// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Tuomas Pietilä
// Synthetic CPU reference to Claude's sealed() generation schedule; no model.
#include "synrex/sealed_cell.hpp"
#include <algorithm>
#include <cmath>
#include <iostream>
#include <map>
#include <stdexcept>

using namespace synrex::generic;
namespace {
int checks = 0, rejects = 0;
void check(bool ok, const char *why) {
    ++checks;
    if (!ok)
        throw std::runtime_error(why);
}
template <typename F> void must_reject(F callback) {
    bool did = false;
    try {
        callback();
    } catch (const std::exception &) {
        did = true;
    }
    check(did, "expected exception");
    ++rejects;
}
SealedCellPolicy policy() {
    SealedCellPolicy p;
    p.idle_gap_points = 4;
    p.max_active_cells = 3;
    p.max_retained_points = 100;
    p.max_points_per_cell = 20;
    p.max_outstanding_windows = 4;
    p.origin_xy_m = {{300000.0, 6700000.0}};
    p.projected_crs = "EPSG:3067";
    return p;
}
synrex::streaming::SourcePoint point(std::uint64_t i, char cell) {
    synrex::streaming::SourcePoint p;
    p.id = 1000000 + i;
    p.timestamp_ns = i;
    p.xyz = {{300001.0 + (cell - 'A') * 40.0, 6700001.0, 50.0}};
    p.intensity = static_cast<std::uint16_t>(100 + i);
    p.return_number = 1;
    p.return_count = 1;
    return p;
}
std::vector<std::uint8_t> labels(const SealedCellWindow &w) {
    return std::vector<std::uint8_t>(w.context->size(), 1);
}
void oracle_schedule() {
    // Python continuous_policies.sealed(X,..., G=4) would cut A members
    // m=[0,2,7] at diff(2,7)=5, releasing first generation at 7.
    // The second A generation releases at index 12, with all A history.
    SealedCellScheduler s(policy());
    const std::string stream = "ABABBBBABBBBB";
    std::vector<SealedCellWindow> windows;
    for (std::uint64_t i = 0; i < stream.size(); ++i) {
        auto got = s.try_ingest(i, point(i, stream[i]), "EPSG:3067");
        check(got.accepted && !got.needs_commit, "sequential point admitted");
        for (const auto &w : got.issued) {
            check(w.release_point_index <= i, "no future index used for emission");
            for (const auto &p : *w.context)
                check(p.id < 1000000 + i, "future cell observation must be unavailable");
            windows.push_back(w);
            auto output = s.commit_new_only(w, labels(w));
            check(output.semantic_ids.size() == w.fresh_count, "only new cell points published");
        }
    }
    check(windows.size() == 2, "two idle A generations expected");
    check(windows[0].cell == std::array<std::int64_t, 2>({0, 0}) &&
              windows[0].release_point_index == 7 && windows[0].fresh_begin == 0 &&
              windows[0].fresh_count == 2 && windows[0].context->at(0).id == 1000000 &&
              windows[0].context->at(1).id == 1000002,
          "first sealed generation identical to Python grouped A indices");
    check(windows[1].release_point_index == 12 && windows[1].fresh_begin == 2 &&
              windows[1].fresh_count == 1 && windows[1].context->size() == 3 &&
              windows[1].context->at(2).id == 1000007,
          "second sealed generation retains all earlier A observations");
    auto eof = s.try_source_eof(stream.size());
    check(eof.accepted && eof.issued.size() == 1,
          "remaining B points released exactly once at true EOF");
    const auto &b = eof.issued.front();
    check(b.eof_release && b.fresh_count == 10 && b.fresh_begin == 0,
          "whole B generation held causally until EOF");
    auto emitted = s.commit_new_only(b, labels(b));
    check(emitted.source_ids.size() == 10 && emitted.source_ids.front() == 1000001 &&
              emitted.source_ids.back() == 1000012,
          "EOF exact B indices");
    check(s.retained_points() == 13 && s.evictions() == 0 && s.accepted_points() == 13 &&
              s.outstanding_windows() == 0 && !s.failed() && s.finished(),
          "unbounded-equivalent fixture completed");
}

void nonpoisoning_capacity_and_explicit_eviction() {
    auto p = policy();
    p.max_active_cells = 2;
    p.max_retained_points = 6;
    p.max_points_per_cell = 6;
    SealedCellScheduler s(p);
    // A:0, B:1, B:2, B:3, B:4, B:5 => at index5,
    // A becomes inactive (last0, gap4), but output remains outstanding.
    const std::string first = "ABBBBB";
    for (std::uint64_t i = 0; i < first.size(); ++i) {
        auto r = s.try_ingest(i, point(i, first[i]), "EPSG:3067");
        check(r.accepted, "accepted fixture input before capacity");
        if (i == 5)
            check(r.issued.size() == 1 && r.issued[0].cell[0] == 0,
                  "A emitted before any eviction");
    }
    check(s.retained_points() == 6 && s.outstanding_windows() == 1, "bounded quota reached");
    auto third = s.try_ingest(6, point(6, 'C'), "EPSG:3067");
    check(!third.accepted && s.accepted_points() == 6 && s.retained_points() == 6 &&
              s.evictions() == 0 && !s.failed(),
          "unacked generation not evicted; rejected input is untouched");
    // No access to internal token necessary: the retry returns outstanding A
    // only if it was issued during this call; hold the issued A window explicitly.
}

void eviction_after_ack_and_history_disclosure() {
    auto p = policy();
    p.max_active_cells = 2;
    p.max_retained_points = 10;
    p.max_points_per_cell = 10;
    SealedCellScheduler s(p);
    auto r0 = s.try_ingest(0, point(0, 'A'), "EPSG:3067");
    check(r0.accepted, "A admitted");
    SealedCellWindow a;
    for (std::uint64_t i = 1; i <= 5; ++i) {
        auto r = s.try_ingest(i, point(i, 'B'), "EPSG:3067");
        check(r.accepted, "B admitted");
        if (i == 5) {
            check(r.issued.size() == 1, "A becomes eligible after four-point inactivity");
            a = r.issued.at(0);
        }
    }
    auto blocked = s.try_ingest(6, point(6, 'C'), "EPSG:3067");
    check(!blocked.accepted && s.evictions() == 0 && !s.failed(),
          "cannot discard unacked A context");
    auto result = s.commit_new_only(a, labels(a));
    check(result.source_ids == std::vector<std::uint64_t>({1000000}), "A produced before eviction");
    auto retry = s.try_ingest(6, point(6, 'C'), "EPSG:3067");
    check(retry.accepted && s.evictions() == 1 && s.active_cells() == 2 &&
              s.history_may_be_truncated(),
          "retired fully sealed A with honest truncation flag");
    auto eof = s.try_source_eof(7);
    check(eof.accepted && eof.issued.size() == 2, "remaining cells flushed at EOF");
    for (auto &w : eof.issued) {
        check(w.previous_history_may_be_lost, "future output must disclose possible history loss");
        s.commit_new_only(w, labels(w));
    }
    check(!s.failed() && s.retained_points() <= p.max_retained_points, "bounded eviction complete");
}

void spoof_and_invalid_input_guards() {
    {
        SealedCellScheduler s(policy());
        s.try_ingest(0, point(0, 'A'), "EPSG:3067");
        for (std::uint64_t i = 1; i <= 5; ++i) {
            auto r = s.try_ingest(i, point(i, 'B'), "EPSG:3067");
            if (i == 5) {
                auto authentic = r.issued.at(0);
                auto tampered = authentic;
                auto fake = std::vector<synrex::streaming::SourcePoint>(*authentic.context);
                fake.at(0).xyz[0] += 0.5;
                tampered.context =
                    std::make_shared<const std::vector<synrex::streaming::SourcePoint>>(
                        std::move(fake));
                must_reject([&] { s.commit_new_only(tampered, labels(tampered)); });
                check(s.failed(), "forged sealed context permanently rejected");
            }
        }
    }
    {
        SealedCellScheduler s(policy());
        auto bad = point(0, 'A');
        bad.xyz[0] += 100000000.0;
        must_reject([&] { s.try_ingest(0, bad, "EPSG:3067"); });
        check(s.failed(), "out-of-domain point poisons input");
    }
    {
        SealedCellScheduler s(policy());
        must_reject([&] { s.try_ingest(0, point(0, 'A'), "EPSG:4326"); });
        check(s.failed(), "geographic coordinates rejected");
    }
    {
        SealedCellScheduler s(policy());
        s.try_ingest(0, point(0, 'A'), "EPSG:3067");
        must_reject([&] { s.try_ingest(2, point(2, 'A'), "EPSG:3067"); });
        check(s.failed(), "source gaps refused");
    }
    {
        auto p = policy();
        p.max_points_per_cell = 2;
        p.max_retained_points = 2;
        SealedCellScheduler s(p);
        check(s.try_ingest(0, point(0, 'A'), "EPSG:3067").accepted, "first A");
        check(s.try_ingest(1, point(1, 'A'), "EPSG:3067").accepted, "second A");
        auto limited = s.try_ingest(2, point(2, 'A'), "EPSG:3067");
        check(!limited.accepted && !s.failed() && s.accepted_points() == 2,
              "per-cell history cap refuses growth rather than silently dropping history");
    }
}

void independent_stream_index_schedule_oracle() {
    // Mathematically independent group-at-gap implementation of the Python
    // sealed() policy: cell memberships are constructed from the entire
    // deterministic fixture only for TEST ASSERTIONS, never for scheduler inputs.
    constexpr std::uint64_t total = 8192;
    constexpr std::uint64_t gap = 256;
    auto p = policy();
    p.idle_gap_points = gap;
    p.max_active_cells = 16;
    p.max_retained_points = 10000;
    p.max_points_per_cell = 2000;
    p.max_outstanding_windows = 32;
    SealedCellScheduler scheduler(p);
    using Key = std::array<std::int64_t, 2>;
    std::map<Key, std::vector<std::uint64_t>> oracle_inputs;
    struct Event {
        Key key{};
        std::uint64_t release{};
        std::vector<std::uint64_t> newly_labelled;
        std::size_t total_context{};
    };
    std::vector<Event> actual;
    auto accept_window = [&](const SealedCellWindow &w) {
        const auto labelled = scheduler.commit_new_only(w, labels(w));
        actual.push_back({w.cell, w.release_point_index, labelled.source_ids, w.context->size()});
        check(w.context->size() == w.fresh_begin + w.fresh_count,
              "sealed output includes all and only prior cell points");
    };
    for (std::uint64_t i = 0; i < total; ++i) {
        // One strip touches one of eight cells; returns create a long gap.
        const char cell = static_cast<char>('A' + (i / 127) % 8);
        const auto pt = point(i, cell);
        const Key key = {
            static_cast<std::int64_t>(std::floor((pt.xyz[0] - p.origin_xy_m[0]) / p.cell_width_m)),
            static_cast<std::int64_t>(std::floor((pt.xyz[1] - p.origin_xy_m[1]) / p.cell_width_m))};
        oracle_inputs[key].push_back(pt.id);
        auto admitted = scheduler.try_ingest(i, pt, "EPSG:3067");
        check(admitted.accepted && !admitted.needs_commit,
              "unbounded-oracle experiment has sufficient memory");
        for (const auto &w : admitted.issued)
            accept_window(w);
    }
    auto eof = scheduler.try_source_eof(total);
    check(eof.accepted && !eof.needs_commit, "oracle source EOF allowed");
    for (const auto &w : eof.issued)
        accept_window(w);
    std::vector<Event> expected;
    for (const auto &[cell, ids] : oracle_inputs) {
        std::size_t begin = 0;
        while (begin < ids.size()) {
            std::size_t end = begin + 1;
            while (end < ids.size() && ids[end] - ids[end - 1] <= gap)
                ++end;
            const std::uint64_t last = ids[end - 1] - 1000000;
            const auto release = std::min<std::uint64_t>(total, last + gap + 1);
            expected.push_back({cell, release,
                                std::vector<std::uint64_t>(ids.begin() + begin, ids.begin() + end),
                                end});
            begin = end;
        }
    }
    const auto order = [](const Event &a, const Event &b) {
        if (a.release != b.release)
            return a.release < b.release;
        const auto last_a = a.newly_labelled.back(), last_b = b.newly_labelled.back();
        if (last_a != last_b)
            return last_a < last_b;
        return a.key < b.key;
    };
    std::sort(expected.begin(), expected.end(), order);
    check(expected.size() == actual.size(),
          "native sealed generation count equals independent oracle");
    for (std::size_t i = 0; i < expected.size(); ++i) {
        check(expected[i].key == actual[i].key && expected[i].release == actual[i].release &&
                  expected[i].newly_labelled == actual[i].newly_labelled &&
                  expected[i].total_context == actual[i].total_context,
              "native seal schedule, fresh IDs, context and release agree with offline gap oracle");
    }
    check(scheduler.accepted_points() == total && scheduler.evictions() == 0 &&
              !scheduler.history_may_be_truncated() && !scheduler.failed(),
          "full oracle equality requires no eviction and no point loss");
}

// Regression: a held point must never wait forever. Before the fix each of these
// returned needs_commit with nothing to commit, for good.
void liveness_under_pressure() {
    struct Case {
        std::size_t cells, retained, per_cell, outstanding;
        std::uint64_t gap;
        int spread;
    };
    const Case cases[] = {
        {8, 64, 32, 4, 16, 1},   {8, 64, 32, 4, 1000, 4},  {4, 64, 32, 4, 1000, 8},
        {2, 8, 4, 1, 1000, 8},   {1, 1, 1, 1, 1000000, 8}, {3, 9, 9, 2, 5, 16},
        {16, 40, 40, 2, 64, 32}, {4, 16, 8, 1, 2, 3},      {64, 512, 64, 8, 256, 64}};
    std::uint64_t seed = 88172645463325252ULL;
    const auto next = [&seed] {
        seed ^= seed << 13;
        seed ^= seed >> 7;
        seed ^= seed << 17;
        return seed;
    };
    for (const auto &c : cases) {
        auto p = policy();
        p.max_active_cells = c.cells;
        p.max_retained_points = c.retained;
        p.max_points_per_cell = c.per_cell;
        p.max_outstanding_windows = c.outstanding;
        p.idle_gap_points = c.gap;
        SealedCellScheduler s(p);
        constexpr std::uint64_t total = 4000;
        std::vector<int> seen(total, 0);
        const auto commit = [&](const SealedCellWindow &w) {
            check(w.context->size() == w.fresh_begin + w.fresh_count, "context layout");
            for (const auto &q : *w.context)
                check(q.id - 1000000 < total, "context holds received points only");
            for (const auto id : s.commit_new_only(w, labels(w)).source_ids)
                ++seen.at(id - 1000000);
        };
        for (std::uint64_t i = 0; i < total; ++i) {
            const auto pt = point(i, static_cast<char>('A' + next() % c.spread));
            for (int attempt = 0;; ++attempt) {
                check(attempt < 8, "held point always makes progress after commits");
                const auto r = s.try_ingest(i, pt, "EPSG:3067");
                for (const auto &w : s.outstanding())
                    commit(w);
                if (r.accepted)
                    break;
            }
            check(s.retained_points() <= p.max_retained_points &&
                      s.active_cells() <= p.max_active_cells,
                  "caps hold under pressure");
        }
        for (;;) {
            const auto r = s.try_source_eof(total);
            for (const auto &w : s.outstanding())
                commit(w);
            if (r.accepted)
                break;
        }
        check(s.complete(), "every point committed");
        for (const auto n : seen)
            check(n == 1, "every point labelled exactly once");
    }
    // Hover: one cell, never idle. Used to stop for good at the per-cell cap.
    auto p = policy();
    p.max_points_per_cell = 32;
    p.idle_gap_points = 16;
    SealedCellScheduler s(p);
    std::uint64_t forced_seen = 0;
    for (std::uint64_t i = 0; i < 500; ++i) {
        for (int attempt = 0;; ++attempt) {
            check(attempt < 8, "hover makes progress");
            const auto r = s.try_ingest(i, point(i, 'A'), "EPSG:3067");
            for (const auto &w : s.outstanding()) {
                forced_seen += w.forced_release;
                s.commit_new_only(w, labels(w));
            }
            if (r.accepted)
                break;
        }
    }
    check(forced_seen > 0 && s.forced_releases() == forced_seen, "pressure releases are disclosed");
    // A held point may not be swapped for another one at retry.
    auto q = policy();
    q.max_active_cells = 1;
    q.idle_gap_points = 1000;
    SealedCellScheduler t(q);
    check(t.try_ingest(0, point(0, 'A'), "EPSG:3067").accepted, "first point");
    const auto held = t.try_ingest(1, point(1, 'B'), "EPSG:3067");
    check(!held.accepted && held.issued.size() == 1, "pressure seals the only pending cell");
    for (const auto &w : t.outstanding())
        t.commit_new_only(w, labels(w));
    auto swapped = point(1, 'B');
    swapped.xyz[2] += 1.0;
    must_reject([&] { t.try_ingest(1, swapped, "EPSG:3067"); });
}

// Only the cell that lost points is flagged, not every later window.
void truncation_flag_is_per_cell() {
    auto p = policy();
    p.max_active_cells = 2;
    p.idle_gap_points = 2;
    SealedCellScheduler s(p);
    std::vector<SealedCellWindow> ws;
    std::uint64_t i = 0;
    for (const char c : std::string("AAABBBBCCCCAAAACCCC")) {
        for (;;) {
            const auto r = s.try_ingest(i, point(i, c), "EPSG:3067");
            for (const auto &w : s.outstanding()) {
                ws.push_back(w);
                s.commit_new_only(w, labels(w));
            }
            if (r.accepted)
                break;
        }
        ++i;
    }
    check(s.evictions() > 0, "fixture evicts");
    bool a_again = false, other_clean = false;
    for (const auto &w : ws) {
        if (w.cell[0] == 0 && w.fresh_begin == 0 && w.context->front().id > 1000000 + 3)
            a_again = a_again || w.cell_history_truncated;
        if (w.cell[0] == 2)
            other_clean =
                other_clean || (!w.cell_history_truncated && w.previous_history_may_be_lost);
    }
    check(a_again && other_clean, "returning cell flagged, untouched cell clean");
}

} // namespace

int main() {
    try {
        oracle_schedule();
        nonpoisoning_capacity_and_explicit_eviction();
        eviction_after_ack_and_history_disclosure();
        spoof_and_invalid_input_guards();
        independent_stream_index_schedule_oracle();
        liveness_under_pressure();
        truncation_flag_is_per_cell();
        std::cout << "SEALED_CELL_CPU_PASS checks=" << checks << " rejections=" << rejects
                  << " GPU_NOT_RUN model_inference_NOT_RUN\n";
        return 0;
    } catch (const std::exception &e) {
        std::cerr << "SEALED_CELL_CPU_FAIL " << e.what() << '\n';
        return 1;
    }
}
