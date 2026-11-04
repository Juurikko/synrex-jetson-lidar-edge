// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Tuomas Pietilä
// Synthetic CPU reference to Claude's sealed() generation schedule; no model.
#include "synrex/sealed_cell.hpp"
#include <iostream>
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

} // namespace

int main() {
    try {
        oracle_schedule();
        nonpoisoning_capacity_and_explicit_eviction();
        eviction_after_ack_and_history_disclosure();
        spoof_and_invalid_input_guards();
        std::cout << "SEALED_CELL_CPU_PASS checks=" << checks << " rejections=" << rejects
                  << " GPU_NOT_RUN model_inference_NOT_RUN\n";
        return 0;
    } catch (const std::exception &e) {
        std::cerr << "SEALED_CELL_CPU_FAIL " << e.what() << '\n';
        return 1;
    }
}
