// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Tuomas Pietilä
// CPU-only causal context and source-index publication tests; no neural/GPU claims.
#include "synrex/causal_context.hpp"
#include <functional>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <type_traits>
#include <utility>

using namespace synrex::generic;
static_assert(std::is_const_v<std::remove_reference_t<decltype(
                  std::declval<const CausalContextWindow &>().context->at(0))>>,
              "A published causal point must be immutable through the native API");
namespace {
std::size_t checks{}, rejections{};
void check(bool ok, const char *message) {
    ++checks;
    if (!ok)
        throw std::runtime_error(message);
}
template <class F> void rejects(F run) {
    bool threw = false;
    try {
        run();
    } catch (const std::exception &) {
        threw = true;
    }
    check(threw, "expected rejection was not raised");
    ++rejections;
}

CausalContextPolicy policy(bool source_rows = true) {
    CausalContextPolicy p;
    p.fresh_points = 4;
    p.history_points = 6;
    p.max_context_points = 10;
    p.max_inflight_windows = 4;
    p.semantic_class_count = 7;
    p.fixed_world_origin_xy_m = {{300000.0, 6700000.0}};
    p.projected_crs = "EPSG:3067";
    p.world_grid_size_m = 0.5;
    p.require_exact_source_rows = source_rows;
    return p;
}

Frame frame(std::uint64_t start, std::size_t count, std::uint64_t seq, std::uint64_t total = 13,
            bool source_rows = true, std::string stream = "sha256-source") {
    Frame f;
    f.identity = {"fixed-descriptor-sha256", "session-abc", 3};
    f.sensor_id = "als-source";
    f.stream_id = std::move(stream);
    f.coordinate_frame = "projected-metre";
    f.crs = "EPSG:3067";
    f.sequence = seq;
    f.frame_id = seq;
    f.frame_start_ns = 100 + seq * 100;
    f.frame_end_ns = 101 + seq * 100;
    if (source_rows) {
        SourceHeader header{};
        header.records = total;
        header.wire_begin = 1000;
        f.source_header = header;
    }
    for (std::size_t i = 0; i < count; ++i) {
        synrex::streaming::SourcePoint p;
        p.id = 1000 + start + i;
        p.timestamp_ns = 100000 + start + i;
        p.xyz = {{300000.0 + double(start + i), 6700000.0, 50.0}};
        p.intensity = 100;
        p.return_number = 1;
        p.return_count = 1;
        f.points.push_back(p);
        if (source_rows) {
            SourceRow row{};
            row.original_record_index = start + i;
            row.integer_xyz = {{static_cast<std::int32_t>(start + i), 0, 5000}};
            f.source_rows.push_back(row);
        }
    }
    return f;
}

std::vector<std::uint8_t> mock_labels(const CausalContextWindow &w) {
    // Intentionally distinctive past labels: a correct output cannot leak them.
    std::vector<std::uint8_t> labels(w.context->size(), 6);
    for (std::size_t i = 0; i < w.fresh_count; ++i)
        labels[w.fresh_begin + i] = static_cast<std::uint8_t>((i + 1) % 6);
    return labels;
}

void basic_history_and_eof() {
    CausalContextAssembler asmbl(policy());
    auto a = asmbl.ingest(frame(0, 4, 0));
    check(a.size() == 1, "first full frame emits one window");
    check(a[0].fresh_begin == 0 && a[0].fresh_count == 4 && !a[0].full_history,
          "first window has no invented history");
    auto first = asmbl.commit_new_only(a[0], mock_labels(a[0]));
    check(first.source_ids == std::vector<std::uint64_t>({1000, 1001, 1002, 1003}),
          "first output source order");
    check(first.semantic_ids == std::vector<std::uint8_t>({1, 2, 3, 4}),
          "fresh-only labels extracted");

    auto b = asmbl.ingest(frame(4, 4, 1));
    check(b.size() == 1 && b[0].fresh_begin == 4 && b[0].fresh_count == 4,
          "second window includes four previous points");
    auto second = asmbl.commit_new_only(b[0], mock_labels(b[0]));
    check(second.source_ids.front() == 1004 && second.source_ids.back() == 1007,
          "second output excludes history ids");
    check(asmbl.retained_history_points() == 6, "history ring is bounded");
    check(asmbl.ingest(frame(8, 2, 2)).empty(), "partial frame stays pending");
    auto d = asmbl.ingest(frame(10, 3, 3));
    check(d.size() == 1 && d[0].full_history && d[0].fresh_begin == 6 && d[0].context->size() == 10,
          "full six-point past carried forward");
    auto third = asmbl.commit_new_only(d[0], mock_labels(d[0]));
    check(third.source_ids == std::vector<std::uint64_t>({1008, 1009, 1010, 1011}),
          "past labels never republished");
    const auto tail = asmbl.source_eof();
    check(tail && tail->fresh_count == 1 && tail->context->size() == 7, "exact one-point EOF tail");
    auto last = asmbl.commit_new_only(*tail, mock_labels(*tail));
    check(last.source_ids == std::vector<std::uint64_t>({1012}), "EOF last point exactly once");
    check(asmbl.accepted_points() == 13 && asmbl.outstanding_windows() == 0 && asmbl.finished() &&
              !asmbl.failed(),
          "complete clean bounded finish");
    rejects([&] { asmbl.source_eof(); });
    check(asmbl.failed(), "duplicate EOF poisons session");
}

void future_invariance() {
    // The exact same prefix must emit identical history, origin, IDs and
    // cutoff regardless of subsequent (or mutated) unseen points.
    CausalContextAssembler a(policy()), b(policy());
    auto prefix = frame(0, 4, 0, 8);
    const auto first_a = a.ingest(prefix);
    const auto first_b = b.ingest(prefix);
    check(first_a.size() == 1 && first_b.size() == 1, "both prefixes emit one window");
    auto future = frame(4, 4, 1, 8);
    for (auto &p : future.points)
        p.intensity = 400;
    auto later = b.ingest(future);
    check(later.size() == 1 && later[0].fresh_begin == 4,
          "future history may be used only by future window");
    check(first_a[0].context->size() == first_b[0].context->size() &&
              first_a[0].fresh_begin == first_b[0].fresh_begin &&
              first_a[0].cutoff_frame_end_ns == first_b[0].cutoff_frame_end_ns &&
              first_a[0].fixed_world_origin_xy_m == first_b[0].fixed_world_origin_xy_m,
          "future append did not revise prefix");
    for (std::size_t i = 0; i < first_a[0].context->size(); ++i)
        check(first_a[0].context->at(i).id == first_b[0].context->at(i).id &&
                  first_a[0].context->at(i).intensity == first_b[0].context->at(i).intensity,
              "historical input cannot be replaced by future");
    auto out_a = a.commit_new_only(first_a[0], mock_labels(first_a[0]));
    auto out_b = b.commit_new_only(first_b[0], mock_labels(first_b[0]));
    check(out_a.source_ids == out_b.source_ids && out_a.semantic_ids == out_b.semantic_ids,
          "committed prefix is immutable");
}

void poison_guards() {
    {
        CausalContextAssembler asmbl(policy());
        auto broken = frame(0, 4, 0);
        broken.points[2].id = broken.points[1].id;
        rejects([&] { asmbl.ingest(broken); });
        check(asmbl.failed(), "index-gap failure is terminal");
        rejects([&] { asmbl.ingest(frame(0, 4, 0)); });
    }
    {
        CausalContextAssembler asmbl(policy());
        asmbl.ingest(frame(0, 3, 0, 4));
        auto changed = frame(3, 1, 1, 4);
        changed.stream_id = "foreign-acquisition";
        rejects([&] { asmbl.ingest(changed); });
        check(asmbl.failed(), "cross-source transition poisoned");
    }
    {
        CausalContextAssembler asmbl(policy());
        asmbl.ingest(frame(0, 3, 0, 13));
        rejects([&] { asmbl.source_eof(); });
        check(asmbl.failed(), "premature EOF poisoned");
    }
    {
        CausalContextAssembler asmbl(policy());
        auto a = asmbl.ingest(frame(0, 4, 0, 4));
        auto labels = mock_labels(a.at(0));
        labels.at(0) = 255;
        rejects([&] { asmbl.commit_new_only(a[0], labels); });
        check(asmbl.failed(), "invalid class poisons window");
    }
    {
        CausalContextAssembler asmbl(policy());
        auto a = asmbl.ingest(frame(0, 4, 0, 4));
        auto labels = mock_labels(a.at(0));
        labels.pop_back();
        rejects([&] { asmbl.commit_new_only(a[0], labels); });
    }
    {
        CausalContextAssembler asmbl(policy());
        asmbl.ingest(frame(0, 3, 0, 8));
        auto seq_gap = frame(3, 1, 2, 8);
        rejects([&] { asmbl.ingest(seq_gap); });
    }
    {
        CausalContextAssembler asmbl(policy());
        auto f = frame(0, 4, 0, 4);
        f.source_rows[0].original_record_index = 1;
        rejects([&] { asmbl.ingest(f); });
    }
    {
        auto p = policy();
        p.fresh_points = 0;
        rejects([&] { CausalContextAssembler asmbl(p); });
        p = policy();
        p.history_points = 10;
        rejects([&] { CausalContextAssembler asmbl(p); });
        p = policy();
        p.fixed_world_origin_xy_m[0] = std::numeric_limits<double>::infinity();
        rejects([&] { CausalContextAssembler asmbl(p); });
    }
    {
        // Optional identity-only mode is explicit, not mislabeled as exact LAS.
        CausalContextAssembler asmbl(policy(false));
        auto a = asmbl.ingest(frame(0, 4, 0, 4, false));
        check(a.size() == 1, "explicit identity-only context supported");
        asmbl.commit_new_only(a[0], mock_labels(a[0]));
        check(!asmbl.source_eof(), "source-v3 EOF accepted without LAS coverage claim");
    }
}

void commit_window_integrity_guards() {
    // Every payload attribute and history record is covered by the PRIVATE
    // issuance digest. No externally mutable window may become authoritative.
    for (int variant = 0; variant < 12; ++variant) {
        CausalContextAssembler assembler(policy());
        auto first = assembler.ingest(frame(0, 4, 0, 8));
        assembler.commit_new_only(first.at(0), mock_labels(first.at(0)));
        auto next = assembler.ingest(frame(4, 4, 1, 8));
        auto altered = next.at(0);
        // Adversarial transport of a counterfeit context copy: even if its
        // fields match an issued window, it must NOT acquire the private handle.
        std::vector<synrex::streaming::SourcePoint> counterfeit(*altered.context);
        const auto fresh = altered.fresh_begin;
        switch (variant) {
        case 0:
            counterfeit.at(fresh).xyz[0] += 0.125;
            break;
        case 1:
            counterfeit.at(0).id += 123;
            break;
        case 2:
            counterfeit.at(0).xyz[1] += 0.25;
            break;
        case 3:
            counterfeit.at(0).timestamp_ns += 1;
            break;
        case 4:
            counterfeit.at(fresh).intensity = 65535;
            break;
        case 5:
            counterfeit.at(fresh).return_number = 2;
            break;
        case 6:
            counterfeit.at(fresh).return_count = 2;
            break;
        case 7:
            counterfeit.at(fresh).timestamp_ns += 1;
            break;
        case 8:
            altered.cutoff_frame_end_ns += 1;
            break;
        case 9:
            altered.full_history = !altered.full_history;
            break;
        case 10:
            counterfeit.at(0).intensity.reset();
            break;
        case 11:
            counterfeit.at(fresh).xyz[2] += 0.01;
            break;
        }
        if (variant != 8 && variant != 9)
            altered.context = std::make_shared<const std::vector<synrex::streaming::SourcePoint>>(
                std::move(counterfeit));
        rejects([&] { assembler.commit_new_only(altered, mock_labels(altered)); });
        check(assembler.failed(), "every mutation of issued context must poison commit");
    }
    {
        // Old authentic issued window still commits unchanged after future
        // ingress, so the hash is a content binding, not a map-state snapshot.
        CausalContextAssembler assembler(policy());
        auto first = assembler.ingest(frame(0, 4, 0, 8));
        auto second = assembler.ingest(frame(4, 4, 1, 8));
        auto out0 = assembler.commit_new_only(first.at(0), mock_labels(first.at(0)));
        auto out1 = assembler.commit_new_only(second.at(0), mock_labels(second.at(0)));
        check(out0.source_ids.front() == 1000 && out1.source_ids.front() == 1004,
              "exact issued window remains valid despite later accepted frames");
        check(!assembler.failed(), "valid delayed commit does not poison session");
    }
}

void backpressure_is_retryable_and_never_consumes_input() {
    {
        auto p = policy();
        p.fresh_points = 16384;
        p.history_points = 114688;
        p.max_context_points = 131072;
        p.max_inflight_windows = 4;
        CausalContextAssembler assembler(p);
        auto initial = assembler.ingest(frame(0, 16384, 0, 200000));
        check(initial.size() == 1 && assembler.outstanding_windows() == 1,
              "initial window outstanding");
        auto large_frame = frame(16384, 65536, 1, 200000);
        auto busy = assembler.try_ingest(large_frame);
        check(!busy.accepted && busy.windows.empty() && !assembler.failed() &&
                  assembler.accepted_points() == 16384 && assembler.outstanding_windows() == 1 &&
                  assembler.pending_new_points() == 0,
              "busy 65k frame returns capacity without mutating source state");
        rejects([&] { assembler.ingest(large_frame); });
        check(!assembler.failed() && assembler.accepted_points() == 16384,
              "backpressure wrapper throws but does not permanently poison");
        assembler.commit_new_only(initial.at(0), mock_labels(initial.at(0)));
        auto retry = assembler.try_ingest(large_frame);
        check(retry.accepted && retry.windows.size() == 4,
              "exact original frame retries after old window completes");
        for (std::size_t i = 0; i < retry.windows.size(); ++i) {
            auto result =
                assembler.commit_new_only(retry.windows[i], mock_labels(retry.windows[i]));
            check(result.source_ids.front() == 1000 + 16384 + i * 16384 &&
                      result.source_ids.size() == 16384,
                  "retry preserves every new point in exact source order");
        }
        check(!assembler.failed() && assembler.accepted_points() == 81920 &&
                  assembler.retained_history_points() == 81920 &&
                  assembler.outstanding_windows() == 0,
              "backpressure recovery keeps bounded history and full accepted coverage");
    }
    {
        auto p = policy();
        p.max_inflight_windows = 1;
        CausalContextAssembler assembler(p);
        auto first = assembler.ingest(frame(0, 4, 0, 5));
        check(assembler.ingest(frame(4, 1, 1, 5)).empty(),
              "one partial EOF point buffered while prior inference outstanding");
        auto busy = assembler.try_source_eof();
        check(!busy.accepted && !busy.window && !assembler.failed() && !assembler.finished() &&
                  assembler.pending_new_points() == 1,
              "EOF capacity wait retains the final point without poisoning");
        rejects([&] { assembler.source_eof(); });
        check(!assembler.failed() && !assembler.finished(),
              "EOF wrapper capacity rejection is retryable");
        assembler.commit_new_only(first.at(0), mock_labels(first.at(0)));
        auto admitted = assembler.try_source_eof();
        check(admitted.accepted && admitted.window && admitted.window->fresh_count == 1 &&
                  assembler.finished(),
              "EOF reattempt publishes single remaining point");
        auto last = assembler.commit_new_only(*admitted.window, mock_labels(*admitted.window));
        check(last.source_ids == std::vector<std::uint64_t>({1004}),
              "EOF after backpressure publishes point exactly once");
    }
    {
        // Invalid data may never hide behind a valid capacity denial.
        auto p = policy();
        p.max_inflight_windows = 1;
        CausalContextAssembler assembler(p);
        assembler.ingest(frame(0, 4, 0, 8));
        auto invalid = frame(4, 4, 1, 8);
        invalid.crs = "EPSG:4326";
        rejects([&] { assembler.try_ingest(invalid); });
        check(assembler.failed(), "invalid data is terminal even under full backpressure");
    }
}

void projected_crs_and_spatial_admission() {
    {
        auto p = policy();
        p.projected_crs = "EPSG:4326";
        rejects([&] { CausalContextAssembler a(p); });
        p.projected_crs = "EPSG:2263"; // projected, but US survey foot
        rejects([&] { CausalContextAssembler a(p); });
        p.projected_crs = "EPSG:3067junk";
        rejects([&] { CausalContextAssembler a(p); });
        p.projected_crs = "EPSG:2154"; // metre-based Lambert-93
        CausalContextAssembler a(p);
        auto f = frame(0, 4, 0, 4);
        f.crs = "EPSG:2154";
        check(a.ingest(f).size() == 1, "admitted French projected-metre CRS works");
    }
    {
        CausalContextAssembler a(policy());
        auto geographic = frame(0, 4, 0, 4);
        geographic.crs = "EPSG:4326";
        rejects([&] { a.ingest(geographic); });
        check(a.failed(), "geographic frame cannot use a metre grid");
    }
    {
        CausalContextAssembler a(policy());
        auto far = frame(0, 4, 0, 4);
        far.points.at(2).xyz[0] += 100000000.0; // 100,000 km from anchor
        rejects([&] { a.ingest(far); });
        check(a.failed(), "absurd X offset rejected before any window issued");
    }
    {
        CausalContextAssembler a(policy());
        auto tall = frame(0, 4, 0, 4);
        tall.points.at(1).xyz[2] = 25001.0;
        rejects([&] { a.ingest(tall); });
        check(a.failed(), "invalid vertical magnitude rejected even without Z anchor");
    }
    {
        // Vertical origin is deliberately unanchored, to avoid silently
        // changing the frozen U-Net's z preprocessing during model research.
        auto p = policy();
        p.fixed_world_origin_xy_m = {{300000.0, 6700000.0}};
        CausalContextAssembler a(p);
        auto f = frame(0, 4, 0, 4);
        f.points.at(2).xyz[2] = 120.0;
        auto w = a.ingest(f);
        check(w.size() == 1 && w[0].fixed_world_origin_xy_m == p.fixed_world_origin_xy_m &&
                  w[0].context->at(2).xyz[2] == 120.0,
              "XY anchor does not shift or replace the measured Z coordinate");
    }
}

} // namespace

int main() {
    try {
        basic_history_and_eof();
        future_invariance();
        poison_guards();
        commit_window_integrity_guards();
        backpressure_is_retryable_and_never_consumes_input();
        projected_crs_and_spatial_admission();
        std::cout << "CAUSAL_CONTEXT_CPU_PASS checks=" << checks << " rejections=" << rejections
                  << " GPU_NOT_RUN model_inference_NOT_RUN" << std::endl;
        return 0;
    } catch (const std::exception &error) {
        std::cerr << "CAUSAL_CONTEXT_CPU_FAIL " << error.what() << std::endl;
        return 1;
    }
}
