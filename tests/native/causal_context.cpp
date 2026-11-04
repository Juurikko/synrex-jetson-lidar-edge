// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Tuomas Pietilä
// CPU-only causal context and source-index publication tests; no neural/GPU claims.
#include "synrex/causal_context.hpp"
#include <functional>
#include <iostream>
#include <stdexcept>

using namespace synrex::generic;
namespace {
std::size_t checks{}, rejections{};
void check(bool ok, const char *message) {
    ++checks;
    if (!ok) throw std::runtime_error(message);
}
template <class F> void rejects(F run) {
    bool threw = false;
    try { run(); } catch (const std::exception &) { threw = true; }
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
    p.fixed_world_origin_m = {{300000.0, 6700000.0, 0.0}};
    p.world_grid_size_m = 0.5;
    p.require_exact_source_rows = source_rows;
    return p;
}

Frame frame(std::uint64_t start, std::size_t count, std::uint64_t seq,
            std::uint64_t total = 13, bool source_rows = true,
            std::string stream = "sha256-source") {
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
        streaming::SourcePoint p;
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
    std::vector<std::uint8_t> labels(w.context.size(), 6);
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
    check(d.size() == 1 && d[0].full_history && d[0].fresh_begin == 6 &&
          d[0].context.size() == 10, "full six-point past carried forward");
    auto third = asmbl.commit_new_only(d[0], mock_labels(d[0]));
    check(third.source_ids == std::vector<std::uint64_t>({1008, 1009, 1010, 1011}),
          "past labels never republished");
    const auto tail = asmbl.source_eof();
    check(tail && tail->fresh_count == 1 && tail->context.size() == 7,
          "exact one-point EOF tail");
    auto last = asmbl.commit_new_only(*tail, mock_labels(*tail));
    check(last.source_ids == std::vector<std::uint64_t>({1012}),
          "EOF last point exactly once");
    check(asmbl.accepted_points() == 13 && asmbl.outstanding_windows() == 0 &&
          asmbl.finished() && !asmbl.failed(), "complete clean bounded finish");
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
    check(first_a.size() == 1 && first_b.size() == 1,
          "both prefixes emit one window");
    auto future = frame(4, 4, 1, 8);
    for (auto &p : future.points) p.intensity = 400;
    auto later = b.ingest(future);
    check(later.size() == 1 && later[0].fresh_begin == 4,
          "future history may be used only by future window");
    check(first_a[0].context.size() == first_b[0].context.size() &&
          first_a[0].fresh_begin == first_b[0].fresh_begin &&
          first_a[0].cutoff_frame_end_ns == first_b[0].cutoff_frame_end_ns &&
          first_a[0].fixed_world_origin_m == first_b[0].fixed_world_origin_m,
          "future append did not revise prefix");
    for (std::size_t i = 0; i < first_a[0].context.size(); ++i)
        check(first_a[0].context[i].id == first_b[0].context[i].id &&
              first_a[0].context[i].intensity == first_b[0].context[i].intensity,
              "historical input cannot be replaced by future");
    auto out_a = a.commit_new_only(first_a[0], mock_labels(first_a[0]));
    auto out_b = b.commit_new_only(first_b[0], mock_labels(first_b[0]));
    check(out_a.source_ids == out_b.source_ids &&
          out_a.semantic_ids == out_b.semantic_ids, "committed prefix is immutable");
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
        auto p = policy(); p.fresh_points = 0;
        rejects([&] { CausalContextAssembler asmbl(p); });
        p = policy(); p.history_points = 10;
        rejects([&] { CausalContextAssembler asmbl(p); });
        p = policy(); p.fixed_world_origin_m[0] = std::numeric_limits<double>::infinity();
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

} // namespace

int main() {
    try {
        basic_history_and_eof();
        future_invariance();
        poison_guards();
        std::cout << "CAUSAL_CONTEXT_CPU_PASS checks=" << checks
                  << " rejections=" << rejections
                  << " GPU_NOT_RUN model_inference_NOT_RUN" << std::endl;
        return 0;
    } catch (const std::exception &error) {
        std::cerr << "CAUSAL_CONTEXT_CPU_FAIL " << error.what() << std::endl;
        return 1;
    }
}
