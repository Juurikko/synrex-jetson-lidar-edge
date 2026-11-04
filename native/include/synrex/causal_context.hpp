// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Tuomas Pietilä
#pragma once
#include "synrex/generic_codec.hpp"
#include <deque>

namespace synrex::generic {

// Opt-in, single-source, arrival-order context policy. It changes NO existing
// wire protocol, model tensor layout, inference session, or released Q1 policy.
// A downstream executor must separately qualify the fixed-origin voxelizer.
struct CausalContextPolicy {
    std::size_t fresh_points{16384};
    std::size_t history_points{114688};
    std::size_t max_context_points{131072};
    std::size_t max_inflight_windows{4};
    std::uint16_t semantic_class_count{7};
    // Only planar X/Y are anchored; Z anchoring has not passed quality gates.
    std::array<double, 2> fixed_world_origin_xy_m{{0.0, 0.0}};
    double world_grid_size_m{1.0};
    std::string projected_crs; // caller must explicitly admit a known projected-metre CRS
    double max_horizontal_offset_m{100000.0}, max_abs_elevation_m{25000.0};
    bool require_exact_source_rows{true};

    void validate() const;
};

struct CausalContextWindow {
    std::string descriptor_sha256, session_id, stream_id, sensor_id, coordinate_frame, crs;
    std::uint64_t map_epoch{}, window_index{}, cutoff_frame_end_ns{};
    std::array<double, 2> fixed_world_origin_xy_m{};
    double world_grid_size_m{};
    std::size_t fresh_begin{}, fresh_count{};
    std::vector<streaming::SourcePoint> context; // [past-only H][new-only W]
    bool full_history{};
};

struct NewOnlyClassification {
    std::uint64_t window_index{};
    std::vector<std::uint64_t> source_ids;
    std::vector<std::uint8_t> semantic_ids;
};

// CPU path for causal window assembly, NOT an inference backend. Every accepted
// input is an already admitted generic::Frame; this class adds immutable
// single-source ordering and optional exact LAS source-row coverage rules.
// No future observations are used, and only the final fresh slice of a
// completed context window can be committed. Flush once at true source EOF.
// Invalid content poisons the session; capacity pressure is retryable and never
// consumes a frame. The caller must stop upstream or retry the exact frame.
// No implicit fallbacks or rewinds.
class CausalContextAssembler final {
    struct Token {
        std::uint64_t window_index{}, first_new_id{};
        std::size_t fresh_count{};
        // Private digest of the complete issued window, including history.
        std::array<unsigned char, 32> window_sha256{};
    };
    CausalContextPolicy policy_;
    std::deque<streaming::SourcePoint> past_, pending_;
    std::deque<Token> in_flight_;
    std::optional<FrameIdentity> identity_;
    std::optional<SourceHeader> source_header_;
    std::string stream_id_, sensor_id_, coordinate_frame_, crs_;
    std::uint64_t next_id_{}, next_original_index_{}, last_sequence_{}, last_end_ns_{};
    std::uint64_t accepted_points_{}, window_index_{}, committed_windows_{}, cutoff_ns_{};
    bool have_last_{}, finished_{}, failed_{};

    void check_frame(const Frame &frame) const;
    CausalContextWindow emit(std::size_t fresh);
    void append_accepted(const Frame &frame, std::vector<CausalContextWindow> &out);

  public:
    explicit CausalContextAssembler(CausalContextPolicy policy);
    CausalContextAssembler(const CausalContextAssembler &) = delete;
    CausalContextAssembler &operator=(const CausalContextAssembler &) = delete;

    struct IngestResult {
        bool accepted{}; // false means retryable capacity pressure; NO input was consumed
        std::vector<CausalContextWindow> windows;
    };
    struct EofResult {
        bool accepted{}; // false means retry after outstanding commits
        std::optional<CausalContextWindow> window;
    };
    IngestResult try_ingest(const Frame &frame);
    std::vector<CausalContextWindow> ingest(const Frame &frame);
    EofResult try_source_eof();
    std::optional<CausalContextWindow> source_eof();
    NewOnlyClassification commit_new_only(const CausalContextWindow &window,
                                          const std::vector<std::uint8_t> &all_context_labels);
    std::size_t retained_history_points() const noexcept { return past_.size(); }
    std::size_t pending_new_points() const noexcept { return pending_.size(); }
    std::size_t outstanding_windows() const noexcept { return in_flight_.size(); }
    std::uint64_t accepted_points() const noexcept { return accepted_points_; }
    bool failed() const noexcept { return failed_; }
    bool finished() const noexcept { return finished_; }
};
} // namespace synrex::generic
