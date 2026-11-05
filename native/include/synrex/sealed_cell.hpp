// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Tuomas Pietilä
#pragma once
#include "synrex/causal_context.hpp"
#include <list>
#include <map>
#include <memory>
#include <set>

namespace synrex::generic {

// Experimental CPU-only stream-index scheduler, NOT trained inference.
// Mirrors continuous_r1/continuous_policies.py sealed() up to exact memory
// admission: XY 32m buckets, G-point inactivity, all earlier cell points as
// context, only uncommitted new points returned. No look-ahead or count gain.
struct SealedCellPolicy {
    std::uint64_t idle_gap_points{131072};
    std::size_t max_active_cells{256};
    std::size_t max_retained_points{1048576};
    std::size_t max_points_per_cell{131072};
    std::size_t max_outstanding_windows{32};
    std::array<double, 2> origin_xy_m{{0.0, 0.0}};
    double cell_width_m{32.0};
    double max_xy_distance_m{100000.0};
    std::string projected_crs;
    std::uint16_t semantic_class_count{7};
    void validate() const;
};

struct SealedCellWindow {
    std::array<std::int64_t, 2> cell{};
    std::uint64_t epoch{}, release_point_index{}, window_index{};
    std::size_t fresh_begin{}, fresh_count{};
    bool previous_history_may_be_lost{}; // true after ANY bounded eviction
    bool cell_history_truncated{};       // true when THIS cell lost earlier points
    bool forced_release{};               // sealed by memory pressure, not idle gap
    bool eof_release{};
    std::array<double, 2> cell_origin_xy_m{};
    std::shared_ptr<const std::vector<streaming::SourcePoint>> context;
};

struct SealedCellLabels {
    std::uint64_t window_index{};
    std::vector<std::uint64_t> source_ids;
    std::vector<std::uint8_t> semantic_ids;
};

class SealedCellScheduler final {
  public:
    struct Result {
        bool accepted{};     // input point accepted or EOF admitted
        bool needs_commit{}; // pending sealed output must be committed and retried
        std::vector<SealedCellWindow> issued;
    };

  private:
    using Key = std::array<std::int64_t, 2>;
    struct Cell {
        std::vector<streaming::SourcePoint> points;
        std::size_t emitted_count{}, outstanding{};
        std::uint64_t last_seen_index{}, epoch{};
        bool history_truncated{}, in_pending{};
        std::list<Key>::iterator pending_at{};
    };
    struct Issued {
        SealedCellWindow window;
    };
    SealedCellPolicy policy_;
    std::map<Key, Cell> cells_;
    std::deque<Issued> issued_;
    std::list<Key> pending_; // cells holding unsealed points, oldest last_seen first
    std::set<Key> evicted_keys_;
    bool tombstones_overflowed_{};
    std::uint64_t next_index_{}, first_id_{}, windows_{}, evictions_{}, rejected_capacity_{};
    std::uint64_t epochs_{}, forced_releases_{}, committed_points_{};
    bool held_{};
    streaming::SourcePoint held_point_{};
    std::size_t retained_points_{};
    bool have_source_{}, finished_{}, failed_{}, may_have_lost_history_{};

    Key cell_of(const streaming::SourcePoint &) const;
    bool seal_expired(std::uint64_t watermark, bool eof, Result &result);
    SealedCellWindow seal(const Key &, std::uint64_t watermark, bool eof, bool forced);
    bool reserve_one(const Key &, std::uint64_t watermark, Result &result);
    void validate_point(std::uint64_t index, const streaming::SourcePoint &point,
                        const std::string &source_crs) const;

  public:
    explicit SealedCellScheduler(SealedCellPolicy policy);
    SealedCellScheduler(const SealedCellScheduler &) = delete;
    SealedCellScheduler &operator=(const SealedCellScheduler &) = delete;

    // Source-index (0..N-1) must be globally sequential; same source_id-base.
    // An unaccepted point is NOT consumed. Issued windows MAY need committing
    // before retrying the identical input. Never silently discard the point.
    // Liveness: after every issued and outstanding window is committed, a retry
    // of the same point is accepted or issues more work. It cannot wait forever.
    Result try_ingest(std::uint64_t global_index, const streaming::SourcePoint &point,
                      const std::string &source_crs);
    Result try_source_eof(std::uint64_t total_points);
    SealedCellLabels commit_new_only(const SealedCellWindow &window,
                                     const std::vector<std::uint8_t> &all_context_labels);

    std::uint64_t accepted_points() const noexcept { return next_index_; }
    std::size_t retained_points() const noexcept { return retained_points_; }
    std::size_t active_cells() const noexcept { return cells_.size(); }
    std::size_t outstanding_windows() const noexcept { return issued_.size(); }
    std::uint64_t evictions() const noexcept { return evictions_; }
    std::uint64_t capacity_rejections() const noexcept { return rejected_capacity_; }
    bool history_may_be_truncated() const noexcept { return may_have_lost_history_; }
    std::uint64_t forced_releases() const noexcept { return forced_releases_; }
    std::uint64_t committed_points() const noexcept { return committed_points_; }
    // Every accepted point has been sealed, issued and committed exactly once.
    bool complete() const noexcept {
        return finished_ && issued_.empty() && committed_points_ == next_index_;
    }
    // Recovery: a caller that lost an issued result can fetch it again.
    std::vector<SealedCellWindow> outstanding() const {
        std::vector<SealedCellWindow> all;
        for (const auto &i : issued_)
            all.push_back(i.window);
        return all;
    }
    bool failed() const noexcept { return failed_; }
    bool finished() const noexcept { return finished_; }
};
} // namespace synrex::generic
