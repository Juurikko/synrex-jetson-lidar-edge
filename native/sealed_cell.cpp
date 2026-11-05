// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Tuomas Pietilä
#include "synrex/sealed_cell.hpp"
#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>

namespace synrex::generic {
namespace {
void demand(bool okay, const char *message) {
    if (!okay)
        throw std::runtime_error(message);
}
} // namespace

void SealedCellPolicy::validate() const {
    demand(idle_gap_points >= 1 && idle_gap_points <= 10000000ULL,
           "sealed cell idle gap outside admitted range");
    demand(max_active_cells >= 1 && max_active_cells <= 65536 && max_retained_points >= 1 &&
               max_retained_points <= 8000000 && max_points_per_cell >= 1 &&
               max_points_per_cell <= 262144 && max_points_per_cell <= max_retained_points,
           "sealed cell bounded retained memory policy invalid");
    demand(max_outstanding_windows >= 1 && max_outstanding_windows <= 128,
           "sealed cell outstanding-window cap invalid");
    demand(std::isfinite(cell_width_m) && cell_width_m >= 0.1 && cell_width_m <= 1000.0,
           "sealed cell size invalid");
    CausalContextPolicy p;
    p.fresh_points = 1;
    p.history_points = 0;
    p.max_context_points = 1;
    p.projected_crs = projected_crs;
    p.fixed_world_origin_xy_m = origin_xy_m;
    p.max_horizontal_offset_m = max_xy_distance_m;
    p.semantic_class_count = semantic_class_count;
    p.validate();
}

SealedCellScheduler::SealedCellScheduler(SealedCellPolicy policy) : policy_(std::move(policy)) {
    policy_.validate();
}

void SealedCellScheduler::validate_point(std::uint64_t index, const streaming::SourcePoint &point,
                                         const std::string &source_crs) const {
    demand(!failed_ && !finished_, "sealed cell session failed or already EOF");
    demand(index == next_index_, "sealed cell missing or duplicate global stream index");
    demand(source_crs == policy_.projected_crs, "sealed cell unadmitted metre CRS");
    if (have_source_) {
        demand(first_id_ <= UINT64_MAX - index && point.id == first_id_ + index,
               "sealed cell source ID not contiguous");
    } else {
        demand(point.id <= UINT64_MAX - index, "sealed cell source ID range overflow");
    }
    for (const auto c : point.xyz)
        demand(std::isfinite(c), "sealed cell nonfinite point coordinate");
    demand(std::abs(point.xyz[0] - policy_.origin_xy_m[0]) <= policy_.max_xy_distance_m &&
               std::abs(point.xyz[1] - policy_.origin_xy_m[1]) <= policy_.max_xy_distance_m &&
               std::abs(point.xyz[2]) <= 25000.0,
           "sealed cell coordinates outside admitted metre-domain bounds");
}

SealedCellScheduler::Key SealedCellScheduler::cell_of(const streaming::SourcePoint &point) const {
    const auto x = std::floor((point.xyz[0] - policy_.origin_xy_m[0]) / policy_.cell_width_m);
    const auto y = std::floor((point.xyz[1] - policy_.origin_xy_m[1]) / policy_.cell_width_m);
    demand(x >= -10000000 && x <= 10000000 && y >= -10000000 && y <= 10000000,
           "sealed cell grid key overflow");
    return {static_cast<std::int64_t>(x), static_cast<std::int64_t>(y)};
}

bool SealedCellScheduler::seal_expired(std::uint64_t watermark, bool eof, Result &result) {
    // pending_ is ordered by last_seen, so only its front can have expired.
    while (!pending_.empty()) {
        const Key key = pending_.front();
        const auto last = cells_.at(key).last_seen_index;
        demand(last < watermark || eof, "sealed cell watermark must follow received index");
        if (!eof && watermark - last <= policy_.idle_gap_points)
            return true;
        if (issued_.size() == policy_.max_outstanding_windows)
            return false;
        result.issued.push_back(seal(key, watermark, eof, false));
    }
    return true;
}

SealedCellWindow SealedCellScheduler::seal(const Key &key, std::uint64_t watermark, bool eof,
                                           bool forced) {
    auto &cell = cells_.at(key);
    demand(cell.emitted_count < cell.points.size(), "sealed cell no fresh points");
    demand(issued_.size() < policy_.max_outstanding_windows,
           "sealed cell issuing beyond bounded outstanding capacity");
    SealedCellWindow out;
    out.cell = key;
    out.epoch = cell.epoch;
    out.window_index = windows_++;
    out.release_point_index =
        eof || forced ? watermark : cell.last_seen_index + policy_.idle_gap_points + 1;
    out.forced_release = forced;
    out.cell_history_truncated = cell.history_truncated;
    out.cell_origin_xy_m = {
        {policy_.origin_xy_m[0] + static_cast<double>(key[0]) * policy_.cell_width_m,
         policy_.origin_xy_m[1] + static_cast<double>(key[1]) * policy_.cell_width_m}};
    if (forced)
        ++forced_releases_;
    out.fresh_begin = cell.emitted_count;
    out.fresh_count = cell.points.size() - cell.emitted_count;
    out.previous_history_may_be_lost = may_have_lost_history_;
    out.eof_release = eof;
    out.context = std::make_shared<const std::vector<streaming::SourcePoint>>(cell.points);
    cell.emitted_count = cell.points.size();
    ++cell.outstanding;
    pending_.erase(cell.pending_at);
    cell.in_pending = false;
    issued_.push_back({out});
    return out;
}

bool SealedCellScheduler::reserve_one(const Key &target, std::uint64_t watermark, Result &result) {
    for (;;) {
        const auto found = cells_.find(target);
        const bool needs_cell = found == cells_.end();
        const bool local_full =
            !needs_cell && found->second.points.size() >= policy_.max_points_per_cell;
        const bool global_full = retained_points_ >= policy_.max_retained_points;
        const bool cells_full = needs_cell && cells_.size() >= policy_.max_active_cells;
        if (!local_full && !global_full && !cells_full)
            return true;

        auto victim = cells_.end();
        for (auto it = cells_.begin(); it != cells_.end(); ++it) {
            const auto &c = it->second;
            if (c.outstanding || c.emitted_count != c.points.size())
                continue; // never evict unsent or unacknowledged classifications
            if (local_full && it->first != target)
                continue; // local cap requires this exact completed cell to retire
            if (victim == cells_.end() || c.last_seen_index < victim->second.last_seen_index)
                victim = it;
        }
        if (victim != cells_.end()) {
            retained_points_ -= victim->second.points.size();
            if (evicted_keys_.size() < 65536)
                evicted_keys_.insert(victim->first);
            else
                tombstones_overflowed_ = true;
            cells_.erase(victim);
            ++evictions_;
            may_have_lost_history_ = true;
            continue;
        }
        // Nothing can be retired. Commits may free a cell, so try those first. With
        // none outstanding, waiting for G more points cannot help, because the next
        // point is the one being held. Seal early so that a commit frees something.
        // The window is causal, since it holds only points already received.
        if (issued_.empty()) {
            const bool own_pending = local_full && found->second.in_pending;
            if (!local_full && !pending_.empty())
                result.issued.push_back(seal(pending_.front(), watermark, false, true));
            else if (own_pending)
                result.issued.push_back(seal(target, watermark, false, true));
        }
        return false;
    }
}

SealedCellScheduler::Result SealedCellScheduler::try_ingest(std::uint64_t global_index,
                                                            const streaming::SourcePoint &point,
                                                            const std::string &source_crs) {
    try {
        validate_point(global_index, point, source_crs);
        const auto key = cell_of(point);
        Result result;
        if (held_) {
            demand(point.id == held_point_.id && point.xyz == held_point_.xyz &&
                       point.timestamp_ns == held_point_.timestamp_ns &&
                       point.intensity == held_point_.intensity &&
                       point.return_number == held_point_.return_number &&
                       point.return_count == held_point_.return_count,
                   "sealed cell retried point differs from the held point");
        }
        // Idle generations become causal only when the global stream index
        // passes last_cell_point + G, not when future cell points are inspected.
        if (!seal_expired(global_index, false, result) || !reserve_one(key, global_index, result)) {
            ++rejected_capacity_;
            demand(!issued_.empty(), "sealed cell internal liveness violation");
            result.needs_commit = true;
            held_ = true;
            held_point_ = point;
            return result; // global_index point not consumed; retry same point
        }
        held_ = false;
        if (!have_source_) {
            first_id_ = point.id;
            have_source_ = true;
        }
        auto &cell = cells_[key];
        if (cell.points.empty()) {
            cell.epoch = ++epochs_; // unique session-local cell instance identity
            cell.history_truncated = tombstones_overflowed_ || evicted_keys_.count(key) > 0;
        }
        cell.points.push_back(point);
        cell.last_seen_index = global_index;
        if (cell.in_pending) {
            pending_.splice(pending_.end(), pending_, cell.pending_at);
        } else {
            cell.pending_at = pending_.insert(pending_.end(), key);
            cell.in_pending = true;
        }
        ++retained_points_;
        ++next_index_;
        result.accepted = true;
        return result;
    } catch (...) {
        failed_ = true;
        throw;
    }
}

SealedCellScheduler::Result SealedCellScheduler::try_source_eof(std::uint64_t total_points) {
    try {
        demand(!failed_ && !finished_ && have_source_ && total_points == next_index_,
               "sealed cell EOF source coverage invalid");
        Result result;
        if (!seal_expired(total_points, true, result)) {
            ++rejected_capacity_;
            result.needs_commit = true;
            return result;
        }
        result.accepted = true;
        finished_ = true;
        return result;
    } catch (...) {
        failed_ = true;
        throw;
    }
}

SealedCellLabels SealedCellScheduler::commit_new_only(const SealedCellWindow &window,
                                                      const std::vector<std::uint8_t> &labels) {
    try {
        demand(!failed_ && !issued_.empty(), "sealed cell no authorized output to commit");
        const auto &issued = issued_.front().window;
        demand(window.context && window.context == issued.context && window.cell == issued.cell &&
                   window.epoch == issued.epoch && window.window_index == issued.window_index &&
                   window.release_point_index == issued.release_point_index &&
                   window.fresh_begin == issued.fresh_begin &&
                   window.fresh_count == issued.fresh_count &&
                   window.previous_history_may_be_lost == issued.previous_history_may_be_lost &&
                   window.eof_release == issued.eof_release &&
                   window.forced_release == issued.forced_release &&
                   window.cell_history_truncated == issued.cell_history_truncated &&
                   window.cell_origin_xy_m == issued.cell_origin_xy_m &&
                   window.context->size() == labels.size() &&
                   window.fresh_count == window.context->size() - window.fresh_begin,
               "sealed cell source window identity or size changed");
        for (const auto code : labels)
            demand(code < policy_.semantic_class_count, "sealed cell invalid semantic class");
        auto &cell = cells_.at(window.cell);
        demand(cell.epoch == window.epoch && cell.outstanding > 0,
               "sealed cell epoch lost before commit");
        SealedCellLabels out;
        out.window_index = window.window_index;
        out.source_ids.reserve(window.fresh_count);
        out.semantic_ids.reserve(window.fresh_count);
        for (std::size_t i = window.fresh_begin; i < window.context->size(); ++i) {
            out.source_ids.push_back(window.context->at(i).id);
            out.semantic_ids.push_back(labels[i]);
        }
        --cell.outstanding;
        committed_points_ += window.fresh_count;
        issued_.pop_front();
        return out;
    } catch (...) {
        failed_ = true;
        throw;
    }
}
} // namespace synrex::generic
