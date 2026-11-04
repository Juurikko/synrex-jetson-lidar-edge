// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Tuomas Pietilä
#include "synrex/causal_context.hpp"
#include <cmath>
#include <limits>
#include <stdexcept>

namespace synrex::generic {
namespace {
void demand(bool yes, const char *why) {
    if (!yes) throw std::runtime_error(why);
}
bool same_identity(const FrameIdentity &a, const FrameIdentity &b) {
    return a.descriptor_sha256 == b.descriptor_sha256 &&
           a.session_id == b.session_id && a.map_epoch == b.map_epoch;
}
bool same_header(const SourceHeader &a, const SourceHeader &b) {
    return a.records == b.records && a.wire_begin == b.wire_begin &&
           a.las14 == b.las14;
}
} // namespace

void CausalContextPolicy::validate() const {
    demand(fresh_points >= 1 && fresh_points <= 131072,
           "causal context fresh window outside 1..131072");
    demand(history_points <= 262144 && max_context_points <= 262144 &&
           max_context_points >= fresh_points &&
           history_points <= max_context_points - fresh_points,
           "causal context total history/fresh budget exceeded");
    demand(max_inflight_windows >= 1 && max_inflight_windows <= 8,
           "causal context in-flight bound outside 1..8");
    demand(semantic_class_count >= 2 && semantic_class_count <= 255,
           "causal context semantic class count outside 2..255");
    demand(std::isfinite(world_grid_size_m) && world_grid_size_m > 0.0 &&
           world_grid_size_m <= 1000.0, "causal context grid size invalid");
    for (const auto c : fixed_world_origin_m)
        demand(std::isfinite(c) && std::abs(c) <= 1e9,
               "causal context fixed-world origin invalid");
}

CausalContextAssembler::CausalContextAssembler(CausalContextPolicy policy) : policy_(policy) {
    policy_.validate();
}

void CausalContextAssembler::check_frame(const Frame &frame) const {
    demand(!failed_ && !finished_, "causal context session poisoned or EOF");
    demand(!frame.points.empty(), "causal context empty accepted frame");
    demand(!frame.identity.descriptor_sha256.empty() && !frame.identity.session_id.empty() &&
           !frame.sensor_id.empty() && !frame.stream_id.empty() &&
           !frame.coordinate_frame.empty() && !frame.crs.empty(),
           "causal context missing source/session identity");
    demand(frame.frame_end_ns >= frame.frame_start_ns, "causal context inverted frame time");
    demand(frame.points.size() <= 65536, "causal context oversized ingress frame");
    demand(accepted_points_ <= UINT64_MAX - frame.points.size(),
           "causal context point count overflow");
    if (have_last_) {
        demand(same_identity(frame.identity, *identity_) &&
               frame.sensor_id == sensor_id_ && frame.stream_id == stream_id_ &&
               frame.coordinate_frame == coordinate_frame_ && frame.crs == crs_,
               "causal context crossed source/session identity");
        demand(last_sequence_ < UINT64_MAX && frame.sequence == last_sequence_ + 1,
               "causal context missing or replayed frame sequence");
        demand(frame.frame_end_ns >= last_end_ns_,
               "causal context frame availability moved backwards");
    }

    const auto first_id = have_last_ ? next_id_ : frame.points.front().id;
    demand(first_id <= UINT64_MAX - frame.points.size(),
           "causal context source point ID overflow");
    for (std::size_t i = 0; i < frame.points.size(); ++i) {
        const auto &p = frame.points[i];
        demand(p.id == first_id + i, "causal context source index gap, duplicate or reorder");
        for (const auto c : p.xyz)
            demand(std::isfinite(c), "causal context nonfinite coordinate");
    }

    if (policy_.require_exact_source_rows) {
        demand(frame.source_header.has_value() &&
               frame.source_rows.size() == frame.points.size(),
               "causal context exact source-v1 rows required");
        const auto &header = *frame.source_header;
        demand(header.records > 0 && header.records >= accepted_points_ + frame.points.size(),
               "causal context source count exceeded");
        demand(!source_header_.has_value() || same_header(*source_header_, header),
               "causal context source header changed");
        for (std::size_t i = 0; i < frame.points.size(); ++i) {
            const auto index = next_original_index_ + i;
            demand(frame.source_rows[i].original_record_index == index,
                   "causal context original record index reordered");
            demand(header.wire_begin <= UINT64_MAX - index &&
                   frame.points[i].id == header.wire_begin + index,
                   "causal context source row/wire ID disagrees");
        }
    } else {
        // v3 identity-only mode deliberately cannot claim exact LAS source proof.
        demand(!frame.source_header && frame.source_rows.empty(),
               "causal context cannot mix identity-only and exact source records");
    }
}

CausalContextWindow CausalContextAssembler::emit(std::size_t fresh) {
    demand(fresh > 0 && fresh <= pending_.size(), "invalid causal context fresh prefix");
    demand(in_flight_.size() < policy_.max_inflight_windows,
           "causal context outstanding-window budget exhausted");
    CausalContextWindow out;
    out.descriptor_sha256 = identity_->descriptor_sha256;
    out.session_id = identity_->session_id;
    out.map_epoch = identity_->map_epoch;
    out.stream_id = stream_id_;
    out.sensor_id = sensor_id_;
    out.coordinate_frame = coordinate_frame_;
    out.crs = crs_;
    out.window_index = window_index_;
    out.cutoff_frame_end_ns = cutoff_ns_; // input publisher's admission boundary, NOT physical time
    out.fixed_world_origin_m = policy_.fixed_world_origin_m;
    out.world_grid_size_m = policy_.world_grid_size_m;
    out.fresh_begin = past_.size();
    out.fresh_count = fresh;
    out.full_history = past_.size() == policy_.history_points;
    out.context.reserve(out.fresh_begin + fresh);
    out.context.insert(out.context.end(), past_.begin(), past_.end());
    for (std::size_t i = 0; i < fresh; ++i) {
        out.context.push_back(pending_.front());
        if (policy_.history_points) {
            past_.push_back(pending_.front());
            while (past_.size() > policy_.history_points) past_.pop_front();
        }
        pending_.pop_front();
    }
    in_flight_.push_back({window_index_, out.context[out.fresh_begin].id, fresh});
    ++window_index_;
    return out;
}

void CausalContextAssembler::append_accepted(const Frame &frame,
                                              std::vector<CausalContextWindow> &out) {
    if (!have_last_) {
        identity_ = frame.identity;
        stream_id_ = frame.stream_id;
        sensor_id_ = frame.sensor_id;
        coordinate_frame_ = frame.coordinate_frame;
        crs_ = frame.crs;
        if (policy_.require_exact_source_rows) source_header_ = frame.source_header;
    }
    last_sequence_ = frame.sequence;
    last_end_ns_ = frame.frame_end_ns;
    cutoff_ns_ = frame.frame_end_ns;
    next_id_ = frame.points.back().id + 1;
    accepted_points_ += frame.points.size();
    next_original_index_ += frame.points.size();
    have_last_ = true;
    for (const auto &p : frame.points) {
        pending_.push_back(p);
        if (pending_.size() == policy_.fresh_points)
            out.push_back(emit(policy_.fresh_points));
    }
}

std::vector<CausalContextWindow> CausalContextAssembler::ingest(const Frame &frame) {
    try {
        check_frame(frame);
        const auto complete = (pending_.size() + frame.points.size()) / policy_.fresh_points;
        demand(complete <= policy_.max_inflight_windows - in_flight_.size(),
               "causal context new windows exceed outstanding budget");
        std::vector<CausalContextWindow> out;
        out.reserve(complete);
        append_accepted(frame, out);
        return out;
    } catch (...) {
        failed_ = true;
        throw;
    }
}

std::optional<CausalContextWindow> CausalContextAssembler::source_eof() {
    try {
        demand(!failed_ && !finished_ && have_last_,
               "causal context invalid/duplicate source EOF");
        if (policy_.require_exact_source_rows)
            demand(source_header_ && accepted_points_ == source_header_->records,
                   "causal context premature source EOF");
        demand(pending_.empty() || in_flight_.size() < policy_.max_inflight_windows,
               "causal context EOF cannot publish another in-flight window");
        finished_ = true;
        if (pending_.empty()) return {};
        return emit(pending_.size());
    } catch (...) {
        failed_ = true;
        throw;
    }
}

NewOnlyClassification CausalContextAssembler::commit_new_only(
    const CausalContextWindow &window, const std::vector<std::uint8_t> &all_context_labels) {
    try {
        demand(!failed_ && !in_flight_.empty(), "causal context has no commit-ready window");
        const auto &token = in_flight_.front();
        demand(window.window_index == token.window_index &&
               window.fresh_count == token.fresh_count &&
               window.context.size() == all_context_labels.size() &&
               window.fresh_begin <= window.context.size() &&
               window.fresh_count == window.context.size() - window.fresh_begin,
               "causal context classifier output/window mismatch");
        demand(same_identity({window.descriptor_sha256, window.session_id, window.map_epoch},
                             *identity_) &&
               window.stream_id == stream_id_ && window.sensor_id == sensor_id_ &&
               window.coordinate_frame == coordinate_frame_ && window.crs == crs_ &&
               window.fixed_world_origin_m == policy_.fixed_world_origin_m &&
               window.world_grid_size_m == policy_.world_grid_size_m,
               "causal context output identity/anchor mismatch");
        NewOnlyClassification result;
        result.window_index = window.window_index;
        result.source_ids.reserve(window.fresh_count);
        result.semantic_ids.reserve(window.fresh_count);
        for (std::size_t i = 0; i < all_context_labels.size(); ++i)
            demand(all_context_labels[i] < policy_.semantic_class_count,
                   "causal context invalid model class, no implicit fallback");
        for (std::size_t i = 0; i < window.fresh_count; ++i) {
            const auto idx = window.fresh_begin + i;
            demand(token.first_new_id <= UINT64_MAX - i &&
                   window.context[idx].id == token.first_new_id + i,
                   "causal context committed point identity mismatch");
            result.source_ids.push_back(window.context[idx].id);
            result.semantic_ids.push_back(all_context_labels[idx]);
        }
        in_flight_.pop_front();
        ++committed_windows_;
        return result;
    } catch (...) {
        failed_ = true;
        throw;
    }
}
} // namespace synrex::generic
