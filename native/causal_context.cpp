// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Tuomas Pietilä
#include "synrex/causal_context.hpp"
#include <cmath>
#include <cstring>
#include <limits>
#include <memory>
#include <openssl/crypto.h>
#include <openssl/evp.h>
#include <stdexcept>

namespace synrex::generic {
namespace {
void demand(bool yes, const char *why) {
    if (!yes)
        throw std::runtime_error(why);
}
bool same_identity(const FrameIdentity &a, const FrameIdentity &b) {
    return a.descriptor_sha256 == b.descriptor_sha256 && a.session_id == b.session_id &&
           a.map_epoch == b.map_epoch;
}
bool same_header(const SourceHeader &a, const SourceHeader &b) {
    return a.records == b.records && a.wire_begin == b.wire_begin && a.las14 == b.las14;
}

// A restrictive set of projected CRS identifiers with metre horizontal units.
// This is a *declared-frame* admission check, not independent WKT inspection.
bool known_projected_metre_crs(const std::string &crs) {
    if (crs.size() <= 5 || crs.compare(0, 5, "EPSG:") != 0)
        return false;
    unsigned code = 0;
    for (std::size_t i = 5; i < crs.size(); ++i) {
        if (crs[i] < '0' || crs[i] > '9' || code > 99999)
            return false;
        code = code * 10 + static_cast<unsigned>(crs[i] - '0');
    }
    return code == 3067 || code == 2154 || code == 2393 || code == 3006 ||
           (code >= 25828 && code <= 25838) || (code >= 32601 && code <= 32660) ||
           (code >= 32701 && code <= 32760);
}

// Stable endian-independent SHA-256 of EVERY field that a downstream model
// could use. The private expected hash never leaves the assembler's queue.
class WindowDigest {
    std::unique_ptr<EVP_MD_CTX, decltype(&EVP_MD_CTX_free)> ctx_{EVP_MD_CTX_new(), EVP_MD_CTX_free};

  public:
    WindowDigest() {
        demand(ctx_ && EVP_DigestInit_ex(ctx_.get(), EVP_sha256(), nullptr) == 1,
               "causal context window digest initialization failed");
    }
    void bytes(const void *data, std::size_t count) {
        demand(EVP_DigestUpdate(ctx_.get(), data, count) == 1,
               "causal context window digest update failed");
    }
    void integer(std::uint64_t value) {
        std::array<unsigned char, 8> le{};
        for (unsigned i = 0; i < le.size(); ++i)
            le[i] = static_cast<unsigned char>((value >> (8 * i)) & 255);
        bytes(le.data(), le.size());
    }
    void text(const std::string &s) {
        integer(s.size());
        bytes(s.data(), s.size());
    }
    void floating(double x) {
        static_assert(sizeof(double) == sizeof(std::uint64_t));
        std::uint64_t bits = 0;
        std::memcpy(&bits, &x, sizeof(bits));
        integer(bits);
    }
    template <class T> void optional(const std::optional<T> &value) {
        integer(value ? 1 : 0);
        if (value)
            integer(*value);
    }
    std::array<unsigned char, 32> finish() {
        std::array<unsigned char, 32> result{};
        unsigned length = 0;
        demand(EVP_DigestFinal_ex(ctx_.get(), result.data(), &length) == 1 &&
                   length == result.size(),
               "causal context window digest finalization failed");
        return result;
    }
};

std::array<unsigned char, 32> window_digest(const CausalContextWindow &w) {
    WindowDigest h;
    h.text("synrex.causal_context.issued_window.v1");
    h.text(w.descriptor_sha256);
    h.text(w.session_id);
    h.text(w.stream_id);
    h.text(w.sensor_id);
    h.text(w.coordinate_frame);
    h.text(w.crs);
    h.integer(w.map_epoch);
    h.integer(w.window_index);
    h.integer(w.cutoff_frame_end_ns);
    h.integer(w.fresh_begin);
    h.integer(w.fresh_count);
    h.integer(w.full_history ? 1 : 0);
    for (double xy : w.fixed_world_origin_xy_m)
        h.floating(xy);
    h.floating(w.world_grid_size_m);
    h.integer(w.context.size());
    for (const auto &p : w.context) {
        h.integer(p.id);
        h.integer(p.timestamp_ns);
        for (double xyz : p.xyz)
            h.floating(xyz);
        h.optional(p.intensity);
        h.optional(p.return_number);
        h.optional(p.return_count);
    }
    return h.finish();
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
               world_grid_size_m <= 1000.0,
           "causal context grid size invalid");
    demand(known_projected_metre_crs(projected_crs),
           "causal context requires an explicitly admitted projected-metre EPSG CRS");
    demand(std::isfinite(max_horizontal_offset_m) && max_horizontal_offset_m >= 1.0 &&
               max_horizontal_offset_m <= 1000000.0,
           "causal context horizontal reach invalid");
    demand(std::isfinite(max_abs_elevation_m) && max_abs_elevation_m >= 1.0 &&
               max_abs_elevation_m <= 100000.0,
           "causal context vertical elevation bound invalid");
    for (const auto c : fixed_world_origin_xy_m)
        demand(std::isfinite(c) && std::abs(c) <= 1e9,
               "causal context fixed-world XY origin invalid");
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
    demand(frame.crs == policy_.projected_crs,
           "causal context source CRS is not the admitted projected-metre frame");
    demand(frame.frame_end_ns >= frame.frame_start_ns, "causal context inverted frame time");
    demand(frame.points.size() <= 65536, "causal context oversized ingress frame");
    demand(accepted_points_ <= UINT64_MAX - frame.points.size(),
           "causal context point count overflow");
    if (have_last_) {
        demand(same_identity(frame.identity, *identity_) && frame.sensor_id == sensor_id_ &&
                   frame.stream_id == stream_id_ && frame.coordinate_frame == coordinate_frame_ &&
                   frame.crs == crs_,
               "causal context crossed source/session identity");
        demand(last_sequence_ < UINT64_MAX && frame.sequence == last_sequence_ + 1,
               "causal context missing or replayed frame sequence");
        demand(frame.frame_end_ns >= last_end_ns_,
               "causal context frame availability moved backwards");
    }

    const auto first_id = have_last_ ? next_id_ : frame.points.front().id;
    demand(first_id <= UINT64_MAX - frame.points.size(), "causal context source point ID overflow");
    for (std::size_t i = 0; i < frame.points.size(); ++i) {
        const auto &p = frame.points[i];
        demand(p.id == first_id + i, "causal context source index gap, duplicate or reorder");
        for (const auto c : p.xyz)
            demand(std::isfinite(c), "causal context nonfinite coordinate");
        demand(std::abs(p.xyz[0] - policy_.fixed_world_origin_xy_m[0]) <=
                       policy_.max_horizontal_offset_m &&
                   std::abs(p.xyz[1] - policy_.fixed_world_origin_xy_m[1]) <=
                       policy_.max_horizontal_offset_m &&
                   std::abs(p.xyz[2]) <= policy_.max_abs_elevation_m,
               "causal context point outside admitted metric spatial bounds");
    }

    if (policy_.require_exact_source_rows) {
        demand(frame.source_header.has_value() && frame.source_rows.size() == frame.points.size(),
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
    out.fixed_world_origin_xy_m = policy_.fixed_world_origin_xy_m;
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
            while (past_.size() > policy_.history_points)
                past_.pop_front();
        }
        pending_.pop_front();
    }
    const auto digest = window_digest(out);
    in_flight_.push_back({window_index_, out.context[out.fresh_begin].id, fresh, digest});
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
        if (policy_.require_exact_source_rows)
            source_header_ = frame.source_header;
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

CausalContextAssembler::IngestResult CausalContextAssembler::try_ingest(const Frame &frame) {
    try {
        // Invalid content is terminal even if capacity is exhausted.
        check_frame(frame);
    } catch (...) {
        failed_ = true;
        throw;
    }
    const auto complete = (pending_.size() + frame.points.size()) / policy_.fresh_points;
    if (complete > policy_.max_inflight_windows - in_flight_.size())
        return {}; // explicitly retryable; not a single point or counter was consumed
    try {
        IngestResult result;
        result.accepted = true;
        result.windows.reserve(complete);
        append_accepted(frame, result.windows);
        return result;
    } catch (...) {
        failed_ = true;
        throw;
    }
}

std::vector<CausalContextWindow> CausalContextAssembler::ingest(const Frame &frame) {
    auto result = try_ingest(frame);
    demand(result.accepted, "causal context capacity busy; retry the exact input frame");
    return std::move(result.windows);
}

CausalContextAssembler::EofResult CausalContextAssembler::try_source_eof() {
    try {
        demand(!failed_ && !finished_ && have_last_, "causal context invalid/duplicate source EOF");
        if (policy_.require_exact_source_rows)
            demand(source_header_ && accepted_points_ == source_header_->records,
                   "causal context premature source EOF");
    } catch (...) {
        failed_ = true;
        throw;
    }
    if (!pending_.empty() && in_flight_.size() == policy_.max_inflight_windows)
        return {}; // retain pending EOF points; retry after a commit
    try {
        EofResult result;
        result.accepted = true;
        if (!pending_.empty())
            result.window = emit(pending_.size());
        finished_ = true;
        return result;
    } catch (...) {
        failed_ = true;
        throw;
    }
}

std::optional<CausalContextWindow> CausalContextAssembler::source_eof() {
    auto result = try_source_eof();
    demand(result.accepted, "causal context EOF capacity busy; retry after commit");
    return std::move(result.window);
}

NewOnlyClassification
CausalContextAssembler::commit_new_only(const CausalContextWindow &window,
                                        const std::vector<std::uint8_t> &all_context_labels) {
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
                   window.fixed_world_origin_xy_m == policy_.fixed_world_origin_xy_m &&
                   window.world_grid_size_m == policy_.world_grid_size_m,
               "causal context output identity/anchor mismatch");
        const auto actual_digest = window_digest(window);
        demand(CRYPTO_memcmp(actual_digest.data(), token.window_sha256.data(),
                             actual_digest.size()) == 0,
               "causal context full issued window content changed before commit");
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
