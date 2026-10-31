// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Tuomas Pietilä
#include "synrex/generic_map.hpp"

namespace synrex::generic {
namespace {
std::size_t capacity(std::size_t value) {
    require(value > 0 && value <= SemanticMap::max_cells,
            "generic snapshot capacity outside 1..65536");
    return value;
}
Json binding(const Descriptor &descriptor) {
    return {{"descriptor_sha256", descriptor.sha256()},
            {"session_id", descriptor.spec().at("session_id")},
            {"map_epoch", descriptor.spec().at("map_epoch")},
            {"taxonomy_sha256", descriptor.taxonomy().sha256()}};
}
} // namespace
SemanticMap::SemanticMap(Descriptor descriptor, double resolution, std::size_t cells,
                         streaming::MapPolicy policy)
    : descriptor_(std::move(descriptor)), map_(resolution, capacity(cells), false, policy) {}
void SemanticMap::alive() const { require(!failed_, "generic map permanently failed"); }
Json SemanticMap::update(const Classification &result) {
    try {
        alive();
        // Encoded-byte validation is part of acceptance, not just field validation.
        (void)encode_classification(descriptor_, result);
        const auto &frame = result.frame;
        const std::array<std::string, 4> source{frame.sensor_id, frame.stream_id,
                                                frame.coordinate_frame, frame.crs};
        require(!source_ || *source_ == source, "generic map source identity changed");
        require(!previous_ ||
                    (frame.sequence > (*previous_)[0] && frame.frame_id > (*previous_)[1] &&
                     frame.frame_start_ns >= (*previous_)[2]),
                "generic map sequence/time replay or regression");
        const std::array<std::string, 5> identity{source[0], source[1], source[2], source[3],
                                                  descriptor_.sha256()};
        const auto deltas = map_.update_points(frame.points, result.semantic_ids, identity);
        source_ = source;
        previous_ = {frame.sequence, frame.frame_id, frame.frame_end_ns};
        Json entries = Json::array();
        for (const auto &delta : deltas) {
            const auto &category = descriptor_.taxonomy().classes().at(delta.semantic_id);
            entries.push_back({{"key", delta.key},
                               {"semantic_id", delta.semantic_id},
                               {"las_code", category.las_code},
                               {"timestamp_ns", delta.timestamp_ns},
                               {"evicted", delta.evicted ? Json(*delta.evicted) : Json(nullptr)}});
        }
        return {{"schema", "synrex.generic_map_delta.v1"},
                {"binding", binding(descriptor_)},
                {"source_identity", source},
                {"frame_sequence", frame.sequence},
                {"frame_id", frame.frame_id},
                {"deltas", std::move(entries)}};
    } catch (...) {
        failed_ = true;
        throw;
    }
}
Json SemanticMap::snapshot(std::uint64_t as_of_ns) {
    try {
        alive();
        auto result = map_.snapshot(as_of_ns);
        // Never expose the internal 5-string kernel identity as a legacy taxonomy.
        result.erase("identity");
        result["schema"] = "synrex.generic_map_snapshot.v1";
        result["binding"] = binding(descriptor_);
        result["descriptor_bytes"] = descriptor_.bytes();
        result["source_identity"] = source_ ? Json(*source_) : Json(nullptr);
        result["as_of_ns"] = as_of_ns;
        result["policy"] = "last_accepted_observation_revision_eviction_v1";
        require(result.dump().size() <= max_snapshot_bytes, "generic encoded snapshot byte limit");
        return result;
    } catch (...) {
        failed_ = true;
        throw;
    }
}
std::string SemanticMap::snapshot_bytes(std::uint64_t as_of_ns) {
    // snapshot validates its complete encoding; a caller never receives a partial
    // successful snapshot. Publication/transport still needs independent admission.
    return snapshot(as_of_ns).dump();
}
} // namespace synrex::generic
