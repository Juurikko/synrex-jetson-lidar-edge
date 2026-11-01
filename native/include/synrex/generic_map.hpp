// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Tuomas Pietilä
#pragma once
#include "synrex/generic_codec.hpp"

namespace synrex::generic {
// Last accepted observation, revision eviction and late timestamp rejection use
// the existing map kernel. No confidence, TTL, fusion or model execution implied.
// This is a CPU output consumer: constructing it does not confer GPU readiness.
class SemanticMap final {
    const Descriptor descriptor_;
    streaming::SemanticMap map_;
    std::optional<std::array<std::string, 4>> source_;
    std::optional<std::array<std::uint64_t, 3>> previous_;
    bool failed_{};
    void alive() const;

  public:
    static constexpr std::size_t max_cells = 65536;
    static constexpr std::size_t max_snapshot_bytes = 16 * 1024 * 1024;
    static constexpr std::size_t snapshot_base_bytes = 1024 * 1024;
    static constexpr std::size_t snapshot_cell_bytes = 224;
    SemanticMap(Descriptor, double resolution, std::size_t capacity,
                streaming::MapPolicy = streaming::MapPolicy::indexed);
    SemanticMap(const SemanticMap &) = delete;
    SemanticMap &operator=(const SemanticMap &) = delete;
    SemanticMap(SemanticMap &&) = delete;
    SemanticMap &operator=(SemanticMap &&) = delete;
    // Full v3 coverage/projection/identity/size validation precedes map mutation.
    // Invalid input permanently fails this consumer. A new descriptor/epoch needs
    // a new consumer; it cannot inherit cells from an earlier model or taxonomy.
    Json update(const Classification &);
    Json snapshot(std::uint64_t as_of_ns);
    std::string snapshot_bytes(std::uint64_t as_of_ns);
    streaming::MapCounters counters() const { return map_.counters(); }
    std::size_t size() const { return map_.size(); }
    bool failed() const { return failed_; }
};
static_assert(SemanticMap::snapshot_base_bytes +
                  SemanticMap::snapshot_cell_bytes * SemanticMap::max_cells <=
              SemanticMap::max_snapshot_bytes);
// Independently admit exact received bytes against an already admitted descriptor.
// Strict schema/types, duplicate keys, bounds, identities, cell order and ages are
// checked before returning a DOM. Semantic IDs refer only to the bound descriptor's
// taxonomy/LAS projection. This CPU check does not authorize execution/readiness.
Json admit_snapshot(const Descriptor &, std::string_view bytes);
} // namespace synrex::generic
