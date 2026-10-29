// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Tuomas Pietilä
// Original revision-scan behavior retained; intrusive index changes victim lookup only.
#include "synrex/stream.hpp"
#include <algorithm>
#include <cmath>
#include <limits>
namespace synrex::streaming {
namespace {
using Clock = std::chrono::steady_clock;
using Time = Clock::time_point;
Time begin(bool enabled) { return enabled ? Clock::now() : Time{}; }
void add(bool enabled, Time start, double &seconds) {
    if (enabled)
        seconds += std::chrono::duration<double>(Clock::now() - start).count();
}
} // namespace
const char *map_policy_name(MapPolicy policy) {
    switch (policy) {
    case MapPolicy::reference:
        return "reference";
    case MapPolicy::indexed:
        return "indexed";
    }
    throw std::runtime_error("unknown map policy");
}
MapPolicy map_policy(const std::string &name) {
    if (name == "reference")
        return MapPolicy::reference;
    if (name == "indexed")
        return MapPolicy::indexed;
    throw std::runtime_error("unknown map policy");
}
SemanticMap::SemanticMap(double resolution, std::size_t capacity, bool profile, MapPolicy policy,
                         bool detailed)
    : resolution_(resolution), capacity_(capacity), profile_(profile), detail_(detailed),
      policy_(policy) {
    require(std::isfinite(resolution) && resolution >= 0.001 && resolution <= 1000 &&
                capacity > 0 && capacity <= 1000000,
            "unsafe semantic map budget");
    (void)map_policy_name(policy);
}
void SemanticMap::unlink(Cell &cell) {
    if (cell.previous)
        cell.previous->next = cell.next;
    else
        oldest_ = cell.next;
    if (cell.next)
        cell.next->previous = cell.previous;
    else
        newest_ = cell.previous;
    cell.previous = cell.next = nullptr;
}
void SemanticMap::refresh(Cell &cell) {
    if (newest_ == &cell)
        return;
    if (cell.previous || cell.next || oldest_ == &cell)
        unlink(cell);
    cell.previous = newest_;
    if (newest_)
        newest_->next = &cell;
    else
        oldest_ = &cell;
    newest_ = &cell;
}
std::vector<MapDelta> SemanticMap::update(const Classification &r) {
    auto tick = begin(profile_);
    validate_classification(r);
    add(profile_, tick, validation_seconds_);
    const auto &h = r.frame.header;
    const std::array<std::string, 5> identity{h.sensor_id, h.stream_id, h.coordinate_frame, h.crs,
                                              h.taxonomy};
    require(!identity_ || *identity_ == identity, "map coordinate/taxonomy identity mismatch");
    identity_ = identity;
    std::vector<MapDelta> deltas;
    for (std::size_t i = 0; i < r.frame.points.size(); ++i) {
        const auto &p = r.frame.points[i];
        Key key{};
        tick = begin(detail_);
        for (int a = 0; a < 3; ++a) {
            const auto v = std::floor(p.xyz[a] / resolution_);
            require(std::isfinite(v) && std::abs(v) <= 1e15, "map coordinate range exceeded");
            key[a] = static_cast<std::int64_t>(v);
        }
        add(detail_, tick, stage_seconds_[0]);
        tick = begin(detail_);
        auto found = cells_.find(key);
        add(detail_, tick, stage_seconds_[1]);
        ++counters_.observations;
        if (found != cells_.end() && p.timestamp_ns < found->second.timestamp_ns) {
            ++counters_.late_rejections;
            continue;
        }
        const bool inserted = found == cells_.end();
        if (inserted)
            ++counters_.insertions;
        else
            ++counters_.existing_updates;
        const auto id = r.semantic_ids[i];
        const bool changed = inserted || found->second.semantic_id != id ||
                             found->second.timestamp_ns != p.timestamp_ns;
        bool indexed = policy_ == MapPolicy::indexed && !overflow_fallback_;
        // Revision wrap is defined unsigned arithmetic in the reference. Its numeric-min
        // order then ceases to be LRU. Preserve exact behavior with a permanent scan fallback.
        if (indexed && revision_ == std::numeric_limits<std::uint64_t>::max()) {
            overflow_fallback_ = true;
            oldest_ = newest_ = nullptr;
            indexed = false;
        }
        std::optional<Key> evicted;
        if (inserted && cells_.size() == capacity_) {
            tick = begin(detail_);
            auto victim = cells_.end();
            if (indexed) {
                require(oldest_ != nullptr, "missing map eviction index");
                evicted = oldest_->key;
            } else {
                counters_.eviction_scanned_cells += cells_.size();
                victim = std::min_element(cells_.begin(), cells_.end(),
                                          [](const auto &a, const auto &b) {
                                              return a.second.revision < b.second.revision;
                                          });
                evicted = victim->first;
            }
            add(detail_, tick, stage_seconds_[4]);
            if (indexed) {
                tick = begin(detail_);
                unlink(*oldest_);
                add(detail_, tick, stage_seconds_[6]);
                tick = begin(detail_);
                cells_.erase(*evicted);
                add(detail_, tick, stage_seconds_[5]);
            } else {
                tick = begin(detail_);
                cells_.erase(victim);
                add(detail_, tick, stage_seconds_[5]);
            }
            ++counters_.removals;
        }
        tick = begin(detail_);
        if (inserted)
            found =
                cells_.emplace(key, Cell{id, p.timestamp_ns, ++revision_, nullptr, nullptr, key})
                    .first;
        else {
            found->second.semantic_id = id;
            found->second.timestamp_ns = p.timestamp_ns;
            found->second.revision = ++revision_;
        }
        add(detail_, tick, stage_seconds_[inserted ? 2 : 3]);
        if (indexed) {
            tick = begin(detail_);
            refresh(found->second);
            add(detail_, tick, stage_seconds_[6]);
        }
        counters_.peak_cells = std::max<std::uint64_t>(counters_.peak_cells, cells_.size());
        if (changed) {
            tick = begin(detail_);
            ++counters_.delta_events;
            deltas.push_back({key, id, p.timestamp_ns, evicted});
            add(detail_, tick, stage_seconds_[7]);
        }
    }
    return deltas;
}
Json SemanticMap::snapshot(std::uint64_t now) const {
    const auto tick = begin(detail_);
    Json cells = Json::array();
    for (const auto &entry : cells_) {
        require(now >= entry.second.timestamp_ns, "map age cannot be negative");
        cells.push_back({{"key", entry.first},
                         {"semantic_id", entry.second.semantic_id},
                         {"timestamp_ns", entry.second.timestamp_ns},
                         {"classification_age_ns", now - entry.second.timestamp_ns},
                         {"confidence", nullptr},
                         {"confidence_age_ns", nullptr}});
    }
    Json result{{"identity", identity_ ? Json(*identity_) : Json(nullptr)},
                {"schema", "synrex.semantic_map.v1"},
                {"resolution_metres", resolution_},
                {"capacity", capacity_},
                {"mode", "map_delta_output"},
                {"cells", cells}};
    add(detail_, tick, stage_seconds_[8]);
    return result;
}
Json SemanticMap::profile() const {
    Json stages = Json::object();
    const char *names[]{"coordinate_key",   "lookup",    "insertion",         "existing_update",
                        "victim_selection", "removal",   "index_maintenance", "delta_construction",
                        "full_snapshot",    "expiration"};
    for (std::size_t i = 0; i < stage_seconds_.size(); ++i)
        stages[names[i]] = stage_seconds_[i];
    return {
        {"requested_policy", map_policy_name(policy_)},
        {"effective_policy", overflow_fallback_ ? "reference" : map_policy_name(policy_)},
        {"detailed_timers_enabled", detail_},
        {"stage_seconds", stages},
        {"validation_seconds", validation_seconds_},
        {"expiration_enabled", false},
        {"expiration_reason", "No TTL or expiry exists in the frozen map policy"},
        {"active_cells", cells_.size()},
        {"index_entries", policy_ == MapPolicy::indexed && !overflow_fallback_ ? cells_.size() : 0},
        {"intrusive_index_bytes_per_cell", sizeof(Cell *) * 2 + sizeof(Key)},
        {"cell_payload_bytes", sizeof(Cell)},
        {"ordered_key_bytes", sizeof(Key)},
        {"node_allocator_overhead_included", false},
        {"revision_overflow_fallback", overflow_fallback_}};
}
void SemanticMap::validate_index() const {
    require(cells_.size() <= capacity_, "map capacity violated");
    if (policy_ != MapPolicy::indexed || overflow_fallback_)
        return;
    const Cell *previous = nullptr;
    std::size_t count = 0;
    for (auto cell = oldest_; cell; cell = cell->next) {
        require(count++ < cells_.size() && cell->previous == previous, "broken/cyclic map index");
        const auto found = cells_.find(cell->key);
        require(found != cells_.end() && &found->second == cell, "index points outside map");
        require(!previous || previous->revision < cell->revision, "map index recency mismatch");
        previous = cell;
    }
    require(count == cells_.size() && previous == newest_, "incomplete map index");
}
void SemanticMap::reset() {
    oldest_ = newest_ = nullptr;
    cells_.clear();
    identity_.reset();
    revision_ = 0;
    overflow_fallback_ = false;
    validation_seconds_ = 0;
    stage_seconds_ = {};
    counters_ = {};
}
} // namespace synrex::streaming
