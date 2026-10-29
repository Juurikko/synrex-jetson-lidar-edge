// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Tuomas Pietilä
#pragma once
// Benchmark/test oracle: independent hash table + list, unlike the production
// ordered map + revision scan. Never feeds labels into neural classification.
#include "stream.hpp"
#include <algorithm>
#include <cmath>
#include <list>
#include <unordered_map>
namespace synrex::streaming {
class MapOracle {
    using Key = std::array<std::int64_t,3>;
    struct Hash { bool collide{}; std::size_t operator()(const Key &key) const {
        if (collide) return 0;
        std::uint64_t h=1469598103934665603ULL;
        for (auto v:key) h=(h^static_cast<std::uint64_t>(v))*1099511628211ULL;
        return h;
    }};
    struct Cell { std::uint8_t id; std::uint64_t time; std::list<Key>::iterator position; };
    std::size_t capacity_;
    std::list<Key> lru_;
    std::unordered_map<Key,Cell,Hash> cells_;
    std::optional<std::array<std::string,5>> identity_;
  public:
    std::uint64_t observations{}, insertions{}, existing{}, removals{}, late{}, peak{};
    explicit MapOracle(std::size_t capacity, bool collisions=false):capacity_(capacity),cells_(0,Hash{collisions}) { cells_.reserve(capacity); }
    // Socket consumers validate every delta, capacity policy and final snapshot;
    // they cannot inspect the producer's resident size until its snapshot arrives.
    // In-process callers additionally provide the actual size on every update.
    void verify(const Classification &r, const std::vector<MapDelta> &actual, std::size_t size=SIZE_MAX) {
        const auto &h=r.frame.header;
        const std::array<std::string,5> identity{h.sensor_id,h.stream_id,h.coordinate_frame,h.crs,h.taxonomy};
        if (identity_ && *identity_!=identity) throw std::runtime_error("oracle coordinate identity mismatch");
        identity_=identity;std::size_t delta=0;
        for (std::size_t i=0;i<r.frame.points.size();++i) {
            const auto &p=r.frame.points[i];Key key{};
            for (int a=0;a<3;++a) key[a]=static_cast<std::int64_t>(std::floor(p.xyz[a]));
            ++observations;
            auto found=cells_.find(key);
            if (found!=cells_.end() && p.timestamp_ns<found->second.time) { ++late;continue; }
            const auto id=r.semantic_ids[i];
            const bool changed=found==cells_.end() || found->second.id!=id || found->second.time!=p.timestamp_ns;
            std::optional<Key> evicted;
            if (found==cells_.end()) {
                ++insertions;
                if (cells_.size()==capacity_) {
                    evicted=lru_.front();cells_.erase(*evicted);lru_.pop_front();++removals;
                }
                lru_.push_back(key);cells_.emplace(key,Cell{id,p.timestamp_ns,std::prev(lru_.end())});
            } else {
                ++existing;
                lru_.splice(lru_.end(),lru_,found->second.position);
                found->second.id=id;found->second.time=p.timestamp_ns;
            }
            peak=std::max<std::uint64_t>(peak,cells_.size());
            if (changed) {
                if (delta>=actual.size()) throw std::runtime_error("missing map delta");
                const auto &d=actual[delta++];
                if (d.key!=key || d.semantic_id!=id || d.timestamp_ns!=p.timestamp_ns || d.evicted!=evicted)
                    throw std::runtime_error("map delta/source timestamp/class/eviction oracle mismatch");
            }
        }
        if (delta!=actual.size() || (size!=SIZE_MAX && size!=cells_.size())) throw std::runtime_error("map delta count/capacity oracle mismatch");
    }
    Json snapshot(std::uint64_t now) const {
        std::vector<Key> keys;keys.reserve(cells_.size());for (const auto &entry:cells_) keys.push_back(entry.first);
        std::sort(keys.begin(),keys.end());Json array=Json::array();
        for (const auto &key:keys) {
            const auto &cell=cells_.at(key);
            if (now<cell.time) throw std::runtime_error("oracle negative age");
            array.push_back({{"key",key},{"semantic_id",cell.id},{"timestamp_ns",cell.time},
                {"classification_age_ns",now-cell.time},{"confidence",nullptr},{"confidence_age_ns",nullptr}});
        }
        return {{"identity",identity_?Json(*identity_):Json(nullptr)},{"schema","synrex.semantic_map.v1"},
            {"resolution_metres",1.0},{"capacity",capacity_},{"mode","map_delta_output"},{"cells",array}};
    }
};
} // namespace synrex::streaming
