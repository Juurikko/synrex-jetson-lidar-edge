// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Tuomas Pietilä
#pragma once
#include "runtime.hpp"
#include <condition_variable>
#include <chrono>
#include <deque>
#include <mutex>
#include <map>
#include <optional>
#include <set>

namespace synrex::generic {
class SemanticMap;
}
namespace synrex::streaming {
inline constexpr const char *schema = "synrex.point_stream.v1";
struct Limits {
    std::size_t max_line_bytes{1024 * 1024}, max_chunk_points{4096}, max_frame_points{65536};
    std::size_t max_chunks{1024};
    std::uint64_t max_frame_span_ns{1000000000}, assembly_timeout_ns{2000000000};
    void validate() const;
};
struct SourcePoint {
    std::uint64_t id{}, timestamp_ns{};
    std::array<double, 3> xyz{};
    std::optional<std::uint16_t> intensity;
    std::optional<std::uint8_t> return_number, return_count;
};
struct Chunk {
    std::string sensor_id, stream_id, coordinate_frame, crs, taxonomy;
    std::uint64_t sequence{}, frame_id{}, frame_start_ns{}, frame_end_ns{};
    std::size_t chunk_index{}, chunk_count{}, frame_points{};
    std::vector<SourcePoint> points;
};
// read_line caps allocation before parsing, including for an unterminated hostile record.
bool read_line(std::istream &in, std::string &line, const Limits &limits);
Chunk parse(const std::string &line, const Limits &limits = {});
std::string serialize(const Chunk &chunk, const Limits &limits = {});
struct Frame {
    Chunk header; // metadata from first chunk; header.points is always empty
    std::vector<SourcePoint> points;
    std::uint64_t first_sequence{}, last_sequence{};
    std::uint64_t accepted_steady_ns{};
    std::uint64_t submitted_steady_ns{};
    std::uint64_t replay_steady_ns{}; // R1 optional same-host replay deadline, not sensor time
    std::uint8_t coordinate_domain{1}; // R1 wire Coordinates, carried unchanged
};
struct AssemblyCounters {
    std::uint64_t incoming_points{}, assembled_points{}, dropped_points{}, late_points{};
    std::uint64_t partial_frames{}, rejected_chunks{}, missing_sequences{};
};
// One ordered sensor/stream per assembler. Reordering is explicitly rejected, never guessed.
// Timestamps are UNIX nanoseconds; assembly deadline uses a separate monotonic host clock.
class Assembler {
    Limits limits_;
    std::optional<Frame> active_;
    std::optional<Chunk> identity_;
    std::set<std::uint64_t> ids_;
    std::optional<std::uint64_t> last_sequence_, last_frame_, last_end_;
    std::size_t next_chunk_{};
    AssemblyCounters counters_;
    void discard();
  public:
    explicit Assembler(Limits limits = {});
    std::optional<Frame> accept(Chunk chunk, std::uint64_t steady_ns);
    void expire(std::uint64_t steady_ns);
    void finish();
    AssemblyCounters counters() const { return counters_; }
};

// Independent bounded input/output queues. Full policy: block producer or reject-new.
// close drains; cancel discards queued items, wakes all waiters, and returns them for accounting.
template <typename T> class Queue {
    std::size_t capacity_;
    std::deque<T> items_;
    std::deque<std::chrono::steady_clock::time_point> enqueue_times_;
    mutable std::mutex mutex_;
    std::size_t high_water_{};
    std::uint64_t waits_{}, rejected_{};
    std::condition_variable changed_;
    bool closed_{}, cancelled_{};
  public:
    explicit Queue(std::size_t capacity) : capacity_(capacity) {
        require(capacity > 0 && capacity <= 1024, "queue capacity outside 1..1024");
    }
    bool push(T value, bool block = true) {
        std::unique_lock<std::mutex> lock(mutex_);
        if (block && !closed_ && !cancelled_ && items_.size() == capacity_) ++waits_;
        if (block)
            changed_.wait(lock, [&] { return closed_ || cancelled_ || items_.size() < capacity_; });
        if (closed_ || cancelled_ || items_.size() == capacity_) {
            ++rejected_; return false;
        }
        items_.push_back(std::move(value));
        enqueue_times_.push_back(std::chrono::steady_clock::now());
        high_water_ = std::max(high_water_, items_.size());
        changed_.notify_all();
        return true;
    }
    std::optional<T> pop() {
        std::unique_lock<std::mutex> lock(mutex_);
        changed_.wait(lock, [&] { return closed_ || cancelled_ || !items_.empty(); });
        if (cancelled_ || items_.empty())
            return {};
        T item = std::move(items_.front());
        items_.pop_front();
        enqueue_times_.pop_front();
        changed_.notify_all();
        return item;
    }
    Json stats() const {
        std::lock_guard<std::mutex> lock(mutex_);
        return {{"capacity_frames", capacity_}, {"depth_frames", items_.size()},
                {"oldest_queue_wait_seconds", enqueue_times_.empty() ? 0.0 :
                    std::chrono::duration<double>(std::chrono::steady_clock::now() - enqueue_times_.front()).count()},
                {"high_water_frames", high_water_},
                {"producer_waits", waits_}, {"rejected_pushes", rejected_}};
    }
    void close() {
        std::lock_guard<std::mutex> lock(mutex_);
        closed_ = true;
        changed_.notify_all();
    }
    std::deque<T> cancel() {
        std::lock_guard<std::mutex> lock(mutex_);
        cancelled_ = true;
        std::deque<T> removed;
        removed.swap(items_);
        enqueue_times_.clear();
        changed_.notify_all();
        return removed;
    }
};

struct Classification {
    Frame frame;
    std::vector<std::uint8_t> semantic_ids, las_codes;
    Json timings;
};
void validate_classification(const Classification &result);
// Original model channel identity is authoritative; LAS projection remains explicitly separate.
Json event(const Classification &result);
struct MapDelta {
    std::array<std::int64_t, 3> key;
    std::uint8_t semantic_id;
    std::uint64_t timestamp_ns;
    std::optional<std::array<std::int64_t, 3>> evicted;
};
struct MapCounters {
    std::uint64_t observations{}, insertions{}, existing_updates{}, removals{}, late_rejections{},
                  delta_events{}, peak_cells{}, eviction_scanned_cells{};
    Json json() const { return {{"observations",observations},{"insertions",insertions},
        {"existing_updates",existing_updates},{"removals",removals},{"late_rejections",late_rejections},
        {"delta_events",delta_events},{"peak_cells",peak_cells},{"eviction_scanned_cells",eviction_scanned_cells}}; }
};
enum class MapPolicy { reference, indexed };
const char *map_policy_name(MapPolicy policy);
MapPolicy map_policy(const std::string &name);
class SemanticMap {
    using Key = std::array<std::int64_t, 3>;
    struct Cell {
        std::uint8_t semantic_id;
        std::uint64_t timestamp_ns, revision;
        Cell *previous{}, *next{};
        Key key{};
    };
    double resolution_;
    std::size_t capacity_;
    std::uint64_t revision_{};
    bool profile_{}, detail_{};
    MapPolicy policy_;
    bool overflow_fallback_{};
    Cell *oldest_{}, *newest_{};
    double validation_seconds_{};
    mutable std::array<double, 10> stage_seconds_{};
    MapCounters counters_;
    std::optional<std::array<std::string, 5>> identity_;
    std::map<Key, Cell> cells_;
    void unlink(Cell &cell);
    void refresh(Cell &cell);
    // Both callers validate their own classification contract first. Generic
    // identities use the exact descriptor digest, never a legacy taxonomy alias.
    std::vector<MapDelta> update_points(const std::vector<SourcePoint> &,
                                        const std::vector<std::uint8_t> &,
                                        const std::array<std::string, 5> &);
    friend class synrex::generic::SemanticMap;
    friend struct SemanticMapTestAccess;
  public:
    SemanticMap(double resolution, std::size_t capacity, bool profile = false,
                MapPolicy policy = MapPolicy::indexed, bool detailed = false);
    SemanticMap(const SemanticMap &) = delete;
    SemanticMap &operator=(const SemanticMap &) = delete;
    SemanticMap(SemanticMap &&) = delete;
    SemanticMap &operator=(SemanticMap &&) = delete;
    MapCounters counters() const { return counters_; }
    double validation_seconds() const { return validation_seconds_; }
    // Last accepted observation refreshes revision, including unchanged/equal-time labels.
    // Timestamp rejection is per existing cell. No TTL expiration is performed.
    std::vector<MapDelta> update(const Classification &result);
    Json snapshot(std::uint64_t as_of_ns) const;
    Json profile() const;
    void validate_index() const;
    void reset();
    std::size_t size() const { return cells_.size(); }
};
std::vector<Chunk> synthetic(std::uint64_t seed, std::size_t frames, std::size_t points_per_frame,
                             std::size_t chunk_points, const std::string &taxonomy);
} // namespace synrex::streaming
