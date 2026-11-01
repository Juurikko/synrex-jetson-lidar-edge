// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Tuomas Pietilä
#pragma once
#include "synrex/replay_reader.hpp"
#include <algorithm>
#include <array>
#include <chrono>
#include <condition_variable>
#include <exception>
#include <limits>
#include <mutex>
#include <optional>
#include <stdexcept>

namespace synrex::replay {
class Cancelled final : public std::runtime_error {
  public:
    Cancelled() : std::runtime_error("replay queue cancelled") {}
};

struct QueueStats {
    std::size_t capacity{}, window_points{}, byte_limit{};
    std::size_t queued_items{}, queued_bytes{}, queued_allocation_bytes{};
    std::size_t high_water_items{}, high_water_bytes{}, high_water_allocation_bytes{};
    std::uint64_t producer_waits{}, consumer_waits{};
    std::uint64_t producer_wait_ns{}, consumer_wait_ns{};
    std::uint64_t oldest_age_ns{}, max_age_ns{};
    std::uint64_t discarded_items{}, discarded_points{}, discarded_allocation_bytes{};
    bool producer_waiting{}, consumer_waiting{}, closed{}, failed{};

    Json json() const {
        return {{"capacity", capacity},
                {"window_points", window_points},
                {"byte_limit", byte_limit},
                {"queued_items", queued_items},
                {"queued_bytes", queued_bytes},
                {"queued_allocation_bytes", queued_allocation_bytes},
                {"high_water_items", high_water_items},
                {"high_water_bytes", high_water_bytes},
                {"high_water_allocation_bytes", high_water_allocation_bytes},
                {"producer_waits", producer_waits},
                {"consumer_waits", consumer_waits},
                {"producer_wait_ns", producer_wait_ns},
                {"consumer_wait_ns", consumer_wait_ns},
                {"oldest_age_ns", oldest_age_ns},
                {"max_age_ns", max_age_ns},
                {"discarded_items", discarded_items},
                {"discarded_points", discarded_points},
                {"discarded_allocation_bytes", discarded_allocation_bytes},
                {"producer_waiting", producer_waiting},
                {"consumer_waiting", consumer_waiting},
                {"closed", closed},
                {"failed", failed}};
    }
};

// Single producer/single consumer, with at most two retained windows. The
// caller owns and joins threads before destruction. Caller-owned producer and
// consumer windows (and decoder internals) are additional pipeline memory.
// No observation or timing samples are retained after a window leaves.
class WindowQueue final {
    using Clock = std::chrono::steady_clock;
    struct Entry {
        Window window;
        Clock::time_point available;
    };
    const std::size_t window_points_, capacity_;
    mutable std::mutex mutex_;
    std::condition_variable changed_;
    std::array<std::optional<Entry>, 2> slots_;
    std::size_t head_{}, size_{};
    QueueStats stats_;
    bool closed_{};
    std::exception_ptr failure_;
    Clock::time_point producer_wait_start_{}, consumer_wait_start_{};

    static std::uint64_t elapsed(Clock::time_point start, Clock::time_point end) {
        return static_cast<std::uint64_t>(
            std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count());
    }
    static void add(std::uint64_t &value, std::uint64_t increment) {
        const auto maximum = std::numeric_limits<std::uint64_t>::max();
        value = increment > maximum - value ? maximum : value + increment;
    }
    void check_failure() const {
        if (failure_)
            std::rethrow_exception(failure_);
    }
    void note_age(Clock::time_point now) {
        if (size_)
            stats_.max_age_ns = std::max(stats_.max_age_ns, elapsed(slots_[head_]->available, now));
    }
    void stop(std::exception_ptr error) {
        std::lock_guard<std::mutex> lock(mutex_);
        if (!failure_)
            failure_ = std::move(error);
        closed_ = true;
        note_age(Clock::now());
        for (auto &slot : slots_) {
            if (slot) {
                add(stats_.discarded_items, 1);
                add(stats_.discarded_points, slot->window.observations.size());
                add(stats_.discarded_allocation_bytes,
                    slot->window.observations.capacity() * sizeof(Observation));
            }
            slot.reset();
        }
        size_ = stats_.queued_items = stats_.queued_bytes = stats_.queued_allocation_bytes = 0;
        changed_.notify_all();
    }

  public:
    explicit WindowQueue(std::size_t window_points, std::size_t capacity = 2)
        : window_points_(window_points), capacity_(capacity) {
        if (!window_points || window_points > 65536 || !capacity || capacity > 2)
            throw std::invalid_argument("replay queue requires 1..65536 points and 1..2 windows");
        stats_.window_points = window_points;
        stats_.capacity = capacity;
        stats_.byte_limit = capacity * window_points * sizeof(Observation);
    }
    WindowQueue(const WindowQueue &) = delete;
    WindowQueue &operator=(const WindowQueue &) = delete;

    // Ownership transfers only on successful enqueue. Capacity admission also
    // bounds backing storage for partial windows; a small size cannot hide an
    // oversized vector allocation. Empty windows are not EOF markers.
    void push(Window &&window) {
        if (window.observations.empty() || window.observations.size() > window_points_ ||
            window.observations.capacity() > window_points_) {
            const auto error = std::make_exception_ptr(std::invalid_argument(
                "replay queue window size/allocation exceeds admitted bound"));
            stop(error);
            std::rethrow_exception(error);
        }
        std::unique_lock<std::mutex> lock(mutex_);
        check_failure();
        if (!closed_ && size_ == capacity_) {
            stats_.producer_waiting = true;
            add(stats_.producer_waits, 1);
            producer_wait_start_ = Clock::now();
            changed_.wait(lock, [&] { return closed_ || size_ < capacity_; });
            add(stats_.producer_wait_ns, elapsed(producer_wait_start_, Clock::now()));
            stats_.producer_waiting = false;
        }
        check_failure();
        if (closed_)
            throw std::runtime_error("push on closed replay queue");
        const auto bytes = window.observations.size() * sizeof(Observation);
        const auto allocation = window.observations.capacity() * sizeof(Observation);
        slots_[(head_ + size_) % capacity_].emplace(Entry{std::move(window), Clock::now()});
        stats_.queued_items = ++size_;
        stats_.queued_bytes += bytes;
        stats_.queued_allocation_bytes += allocation;
        stats_.high_water_items = std::max(stats_.high_water_items, size_);
        stats_.high_water_bytes = std::max(stats_.high_water_bytes, stats_.queued_bytes);
        stats_.high_water_allocation_bytes =
            std::max(stats_.high_water_allocation_bytes, stats_.queued_allocation_bytes);
        changed_.notify_all();
    }

    // false means clean close after drain only. Failures and cancellation clear
    // queued values and always throw, including when discovered after close.
    bool pop(Window &window) {
        std::unique_lock<std::mutex> lock(mutex_);
        check_failure();
        if (!closed_ && !size_) {
            stats_.consumer_waiting = true;
            add(stats_.consumer_waits, 1);
            consumer_wait_start_ = Clock::now();
            changed_.wait(lock, [&] { return closed_ || size_; });
            add(stats_.consumer_wait_ns, elapsed(consumer_wait_start_, Clock::now()));
            stats_.consumer_waiting = false;
        }
        check_failure();
        if (!size_) {
            window = Window{};
            return false;
        }
        note_age(Clock::now());
        auto &entry = *slots_[head_];
        stats_.queued_bytes -= entry.window.observations.size() * sizeof(Observation);
        stats_.queued_allocation_bytes -=
            entry.window.observations.capacity() * sizeof(Observation);
        window = std::move(entry.window);
        slots_[head_].reset();
        head_ = (head_ + 1) % capacity_;
        stats_.queued_items = --size_;
        changed_.notify_all();
        return true;
    }

    void close() {
        std::lock_guard<std::mutex> lock(mutex_);
        closed_ = true;
        changed_.notify_all();
    }
    void fail(std::exception_ptr error) {
        if (!error)
            throw std::invalid_argument("replay queue failure needs an exception");
        stop(std::move(error));
    }
    void cancel() { stop(std::make_exception_ptr(Cancelled{})); }

    QueueStats stats() const {
        std::lock_guard<std::mutex> lock(mutex_);
        auto result = stats_;
        const auto now = Clock::now();
        result.closed = closed_;
        result.failed = bool(failure_);
        if (size_)
            result.oldest_age_ns = elapsed(slots_[head_]->available, now);
        result.max_age_ns = std::max(result.max_age_ns, result.oldest_age_ns);
        if (result.producer_waiting)
            add(result.producer_wait_ns, elapsed(producer_wait_start_, now));
        if (result.consumer_waiting)
            add(result.consumer_wait_ns, elapsed(consumer_wait_start_, now));
        return result;
    }
};
} // namespace synrex::replay
