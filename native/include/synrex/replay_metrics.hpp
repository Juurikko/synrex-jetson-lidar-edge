// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Tuomas Pietilä
#pragma once
#include <algorithm>
#include <array>
#include <chrono>
#include <cstdint>
#include <limits>
#include <optional>
#include <stdexcept>
#include <type_traits>

namespace synrex::replay {
struct QuantileBounds {
    std::uint64_t lower_ns{}, upper_ns{};
};

// Single-owner, allocation-free logarithmic histogram. Each observation is one
// unweighted latency sample unless occurrences is explicitly supplied. Do not
// mix frame-weighted and point-weighted populations in one histogram.
//
// Nearest-rank quantiles are intervals containing the true sample quantile:
// 0..127 ns are exact; larger bins have width <= lower_ns/64. Use upper_ns
// for a conservative deadline comparison; neither endpoint is an exact p99.
// Count, sum and maximum are exact. Overflow throws BEFORE any state changes;
// callers must fail qualification rather than silently discard that observation.
class LatencyHistogram final {
    static constexpr auto maximum = std::numeric_limits<std::uint64_t>::max();
    std::array<std::uint64_t, 3776> bins_{};
    std::uint64_t count_{}, sum_ns_{}, max_ns_{};

    static std::size_t index(std::uint64_t ns) {
        if (ns < 64)
            return static_cast<std::size_t>(ns);
        unsigned shift{};
        auto mantissa = ns;
        while (mantissa >= 128) {
            mantissa >>= 1;
            ++shift;
        }
        return 64 + 64 * shift + static_cast<std::size_t>(mantissa - 64);
    }
    static QuantileBounds bounds(std::size_t bin) {
        if (bin < 64)
            return {bin, bin};
        const auto shift = (bin - 64) / 64;
        const auto lower = (std::uint64_t{64} + (bin - 64) % 64) << shift;
        return {lower, lower + ((std::uint64_t{1} << shift) - 1)};
    }

  public:
    static constexpr std::size_t bin_count = 3776;
    void add_ns(std::uint64_t ns, std::uint64_t occurrences = 1) {
        if (!occurrences)
            throw std::invalid_argument("latency sample occurrences must be positive");
        if (occurrences > maximum - count_)
            throw std::overflow_error("latency sample count overflow");
        if (ns && occurrences > (maximum - sum_ns_) / ns)
            throw std::overflow_error("latency nanosecond sum overflow");
        bins_[index(ns)] += occurrences;
        count_ += occurrences;
        sum_ns_ += ns * occurrences;
        max_ns_ = std::max(max_ns_, ns);
    }
    // Merge sample populations, never their reported quantile endpoints. The
    // count bound also proves every resulting bin fits. Check both bounds
    // before changing any state so failed trial aggregation remains explicit.
    void merge(const LatencyHistogram &other) {
        if (other.count_ > maximum - count_)
            throw std::overflow_error("latency sample count overflow");
        if (other.sum_ns_ > maximum - sum_ns_)
            throw std::overflow_error("latency nanosecond sum overflow");
        for (std::size_t bin = 0; bin < bins_.size(); ++bin)
            bins_[bin] += other.bins_[bin];
        count_ += other.count_;
        sum_ns_ += other.sum_ns_;
        max_ns_ = std::max(max_ns_, other.max_ns_);
    }
    std::uint64_t count() const { return count_; }
    std::uint64_t sum_ns() const { return sum_ns_; }
    std::optional<std::uint64_t> max_ns() const {
        return count_ ? std::optional<std::uint64_t>(max_ns_) : std::nullopt;
    }
    std::optional<QuantileBounds> percentile(unsigned percent) const {
        if (!percent || percent > 100)
            throw std::invalid_argument("latency percentile must be 1..100");
        if (!count_)
            return std::nullopt;
        // ceil(count*percent/100), without overflowing the product.
        const auto rank = (count_ / 100) * percent + ((count_ % 100) * percent + 99) / 100;
        std::uint64_t seen{};
        for (std::size_t bin = 0; bin < bins_.size(); ++bin) {
            seen += bins_[bin];
            if (seen >= rank)
                return bounds(bin);
        }
        throw std::logic_error("latency histogram count invariant");
    }
    std::optional<QuantileBounds> p50() const { return percentile(50); }
    std::optional<QuantileBounds> p95() const { return percentile(95); }
    std::optional<QuantileBounds> p99() const { return percentile(99); }
};

using MetricsClock = std::chrono::steady_clock;

// Receiver/service latency uses two timestamps from the SAME process's steady
// clock. Choose and report actual boundaries (e.g. frame read to output write).
// Never subtract sender/receiver clocks or add overlapping stage durations.
inline std::uint64_t elapsed_ns(MetricsClock::time_point begin, MetricsClock::time_point end) {
    static_assert(std::ratio_equal<MetricsClock::period, std::nano>::value,
                  "metrics clock requires nanosecond ticks");
    static_assert(std::is_integral<MetricsClock::rep>::value &&
                      std::is_signed<MetricsClock::rep>::value && sizeof(MetricsClock::rep) <= 8,
                  "metrics clock requires a signed at-most-64-bit tick count");
    if (end < begin)
        throw std::invalid_argument("latency end precedes begin on monotonic clock");
    // Unsigned subtraction also handles negative epochs and the full signed
    // clock range without overflowing a signed duration subtraction.
    return static_cast<std::uint64_t>(end.time_since_epoch().count()) -
           static_cast<std::uint64_t>(begin.time_since_epoch().count());
}
inline std::uint64_t scheduling_lag_ns(MetricsClock::time_point due,
                                       MetricsClock::time_point actual) {
    return actual <= due ? 0 : elapsed_ns(due, actual);
}

// A synthetic file-replay estimate only: declared frame accumulation plus
// measured service duration. It excludes unmeasured stages, transport and
// scheduling lag. Physical capture-to-output age is UNAVAILABLE without actual
// acquisition timestamps and clock provenance; never relabel this estimate.
inline std::uint64_t synthetic_accumulation_plus_service_ns(std::uint64_t accumulation_ns,
                                                            std::uint64_t service_ns) {
    if (service_ns > std::numeric_limits<std::uint64_t>::max() - accumulation_ns)
        throw std::overflow_error("synthetic accumulation plus service overflow");
    return accumulation_ns + service_ns;
}
} // namespace synrex::replay
