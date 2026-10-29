// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Tuomas Pietilä
#pragma once
#include "runtime.hpp"
#include <algorithm>
#include <array>
#include <cmath>
namespace synrex::streaming {
// Fixed 64*32 histogram: every sample is counted, upper edge has at most 3.125% relative error.
// No rolling sample truncation and no memory growth with stream duration.
class LatencyHistogram {
    std::array<std::uint64_t, 2048> bins_{};
    std::uint64_t count_{};
    long double sum_ns_{};
    std::uint64_t maximum_ns_{};
  public:
    void add(std::uint64_t ns) {
        const auto exponent = ns ? 63U - static_cast<unsigned>(__builtin_clzll(ns)) : 0;
        const auto base = std::uint64_t{1} << exponent;
        const auto fraction = ns > base ? static_cast<unsigned>((ns - base) * (32.0L / base)) : 0;
        ++bins_[exponent * 32 + std::min(fraction, 31U)];
        ++count_; sum_ns_ += ns; maximum_ns_ = std::max(maximum_ns_, ns);
    }
    double quantile(double q) const {
        if (!count_) return 0;
        const auto target = static_cast<std::uint64_t>(std::ceil(count_ * q));
        std::uint64_t cumulative = 0;
        for (std::size_t i = 0; i < bins_.size(); ++i) {
            cumulative += bins_[i];
            if (cumulative >= target)
                return std::ldexp(1.0 + double(i % 32 + 1) / 32, static_cast<int>(i / 32)) / 1e9;
        }
        return maximum_ns_ / 1e9;
    }
    Json json() const {
        return {{"count", count_}, {"p50_seconds_upper_bound", quantile(.5)},
                {"p95_seconds_upper_bound", quantile(.95)}, {"p99_seconds_upper_bound", quantile(.99)},
                {"mean_seconds", count_ ? double(sum_ns_ / count_ / 1e9) : 0},
                {"maximum_seconds", maximum_ns_ / 1e9}, {"relative_bin_width_maximum", .03125}};
    }
};
} // namespace synrex::streaming
