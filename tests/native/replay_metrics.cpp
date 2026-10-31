// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Tuomas Pietilä
#include "synrex/replay_metrics.hpp"
#include <ctime>
#include <iostream>
#include <string>
#include <vector>

namespace {
using namespace synrex::replay;
unsigned checks{};
void check(bool condition, const char *message) {
    ++checks;
    if (!condition)
        throw std::runtime_error(message);
}
template <class Exception, class F> void rejects(F operation, const char *message) {
    try {
        operation();
    } catch (const Exception &error) {
        check(std::string(error.what()) == message, "unexpected metrics rejection");
        return;
    }
    throw std::runtime_error("expected metrics rejection");
}
void contains(const QuantileBounds &bounds, std::uint64_t value) {
    check(bounds.lower_ns <= value && value <= bounds.upper_ns, "quantile containment");
    check(bounds.upper_ns - bounds.lower_ns <= bounds.lower_ns / 64, "quantile error bound");
}
void histogram() {
    const auto maximum = std::numeric_limits<std::uint64_t>::max();
    LatencyHistogram empty;
    check(!empty.count() && !empty.sum_ns() && !empty.max_ns(), "empty accounting");
    check(!empty.p50() && !empty.p95() && !empty.p99(), "empty quantiles unavailable");
    rejects<std::invalid_argument>([&] { empty.percentile(0); },
                                   "latency percentile must be 1..100");
    rejects<std::invalid_argument>([&] { empty.percentile(101); },
                                   "latency percentile must be 1..100");
    rejects<std::invalid_argument>([&] { empty.add_ns(1, 0); },
                                   "latency sample occurrences must be positive");
    check(!empty.count() && !empty.sum_ns(), "invalid weight preserves accounting");

    LatencyHistogram sequence;
    for (std::uint64_t ns = 1; ns <= 100; ++ns)
        sequence.add_ns(ns);
    check(sequence.count() == 100 && sequence.sum_ns() == 5050 && sequence.max_ns() == 100,
          "exact sequence accounting");
    check(sequence.p50()->lower_ns == 50 && sequence.p50()->upper_ns == 50 &&
              sequence.p95()->lower_ns == 95 && sequence.p95()->upper_ns == 95 &&
              sequence.p99()->lower_ns == 99 && sequence.p99()->upper_ns == 99,
          "nearest-rank exact quantiles");
    LatencyHistogram tail;
    tail.add_ns(10, 99);
    tail.add_ns(1000000);
    check(tail.p99()->upper_ns == 10, "nearest-rank p99 excludes single 1 percent tail");
    contains(*tail.percentile(100), 1000000);
    check(tail.max_ns() == 1000000, "maximum is exact despite histogram bin width");

    // Every bin edge, its predecessor and its successor, including UINT64_MAX.
    for (unsigned shift = 0; shift <= 57; ++shift) {
        for (std::uint64_t mantissa = 64; mantissa < 128; ++mantissa) {
            const auto edge = mantissa << shift;
            for (const auto value : {edge - 1, edge, edge + 1}) {
                LatencyHistogram single;
                single.add_ns(value);
                contains(*single.p50(), value);
                contains(*single.p95(), value);
                contains(*single.p99(), value);
                check(single.count() == 1 && single.sum_ns() == value && single.max_ns() == value,
                      "edge accounting exact");
            }
        }
    }
    for (std::uint64_t value = 0; value < 128; ++value) {
        LatencyHistogram single;
        single.add_ns(value);
        check(single.p99()->lower_ns == value && single.p99()->upper_ns == value,
              "small nanosecond values exact");
    }
    LatencyHistogram sum_limit;
    sum_limit.add_ns(maximum);
    contains(*sum_limit.p99(), maximum);
    rejects<std::overflow_error>([&] { sum_limit.add_ns(1); }, "latency nanosecond sum overflow");
    check(sum_limit.count() == 1 && sum_limit.sum_ns() == maximum && sum_limit.max_ns() == maximum,
          "sum overflow has no partial mutation");
    contains(*sum_limit.p50(), maximum);
    sum_limit.add_ns(0);
    check(sum_limit.count() == 2 && sum_limit.sum_ns() == maximum, "zero allowed at sum limit");
    LatencyHistogram product_limit;
    rejects<std::overflow_error>([&] { product_limit.add_ns(maximum / 2 + 1, 2); },
                                 "latency nanosecond sum overflow");
    check(!product_limit.count() && !product_limit.max_ns(), "product overflow is atomic");
    product_limit.add_ns(3, maximum / 3);
    check(product_limit.sum_ns() == maximum, "weighted exact uint64 maximum sum");
    LatencyHistogram count_limit;
    count_limit.add_ns(0, maximum);
    check(count_limit.count() == maximum && count_limit.sum_ns() == 0, "uint64 maximum count");
    check(count_limit.p50()->upper_ns == 0 && count_limit.p95()->upper_ns == 0 &&
              count_limit.p99()->upper_ns == 0 && count_limit.percentile(100)->upper_ns == 0,
          "percentile ranks avoid uint64 multiplication overflow");
    rejects<std::overflow_error>([&] { count_limit.add_ns(0); }, "latency sample count overflow");
    check(count_limit.count() == maximum && count_limit.p99()->upper_ns == 0,
          "count overflow has no partial mutation");

    // Compare weighted bins against an independently sorted deterministic corpus.
    std::vector<std::uint64_t> corpus;
    LatencyHistogram mixed;
    for (std::uint64_t i = 0; i < 10003; ++i) {
        const auto value = (i * 7919) % 2000001;
        corpus.push_back(value);
        mixed.add_ns(value);
    }
    std::sort(corpus.begin(), corpus.end());
    std::uint64_t previous_lower{}, previous_upper{};
    for (unsigned percent = 1; percent <= 100; ++percent) {
        const auto quantile = *mixed.percentile(percent);
        contains(quantile, corpus[(corpus.size() * percent + 99) / 100 - 1]);
        check(quantile.lower_ns >= previous_lower && quantile.upper_ns >= previous_upper,
              "ordered percentile intervals are monotonic");
        previous_lower = quantile.lower_ns;
        previous_upper = quantile.upper_ns;
    }
}
void clocks() {
    using Time = MetricsClock::time_point;
    using Ticks = MetricsClock::duration;
    check(elapsed_ns(Time(Ticks(100)), Time(Ticks(150))) == 50, "monotonic elapsed duration");
    check(elapsed_ns(Time(Ticks(-100)), Time(Ticks(-50))) == 50, "negative clock epoch");
    check(elapsed_ns(Time(Ticks(-50)), Time(Ticks(50))) == 100, "clock epoch crossing");
    check(elapsed_ns(Time{}, Time{}) == 0, "zero service duration");
    check(elapsed_ns(Time::min(), Time::max()) == std::numeric_limits<std::uint64_t>::max(),
          "full clock span avoids signed subtraction overflow");
    rejects<std::invalid_argument>([&] { elapsed_ns(Time(Ticks(2)), Time(Ticks(1))); },
                                   "latency end precedes begin on monotonic clock");
    check(scheduling_lag_ns(Time(Ticks(100)), Time(Ticks(99))) == 0 &&
              scheduling_lag_ns(Time(Ticks(100)), Time(Ticks(100))) == 0 &&
              scheduling_lag_ns(Time(Ticks(100)), Time(Ticks(125))) == 25,
          "early/on-time/late sender scheduling");
    check(synthetic_accumulation_plus_service_ns(409600000, 1000000) == 410600000,
          "16384 observations at 40k/s plus service remains above 250 ms");
    check(synthetic_accumulation_plus_service_ns(0, 0) == 0, "zero synthetic duration");
    const auto maximum = std::numeric_limits<std::uint64_t>::max();
    check(synthetic_accumulation_plus_service_ns(maximum, 0) == maximum,
          "maximum synthetic duration");
    rejects<std::overflow_error>([&] { synthetic_accumulation_plus_service_ns(maximum, 1); },
                                 "synthetic accumulation plus service overflow");
    const auto begin = MetricsClock::now();
    const auto end = MetricsClock::now();
    check(end >= begin &&
              elapsed_ns(begin, end) == static_cast<std::uint64_t>((end - begin).count()),
          "actual steady clock smoke");
}
volatile std::uint64_t sink{};
void benchmark() {
    constexpr std::uint64_t iterations = 1000000;
    constexpr unsigned repetitions = 5;
    // Fixtures are CPU-only loops, not runtime throughput. Include histogram
    // updates and separately clock reads; queries are measured separately.
    std::cout << "fixture=bounded_metrics_cpu_v1 iterations=" << iterations
              << " repetitions=" << repetitions << " sizeof_histogram=" << sizeof(LatencyHistogram)
              << " CLOCKS_PER_SEC=" << CLOCKS_PER_SEC << '\n';
    for (unsigned repetition = 0; repetition < repetitions; ++repetition) {
        std::uint64_t baseline{};
        const auto before = std::clock();
        for (std::uint64_t i = 0; i < iterations; ++i)
            baseline += (i * 7919) % 1000000000;
        sink = baseline;
        const auto after_baseline = std::clock();
        LatencyHistogram measured;
        for (std::uint64_t i = 0; i < iterations; ++i)
            measured.add_ns((i * 7919) % 1000000000);
        sink = measured.sum_ns();
        const auto after_updates = std::clock();
        LatencyHistogram clocked;
        for (std::uint64_t i = 0; i < iterations; ++i) {
            const auto begin = MetricsClock::now();
            const auto end = MetricsClock::now();
            clocked.add_ns(elapsed_ns(begin, end));
        }
        sink = clocked.sum_ns() + clocked.p99()->upper_ns;
        const auto after_clocked = std::clock();
        constexpr std::uint64_t queries = 10000;
        for (std::uint64_t i = 0; i < queries; ++i)
            sink = measured.percentile(static_cast<unsigned>(1 + i % 100))->upper_ns;
        const auto after_queries = std::clock();
        check(before != -1 && after_baseline != -1 && after_updates != -1 && after_clocked != -1 &&
                  after_queries != -1,
              "CPU clock unavailable");
        auto ns_per = [](std::clock_t start, std::clock_t stop, std::uint64_t count) {
            return 1e9 * static_cast<double>(stop - start) / CLOCKS_PER_SEC / count;
        };
        std::cout << "repetition=" << repetition
                  << " baseline_cpu_ns_per_item=" << ns_per(before, after_baseline, iterations)
                  << " histogram_cpu_ns_per_item="
                  << ns_per(after_baseline, after_updates, iterations)
                  << " two_clocks_and_histogram_cpu_ns_per_item="
                  << ns_per(after_updates, after_clocked, iterations)
                  << " percentile_cpu_ns_per_query="
                  << ns_per(after_clocked, after_queries, queries) << '\n';
    }
}
} // namespace
int main(int argc, char **argv) {
    try {
        if (argc == 2 && std::string(argv[1]) == "--benchmark") {
            benchmark();
        } else {
            check(argc == 1, "usage: replay-metrics-tests [--benchmark]");
            histogram();
            clocks();
            std::cout << "replay_metrics=PASS checks=" << checks << '\n';
        }
        return 0;
    } catch (const std::exception &error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
