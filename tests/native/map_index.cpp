// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Tuomas Pietilä
// Differential reference scan vs intrusive index plus separately allocated hash/list oracle.
#include "synrex/map_oracle.hpp"
#include <cmath>
#include <fstream>
#include <iostream>
#include <limits>
#include <unistd.h>
using namespace synrex;
using namespace synrex::streaming;
namespace synrex::streaming {
struct SemanticMapTestAccess {
    static void near_wrap(SemanticMap &map) { map.revision_ = UINT64_MAX - 1; }
};
} // namespace synrex::streaming
namespace {
std::uint64_t resident() {
    std::ifstream f("/proc/self/statm");
    std::uint64_t total{}, pages{};
    require(bool(f >> total >> pages), "RSS unavailable");
    return pages * static_cast<std::uint64_t>(sysconf(_SC_PAGESIZE));
}
void same(const std::vector<MapDelta> &a, const std::vector<MapDelta> &b) {
    require(a.size() == b.size(), "delta count differs");
    for (std::size_t i = 0; i < a.size(); ++i)
        require(a[i].key == b[i].key && a[i].semantic_id == b[i].semantic_id &&
                    a[i].timestamp_ns == b[i].timestamp_ns && a[i].evicted == b[i].evicted,
                "exact delta differs");
}
Classification frame(std::size_t start, std::size_t n, std::uint64_t stamp, bool repeats = false) {
    Classification r;
    auto &h = r.frame.header;
    h.sensor_id = "cpu-fixture";
    h.stream_id = "exact-map-index";
    h.coordinate_frame = "fixed-world";
    h.crs = "LOCAL:metres";
    h.taxonomy = "synrex.fractal7.v1";
    h.frame_points = n;
    h.frame_start_ns = stamp;
    h.frame_end_ns = stamp + n;
    const std::uint8_t codes[]{1, 2, 200, 6, 9, 17, 201};
    for (std::size_t i = 0; i < n; ++i) {
        const auto k = start + (repeats ? i % 3 : i);
        const auto x = double(k) - 32768.0;
        r.frame.points.push_back({k, stamp + i, {{x, (i % 2) ? -0.001 : 0.0, -0.0}}, 123, 1, 2});
        r.semantic_ids.push_back(k % 7);
        r.las_codes.push_back(codes[k % 7]);
    }
    return r;
}
} // namespace
int main(int argc, char **argv) {
    try {
        require(argc == 1 || argc == 2, "optional NEW_RECEIPT");
        Json cases = Json::array();
        std::vector<std::size_t> capacities;
        for (std::size_t c = 1; c <= 64; ++c)
            capacities.push_back(c);
        for (std::size_t c : {257, 1024, 4096, 16384, 65536})
            capacities.push_back(c);
        std::uint64_t compared = 0;
        for (auto capacity : capacities) {
            SemanticMap ref(1, capacity, true, MapPolicy::reference, true);
            SemanticMap fast(1, capacity, true, MapPolicy::indexed, true);
            MapOracle oracle(capacity, capacity == 257);
            auto verify = [&](const Classification &r) {
                const auto a = ref.update(r), b = fast.update(r);
                same(a, b);
                oracle.verify(r, b, fast.size());
                fast.validate_index();
                compared += r.frame.points.size();
            };
            for (std::size_t start = 0; start < capacity; start += 256)
                verify(frame(start, std::min<std::size_t>(256, capacity - start), 1000 + start));
            auto r = frame(capacity - 1, 1, 1000000);
            verify(r);
            verify(r); // Unchanged equal time still refreshes.
            r.semantic_ids[0] = (r.semantic_ids[0] + 1) % 7;
            const std::uint8_t codes[]{1, 2, 200, 6, 9, 17, 201};
            r.las_codes[0] = codes[r.semantic_ids[0]];
            verify(r); // Equal time class change.
            --r.frame.points[0].timestamp_ns;
            verify(r); // Late point changes nothing, including recency.
            for (std::size_t f = 0; f < 32; ++f)
                verify(frame(capacity + f * 32, 32, 2000000 + f * 64));
            // Boundaries, very large offsets, negative floors, coincident observations.
            r = frame(capacity + 1024, 16, 3000000);
            for (std::size_t i = 0; i < r.frame.points.size(); ++i) {
                double x = (i % 2) ? -100000000.0 : 700000.0;
                if (i % 3 == 0)
                    x = std::nextafter(x, -std::numeric_limits<double>::infinity());
                if (i % 3 == 1)
                    x = std::nextafter(x, std::numeric_limits<double>::infinity());
                r.frame.points[i].xyz = {x, -0.001, 0};
            }
            verify(r);
            require(ref.snapshot(4000000) == fast.snapshot(4000000) &&
                        fast.snapshot(4000000) == oracle.snapshot(4000000),
                    "full snapshot/age differs");
            const auto before = fast.snapshot(4000000);
            bool negative_age = false;
            try {
                (void)fast.snapshot(0);
            } catch (const std::exception &) {
                negative_age = true;
            }
            require(negative_age && before == fast.snapshot(4000000), "age rejection mutated map");
            r.frame.header.crs = "OTHER:frame";
            bool rejected = false;
            try {
                (void)fast.update(r);
            } catch (const std::exception &) {
                rejected = true;
            }
            require(rejected && before == fast.snapshot(4000000), "identity mismatch mutated map");
            const auto a = ref.counters(), b = fast.counters();
            auto ac = a.json(), bc = b.json();
            ac.erase("eviction_scanned_cells");
            bc.erase("eviction_scanned_cells");
            require(ac == bc && b.eviction_scanned_cells == 0, "operation or scan parity failure");
            cases.push_back({{"capacity", capacity},
                             {"status", "PASS"},
                             {"operations", b.json()},
                             {"reference_profile", ref.profile()},
                             {"indexed_profile", fast.profile()}});
            ref.reset();
            fast.reset();
            require(fast.snapshot(0) == ref.snapshot(0) && fast.size() == 0, "reset differs");
            r = frame(0, 3, 100);
            r.frame.header.sensor_id = "after-reset";
            same(ref.update(r), fast.update(r));
            fast.validate_index();
            require(ref.snapshot(500) == fast.snapshot(500), "reset/reinsert parity");
        }
        // Millions of accepted refreshes retain exactly capacity nodes, no stale entries.
        SemanticMap ref(1, 65536, false, MapPolicy::reference),
            fast(1, 65536, false, MapPolicy::indexed);
        for (std::size_t start = 0; start < 65536; start += 256) {
            auto r = frame(start, 256, 1000);
            same(ref.update(r), fast.update(r));
        }
        auto repeated = frame(50000, 256, 1000000, true);
        auto refresh = [&] {
            repeated.frame.header.frame_start_ns += 1000;
            repeated.frame.header.frame_end_ns += 1000;
            for (auto &point : repeated.frame.points)
                point.timestamp_ns += 1000;
            same(ref.update(repeated), fast.update(repeated));
        };
        for (int i = 0; i < 64; ++i)
            refresh();
        const auto rss_before = resident();
        const auto accepted_before = fast.counters().existing_updates;
        for (int i = 0; i < 8192; ++i) {
            refresh();
        }
        const auto accepted_refreshes = fast.counters().existing_updates - accepted_before;
        require(accepted_refreshes == 2097152, "memory phase did not accept every refresh");
        const auto rss_after = resident();
        fast.validate_index();
        require(fast.profile().at("index_entries") == 65536 && fast.size() == 65536,
                "stale entries accumulated");
        require(rss_after <= rss_before + 8 * 1024 * 1024,
                "repeated updates grew memory over 8MiB");
        require(ref.snapshot(20000000) == fast.snapshot(20000000),
                "repeated refresh snapshot differs");
        // Exhaustion of unsigned revisions must preserve the original numeric minimum order.
        SemanticMap wrap_ref(1, 3, false, MapPolicy::reference),
            wrap_fast(1, 3, false, MapPolicy::indexed);
        same(wrap_ref.update(frame(0, 3, 100)), wrap_fast.update(frame(0, 3, 100)));
        SemanticMapTestAccess::near_wrap(wrap_ref);
        SemanticMapTestAccess::near_wrap(wrap_fast);
        for (std::size_t i = 0; i < 8; ++i) {
            auto r = frame(50 + i, 1, 1000 + i);
            same(wrap_ref.update(r), wrap_fast.update(r));
        }
        require(wrap_fast.profile().at("revision_overflow_fallback") == true &&
                    wrap_ref.snapshot(2000) == wrap_fast.snapshot(2000),
                "revision wrap fallback differs");
        Json receipt{
            {"schema", "synrex.map_index_parity.v1"},
            {"status", "PASS"},
            {"capacity_cases", cases},
            {"observations_differential_compared", compared + 65536 + 8256ULL * 256},
            {"memory_refresh_observations", 2097152},
            {"accepted_memory_refreshes", accepted_refreshes},
            {"memory_phase_late_rejections", fast.counters().late_rejections},
            {"rss_before_refresh_bytes", rss_before},
            {"rss_after_refresh_bytes", rss_after},
            {"reference_overflow_parity", "PASS"},
            {"forced_hash_collision_oracle_capacity", 257},
            {"expiration_policy", "No expiration exists; snapshot age/rejection preserved"},
            {"scope",
             "Model-free CPU semantic-map differential tests; no neural throughput claim"}};
        if (argc == 2)
            write_new(argv[1], receipt.dump(2) + "\n");
        std::cout << "map_index_parity=PASS\n";
        return 0;
    } catch (const std::exception &e) {
        std::cerr << e.what() << '\n';
        return 1;
    }
}
