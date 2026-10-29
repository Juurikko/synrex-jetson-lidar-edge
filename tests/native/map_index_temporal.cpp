// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Tuomas Pietilä
// Explicit policy expectations independent of the shared update code.
#include "synrex/stream.hpp"
#include <iostream>
using namespace synrex;
using namespace synrex::streaming;
int main(int argc, char **argv) {
    try {
        require(argc == 1 || argc == 2, "optional NEW_RECEIPT");
        Json cases = Json::array();
        for (double resolution : {0.001, 0.125, 1.0, 32.5, 1000.0}) {
            SemanticMap reference(resolution, 3, false, MapPolicy::reference);
            SemanticMap indexed(resolution, 3, false, MapPolicy::indexed);
            Classification r;
            auto &h = r.frame.header;
            h.sensor_id = "cpu-fixture";
            h.stream_id = "temporal-map-policy";
            h.coordinate_frame = "fixed-world";
            h.crs = "LOCAL:metres";
            h.taxonomy = "synrex.fractal7.v1";
            h.frame_points = 1;
            const std::uint8_t codes[]{1, 2, 200, 6, 9, 17, 201};
            auto observe = [&](int key, std::uint64_t time, int id, int victim = -1,
                               int delta_count = 1) {
                ++h.frame_id;
                h.frame_start_ns = time;
                h.frame_end_ns = time + 1;
                r.frame.points = {SourcePoint{static_cast<std::uint64_t>(key + 10),
                                              time,
                                              {{(key + 0.25) * resolution, 0, 0}},
                                              123,
                                              1,
                                              2}};
                r.semantic_ids = {static_cast<std::uint8_t>(id)};
                r.las_codes = {codes[id]};
                const auto a = reference.update(r), b = indexed.update(r);
                require(a.size() == b.size() && b.size() == static_cast<std::size_t>(delta_count),
                        "unexpected temporal delta count");
                if (delta_count) {
                    require(b[0].key[0] == key && b[0].semantic_id == id &&
                                b[0].timestamp_ns == time,
                            "temporal delta differs");
                    require(a[0].key == b[0].key && a[0].evicted == b[0].evicted &&
                                a[0].semantic_id == b[0].semantic_id &&
                                a[0].timestamp_ns == b[0].timestamp_ns,
                            "reference differs");
                    require(victim < 0 ? !b[0].evicted
                                       : (b[0].evicted && (*b[0].evicted)[0] == victim),
                            "wrong explicitly expected eviction victim");
                }
                indexed.validate_index();
                require(reference.snapshot(UINT64_MAX) == indexed.snapshot(UINT64_MAX),
                        "temporal snapshot differs");
            };
            observe(0, 1000, 0);
            observe(1, 1, 1);
            observe(2, 1000, 2);
            observe(1, 1, 1, -1, 0);   // Equal timestamp/id refreshes recency without delta.
            observe(0, 999, 6, -1, 0); // Late existing point does not refresh recency or class.
            observe(3, 0, 3, 0);     // New point older than every resident time is still accepted.
            observe(0, 0, 4, 2);     // Recency, not timestamp-minimum, selects cell 2.
            observe(3, 0, 3, -1, 0); // Exact duplicate refreshes without emitting a delta.
            observe(4, 0, 5, 1);
            observe(0, 0, 6); // Equal-time class change is accepted, refreshing cell 0.
            observe(5, 0, 0, 3);
            const auto snapshot = indexed.snapshot(UINT64_MAX);
            require(indexed.size() == 3 && indexed.counters().late_rejections == 1,
                    "aging expired cells or late count differs");
            for (const auto &cell : snapshot.at("cells"))
                require(cell.at("timestamp_ns") == 0 &&
                            cell.at("classification_age_ns") == UINT64_MAX,
                        "age arithmetic differs");
            cases.push_back({{"resolution_metres", resolution},
                             {"status", "PASS"},
                             {"operations", indexed.counters().json()},
                             {"explicit_eviction_order", {0, 2, 1, 3}}});
        }
        const Json receipt{
            {"schema", "synrex.map_temporal_policy.v1"},
            {"status", "PASS"},
            {"cases", cases},
            {"scope", "CPU dummy classes; explicit recency/equal-time/old-new-cell expectations, "
                      "generic resolution and non-expiring age behavior; not neural throughput"}};
        if (argc == 2)
            write_new(argv[1], receipt.dump(2) + "\n");
        std::cout << "map_temporal_policy=PASS\n";
        return 0;
    } catch (const std::exception &e) {
        std::cerr << e.what() << '\n';
        return 1;
    }
}
