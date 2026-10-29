// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Tuomas Pietilä
// Exact map A/B oracle over previously generated native neural point events.
// Extra aging/class/coordinate fixtures are map-only tests, never neural inference claims.
#include "synrex/stream.hpp"
#include <fstream>
#include <iostream>
using namespace synrex;
using namespace synrex::streaming;
namespace {
Classification load(const fs::path &path) {
    require(fs::file_size(path) <= 64 * 1024 * 1024, "map replay input budget");
    std::ifstream in(path); Json j; in >> j;
    require(j.at("schema") == "synrex.classified_frame.v1" && j.at("mode") == "all_points_freshly_classified", "native classified events required");
    Classification r;
    auto &h = r.frame.header;
    h.sensor_id = j.at("sensor_id"); h.stream_id = j.at("stream_id");
    h.coordinate_frame = j.at("coordinate_frame"); h.crs = j.at("crs"); h.taxonomy = j.at("taxonomy");
    h.frame_id = j.at("frame_id"); h.frame_points = j.at("frame_points");
    h.frame_start_ns = j.at("frame_start_ns"); h.frame_end_ns = j.at("frame_end_ns");
    r.frame.first_sequence = j.at("first_sequence"); r.frame.last_sequence = j.at("last_sequence");
    require(j.at("points").size() == h.frame_points && h.frame_points <= 65536, "classified frame budget");
    for (const auto &point : j.at("points")) {
        SourcePoint p;
        p.id = point.at("id"); p.timestamp_ns = point.at("timestamp_ns"); p.xyz = point.at("xyz").get<std::array<double, 3>>();
        if (!point.at("intensity").is_null()) p.intensity = point.at("intensity");
        if (!point.at("return_number").is_null()) p.return_number = point.at("return_number");
        if (!point.at("return_count").is_null()) p.return_count = point.at("return_count");
        r.frame.points.push_back(p);
        r.semantic_ids.push_back(point.at("semantic_id")); r.las_codes.push_back(point.at("model_las_code"));
    }
    r.timings = Json::object();
    require(event(r).at("points") == j.at("points"), "point metadata/class reconstruction mismatch");
    return r;
}
}
int main(int argc, char **argv) {
    try {
        require(argc >= 3 && argc <= 35, "usage: synrex-map-replay-tests NEW_OUTPUT CAPACITY [EVENT.json ...]");
        const auto capacity = std::stoull(argv[2]);
        SemanticMap map(1.0, capacity);
        Json output{{"schema", "synrex.map_reference_replay.v1"}, {"status", "PASS"},
                    {"scope", "Map-only exact A/B test; supplied classes originate from real native neural events; stress updates are model-free fixtures"},
                    {"frames", Json::array()}};
        std::uint64_t as_of = 0, points = 0;
        Classification last;
        auto apply = [&](const Classification &r, const std::string &kind) {
            const auto deltas = map.update(r);
            Json changes = Json::array();
            for (const auto &d : deltas) changes.push_back({{"key", d.key}, {"semantic_id", d.semantic_id},
                {"timestamp_ns", d.timestamp_ns}, {"evicted", d.evicted ? Json(*d.evicted) : Json(nullptr)}});
            as_of = std::max(as_of, r.frame.header.frame_end_ns);
            auto snapshot = map.snapshot(as_of);
            require(map.size() <= capacity, "map capacity exceeded");
            for (const auto &cell : snapshot.at("cells"))
                require(cell.at("confidence").is_null() && cell.at("confidence_age_ns").is_null(), "uncalibrated confidence must stay unknown");
            output["frames"].push_back({{"fixture", kind}, {"source_frame", r.frame.header.frame_id},
                {"point_events", event(r).at("points")}, {"deltas", changes}, {"snapshot", snapshot}});
        };
        for (int i = 3; i < argc; ++i) { last = load(argv[i]); points += last.frame.points.size(); require(points <= 1000000, "map test point budget"); apply(last, "native_neural_event"); }
        require(points > 0, "neural events required");
        apply(last, "repeated_observation");
        last.frame.header.frame_start_ns += 1000000; last.frame.header.frame_end_ns += 1000000;
        for (auto &p : last.frame.points) p.timestamp_ns += 1000000;
        apply(last, "observation_age_refresh");
        const std::vector<std::uint8_t> mapping = last.frame.header.taxonomy == "synrex.fractal7.v1" ?
            std::vector<std::uint8_t>{1,2,200,6,9,17,201} : std::vector<std::uint8_t>{2,5,6};
        for (std::size_t i = 0; i < last.semantic_ids.size(); ++i) {
            last.semantic_ids[i] = (last.semantic_ids[i] + 1) % mapping.size();
            last.las_codes[i] = mapping[last.semantic_ids[i]];
        }
        apply(last, "explicit_map_class_update_fixture");
        last.frame.header.frame_start_ns -= 2000000; last.frame.header.frame_end_ns -= 2000000;
        for (auto &p : last.frame.points) p.timestamp_ns -= 2000000;
        apply(last, "late_observation");
        const auto before = map.snapshot(as_of);
        last.frame.header.crs = "INCOMPATIBLE:fixture";
        bool refused = false;
        try { (void)map.update(last); } catch (const std::exception &) { refused = true; }
        require(refused && before == map.snapshot(as_of), "coordinate mismatch must reject without map mutation");
        output["coordinate_mismatch_rejection"] = "PASS";
        output["capacity"] = capacity; output["native_neural_points"] = points;
        write_new(argv[1], output.dump() + "\n");
        std::cout << "map_reference_replay=PASS\n"; return 0;
    } catch (const std::exception &e) { std::cerr << "map_reference_replay=FAIL: " << e.what() << '\n'; return 1; }
}
