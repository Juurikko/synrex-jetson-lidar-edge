// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Tuomas Pietilä
// CPU-only differential profile. Fixture classes are invented, never neural results.
#include "synrex/map_oracle.hpp"
#include "synrex/transport.hpp"
#include <iostream>
using namespace synrex;
using namespace synrex::streaming;
namespace w = synrex::streaming::wire;
int main(int argc, char **argv) {
    try {
        require(argc == 4, "usage: transport-profile MODEL SOURCE NEW_RECEIPT");
        const Package package(argv[1]);
        const auto corpus = decode(argv[2], package.contract);
        require(corpus.size() >= 16384, "profile corpus too small");
        std::array<w::Sequence, 2> sequences;
        std::array<double, 2> decode_time{}, oracle_time{};
        MapOracle reference(65536), optimized(65536);
        SemanticMap map(1, 65536);
        std::set<std::array<std::int64_t, 3>> active;
        auto elapsed = [](std::uint64_t start) { return (steady_ns() - start) / 1e9; };
        std::uint64_t observations = 0;
        for (std::uint64_t frame = 0; frame < 128; ++frame) {
            w::Output out;
            auto &in = out.input;
            auto &h = in.frame.header;
            h.sensor_id = "cpu-profile";
            h.stream_id = "fixture-only";
            h.coordinate_frame = "world";
            h.crs = "LOCAL:world";
            h.taxonomy = "synrex.fractal7.v1";
            h.sequence = h.frame_id = frame;
            in.frame.first_sequence = in.frame.last_sequence = frame;
            h.frame_start_ns = h.frame_end_ns = 1000 + frame;
            h.frame_points = 16384;
            h.chunk_count = 1;
            in.coordinates = w::Coordinates::world;
            for (std::size_t i = 0; i < 16384; ++i) {
                const auto index = (frame * 16384 + i) % corpus.size();
                auto xyz = corpus[index].xyz;
                xyz[0] += frame * 64;
                in.frame.points.push_back(
                    {index, h.frame_start_ns, xyz, corpus[index].intensity, {}, {}});
                out.semantic_ids.push_back(i % 7);
                out.las_codes.push_back(package.contract.las_classes[i % 7]);
            }
            out.model_manifest_sha256 = sha256(package.root / "manifest.json");
            Classification c{in.frame, out.semantic_ids, out.las_codes, {}};
            out.deltas = map.update(c);
            const auto bytes = w::encode(out);
            for (unsigned mode = 0; mode < 2; ++mode) {
                const auto begin = steady_ns();
                auto decoded =
                    mode ? sequences[mode].decode_classified(bytes) : w::decode_output(bytes);
                if (!mode)
                    sequences[mode].accept(decoded.input);
                decode_time[mode] += elapsed(begin);
                require(w::encode(decoded) == bytes,
                        "profile exact output/source/delta/class byte parity");
                Classification actual{
                    decoded.input.frame, decoded.semantic_ids, decoded.las_codes, {}};
                const auto oracle_begin = steady_ns();
                if (mode)
                    optimized.verify(actual, decoded.deltas);
                else {
                    for (const auto &d : decoded.deltas) {
                        if (d.evicted)
                            require(active.erase(*d.evicted) == 1, "profile eviction");
                        active.insert(d.key);
                    }
                    reference.verify(actual, decoded.deltas, active.size());
                }
                oracle_time[mode] += elapsed(oracle_begin);
            }
            observations += 16384;
        }
        require(reference.snapshot(1127) == optimized.snapshot(1127) &&
                    reference.snapshot(1127) == map.snapshot(1127),
                "profile full snapshot parity");
        const Json receipt{
            {"schema", "synrex.streaming_r1.consumer_cpu_profile.v1"},
            {"status", "PASS"},
            {"scope", "MODEL_FREE_TEST_ONLY; actual licensed geometry/intensity, invented fixture "
                      "classes. CPU validation service cost only; no neural throughput claim."},
            {"source_basename", fs::path(argv[2]).filename().string()},
            {"source_sha256", sha256(argv[2])},
            {"fixture_points", observations},
            {"capacity", 65536},
            {"reference_decode_and_sequence_seconds", decode_time[0]},
            {"decoded_once_sequence_seconds", decode_time[1]},
            {"reference_oracle_plus_occupancy_tree_seconds", oracle_time[0]},
            {"oracle_without_redundant_tree_seconds", oracle_time[1]},
            {"exact_encoded_point_delta_class_source_bytes", "PASS"},
            {"full_snapshot_parity", "PASS"},
            {"bounded_map_size", map.size()}};
        write_new(argv[3], receipt.dump(2) + "\n");
        std::cout << receipt.dump(2) << '\n';
        return 0;
    } catch (const std::exception &e) {
        std::cerr << e.what() << '\n';
        return 1;
    }
}
