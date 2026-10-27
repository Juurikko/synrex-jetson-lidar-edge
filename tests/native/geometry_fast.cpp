// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Tuomas Pietilä
#include "synrex/runtime.hpp"
#include <chrono>
#include <cmath>
#include <cstring>
#include <iostream>
using namespace synrex;
namespace {
bool compare(const std::vector<Point> &points, const Contract &c, GeometryMode mode) {
    Plan reference, actual;
    bool reference_failed = false, actual_failed = false;
    try {
        reference = geometry(points, c);
    } catch (const std::exception &) {
        reference_failed = true;
    }
    try {
        actual = geometry(points, c, mode);
    } catch (const std::exception &) {
        actual_failed = true;
    }
    require(reference_failed == actual_failed, "geometry rejection parity");
    if (reference_failed)
        return false;
    require(reference.memberships == actual.memberships &&
                reference.blocks.size() == actual.blocks.size(),
            "membership/block count parity");
    for (std::size_t b = 0; b < reference.blocks.size(); ++b) {
        const auto &x = reference.blocks[b], &y = actual.blocks[b];
        require(std::memcmp(x.origin.data(), y.origin.data(), sizeof(double) * 3) == 0 &&
                    x.members.size() == y.members.size(),
                "exact origin/member count parity");
        for (std::size_t i = 0; i < x.members.size(); ++i)
            require(x.members[i].point == y.members[i].point &&
                        x.members[i].voxel == y.members[i].voxel,
                    "source/voxel/order parity");
        const auto a = reference_tensor(points, x, c), z = reference_tensor(points, y, c);
        require(a.size() == z.size() &&
                    std::memcmp(a.data(), z.data(), a.size() * sizeof(float)) == 0,
                "tensor feature byte parity");
    }
    return true;
}
} // namespace
int main(int argc, char **argv) {
    try {
        require(argc == 2 || argc == 3, "usage: geometry-fast-tests SPEC NEW_RECEIPT");
        const auto original = read_json(argv[1]);
        std::uint64_t state = 91, cases = 0, tested_points = 0, successful = 0, rejections = 0;
        auto random = [&]() {
            state = state * 6364136223846793005ULL + 1442695040888963407ULL;
            return state;
        };
        for (double overlap : {0.0, 0.25})
            for (double unit : {1.0, 0.1, 0.003})
                for (double base :
                     {-1e15, -6390000.123, -4096.0, -0.0, 0.0, 700000.001, 1e12, 1e15}) {
                    auto spec = original;
                    spec["geometry"]["voxel_size_xyz"] = {unit, unit, unit};
                    spec["geometry"]["block_size_xyz"] = {32 * unit, 32 * unit, 32 * unit};
                    spec["geometry"]["overlap_xyz"] = {overlap, overlap, overlap};
                    const Contract c(spec);
                    std::vector<Point> p{{{base, base, base}, 0}};
                    for (int i = 0; i < 384; ++i) {
                        Point point{};
                        for (int a = 0; a < 3; ++a) {
                            const auto index = random() % 5;
                            const auto boundary = base + index * 32 * unit;
                            const auto which = random() % 6;
                            point.xyz[a] = which == 0   ? boundary
                                           : which == 1 ? std::nextafter(boundary, -INFINITY)
                                           : which == 2 ? std::nextafter(boundary, INFINITY)
                                                        : base + (random() % 160001) * unit / 1000;
                        }
                        point.intensity = static_cast<std::uint16_t>(random());
                        p.push_back(point);
                        if (i % 17 == 0)
                            p.push_back(point); // coincident points retain distinct indices
                    }
                    for (auto mode :
                         {GeometryMode::direct, GeometryMode::flat, GeometryMode::radix})
                        if (compare(p, c, mode))
                            ++successful;
                        else
                            ++rejections;
                    ++cases;
                    tested_points += p.size();
                }
        // Deterministic LAS integer/scale reconstruction, sparse empty regions and arbitrary grids.
        for (int repeat = 0; repeat < 150; ++repeat) {
            auto spec = original;
            const std::array<int, 3> dims{3 + repeat % 9, 5, 7};
            spec["geometry"]["dimensions_xyz"] = dims;
            spec["geometry"]["voxel_size_xyz"] = {0.25, 0.5, 0.125};
            spec["geometry"]["block_size_xyz"] = {dims[0] * 0.25, 2.5, 0.875};
            spec["input"]["shape"] = {1, 4, 7, 5, dims[0]};
            spec["output"]["shape"] = {1, 7, 7, 5, dims[0]};
            spec["geometry"]["overlap_xyz"] = {0, 0, 0};
            const Contract c(spec);
            std::vector<Point> p;
            for (int i = 0; i < 128; ++i) {
                Point point{};
                for (int a = 0; a < 3; ++a)
                    point.xyz[a] = static_cast<std::int32_t>(random() % 5000) * 0.001 +
                                   (a == 0 ? -700000.123 : 6390000.002);
                point.intensity = static_cast<std::uint16_t>(random());
                p.push_back(point);
            }
            for (auto mode : {GeometryMode::direct, GeometryMode::flat, GeometryMode::radix})
                if (compare(p, c, mode))
                    ++successful;
                else
                    ++rejections;
            ++cases;
            tested_points += p.size();
        }
        auto spec = original;
        spec["geometry"]["overlap_xyz"] = {0, 0, 0};
        const Contract c(spec);
        const std::vector<Point> sparse{{{-100, -100, -100}, 0},
                                        {{3100000.125, 3100000.125, 3100000.125}, 65535}};
        for (auto mode : {GeometryMode::direct, GeometryMode::flat, GeometryMode::radix})
            if (compare(sparse, c, mode))
                ++successful;
            else
                ++rejections;
        for (const auto &invalid :
             std::vector<std::vector<Point>>{{},
                                             {{{NAN, 0, 0}, 1}},
                                             {{{0, INFINITY, 0}, 1}},
                                             {{{0, 0, 0}, 1}, {{1e20, 0, 0}, 1}}})
            for (auto mode : {GeometryMode::direct, GeometryMode::flat, GeometryMode::radix}) {
                require(!compare(invalid, c, mode), "unsafe geometry must reject");
                ++rejections;
            }
        std::vector<Point> corpus;
        for (int i = 0; i < 16384; ++i)
            corpus.push_back(
                {{double(random() % 50000) / 1000 - 700000,
                  double(random() % 50000) / 1000 + 6390000, double(random() % 12000) / 1000 - 500},
                 static_cast<std::uint16_t>(random())});
        // Exercise radix-sized blocks, stable coincident reductions, and all-zero voxels.
        for (auto mode : {GeometryMode::direct, GeometryMode::flat, GeometryMode::radix}) {
            require(compare(corpus, c, mode), "dense radix tensor parity");
            auto coincident = corpus;
            for (auto &p : coincident)
                p.xyz = corpus.front().xyz;
            require(compare(coincident, c, mode), "coincident stable reduction parity");
        }
        successful += 6;
        cases += 2;
        tested_points += 2 * corpus.size();
        Json timings = Json::array();
        for (auto mode : {GeometryMode::reference, GeometryMode::direct, GeometryMode::flat,
                          GeometryMode::radix}) {
            const auto start = std::chrono::steady_clock::now();
            GeometryProfile totals;
            std::uint64_t blocks = 0;
            for (int trial = 0; trial < 40; ++trial) {
                GeometryProfile p;
                const auto plan = geometry(corpus, c, mode, &p);
                blocks += plan.blocks.size();
                totals.minimum_seconds += p.minimum_seconds;
                totals.assignment_seconds += p.assignment_seconds;
                totals.grouping_seconds += p.grouping_seconds;
                totals.record_sort_seconds += p.record_sort_seconds;
                totals.member_sort_seconds += p.member_sort_seconds;
                totals.tree_lookups += p.tree_lookups;
                totals.member_capacity_growths += p.member_capacity_growths;
                totals.boundary_fallback_points += p.boundary_fallback_points;
            }
            timings.push_back(
                {{"mode", geometry_name(mode)},
                 {"wall_seconds",
                  std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count()},
                 {"points", corpus.size() * 40},
                 {"blocks", blocks},
                 {"minimum_seconds", totals.minimum_seconds},
                 {"assignment_seconds", totals.assignment_seconds},
                 {"grouping_seconds", totals.grouping_seconds},
                 {"record_sort_seconds", totals.record_sort_seconds},
                 {"member_sort_seconds", totals.member_sort_seconds},
                 {"tree_lookups", totals.tree_lookups},
                 {"member_capacity_growths", totals.member_capacity_growths},
                 {"boundary_fallback_points", totals.boundary_fallback_points}});
        }
        const auto receipt = Json{{"schema", "synrex.geometry_parity.v1"},
                                  {"status", "PASS"},
                                  {"cases", cases + 5},
                                  {"successful_mode_comparisons", successful},
                                  {"matching_rejections", rejections},
                                  {"points", tested_points + 2},
                                  {"scope", "CPU-only membership/index/order/tensor byte "
                                            "comparisons; no neural throughput claim"},
                                  {"cpu_microbenchmark", timings}}
                                 .dump(2) +
                             "\n";
        if (argc == 3)
            write_new(argv[2], receipt);
        std::cout << "geometry_parity=PASS\n";
        return 0;
    } catch (const std::exception &e) {
        std::cerr << e.what() << '\n';
        return 1;
    }
}
