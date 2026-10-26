// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Tuomas Pietilä
#include "synrex/runtime.hpp"
#include <cmath>
#include <iostream>
#include <stdexcept>

int main(int argc, char **argv) {
    using namespace synrex;
    try {
        require(argc == 2, "spec argument required");
        auto spec = read_json(argv[1]);
        const Contract c(spec);
        const std::vector<Point> points{
            {{0, 0, 0}, 0}, {{7.5, 0, 0}, 65535}, {{7.25, 0, 0}, 0}, {{8, 0, 0}, 32768}};
        const auto plan = geometry(points, c);
        require(plan.blocks.size() == 2 && plan.memberships == 6, "overlap coverage");
        require(plan.blocks[0].members[1].point == 1 && plan.blocks[0].members[2].point == 2,
                "same voxel retains original point identities");
        const auto tensor = reference_tensor(points, plan.blocks[0], c);
        require(tensor[7] == 1 && std::abs(tensor[c.voxels + 7] - 0.5f) < 1e-6,
                "voxel mean feature");
        require(tensor[8] == 0 && tensor[c.voxels + 8] == 0, "empty voxels are zero");
        auto all_features = spec;
        all_features["features"] =
            Json::array({{{"kind", "occupancy"}, {"offset", 0.25}, {"scale", 2}},
                         {{"kind", "count"}, {"offset", 1}, {"scale", 2}},
                         {{"kind", "mean_intensity"}, {"offset", 0}, {"scale", 1.0 / 65535}},
                         {{"kind", "local_x"}, {"offset", 1}, {"scale", 2}},
                         {{"kind", "local_y"}, {"offset", 0}, {"scale", 1}},
                         {{"kind", "local_z"}, {"offset", 0}, {"scale", 1}}});
        all_features["input"]["shape"][1] = 6;
        const Contract full(all_features);
        const auto values = reference_tensor(points, plan.blocks[0], full);
        require(values[7] == 1.5f && values[c.voxels + 7] == 2 &&
                    std::abs(values[2 * c.voxels + 7] - 0.5f) < 1e-6 &&
                    values[3 * c.voxels + 7] == 12.75f && values[4 * c.voxels + 7] == 0 &&
                    values[5 * c.voxels + 7] == 0 && values[8] == 0,
                "all feature reductions and normalization");
        auto bad = spec;
        bad["input"]["shape"][4] = 112;
        bool refused = false;
        try {
            const Contract ignored(bad);
        } catch (const std::exception &) {
            refused = true;
        }
        require(refused, "shape mismatch must fail closed");
        bad = spec;
        bad["geometry"]["overlap_xyz"][0] = 1;
        refused = false;
        try {
            const Contract ignored(bad);
        } catch (const std::exception &) {
            refused = true;
        }
        require(refused, "unbounded overlap must fail closed");
        // A completely different grid/channel contract proves shape configurability.
        auto other = spec;
        other["geometry"]["dimensions_xyz"] = {3, 2, 5};
        other["geometry"]["block_size_xyz"] = {3, 2, 5};
        other["input"]["shape"] = {1, 2, 5, 2, 3};
        other["las_classes"] = {0, 2, 5, 6};
        other["output"]["shape"] = {1, 4, 5, 2, 3};
        require(Contract(other).voxels == 30, "configurable non-reference model dimensions");
        std::cout << "dense_voxel_contract=PASS\n";
        return 0;
    } catch (const std::exception &e) {
        std::cerr << e.what() << '\n';
        return 1;
    }
}
