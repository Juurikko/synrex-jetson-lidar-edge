// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Tuomas Pietilä
// Model-free export boundary test; no neural inference or performance assertion.
#include "synrex/export.hpp"
#include <iostream>
using namespace synrex;
using namespace synrex::streaming;
template <class F> void rejects(F f) {
    bool failed = false;
    try {
        f();
    } catch (const std::exception &) {
        failed = true;
    }
    require(failed, "expected export rejection");
}
int main(int argc, char **argv) {
    try {
        require(argc == 5, "usage: export-tests SOURCE MODEL TAXONOMY NEW_WORK");
        const fs::path work(argv[4]);
        require(fs::create_directory(work), "new test directory");
        const Package package(argv[2]);
        const auto points = decode(argv[1], package.contract);
        Classification c;
        c.frame.header.sensor_id = "export-test";
        c.frame.header.stream_id = "export-test";
        c.frame.header.coordinate_frame = "explicit_local";
        c.frame.header.crs = "LOCAL:explicit_local";
        c.frame.header.taxonomy = "synrex.fractal7.v1";
        c.frame.header.frame_start_ns = 100;
        c.frame.header.frame_end_ns = 200;
        c.frame.header.frame_points = points.size();
        c.frame.header.chunk_count = 1;
        for (std::size_t i = 0; i < points.size(); ++i) {
            c.frame.points.push_back({i, 100, points[i].xyz, points[i].intensity, {}, {}});
            c.semantic_ids.push_back(i % 7);
            c.las_codes.push_back(package.contract.las_classes[i % 7]);
        }
        {
            AsyncLazExport partial(argv[1], argv[2], argv[3], work / "partial");
            auto one = c;
            one.frame.points.resize(1);
            one.frame.header.frame_points = 1;
            one.semantic_ids.resize(1);
            one.las_codes.resize(1);
            partial.observe(one);
            rejects([&] { partial.finish(); });
        }
        require(!fs::exists(work / "partial"), "partial export must not publish");
        {
            AsyncLazExport invalid(argv[1], argv[2], argv[3], work / "invalid");
            auto moved = c;
            moved.frame.points[0].xyz[0] += 1;
            rejects([&] { invalid.observe(moved); });
        }
        AsyncLazExport good(argv[1], argv[2], argv[3], work / "complete");
        good.observe(c);
        good.observe(c);
        const auto receipt = good.finish();
        require(receipt.at("coverage") == 1, "export full coverage");
        rejects([&] { AsyncLazExport existing(argv[1], argv[2], argv[3], work / "complete"); });
        std::cout << Json{{"status", "PASS"},
                          {"scope", "MODEL_FREE_TEST_ONLY; preservation, atomic publication, "
                                    "partial/transform/overwrite rejection"},
                          {"product", receipt}}
                         .dump(2)
                  << '\n';
        return 0;
    } catch (const std::exception &e) {
        std::cerr << e.what() << '\n';
        return 1;
    }
}
