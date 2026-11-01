// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Tuomas Pietilä
// Synthetic aggregation controls only: no inference, CUDA or GPU target query.
#include "synrex/backend_profile.hpp"
#include <cmath>
#include <iostream>
#include <limits>

namespace {
using namespace synrex;
using namespace synrex::generic;
unsigned checks{}, rejected{};
void check(bool good, const char *message) {
    ++checks;
    require(good, message);
}
template <class F> void rejects(F f) {
    try {
        f();
    } catch (const std::exception &) {
        ++rejected;
        return;
    }
    throw std::runtime_error("expected profile rejection");
}
Json identity(const Json &contract, const char *mode = "reference") {
    return {{"schema", "synrex.persistent_session.v1"},
            {"backend", "CUDA/TensorRT"},
            {"contract", contract},
            {"geometry_mode", mode},
            {"manifest_sha256", std::string(64, 'a')}};
}
Json timings(double multiplier = 1) {
    return {{"cpu_geometry_seconds", .01 * multiplier},
            {"h2d_and_zero_seconds", .02 * multiplier},
            {"cuda_preparation_seconds", .03 * multiplier},
            {"tensorrt_seconds", .04 * multiplier},
            {"cuda_scatter_seconds", .05 * multiplier},
            {"d2h_seconds", .06 * multiplier},
            {"window_seconds", .1 * multiplier},
            {"completed_inference_blocks", 2},
            {"freshly_classified_points", 3},
            {"occupied_voxels", 5}};
}
} // namespace
int main(int argc, char **argv) {
    try {
        require(argc == 2, "profile fixture contract path");
        const auto spec = read_json(argv[1]);
        const auto id = identity(spec);
        BackendProfile one(id), two(id), trial(id);
        check(one.receipt().at("availability") == "UNAVAILABLE_NO_SAMPLES", "no samples explicit");
        check(one.receipt().at("phase_seconds").at("window_seconds").at("sum_seconds").is_null(),
              "missing values null");
        one.observe(timings(), 3);
        two.observe(timings(3), 3);
        two.observe(timings(2), 3);
        trial.merge(one);
        trial.merge(two);
        const auto row = trial.receipt();
        check(row.at("availability") == "BACKEND_REPORTED",
              "synthetic reports recognized, not GPU proof");
        check(row.at("observed_returned_windows") == 3 &&
                  row.at("observed_returned_frame_points") == 9,
              "unweighted window and point totals");
        check(row.at("phase_seconds").at("window_seconds").at("samples") == 3 &&
                  std::abs(
                      row.at("phase_seconds").at("window_seconds").at("sum_seconds").get<double>() -
                      .6) < 1e-14 &&
                  row.at("phase_seconds").at("window_seconds").at("minimum_seconds") == .1 &&
                  row.at("phase_seconds").at("window_seconds").at("maximum_seconds") == .1 * 3,
              "trial sums samples rather than source means; preserves extrema");
        check(row.at("counters").at("completed_inference_blocks").at("sum") == 6 &&
                  row.at("counters").at("occupied_voxels").at("sum") == 15,
              "actual backend counters summed");
        check(row.at("workload").at("feature_count") == 2 &&
                  row.at("workload").at("class_count") == 3 &&
                  row.at("workload").at("voxels_per_block") == 192 &&
                  row.at("workload").at("effective_geometry") == "reference",
              "actual noncubic contract workload identity");
        check(row.at("storage_bytes").get<std::size_t>() <= 4096 &&
                  row.dump().size() <= BackendProfile::max_receipt_bytes,
              "bounded storage and serialized receipt");
        auto zeros = timings();
        zeros["window_seconds"] = 0;
        BackendProfile partial(id);
        partial.observe(zeros, 3);
        auto missing = timings();
        missing.erase("tensorrt_seconds");
        missing.erase("occupied_voxels");
        partial.observe(missing, 3);
        check(
            partial.receipt().at("availability") == "PARTIAL_MISSING_FIELDS" &&
                partial.receipt()
                        .at("phase_seconds")
                        .at("tensorrt_seconds")
                        .at("missing_windows") == 1 &&
                partial.receipt().at("phase_seconds").at("window_seconds").at("minimum_seconds") ==
                    0,
            "missing measurements remain missing while actual zeros remain zero");
        BackendProfile absent(id);
        absent.observe(Json::object(), 3);
        check(absent.receipt()
                      .at("phase_seconds")
                      .at("tensorrt_seconds")
                      .at("sum_seconds")
                      .is_null() &&
                  absent.receipt().at("counters").at("occupied_voxels").at("sum").is_null(),
              "all missing not fabricated zero");
        for (const auto &fault : {"negative", "string", "nan", "infinity", "floating_count",
                                  "point_mismatch", "blocks", "occupancy", "late_field"}) {
            auto bad = timings();
            const std::string name = fault;
            if (name == "negative")
                bad["cpu_geometry_seconds"] = -1;
            if (name == "string")
                bad["window_seconds"] = "0.1";
            if (name == "nan")
                bad["tensorrt_seconds"] = std::numeric_limits<double>::quiet_NaN();
            if (name == "infinity")
                bad["d2h_seconds"] = std::numeric_limits<double>::infinity();
            if (name == "floating_count")
                bad["occupied_voxels"] = 5.0;
            if (name == "point_mismatch")
                bad["freshly_classified_points"] = 4;
            if (name == "blocks")
                bad["completed_inference_blocks"] = UINT64_MAX;
            if (name == "occupancy")
                bad["occupied_voxels"] = 385;
            if (name == "late_field")
                bad["window_seconds"] = -1;
            const auto before = trial.receipt();
            rejects([&] { trial.observe(bad, 3); });
            check(trial.receipt() == before,
                  "malformed observation leaves prior measured prefix unchanged");
        }
        auto nonoverlap = spec;
        nonoverlap["geometry"]["overlap_xyz"] = Json::array({0.0, 0.0, 0.0});
        BackendProfile nonoverlap_profile(identity(nonoverlap));
        // Independent reviewer reproduced these impossible actual-geometry tuples.
        for (const auto tuple : {std::array<std::uint64_t, 3>{1, 1, 2}, {3, 2, 1}}) {
            auto bad = timings();
            bad["freshly_classified_points"] = tuple[0];
            bad["completed_inference_blocks"] = tuple[1];
            bad["occupied_voxels"] = tuple[2];
            const auto saved = nonoverlap_profile.receipt();
            rejects([&] { nonoverlap_profile.observe(bad, tuple[0]); });
            check(nonoverlap_profile.receipt() == saved,
                  "impossible occupied/block/point counter tuple rejected atomically");
        }
        auto possible = timings();
        possible["freshly_classified_points"] = 1;
        possible["completed_inference_blocks"] = 2;
        possible["occupied_voxels"] = 2;
        nonoverlap_profile.observe(possible, 1);
        check(nonoverlap_profile.receipt().at("observed_returned_windows") == 1,
              "nominal zero overlap is not evidence of exact floating-coordinate partition");
        auto precision_spec = nonoverlap;
        precision_spec["geometry"]["dimensions_xyz"] = Json::array({1, 1, 1});
        precision_spec["geometry"]["voxel_size_xyz"] = Json::array({1e-12, 1e-12, 1e-12});
        precision_spec["geometry"]["block_size_xyz"] = Json::array({1e-12, 1e-12, 1e-12});
        precision_spec["input"]["shape"] = Json::array({1, 2, 1, 1, 1});
        precision_spec["output"]["shape"] = Json::array({1, 3, 1, 1, 1});
        const Contract precision_contract(precision_spec);
        const auto actual = geometry(std::vector<Point>{{{1e6, 1e6, 1e6}, 0}}, precision_contract);
        check(actual.blocks.size() == 8 && actual.memberships == 8,
              "existing sub-coordinate-precision geometry behavior retained visibly");
        BackendProfile precision_profile(identity(precision_spec));
        auto precision_timings = timings();
        precision_timings["freshly_classified_points"] = 1;
        precision_timings["completed_inference_blocks"] = actual.blocks.size();
        precision_timings["occupied_voxels"] = actual.memberships;
        precision_profile.observe(precision_timings, 1);
        check(precision_profile.receipt().at("counters").at("occupied_voxels").at("sum") == 8,
              "profiler accepts actual geometry counts without changing geometry");
        auto valid = timings();
        valid["occupied_voxels"] = 2;
        nonoverlap_profile.observe(valid, 3);
        check(nonoverlap_profile.receipt().at("observed_returned_windows") == 2,
              "ordinary valid nonoverlap reports aggregate without inferring partition bounds");
        auto tiny = spec;
        tiny["limits"]["max_memberships"] = 3;
        BackendProfile bounded(identity(tiny));
        auto bad = timings();
        bad["occupied_voxels"] = 4;
        rejects([&] { bounded.observe(bad, 3); });
        bad = timings();
        bad["occupied_voxels"] = 1;
        rejects([&] { bounded.observe(bad, 3); });
        bad = timings();
        bad.erase("occupied_voxels");
        bad["completed_inference_blocks"] = 4;
        rejects([&] { bounded.observe(bad, 3); });
        bad = timings();
        bad["freshly_classified_points"] = 4;
        rejects([&] { bounded.observe(bad, 4); });
        check(bounded.receipt().at("observed_returned_windows") == 0,
              "overlap membership and nonempty-block limits preserve no-sample prefix");
        const auto before = trial.receipt();
        BackendProfile other(identity(spec, "direct"));
        rejects([&] { trial.merge(other); });
        check(trial.receipt() == before, "workload merge drift fails atomically");
        auto fixture = id;
        fixture["fixture"] = "opaque CPU test";
        BackendProfile cpu(fixture);
        cpu.observe(timings(), 3);
        check(cpu.receipt().at("availability") == "UNSUPPORTED_BACKEND" &&
                  cpu.receipt().at("backend").is_null() &&
                  cpu.receipt().at("phase_seconds").at("tensorrt_seconds").is_null() &&
                  cpu.receipt().at("workload").is_null(),
              "CPU fixture stays unsupported even with synthetic CUDA-looking values");
        ProfileSeconds sums;
        sums.add(std::numeric_limits<double>::max());
        const auto sum_before = sums.receipt();
        rejects([&] { sums.add(std::numeric_limits<double>::max()); });
        check(sums.receipt() == sum_before, "finite seconds overflow preserves state");
        ProfileCounter counts;
        counts.add(UINT64_MAX);
        const auto count_before = counts.receipt();
        rejects([&] { counts.add(1); });
        check(counts.receipt() == count_before, "counter overflow preserves state");
        ProfileCounter extra;
        extra.add(1);
        rejects([&] { counts.merge(extra); });
        check(counts.receipt() == count_before, "merge overflow preserves state");
        std::cout << Json{{"status", "PASS_CPU_BACKEND_PROFILE_AGGREGATION"},
                          {"checks", checks},
                          {"expected_rejections", rejected},
                          {"scope", "synthetic helper inputs; no model or GPU execution"}}
                         .dump()
                  << '\n';
        return 0;
    } catch (const std::exception &error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
