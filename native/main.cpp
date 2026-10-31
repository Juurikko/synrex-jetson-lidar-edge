// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Tuomas Pietilä
#include "synrex/runtime.hpp"
#include <chrono>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <map>
#include <stdexcept>

namespace {
class Scratch {
    synrex::fs::path root_;

  public:
    explicit Scratch(const synrex::fs::path &output) {
        auto pattern = (output.parent_path() / ".synrex-byom-XXXXXX").string();
        std::vector<char> data(pattern.begin(), pattern.end());
        data.push_back('\0');
        auto p = ::mkdtemp(data.data());
        synrex::require(p != nullptr, "cannot create output scratch");
        root_ = p;
    }
    ~Scratch() {
        std::error_code ec;
        synrex::fs::remove_all(root_, ec);
    }
    synrex::fs::path labels() const { return root_ / "labels.u8"; }
    synrex::fs::path semantics() const { return root_ / "semantics.json"; }
};
void usage() {
    std::cout << "SynRex BYOM R0 — DENSE_VOXEL_SEGMENTATION_V1\n"
                 "synrex-native version\n"
                 "synrex-native target\n"
                 "synrex-native pack --spec SPEC.json --onnx MODEL.onnx --license NOTICE --out "
                 "MODEL.srxmodel [--taxonomy TAXONOMY.json]\n"
                 "synrex-native pack --spec SPEC.json --engine MODEL.engine --license NOTICE --out "
                 "MODEL.srxmodel [--taxonomy TAXONOMY.json]\n"
                 "synrex-native repack --model OLD.srxmodel --taxonomy TAXONOMY.json --out "
                 "NEW.srxmodel\n"
                 "synrex-native inspect --model MODEL.srxmodel\n"
                 "synrex-native build-engine --model ONNX.srxmodel --out MODEL.engine [--precision "
                 "fp16]\n"
                 "synrex-native plan --model MODEL.srxmodel --input INPUT.laz --out PLAN.json\n"
                 "synrex-native classify --model ENGINE.srxmodel --input INPUT.laz --out "
                 "OUTPUT.laz [--semantics IDS.json]\n"
                 "synrex-native benchmark --model ENGINE.srxmodel --input INPUT.laz --out "
                 "RECEIPT.json --warmup 3 --iterations 10\n";
}
} // namespace
int main(int argc, char **argv) {
    using namespace synrex;
    using Clock = std::chrono::steady_clock;
    const auto invocation_start = Clock::now();
    const auto seconds = [](Clock::time_point start, Clock::time_point end) {
        return std::chrono::duration<double>(end - start).count();
    };
    try {
        if (argc < 2) {
            usage();
            return 2;
        }
        const std::string command(argv[1]);
        if (command == "--help" && argc == 2) {
            usage();
            return 0;
        }
        if (command == "version" && argc == 2) {
            std::cout << Json{{"runtime", "synrex-native-r0"},
                              {"adapter", "DENSE_VOXEL_SEGMENTATION_V1"},
                              {"extensions",
                               {"synrex.point_semantics.v1", "synrex.dense_tensor_trace.v1",
                                "persistent_benchmark.v1"}},
                              {"cuda_tensorrt", gpu_enabled()}}
                             .dump(2)
                      << '\n';
            return 0;
        }
        if (command == "target" && argc == 2) {
            std::cout << target_info().dump(2) << '\n';
            return 0;
        }
        std::map<std::string, std::string> args;
        for (int i = 2; i < argc; i += 2) {
            require(i + 1 < argc && std::string(argv[i]).rfind("--", 0) == 0,
                    "expected --option value pairs");
            require(args.emplace(argv[i], argv[i + 1]).second, "duplicate CLI option");
        }
        auto get = [&args](const std::string &key) {
            require(args.count(key) == 1, "missing option: " + key);
            return args.at(key);
        };
        if (command == "pack") {
            require(args.size() == 4 + args.count("--taxonomy") &&
                        (args.count("--onnx") != args.count("--engine")),
                    "pack requires spec, model, license and out");
            const auto kind = args.count("--onnx") ? "onnx" : "engine";
            if (args.count("--taxonomy"))
                pack(get("--spec"), get(std::string("--") + kind), kind, get("--license"),
                     get("--out"), get("--taxonomy"));
            else
                pack(get("--spec"), get(std::string("--") + kind), kind, get("--license"),
                     get("--out"));
            std::cout << "{\"status\":\"PASS\",\"operation\":\"package_creation\"}\n";
            return 0;
        }
        if (command == "repack") {
            require(args.size() == 3, "repack requires model, taxonomy and out");
            repack(get("--model"), get("--taxonomy"), get("--out"));
            std::cout << "{\"status\":\"PASS\",\"operation\":\"taxonomy_repack\"}\n";
            return 0;
        }
        require(command == "inspect" || command == "build-engine" || command == "plan" ||
                    command == "classify" || command == "benchmark",
                "unknown command");
        const auto expected = command == "inspect" ? 1U : command == "build-engine" ? 2U : 3U;
        require(args.size() == expected ||
                    (command == "build-engine" && args.size() == expected + 1 &&
                     args.count("--precision")) ||
                    (command == "classify" &&
                     args.size() == 3 + args.count("--semantics") + args.count("--trace")) ||
                    (command == "benchmark" && args.size() == 5 && args.count("--warmup") &&
                     args.count("--iterations")),
                "unexpected CLI options");
        const Package package(get("--model"));
        if (command == "inspect") {
            std::cout << package.manifest.dump(2) << '\n';
            return 0;
        }
        const fs::path output(get("--out"));
        require(!fs::exists(output) && !fs::is_symlink(output), "refusing output replacement");
        if (command == "build-engine") {
            const auto precision = args.count("--precision") ? get("--precision") : "default";
            require(precision == "default" || precision == "fp16", "unsupported engine precision");
            build_engine(package, output, precision);
            return 0;
        }
        const fs::path input(get("--input"));
        if (command == "classify" || command == "benchmark")
            require(gpu_enabled(), "CUDA/TensorRT backend not built; inference NOT_RUN");
        fs::path sidecar;
        if (args.count("--semantics")) {
            sidecar = fs::absolute(get("--semantics")).lexically_normal();
            require(!fs::exists(sidecar) && !fs::is_symlink(sidecar),
                    "refusing semantic sidecar replacement");
            require(sidecar.parent_path() ==
                            fs::absolute(output).lexically_normal().parent_path() &&
                        sidecar != fs::absolute(output).lexically_normal(),
                    "semantic sidecar must be distinct and beside output LAZ");
        }
        const auto decode_start = Clock::now();
        const auto points = decode(input, package.contract);
        const auto geometry_start = Clock::now();
        const auto plan = geometry(points, package.contract);
        const auto preprocessing_end = Clock::now();
        if (command == "plan") {
            write_new(output, describe_plan(points, plan, package.contract).dump(2) + "\n");
            return 0;
        }
        InferenceMetrics metrics;
        InferenceOptions options;
        if (args.count("--trace"))
            options.tensor_trace = get("--trace");
        if (command == "benchmark") {
            const auto warmup = Json::parse(get("--warmup")),
                       count = Json::parse(get("--iterations"));
            require(warmup.is_number_integer() && count.is_number_integer() && warmup >= 0 &&
                        warmup <= 20 && count > 0 && count <= 100,
                    "invalid persistent benchmark budgets");
            options.warmup = warmup.get<unsigned>();
            options.iterations = count.get<unsigned>();
        }
        std::vector<std::uint8_t> semantic_ids;
        auto labels = infer(package, points, plan, &metrics,
                            sidecar.empty() ? nullptr : &semantic_ids, options);
        require(labels.size() == points.size(), "inference label count mismatch");
        Scratch scratch(output);
        write_new(scratch.labels(),
                  std::string(reinterpret_cast<const char *>(labels.data()), labels.size()));
        if (command == "benchmark") {
            const Json receipt{
                {"status", "PASS"},
                {"backend", "CUDA/TensorRT"},
                {"points", points.size()},
                {"blocks", plan.blocks.size()},
                {"warmup", options.warmup},
                {"iterations", options.iterations},
                {"scope", "Persistent engine/context/buffers and decoded cloud; LAZ I/O excluded; "
                          "not a streaming service"},
                {"initialization_seconds", metrics.initialization_and_upload_seconds},
                {"laz_decode_seconds", seconds(decode_start, geometry_start)},
                {"geometry_seconds", seconds(geometry_start, preprocessing_end)},
                {"cycles", metrics.persistent_cycles},
                {"source_order_labels_sha256", sha256(scratch.labels())},
                {"tracked_peak_requested_cuda_bytes", metrics.tracked_peak_requested_cuda_bytes},
                {"peak_scope", "Adapter and TensorRT allocator callbacks; excludes CUDA context "
                               "and untracked vendor allocations"}};
            write_new(output, receipt.dump(2) + "\n");
            std::cout << receipt.dump(2) << '\n';
            return 0;
        }
        if (!sidecar.empty()) {
            const Json semantic{{"schema", "synrex.point_semantics.v1"},
                                {"source_sha256", sha256(input)},
                                {"model_manifest_sha256",
                                 package.taxonomy ? package.manifest_sha256
                                                  : sha256(package.root / "manifest.json")},
                                {"point_count", points.size()},
                                {"index_order", "original source point record order"},
                                {"semantic_id_definition",
                                 "zero-based model output channel; no LAS class collapse"},
                                {"las_classes", package.contract.las_classes},
                                {"semantic_ids", semantic_ids},
                                {"mapped_labels_sha256", sha256(scratch.labels())}};
            write_new(scratch.semantics(), semantic.dump() + "\n");
        }
        const auto encode_start = Clock::now();
        const auto changed = apply_labels(input, scratch.labels(), output);
        if (!sidecar.empty()) {
            try {
                fs::create_hard_link(scratch.semantics(), sidecar);
            } catch (...) {
                std::error_code ignored;
                fs::remove(output, ignored);
                throw;
            }
        }
        const auto encode_end = Clock::now();
        const auto output_hash = sha256(output);
        const auto total_end = Clock::now();
        std::cout << Json{{"status", "PASS"},
                          {"backend", "CUDA/TensorRT"},
                          {"points", points.size()},
                          {"blocks", plan.blocks.size()},
                          {"memberships", plan.memberships},
                          {"changed", changed},
                          {"output_sha256", output_hash},
                          {"completed_inference_blocks", metrics.completed_inference_blocks},
                          {"semantic_sidecar", sidecar.empty()
                                                   ? Json(nullptr)
                                                   : Json{{"schema", "synrex.point_semantics.v1"},
                                                          {"sha256", sha256(sidecar)},
                                                          {"points", semantic_ids.size()}}},
                          {"gpu_memory_bytes",
                           {{"free_before_initialization",
                             metrics.gpu_free_before_initialization_bytes},
                            {"free_after_allocation", metrics.gpu_free_after_allocation_bytes},
                            {"tracked_peak_requested_cuda_bytes",
                             metrics.tracked_peak_requested_cuda_bytes},
                            {"tracked_peak_scope", "Adapter plus TensorRT callbacks; excludes "
                                                   "context/untracked vendor allocations"},
                            {"measurement",
                             "cudaMemGetInfo snapshots; device-wide delta, not allocator peak"}}},
                          {"timings_seconds",
                           {{"laz_decode_cpu", seconds(decode_start, geometry_start)},
                            {"geometry_cpu", seconds(geometry_start, preprocessing_end)},
                            {"preprocessing_cpu", seconds(decode_start, preprocessing_end)},
                            {"gpu_initialization_and_initial_upload",
                             metrics.initialization_and_upload_seconds},
                            {"cuda_tensor_preparation", metrics.cuda_preparation_seconds},
                            {"tensorrt_inference", metrics.tensorrt_inference_seconds},
                            {"cuda_postprocessing", metrics.cuda_postprocessing_seconds},
                            {"laz_encoding_and_preservation", seconds(encode_start, encode_end)},
                            {"end_to_end", seconds(invocation_start, total_end)}}}}
                         .dump(2)
                  << '\n';
        return 0;
    } catch (const std::exception &e) {
        std::cerr << synrex::Json{{"status", "FAIL"}, {"reason", e.what()}}.dump() << '\n';
        return 1;
    }
}
