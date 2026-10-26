// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Tuomas Pietilä
#include "synrex/runtime.hpp"
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
};
void usage() {
    std::cout
        << "SynRex BYOM R0 — DENSE_VOXEL_SEGMENTATION_V1\n"
           "synrex-native version\n"
           "synrex-native target\n"
           "synrex-native pack --spec SPEC.json --onnx MODEL.onnx --license NOTICE --out "
           "MODEL.srxmodel\n"
           "synrex-native pack --spec SPEC.json --engine MODEL.engine --license NOTICE --out "
           "MODEL.srxmodel\n"
           "synrex-native inspect --model MODEL.srxmodel\n"
           "synrex-native build-engine --model ONNX.srxmodel --out MODEL.engine\n"
           "synrex-native plan --model MODEL.srxmodel --input INPUT.laz --out PLAN.json\n"
           "synrex-native classify --model ENGINE.srxmodel --input INPUT.laz --out OUTPUT.laz\n";
}
} // namespace
int main(int argc, char **argv) {
    using namespace synrex;
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
            require(args.size() == 4 && (args.count("--onnx") != args.count("--engine")),
                    "pack requires spec, model, license and out");
            const auto kind = args.count("--onnx") ? "onnx" : "engine";
            pack(get("--spec"), get(std::string("--") + kind), kind, get("--license"),
                 get("--out"));
            std::cout << "{\"status\":\"PASS\",\"operation\":\"package_creation\"}\n";
            return 0;
        }
        require(command == "inspect" || command == "build-engine" || command == "plan" ||
                    command == "classify",
                "unknown command");
        const auto expected = command == "inspect" ? 1U : command == "build-engine" ? 2U : 3U;
        require(args.size() == expected, "unexpected CLI options");
        const Package package(get("--model"));
        if (command == "inspect") {
            std::cout << package.manifest.dump(2) << '\n';
            return 0;
        }
        const fs::path output(get("--out"));
        require(!fs::exists(output) && !fs::is_symlink(output), "refusing output replacement");
        if (command == "build-engine") {
            build_engine(package, output);
            return 0;
        }
        const fs::path input(get("--input"));
        if (command == "classify")
            require(gpu_enabled(), "CUDA/TensorRT backend not built; inference NOT_RUN");
        const auto points = decode(input, package.contract);
        const auto plan = geometry(points, package.contract);
        if (command == "plan") {
            write_new(output, describe_plan(points, plan, package.contract).dump(2) + "\n");
            return 0;
        }
        auto labels = infer(package, points, plan);
        require(labels.size() == points.size(), "inference label count mismatch");
        Scratch scratch(output);
        write_new(scratch.labels(),
                  std::string(reinterpret_cast<const char *>(labels.data()), labels.size()));
        const auto changed = apply_labels(input, scratch.labels(), output);
        std::cout << Json{{"status", "PASS"},
                          {"backend", "CUDA/TensorRT"},
                          {"points", points.size()},
                          {"blocks", plan.blocks.size()},
                          {"memberships", plan.memberships},
                          {"changed", changed},
                          {"output_sha256", sha256(output)}}
                         .dump(2)
                  << '\n';
        return 0;
    } catch (const std::exception &e) {
        std::cerr << synrex::Json{{"status", "FAIL"}, {"reason", e.what()}}.dump() << '\n';
        return 1;
    }
}
