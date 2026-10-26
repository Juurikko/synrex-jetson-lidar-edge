// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Tuomas Pietilä
#pragma once
#include <array>
#include <cstdint>
#include <filesystem>
#include <nlohmann/json.hpp>
#include <string>
#include <vector>

namespace synrex {
namespace fs = std::filesystem;
using Json = nlohmann::json;
void require(bool ok, const std::string &message);
std::string sha256(const fs::path &path);
Json read_json(const fs::path &path);
void write_new(const fs::path &path, const std::string &bytes);

struct Feature {
    std::string kind;
    float offset, scale;
};
struct Contract {
    Json spec;
    std::array<int, 3> dimensions;
    std::array<double, 3> size, extent, overlap;
    std::vector<Feature> features;
    std::vector<std::uint8_t> las_classes;
    std::string input_name, output_name, input_dtype, output_dtype, policy;
    std::uint64_t voxels, max_points, max_blocks, max_memberships, max_tensor_bytes;
    explicit Contract(const Json &spec);
};
struct Package {
    fs::path root;
    Json manifest;
    Contract contract;
    explicit Package(const fs::path &root);
    fs::path artifact(const std::string &name) const;
};
void pack(const fs::path &spec, const fs::path &artifact, const std::string &kind,
          const fs::path &license, const fs::path &output);

struct Point {
    std::array<double, 3> xyz;
    std::uint16_t intensity;
};
struct Member {
    std::uint32_t point, voxel;
};
struct Block {
    std::array<double, 3> origin;
    std::vector<Member> members; // sorted by voxel, then original point index
};
struct Plan {
    std::vector<Block> blocks;
    std::uint64_t memberships{};
};
std::uint64_t validate_laz(const fs::path &input);
std::uint64_t apply_labels(const fs::path &input, const fs::path &labels, const fs::path &output);
std::vector<Point> decode(const fs::path &input, const Contract &contract);
Plan geometry(const std::vector<Point> &points, const Contract &contract);
std::vector<float> reference_tensor(const std::vector<Point> &points, const Block &block,
                                    const Contract &contract);
Json describe_plan(const std::vector<Point> &points, const Plan &plan, const Contract &contract);

bool gpu_enabled();
Json target_info();
void build_engine(const Package &package, const fs::path &output);
struct InferenceMetrics {
    double initialization_and_upload_seconds{};
    double cuda_preparation_seconds{};
    double tensorrt_inference_seconds{};
    double cuda_postprocessing_seconds{};
    std::uint64_t completed_inference_blocks{};
};
std::vector<std::uint8_t> infer(const Package &package, const std::vector<Point> &points,
                                const Plan &plan, InferenceMetrics *metrics = nullptr);
} // namespace synrex
