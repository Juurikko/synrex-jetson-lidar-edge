// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Tuomas Pietilä
#pragma once
#include "taxonomy.hpp"
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
// Path-only streaming sessions/exporters retain the legacy contract. Generic
// sessions use a separate retained-artifact and descriptor/readiness path.
enum class PackageUse { file, legacy_stream };
struct Package {
    fs::path root;
    std::string manifest_bytes, manifest_sha256; // Same captured bytes are parsed and hashed.
    Json manifest;
    Contract contract;
    std::optional<Taxonomy> taxonomy; // Only explicit package-v3 metadata; no inference from K.
    explicit Package(const fs::path &root);
    Package(const fs::path &root, PackageUse use);
    fs::path artifact(const std::string &name) const;
};
void pack(const fs::path &spec, const fs::path &artifact, const std::string &kind,
          const fs::path &license, const fs::path &output);
void pack(const fs::path &spec, const fs::path &artifact, const std::string &kind,
          const fs::path &license, const fs::path &output, const fs::path &taxonomy);
void repack(const fs::path &source, const fs::path &taxonomy, const fs::path &output);

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
enum class GeometryMode { reference, direct, flat, radix };
const char *geometry_name(GeometryMode mode);
GeometryMode geometry_mode(const std::string &name);
struct GeometryProfile {
    double reference_seconds{}, minimum_seconds{}, assignment_seconds{}, grouping_seconds{},
           record_sort_seconds{}, member_sort_seconds{};
    std::uint64_t direct_points{}, boundary_fallback_points{}, tree_lookups{},
                  record_capacity_growths{}, member_capacity_growths{};
    bool reference_fallback{};
};
Plan geometry(const std::vector<Point> &points, const Contract &contract, GeometryMode mode,
              GeometryProfile *profile = nullptr);

std::vector<float> reference_tensor(const std::vector<Point> &points, const Block &block,
                                    const Contract &contract);
Json describe_plan(const std::vector<Point> &points, const Plan &plan, const Contract &contract);

bool gpu_enabled();
Json target_info();
void build_engine(const Package &package, const fs::path &output,
                  const std::string &precision = "default");
struct InferenceMetrics {
    double initialization_and_upload_seconds{};
    double cuda_preparation_seconds{};
    double tensorrt_inference_seconds{};
    double cuda_postprocessing_seconds{};
    std::uint64_t completed_inference_blocks{};
    std::uint64_t gpu_free_before_initialization_bytes{}, gpu_free_after_allocation_bytes{};
    std::uint64_t tracked_peak_requested_cuda_bytes{};
    Json persistent_cycles = Json::array();
};
struct InferenceOptions {
    unsigned warmup{}, iterations{1};
    fs::path tensor_trace;
};
std::vector<std::uint8_t> infer(const Package &package, const std::vector<Point> &points,
                                const Plan &plan, InferenceMetrics *metrics = nullptr,
                                std::vector<std::uint8_t> *semantic_ids = nullptr,
                                const InferenceOptions &options = {});
} // namespace synrex
