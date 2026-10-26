// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Tuomas Pietilä
// Original generic bounded tiling, deliberately distinct from model-specific geometry.
#include "synrex/runtime.hpp"
#include <algorithm>
#include <cmath>
#include <cstring>
#include <fstream>
#include <lazperf/readers.hpp>
#include <limits>
#include <map>

namespace synrex {
static std::uint64_t little(const unsigned char *p, int width) {
    std::uint64_t out = 0;
    for (int i = 0; i < width; ++i)
        out |= std::uint64_t(p[i]) << (8 * i);
    return out;
}
static double number(const unsigned char *p) {
    auto bits = little(p, 8);
    double value{};
    std::memcpy(&value, &bits, 8);
    require(std::isfinite(value), "nonfinite LAS scale/offset");
    return value;
}
std::vector<Point> decode(const fs::path &input, const Contract &c) {
    const auto count = validate_laz(input); // same fail-closed preservation contract as writer
    require(count <= c.max_points, "point budget exceeded");
    std::ifstream in(input, std::ios::binary);
    std::array<unsigned char, 375> header{};
    require(bool(in.read(reinterpret_cast<char *>(header.data()), header.size())),
            "truncated header");
    std::array<double, 3> scale{}, offset{};
    for (int a = 0; a < 3; ++a) {
        scale[a] = number(header.data() + 131 + 8 * a);
        offset[a] = number(header.data() + 155 + 8 * a);
        require(scale[a] > 0, "LAS coordinate scales must be positive");
    }
    lazperf::reader::named_file reader(input.string());
    require(reader.pointCount() == count && reader.header().point_record_length == 42 &&
                reader.header().point_format_id == 7,
            "decoder contract mismatch");
    std::vector<Point> points;
    points.reserve(count);
    std::array<unsigned char, 42> record{};
    for (std::uint64_t i = 0; i < count; ++i) {
        reader.readPoint(reinterpret_cast<char *>(record.data()));
        Point p{};
        for (int a = 0; a < 3; ++a) {
            const auto raw = static_cast<std::uint32_t>(little(record.data() + a * 4, 4));
            std::int32_t integer{};
            std::memcpy(&integer, &raw, 4);
            p.xyz[a] = integer * scale[a] + offset[a];
            require(std::isfinite(p.xyz[a]), "nonfinite scaled coordinate");
        }
        p.intensity = static_cast<std::uint16_t>(little(record.data() + 12, 2));
        points.push_back(p);
    }
    return points;
}
Plan geometry(const std::vector<Point> &points, const Contract &c) {
    require(!points.empty() && points.size() <= c.max_points, "invalid point count");
    auto minimum = points.front().xyz;
    for (const auto &p : points)
        for (int a = 0; a < 3; ++a) {
            require(std::isfinite(p.xyz[a]), "nonfinite coordinate");
            minimum[a] = std::min(minimum[a], p.xyz[a]);
        }
    std::array<double, 3> stride{};
    for (int a = 0; a < 3; ++a)
        stride[a] = c.extent[a] * (1 - c.overlap[a]);
    // Sparse point-driven block discovery avoids constructing a huge empty spatial grid.
    // Array lexicographic order defines x-major, then y, then z block traversal.
    std::map<std::array<int, 3>, Block> blocks;
    Plan plan;
    for (std::size_t i = 0; i < points.size(); ++i) {
        const auto &point = points[i];
        std::array<int, 3> lo{}, hi{};
        for (int a = 0; a < 3; ++a) {
            const auto relative = point.xyz[a] - minimum[a];
            require(std::isfinite(relative) && relative / stride[a] < 1000000000.0,
                    "coordinate range exceeds block index limit");
            // Conservative candidates, followed by exact half-open membership checks.
            lo[a] = static_cast<int>(
                std::max(0.0, std::floor((relative - c.extent[a]) / stride[a]) - 1));
            hi[a] = static_cast<int>(std::floor(relative / stride[a]) + 1);
        }
        bool covered = false;
        for (int x = lo[0]; x <= hi[0]; ++x)
            for (int y = lo[1]; y <= hi[1]; ++y)
                for (int z = lo[2]; z <= hi[2]; ++z) {
                    const std::array<int, 3> index{x, y, z};
                    std::array<double, 3> origin{};
                    std::array<int, 3> voxel{};
                    bool inside = true;
                    for (int a = 0; a < 3; ++a) {
                        origin[a] = minimum[a] + index[a] * stride[a];
                        const auto local = point.xyz[a] - origin[a];
                        if (!(local >= 0 && local < c.extent[a])) {
                            inside = false;
                            break;
                        }
                        const double v = std::floor(local / c.size[a]);
                        // Floating division may round a value just below extent onto dimensions.
                        voxel[a] = std::min(static_cast<int>(v), c.dimensions[a] - 1);
                    }
                    if (!inside)
                        continue;
                    require(++plan.memberships <= c.max_memberships, "membership budget exceeded");
                    if (!blocks.count(index)) {
                        require(blocks.size() < c.max_blocks, "block budget exceeded");
                        blocks.emplace(index, Block{origin, {}});
                    }
                    const auto linear =
                        (voxel[2] * c.dimensions[1] + voxel[1]) * c.dimensions[0] + voxel[0];
                    blocks.at(index).members.push_back(
                        {static_cast<std::uint32_t>(i), static_cast<std::uint32_t>(linear)});
                    covered = true;
                }
        require(covered, "point uncovered by geometry (coordinate precision)");
    }
    for (auto &pair : blocks) {
        auto &b = pair.second;
        std::sort(b.members.begin(), b.members.end(), [](const Member &a, const Member &b) {
            return a.voxel < b.voxel || (a.voxel == b.voxel && a.point < b.point);
        });
        plan.blocks.push_back(std::move(b));
    }
    return plan;
}
std::vector<float> reference_tensor(const std::vector<Point> &points, const Block &b,
                                    const Contract &c) {
    std::vector<float> tensor(c.voxels * c.features.size(), 0);
    for (std::size_t start = 0; start < b.members.size();) {
        auto end = start + 1;
        while (end < b.members.size() && b.members[end].voxel == b.members[start].voxel)
            ++end;
        double intensity = 0;
        std::array<double, 3> local{};
        for (auto i = start; i < end; ++i) {
            const auto &p = points.at(b.members[i].point);
            intensity += p.intensity;
            for (int a = 0; a < 3; ++a)
                local[a] += p.xyz[a] - b.origin[a];
        }
        const auto n = static_cast<double>(end - start);
        for (std::size_t f = 0; f < c.features.size(); ++f) {
            const auto &feature = c.features[f];
            double value = 1;
            if (feature.kind == "count")
                value = n;
            if (feature.kind == "mean_intensity")
                value = intensity / n;
            if (feature.kind == "local_x")
                value = local[0] / n;
            if (feature.kind == "local_y")
                value = local[1] / n;
            if (feature.kind == "local_z")
                value = local[2] / n;
            const auto normalized = (static_cast<float>(value) - feature.offset) * feature.scale;
            require(std::isfinite(normalized), "nonfinite feature");
            tensor[f * c.voxels + b.members[start].voxel] = normalized;
        }
        start = end;
    }
    return tensor;
}
Json describe_plan(const std::vector<Point> &points, const Plan &plan, const Contract &c) {
    require(plan.memberships <= 1000000, "plan diagnostic limited to one million memberships");
    Json blocks = Json::array();
    for (const auto &b : plan.blocks) {
        Json members = Json::array();
        for (const auto &m : b.members)
            members.push_back({m.point, m.voxel});
        blocks.push_back({{"origin", b.origin}, {"members", members}});
    }
    return {{"adapter", c.spec.at("adapter")},
            {"points", points.size()},
            {"memberships", plan.memberships},
            {"blocks", blocks}};
}
} // namespace synrex
