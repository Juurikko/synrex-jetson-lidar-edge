// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Tuomas Pietilä
// Optional exact dense geometry strategies. geometry.cpp is the unchanged oracle.
#include "synrex/runtime.hpp"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <limits>
#include <map>

namespace synrex {
const char *geometry_name(GeometryMode mode) {
    switch (mode) {
    case GeometryMode::reference:
        return "reference";
    case GeometryMode::direct:
        return "direct";
    case GeometryMode::flat:
        return "flat";
    case GeometryMode::radix:
        return "radix";
    }
    throw std::runtime_error("unknown geometry strategy");
}
GeometryMode geometry_mode(const std::string &name) {
    if (name == "reference")
        return GeometryMode::reference;
    if (name == "direct")
        return GeometryMode::direct;
    if (name == "flat")
        return GeometryMode::flat;
    if (name == "radix")
        return GeometryMode::radix;
    throw std::runtime_error("unknown geometry strategy");
}
namespace {
using Clock = std::chrono::steady_clock;
double elapsed(Clock::time_point start) {
    return std::chrono::duration<double>(Clock::now() - start).count();
}
struct Record {
    std::array<int, 3> key;
    Member member;
};
bool member_less(const Member &a, const Member &b) {
    return a.voxel < b.voxel || (a.voxel == b.voxel && a.point < b.point);
}
} // namespace
Plan geometry(const std::vector<Point> &points, const Contract &c, GeometryMode mode,
              GeometryProfile *profile) {
    GeometryProfile local;
    auto &stats = profile ? *profile : local;
    stats = {};
    const auto start = Clock::now();
    if (mode == GeometryMode::reference ||
        std::any_of(c.overlap.begin(), c.overlap.end(), [](double x) { return x != 0; })) {
        auto plan = geometry(points, c);
        stats.reference_fallback = true;
        stats.reference_seconds = elapsed(start);
        return plan;
    }
    if (points.empty() || points.size() > c.max_points)
        require(false, "invalid point count");
    auto minimum = points.front().xyz;
    for (const auto &p : points)
        for (int a = 0; a < 3; ++a) {
            if (!std::isfinite(p.xyz[a]))
                require(false, "nonfinite coordinate");
            minimum[a] = std::min(minimum[a], p.xyz[a]);
        }
    stats.minimum_seconds = elapsed(start);
    const auto assign_start = Clock::now();
    std::vector<Record> records;
    records.reserve(std::min<std::uint64_t>(points.size(), c.max_memberships));
    auto append = [&](const std::array<int, 3> &key, std::size_t i,
                      const std::array<int, 3> &voxel) {
        if (records.size() >= c.max_memberships)
            require(false, "membership budget exceeded");
        if (records.size() == records.capacity())
            ++stats.record_capacity_growths;
        records.push_back(
            {key,
             {static_cast<std::uint32_t>(i),
              static_cast<std::uint32_t>((voxel[2] * c.dimensions[1] + voxel[1]) * c.dimensions[0] +
                                         voxel[0])}});
    };
    for (std::size_t i = 0; i < points.size(); ++i) {
        const auto &p = points[i];
        std::array<int, 3> key{}, voxel{}, lo{}, hi{};
        bool interior = true;
        for (int a = 0; a < 3; ++a) {
            const auto relative = p.xyz[a] - minimum[a];
            if (!(std::isfinite(relative) && relative / c.extent[a] < 1000000000.0))
                require(false, "coordinate range exceeds block index limit");
            key[a] = static_cast<int>(std::floor(relative / c.extent[a]));
            const auto product = key[a] * c.extent[a];
            const auto origin = minimum[a] + product;
            const auto offset = p.xyz[a] - origin;
            // Conservative roundoff guard for subtraction, quotient, multiplication and
            // translated origins. Boundary/low-precision points use the exact old search,
            // including its possible multiple memberships or uncovered-point rejection.
            const auto guard =
                64 * std::numeric_limits<double>::epsilon() *
                (std::abs(p.xyz[a]) + std::abs(minimum[a]) + std::abs(product) + c.extent[a]);
            if (!(offset >= 0 && offset < c.extent[a]) || guard >= c.extent[a] / 8 ||
                (key[a] > 0 && offset <= guard) || c.extent[a] - offset <= guard)
                interior = false;
            if (offset >= 0 && offset < c.extent[a])
                voxel[a] =
                    std::min(static_cast<int>(std::floor(offset / c.size[a])), c.dimensions[a] - 1);
        }
        if (interior) {
            append(key, i, voxel);
            ++stats.direct_points;
            continue;
        }
        ++stats.boundary_fallback_points;
        for (int a = 0; a < 3; ++a) {
            const auto relative = p.xyz[a] - minimum[a];
            lo[a] = static_cast<int>(
                std::max(0.0, std::floor((relative - c.extent[a]) / c.extent[a]) - 1));
            hi[a] = static_cast<int>(std::floor(relative / c.extent[a]) + 1);
        }
        bool covered = false;
        for (int x = lo[0]; x <= hi[0]; ++x)
            for (int y = lo[1]; y <= hi[1]; ++y)
                for (int z = lo[2]; z <= hi[2]; ++z) {
                    const std::array<int, 3> candidate{x, y, z};
                    bool inside = true;
                    for (int a = 0; a < 3; ++a) {
                        const auto origin = minimum[a] + candidate[a] * c.extent[a];
                        const auto offset = p.xyz[a] - origin;
                        if (!(offset >= 0 && offset < c.extent[a])) {
                            inside = false;
                            break;
                        }
                        voxel[a] = std::min(static_cast<int>(std::floor(offset / c.size[a])),
                                            c.dimensions[a] - 1);
                    }
                    if (inside) {
                        append(candidate, i, voxel);
                        covered = true;
                    }
                }
        if (!covered)
            require(false, "point uncovered by geometry (coordinate precision)");
    }
    stats.assignment_seconds = elapsed(assign_start);
    // Records were appended in ascending source index. A stable voxel-only radix
    // sort therefore produces exactly the authoritative (voxel, point) order.
    // Scratch is bounded by one block's membership count; no grid-sized allocation.
    std::vector<Member> scratch;
    auto order = [&](std::vector<Member> &members) {
        if (mode != GeometryMode::radix || members.size() < 256) {
            std::sort(members.begin(), members.end(), member_less);
            return;
        }
        scratch.resize(members.size());
        std::uint32_t maximum = 0;
        for (const auto &m : members)
            maximum = std::max(maximum, m.voxel);
        for (unsigned shift = 0; shift < 32 && (maximum >> shift); shift += 8) {
            std::array<std::size_t, 256> offsets{};
            for (const auto &m : members)
                ++offsets[(m.voxel >> shift) & 255];
            std::size_t start = 0;
            for (auto &offset : offsets) {
                const auto count = offset;
                offset = start;
                start += count;
            }
            for (const auto &m : members)
                scratch[offsets[(m.voxel >> shift) & 255]++] = m;
            members.swap(scratch);
        }
    };
    Plan plan;
    plan.memberships = records.size();
    auto origin_for = [&](const std::array<int, 3> &key) {
        std::array<double, 3> origin{};
        for (int a = 0; a < 3; ++a)
            origin[a] = minimum[a] + key[a] * c.extent[a];
        return origin;
    };
    if (mode == GeometryMode::flat) {
        const auto sort_start = Clock::now();
        std::sort(records.begin(), records.end(), [](const Record &a, const Record &b) {
            return a.key < b.key || (a.key == b.key && member_less(a.member, b.member));
        });
        stats.record_sort_seconds = elapsed(sort_start);
        const auto group_start = Clock::now();
        for (std::size_t begin = 0; begin < records.size();) {
            auto end = begin + 1;
            while (end < records.size() && records[end].key == records[begin].key)
                ++end;
            if (plan.blocks.size() >= c.max_blocks)
                require(false, "block budget exceeded");
            Block b{origin_for(records[begin].key), {}};
            b.members.reserve(end - begin);
            for (auto at = begin; at < end; ++at)
                b.members.push_back(records[at].member);
            plan.blocks.push_back(std::move(b));
            begin = end;
        }
        stats.grouping_seconds = elapsed(group_start);
    } else {
        const auto group_start = Clock::now();
        std::map<std::array<int, 3>, Block> blocks;
        for (const auto &record : records) {
            ++stats.tree_lookups;
            const auto found = blocks.try_emplace(record.key, Block{origin_for(record.key), {}});
            if (blocks.size() > c.max_blocks)
                require(false, "block budget exceeded");
            auto &members = found.first->second.members;
            if (members.size() == members.capacity())
                ++stats.member_capacity_growths;
            members.push_back(record.member);
        }
        stats.grouping_seconds = elapsed(group_start);
        const auto sort_start = Clock::now();
        plan.blocks.reserve(blocks.size());
        for (auto &entry : blocks) {
            order(entry.second.members);
            plan.blocks.push_back(std::move(entry.second));
        }
        stats.member_sort_seconds = elapsed(sort_start);
    }
    return plan;
}
} // namespace synrex
