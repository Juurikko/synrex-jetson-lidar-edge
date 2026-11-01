// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Tuomas Pietilä
#pragma once
#include "synrex/runtime.hpp"

namespace synrex {
// Explicit serialization avoids native endianness, alignment or struct layout.
inline std::string trace_u32_le(const std::vector<std::uint32_t> &values) {
    require(values.size() <= 256ULL * 1024 * 1024 / 4, "trace mapping byte bound");
    std::string bytes(values.size() * 4, '\0');
    for (std::size_t i = 0; i < values.size(); ++i)
        for (unsigned b = 0; b < 4; ++b)
            bytes[4 * i + b] = static_cast<char>((values[i] >> (8 * b)) & 255);
    return bytes;
}
inline Json trace_membership_semantics(const Contract &contract) {
    return {{"membership_indexing", "zero-based local input-window point index"},
            {"voxel_linearization", "(iz * ny + iy) * nx + ix; x fastest"},
            {"membership_order", "actual backend traversal: voxel ascending, then point ascending"},
            {"overlap_policy", contract.policy},
            {"block_order", "actual execution order; overlap resolved by overlap_policy"}};
}
inline void trace_write_memberships(const fs::path &root, const std::string &stem,
                                    const std::vector<std::uint32_t> &points,
                                    const std::vector<std::uint32_t> &voxels, Json &block,
                                    std::uint64_t limit, std::uint64_t &written) {
    require(points.size() == voxels.size(), "trace mapping count mismatch");
    require(written <= limit && points.size() <= (limit - written) / 8,
            "trace mapping write byte budget");
    block["membership_count"] = points.size();
    for (const auto &entry : {std::make_pair("membership_points", &points),
                              std::make_pair("membership_voxels", &voxels)}) {
        const std::string key = entry.first;
        const auto filename =
            stem + (key == "membership_points" ? ".points.u32.bin" : ".voxels.u32.bin");
        const auto bytes = trace_u32_le(*entry.second);
        write_new(root / filename, bytes);
        written += bytes.size();
        block[key] = filename;
        block[key + "_sha256"] = sha256(root / filename);
    }
}
} // namespace synrex
