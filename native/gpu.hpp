// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Tuomas Pietilä
#pragma once
#include <cstdint>
#include <cuda_runtime.h>
namespace synrex {
struct GpuPoint {
    double xyz[3];
    double intensity;
};
struct GpuFeature {
    int kind;
    float offset, scale;
};
struct GpuGroup {
    std::uint32_t voxel, begin, end;
};
struct GpuOrigin {
    double xyz[3];
};
void prepare_cuda(void *tensor, bool half, std::uint64_t voxels, const GpuPoint *points,
                  const std::uint32_t *members, const GpuGroup *groups, unsigned group_count,
                  const GpuFeature *features, unsigned feature_count, GpuOrigin origin, int *bad,
                  cudaStream_t stream);
void scatter_cuda(const void *logits, bool half, std::uint64_t voxels, unsigned classes,
                  const std::uint32_t *point_ids, const std::uint32_t *voxel_ids, unsigned count,
                  const std::uint8_t *mapping, std::uint32_t *labels, std::uint32_t *semantic_ids,
                  bool first_wins, int *bad, cudaStream_t stream);
} // namespace synrex
