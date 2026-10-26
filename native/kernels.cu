// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Tuomas Pietilä
// Original generic CUDA feature reduction and source-index reconstruction.
#include "gpu.hpp"
#include <cmath>
#include <cuda_fp16.h>

namespace synrex {
template <typename T> __device__ float load(const T *p) { return static_cast<float>(*p); }
template <typename T> __device__ void store(T *p, float value) { *p = static_cast<T>(value); }
template <> __device__ float load<__half>(const __half *p) { return __half2float(*p); }
template <> __device__ void store<__half>(__half *p, float value) { *p = __float2half_rn(value); }

template <typename T>
__global__ void prepare_kernel(T *tensor, std::uint64_t voxels, const GpuPoint *points,
                               const std::uint32_t *members, const GpuGroup *groups,
                               unsigned group_count, const GpuFeature *features,
                               unsigned feature_count, GpuOrigin origin, int *bad) {
    const auto i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= group_count)
        return;
    const auto group = groups[i];
    double sums[4]{};
    // Fixed source-index order within each voxel; no nondeterministic atomic floating sums.
    for (auto m = group.begin; m < group.end; ++m) {
        const auto p = points[members[m]];
        for (int a = 0; a < 3; ++a)
            sums[a] += p.xyz[a] - origin.xyz[a];
        sums[3] += p.intensity;
    }
    const double count = group.end - group.begin;
    for (unsigned f = 0; f < feature_count; ++f) {
        const auto feature = features[f];
        double value = 1;
        if (feature.kind == 1)
            value = count;
        if (feature.kind >= 2)
            value = sums[feature.kind == 2 ? 3 : feature.kind - 3] / count;
        const float normalized = (static_cast<float>(value) - feature.offset) * feature.scale;
        const auto index = f * voxels + group.voxel;
        store(tensor + index, normalized);
        if (!isfinite(load(tensor + index)))
            atomicExch(bad, 1);
    }
}
template <typename T>
__global__ void scatter_kernel(const T *logits, std::uint64_t voxels, unsigned classes,
                               const std::uint32_t *points, const std::uint32_t *voxel_ids,
                               unsigned count, const std::uint8_t *mapping, std::uint32_t *labels,
                               std::uint32_t *semantic_ids, bool first_wins, int *bad) {
    const auto i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= count)
        return;
    const auto voxel = voxel_ids[i];
    float best = load(logits + voxel);
    unsigned winner = 0;
    if (!isfinite(best))
        atomicExch(bad, 1);
    for (unsigned c = 1; c < classes; ++c) {
        const float value = load(logits + c * voxels + voxel);
        if (!isfinite(value))
            atomicExch(bad, 1);
        if (value > best) {
            best = value;
            winner = c;
        }
    }
    // Geometry guarantees one reference per source point per block. Blocks share one stream.
    if (!first_wins || labels[points[i]] == 0xffffffffU) {
        labels[points[i]] = mapping[winner];
        semantic_ids[points[i]] = winner;
    }
}
void prepare_cuda(void *tensor, bool half, std::uint64_t voxels, const GpuPoint *points,
                  const std::uint32_t *members, const GpuGroup *groups, unsigned n,
                  const GpuFeature *features, unsigned feature_count, GpuOrigin origin, int *bad,
                  cudaStream_t stream) {
    if (half)
        prepare_kernel<<<(n + 255) / 256, 256, 0, stream>>>(static_cast<__half *>(tensor), voxels,
                                                            points, members, groups, n, features,
                                                            feature_count, origin, bad);
    else
        prepare_kernel<<<(n + 255) / 256, 256, 0, stream>>>(static_cast<float *>(tensor), voxels,
                                                            points, members, groups, n, features,
                                                            feature_count, origin, bad);
}
void scatter_cuda(const void *logits, bool half, std::uint64_t voxels, unsigned classes,
                  const std::uint32_t *points, const std::uint32_t *voxel_ids, unsigned n,
                  const std::uint8_t *mapping, std::uint32_t *labels, std::uint32_t *semantic_ids,
                  bool first_wins, int *bad, cudaStream_t stream) {
    if (half)
        scatter_kernel<<<(n + 255) / 256, 256, 0, stream>>>(
            static_cast<const __half *>(logits), voxels, classes, points, voxel_ids, n, mapping,
            labels, semantic_ids, first_wins, bad);
    else
        scatter_kernel<<<(n + 255) / 256, 256, 0, stream>>>(
            static_cast<const float *>(logits), voxels, classes, points, voxel_ids, n, mapping,
            labels, semantic_ids, first_wins, bad);
}
} // namespace synrex
