// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Tuomas Pietilä
#include "synrex/runtime.hpp"
#include <stdexcept>
namespace synrex {
bool gpu_enabled() { return false; }
Json target_info() {
    throw std::runtime_error("CUDA/TensorRT backend not built; GPU execution NOT_RUN");
}
void build_engine(const Package &, const fs::path &) { (void)target_info(); }
std::vector<std::uint8_t> infer(const Package &, const std::vector<Point> &, const Plan &,
                                InferenceMetrics *) {
    (void)target_info();
    return {};
}
} // namespace synrex
