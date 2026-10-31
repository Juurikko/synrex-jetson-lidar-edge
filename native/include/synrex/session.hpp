// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Tuomas Pietilä
#pragma once
#include "runtime.hpp"
#include <memory>
#include <stdexcept>
namespace synrex {
struct SessionOptions {
    bool pinned_staging{false};
    bool timeline{false};
    GeometryMode geometry{GeometryMode::reference};
};
struct WindowResult {
    std::vector<std::uint8_t> semantic_ids, las_codes;
    Json timings;
};
// Session is owned by exactly one inference worker and may not be called concurrently.
// CUDA context/stream, engine, execution context and all device buffers live for the session.
class InferenceSession {
  public:
    virtual ~InferenceSession() = default;
    virtual const Contract &contract() const = 0;
    virtual Json identity() const = 0;
    // Diagnostic model-only path: actual repeated enqueueV3 on a prepared occupied block.
    // No source points are emitted; points-equivalent rate is explicitly an upper bound.
    virtual Json benchmark_model(const std::vector<Point> &, unsigned, unsigned) {
        throw std::runtime_error("model-only benchmark unavailable for this backend");
    }
    virtual WindowResult run(const std::vector<Point> &points,
                              const fs::path &tensor_trace = {}) = 0;
};
Json session_resources(); // Owned tracked CUDA and pinned-host lifetimes; excludes driver/vendor caches.
// Legacy streaming factory: loads/validates the retained package with PackageUse::legacy_stream
// before any GPU resource construction. Explicit v3 packages require a separate generic factory
// and descriptor/readiness state path; this path-only API cannot waive that boundary.
// No ONNX compilation or Python execution in this path.
std::unique_ptr<InferenceSession> create_session(const fs::path &package,
                                                std::size_t max_window_points,
                                                const SessionOptions &options = {});
} // namespace synrex
