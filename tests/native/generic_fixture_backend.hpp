// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Tuomas Pietilä
#pragma once
// Include in exactly one translation unit of a CPU test executable. This opaque
// fixture implements no neural inference and performs no CUDA/TensorRT operation.
#include "../../native/generic_backend.hpp"
#include <atomic>
#include <chrono>
#include <functional>
#include <thread>

namespace fixture_backend {
enum class InitFault { none, throw_error, null_session, wrong_identity };
enum class RunFault { none, throw_error, missing_output, bad_semantic_id, bad_projection };
inline std::atomic<unsigned> init_calls{}, constructions{}, destructions{}, runs{}, live{};
inline std::atomic<unsigned> init_delay_ms{}, run_delay_ms{};
inline std::atomic<InitFault> init_fault{InitFault::none};
inline std::atomic<RunFault> run_fault{RunFault::none};
// Access only before worker launch or after worker join, as with fixture controls.
inline std::weak_ptr<const synrex::EngineArtifact> last_artifact;
inline const synrex::EngineArtifact *last_pointer{};
inline std::size_t last_capacity{};
inline synrex::SessionOptions last_options;
inline std::function<void()> init_hook;
inline void reset() {
    synrex::require(live == 0, "cannot reset live CPU backend fixture");
    init_calls = constructions = destructions = runs = 0;
    init_delay_ms = run_delay_ms = 0;
    init_fault = InitFault::none;
    run_fault = RunFault::none;
    last_artifact.reset();
    last_pointer = nullptr;
    last_capacity = 0;
    last_options = {};
    init_hook = {};
}
class Backend final : public synrex::InferenceSession {
    const std::shared_ptr<const synrex::EngineArtifact> retained_;
    const std::size_t capacity_;
    const bool wrong_identity_;

  public:
    Backend(std::shared_ptr<const synrex::EngineArtifact> retained, std::size_t capacity)
        : retained_(std::move(retained)), capacity_(capacity),
          wrong_identity_(init_fault == InitFault::wrong_identity) {
        ++constructions;
        ++live;
    }
    ~Backend() override {
        ++destructions;
        --live;
    }
    const synrex::Contract &contract() const override { return retained_->contract(); }
    synrex::Json identity() const override {
        return {{"backend", "CUDA/TensorRT"},
                {"manifest_sha256",
                 wrong_identity_ ? std::string(64, '0') : retained_->manifest_sha256()},
                {"artifacts", retained_->manifest().at("artifacts")},
                {"target", retained_->manifest().at("target")},
                {"max_window_points", capacity_},
                {"fixture", "opaque CPU test only; no neural inference or GPU execution"}};
    }
    synrex::WindowResult run(const std::vector<synrex::Point> &points,
                             const synrex::fs::path & = {}) override {
        ++runs;
        std::this_thread::sleep_for(std::chrono::milliseconds(run_delay_ms.load()));
        synrex::require(run_fault != RunFault::throw_error, "CPU fixture run failure");
        synrex::require(points.size() <= capacity_, "CPU fixture capacity exceeded");
        synrex::WindowResult result;
        result.timings = {{"cpu_fixture_seconds", 0.0}};
        const auto count = contract().las_classes.size();
        for (std::size_t i = 0; i < points.size(); ++i) {
            const auto id = static_cast<std::uint8_t>(i % 2 ? i % count : count - 1);
            result.semantic_ids.push_back(id);
            result.las_codes.push_back(contract().las_classes.at(id));
        }
        if (run_fault == RunFault::missing_output && !result.semantic_ids.empty())
            result.semantic_ids.pop_back();
        if (run_fault == RunFault::bad_semantic_id && !result.semantic_ids.empty())
            result.semantic_ids[0] = 254;
        if (run_fault == RunFault::bad_projection && !result.las_codes.empty())
            result.las_codes[0] ^= 1;
        return result;
    }
};
} // namespace fixture_backend

namespace synrex::generic::detail {
// Deliberately out of line, satisfying production's backend linkage seam.
std::unique_ptr<InferenceSession>
create_persistent_session(std::shared_ptr<const EngineArtifact> retained, std::size_t capacity,
                          const SessionOptions &options) {
    using namespace fixture_backend;
    ++init_calls;
    if (init_hook)
        init_hook();
    last_pointer = retained.get();
    last_artifact = retained;
    last_capacity = capacity;
    last_options = options;
    std::this_thread::sleep_for(std::chrono::milliseconds(init_delay_ms.load()));
    require(init_fault != InitFault::throw_error, "CPU fixture initialization failure");
    if (init_fault == InitFault::null_session)
        return {};
    return std::make_unique<Backend>(std::move(retained), capacity);
}
} // namespace synrex::generic::detail
