// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Tuomas Pietilä
#pragma once
#include "session.hpp"
#include "stream.hpp"
#include <atomic>
#include <exception>
#include <functional>
#include <thread>
namespace synrex::streaming {
class Service {
    std::unique_ptr<InferenceSession> session_;
    Queue<Frame> input_;
    Queue<Classification> output_;
    std::function<void(const Classification &)> sink_;
    std::thread worker_, writer_;
    std::atomic<bool> cancelled_{false};
    std::atomic<std::uint64_t> submitted_{0}, classified_{0}, emitted_{0}, dropped_{0},
                               rejected_frames_{0}, completed_windows_{0};
    std::mutex error_mutex_;
    std::exception_ptr error_;
    void fail(std::exception_ptr error);
    void work();
    void write();
  public:
    Service(std::unique_ptr<InferenceSession> session, std::size_t queue_capacity,
             std::function<void(const Classification &)> sink);
    ~Service();
    // The single producer supplies monotonic acceptance times from its assembler.
    bool submit(Frame frame, bool block = true);
    void cancel(); // Finishes any active GPU window; never tears resources down mid-kernel.
    void finish(); // Drain and join; throws worker/sink errors after all threads are joined.
    Json progress() const; // Safe while running; does not inspect worker-owned session state.
    Json receipt() const;
};
std::uint64_t steady_ns();
} // namespace synrex::streaming
