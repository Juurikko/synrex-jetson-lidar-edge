// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Tuomas Pietilä
#pragma once
#include "synrex/generic_session.hpp"
#include <atomic>
#include <thread>

namespace synrex::generic {
// Complete v3 frames only; one producer, one inference worker, one output writer.
// Does not route global source IDs to the legacy single-file exporter/map path.
class Service final {
    std::unique_ptr<Session> session_;
    streaming::Queue<Frame> input_;
    streaming::Queue<Classification> output_;
    const std::function<void(const Classification &)> sink_;
    std::thread worker_, writer_;
    std::atomic<bool> cancelled_{false}, finished_{false};
    std::atomic<std::uint64_t> submitted_{0}, classified_{0}, emitted_{0}, dropped_{0},
        rejected_{0}, windows_{0};
    std::mutex error_mutex_;
    std::exception_ptr error_;
    void fail(std::exception_ptr);
    void work();
    void write();

  public:
    Service(std::unique_ptr<Session>, std::size_t queue_capacity,
            std::function<void(const Classification &)> sink);
    ~Service();
    Service(const Service &) = delete;
    Service &operator=(const Service &) = delete;
    // Sets local acceptance/submission time here; ignores any incoming host clock.
    bool submit(Frame frame, bool block = true);
    void cancel();
    void finish();         // Drain/join and propagate inference/sink failures.
    Json progress() const; // RUNNING until terminal drain has completed.
    Json receipt() const;  // Only after finish; never races session-owned counters.
};
} // namespace synrex::generic
