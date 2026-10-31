// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Tuomas Pietilä
#include "synrex/generic_service.hpp"

namespace synrex::generic {
namespace {
std::uint64_t now_ns() {
    return std::chrono::duration_cast<std::chrono::nanoseconds>(
               std::chrono::steady_clock::now().time_since_epoch())
        .count();
}
std::size_t queue_capacity(std::size_t value) {
    require(value > 0 && value <= 8, "generic service queues capped at eight frames each");
    return value;
}
} // namespace
Service::Service(std::unique_ptr<Session> session, std::size_t capacity,
                 std::function<void(const Classification &)> sink)
    : session_(std::move(session)), input_(queue_capacity(capacity)),
      output_(queue_capacity(capacity)), sink_(std::move(sink)) {
    require(bool(session_) && bool(sink_), "generic session and sink required");
    writer_ = std::thread([this] { write(); });
    try {
        worker_ = std::thread([this] { work(); });
    } catch (...) {
        cancel();
        writer_.join();
        throw;
    }
}
Service::~Service() {
    cancel();
    if (worker_.joinable())
        worker_.join();
    if (writer_.joinable())
        writer_.join();
}
void Service::fail(std::exception_ptr error) {
    {
        std::lock_guard<std::mutex> lock(error_mutex_);
        if (!error_)
            error_ = error;
    }
    cancel();
}
void Service::cancel() {
    cancelled_ = true;
    for (const auto &frame : input_.cancel())
        dropped_ += frame.points.size();
    for (const auto &result : output_.cancel())
        dropped_ += result.frame.points.size();
}
bool Service::submit(Frame frame, bool block) {
    require(!finished_, "generic service already finished");
    try {
        validate_frame(session_->descriptor(), frame);
    } catch (...) {
        ++rejected_;
        fail(std::current_exception());
        throw;
    }
    const auto count = frame.points.size();
    frame.accepted_steady_ns = now_ns();
    frame.submitted_steady_ns = frame.accepted_steady_ns;
    submitted_ += count;
    if (!input_.push(std::move(frame), block)) {
        dropped_ += count;
        ++rejected_;
        return false;
    }
    return true;
}
void Service::work() {
    while (auto frame = input_.pop()) {
        const auto count = frame->points.size();
        try {
            const auto started = now_ns();
            const auto accepted = frame->accepted_steady_ns;
            auto result = session_->run(std::move(*frame));
            result.timings["service_input_queue_wait_seconds"] = (started - accepted) / 1e9;
            result.timings["service_inference_complete_steady_ns"] = now_ns();
            result.timings["service_acceptance_to_inference_seconds"] = (now_ns() - accepted) / 1e9;
            // Added measurements count against the same declared output budget.
            (void)encode_classification(session_->descriptor(), result);
            classified_ += count;
            ++windows_;
            if (!output_.push(std::move(result)))
                dropped_ += count;
        } catch (...) {
            dropped_ += count;
            fail(std::current_exception());
            break;
        }
    }
    output_.close();
}
void Service::write() {
    while (auto result = output_.pop()) {
        try {
            sink_(*result);
            emitted_ += result->frame.points.size();
        } catch (...) {
            dropped_ += result->frame.points.size();
            fail(std::current_exception());
            break;
        }
    }
}
void Service::finish() {
    input_.close();
    if (worker_.joinable())
        worker_.join();
    if (writer_.joinable())
        writer_.join();
    finished_ = true;
    std::lock_guard<std::mutex> lock(error_mutex_);
    if (error_)
        std::rethrow_exception(error_);
}
Json Service::progress() const {
    const bool okay =
        !cancelled_ && dropped_ == 0 && rejected_ == 0 && submitted_ > 0 && submitted_ == emitted_;
    return {{"schema", "synrex.generic_service.v1"},
            {"status", finished_ ? (okay ? "PASS" : "FAIL") : (cancelled_ ? "FAIL" : "RUNNING")},
            {"submitted_points", submitted_.load()},
            {"freshly_classified_points", classified_.load()},
            {"emitted_points", emitted_.load()},
            {"dropped_points", dropped_.load()},
            {"rejected_frames", rejected_.load()},
            {"completed_windows", windows_.load()},
            {"cancelled", cancelled_.load()},
            {"finished", finished_.load()},
            {"input_queue", input_.stats()},
            {"output_queue", output_.stats()},
            {"time_boundary", "local_service_acceptance_to_inference; accumulation excluded"}};
}
Json Service::receipt() const {
    require(finished_, "generic service receipt requires terminal drain");
    auto result = progress();
    result["session"] = session_->identity();
    return result;
}
} // namespace synrex::generic
