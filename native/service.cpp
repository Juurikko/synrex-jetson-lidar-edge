// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Tuomas Pietilä
#include "synrex/service.hpp"
#include <chrono>
#include <algorithm>
namespace synrex::streaming {
std::uint64_t steady_ns() {
    return std::chrono::duration_cast<std::chrono::nanoseconds>(
        std::chrono::steady_clock::now().time_since_epoch()).count();
}
Service::Service(std::unique_ptr<InferenceSession> session, std::size_t capacity,
                 std::function<void(const Classification &)> sink)
    : session_(std::move(session)), input_(capacity), output_(capacity), sink_(std::move(sink)) {
    require(bool(session_) && bool(sink_), "inference session and sink required");
    require(capacity <= 8, "stream service queues capped at eight frames each");
    writer_ = std::thread([this] { write(); });
    try { worker_ = std::thread([this] { work(); }); }
    catch (...) { cancel(); writer_.join(); throw; }
}
Service::~Service() {
    cancel();
    if (worker_.joinable()) worker_.join();
    if (writer_.joinable()) writer_.join();
}
void Service::fail(std::exception_ptr error) {
    { std::lock_guard<std::mutex> lock(error_mutex_); if (!error_) error_ = error; }
    cancel();
}
void Service::cancel() {
    cancelled_ = true;
    for (const auto &f : input_.cancel()) dropped_ += f.points.size();
    for (const auto &r : output_.cancel()) dropped_ += r.frame.points.size();
}
bool Service::submit(Frame frame, bool block) {
    const auto count = frame.points.size();
    require(count > 0 && count <= 65536 && count == frame.header.frame_points,
            "invalid complete frame submission");
    frame.submitted_steady_ns = steady_ns();
    submitted_ += count;
    if (!input_.push(std::move(frame), block)) {
        dropped_ += count; ++rejected_frames_; return false;
    }
    return true;
}
void Service::work() {
    while (auto frame = input_.pop()) {
        const auto count = frame->points.size();
        try {
            require(frame->accepted_steady_ns <= steady_ns(), "acceptance timestamp is in future");
            const auto conversion_start = steady_ns();
            const auto queue_wait = frame->submitted_steady_ns ?
                (conversion_start - frame->submitted_steady_ns) / 1e9 : 0.0;
            const auto &c = session_->contract();
            const auto &taxonomy = frame->header.taxonomy;
            require((taxonomy == "synrex.fractal7.v1" && c.las_classes ==
                         std::vector<std::uint8_t>{1, 2, 200, 6, 9, 17, 201}) ||
                    (taxonomy == "synrex.synthetic3.v1" && c.las_classes ==
                         std::vector<std::uint8_t>{2, 5, 6}),
                    "stream taxonomy/model mapping mismatch; no implicit LAS remap");
            std::vector<Point> points;
            points.reserve(count);
            const bool needs_intensity = std::any_of(c.features.begin(), c.features.end(),
                [](const auto &f) { return f.kind == "mean_intensity"; });
            for (const auto &p : frame->points) {
                require(p.intensity || !needs_intensity, "model needs missing source intensity");
                points.push_back({p.xyz, p.intensity.value_or(0)});
            }
            const auto conversion_seconds = (steady_ns() - conversion_start) / 1e9;
            auto result = session_->run(points);
            result.timings["cpu_point_conversion_seconds"] = conversion_seconds;
            result.timings["input_queue_wait_seconds"] = queue_wait;
            result.timings["inference_complete_steady_ns"] = steady_ns();
            require(result.semantic_ids.size() == count && result.las_codes.size() == count,
                    "inference coverage mismatch");
            classified_ += count; ++completed_windows_;
            Classification out{std::move(*frame), std::move(result.semantic_ids),
                                std::move(result.las_codes), std::move(result.timings)};
            out.timings["acceptance_to_inference_complete_seconds"] =
                (steady_ns() - out.frame.accepted_steady_ns) / 1e9;
            if (!output_.push(std::move(out))) dropped_ += count;
        } catch (...) { dropped_ += count; fail(std::current_exception()); break; }
    }
    output_.close();
}
void Service::write() {
    while (auto result = output_.pop()) {
        try {
            const auto complete = result->timings.value("inference_complete_steady_ns", std::uint64_t{});
            result->timings["output_queue_wait_seconds"] = complete ? (steady_ns() - complete) / 1e9 : 0.0;
            sink_(*result); emitted_ += result->frame.points.size();
        }
        catch (...) { dropped_ += result->frame.points.size(); fail(std::current_exception()); break; }
    }
}
void Service::finish() {
    input_.close();
    if (worker_.joinable()) worker_.join();
    if (writer_.joinable()) writer_.join();
    std::lock_guard<std::mutex> lock(error_mutex_);
    if (error_) std::rethrow_exception(error_);
}
Json Service::progress() const {
    return {{"status", !cancelled_ && dropped_ == 0 && submitted_ == emitted_ ? "PASS" : "FAIL"},
            {"submitted_points", submitted_.load()}, {"freshly_classified_points", classified_.load()},
            {"emitted_points", emitted_.load()}, {"dropped_points", dropped_.load()},
            {"rejected_frames", rejected_frames_.load()}, {"completed_windows", completed_windows_.load()},
            {"cancelled", cancelled_.load()}, {"input_queue", input_.stats()}, {"output_queue", output_.stats()}, {"inferred_classes_reused_points", 0},
            {"geometry_only_ingested_points", 0}};
}
Json Service::receipt() const {
    auto result = progress();
    result["session"] = session_->identity();
    return result;
}
} // namespace synrex::streaming
