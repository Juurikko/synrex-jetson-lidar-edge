// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Tuomas Pietilä
#include "synrex/generic_session.hpp"
#include "generic_backend.hpp"
#include "synrex/generic_transport.hpp"
#include <algorithm>

namespace synrex::generic {
Session::Session(std::unique_ptr<InferenceSession> backend, Descriptor descriptor,
                 std::function<void(std::uint64_t, std::uint64_t)> authorize,
                 std::function<void()> abort)
    : backend_(std::move(backend)), descriptor_(std::move(descriptor)),
      authorize_(std::move(authorize)), abort_(std::move(abort)) {}

std::unique_ptr<Session> AuthorityTransport::create_session(const SessionOptions &options) {
    try {
        require(!options.timeline, "generic wire output does not support timeline arrays");
        const auto artifact = begin_session();
        const auto capacity = negotiated_limits().max_frame_points;
        auto backend = detail::create_persistent_session(artifact, capacity, options);
        require(bool(backend), "generic backend returned no session");
        const auto identity = backend->identity();
        require(backend->contract().spec == artifact->contract().spec &&
                    identity.at("manifest_sha256") == artifact->manifest_sha256() &&
                    identity.at("artifacts") == artifact->manifest().at("artifacts") &&
                    identity.at("target") == artifact->manifest().at("target") &&
                    identity.at("max_window_points") == capacity &&
                    identity.at("backend") == "CUDA/TensorRT",
                "loaded generic backend identity mismatch");
        auto session = std::unique_ptr<Session>(
            new Session(std::move(backend), descriptor(), frame_authorizer(), session_aborter()));
        // Check the startup deadline after synchronous vendor construction as well
        // as before it. Expiry disqualifies the session; vendor work is not preempted.
        complete_session();
        return session;
    } catch (...) {
        abort_session();
        throw;
    }
}

Classification Session::run(Frame frame) {
    try {
        require(!poisoned_, "generic session permanently failed");
        // The codec performs size, full identity and source validation. No legacy
        // taxonomy allowlist or parser is used, and host clocks are not wire data.
        const auto encoded = encode_frame(descriptor_, frame);
        authorize_(frame.points.size(), encoded.size());
        const std::array<std::string, 4> source{frame.sensor_id, frame.stream_id,
                                                frame.coordinate_frame, frame.crs};
        require(!source_identity_ || *source_identity_ == source,
                "generic source identity changed within session");
        require(!last_frame_ ||
                    (frame.sequence > (*last_frame_)[0] && frame.frame_id > (*last_frame_)[1] &&
                     frame.frame_start_ns >= (*last_frame_)[2]),
                "generic frame sequence/time replay or regression");
        source_identity_ = source;
        last_frame_ = {frame.sequence, frame.frame_id, frame.frame_end_ns};
        const auto &contract = backend_->contract();
        const bool intensity =
            std::any_of(contract.features.begin(), contract.features.end(),
                        [](const auto &f) { return f.kind == "mean_intensity"; });
        std::vector<Point> points;
        points.reserve(frame.points.size());
        for (const auto &point : frame.points) {
            require(point.intensity || !intensity, "model needs missing source intensity");
            points.push_back({point.xyz, point.intensity.value_or(0)});
        }
        auto result = backend_->run(points);
        Classification output{std::move(frame), std::move(result.semantic_ids),
                              std::move(result.las_codes), std::move(result.timings)};
        // Validate full coverage, ID255/K256 and exact admitted LAS projection,
        // including the actual encoded output budget before exposing results.
        (void)encode_classification(descriptor_, output);
        return output;
    } catch (...) {
        poisoned_ = true;
        abort_();
        throw;
    }
}
Json Session::identity() const {
    auto result = backend_->identity();
    result["generic_session"] = {{"descriptor_sha256", descriptor_.sha256()},
                                 {"descriptor", descriptor_.spec()},
                                 {"ready_exchange_completed", true},
                                 {"backend_failed", poisoned_}};
    return result;
}
} // namespace synrex::generic
