// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Tuomas Pietilä
#include "synrex/replay_frames.hpp"
#include <algorithm>
#include <cstring>
#include <limits>

namespace synrex::replay {
namespace {
constexpr std::uint64_t billion = 1000000000;
const Source &source_at(const Catalog &catalog, std::size_t index) {
    require(index < catalog.sources().size(), "replay frame source index");
    return catalog.sources()[index];
}
void validate_window(const Catalog &catalog, const FrameOptions &options, const Window &window) {
    const auto &source = source_at(catalog, window.source_index);
    require(!window.observations.empty() && window.observations.size() <= options.window_points,
            "replay frame window point count");
    const auto begin = window.observations.front().original_record_index;
    require(begin < source.records && begin % options.window_points == 0,
            "replay frame window boundary");
    require(window.observations.size() ==
                std::min<std::uint64_t>(options.window_points, source.records - begin),
            "replay frame premature partial window");
    for (std::size_t i = 0; i < window.observations.size(); ++i) {
        const auto &point = window.observations[i];
        require(point.original_record_index == begin + i &&
                    point.wire_id == catalog.wire_id(window.source_index, begin + i),
                "replay frame original/global row identity");
    }
}
} // namespace
void FrameOptions::validate(const generic::Descriptor &descriptor) const {
    require(window_points > 0 && window_points <= 65536 &&
                window_points <=
                    descriptor.spec().at("limits").at("max_frame_points").get<std::uint64_t>(),
            "replay window outside descriptor point limit");
    require(points_per_second > 0 && window_points <= points_per_second,
            "replay rate must be positive and window span at most one second");
}
std::uint64_t synthetic_time_ns(std::uint64_t wire_id, std::uint64_t rate) {
    require(rate > 0, "replay rate must be positive");
    // GCC/Clang extension used only for bounded integer arithmetic, not storage
    // or ABI. __extension__ keeps the project's -Wpedantic contract intact.
    __extension__ using Wide = unsigned __int128;
    const Wide time = Wide(wire_id) * billion / rate;
    require(time <= std::numeric_limits<std::uint64_t>::max(), "replay synthetic clock overflow");
    return static_cast<std::uint64_t>(time);
}
generic::Frame make_frame(const Catalog &catalog, const generic::Descriptor &descriptor,
                          const FrameOptions &options, const Window &window,
                          std::uint64_t sequence) {
    options.validate(descriptor);
    validate_window(catalog, options, window);
    const auto &source = catalog.sources()[window.source_index];
    generic::Frame frame;
    frame.identity = {descriptor.sha256(), descriptor.spec().at("session_id"),
                      descriptor.spec().at("map_epoch")};
    frame.sensor_id = "file-order-replay";
    frame.stream_id = source.sha256;
    frame.coordinate_frame = source.coordinate_frame;
    frame.crs = source.crs;
    frame.sequence = frame.frame_id = sequence;
    frame.frame_start_ns =
        synthetic_time_ns(window.observations.front().wire_id, options.points_per_second);
    frame.frame_end_ns =
        synthetic_time_ns(window.observations.back().wire_id, options.points_per_second);
    frame.points.reserve(window.observations.size());
    for (const auto &point : window.observations)
        frame.points.push_back(
            {point.wire_id, synthetic_time_ns(point.wire_id, options.points_per_second), point.xyz,
             point.intensity, point.return_number, point.return_count});
    generic::validate_frame(descriptor, frame);
    return frame;
}
Coverage::Coverage(const Catalog &catalog, const generic::Descriptor &descriptor,
                   FrameOptions options, std::size_t source_index, std::uint64_t sequence_begin)
    : catalog_(catalog), descriptor_(descriptor), options_(options), source_index_(source_index),
      sequence_begin_(sequence_begin) {
    options_.validate(descriptor_);
    const auto &source = source_at(catalog_, source_index_);
    // End-exclusive accounting must be representable even if the last point is.
    (void)synthetic_time_ns(source.wire_begin + source.records, options_.points_per_second);
    const auto count =
        source.records / options_.window_points + (source.records % options_.window_points != 0);
    require(count - 1 <= UINT64_MAX - sequence_begin_, "replay sequence overflow");
}
void Coverage::validate_next(const generic::Frame &frame) const {
    require(!failed_ && !eof_ && !finished_, "replay coverage failed or terminal");
    const auto &source = catalog_.sources()[source_index_];
    require(points_ < source.records, "replay coverage extra frame");
    generic::validate_frame(descriptor_, frame);
    const auto expected = std::min<std::uint64_t>(options_.window_points, source.records - points_);
    require(frame.sensor_id == "file-order-replay" && frame.stream_id == source.sha256 &&
                frame.coordinate_frame == source.coordinate_frame && frame.crs == source.crs,
            "replay coverage source/header identity");
    require(frame.sequence == sequence_begin_ + windows_ && frame.frame_id == frame.sequence,
            "replay coverage frame sequence");
    require(frame.points.size() == expected, "replay coverage partial/missing rows");
    const auto first = source.wire_begin + points_;
    require(frame.frame_start_ns == synthetic_time_ns(first, options_.points_per_second) &&
                frame.frame_end_ns ==
                    synthetic_time_ns(first + expected - 1, options_.points_per_second),
            "replay coverage frame timing");
    for (std::size_t i = 0; i < frame.points.size(); ++i)
        require(frame.points[i].id == first + i &&
                    frame.points[i].timestamp_ns ==
                        synthetic_time_ns(first + i, options_.points_per_second),
                "replay coverage global row/time identity");
}
void Coverage::advance(bool exact, std::size_t count) {
    points_ += count;
    ++windows_;
    if (exact)
        exact_points_ += count;
    else
        identity_points_ += count;
}
void Coverage::accept_frame(const generic::Frame &frame) {
    try {
        validate_next(frame);
        advance(false, frame.points.size());
    } catch (...) {
        failed_ = true;
        throw;
    }
}
void Coverage::accept(const Window &window, const generic::Frame &frame) {
    try {
        validate_next(frame);
        require(window.source_index == source_index_, "replay coverage decoded source identity");
        validate_window(catalog_, options_, window);
        require(window.observations.size() == frame.points.size(), "replay coverage decoded count");
        for (std::size_t i = 0; i < frame.points.size(); ++i) {
            const auto &expected = window.observations[i];
            const auto &actual = frame.points[i];
            require(expected.original_record_index == points_ + i && expected.wire_id == actual.id,
                    "replay coverage decoded row identity");
            require(std::memcmp(expected.xyz.data(), actual.xyz.data(), sizeof(expected.xyz)) ==
                            0 &&
                        actual.intensity == expected.intensity &&
                        actual.return_number == expected.return_number &&
                        actual.return_count == expected.return_count,
                    "replay coverage exact attributes");
        }
        advance(true, frame.points.size());
    } catch (...) {
        failed_ = true;
        throw;
    }
}
void Coverage::source_eof() {
    try {
        require(!failed_ && !finished_ && !eof_, "replay coverage failed or terminal");
        require(points_ == catalog_.sources()[source_index_].records,
                "replay coverage missing source rows");
        eof_ = true;
    } catch (...) {
        failed_ = true;
        throw;
    }
}
void Coverage::finish() {
    try {
        require(!failed_ && !finished_ && eof_,
                "replay coverage requires complete declared frames");
        finished_ = true;
    } catch (...) {
        failed_ = true;
        throw;
    }
}
Json Coverage::receipt() const {
    const auto &source = catalog_.sources()[source_index_];
    return {{"schema", "synrex.replay_source_coverage.v1"},
            {"state", failed_ ? "FAILED" : (finished_ ? "SUCCEEDED" : "RUNNING")},
            {"manifest_sha256", catalog_.manifest_sha256()},
            {"source_index", source_index_},
            {"source_sha256", source.sha256},
            {"descriptor_sha256", descriptor_.sha256()},
            {"session_id", descriptor_.spec().at("session_id")},
            {"map_epoch", descriptor_.spec().at("map_epoch")},
            {"coordinate_frame", source.coordinate_frame},
            {"crs", source.crs},
            {"window_point_limit", options_.window_points},
            {"sequence_begin", sequence_begin_},
            {"source_records", source.records},
            {"wire_begin", source.wire_begin},
            {"accepted_points", points_},
            {"windows", windows_},
            {"exact_attribute_points", exact_points_},
            {"identity_only_points", identity_points_},
            {"declared_frame_coverage_complete", eof_},
            {"points_per_second", options_.points_per_second},
            {"synthetic_duration_ns",
             synthetic_time_ns(source.wire_begin + source.records, options_.points_per_second) -
                 synthetic_time_ns(source.wire_begin, options_.points_per_second)},
            {"clock", "file-order-synthetic; not source acquisition/GPS time"}};
}
} // namespace synrex::replay
