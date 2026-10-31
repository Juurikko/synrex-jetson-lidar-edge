// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Tuomas Pietilä
#pragma once
#include "synrex/generic_startup.hpp"
#include "synrex/stream.hpp"

namespace synrex::generic {
struct FrameIdentity {
    std::string descriptor_sha256, session_id;
    std::uint64_t map_epoch{};
};
struct Frame {
    FrameIdentity identity;
    std::string sensor_id, stream_id, coordinate_frame, crs;
    std::uint64_t sequence{}, frame_id{}, frame_start_ns{}, frame_end_ns{};
    std::vector<streaming::SourcePoint> points;
    std::uint64_t accepted_steady_ns{}, submitted_steady_ns{};
};
struct Classification {
    Frame frame;
    std::vector<std::uint8_t> semantic_ids, las_codes;
    Json timings = Json::object();
};
// CPU payload validation only. Callers must separately authorize readiness through
// the admitted transport/session before accepting or executing any frame.
// Encodings use SRX3/version 3, kinds 16 (frame) and 17 (classification), zero
// reserved bits, and a big-endian u32 payload length. The whole encoding counts
// against descriptor max_frame_bytes. No legacy codec/validation is used.
// Strings are nonempty printable ASCII, at most 256 bytes. Frames span <=1s;
// finite f64 coordinates are bounded to +/-1e9. Points use unique source IDs.
// Timings are <=64 flat finite nonnegative numeric measurements, <=16KiB JSON;
// they are untrusted remote measurements, never local host clock timestamps.
// Local accepted/submitted timestamps are not serialized; decode resets to zero.
void validate_frame(const Descriptor &, const Frame &);
void validate_classification(const Descriptor &, const Classification &);
std::string encode_frame(const Descriptor &, const Frame &);
Frame decode_frame(const Descriptor &, std::string_view);
std::string encode_classification(const Descriptor &, const Classification &);
Classification decode_classification(const Descriptor &, std::string_view);
} // namespace synrex::generic
