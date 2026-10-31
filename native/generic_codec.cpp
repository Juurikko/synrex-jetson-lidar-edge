// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Tuomas Pietilä
#include "synrex/generic_codec.hpp"
#include <algorithm>
#include <cmath>
#include <cstring>
#include <limits>
#include <set>
#include <stdexcept>

namespace synrex::generic {
namespace {
constexpr std::size_t header_bytes = 12, point_bytes = 48, string_bound = 256;
constexpr std::size_t timings_bound = 16384, timing_fields = 64;
constexpr unsigned frame_kind = 16, classification_kind = 17;
static_assert(sizeof(double) == 8 && std::numeric_limits<double>::is_iec559,
              "binary-v3 requires IEEE754 binary64");
void check(bool value, const char *message) {
    if (!value)
        throw std::runtime_error(message);
}
std::uint64_t limit(const Descriptor &descriptor, const char *name) {
    return descriptor.spec().at("limits").at(name).get<std::uint64_t>();
}
void identifier(std::string_view value) {
    check(!value.empty() && value.size() <= string_bound &&
              std::all_of(value.begin(), value.end(),
                          [](unsigned char c) { return c >= 32 && c <= 126; }),
          "generic string bound/printable ASCII");
}
void identity(const Descriptor &descriptor, const FrameIdentity &value) {
    check(value.descriptor_sha256 == descriptor.sha256() &&
              value.session_id == descriptor.spec().at("session_id").get<std::string>() &&
              value.map_epoch == descriptor.spec().at("map_epoch").get<std::uint64_t>(),
          "generic descriptor/session/epoch mismatch");
}
std::size_t frame_size(const Frame &frame) {
    return header_bytes + 64 + 32 + 8 + 8 + frame.sensor_id.size() + frame.stream_id.size() +
           frame.coordinate_frame.size() + frame.crs.size() + 40 +
           point_bytes * frame.points.size();
}
void timing_values(const Json &timings) {
    check(timings.is_object() && timings.size() <= timing_fields, "generic timings object bound");
    for (auto entry = timings.begin(); entry != timings.end(); ++entry) {
        identifier(entry.key());
        check(entry.value().is_number(), "generic timing numeric value required");
        const auto value = entry.value().get<double>();
        check(std::isfinite(value) && value >= 0, "generic timing finite nonnegative value");
    }
}
std::string timing_bytes(const Json &timings) {
    timing_values(timings);
    auto bytes = timings.dump();
    check(bytes.size() <= timings_bound, "generic timings byte limit");
    return bytes;
}
// Reject nested values and duplicate decoded keys before creating a DOM.
struct TimingGuard : nlohmann::json_sax<Json> {
    bool root{}, ended{};
    std::set<std::string> keys;
    bool null() override { return false; }
    bool boolean(bool) override { return false; }
    bool number_integer(number_integer_t value) override { return root && !ended && value >= 0; }
    bool number_unsigned(number_unsigned_t) override { return root && !ended; }
    bool number_float(number_float_t value, const string_t &) override {
        return root && !ended && std::isfinite(value) && value >= 0;
    }
    bool string(string_t &) override { return false; }
    bool binary(binary_t &) override { return false; }
    bool start_object(std::size_t) override {
        if (root)
            return false;
        return root = true;
    }
    bool key(string_t &key) override {
        identifier(key);
        return keys.size() < timing_fields && keys.insert(key).second;
    }
    bool end_object() override { return ended = true; }
    bool start_array(std::size_t) override { return false; }
    bool end_array() override { return false; }
    bool parse_error(std::size_t, const std::string &, const Json::exception &) override {
        return false;
    }
};
void put(std::string &bytes, std::uint64_t value, unsigned width) {
    for (unsigned i = width; i; --i)
        bytes.push_back(static_cast<char>(value >> (8 * (i - 1))));
}
void put_string(std::string &bytes, const std::string &value) {
    put(bytes, value.size(), 2);
    bytes.append(value);
}
struct Reader {
    std::string_view bytes;
    std::size_t offset{};
    std::size_t left() const { return bytes.size() - offset; }
    std::string_view take(std::size_t count) {
        check(count <= left(), "generic truncated payload");
        const auto value = bytes.substr(offset, count);
        offset += count;
        return value;
    }
    std::uint64_t get(unsigned width) {
        std::uint64_t value{};
        for (unsigned char byte : take(width))
            value = (value << 8) | byte;
        return value;
    }
    std::string text() {
        const auto size = get(2);
        check(size >= 1 && size <= string_bound, "generic string byte limit");
        const auto value = take(size);
        identifier(value);
        return std::string(value);
    }
};
Reader payload(const Descriptor &descriptor, std::string_view bytes, unsigned kind) {
    check(bytes.size() >= header_bytes, "generic truncated header");
    check(bytes.size() <= limit(descriptor, "max_frame_bytes"), "generic frame byte limit");
    Reader reader{bytes};
    check(reader.take(4) == "SRX3", "generic magic");
    check(reader.get(1) == 3, "generic version");
    check(reader.get(1) == kind, "generic message kind");
    check(reader.get(2) == 0, "generic reserved header");
    const auto size = reader.get(4);
    check(size == reader.left(), "generic payload length/trailing bytes");
    return reader;
}
void encode_body(std::string &bytes, const Frame &frame) {
    bytes.append(frame.identity.descriptor_sha256);
    bytes.append(frame.identity.session_id);
    put(bytes, frame.identity.map_epoch, 8);
    for (const auto *text :
         {&frame.sensor_id, &frame.stream_id, &frame.coordinate_frame, &frame.crs})
        put_string(bytes, *text);
    for (auto value : {frame.sequence, frame.frame_id, frame.frame_start_ns, frame.frame_end_ns,
                       static_cast<std::uint64_t>(frame.points.size())})
        put(bytes, value, 8);
    for (const auto &point : frame.points) {
        put(bytes, point.id, 8);
        put(bytes, point.timestamp_ns, 8);
        for (const auto axis : point.xyz) {
            std::uint64_t bits;
            std::memcpy(&bits, &axis, 8);
            put(bytes, bits, 8);
        }
        put(bytes, (point.intensity ? 1 : 0) | (point.return_number ? 2 : 0), 1);
        put(bytes, point.return_number.value_or(0), 1);
        put(bytes, point.return_count.value_or(0), 1);
        put(bytes, 0, 1);
        put(bytes, point.intensity.value_or(0), 2);
        put(bytes, 0, 2);
    }
}
Frame decode_body(const Descriptor &descriptor, Reader &reader, bool output) {
    Frame frame;
    frame.identity.descriptor_sha256 = std::string(reader.take(64));
    frame.identity.session_id = std::string(reader.take(32));
    frame.identity.map_epoch = reader.get(8);
    identity(descriptor, frame.identity);
    frame.sensor_id = reader.text();
    frame.stream_id = reader.text();
    frame.coordinate_frame = reader.text();
    frame.crs = reader.text();
    frame.sequence = reader.get(8);
    frame.frame_id = reader.get(8);
    frame.frame_start_ns = reader.get(8);
    frame.frame_end_ns = reader.get(8);
    const auto count = reader.get(8);
    check(count > 0 && count <= limit(descriptor, "max_frame_points"), "generic frame point limit");
    // count is bounded before multiplication or allocation; include labels and
    // the minimum timing JSON for an output, with no input trailing bytes.
    const auto record = point_bytes + (output ? 2 : 0);
    check(count <= reader.left() / record, "generic point count exceeds payload");
    if (output)
        check(reader.left() - count * record >= 6, "generic missing timings");
    else
        check(count * record == reader.left(), "generic frame trailing bytes");
    frame.points.reserve(static_cast<std::size_t>(count));
    for (std::uint64_t i = 0; i < count; ++i) {
        streaming::SourcePoint point;
        point.id = reader.get(8);
        point.timestamp_ns = reader.get(8);
        for (auto &axis : point.xyz) {
            const auto bits = reader.get(8);
            std::memcpy(&axis, &bits, 8);
        }
        const auto flags = reader.get(1), number = reader.get(1), returns = reader.get(1);
        check((flags & ~3ULL) == 0 && reader.get(1) == 0, "generic point flags/reserved");
        const auto intensity = reader.get(2);
        check(reader.get(2) == 0, "generic point reserved");
        check((flags & 1) || intensity == 0, "generic absent intensity must be zero");
        check((flags & 2) || (number == 0 && returns == 0), "generic absent returns must be zero");
        if (flags & 1)
            point.intensity = static_cast<std::uint16_t>(intensity);
        if (flags & 2) {
            point.return_number = static_cast<std::uint8_t>(number);
            point.return_count = static_cast<std::uint8_t>(returns);
        }
        frame.points.push_back(point);
    }
    validate_frame(descriptor, frame);
    return frame;
}
std::string envelope(std::string bytes, unsigned kind) {
    const auto size = bytes.size() - header_bytes;
    bytes.replace(0, 4, "SRX3");
    bytes[4] = 3;
    bytes[5] = static_cast<char>(kind);
    for (unsigned i = 0; i < 4; ++i)
        bytes[8 + i] = static_cast<char>(size >> (8 * (3 - i)));
    return bytes;
}
} // namespace
void validate_frame(const Descriptor &descriptor, const Frame &frame) {
    identity(descriptor, frame.identity);
    for (const auto *text :
         {&frame.sensor_id, &frame.stream_id, &frame.coordinate_frame, &frame.crs})
        identifier(*text);
    check(frame.points.size() > 0 && frame.points.size() <= limit(descriptor, "max_frame_points"),
          "generic frame point limit");
    check(frame_size(frame) <= limit(descriptor, "max_frame_bytes"), "generic frame byte limit");
    check(frame.frame_end_ns >= frame.frame_start_ns &&
              frame.frame_end_ns - frame.frame_start_ns <= 1000000000ULL,
          "generic unsafe frame span");
    std::set<std::uint64_t> ids;
    for (const auto &point : frame.points) {
        check(ids.insert(point.id).second, "generic duplicate source ID");
        check(point.timestamp_ns >= frame.frame_start_ns &&
                  point.timestamp_ns <= frame.frame_end_ns,
              "generic source timestamp outside frame");
        for (const auto axis : point.xyz)
            check(std::isfinite(axis) && std::abs(axis) <= 1e9, "generic unsafe coordinate");
        check(point.return_number.has_value() == point.return_count.has_value(),
              "generic paired return attributes required");
        if (point.return_number)
            check(*point.return_number > 0 && *point.return_number <= *point.return_count &&
                      *point.return_count <= 15,
                  "generic invalid return attributes");
    }
}
void validate_classification(const Descriptor &descriptor, const Classification &result) {
    validate_frame(descriptor, result.frame);
    check(result.semantic_ids.size() == result.frame.points.size() &&
              result.las_codes.size() == result.frame.points.size(),
          "generic output label count");
    const auto &classes = descriptor.taxonomy().classes();
    for (std::size_t i = 0; i < result.semantic_ids.size(); ++i) {
        const auto id = result.semantic_ids[i];
        check(static_cast<std::size_t>(id) < classes.size(),
              "generic semantic ID outside taxonomy");
        check(result.las_codes[i] == classes[id].las_code, "generic LAS projection mismatch");
    }
    const auto timings = timing_bytes(result.timings);
    check(frame_size(result.frame) + 2 * result.frame.points.size() + 4 + timings.size() <=
              limit(descriptor, "max_frame_bytes"),
          "generic output byte limit");
}
std::string encode_frame(const Descriptor &descriptor, const Frame &frame) {
    validate_frame(descriptor, frame);
    std::string bytes(header_bytes, '\0');
    bytes.reserve(frame_size(frame));
    encode_body(bytes, frame);
    return envelope(std::move(bytes), frame_kind);
}
Frame decode_frame(const Descriptor &descriptor, std::string_view bytes) {
    auto reader = payload(descriptor, bytes, frame_kind);
    auto result = decode_body(descriptor, reader, false);
    check(reader.left() == 0, "generic frame trailing bytes");
    return result;
}
std::string encode_classification(const Descriptor &descriptor, const Classification &result) {
    validate_classification(descriptor, result);
    const auto timings = timing_bytes(result.timings);
    std::string bytes(header_bytes, '\0');
    bytes.reserve(frame_size(result.frame) + 2 * result.frame.points.size() + 4 + timings.size());
    encode_body(bytes, result.frame);
    for (std::size_t i = 0; i < result.semantic_ids.size(); ++i) {
        put(bytes, result.semantic_ids[i], 1);
        put(bytes, result.las_codes[i], 1);
    }
    put(bytes, timings.size(), 4);
    bytes.append(timings);
    return envelope(std::move(bytes), classification_kind);
}
Classification decode_classification(const Descriptor &descriptor, std::string_view bytes) {
    auto reader = payload(descriptor, bytes, classification_kind);
    Classification result;
    result.frame = decode_body(descriptor, reader, true);
    const auto count = result.frame.points.size();
    result.semantic_ids.reserve(count);
    result.las_codes.reserve(count);
    for (std::size_t i = 0; i < count; ++i) {
        result.semantic_ids.push_back(static_cast<std::uint8_t>(reader.get(1)));
        result.las_codes.push_back(static_cast<std::uint8_t>(reader.get(1)));
    }
    const auto size = reader.get(4);
    check(size >= 2 && size <= timings_bound && size == reader.left(),
          "generic timings byte limit/trailing");
    const auto timings = reader.take(size);
    TimingGuard guard;
    check(Json::sax_parse(timings.begin(), timings.end(), &guard), "generic malformed timings");
    result.timings = Json::parse(timings);
    validate_classification(descriptor, result);
    return result;
}
} // namespace synrex::generic
