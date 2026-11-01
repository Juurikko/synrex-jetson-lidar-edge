// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Tuomas Pietilä
// CPU-only codec contract tests; opaque ONNX fixtures are never executed.
#include "synrex/generic_codec.hpp"
#include <cmath>
#include <cstring>
#include <fstream>
#include <iostream>
#include <limits>
#include <unistd.h>

namespace {
using namespace synrex;
using namespace synrex::generic;
unsigned checks{}, rejections{}, fixtures{}, target_queries{};
void check(bool value, const std::string &message) {
    ++checks;
    if (!value)
        throw std::runtime_error(message);
}
template <class F> void rejects(F operation, const std::string &text = "") {
    try {
        operation();
    } catch (const std::exception &error) {
        check(text.empty() || std::string(error.what()).find(text) != std::string::npos,
              "wrong rejection: " + std::string(error.what()) + "; expected " + text);
        ++rejections;
        return;
    }
    throw std::runtime_error("unexpected acceptance: " + text);
}
void write(const fs::path &path, const std::string &value) {
    std::ofstream stream(path, std::ios::binary);
    stream << value;
    stream.close();
    check(bool(stream), "fixture write");
}
struct Temporary {
    fs::path path;
    explicit Temporary(const fs::path &parent) {
        auto pattern = (parent / "codec-fixture-XXXXXX").string();
        const auto created = ::mkdtemp(pattern.data());
        check(created != nullptr, "temporary fixture");
        path = created;
    }
    ~Temporary() {
        std::error_code error;
        fs::remove_all(path, error);
    }
};
Json taxonomy(unsigned count) {
    Json classes = Json::array();
    const unsigned nonlegacy[] = {1, 2, 200, 6, 9, 17, 201};
    for (unsigned i = 0; i < count; ++i)
        classes.push_back({{"semantic_id", i},
                           {"key", "private_" + std::to_string(i)},
                           {"name", "Meaning " + std::to_string(i)},
                           {"las_code", count == 7 ? nonlegacy[i] : i % 3 + 40}});
    return {
        {"schema", "synrex.taxonomy.v1"},
        {"taxonomy_id", "example.reused.v1"},
        {"namespace", "example.cpu"},
        {"classes", classes},
        {"provenance", {{"source", "CPU fixture"}, {"author", "SynRex tests"}, {"license", "MIT"}}},
        {"unknown_semantic_id", nullptr}};
}
fs::path package(const fs::path &parent, const Json &base, const Json &tax) {
    const auto path = parent / ("package-" + std::to_string(++fixtures));
    fs::create_directory(path);
    auto contract = base;
    const Taxonomy admitted(tax.dump());
    Json projection = Json::array();
    for (const auto &entry : admitted.classes())
        projection.push_back(entry.las_code);
    contract["las_classes"] = projection;
    contract["output"]["shape"][1] = admitted.classes().size();
    write(path / "model.onnx", "Opaque CPU fixture: never executed");
    write(path / "MODEL_LICENSE.txt", "MIT CPU fixture");
    write(path / "taxonomy.json", tax.dump());
    write(path / "manifest.json",
          Json{{"schema", "synrex.srxmodel.v3"},
               {"contract", contract},
               {"artifacts",
                {{"onnx", {{"path", "model.onnx"}, {"sha256", sha256(path / "model.onnx")}}},
                 {"license",
                  {{"path", "MODEL_LICENSE.txt"}, {"sha256", sha256(path / "MODEL_LICENSE.txt")}}},
                 {"taxonomy", {{"path", "taxonomy.json"}, {"sha256", admitted.sha256()}}}}},
               {"target", nullptr}}
              .dump());
    return path;
}
Frame frame(const Descriptor &descriptor) {
    Frame value;
    value.identity = {descriptor.sha256(), descriptor.spec().at("session_id"),
                      descriptor.spec().at("map_epoch")};
    value.sensor_id = "sensor";
    value.stream_id = "stream";
    value.coordinate_frame = "local";
    value.crs = "LOCAL:test";
    value.sequence = UINT64_MAX;
    value.frame_id = UINT64_MAX - 1;
    value.frame_start_ns = 1700000000000000000ULL;
    value.frame_end_ns = value.frame_start_ns + 1000000000ULL;
    value.points = {
        {UINT64_MAX, value.frame_start_ns, {std::nextafter(1.0, 2.0), -0.0, 1e9}, {}, {}, {}},
        {0,
         value.frame_end_ns,
         {-1e9, std::nextafter(10000000.0, 10000001.0), 0.0},
         65535,
         15,
         15}};
    value.accepted_steady_ns = 900;
    value.submitted_steady_ns = 901;
    return value;
}
void equal_frame(const Frame &expected, const Frame &actual) {
    check(actual.identity.descriptor_sha256 == expected.identity.descriptor_sha256 &&
              actual.identity.session_id == expected.identity.session_id &&
              actual.identity.map_epoch == expected.identity.map_epoch,
          "identity preserved");
    check(actual.sensor_id == expected.sensor_id && actual.stream_id == expected.stream_id &&
              actual.coordinate_frame == expected.coordinate_frame && actual.crs == expected.crs &&
              actual.sequence == expected.sequence && actual.frame_id == expected.frame_id &&
              actual.frame_start_ns == expected.frame_start_ns &&
              actual.frame_end_ns == expected.frame_end_ns,
          "frame metadata preserved");
    check(actual.points.size() == expected.points.size(), "point count preserved");
    for (std::size_t i = 0; i < actual.points.size(); ++i) {
        const auto &a = actual.points[i], &b = expected.points[i];
        check(a.id == b.id && a.timestamp_ns == b.timestamp_ns && a.intensity == b.intensity &&
                  a.return_number == b.return_number && a.return_count == b.return_count &&
                  std::memcmp(a.xyz.data(), b.xyz.data(), 24) == 0,
              "point fields and exact binary64 bits preserved");
    }
    check(actual.accepted_steady_ns == 0 && actual.submitted_steady_ns == 0,
          "remote host clocks never trusted");
}
void length(std::string &bytes) {
    const auto size = bytes.size() - 12;
    for (unsigned i = 0; i < 4; ++i)
        bytes[8 + i] = static_cast<char>(size >> (8 * (3 - i)));
}
void u64(std::string &bytes, std::size_t offset, std::uint64_t value) {
    for (unsigned i = 0; i < 8; ++i)
        bytes[offset + i] = static_cast<char>(value >> (8 * (7 - i)));
}
} // namespace
namespace synrex {
Json target_info() {
    ++target_queries;
    throw std::runtime_error("UNREACHABLE GPU TARGET SENTINEL");
}
} // namespace synrex
int main(int argc, char **argv) {
    try {
        check(argc == 3, "source contract and scratch parent required");
        Temporary temporary(argv[2]);
        const auto base = read_json(argv[1]);
        const std::string session(32, 'a');
        const CpuAdmittedPackage p4(package(temporary.path, base, taxonomy(4)));
        const auto descriptor = Descriptor::from_package(p4, {}, session, 1);
        const auto input = frame(descriptor);
        check(Classification{}.timings.is_object() && Classification{}.timings.empty(),
              "default output timings is an empty object");
        const auto encoded = encode_frame(descriptor, input);
        equal_frame(input, decode_frame(descriptor, encoded));
        check(encoded.substr(0, 8) == std::string("SRX3\3\20\0\0", 8),
              "fixed binary-v3 frame header");
        const auto point_offset = encoded.size() - input.points.size() * 48;
        check(encoded.substr(point_offset + 16, 8) == std::string("\x3f\xf0\0\0\0\0\0\1", 8),
              "binary64 network byte order, no float narrowing");
        // Every proper prefix, including partial points and header, is rejected.
        for (std::size_t i = 0; i < encoded.size(); ++i)
            rejects([&] { decode_frame(descriptor, std::string_view(encoded).substr(0, i)); });
        // A lying sender can make the envelope internally consistent while
        // truncating the actual body. Bounds checks still precede point reserve.
        for (std::size_t i = 12; i < encoded.size(); ++i) {
            auto bytes = encoded.substr(0, i);
            length(bytes);
            rejects([&] { decode_frame(descriptor, bytes); });
        }
        for (const auto index : {0u, 4u, 5u, 6u, 7u}) {
            auto bytes = encoded;
            bytes[index] ^= 1;
            rejects([&] { decode_frame(descriptor, bytes); });
        }
        {
            auto bytes = encoded + "x";
            rejects([&] { decode_frame(descriptor, bytes); }, "payload length");
            length(bytes);
            rejects([&] { decode_frame(descriptor, bytes); }, "trailing");
        }
        for (auto offset : {12u, 76u, 115u}) {
            auto bytes = encoded;
            bytes[offset] ^= 1;
            rejects([&] { decode_frame(descriptor, bytes); }, "mismatch");
        }
        for (const auto count : {std::uint64_t{0}, std::uint64_t{3}, UINT64_MAX}) {
            auto bytes = encoded;
            u64(bytes, point_offset - 8, count);
            rejects([&] { decode_frame(descriptor, bytes); });
        }
        {
            auto bytes = encoded;
            u64(bytes, point_offset + 48, UINT64_MAX);
            rejects([&] { decode_frame(descriptor, bytes); }, "duplicate");
            bytes = encoded;
            u64(bytes, point_offset + 8, input.frame_start_ns - 1);
            rejects([&] { decode_frame(descriptor, bytes); }, "timestamp");
        }
        {
            auto bytes = encoded;
            bytes[116] = 1;
            bytes[117] = 1; // u16 length 257 rejects before string allocation
            rejects([&] { decode_frame(descriptor, bytes); }, "string byte limit");
        }
        for (auto offset : {40u, 41u, 43u, 44u, 46u}) {
            auto bytes = encoded;
            bytes[point_offset + offset] = 4;
            rejects([&] { decode_frame(descriptor, bytes); });
        }
        {
            auto bytes = encoded;
            u64(bytes, point_offset + 16, 0x7ff0000000000000ULL);
            rejects([&] { decode_frame(descriptor, bytes); }, "coordinate");
        }
        auto invalid = input;
        invalid.points[1].id = invalid.points[0].id;
        rejects([&] { encode_frame(descriptor, invalid); }, "duplicate");
        invalid = input;
        invalid.points[0].timestamp_ns--;
        rejects([&] { validate_frame(descriptor, invalid); }, "timestamp");
        invalid = input;
        invalid.frame_start_ns = invalid.frame_end_ns + 1;
        rejects([&] { validate_frame(descriptor, invalid); }, "span");
        invalid = input;
        invalid.frame_end_ns++;
        rejects([&] { validate_frame(descriptor, invalid); }, "span");
        for (const auto value : {std::numeric_limits<double>::quiet_NaN(),
                                 std::numeric_limits<double>::infinity(), 1000000001.0}) {
            invalid = input;
            invalid.points[0].xyz[0] = value;
            rejects([&] { validate_frame(descriptor, invalid); }, "coordinate");
        }
        invalid = input;
        invalid.points[0].return_number = 1;
        rejects([&] { validate_frame(descriptor, invalid); }, "paired");
        for (const auto &pair : {std::pair<unsigned, unsigned>{0, 1}, {2, 1}, {1, 16}}) {
            invalid = input;
            invalid.points[0].return_number = pair.first;
            invalid.points[0].return_count = pair.second;
            rejects([&] { validate_frame(descriptor, invalid); }, "return");
        }
        for (const auto &text :
             {std::string(), std::string(257, 'x'), std::string("a\0b", 3), std::string("\xff")}) {
            invalid = input;
            invalid.sensor_id = text;
            rejects([&] { validate_frame(descriptor, invalid); }, "string");
        }
        // Optional presence is distinct from a zero intensity value. Return-only,
        // intensity-only, absent and paired attributes all round trip exactly.
        for (const auto flags : {0u, 1u, 2u, 3u}) {
            auto optional = input;
            optional.points.resize(1);
            if (flags & 1)
                optional.points[0].intensity = 0;
            if (flags & 2) {
                optional.points[0].return_number = 1;
                optional.points[0].return_count = 1;
            }
            equal_frame(optional, decode_frame(descriptor, encode_frame(descriptor, optional)));
        }
        // Separate point and encoded-byte negotiation; exact boundary accepted.
        {
            const auto narrow = Descriptor::from_package(p4, {1, 1048576, 30000}, session, 1);
            rejects([&] { decode_frame(narrow, encoded); }, "mismatch");
            invalid = frame(narrow);
            rejects([&] { encode_frame(narrow, invalid); }, "point limit");
            const auto exact = Descriptor::from_package(p4, {2, encoded.size(), 30000}, session, 1);
            check(encode_frame(exact, frame(exact)).size() == encoded.size(),
                  "exact frame byte budget");
            const auto small =
                Descriptor::from_package(p4, {2, encoded.size() - 1, 30000}, session, 1);
            rejects([&] { encode_frame(small, frame(small)); }, "byte limit");
            rejects([&] { decode_frame(small, encoded); }, "byte limit");
        }
        // K carries no meaning: distinct explicit taxonomy projections, including id255.
        for (const auto count : {2u, 4u, 7u, 256u}) {
            const CpuAdmittedPackage p(package(temporary.path, base, taxonomy(count)));
            const auto d = Descriptor::from_package(p, {}, session, 1);
            Classification result;
            result.frame = frame(d);
            result.semantic_ids = {0, static_cast<std::uint8_t>(count - 1)};
            result.las_codes = {d.taxonomy().classes()[0].las_code,
                                d.taxonomy().classes()[count - 1].las_code};
            result.timings = {{"service_seconds", 0.125},
                              {"remote_measurement", 1e20},
                              {"window_started_steady_ns", UINT64_MAX}};
            const auto wire = encode_classification(d, result);
            const auto output = decode_classification(d, wire);
            equal_frame(result.frame, output.frame);
            check(output.semantic_ids == result.semantic_ids &&
                      output.las_codes == result.las_codes && output.timings == result.timings,
                  "taxonomy and timings round trip");
            rejects([&] { decode_frame(d, wire); }, "kind");
            rejects([&] { decode_classification(d, encode_frame(d, result.frame)); }, "kind");
            for (std::size_t i = 0; i < wire.size(); ++i)
                rejects([&] { decode_classification(d, std::string_view(wire).substr(0, i)); });
            for (std::size_t i = 12; i < wire.size(); ++i) {
                auto bytes = wire.substr(0, i);
                length(bytes);
                rejects([&] { decode_classification(d, bytes); });
            }
            auto bad = result;
            bad.semantic_ids.pop_back();
            rejects([&] { encode_classification(d, bad); }, "label count");
            bad = result;
            bad.las_codes[0] ^= 1;
            rejects([&] { encode_classification(d, bad); }, "projection");
            for (const auto &timings : {Json::array(), Json{{"x", -1}}, Json{{"x", "1"}},
                                        Json{{"x", nullptr}}, Json{{"x", Json::object()}},
                                        Json{{"x", std::numeric_limits<double>::infinity()}}}) {
                bad = result;
                bad.timings = timings;
                rejects([&] { encode_classification(d, bad); }, "timing");
            }
            bad = result;
            for (unsigned i = 0; i < 65; ++i)
                bad.timings["field" + std::to_string(i)] = 0;
            rejects([&] { encode_classification(d, bad); }, "timings object bound");
            bad = result;
            bad.timings = Json::object();
            for (unsigned i = 0; i < 64; ++i)
                bad.timings[std::string(253, 'x') + std::to_string(100 + i)] = UINT64_MAX;
            rejects([&] { encode_classification(d, bad); }, "timings byte limit");
            auto corrupt = wire;
            const auto label_offset = encode_frame(d, result.frame).size();
            corrupt[label_offset + 1] ^= 1;
            rejects([&] { decode_classification(d, corrupt); }, "projection");
            if (count < 256) {
                bad = result;
                bad.semantic_ids[0] = 255;
                rejects([&] { encode_classification(d, bad); }, "semantic ID");
                corrupt = wire;
                corrupt[label_offset] = static_cast<char>(255);
                rejects([&] { decode_classification(d, corrupt); }, "semantic ID");
            }
            const auto empty_timing_wire = encode_classification(
                d, {result.frame, result.semantic_ids, result.las_codes, Json::object()});
            for (const auto malformed :
                 {"[]", "{\"x\":-1}", "{\"x\":{}}", "{\"x\":true}", "{\"x\":1,\"x\":2}",
                  "{\"x\":1,\"\\u0078\":2}", "{\"x\":1e999}"}) {
                auto bytes = empty_timing_wire.substr(0, empty_timing_wire.size() - 6);
                const std::string timing(malformed);
                bytes.append(4, '\0');
                for (unsigned i = 0; i < 4; ++i)
                    bytes[bytes.size() - 4 + i] = static_cast<char>(timing.size() >> (8 * (3 - i)));
                bytes.append(timing);
                length(bytes);
                rejects([&] { decode_classification(d, bytes); });
            }
            const auto exact = Descriptor::from_package(p, {2, wire.size(), 30000}, session, 1);
            bad = result;
            bad.frame = frame(exact);
            check(encode_classification(exact, bad).size() == wire.size(),
                  "exact output byte budget");
            const auto small = Descriptor::from_package(p, {2, wire.size() - 1, 30000}, session, 1);
            bad.frame = frame(small);
            rejects([&] { encode_classification(small, bad); }, "output byte limit");
        }
        // Same K/id/projection with changed meaning/order has a distinct descriptor.
        for (const auto reorder : {false, true}) {
            auto tax = taxonomy(4);
            if (reorder) {
                std::swap(tax["classes"][0]["name"], tax["classes"][1]["name"]);
                std::swap(tax["classes"][0]["key"], tax["classes"][1]["key"]);
            } else
                tax["classes"][0]["name"] = "Changed meaning";
            const CpuAdmittedPackage other(package(temporary.path, base, tax));
            const auto d = Descriptor::from_package(other, {}, session, 1);
            check(d.taxonomy().taxonomy_id() == descriptor.taxonomy().taxonomy_id() &&
                      d.spec().at("las_projection") == descriptor.spec().at("las_projection"),
                  "same K/id/LAS fixture");
            rejects([&] { decode_frame(d, encoded); }, "mismatch");
        }
        check(target_queries == 0, "no GPU target calls");
        std::cout
            << Json{{"status", "PASS_CPU_GENERIC_V3_CODEC"}, {"checks", checks},
                    {"expected_rejections", rejections},     {"gpu_target_calls", target_queries},
                    {"readiness_authorized", false},         {"service_qualified", false},
                    {"class_counts", {2, 4, 7, 256}},        {"semantic_id_255", true}}
                   .dump()
            << '\n';
        return 0;
    } catch (const std::exception &error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
