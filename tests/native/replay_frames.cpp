// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Tuomas Pietilä
// CPU-only input/frame/coverage contracts. No model or engine is executed.
#include "synrex/replay_frames.hpp"
#include <cmath>
#include <cstring>
#include <fstream>
#include <iostream>
#include <unistd.h>

namespace {
using namespace synrex;
using namespace synrex::generic;
using namespace synrex::replay;
unsigned checks{}, rejects_count{}, target_queries{};
void check(bool okay, const std::string &why) {
    ++checks;
    if (!okay)
        throw std::runtime_error(why);
}
template <class F> void rejects(F operation) {
    try {
        operation();
    } catch (const std::exception &) {
        ++rejects_count;
        return;
    }
    throw std::runtime_error("unexpected acceptance");
}
void write(const fs::path &path, const std::string &bytes) {
    std::ofstream file(path, std::ios::binary);
    file << bytes;
    file.close();
    check(bool(file), "fixture write");
}
struct Temporary {
    fs::path path;
    explicit Temporary(const fs::path &parent) {
        auto pattern = (parent / "replay-frames-XXXXXX").string();
        const auto created = ::mkdtemp(pattern.data());
        check(created != nullptr, "temporary fixture");
        path = created;
    }
    ~Temporary() {
        std::error_code ignored;
        fs::remove_all(path, ignored);
    }
};
fs::path package(const fs::path &parent, Json contract, unsigned k) {
    const auto path = parent / ("package-" + std::to_string(k));
    fs::create_directory(path);
    Json classes = Json::array(), projection = Json::array();
    for (unsigned i = 0; i < k; ++i) {
        const auto las = (i * 17 + 41) % 256;
        classes.push_back({{"semantic_id", i},
                           {"key", "test_" + std::to_string(i)},
                           {"name", "Nonlegacy meaning " + std::to_string(i)},
                           {"las_code", las}});
        projection.push_back(las);
    }
    const Taxonomy tax(Json{
        {"schema", "synrex.taxonomy.v1"},
        {"taxonomy_id", "example.replay"},
        {"namespace", "example.cpu"},
        {"classes", classes},
        {"provenance", {{"source", "CPU fixture"}, {"author", "SynRex tests"}, {"license", "MIT"}}},
        {"unknown_semantic_id",
         nullptr}}.dump());
    contract["las_classes"] = projection;
    contract["output"]["shape"][1] = k;
    write(path / "model.onnx", "CPU fixture only: never executed");
    write(path / "MODEL_LICENSE.txt", "MIT CPU fixture");
    write(path / "taxonomy.json", tax.raw_bytes());
    write(path / "manifest.json",
          Json{{"schema", "synrex.srxmodel.v3"},
               {"contract", contract},
               {"artifacts",
                {{"onnx", {{"path", "model.onnx"}, {"sha256", sha256(path / "model.onnx")}}},
                 {"license",
                  {{"path", "MODEL_LICENSE.txt"}, {"sha256", sha256(path / "MODEL_LICENSE.txt")}}},
                 {"taxonomy", {{"path", "taxonomy.json"}, {"sha256", tax.sha256()}}}}},
               {"target", nullptr}}
              .dump());
    return path;
}
fs::path manifest(const fs::path &path, std::uint64_t first_count = 5) {
    write(path / "a.laz", "metadata-only source a");
    write(path / "b.laz", "metadata-only source b");
    Json sources = Json::array();
    for (const auto &entry :
         {std::make_pair("a.laz", first_count), std::make_pair("b.laz", std::uint64_t(4))})
        sources.push_back({{"path", entry.first},
                           {"sha256", sha256(path / entry.first)},
                           {"records", entry.second},
                           {"coordinate_frame", entry.first},
                           {"crs", "LOCAL:fixture"}});
    const auto file = path / "replay.json";
    write(file, Json{{"schema", "synrex.distinct_replay.v1"},
                     {"workload", "file_order_synthetic"},
                     {"boundary_policy", "reset_each_file"},
                     {"sources", sources}}
                    .dump());
    return file;
}
Window window(const Catalog &catalog, std::size_t source, std::uint64_t begin, unsigned count) {
    Window value;
    value.source_index = source;
    for (unsigned i = 0; i < count; ++i) {
        Observation point;
        point.wire_id = catalog.wire_id(source, begin + i);
        point.original_record_index = begin + i;
        point.integer_xyz = {-1, 2, 3};
        point.xyz = {std::nextafter(10000000.0 + i, 10000001.0 + i), -0.0, -42.123456789};
        point.intensity = 65535 - i;
        point.return_number = 2;
        point.return_count = 3;
        point.gps_time_bits = UINT64_MAX - i; // Deliberately invalid clock interpretation.
        value.observations.push_back(point);
    }
    return value;
}
Window source_window(const Catalog &catalog, std::uint64_t begin, unsigned count) {
    auto result = window(catalog, 0, begin, count);
    auto &h = result.source_header;
    auto put = [&](std::size_t offset, std::uint64_t bits, unsigned width) {
        for (unsigned i = 0; i < width; ++i)
            h.at(offset + i) = bits >> (8 * i);
    };
    std::memcpy(h.data(), "LASF", 4);
    h[24] = 1;
    h[25] = 4;
    h[104] = 128 + 6;
    put(94, 375, 2);
    put(105, 30, 2);
    put(247, catalog.sources()[0].records, 8);
    const std::array<double, 3> scales{0.0001, 0.00000001, 0.5};
    const std::array<double, 3> offsets{10000000.0000001, -0.0, -42.0};
    for (unsigned axis = 0; axis < 3; ++axis) {
        std::uint64_t bits;
        std::memcpy(&bits, &scales[axis], 8);
        put(131 + axis * 8, bits, 8);
        std::memcpy(&bits, &offsets[axis], 8);
        put(155 + axis * 8, bits, 8);
    }
    for (unsigned i = 0; i < count; ++i) {
        auto &point = result.observations[i];
        point.integer_xyz = {INT32_MIN + static_cast<int>(i), INT32_MAX, -1};
        for (unsigned axis = 0; axis < 3; ++axis)
            point.xyz[axis] = point.integer_xyz[axis] * scales[axis] + offsets[axis];
        point.return_number = point.return_count = 0;
        point.gps_time_bits = i == 0   ? 0x8000000000000000ULL
                              : i == 1 ? 0x7ff8000000000001ULL
                                       : UINT64_MAX;
    }
    return result;
}
} // namespace
namespace synrex {
Json target_info() {
    ++target_queries;
    throw std::runtime_error("GPU query forbidden in CPU fixture");
}
} // namespace synrex
int main(int argc, char **argv) {
    try {
        check(argc == 3, "dense contract and output build root required");
        Temporary tmp(argv[2]);
        Catalog catalog(manifest(tmp.path));
        const CpuAdmittedPackage pkg(package(tmp.path, read_json(argv[1]), 4));
        const auto descriptor = Descriptor::from_package(pkg, {}, std::string(32, 'd'), 1);
        const FrameOptions options{3, 7};
        const auto w0 = window(catalog, 0, 0, 3), w1 = window(catalog, 0, 3, 2);
        const auto f0 = make_frame(catalog, descriptor, options, w0, 0);
        const auto f1 = make_frame(catalog, descriptor, options, w1, 1);
        check(f0.frame_start_ns == 0 && f0.frame_end_ns == 285714285 &&
                  f0.points[1].timestamp_ns == 142857142 && f1.frame_end_ns == 571428571,
              "exact rational synthetic time and finite partial frame");
        check(f0.stream_id == catalog.sources()[0].sha256 && f0.coordinate_frame == "a.laz" &&
                  f0.sensor_id == "file-order-replay" && f0.points[0].id == 0,
              "source SHA identity and exact metadata");
        Coverage exact(catalog, descriptor, options, 0);
        check(exact.receipt()["state"] == "RUNNING", "never premature success");
        exact.accept(w0, decode_frame(descriptor, encode_frame(descriptor, f0)));
        exact.accept(w1, f1);
        exact.source_eof();
        exact.finish();
        check(exact.receipt()["state"] == "SUCCEEDED" && exact.receipt()["accepted_points"] == 5 &&
                  exact.receipt()["windows"] == 2 &&
                  exact.receipt()["exact_attribute_points"] == 5 &&
                  exact.receipt()["synthetic_duration_ns"] == 714285714,
              "finite exact attribute coverage");
        const CpuAdmittedPackage pkg7(package(tmp.path, read_json(argv[1]), 7));
        const auto d7 = Descriptor::from_package(pkg7, {}, std::string(32, 'd'), 2);
        auto second0 = make_frame(catalog, d7, options, window(catalog, 1, 0, 3), 0);
        auto second1 = make_frame(catalog, d7, options, window(catalog, 1, 3, 1), 1);
        Coverage second(catalog, d7, options, 1);
        second.accept_frame(second0);
        second.accept_frame(second1);
        second.source_eof();
        second.finish();
        check(second0.points[0].id == 5 && second0.points[0].timestamp_ns == 714285714 &&
                  second.receipt()["identity_only_points"] == 4 &&
                  second.receipt()["exact_attribute_points"] == 0,
              "source reset keeps global IDs/time; nonlegacy descriptor and identity-only claim");
        auto poison = [&](auto mutation, bool exact_attributes = false) {
            Coverage coverage(catalog, descriptor, options, 0);
            auto changed = f0;
            mutation(changed);
            rejects([&] {
                if (exact_attributes)
                    coverage.accept(w0, changed);
                else
                    coverage.accept_frame(changed);
            });
            check(coverage.receipt()["state"] == "FAILED", "bad frame permanently fails coverage");
            rejects([&] { coverage.accept(w0, f0); });
            rejects([&] { coverage.source_eof(); });
            rejects([&] { coverage.finish(); });
        };
        poison([](auto &f) { f.sensor_id = "wrong"; });
        poison([](auto &f) { f.stream_id = std::string(64, '0'); });
        poison([](auto &f) { f.coordinate_frame = "other"; });
        poison([](auto &f) { f.crs = "other"; });
        poison([](auto &f) { f.identity.descriptor_sha256 = std::string(64, '0'); });
        poison([](auto &f) { ++f.identity.map_epoch; });
        poison([](auto &f) { ++f.sequence; });
        poison([](auto &f) { ++f.frame_id; });
        poison([](auto &f) { --f.frame_end_ns; });
        poison([](auto &f) { f.points[0].id = 6; });
        poison([](auto &f) { ++f.points[0].timestamp_ns; });
        poison([](auto &f) { std::swap(f.points[0], f.points[1]); });
        poison([](auto &f) { f.points.pop_back(); });
        poison([](auto &f) { f.points[0].xyz[1] = 0.0; }, true); // -0 bits matter.
        poison([](auto &f) { f.points[0].xyz[0] = std::nextafter(f.points[0].xyz[0], 1e9); }, true);
        poison([](auto &f) { f.points[0].intensity.reset(); }, true);
        poison([](auto &f) { f.points[0].return_number = 1; }, true);
        poison([](auto &f) { f.points[0].return_count = 4; }, true);
        for (unsigned scenario = 0; scenario < 5; ++scenario) {
            Coverage coverage(catalog, descriptor, options, 0);
            if (scenario == 0)
                rejects([&] { coverage.accept_frame(f1); }); // Missing first.
            if (scenario == 1) {
                coverage.accept_frame(f0);
                rejects([&] { coverage.accept_frame(f0); });
            }
            if (scenario == 2) {
                coverage.accept_frame(f0);
                rejects([&] { coverage.source_eof(); });
            }
            if (scenario == 3)
                rejects([&] { coverage.finish(); });
            if (scenario == 4) {
                auto wrong = w0;
                wrong.source_index = 1;
                rejects([&] { coverage.accept(wrong, f0); });
            }
            check(coverage.receipt()["state"] == "FAILED", "terminal/fault fail closed");
        }
        Coverage late(catalog, descriptor, options, 0);
        late.accept_frame(f0);
        late.accept_frame(f1);
        late.source_eof();
        rejects([&] { late.accept_frame(f1); });
        check(late.receipt()["state"] == "FAILED", "extra frame at EOF fails");
        rejects([&] { make_frame(catalog, descriptor, options, window(catalog, 0, 0, 2), 0); });
        rejects([&] { make_frame(catalog, descriptor, options, window(catalog, 0, 1, 3), 0); });
        auto wrong = w0;
        ++wrong.observations[1].wire_id;
        rejects([&] { make_frame(catalog, descriptor, options, wrong, 0); });
        wrong = w0;
        ++wrong.observations[1].original_record_index;
        rejects([&] { make_frame(catalog, descriptor, options, wrong, 0); });
        for (const auto bad : {FrameOptions{0, 7}, FrameOptions{65537, 65537}, FrameOptions{3, 0},
                               FrameOptions{3, 2}})
            rejects([&] { bad.validate(descriptor); });
        rejects([&] { Coverage invalid(catalog, descriptor, options, 2); });
        rejects([&] { Coverage invalid(catalog, descriptor, options, 0, UINT64_MAX); });
        check(synthetic_time_ns(UINT64_MAX, UINT64_MAX) == 1000000000 &&
                  synthetic_time_ns(UINT64_MAX - 1, UINT64_MAX) == 999999999,
              "wide intermediate rational division");
        rejects([&] { synthetic_time_ns(UINT64_MAX, 1); });
        rejects([&] { synthetic_time_ns(0, 0); });
        const auto one_second = make_frame(catalog, descriptor, {3, 3}, w0, 0);
        check(one_second.frame_end_ns == 666666666, "at most one second frame");
        auto gps = w0;
        for (auto &p : gps.observations)
            p.gps_time_bits = 0;
        check(encode_frame(descriptor, make_frame(catalog, descriptor, options, gps, 0)) ==
                  encode_frame(descriptor, f0),
              "GPS is never used as synthetic or host clock");
        Catalog huge(manifest(tmp.path, UINT64_MAX - 4));
        rejects([&] { Coverage invalid(huge, descriptor, options, 0); });
        const FrameOptions tiny_time{4, UINT64_MAX};
        auto last = window(huge, 1, 0, 4);
        const auto last_frame = make_frame(huge, descriptor, tiny_time, last, UINT64_MAX);
        Coverage last_coverage(huge, descriptor, tiny_time, 1, UINT64_MAX);
        last_coverage.accept(last, last_frame);
        last_coverage.source_eof();
        last_coverage.finish();
        check(last_frame.points.back().id == UINT64_MAX - 1 &&
                  last_frame.frame_end_ns == 999999999 &&
                  last_coverage.receipt()["state"] == "SUCCEEDED",
              "max global row and last permitted sequence preserve finite arithmetic");
        const auto source_descriptor = Descriptor::from_package(pkg, {}, std::string(32, 'd'), 1,
                                                                WireProtocol::binary_source_v1);
        check(source_descriptor.sha256() != descriptor.sha256() &&
                  source_descriptor.source_fidelity(),
              "new descriptor explicitly binds codec");
        auto wrong_spec = source_descriptor.spec();
        wrong_spec["source_attributes_schema"] = "unknown";
        rejects([&] { Descriptor::parse(wrong_spec.dump()); });
        wrong_spec = source_descriptor.spec();
        wrong_spec["new_attribute"] = 1;
        rejects([&] { Descriptor::parse(wrong_spec.dump()); });
        wrong_spec = source_descriptor.spec();
        wrong_spec["binary_wire_version"] = 2;
        rejects([&] { Descriptor::parse(wrong_spec.dump()); });
        rejects([&] {
            Descriptor::from_package(pkg, {}, std::string(32, 'd'), 1,
                                     static_cast<WireProtocol>(99));
        });
        const auto sw0 = source_window(catalog, 0, 3), sw1 = source_window(catalog, 3, 2);
        const auto sf0 = make_frame(catalog, source_descriptor, options, sw0, 0);
        const auto sf1 = make_frame(catalog, source_descriptor, options, sw1, 1);
        auto source_bytes = encode_frame(source_descriptor, sf0);
        check(source_bytes.substr(0, 5) == std::string("SRXS\1", 5) &&
                  source_bytes.size() == encode_frame(descriptor, f0).size() + 391 + 28 * 3,
              "fixed bounded source header and raw row records");
        const auto decoded_source = decode_frame(source_descriptor, source_bytes);
        check(decoded_source.source_header->las14 == sw0.source_header &&
                  decoded_source.source_rows[0].integer_xyz == sw0.observations[0].integer_xyz &&
                  decoded_source.source_rows[1].gps_time_bits == 0x7ff8000000000001ULL &&
                  decoded_source.points[0].return_number == 0 &&
                  decoded_source.points[0].return_count == 0,
              "raw int extremes, signed zero GPS, NaN GPS payload and paired zero preserved");
        Coverage source_exact(catalog, source_descriptor, options, 0);
        source_exact.accept(sw0, decoded_source);
        source_exact.accept(sw1, sf1);
        source_exact.source_eof();
        source_exact.finish();
        check(source_exact.receipt()["raw_source_attribute_points"] == 5,
              "independent source bit coverage");
        Classification classified{sf0, {0, 1, 2}, {41, 58, 75}, Json::object()};
        const auto output = decode_classification(
            source_descriptor, encode_classification(source_descriptor, classified));
        Coverage output_exact(catalog, source_descriptor, options, 0);
        output_exact.accept(sw0, output.frame);
        auto source_poison = [&](auto mutation) {
            auto changed = sf0;
            mutation(changed);
            Coverage coverage(catalog, source_descriptor, options, 0);
            rejects([&] { coverage.accept(sw0, changed); });
            check(coverage.receipt()["state"] == "FAILED",
                  "raw attribute corruption poisons coverage");
        };
        source_poison([](auto &f) { f.source_header->las14[26] ^= 1; });  // provenance header
        source_poison([](auto &f) { f.source_header->las14[131] ^= 1; }); // exact scale
        source_poison([](auto &f) { f.source_header->las14[155] ^= 1; }); // exact offset
        source_poison([](auto &f) { ++f.source_header->records; });
        source_poison([](auto &f) { ++f.source_header->wire_begin; });
        source_poison([](auto &f) { ++f.source_rows[0].integer_xyz[0]; });
        source_poison([](auto &f) { f.source_rows[1].gps_time_bits ^= 1; });
        source_poison([](auto &f) { ++f.source_rows[1].original_record_index; });
        source_poison([](auto &f) { f.points[0].return_count = 1; });
        source_poison([](auto &f) { f.points[0].return_number = 1; });
        source_poison([](auto &f) { f.source_header->las14[105] = 31; }); // unknown extra attrs
        source_poison([](auto &f) { f.source_rows.pop_back(); });
        source_poison([](auto &f) { f.source_header.reset(); });
        Coverage header_change(catalog, source_descriptor, options, 0);
        header_change.accept_frame(sf0);
        auto changed_header = sf1;
        changed_header.source_header->las14[26] ^= 1;
        rejects([&] { header_change.accept_frame(changed_header); });
        auto legacy_zero = f0;
        legacy_zero.points[0].return_number = legacy_zero.points[0].return_count = 0;
        rejects([&] { encode_frame(descriptor, legacy_zero); });
        auto legacy_attrs = f0;
        legacy_attrs.source_header = sf0.source_header;
        legacy_attrs.source_rows = sf0.source_rows;
        rejects([&] { encode_frame(descriptor, legacy_attrs); });
        rejects([&] { decode_frame(descriptor, source_bytes); });
        rejects([&] { decode_frame(source_descriptor, encode_frame(descriptor, f0)); });
        for (const auto offset : {4U, 6U}) {
            auto corrupt = source_bytes;
            corrupt[offset] ^= 1;
            rejects([&] { decode_frame(source_descriptor, corrupt); });
        }
        auto trailing = source_bytes + "x";
        rejects([&] { decode_frame(source_descriptor, trailing); });
        for (std::size_t length : {std::size_t(0), std::size_t(11), source_bytes.size() - 1})
            rejects([&] { decode_frame(source_descriptor, source_bytes.substr(0, length)); });
        const auto tight =
            Descriptor::from_package(pkg, {3, source_bytes.size() - 1, 30000}, std::string(32, 'd'),
                                     1, WireProtocol::binary_source_v1);
        auto oversized = sf0;
        oversized.identity.descriptor_sha256 = tight.sha256();
        rejects([&] { encode_frame(tight, oversized); });
        check(target_queries == 0, "no GPU target query");
        std::cout << "replay frames: PASS checks=" << checks << " rejections=" << rejects_count
                  << '\n';
        return 0;
    } catch (const std::exception &error) {
        std::cerr << "replay frames: FAIL: " << error.what() << '\n';
        return 1;
    }
}
