// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Tuomas Pietilä
// CPU-only generic map consumer; opaque model artifacts are never executed.
#include "synrex/generic_map.hpp"
#include <cmath>
#include <fstream>
#include <iostream>
#include <limits>
#include <unistd.h>

namespace {
using namespace synrex;
namespace g = synrex::generic;
unsigned checks{}, rejections{}, fixtures{}, target_queries{};
std::size_t full_snapshot_bytes{};
void check(bool okay, const std::string &message) {
    ++checks;
    if (!okay)
        throw std::runtime_error(message);
}
template <class F> void rejects(F operation, const std::string &diagnostic = {}) {
    try {
        operation();
    } catch (const std::exception &error) {
        check(diagnostic.empty() || std::string(error.what()).find(diagnostic) != std::string::npos,
              "unexpected rejection: " + std::string(error.what()));
        ++rejections;
        return;
    }
    throw std::runtime_error("unexpected acceptance case " + std::to_string(rejections + 1) + ": " +
                             diagnostic);
}
void write(const fs::path &path, const std::string &bytes) {
    std::ofstream stream(path, std::ios::binary);
    stream << bytes;
    stream.close();
    check(bool(stream), "fixture write");
}
struct Temporary {
    fs::path path;
    explicit Temporary(const fs::path &parent) {
        auto pattern = (parent / "generic-map-XXXXXX").string();
        const auto created = ::mkdtemp(pattern.data());
        check(created != nullptr, "temporary fixture");
        path = created;
    }
    ~Temporary() {
        std::error_code error;
        fs::remove_all(path, error);
    }
};
g::Descriptor descriptor(const fs::path &parent, Json contract, unsigned count,
                         const std::string &meaning = "Private meaning ", std::uint64_t epoch = 1) {
    const auto path = parent / ("package-" + std::to_string(++fixtures));
    fs::create_directory(path);
    Json classes = Json::array(), projection = Json::array();
    const unsigned legacy_projection[] = {1, 2, 200, 6, 9, 17, 201};
    for (unsigned i = 0; i < count; ++i) {
        const unsigned las = count == 7 ? legacy_projection[i] : 40 + i % 3;
        classes.push_back({{"semantic_id", i},
                           {"key", "private_" + std::to_string(i)},
                           {"name", meaning + std::to_string(i)},
                           {"las_code", las}});
        projection.push_back(las);
    }
    const Taxonomy taxonomy(Json{
        {"schema", "synrex.taxonomy.v1"},
        {"taxonomy_id", "example.reused.v1"},
        {"namespace", "example.map"},
        {"classes", classes},
        {"provenance", {{"source", "CPU fixture"}, {"author", "SynRex"}, {"license", "MIT"}}},
        {"unknown_semantic_id",
         nullptr}}.dump());
    contract["las_classes"] = projection;
    contract["output"]["shape"][1] = count;
    write(path / "model.onnx", "Opaque fixture, never executed");
    write(path / "MODEL_LICENSE.txt", "MIT");
    write(path / "taxonomy.json", taxonomy.raw_bytes());
    write(path / "manifest.json",
          Json{{"schema", "synrex.srxmodel.v3"},
               {"contract", contract},
               {"artifacts",
                {{"onnx", {{"path", "model.onnx"}, {"sha256", sha256(path / "model.onnx")}}},
                 {"license",
                  {{"path", "MODEL_LICENSE.txt"}, {"sha256", sha256(path / "MODEL_LICENSE.txt")}}},
                 {"taxonomy", {{"path", "taxonomy.json"}, {"sha256", taxonomy.sha256()}}}}},
               {"target", nullptr}}
              .dump());
    const g::CpuAdmittedPackage package(path);
    return g::Descriptor::from_package(package, {65536, 16777216, 30000}, std::string(32, 'a'),
                                       epoch);
}
g::Classification classification(const g::Descriptor &descriptor, std::uint64_t sequence,
                                 std::uint64_t timestamp, double x, std::uint8_t id) {
    g::Frame frame;
    frame.identity = {descriptor.sha256(), descriptor.spec().at("session_id"),
                      descriptor.spec().at("map_epoch")};
    frame.sensor_id = "sensor";
    frame.stream_id = "stream";
    frame.coordinate_frame = "projected";
    frame.crs = "LOCAL:test";
    frame.sequence = sequence;
    frame.frame_id = sequence;
    frame.frame_start_ns = timestamp;
    frame.frame_end_ns = timestamp;
    frame.points = {{UINT64_MAX - sequence, timestamp, {x, 0, 0}, {}, {}, {}}};
    return {
        std::move(frame), {id}, {descriptor.taxonomy().classes().at(id).las_code}, Json::object()};
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
        Json fixture;
        std::optional<g::Descriptor> expected;
        for (unsigned count : {4, 7, 256}) {
            const auto d = descriptor(temporary.path, base, count);
            g::SemanticMap map(d, 1, 4);
            const auto empty = map.snapshot_bytes(0);
            check(g::admit_snapshot(d, empty) == Json::parse(empty), "empty serializer admission");
            map.update(classification(d, 1, 1, -0.1, static_cast<std::uint8_t>(count - 1)));
            map.update(classification(d, 2, 2, 0.1, 0));
            const auto bytes = map.snapshot_bytes(UINT64_MAX);
            const auto admitted = g::admit_snapshot(d, bytes);
            check(admitted == Json::parse(bytes) && admitted.at("cells").size() == 2 &&
                      admitted.at("cells").at(0).at("semantic_id") == count - 1 &&
                      admitted.at("cells").at(0).at("classification_age_ns") == UINT64_MAX - 1,
                  "K4/nonFRACTAL7/K256 ID255 exact byte roundtrip");
            const auto reconstructed = g::Descriptor::parse(
                admitted.at("descriptor_bytes").get_ref<const std::string &>());
            check(reconstructed.bytes() == d.bytes() &&
                      reconstructed.taxonomy().classes().at(count - 1).las_code ==
                          d.taxonomy().classes().at(count - 1).las_code,
                  "independent bound descriptor and LAS projection reconstruction");
            auto bad = admitted;
            bad["cells"][0]["semantic_id"] = count;
            rejects([&] { g::admit_snapshot(d, bad.dump()); });
            if (count == 4) {
                fixture = admitted;
                expected = d;
            }
        }
        const auto &d = *expected;
        auto mutation = [&](auto change) {
            auto bad = fixture;
            change(bad);
            rejects([&] { g::admit_snapshot(d, bad.dump()); });
        };
        // Every required field is strict, including nested objects and cells.
        for (const auto &key :
             {"schema", "binding", "descriptor_bytes", "source_identity", "as_of_ns", "policy",
              "resolution_metres", "capacity", "mode", "cells"}) {
            mutation([&](Json &v) { v.erase(key); });
        }
        mutation([](Json &v) { v["extra"] = nullptr; });
        mutation([](Json &v) { v["identity"] = Json::array(); });
        for (const auto &key :
             {"descriptor_sha256", "session_id", "map_epoch", "taxonomy_sha256"}) {
            mutation([&](Json &v) { v["binding"].erase(key); });
            mutation([&](Json &v) { v["binding"][key] = nullptr; });
        }
        mutation([](Json &v) { v["binding"]["extra"] = 0; });
        mutation([](Json &v) { v["schema"] = "synrex.semantic_map.v1"; });
        mutation([](Json &v) { v["mode"] = "stream_output"; });
        mutation([](Json &v) { v["policy"] = "ttl"; });
        mutation([](Json &v) { v["binding"]["descriptor_sha256"] = std::string(64, '0'); });
        mutation([](Json &v) { v["binding"]["taxonomy_sha256"] = std::string(64, '0'); });
        mutation([](Json &v) { v["binding"]["session_id"] = std::string(32, 'b'); });
        mutation([](Json &v) { v["binding"]["map_epoch"] = 2; });
        mutation([](Json &v) { v["binding"]["map_epoch"] = 1.0; });
        mutation([](Json &v) {
            v["descriptor_bytes"] = v["descriptor_bytes"].get<std::string>() + " ";
        });
        mutation([](Json &v) {
            v["descriptor_bytes"] = Json::parse(v["descriptor_bytes"].get<std::string>());
        });
        // A changed meaning with identical taxonomy ID, K and projection is incompatible.
        const auto changed = descriptor(temporary.path, base, 4, "Changed meaning ");
        check(changed.spec().at("las_projection") == d.spec().at("las_projection") &&
                  changed.taxonomy().taxonomy_id() == d.taxonomy().taxonomy_id(),
              "same ID/K/projection adversary");
        rejects([&] { g::admit_snapshot(changed, fixture.dump()); });
        const auto epoch = descriptor(temporary.path, base, 4, "Private meaning ", 2);
        rejects([&] { g::admit_snapshot(epoch, fixture.dump()); });
        for (const auto &bad : {Json(nullptr), Json(true), Json("1"), Json(-1), Json(1.5), Json(0),
                                Json(65537), Json(UINT64_MAX)}) {
            mutation([&](Json &v) { v["capacity"] = bad; });
        }
        mutation([](Json &v) { v["capacity"] = 1; });
        for (const auto &bad : {Json(nullptr), Json(true), Json("1"), Json(0), Json(-1),
                                Json(0.000999), Json(1000.01), Json(1e300)}) {
            mutation([&](Json &v) { v["resolution_metres"] = bad; });
        }
        for (const auto &bad : {Json(nullptr), Json(true), Json("1"), Json(-1), Json(1.0)}) {
            mutation([&](Json &v) { v["as_of_ns"] = bad; });
            mutation([&](Json &v) { v["cells"][0]["timestamp_ns"] = bad; });
            mutation([&](Json &v) { v["cells"][0]["classification_age_ns"] = bad; });
            mutation([&](Json &v) { v["cells"][0]["semantic_id"] = bad; });
        }
        mutation([](Json &v) { v["as_of_ns"] = 0; });
        mutation([](Json &v) { v["cells"][0]["classification_age_ns"] = UINT64_MAX; });
        mutation([](Json &v) { v["cells"][0]["timestamp_ns"] = UINT64_MAX; });
        mutation([](Json &v) { v["cells"][0]["semantic_id"] = 256; });
        mutation([](Json &v) { v["cells"][0]["las_code"] = 40; });
        mutation([](Json &v) { v["cells"][0]["confidence"] = 0.9; });
        mutation([](Json &v) { v["cells"][0]["confidence_age_ns"] = 0; });
        for (const auto &key : {"key", "semantic_id", "timestamp_ns", "classification_age_ns",
                                "confidence", "confidence_age_ns"}) {
            mutation([&](Json &v) { v["cells"][0].erase(key); });
        }
        mutation([](Json &v) { v["cells"] = Json::object(); });
        mutation([](Json &v) { v["cells"] = Json::array(); });
        mutation([](Json &v) { v["cells"][0] = nullptr; });
        mutation([](Json &v) { v["cells"][0]["key"] = Json::array({0, 0}); });
        mutation([](Json &v) { v["cells"][0]["key"] = Json::array({0, 0, 0, 0}); });
        for (const auto &bad :
             {Json(nullptr), Json(true), Json("0"), Json(0.0), Json(UINT64_MAX), Json(INT64_MIN),
              Json(INT64_MAX), Json(1000000001), Json(-1000000001)}) {
            mutation([&](Json &v) { v["cells"][0]["key"][0] = bad; });
        }
        mutation([](Json &v) { v["cells"][1]["key"] = v["cells"][0]["key"]; });
        mutation([](Json &v) { std::swap(v["cells"][0], v["cells"][1]); });
        for (const auto &bad :
             {Json(nullptr), Json("sensor"), Json::array(), Json::array({1, 2, 3, 4}),
              Json::array({"a", "b", "c"}), Json::array({"a", "b", "c", "d", "e"})}) {
            mutation([&](Json &v) { v["source_identity"] = bad; });
        }
        for (const auto &bad : {std::string(), std::string(257, 'x'), std::string("x\ny"),
                                std::string("\0", 1), std::string("\x7f", 1), std::string("é")}) {
            mutation([&](Json &v) { v["source_identity"][0] = bad; });
        }
        // Escaped source identifiers exactly at the codec bound remain legal.
        {
            auto valid = fixture;
            valid["source_identity"] =
                std::array<std::string, 4>{std::string(256, '\\'), std::string(256, '"'),
                                           std::string(256, '~'), std::string(256, ' ')};
            check(g::admit_snapshot(d, valid.dump()) == valid, "escaped max source strings");
            valid["resolution_metres"] = 1000;
            check(g::admit_snapshot(d, valid.dump()) == valid, "supported integer resolution");
        }
        {
            g::SemanticMap zero(d, 1, 1);
            zero.update(classification(d, 1, 0, 0, 0));
            check(g::admit_snapshot(d, zero.snapshot_bytes(0))
                          .at("cells")[0]
                          .at("classification_age_ns") == 0,
                  "zero timestamp and age");
            g::SemanticMap maximum(d, 1, 1);
            maximum.update(classification(d, 1, UINT64_MAX, 0, 0));
            check(g::admit_snapshot(d, maximum.snapshot_bytes(UINT64_MAX))
                          .at("cells")[0]
                          .at("classification_age_ns") == 0,
                  "maximum timestamp zero age without overflow");
        }
        const auto bytes = fixture.dump();
        for (const auto &bad :
             {std::string(), std::string("null"), std::string("[]"), std::string("true"),
              bytes.substr(0, bytes.size() - 1), bytes + "x", bytes + "{}",
              bytes + std::string("\0", 1), std::string("{\"x\":1e999}"),
              std::string("{\"x\":NaN}"), std::string("{\"x\":\"\xff\"}"),
              std::string("{\"x\":\"\\ud800\"}"), std::string("{/*x*/}"),
              std::string("[[[[[0]]]]]"), std::string("{\"a\":{\"b\":{\"c\":{\"d\":{}}}}}")}) {
            rejects([&] { g::admit_snapshot(d, bad); });
        }
        auto raw_reject = [&](std::string bad) { rejects([&] { g::admit_snapshot(d, bad); }); };
        raw_reject("{\"schema\":0," + bytes.substr(1));
        raw_reject("{\"\\u0073chema\":0," + bytes.substr(1));
        auto nested = bytes;
        nested.insert(nested.find("\"binding\":{") + std::string("\"binding\":{").size(),
                      "\"map_epoch\":1,");
        raw_reject(nested);
        nested = bytes;
        nested.insert(nested.find("\"cells\":[{") + std::string("\"cells\":[{").size(),
                      "\"semantic_id\":0,");
        raw_reject(nested);
        nested = bytes;
        nested.replace(nested.find("\"as_of_ns\":18446744073709551615"),
                       std::string("\"as_of_ns\":18446744073709551615").size(),
                       "\"as_of_ns\":18446744073709551616");
        raw_reject(nested);
        check(g::admit_snapshot(
                  d, bytes + std::string(g::SemanticMap::max_snapshot_bytes - bytes.size(), ' ')) ==
                  fixture,
              "exact wire byte budget including JSON whitespace");
        raw_reject(std::string(g::SemanticMap::max_snapshot_bytes + 1, ' '));
        raw_reject(bytes + std::string("\0hidden suffix", 14));
        raw_reject("{\"x\":\"" + std::string(g::Descriptor::max_bytes + 1, 'a') + "\"}");
        nested = "[";
        for (std::size_t i = 0; i <= g::SemanticMap::max_cells; ++i)
            nested += i ? ",0" : "0";
        raw_reject(nested + "]");
        // Full capacity, exact64KiB descriptor, escaped source identities and
        // extreme supported coordinates. Independently reconstruct all cells.
        {
            const auto large = g::Descriptor::parse(
                d.bytes() + std::string(g::Descriptor::max_bytes - d.bytes().size(), ' '));
            g::SemanticMap map(large, 0.001, g::SemanticMap::max_cells);
            auto r = classification(large, UINT64_MAX, 0, 1e9, 3);
            r.frame.sensor_id = r.frame.stream_id = r.frame.coordinate_frame = r.frame.crs =
                std::string(256, '\\');
            r.frame.points.clear();
            r.semantic_ids.clear();
            r.las_codes.clear();
            for (std::size_t i = 0; i < g::SemanticMap::max_cells; ++i) {
                r.frame.points.push_back({UINT64_MAX - i, 0, {-1e9 + i, 1e9, -1e9}, {}, {}, {}});
                r.semantic_ids.push_back(3);
                r.las_codes.push_back(40);
            }
            map.update(r);
            const auto full = map.snapshot_bytes(UINT64_MAX);
            full_snapshot_bytes = full.size();
            const auto admitted = g::admit_snapshot(large, full);
            check(admitted.at("cells").size() == g::SemanticMap::max_cells &&
                      admitted.at("descriptor_bytes") == large.bytes(),
                  "full exact snapshot admission");
            const auto recovered =
                g::Descriptor::parse(admitted.at("descriptor_bytes").get<std::string>());
            check(recovered.bytes() == large.bytes() &&
                      recovered.taxonomy().sha256() == large.taxonomy().sha256(),
                  "full independent descriptor reconstruction");
            for (std::size_t i = 0; i < g::SemanticMap::max_cells; ++i) {
                const auto &cell = admitted.at("cells").at(i);
                check(
                    cell.at("key") ==
                            Json::array({static_cast<std::int64_t>(std::floor((-1e9 + i) / 0.001)),
                                         static_cast<std::int64_t>(std::floor(1e9 / 0.001)),
                                         static_cast<std::int64_t>(std::floor(-1e9 / 0.001))}) &&
                        cell.at("semantic_id") == 3 && cell.at("timestamp_ns") == 0 &&
                        cell.at("classification_age_ns") == UINT64_MAX &&
                        recovered.taxonomy().classes().at(3).las_code == 40,
                    "full independent key/semantic/projection/time reconstruction");
            }
        }
        check(target_queries == 0, "CPU-only: no GPU target query");
        std::cout << Json{{"status", "PASS"},
                          {"checks", checks},
                          {"rejections", rejections},
                          {"full_snapshot_bytes", full_snapshot_bytes},
                          {"gpu_calls", target_queries}}
                         .dump()
                  << '\n';
        return 0;
    } catch (const std::exception &error) {
        std::cerr << "FAIL: " << error.what() << '\n';
        return 1;
    }
}
