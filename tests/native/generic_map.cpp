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
    throw std::runtime_error("unexpected acceptance: " + diagnostic);
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
        for (unsigned count : {4, 7, 256}) {
            const auto d = descriptor(temporary.path, base, count);
            g::SemanticMap map(d, 1, 2);
            check(map.snapshot(0).at("cells").empty(), "empty generic snapshot");
            auto r = classification(d, 1, 1, 0.1, static_cast<std::uint8_t>(count - 1));
            const auto delta = map.update(r);
            check(delta.at("deltas").size() == 1 &&
                      delta.at("deltas").at(0).at("semantic_id") == count - 1 &&
                      delta.at("binding").at("taxonomy_sha256") == d.taxonomy().sha256(),
                  "K4/nonFRACTAL7/K256 semantic and taxonomy identity");
            auto snapshot = Json::parse(map.snapshot_bytes(UINT64_MAX));
            const auto reconstructed =
                g::Descriptor::parse(snapshot.at("descriptor_bytes").get<std::string>());
            check(reconstructed.bytes() == d.bytes() && reconstructed.sha256() == d.sha256() &&
                      reconstructed.taxonomy().raw_bytes() == d.taxonomy().raw_bytes() &&
                      !snapshot.contains("identity") &&
                      snapshot.at("schema") == "synrex.generic_map_snapshot.v1",
                  "independent exact descriptor/taxonomy reconstruction and new schema");
            check(snapshot.at("cells").at(0).at("classification_age_ns") == UINT64_MAX - 1,
                  "uint64 age preserved");
            map.update(classification(d, 2, 2, 1.1, 0));
            // Equal-time acceptance refreshes revision even for unchanged labels.
            check(map.update(classification(d, 3, 2, 0.1, static_cast<std::uint8_t>(count - 1)))
                          .at("deltas")
                          .size() == 1,
                  "timestamp change emits delta");
            check(map.update(classification(d, 4, 2, 0.1, static_cast<std::uint8_t>(count - 1)))
                      .at("deltas")
                      .empty(),
                  "equal-time unchanged label refreshes without delta");
            const auto evicted = map.update(classification(d, 5, 2, 2.1, 1));
            check(evicted.at("deltas").at(0).at("evicted") == Json::array({1, 0, 0}),
                  "revision victim matches retained policy");
            check(map.size() == 2 && map.counters().observations == 5 &&
                      map.counters().removals == 1,
                  "bounded distinct cell accounting");
            const auto before = map.counters().observations;
            r = classification(d, 6, 3, 0, 0);
            r.las_codes[0] ^= 1;
            rejects([&] { map.update(r); }, "projection");
            check(map.failed() && map.counters().observations == before,
                  "invalid projection rejected before mutation");
            rejects([&] { map.snapshot(3); }, "permanently");
        }
        const auto d = descriptor(temporary.path, base, 4);
        rejects([&] { g::SemanticMap map(d, 1, 0); }, "capacity");
        rejects([&] { g::SemanticMap map(d, 1, 65537); }, "capacity");
        rejects([&] { g::SemanticMap map(d, 0, 1); }, "budget");
        // A scalar independent revision oracle checks indexed and scan policies.
        for (auto policy : {streaming::MapPolicy::reference, streaming::MapPolicy::indexed}) {
            g::SemanticMap map(d, 1, 7, policy);
            struct Cell {
                unsigned id;
                std::uint64_t time, revision;
            };
            std::map<int, Cell> oracle;
            std::uint64_t revision{};
            for (std::uint64_t i = 1; i <= 250; ++i) {
                const int key = static_cast<int>((i * 17) % 19) - 9;
                const auto id = static_cast<std::uint8_t>(i % 4);
                auto found = oracle.find(key);
                if (found == oracle.end() && oracle.size() == 7) {
                    const auto victim = std::min_element(
                        oracle.begin(), oracle.end(), [](const auto &a, const auto &b) {
                            return a.second.revision < b.second.revision;
                        });
                    oracle.erase(victim);
                }
                oracle[key] = {id, i, ++revision};
                map.update(classification(d, i, i, key + 0.25, id));
                const auto cells = map.snapshot(i).at("cells");
                check(cells.size() == oracle.size(), "scalar oracle cell count");
                std::size_t n{};
                for (const auto &[expected, cell] : oracle) {
                    check(cells[n].at("key") == Json::array({expected, 0, 0}) &&
                              cells[n].at("semantic_id") == cell.id &&
                              cells[n].at("timestamp_ns") == cell.time,
                          "independent scalar revision oracle");
                    ++n;
                }
            }
        }
        // Same taxonomy ID/K/projection with changed meaning, new epoch or model
        // descriptor must never inherit an existing map. Each input fails closed.
        const auto changed = descriptor(temporary.path, base, 4, "Changed meaning ");
        check(changed.taxonomy().taxonomy_id() == d.taxonomy().taxonomy_id() &&
                  changed.spec().at("las_projection") == d.spec().at("las_projection") &&
                  changed.taxonomy().sha256() != d.taxonomy().sha256(),
              "adversarial same-ID same-K same-projection fixture");
        const auto epoch = descriptor(temporary.path, base, 4, "Private meaning ", 2);
        for (const auto &different : {changed, epoch}) {
            g::SemanticMap map(d, 1, 4);
            map.update(classification(d, 1, 1, 0, 0));
            rejects([&] { map.update(classification(different, 2, 2, 1, 0)); }, "mismatch");
            check(map.failed() && map.size() == 1, "identity failed before cell mutation");
        }
        for (unsigned failure = 0; failure < 4; ++failure) {
            g::SemanticMap map(d, 1, 4);
            map.update(classification(d, 1, 1, 0, 0));
            auto r = classification(d, 2, 2, 1, 0);
            if (failure == 0)
                r.frame.sensor_id = "different";
            if (failure == 1)
                r.frame.sequence = 1;
            if (failure == 2)
                r.frame.frame_start_ns = r.frame.frame_end_ns = 0,
                r.frame.points[0].timestamp_ns = 0;
            if (failure == 3)
                r.semantic_ids.clear();
            rejects([&] { map.update(r); });
            check(map.failed() && map.size() == 1, "source/order/coverage fails before mutation");
        }
        {
            g::SemanticMap map(d, 1, 4);
            map.update(classification(d, 1, 1, 0, 0));
            rejects([&] { map.snapshot(0); }, "negative");
            rejects([&] { map.update(classification(d, 2, 2, 1, 0)); }, "permanently");
        }
        // Full capacity, exact64KiB descriptor, all escaped source identities,
        // extreme keys and uint64 age. Reconstruct every cell independently.
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
            const auto bytes = map.snapshot_bytes(UINT64_MAX);
            full_snapshot_bytes = bytes.size();
            check(bytes.size() <= g::SemanticMap::max_snapshot_bytes &&
                      bytes.size() <=
                          g::SemanticMap::snapshot_base_bytes +
                              g::SemanticMap::snapshot_cell_bytes * g::SemanticMap::max_cells,
                  "full actual encoded snapshot fits admitted proof and wire budget");
            const auto reconstructed = Json::parse(bytes);
            check(reconstructed.at("cells").size() == g::SemanticMap::max_cells &&
                      reconstructed.at("descriptor_bytes") == large.bytes(),
                  "full snapshot preserves capacity and exact max descriptor");
            for (std::size_t i = 0; i < g::SemanticMap::max_cells; ++i) {
                const auto &cell = reconstructed.at("cells").at(i);
                check(cell.at("key").at(0) ==
                              static_cast<std::int64_t>(std::floor((-1e9 + i) / 0.001)) &&
                          cell.at("semantic_id") == 3 && cell.at("timestamp_ns") == 0 &&
                          cell.at("classification_age_ns") == UINT64_MAX,
                      "independent full cell identity/label/time reconstruction");
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
