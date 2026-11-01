// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Tuomas Pietilä
// Actual generic session/service with opaque CPU backend; no neural/GPU claim.
#include "generic_fixture_backend.hpp"
#include "synrex/generic_service.hpp"
#include "synrex/generic_transport.hpp"
#include "synrex/common_session.hpp"
#include "synrex/tensor_trace.hpp"
#include <fstream>
#include <future>
#include <iostream>
#include <limits>
#include <sys/socket.h>
#include <unistd.h>

namespace {
using namespace synrex;
using namespace synrex::generic;
namespace fb = fixture_backend;
unsigned checks{}, rejections{}, fixture_index{}, target_queries{};
std::vector<std::string> cases;
std::string current_case;
const std::string session_id(32, 'c');
const std::string model = "opaque retained engine bytes; CPU fixture NEVER executes them";
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
              "unexpected rejection: " + std::string(error.what()) + "; expected " + diagnostic);
        ++rejections;
        return;
    }
    throw std::runtime_error("unexpected acceptance: " + diagnostic);
}
void write_file(const fs::path &path, const std::string &bytes) {
    std::ofstream out(path, std::ios::binary);
    out << bytes;
    out.close();
    check(bool(out), "fixture write");
}
struct Temporary {
    fs::path path;
    explicit Temporary(const fs::path &parent) {
        auto pattern = (parent / "session-fixture-XXXXXX").string();
        const auto created = ::mkdtemp(pattern.data());
        check(created != nullptr, "temporary fixture");
        path = created;
    }
    ~Temporary() {
        std::error_code error;
        fs::remove_all(path, error);
    }
};
fs::path fixture(const fs::path &parent, Json base, unsigned count, bool engine = true) {
    const auto path = parent / ("package-" + std::to_string(++fixture_index));
    fs::create_directory(path);
    Json classes = Json::array(), projection = Json::array();
    // K7 deliberately has neither FRACTAL meanings nor its LAS projection.
    for (unsigned i = 0; i < count; ++i) {
        const auto las = static_cast<unsigned>((31 + i * 7) % 256);
        classes.push_back({{"semantic_id", i},
                           {"key", "private_" + std::to_string(i)},
                           {"name", "Private meaning " + std::to_string(i)},
                           {"las_code", las}});
        projection.push_back(las);
    }
    const Json tax{
        {"schema", "synrex.taxonomy.v1"},
        {"taxonomy_id", "example.cpu.session"},
        {"namespace", "example.cpu"},
        {"classes", classes},
        {"provenance", {{"source", "CPU fixture"}, {"author", "SynRex tests"}, {"license", "MIT"}}},
        {"unknown_semantic_id", nullptr}};
    base["las_classes"] = projection;
    base["output"]["shape"][1] = count;
    const std::string kind = engine ? "engine" : "onnx";
    const std::string basename = "model." + kind;
    write_file(path / basename, model);
    write_file(path / "MODEL_LICENSE.txt", "MIT CPU fixture");
    write_file(path / "taxonomy.json", tax.dump());
    const Json target = engine ? Json{{"architecture", "CPU-test-only"},
                                      {"gpu_name", "NO GPU"},
                                      {"sm", 89},
                                      {"tensorrt", "10.test"},
                                      {"cuda_runtime", 12010}}
                               : Json(nullptr);
    write_file(
        path / "manifest.json",
        Json{{"schema", "synrex.srxmodel.v3"},
             {"contract", base},
             {"artifacts",
              {{kind, {{"path", basename}, {"sha256", sha256(path / basename)}}},
               {"license",
                {{"path", "MODEL_LICENSE.txt"}, {"sha256", sha256(path / "MODEL_LICENSE.txt")}}},
               {"taxonomy",
                {{"path", "taxonomy.json"}, {"sha256", Taxonomy(tax.dump()).sha256()}}}}},
             {"target", target}}
            .dump());
    return path;
}
struct Connection {
    int a{-1}, b{-1};
    std::unique_ptr<AuthorityTransport> authority;
    std::unique_ptr<PeerTransport> peer;
    Connection(const CpuAdmittedPackage &package, Limits limits = {32, 65536, 1000}) {
        int sockets[2];
        check(::socketpair(AF_UNIX, SOCK_STREAM, 0, sockets) == 0, "socketpair");
        a = sockets[0];
        b = sockets[1];
        authority = std::make_unique<AuthorityTransport>(WireProtocol::binary_v3, a, package,
                                                         limits, session_id, 42);
        peer = std::make_unique<PeerTransport>(WireProtocol::binary_v3, b, package, limits,
                                               session_id, 42);
    }
    void admit() {
        auto remote = std::async(std::launch::async, [&] { peer->metadata_admission(); });
        authority->metadata_admission();
        remote.get();
    }
    std::unique_ptr<Session> ready(const DiagnosticTraceOptions &trace = {}) {
        auto remote = std::async(std::launch::async, [&] {
            try {
                peer->acknowledge_execution_ready();
                return true;
            } catch (...) {
                return false;
            }
        });
        try {
            auto result = authority->create_session({}, trace);
            check(remote.get(), "peer ready acknowledgement");
            return result;
        } catch (...) {
            remote.get();
            throw;
        }
    }
    ~Connection() {
        authority.reset();
        peer.reset();
        ::close(a);
        ::close(b);
    }
};
Frame frame(const Session &session, unsigned sequence = 1) {
    Frame result;
    result.identity = {session.descriptor().sha256(), session_id, 42};
    result.sensor_id = "sensor";
    result.stream_id = "stream";
    result.coordinate_frame = "world";
    result.crs = "EPSG:3067";
    result.sequence = sequence;
    result.frame_id = sequence;
    result.frame_start_ns = 1000000ULL * sequence;
    result.frame_end_ns = result.frame_start_ns + 100;
    for (unsigned i = 0; i < 3; ++i)
        result.points.push_back({(1ULL << 60) + sequence * 10 + i,
                                 result.frame_start_ns + i,
                                 {double(sequence), double(i), 3.0},
                                 std::uint16_t(100 + i),
                                 std::uint8_t(1),
                                 std::uint8_t(1)});
    result.accepted_steady_ns = std::numeric_limits<std::uint64_t>::max();
    result.submitted_steady_ns = std::numeric_limits<std::uint64_t>::max();
    return result;
}
void failed_gate(Connection &connection, unsigned expected_calls) {
    check(connection.authority->failed(), "failure poisons authority");
    check(fb::live == 0 && fb::constructions == fb::destructions,
          "failed startup destroys backend");
    rejects([&] { connection.authority->create_session(); }, "permanently failed");
    check(fb::init_calls == expected_calls, "poisoned gate never retries backend");
}
template <class F> void run_case(const std::string &name, F body) {
    current_case = name;
    fb::reset();
    body();
    check(fb::live == 0 && fb::constructions == fb::destructions, "backend lifetime balanced");
    cases.push_back(name);
}
} // namespace
namespace synrex {
Json target_info() {
    ++target_queries;
    throw std::runtime_error("UNREACHABLE GPU target sentinel");
}
} // namespace synrex
int main(int argc, char **argv) {
    try {
        check(argc == 3, "source contract and scratch parent required");
        Temporary temporary(argv[2]);
        const auto base = read_json(argv[1]);
        const auto p4 = fixture(temporary.path, base, 4);
        const CpuAdmittedPackage package(p4);
        run_case("common_dense_wrapper", [&] {
            Connection c(package);
            c.admit();
            auto dense = c.ready();
            auto input = frame(*dense);
            synrex::common::Session common(std::move(dense));
            check(common.backend() == synrex::common::Backend::dense, "explicit dense wrapper");
            rejects([&] { common.point_authority(); }, "point authority");
            rejects([&] { common.run(synrex::point::PreparedBatch{}); }, "backend/input");
            const auto result = std::get<synrex::generic::Classification>(common.run(input));
            check(result.semantic_ids == std::vector<std::uint8_t>({3, 1, 3}), "unchanged dense output");
            check(fb::runs == 1 && fb::init_calls == 1, "dense wrapper initializes/runs once");
        });
        run_case("no_metadata_no_backend", [&] {
            Connection c(package);
            check(!c.authority->initialization_permission(), "fresh gate denies initialization");
            rejects([&] { c.authority->create_session(); });
            failed_gate(c, 0);
        });
        run_case("ONNX_prebackend_rejection", [&] {
            const CpuAdmittedPackage onnx(fixture(temporary.path, base, 4, false));
            Connection c(onnx);
            c.admit();
            rejects([&] { c.authority->create_session(); }, "ONNX");
            failed_gate(c, 0);
        });
        run_case("capacity_prebackend_rejection", [&] {
            auto large_contract = base;
            large_contract["limits"]["max_points"] = 2000000;
            const CpuAdmittedPackage large(fixture(temporary.path, large_contract, 4));
            Connection c(large, {1000001, 65536, 1000});
            c.admit();
            rejects([&] { c.authority->create_session(); }, "point budget");
            failed_gate(c, 0);
        });
        run_case("timeline_prebackend_rejection", [&] {
            Connection c(package);
            c.admit();
            SessionOptions options;
            options.timeline = true;
            rejects([&] { c.authority->create_session(options); }, "timeline");
            failed_gate(c, 0);
        });
        for (const auto fault : {fb::InitFault::throw_error, fb::InitFault::null_session,
                                 fb::InitFault::wrong_identity})
            run_case("backend_init_fault_" + std::to_string(static_cast<int>(fault)), [&] {
                fb::init_fault = fault;
                Connection c(package);
                c.admit();
                rejects([&] { c.authority->create_session(); });
                failed_gate(c, 1);
            });
        run_case("late_initialization_destroys_backend", [&] {
            Connection c(package, {32, 65536, 30});
            c.admit();
            fb::init_delay_ms = 65;
            rejects([&] { c.ready(); }, "deadline");
            failed_gate(c, 1);
        });
        run_case("ready_ack_disconnect_destroys_backend", [&] {
            Connection c(package);
            c.admit();
            ::shutdown(c.b, SHUT_RDWR);
            rejects([&] { c.authority->create_session(); });
            failed_gate(c, 1);
        });
        for (const auto count : {4u, 7u, 256u})
            run_case("actual_session_K" + std::to_string(count), [&] {
                const CpuAdmittedPackage p(fixture(temporary.path, base, count));
                Connection c(p);
                c.admit();
                auto s = c.ready();
                const auto original = frame(*s);
                const auto result = s->run(original);
                check(result.semantic_ids.at(0) == count - 1, "generic semantic ID including255");
                check(result.las_codes.at(0) ==
                          s->descriptor().taxonomy().classes().back().las_code,
                      "exact admitted projection");
                check(result.frame.points.at(0).id == original.points.at(0).id &&
                          result.frame.points.at(0).id > (1ULL << 60),
                      "global source IDs preserved");
                const auto identity = s->identity();
                check(identity.at("generic_session").at("ready_exchange_completed") == true &&
                          identity.at("generic_session").at("backend_failed") == false &&
                          identity.at("generic_session").at("descriptor_sha256") ==
                              s->descriptor().sha256(),
                      "session identity binds admitted descriptor");
                check(fb::runs == 1 && fb::last_capacity == 32 &&
                          fb::last_artifact.lock().get() == fb::last_pointer,
                      "single backend owns exact retained artifact");
            });
        run_case("unsupported_bounded_backend_never_falls_back", [&] {
            struct Unsupported final : InferenceSession {
                const Contract &c;
                explicit Unsupported(const Contract &value) : c(value) {}
                const Contract &contract() const override { return c; }
                Json identity() const override { return Json::object(); }
                WindowResult run(const std::vector<Point> &, const fs::path &) override {
                    throw std::runtime_error("UNREACHABLE legacy trace fallback");
                }
            };
            const Package original(p4);
            Unsupported backend(original.contract);
            rejects([&] { backend.run_traced({}, temporary.path / "unsupported", 1024); },
                    "unavailable");
            check(!fs::exists(temporary.path / "unsupported"),
                  "unsupported bounded method writes nothing");
        });
        run_case("traced_invalid_frame_preserves_forward_admission", [&] {
            DiagnosticTraceOptions trace{temporary.path / "trace-invalid-frame", 4ULL * 1024 * 1024,
                                         2, std::string(40, 'a')};
            Connection c(package);
            c.admit();
            auto session = c.ready(trace);
            auto invalid = frame(*session);
            invalid.identity.descriptor_sha256 = std::string(64, '0');
            rejects([&] { session->run(invalid); });
            check(fb::trace_paths.empty() && fb::runs == 0 && c.authority->failed() &&
                      !fs::exists(trace.root / "window-0"),
                  "traced frame authorization still precedes backend and window writes");
        });
        run_case("diagnostic_mapping_serialization_order_and_budget", [&] {
            const std::vector<std::uint32_t> values{0, 1, 0x12345678, UINT32_MAX};
            const auto bytes = trace_u32_le(values);
            check(bytes.size() == 16 && static_cast<unsigned char>(bytes[8]) == 0x78 &&
                      static_cast<unsigned char>(bytes[11]) == 0x12 &&
                      static_cast<unsigned char>(bytes[15]) == 0xff,
                  "explicit little endian u32");
            const Package original(p4);
            // Input point order differs from voxel order; one point belongs to overlapping blocks.
            const std::vector<Point> points{{{7, 2, 2}, 100}, {{0, 0, 0}, 101}, {{6, 1, 1}, 102}};
            const auto plan = geometry(points, original.contract);
            check(plan.memberships > points.size(), "actual overlapping membership count");
            const auto path = temporary.path / "mapping";
            fs::create_directory(path);
            std::uint64_t written{};
            bool reordered{};
            for (std::size_t bi = 0; bi < plan.blocks.size(); ++bi) {
                const auto &block = plan.blocks[bi];
                std::vector<std::uint32_t> ids, voxels;
                for (const auto &m : block.members) {
                    ids.push_back(m.point);
                    voxels.push_back(m.voxel);
                }
                if (bi == 0)
                    check(ids == std::vector<std::uint32_t>{1, 2, 0} &&
                              voxels == std::vector<std::uint32_t>{0, 62, 119},
                          "noncubic x-fastest mapping and actual sorted order");
                reordered = reordered || !std::is_sorted(ids.begin(), ids.end());
                Json metadata;
                trace_write_memberships(path, std::to_string(bi), ids, voxels, metadata,
                                        8 * plan.memberships, written);
                check(metadata.at("membership_count") == ids.size() &&
                          metadata.at("membership_points_sha256") ==
                              sha256(path / metadata.at("membership_points").get<std::string>()) &&
                          metadata.at("membership_voxels_sha256") ==
                              sha256(path / metadata.at("membership_voxels").get<std::string>()),
                      "mapping count and both hashes");
                std::ifstream input(path / metadata.at("membership_points").get<std::string>(),
                                    std::ios::binary);
                const std::string saved((std::istreambuf_iterator<char>(input)), {});
                check(saved == trace_u32_le(ids), "actual backend traversal bytes preserved");
            }
            check(reordered && written == 8 * plan.memberships,
                  "sorted geometry order differs from point input");
            Json denied;
            const auto before = written;
            rejects(
                [&] {
                    trace_write_memberships(path, "rejected", values, values, denied, written + 31,
                                            written);
                },
                "byte budget");
            check(written == before && !fs::exists(path / "rejected.points.u32.bin"),
                  "full pair reserved before any mapping write");
            rejects(
                [&] {
                    trace_write_memberships(path, "mismatch", values, {}, denied, written + 32,
                                            written);
                },
                "count mismatch");
            check(
                diagnostic_tensor_bound(original.contract, plan.blocks.size(), plan.memberships) ==
                    diagnostic_tensor_bound(original.contract, plan.blocks.size()) +
                        8 * plan.memberships,
                "reservation uses actual membership count");
            rejects([&] { diagnostic_tensor_bound(original.contract, 1, UINT64_MAX); },
                    "mapping total byte bound");
            check(trace_membership_semantics(original.contract).at("voxel_linearization") ==
                      "(iz * ny + iy) * nx + ix; x fastest",
                  "x fastest semantics");
        });
        run_case("diagnostic_disabled_unchanged_forwarding", [&] {
            Connection c(package);
            c.admit();
            auto session = c.ready();
            check(session->run(frame(*session)).semantic_ids.size() == 3 && fb::trace_paths.empty(),
                  "off-by-default uses original backend run without trace I/O");
            check(!session->identity().contains("diagnostic_trace"), "off identity unchanged");
        });
        for (const auto fault : {"partial", "bytes", "minimum", "windows", "source", "existing"})
            run_case(std::string("diagnostic_preGPU_") + fault, [&] {
                DiagnosticTraceOptions trace{temporary.path / (std::string("trace-") + fault),
                                             4ULL * 1024 * 1024, 2, std::string(40, 'a')};
                const std::string name(fault);
                if (name == "partial")
                    trace.root.clear();
                if (name == "bytes")
                    trace.total_bytes = 512 * 1024;
                if (name == "minimum")
                    trace.total_bytes = 512 * 1024 + 1;
                if (name == "windows")
                    trace.windows = 65;
                if (name == "source")
                    trace.declared_source_commit = "unbound";
                if (name == "existing")
                    fs::create_directory(trace.root);
                Connection c(package);
                c.admit();
                rejects([&] { c.authority->create_session({}, trace); }, "diagnostic");
                failed_gate(c, 0);
            });
        run_case("diagnostic_relative_root_is_captured_across_cwd_change", [&] {
            const auto saved = fs::current_path();
            fs::current_path(temporary.path);
            try {
                DiagnosticTraceOptions trace{"relative-trace", 4ULL * 1024 * 1024, 1,
                                             std::string(40, 'a')};
                Connection c(package);
                c.admit();
                auto session = c.ready(trace);
                fs::current_path(saved);
                session->run(frame(*session));
                check(fb::trace_paths.back() == temporary.path / "relative-trace/window-0" &&
                          fs::exists(temporary.path / "relative-trace/window-0/lineage.json"),
                      "writes use the captured absolute admitted root after cwd changes");
            } catch (...) {
                fs::current_path(saved);
                throw;
            }
            fs::current_path(saved);
        });
        run_case("diagnostic_forwarding_lineage_and_window_budget", [&] {
            DiagnosticTraceOptions trace{temporary.path / "trace-good", 4ULL * 1024 * 1024, 2,
                                         std::string(40, 'a')};
            Connection c(package);
            c.admit();
            auto session = c.ready(trace);
            const auto received = std::chrono::duration_cast<std::chrono::nanoseconds>(
                                      std::chrono::steady_clock::now().time_since_epoch())
                                      .count();
            const auto first = session->run(frame(*session, 1), received);
            const auto second = session->run(frame(*session, 2));
            check(first.semantic_ids == second.semantic_ids && fb::trace_paths.size() == 2 &&
                      fb::trace_paths[0] == trace.root / "window-0" &&
                      fb::trace_paths[1] == trace.root / "window-1",
                  "exact budgeted path forwarding");
            check(fs::exists(trace.root / "admission.json") &&
                      read_json(trace.root / "window-0/lineage.json").at("descriptor_sha256") ==
                          session->descriptor().sha256(),
                  "actual input/source/policy lineage persisted");
            check(first.timings.at("diagnostic_service_seconds").get<double>() >= 0 &&
                      session->identity().at("diagnostic_trace").at("completed_windows") == 2,
                  "trace timing/counters exposed");
            const auto timing = read_json(trace.root / "window-0/timing.json");
            check(timing.at("receiver_frame_available_steady_ns") == received &&
                      timing.at("receiver_frame_available_steady_ns") <=
                          timing.at("service_started_steady_ns") &&
                      timing.at("service_started_steady_ns") <=
                          timing.at("trace_payload_published_steady_ns") &&
                      timing.at("trace_payload_published_steady_ns") <=
                          timing.at("classification_validated_steady_ns") &&
                      read_json(trace.root / "window-1/timing.json")
                          .at("receiver_frame_available_steady_ns")
                          .is_null(),
                  "durable local boundaries and explicit missing receive sample");
            auto sampled = [] {
                return std::chrono::duration_cast<std::chrono::nanoseconds>(
                           std::chrono::steady_clock::now().time_since_epoch())
                    .count();
            };
            const auto ready = sampled(), started = sampled(), sent = sampled();
            session->record_output_timing(0, ready, started, sent);
            check(read_json(trace.root / "window-0/output-timing.json")
                          .at("output_ready_steady_ns") == ready,
                  "post-run output-ready boundary persisted after send");
            rejects([&] { session->run(frame(*session, 3)); }, "window budget");
            check(fb::runs == 2 && c.authority->failed(),
                  "trace window limit rejects before inference");
            rejects([&] { session->run(frame(*session, 4)); }, "permanently failed");
        });
        run_case("diagnostic_aggregate_byte_budget_not_per_window", [&] {
            const Package original(p4);
            const auto plan =
                geometry(std::vector<Point>{{{1, 0, 3}, 100}, {{1, 1, 3}, 101}, {{1, 2, 3}, 102}},
                         original.contract);
            const auto allowance =
                diagnostic_tensor_bound(original.contract, plan.blocks.size(), plan.memberships) +
                3 * 4096;
            DiagnosticTraceOptions trace{temporary.path / "trace-small", 512 * 1024 + allowance, 2,
                                         std::string(40, 'a')};
            Connection c(package);
            c.admit();
            auto session = c.ready(trace);
            session->run(frame(*session, 1));
            rejects([&] { session->run(frame(*session, 2)); }, "session byte budget");
            check(fb::runs == 1 && !fs::exists(trace.root / "window-1") && c.authority->failed(),
                  "aggregate reserved budget prevents second backend/window before write");
        });
        run_case("diagnostic_window_collision_preserves_prior_evidence", [&] {
            DiagnosticTraceOptions trace{temporary.path / "trace-collision", 4ULL * 1024 * 1024, 2,
                                         std::string(40, 'a')};
            Connection c(package);
            c.admit();
            auto session = c.ready(trace);
            fs::create_directory(trace.root / "window-0");
            write_file(trace.root / "window-0/prior.txt", "immutable prior evidence");
            rejects([&] { session->run(frame(*session)); }, "collision");
            check(fb::runs == 0 && c.authority->failed() &&
                      fs::file_size(trace.root / "window-0/prior.txt") == 24,
                  "colliding trace window cannot overwrite or infer");
        });
        run_case("diagnostic_timing_publication_failure_poisoning", [&] {
            DiagnosticTraceOptions trace{temporary.path / "trace-timing-collision",
                                         4ULL * 1024 * 1024, 2, std::string(40, 'a')};
            Connection c(package);
            c.admit();
            auto session = c.ready(trace);
            fb::trace_timing_collision = true;
            rejects([&] { session->run(frame(*session)); });
            check(fb::runs == 1 && c.authority->failed() &&
                      fs::exists(trace.root / "window-0/lineage.json") &&
                      fs::is_directory(trace.root / "window-0/timing.json"),
                  "partial trace remains, timing failure prevents classification return");
            rejects([&] { session->run(frame(*session, 2)); }, "permanently failed");
        });
        run_case("diagnostic_write_failure_poisoning", [&] {
            DiagnosticTraceOptions trace{temporary.path / "trace-failure", 4ULL * 1024 * 1024, 2,
                                         std::string(40, 'a')};
            Connection c(package);
            c.admit();
            auto session = c.ready(trace);
            fb::trace_write_failure = true;
            rejects([&] { session->run(frame(*session, 1)); }, "trace write failure");
            check(fb::runs == 0 && c.authority->failed(),
                  "trace failure stops inference/authority");
            rejects([&] { session->run(frame(*session, 2)); }, "permanently failed");
        });
        run_case("duplicate_create_rejected_without_second_backend", [&] {
            Connection c(package);
            c.admit();
            auto s = c.ready();
            rejects([&] { c.authority->create_session(); });
            check(c.authority->failed() && fb::init_calls == 1,
                  "duplicate poisons without backend retry");
            rejects([&] { s->run(frame(*s)); });
            check(fb::runs == 0, "duplicate creation revokes existing frame permission");
        });
        run_case("retained_bytes_outlive_wrappers_and_rewritten_deleted_files", [&] {
            const auto path = fixture(temporary.path, base, 4);
            std::unique_ptr<Connection> c;
            std::unique_ptr<Session> s;
            {
                const CpuAdmittedPackage p(path);
                c = std::make_unique<Connection>(p);
                c->admit();
                write_file(path / "model.engine", "rewritten after CPU capture");
                fs::remove_all(path);
                s = c->ready();
            }
            check(fb::last_artifact.lock()->engine_bytes() == model,
                  "captured bytes survive original package deletion");
            // Session's authorizer retains connection and artifact after authority wrapper dies.
            c->authority.reset();
            c->peer.reset();
            check(s->run(frame(*s)).semantic_ids.size() == 3,
                  "session survives transport wrapper lifetime");
            s.reset();
            check(fb::last_artifact.expired(),
                  "retained artifact released with session and control owners");
        });
        for (const auto fault : {"descriptor", "session", "epoch", "missing_intensity"})
            run_case(std::string("frame_fault_") + fault, [&] {
                Connection c(package);
                c.admit();
                auto s = c.ready();
                auto f = frame(*s);
                const std::string name(fault);
                if (name == "descriptor")
                    f.identity.descriptor_sha256 = std::string(64, '0');
                if (name == "session")
                    f.identity.session_id = std::string(32, '0');
                if (name == "epoch")
                    ++f.identity.map_epoch;
                if (name == "missing_intensity")
                    f.points.at(0).intensity.reset();
                rejects([&] { s->run(f); });
                rejects([&] { s->run(frame(*s)); }, "permanently failed");
                check(fb::runs == 0 && c.authority->failed() &&
                          s->identity().at("generic_session").at("backend_failed") == true,
                      "invalid frame permanently poisons session before inference");
            });
        for (const auto fault : {fb::RunFault::throw_error, fb::RunFault::missing_output,
                                 fb::RunFault::bad_semantic_id, fb::RunFault::bad_projection})
            run_case("backend_output_fault_" + std::to_string(static_cast<int>(fault)), [&] {
                Connection c(package);
                c.admit();
                auto s = c.ready();
                fb::run_fault = fault;
                rejects([&] { s->run(frame(*s)); });
                fb::run_fault = fb::RunFault::none;
                rejects([&] { s->run(frame(*s)); }, "permanently failed");
                check(fb::runs == 1 && c.authority->failed(),
                      "invalid backend output fail-stops session");
            });
        for (const auto fault : {"repeat_frame", "sequence", "frame_id", "time_overlap", "sensor",
                                 "stream", "coordinate_frame", "crs"})
            run_case(std::string("session_continuity_") + fault, [&] {
                Connection c(package);
                c.admit();
                auto s = c.ready();
                const auto first = frame(*s, 1);
                check(s->run(first).semantic_ids.size() == 3,
                      "first frame establishes stream identity");
                auto next = frame(*s, 2);
                const std::string name(fault);
                if (name == "repeat_frame")
                    next = first;
                if (name == "sequence")
                    next.sequence = first.sequence;
                if (name == "frame_id")
                    next.frame_id = first.frame_id;
                if (name == "time_overlap") {
                    next.frame_start_ns = first.frame_end_ns - 1;
                    next.frame_end_ns = next.frame_start_ns + 100;
                    for (auto &p : next.points)
                        p.timestamp_ns = next.frame_start_ns;
                }
                if (name == "sensor")
                    next.sensor_id += "_other";
                if (name == "stream")
                    next.stream_id += "_other";
                if (name == "coordinate_frame")
                    next.coordinate_frame += "_other";
                if (name == "crs")
                    next.crs += "_other";
                rejects([&] { s->run(next); });
                rejects([&] { s->run(frame(*s, 3)); }, "permanently failed");
                check(fb::runs == 1 && c.authority->failed(),
                      "replay or stream rebind fail-stops before backend");
            });
        run_case("queued_distinct_service_frames_terminal_receipt_and_local_clocks", [&] {
            Connection c(package);
            c.admit();
            auto s = c.ready();
            std::vector<Classification> emitted;
            const auto f1 = frame(*s, 1), f2 = frame(*s, 2), f3 = frame(*s, 3);
            Service service(std::move(s), 2,
                            [&](const Classification &result) { emitted.push_back(result); });
            check(service.progress().at("status") == "RUNNING", "service starts RUNNING");
            rejects([&] { service.receipt(); }, "terminal drain");
            check(service.submit(f1) && service.submit(f2) && service.submit(f3),
                  "distinct frames queued");
            // Workers may finish quickly, but terminal qualification requires explicit finish.
            check(service.progress().at("status") == "RUNNING",
                  "drain alone cannot claim terminal PASS");
            service.finish();
            const auto receipt = service.receipt();
            check(receipt.at("status") == "PASS" && receipt.at("submitted_points") == 9 &&
                      receipt.at("freshly_classified_points") == 9 &&
                      receipt.at("emitted_points") == 9 && receipt.at("completed_windows") == 3 &&
                      receipt.at("dropped_points") == 0 && receipt.at("rejected_frames") == 0 &&
                      emitted.size() == 3,
                  "service terminal counters match distinct emitted coverage");
            for (unsigned i = 0; i < emitted.size(); ++i) {
                const auto &r = emitted.at(i);
                check(r.frame.sequence == i + 1 &&
                          r.frame.points.at(0).id == (1ULL << 60) + (i + 1) * 10,
                      "ordered global IDs never interpreted as single-file indexes");
                check(r.frame.accepted_steady_ns > 0 &&
                          r.frame.accepted_steady_ns != f1.accepted_steady_ns &&
                          r.frame.submitted_steady_ns == r.frame.accepted_steady_ns,
                      "service overwrites incoming host clocks with local acceptance");
                check(r.timings.at("service_input_queue_wait_seconds").get<double>() >= 0 &&
                          r.timings.at("service_acceptance_to_inference_seconds").get<double>() >=
                              0,
                      "local service timing finite and nonnegative");
            }
            rejects([&] { service.submit(f1); }, "finished");
        });
        run_case("service_sink_failure_nonpass", [&] {
            Connection c(package);
            c.admit();
            auto s = c.ready();
            const auto f = frame(*s);
            Service service(std::move(s), 1, [](const Classification &) {
                throw std::runtime_error("sink sentinel");
            });
            check(service.submit(f), "submit before sink failure");
            rejects([&] { service.finish(); }, "sink sentinel");
            const auto receipt = service.receipt();
            check(receipt.at("status") == "FAIL" && receipt.at("emitted_points") == 0 &&
                      receipt.at("dropped_points") == 3,
                  "sink failure prevents successful receipt");
        });
        run_case("service_invalid_frame_nonpass", [&] {
            Connection c(package);
            c.admit();
            auto s = c.ready();
            auto f = frame(*s);
            ++f.identity.map_epoch;
            Service service(std::move(s), 1, [](const Classification &) {});
            rejects([&] { service.submit(f); });
            rejects([&] { service.finish(); });
            check(service.receipt().at("status") == "FAIL" &&
                      service.receipt().at("rejected_frames") == 1 && fb::runs == 0,
                  "invalid service frame cannot qualify or execute backend");
        });
        run_case("service_inference_failure_nonpass", [&] {
            Connection c(package);
            c.admit();
            auto s = c.ready();
            const auto f = frame(*s);
            fb::run_fault = fb::RunFault::missing_output;
            Service service(std::move(s), 1, [](const Classification &) {});
            check(service.submit(f), "submit before malformed backend output");
            rejects([&] { service.finish(); });
            check(service.receipt().at("status") == "FAIL" &&
                      service.receipt().at("emitted_points") == 0 &&
                      service.receipt().at("dropped_points") == 3 && fb::runs == 1,
                  "malformed backend coverage cannot become successful service result");
        });
        run_case("empty_finished_service_nonpass", [&] {
            Connection c(package);
            c.admit();
            Service service(c.ready(), 1, [](const Classification &) {});
            service.finish();
            check(service.receipt().at("status") == "FAIL" && fb::runs == 0,
                  "empty drain does not claim work completion");
        });
        for (const auto capacity : {0u, 9u})
            run_case("service_queue_capacity_" + std::to_string(capacity), [&] {
                Connection c(package);
                c.admit();
                rejects(
                    [&] { Service service(c.ready(), capacity, [](const Classification &) {}); },
                    "eight");
                check(fb::live == 0, "constructor rejection destroys already-ready session");
            });
        check(target_queries == 0, "CPU tests never call GPU target detection");
        std::cout << Json{{"status", "PASS_CPU_GENERIC_SESSION_SERVICE"},
                          {"checks", checks},
                          {"expected_rejections", rejections},
                          {"cases", cases},
                          {"gpu_target_calls", target_queries},
                          {"gpu_initializations", 0},
                          {"backend", "opaque CPU fixture, no neural/GPU execution"}}
                         .dump()
                  << '\n';
        return 0;
    } catch (const std::exception &error) {
        std::cerr << current_case << ": " << error.what() << '\n';
        return 1;
    }
}
