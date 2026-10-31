// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Tuomas Pietilä
// Actual generic session/service with opaque CPU backend; no neural/GPU claim.
#include "generic_fixture_backend.hpp"
#include "synrex/generic_service.hpp"
#include "synrex/generic_transport.hpp"
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
    std::unique_ptr<Session> ready() {
        auto remote = std::async(std::launch::async, [&] {
            try {
                peer->acknowledge_execution_ready();
                return true;
            } catch (...) {
                return false;
            }
        });
        try {
            auto result = authority->create_session();
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
