// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Tuomas Pietilä
// Coordinated producer/subscriber transport over real CPU socketpairs only.
#include "generic_fixture_backend.hpp"
#include "synrex/generic_output.hpp"
#include "synrex/generic_session.hpp"
#include "synrex/generic_transport.hpp"
#include <algorithm>
#include <array>
#include <cmath>
#include <cstring>
#include <fstream>
#include <future>
#include <iostream>
#include <limits>
#include <openssl/sha.h>
#include <sys/socket.h>
#include <unistd.h>

namespace {
using namespace synrex;
using namespace synrex::generic;
namespace fb = fixture_backend;
using Clock = Startup::Clock;
std::atomic<unsigned> checks{};
unsigned rejections{}, fixture_index{}, target_queries{};
std::vector<std::string> cases;
std::string current_case;
const std::string session_id(32, 'd');
constexpr Limits limits{1024, 65536, 1000};
void check(bool value, const std::string &message) {
    ++checks;
    if (!value)
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
Startup::Time deadline(unsigned milliseconds = 1000) {
    return Clock::now() + std::chrono::milliseconds(milliseconds);
}
void write_file(const fs::path &path, const std::string &bytes) {
    std::ofstream stream(path, std::ios::binary);
    stream << bytes;
    stream.close();
    check(bool(stream), "fixture write");
}
struct Temporary {
    fs::path path;
    explicit Temporary(const fs::path &parent) {
        auto pattern = (parent / "duplex-fixture-XXXXXX").string();
        const auto created = ::mkdtemp(pattern.data());
        check(created != nullptr, "temporary fixture");
        path = created;
    }
    ~Temporary() {
        std::error_code error;
        fs::remove_all(path, error);
    }
};
fs::path fixture(const fs::path &parent, Json base, unsigned count, bool changed_meaning = false) {
    const auto path = parent / ("package-" + std::to_string(++fixture_index));
    fs::create_directory(path);
    Json classes = Json::array(), projection = Json::array();
    const unsigned nonlegacy[] = {1, 2, 200, 6, 9, 17, 201};
    for (unsigned i = 0; i < count; ++i) {
        const auto las = count == 7 ? nonlegacy[i] : (31 + i * 7) % 256;
        classes.push_back({{"semantic_id", i},
                           {"key", "private_" + std::to_string(i)},
                           {"name", (changed_meaning ? "Changed meaning " : "Private meaning ") +
                                        std::to_string(i)},
                           {"las_code", las}});
        projection.push_back(las);
    }
    const Json taxonomy{
        {"schema", "synrex.taxonomy.v1"},
        {"taxonomy_id", "example.reused.socket"},
        {"namespace", "example.cpu"},
        {"classes", classes},
        {"provenance", {{"source", "CPU fixture"}, {"author", "SynRex tests"}, {"license", "MIT"}}},
        {"unknown_semantic_id", nullptr}};
    base["las_classes"] = projection;
    base["output"]["shape"][1] = count;
    write_file(path / "model.engine", "Opaque CPU fixture bytes: NEVER executed");
    write_file(path / "MODEL_LICENSE.txt", "MIT CPU fixture");
    write_file(path / "taxonomy.json", taxonomy.dump());
    write_file(
        path / "manifest.json",
        Json{{"schema", "synrex.srxmodel.v3"},
             {"contract", base},
             {"artifacts",
              {{"engine", {{"path", "model.engine"}, {"sha256", sha256(path / "model.engine")}}},
               {"license",
                {{"path", "MODEL_LICENSE.txt"}, {"sha256", sha256(path / "MODEL_LICENSE.txt")}}},
               {"taxonomy",
                {{"path", "taxonomy.json"}, {"sha256", Taxonomy(taxonomy.dump()).sha256()}}}}},
             {"target",
              {{"architecture", "CPU-test-only"},
               {"gpu_name", "NO GPU"},
               {"sm", 89},
               {"tensorrt", "10.test"},
               {"cuda_runtime", 12010}}}}
            .dump());
    return path;
}
struct Connection {
    int a{-1}, b{-1};
    std::unique_ptr<AuthorityTransport> authority;
    std::unique_ptr<PeerTransport> peer;
    std::unique_ptr<Session> session;
    explicit Connection(const CpuAdmittedPackage &package,
                        const CpuAdmittedPackage *expected = nullptr, Limits supported = limits) {
        int sockets[2];
        check(::socketpair(AF_UNIX, SOCK_STREAM, 0, sockets) == 0, "socketpair");
        a = sockets[0];
        b = sockets[1];
        authority = std::make_unique<AuthorityTransport>(WireProtocol::binary_v3, a, package,
                                                         supported, session_id, 42);
        peer = std::make_unique<PeerTransport>(
            WireProtocol::binary_v3, b, expected ? *expected : package, supported, session_id, 42);
    }
    void admit() {
        auto remote = std::async(std::launch::async, [&] { peer->metadata_admission(); });
        try {
            authority->metadata_admission();
        } catch (...) {
            try {
                remote.get();
            } catch (...) {
            }
            throw;
        }
        remote.get();
    }
    void ready() {
        admit();
        auto remote = std::async(std::launch::async, [&] { peer->acknowledge_execution_ready(); });
        try {
            session = authority->create_session();
        } catch (...) {
            try {
                remote.get();
            } catch (...) {
            }
            throw;
        }
        remote.get();
        check(fb::init_calls == 1 && fb::live == 1, "actual startup constructs CPU fixture once");
    }
    ~Connection() {
        session.reset();
        authority.reset();
        peer.reset();
        ::close(a);
        ::close(b);
    }
};
Classification classification(const Descriptor &descriptor, std::uint64_t sequence,
                              double x = 0.25) {
    Classification result;
    auto &frame = result.frame;
    frame.identity = {descriptor.sha256(), session_id, 42};
    frame.sensor_id = "sensor";
    frame.stream_id = "stream";
    frame.coordinate_frame = "world";
    frame.crs = "LOCAL:output-test";
    frame.sequence = frame.frame_id = sequence;
    frame.frame_start_ns = frame.frame_end_ns = sequence;
    frame.points = {{sequence, sequence, {x, 0, 0}, 500, {}, {}}};
    const auto id = static_cast<std::uint8_t>(descriptor.taxonomy().classes().size() - 1);
    result.semantic_ids = {id};
    result.las_codes = {descriptor.taxonomy().classes().at(id).las_code};
    return result;
}
std::string digest(const std::string &bytes) {
    std::array<unsigned char, SHA256_DIGEST_LENGTH> hash{};
    SHA256(reinterpret_cast<const unsigned char *>(bytes.data()), bytes.size(), hash.data());
    std::string result;
    const char hex[] = "0123456789abcdef";
    for (auto byte : hash) {
        result += hex[byte >> 4];
        result += hex[byte & 15];
    }
    return result;
}
std::string envelope(unsigned kind, std::string_view bytes) {
    const auto size = static_cast<std::uint32_t>(bytes.size());
    std::string result{'S',
                       'R',
                       'X',
                       '3',
                       3,
                       static_cast<char>(kind),
                       0,
                       0,
                       static_cast<char>(size >> 24),
                       static_cast<char>(size >> 16),
                       static_cast<char>(size >> 8),
                       static_cast<char>(size)};
    result.append(bytes);
    return result;
}
void raw_write(int socket, const std::string &bytes) {
    for (std::size_t offset = 0; offset < bytes.size();) {
        const auto count =
            ::send(socket, bytes.data() + offset, bytes.size() - offset, MSG_NOSIGNAL);
        check(count > 0, "raw output write");
        offset += static_cast<std::size_t>(count);
    }
}
std::string raw_read(int socket, std::size_t size) {
    std::string result(size, '\0');
    for (std::size_t offset = 0; offset < size;) {
        const auto count = ::recv(socket, result.data() + offset, size - offset, 0);
        check(count > 0, "raw output read");
        offset += static_cast<std::size_t>(count);
    }
    return result;
}
std::string raw_message(int socket, unsigned kind) {
    const auto header = raw_read(socket, 12);
    check(static_cast<unsigned char>(header[5]) == kind, "raw terminal kind");
    std::uint32_t size{};
    for (unsigned i = 8; i < 12; ++i)
        size = (size << 8) | static_cast<unsigned char>(header[i]);
    check(size > 0 && size <= 2048, "raw terminal bound");
    return raw_read(socket, size);
}
void raw_eof(int socket) {
    char byte{};
    check(::recv(socket, &byte, 1, 0) == 0, "raw terminal EOF");
}
struct Duplex {
    Connection input, output;
    std::unique_ptr<Session> session;
    explicit Duplex(const CpuAdmittedPackage &package) : input(package), output(package) {}
    void admit() {
        input.admit();
        output.admit();
    }
    void ready() {
        admit();
        auto producer =
            std::async(std::launch::async, [&] { input.peer->acknowledge_execution_ready(); });
        auto subscriber =
            std::async(std::launch::async, [&] { output.peer->acknowledge_execution_ready(); });
        std::exception_ptr error;
        try {
            session = input.authority->create_session_with_output(*output.authority);
        } catch (...) {
            error = std::current_exception();
        }
        try {
            producer.get();
        } catch (...) {
            if (!error)
                error = std::current_exception();
        }
        try {
            subscriber.get();
        } catch (...) {
            if (!error)
                error = std::current_exception();
        }
        if (error)
            std::rethrow_exception(error);
        check(fb::init_calls == 1 && fb::live == 1,
              "one retained backend on two ready connections");
    }
    void both_failed() {
        check(input.authority->failed() && output.authority->failed(), "both authorities poisoned");
    }
};
void feed(Connection &connection, unsigned sequence = 1) {
    auto frame = classification(connection.authority->descriptor(), sequence).frame;
    connection.peer->send_frame(frame, deadline());
    check(connection.authority->receive_frame(deadline()).sequence == sequence,
          "typed input frame received");
}
Json input_record(const Descriptor &descriptor, const std::string &transcript, unsigned frames = 1,
                  unsigned points = 1) {
    return {{"schema", "synrex.input.terminal.v3"},
            {"kind", "input_complete"},
            {"descriptor_sha256", descriptor.sha256()},
            {"session_id", session_id},
            {"map_epoch", 42},
            {"taxonomy_sha256", descriptor.taxonomy().sha256()},
            {"input_frames", frames},
            {"input_points", points},
            {"input_sha256", digest(transcript)},
            {"completion_challenge", std::string(32, 'e')}};
}
void raw_feed(Connection &connection) {
    const auto frame = classification(connection.authority->descriptor(), 1).frame;
    raw_write(connection.b, encode_frame(connection.authority->descriptor(), frame));
    (void)connection.authority->receive_frame(deadline());
}
template <class F> void run_case(const std::string &name, F operation) {
    current_case = name;
    fb::reset();
    operation();
    check(fb::live == 0, "fixture session destroyed");
    cases.push_back(name);
}
} // namespace
namespace synrex {
Json target_info() {
    ++target_queries;
    throw std::runtime_error("UNREACHABLE GPU TARGET");
}
} // namespace synrex
int main(int argc, char **argv) {
    try {
        check(argc == 3, "contract and scratch parent required");
        Temporary temporary(argv[2]);
        const auto base = read_json(argv[1]);
        const CpuAdmittedPackage p4(fixture(temporary.path, base, 4));
        for (const unsigned count : {4U, 7U, 256U}) {
            const CpuAdmittedPackage package(fixture(temporary.path, base, count));
            run_case("duplex_input_output_completion_K" + std::to_string(count), [&] {
                Duplex d(package);
                fb::init_hook = [&] {
                    check(d.input.authority->initialization_permission() &&
                              d.output.authority->initialization_permission(),
                          "both metadata gates consumed before backend");
                };
                d.ready();
                OutputSender sender(*d.output.authority, 1, 8);
                OutputReceiver receiver(*d.output.peer, 1, 8);
                std::string encoded;
                for (unsigned sequence = 1; sequence <= 3; ++sequence) {
                    auto frame = classification(d.input.authority->descriptor(), sequence).frame;
                    encoded += encode_frame(d.input.authority->descriptor(), frame);
                    d.input.peer->send_frame(frame, deadline());
                    auto received = d.input.authority->receive_frame(deadline());
                    auto result = d.session->run(std::move(received));
                    sender.send(result, deadline());
                    check(receiver.receive(deadline()).semantic_ids.front() == count - 1,
                          "one backend returns full taxonomy including K256 ID255");
                }
                auto producer = std::async(std::launch::async,
                                           [&] { return d.input.peer->finish_input(deadline()); });
                const auto input = d.input.authority->finish_input(3, 3, deadline());
                const auto remote = producer.get();
                check(input.at("input_sha256") == digest(encoded) &&
                          remote.at("input_sha256") == input.at("input_sha256") &&
                          remote.at("input_frames") == 3 && remote.at("input_points") == 3 &&
                          input.at("completion_challenge") == remote.at("completion_challenge") &&
                          input.at("acknowledgement_challenge") ==
                              remote.at("acknowledgement_challenge") &&
                          remote.at("status") == "PEER_VALIDATED" &&
                          input.at("time_boundary") ==
                              "input_delivery_only; no_inference_or_output_success_claim",
                      "input transcript independent SHA/count/challenge receipt");
                check(!d.output.authority->failed(), "output remains independently ready");
                auto subscriber =
                    std::async(std::launch::async, [&] { return receiver.finish(deadline()); });
                const auto output = sender.finish(deadline());
                const auto received_output = subscriber.get();
                check(output.at("status") == "PEER_VALIDATED" &&
                          received_output.at("classification_frames") == 3,
                      "independently reconstructed output terminal completion");
                check(fb::init_calls == 1 && fb::runs == 3,
                      "no second backend or fake initialization");
                rejects([&] { d.input.authority->finish_input(3, 3, deadline()); },
                        "before execution-ready");
                d.both_failed();
                rejects([&] { d.input.peer->finish_input(deadline()); });
            });
        }
        for (unsigned fault = 0; fault < 7; ++fault)
            run_case("pre_backend_gate_" + std::to_string(fault), [&] {
                Connection input(p4), output(p4);
                if (fault != 0)
                    input.admit();
                if (fault != 1 && fault != 0)
                    output.admit();
                if (fault == 2) {
                    rejects([&] { output.authority->receive_frame(deadline()); });
                }
                if (fault == 3) {
                    rejects([&] { input.authority->create_session_with_output(*input.authority); },
                            "distinct");
                    check(input.authority->failed() && fb::init_calls == 0,
                          "self connection rejected");
                    return;
                }
                if (fault == 4) {
                    const CpuAdmittedPackage changed(fixture(temporary.path, base, 4, true));
                    Connection other(changed);
                    other.admit();
                    rejects([&] { input.authority->create_session_with_output(*other.authority); },
                            "descriptor mismatch");
                    check(input.authority->failed() && other.authority->failed() &&
                              fb::init_calls == 0,
                          "changed meaning rejected before backend");
                    return;
                }
                if (fault == 5) {
                    Connection other(p4, nullptr, {512, 65536, 1000});
                    other.admit();
                    rejects([&] { input.authority->create_session_with_output(*other.authority); },
                            "descriptor mismatch");
                    check(input.authority->failed() && other.authority->failed() &&
                              fb::init_calls == 0,
                          "different agreed capacity rejected before backend");
                    return;
                }
                if (fault == 6) {
                    const int duplicate = ::dup(input.a);
                    check(duplicate >= 0, "dup socket");
                    AuthorityTransport alias(WireProtocol::binary_v3, duplicate, p4, limits,
                                             session_id, 42);
                    rejects([&] { input.authority->create_session_with_output(alias); },
                            "distinct");
                    check(alias.failed() && input.authority->failed() && fb::init_calls == 0,
                          "duplicate file description rejected before backend");
                    ::close(duplicate);
                    return;
                }
                rejects([&] { input.authority->create_session_with_output(*output.authority); });
                check(input.authority->failed() && output.authority->failed() &&
                          fb::init_calls == 0,
                      "unadmitted or failed output prevents backend");
            });
        for (const auto fault : {fb::InitFault::throw_error, fb::InitFault::null_session,
                                 fb::InitFault::wrong_identity})
            run_case("backend_failure_" + std::to_string(static_cast<int>(fault)), [&] {
                Duplex d(p4);
                fb::init_fault = fault;
                rejects([&] { d.ready(); });
                d.both_failed();
                check(fb::live == 0 && fb::init_calls == 1, "failed backend destroyed");
            });
        run_case("expired_output_gate_prevents_backend", [&] {
            const Limits brief{1024, 65536, 20};
            Connection input(p4, nullptr, brief), output(p4, nullptr, brief);
            output.admit();
            std::this_thread::sleep_for(std::chrono::milliseconds(35));
            input.admit();
            rejects([&] { input.authority->create_session_with_output(*output.authority); },
                    "deadline");
            check(input.authority->failed() && output.authority->failed() && fb::init_calls == 0,
                  "expired output gate poisons both before backend creation");
        });
        run_case("missing_subscriber_ready_ACK", [&] {
            Duplex d(p4);
            d.admit();
            auto producer = std::async(std::launch::async,
                                       [&] { d.input.peer->acknowledge_execution_ready(); });
            rejects(
                [&] {
                    d.session = d.input.authority->create_session_with_output(*d.output.authority);
                },
                "deadline");
            producer.get();
            d.both_failed();
            check(fb::live == 0 && fb::init_calls == 1, "ready failure destroys single backend");
        });
        run_case("session_failure_aborts_both", [&] {
            Duplex d(p4);
            d.ready();
            fb::run_fault = fb::RunFault::throw_error;
            rejects(
                [&] { d.session->run(classification(d.input.authority->descriptor(), 1).frame); });
            d.both_failed();
        });
        run_case("output_transport_failure_aborts_both", [&] {
            Duplex d(p4);
            d.ready();
            rejects([&] { d.output.authority->receive_frame(Clock::now()); }, "deadline");
            d.both_failed();
        });
        for (bool authority : {false, true})
            run_case(std::string("empty_input_") + (authority ? "authority" : "peer"), [&] {
                Connection c(p4);
                c.ready();
                if (authority)
                    rejects([&] { c.authority->finish_input(0, 0, deadline()); }, "empty input");
                else
                    rejects([&] { c.peer->finish_input(deadline()); }, "empty input");
                check(authority ? c.authority->failed() : c.peer->failed(),
                      "empty input poisons caller");
            });
        for (unsigned fault = 0; fault < 3; ++fault)
            run_case("authority_wrong_expected_" + std::to_string(fault), [&] {
                Connection c(p4);
                c.ready();
                feed(c);
                rejects(
                    [&] {
                        c.authority->finish_input(fault == 0 ? 0 : 1,
                                                  fault == 1   ? 0
                                                  : fault == 2 ? 2
                                                               : 1,
                                                  deadline());
                    },
                    "count mismatch");
                check(c.authority->failed(), "wrong expected count poisons");
            });
        for (unsigned fault = 0; fault < 17; ++fault)
            run_case("authority_terminal_fault_" + std::to_string(fault), [&] {
                Connection c(p4);
                c.ready();
                raw_feed(c);
                const auto bytes = encode_frame(c.authority->descriptor(),
                                                classification(c.authority->descriptor(), 1).frame);
                auto terminal = input_record(c.authority->descriptor(), bytes);
                if (fault == 0)
                    terminal["input_sha256"] = std::string(64, '0');
                if (fault == 1)
                    terminal["input_frames"] = 2;
                if (fault == 2)
                    terminal["input_points"] = 2;
                if (fault == 3)
                    terminal["descriptor_sha256"] = std::string(64, '0');
                if (fault == 4)
                    terminal["completion_challenge"] = std::string(32, 'z');
                if (fault == 5)
                    terminal["session_id"] = std::string(32, '0');
                if (fault == 6)
                    terminal["map_epoch"] = 43;
                if (fault == 7)
                    terminal["taxonomy_sha256"] = std::string(64, '0');
                if (fault == 8)
                    terminal["extra"] = true;
                auto payload = terminal.dump();
                if (fault == 9)
                    payload.insert(1, "\"kind\":\"input_complete\",");
                if (fault == 10)
                    payload += std::string(1, '\0');
                auto wire = envelope(fault == 11 ? 33 : 35, payload);
                if (fault == 12)
                    wire = envelope(35, std::string(2049, 'x')).substr(0, 12);
                if (fault == 13)
                    wire.resize(11);
                if (fault == 14)
                    wire.pop_back();
                if (fault == 15)
                    wire += "x";
                raw_write(c.b, wire);
                if (fault != 16)
                    check(::shutdown(c.b, SHUT_WR) == 0, "raw terminal half close");
                rejects(
                    [&] { c.authority->finish_input(1, 1, deadline(fault == 16 ? 30 : 1000)); });
                check(c.authority->failed(), "malformed terminal permanently poisons");
            });
        for (unsigned fault = 0; fault < 9; ++fault)
            run_case("peer_ACK_fault_" + std::to_string(fault), [&] {
                Connection c(p4);
                c.ready();
                feed(c);
                auto authority = std::async(std::launch::async, [&] {
                    auto terminal = Json::parse(raw_message(c.a, 35));
                    raw_eof(c.a);
                    terminal["kind"] = "input_complete_ack";
                    terminal["acknowledgement_challenge"] = std::string(32, 'f');
                    if (fault == 0)
                        terminal["input_sha256"] = std::string(64, '0');
                    if (fault == 1)
                        terminal["completion_challenge"] = std::string(32, 'a');
                    if (fault == 2)
                        terminal["acknowledgement_challenge"] = std::string(32, 'z');
                    if (fault == 3)
                        terminal.erase("acknowledgement_challenge");
                    if (fault == 4)
                        terminal["input_frames"] = 99;
                    auto wire = envelope(fault == 5 ? 34 : 36, terminal.dump());
                    if (fault == 6)
                        wire.pop_back();
                    if (fault == 7)
                        wire += "x";
                    raw_write(c.a, wire);
                    if (fault != 8)
                        check(::shutdown(c.a, SHUT_WR) == 0, "raw ACK half close");
                });
                rejects([&] { c.peer->finish_input(deadline(fault == 8 ? 50 : 1000)); });
                authority.get();
                check(c.peer->failed(), "invalid ACK/EOF poisons producer");
            });
        for (bool authority : {false, true})
            run_case(std::string("expired_terminal_deadline_") + (authority ? "authority" : "peer"),
                     [&] {
                         Connection c(p4);
                         c.ready();
                         feed(c);
                         if (authority)
                             rejects([&] { c.authority->finish_input(1, 1, Clock::now()); },
                                     "deadline");
                         else
                             rejects([&] { c.peer->finish_input(Clock::now()); }, "deadline");
                         check(authority ? c.authority->failed() : c.peer->failed(),
                               "expired deadline poisons");
                     });
        check(target_queries == 0, "no GPU target query");
        std::cout << Json{{"status", "PASS"},
                          {"checks", checks.load()},
                          {"rejections", rejections},
                          {"cases", cases},
                          {"gpu_target_queries", target_queries},
                          {"backend", "opaque CPU fixture; no neural inference or GPU execution"},
                          {"limitations",
                           {"AF_UNIX sockets only; no TCP network qualification",
                            "No actual TensorRT or scientific quality claim"}}}
                         .dump(2)
                  << '\n';
        return 0;
    } catch (const std::exception &error) {
        std::cerr << Json{{"status", "FAIL"},
                          {"case", current_case},
                          {"checks", checks.load()},
                          {"rejections", rejections},
                          {"error", error.what()}}
                         .dump(2)
                  << '\n';
        return 1;
    }
}
