// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Tuomas Pietilä
// Actual binary-v3 socket I/O with an opaque CPU backend; no neural/GPU claim.
#include "generic_fixture_backend.hpp"
#include "synrex/generic_codec.hpp"
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
        auto pattern = (parent / "socket-fixture-XXXXXX").string();
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
                        const CpuAdmittedPackage *expected = nullptr) {
        int sockets[2];
        check(::socketpair(AF_UNIX, SOCK_STREAM, 0, sockets) == 0, "socketpair");
        a = sockets[0];
        b = sockets[1];
        authority = std::make_unique<AuthorityTransport>(WireProtocol::binary_v3, a, package,
                                                         limits, session_id, 42);
        peer = std::make_unique<PeerTransport>(
            WireProtocol::binary_v3, b, expected ? *expected : package, limits, session_id, 42);
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
Frame frame(const Descriptor &descriptor, unsigned points = 2) {
    Frame value;
    value.identity = {descriptor.sha256(), session_id, 42};
    value.sensor_id = "sensor";
    value.stream_id = "stream";
    value.coordinate_frame = "world";
    value.crs = "EPSG:3067";
    value.sequence = UINT64_MAX;
    value.frame_id = UINT64_MAX - 1;
    value.frame_start_ns = 1700000000000000000ULL;
    value.frame_end_ns = value.frame_start_ns + 1000000000ULL;
    for (unsigned i = 0; i < points; ++i)
        value.points.push_back(
            {UINT64_MAX - i,
             value.frame_start_ns + i,
             {std::nextafter(1.0, 2.0), -0.0, std::nextafter(10000000.0, 10000001.0)},
             std::uint16_t(65535),
             std::uint8_t(15),
             std::uint8_t(15)});
    value.accepted_steady_ns = UINT64_MAX;
    value.submitted_steady_ns = UINT64_MAX;
    return value;
}
Classification classification(const Descriptor &descriptor, unsigned points = 2) {
    Classification value;
    value.frame = frame(descriptor, points);
    const auto &classes = descriptor.taxonomy().classes();
    for (unsigned i = 0; i < points; ++i) {
        const auto id = static_cast<std::uint8_t>(i % 2 ? 0 : classes.size() - 1);
        value.semantic_ids.push_back(id);
        value.las_codes.push_back(classes.at(id).las_code);
    }
    value.timings = {{"remote_seconds", 0.125}};
    return value;
}
void equal_frame(const Frame &expected, const Frame &actual) {
    check(actual.identity.descriptor_sha256 == expected.identity.descriptor_sha256 &&
              actual.identity.session_id == expected.identity.session_id &&
              actual.identity.map_epoch == expected.identity.map_epoch,
          "session identity preserved");
    check(actual.sensor_id == expected.sensor_id && actual.stream_id == expected.stream_id &&
              actual.coordinate_frame == expected.coordinate_frame && actual.crs == expected.crs &&
              actual.sequence == expected.sequence && actual.frame_id == expected.frame_id &&
              actual.frame_start_ns == expected.frame_start_ns &&
              actual.frame_end_ns == expected.frame_end_ns &&
              actual.points.size() == expected.points.size(),
          "frame metadata preserved");
    for (std::size_t i = 0; i < actual.points.size(); ++i) {
        const auto &a = actual.points[i], &b = expected.points[i];
        check(a.id == b.id && a.timestamp_ns == b.timestamp_ns && a.intensity == b.intensity &&
                  a.return_number == b.return_number && a.return_count == b.return_count &&
                  std::memcmp(a.xyz.data(), b.xyz.data(), sizeof(double) * 3) == 0,
              "source IDs, attributes and exact f64 bits preserved");
    }
    check(actual.accepted_steady_ns == 0 && actual.submitted_steady_ns == 0,
          "received host-local timestamps reset");
}
void equal_classification(const Classification &expected, const Classification &actual) {
    equal_frame(expected.frame, actual.frame);
    check(actual.semantic_ids == expected.semantic_ids && actual.las_codes == expected.las_codes &&
              actual.timings == expected.timings,
          "semantic IDs, taxonomy projection and remote measurements preserved");
}
void raw_write(int socket, const std::string &bytes, std::size_t chunk = SIZE_MAX,
               unsigned delay_ms = 0) {
    std::size_t offset{};
    while (offset < bytes.size()) {
        const auto sent = ::send(socket, bytes.data() + offset,
                                 std::min(chunk, bytes.size() - offset), MSG_NOSIGNAL);
        check(sent > 0, "raw socket write");
        offset += static_cast<std::size_t>(sent);
        if (delay_ms)
            std::this_thread::sleep_for(std::chrono::milliseconds(delay_ms));
    }
}
std::string raw_read(int socket, std::size_t size) {
    std::string result;
    std::array<char, 7> chunk{};
    while (result.size() < size) {
        const auto count =
            ::recv(socket, chunk.data(), std::min(chunk.size(), size - result.size()), 0);
        check(count > 0, "raw socket read");
        result.append(chunk.data(), static_cast<std::size_t>(count));
    }
    return result;
}
void set_length(std::string &bytes, std::uint32_t payload) {
    for (unsigned i = 0; i < 4; ++i)
        bytes[8 + i] = static_cast<char>(payload >> (8 * (3 - i)));
}
template <class Endpoint> void poisoned(Endpoint &endpoint) {
    check(endpoint.failed(), "I/O failure permanently poisons endpoint");
    rejects([&] { endpoint.receive_frame(deadline()); }, "permanently failed");
    rejects([&] { endpoint.receive_classification(deadline()); }, "permanently failed");
}
template <class F> void run_case(const std::string &name, F operation) {
    current_case = name;
    fb::reset();
    operation();
    check(fb::live == 0 && fb::constructions == fb::destructions, "CPU fixture lifetime balanced");
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
        const CpuAdmittedPackage p4(fixture(temporary.path, base, 4));
        const auto unattached = Descriptor::from_package(p4, limits, session_id, 42);
        for (const bool admitted : {false, true})
            for (const bool authority : {false, true})
                for (unsigned method = 0; method < 4; ++method)
                    run_case("before_" + std::string(admitted ? "ready" : "descriptor") +
                                 (authority ? "_authority_" : "_peer_") + std::to_string(method),
                             [&] {
                                 Connection c(p4);
                                 if (admitted)
                                     c.admit();
                                 const auto input = frame(unattached);
                                 const auto output = classification(unattached);
                                 const auto started = Clock::now();
                                 auto attempt = [&](auto &endpoint) {
                                     rejects([&] {
                                         switch (method) {
                                         case 0:
                                             endpoint.send_frame(input, deadline(800));
                                             break;
                                         case 1:
                                             (void)endpoint.receive_frame(deadline(800));
                                             break;
                                         case 2:
                                             endpoint.send_classification(output, deadline(800));
                                             break;
                                         default:
                                             (void)endpoint.receive_classification(deadline(800));
                                             break;
                                         }
                                     });
                                     poisoned(endpoint);
                                 };
                                 if (authority)
                                     attempt(*c.authority);
                                 else
                                     attempt(*c.peer);
                                 check(Clock::now() - started < std::chrono::milliseconds(250),
                                       "startup gate rejects before socket wait");
                                 check(fb::init_calls == 0,
                                       "startup refusal never initializes backend");
                             });
        for (const unsigned count : {4, 7, 256})
            run_case("typed_bidirectional_K" + std::to_string(count), [&] {
                const CpuAdmittedPackage package(fixture(temporary.path, base, count));
                Connection c(package);
                c.ready();
                const auto &descriptor = c.authority->descriptor();
                const auto input = frame(descriptor);
                const auto output = classification(descriptor);
                check(output.semantic_ids.front() == count - 1, "highest semantic ID fixture");
                auto peer_receive = std::async(std::launch::async,
                                               [&] { return c.peer->receive_frame(deadline()); });
                c.authority->send_frame(input, deadline());
                equal_frame(input, peer_receive.get());
                c.peer->send_classification(output, deadline());
                equal_classification(output, c.authority->receive_classification(deadline()));
                c.peer->send_frame(input, deadline());
                equal_frame(input, c.authority->receive_frame(deadline()));
                c.authority->send_classification(output, deadline());
                equal_classification(output, c.peer->receive_classification(deadline()));
                if (count == 7)
                    check(descriptor.taxonomy().classes().at(6).las_code == 201 &&
                              descriptor.taxonomy().classes().at(6).key == "private_6",
                          "non-FRACTAL seven-class meanings preserved");
                check(!c.authority->failed() && !c.peer->failed(), "successful I/O remains ready");
            });
        for (const bool authority : {false, true})
            for (const bool output : {false, true})
                run_case(std::string("fragmented_receive_") + (authority ? "authority" : "peer") +
                             (output ? "_classification" : "_frame"),
                         [&] {
                             Connection c(p4);
                             c.ready();
                             const auto input = frame(c.authority->descriptor());
                             const auto labels = classification(c.authority->descriptor());
                             const auto bytes =
                                 output ? encode_classification(c.authority->descriptor(), labels)
                                        : encode_frame(c.authority->descriptor(), input);
                             auto writer = std::async(std::launch::async, [&] {
                                 raw_write(authority ? c.b : c.a, bytes, 3, 1);
                             });
                             auto receive = [&](auto &endpoint) {
                                 if (output)
                                     equal_classification(
                                         labels, endpoint.receive_classification(deadline()));
                                 else
                                     equal_frame(input, endpoint.receive_frame(deadline()));
                             };
                             if (authority)
                                 receive(*c.authority);
                             else
                                 receive(*c.peer);
                             writer.get();
                         });
        for (const bool authority : {false, true})
            for (const bool output : {false, true})
                run_case(std::string("fragmented_send_") + (authority ? "authority" : "peer") +
                             (output ? "_classification" : "_frame"),
                         [&] {
                             Connection c(p4);
                             c.ready();
                             const int buffer = 1024;
                             const int socket = authority ? c.a : c.b;
                             check(::setsockopt(socket, SOL_SOCKET, SO_SNDBUF, &buffer,
                                                sizeof(buffer)) == 0,
                                   "small socket send buffer");
                             const auto input = frame(c.authority->descriptor(), 900);
                             const auto labels = classification(c.authority->descriptor(), 900);
                             const auto expected =
                                 output ? encode_classification(c.authority->descriptor(), labels)
                                        : encode_frame(c.authority->descriptor(), input);
                             auto reader = std::async(std::launch::async, [&] {
                                 return raw_read(authority ? c.b : c.a, expected.size());
                             });
                             auto send = [&](auto &endpoint) {
                                 if (output)
                                     endpoint.send_classification(labels, deadline());
                                 else
                                     endpoint.send_frame(input, deadline());
                             };
                             if (authority)
                                 send(*c.authority);
                             else
                                 send(*c.peer);
                             const auto bytes = reader.get();
                             check(bytes == expected,
                                   "production socket sender emits exact codec envelope");
                             if (output)
                                 equal_classification(
                                     labels,
                                     decode_classification(c.authority->descriptor(), bytes));
                             else
                                 equal_frame(input, decode_frame(c.authority->descriptor(), bytes));
                         });
        for (unsigned mutation = 0; mutation < 10; ++mutation)
            run_case("fixed_header_only_rejection_" + std::to_string(mutation), [&] {
                Connection c(p4);
                c.ready();
                auto bytes =
                    encode_frame(c.authority->descriptor(), frame(c.authority->descriptor()))
                        .substr(0, 12);
                switch (mutation) {
                case 0:
                    bytes[0] = 'X';
                    break;
                case 1:
                    bytes[4] = 2;
                    break; // Explicit legacy wire-v2 never falls back.
                case 2:
                    bytes[5] = 17;
                    break;
                case 3:
                    bytes[5] = 5;
                    break; // Startup control cannot become a frame.
                case 4:
                    bytes[6] = 1;
                    break;
                case 5:
                    bytes[7] = 1;
                    break;
                case 6:
                    set_length(bytes, 0);
                    break;
                case 7:
                    set_length(bytes, static_cast<std::uint32_t>(limits.max_frame_bytes - 11));
                    break;
                case 8:
                    set_length(bytes, UINT32_MAX);
                    break;
                default:
                    bytes.replace(0, 4, "SRX2");
                    break;
                }
                raw_write(c.b, bytes);
                const auto started = Clock::now();
                rejects([&] { c.authority->receive_frame(deadline(800)); },
                        mutation >= 6 && mutation <= 8 ? "bound" : "");
                check(Clock::now() - started < std::chrono::milliseconds(250),
                      "bad fixed header rejects without waiting for any payload");
                poisoned(*c.authority);
            });
        for (unsigned field = 0; field < 3; ++field)
            run_case("incoming_identity_mismatch_" + std::to_string(field), [&] {
                Connection c(p4);
                c.ready();
                auto bytes =
                    encode_frame(c.authority->descriptor(), frame(c.authority->descriptor()));
                bytes[field == 0 ? 12 : field == 1 ? 76 : 115] ^= 1;
                raw_write(c.b, bytes);
                rejects([&] { c.authority->receive_frame(deadline()); }, "mismatch");
                poisoned(*c.authority);
            });
        for (const bool projection : {false, true})
            run_case(std::string("incoming_invalid_") + (projection ? "projection" : "taxonomy_id"),
                     [&] {
                         Connection c(p4);
                         c.ready();
                         const auto output = classification(c.authority->descriptor());
                         auto bytes = encode_classification(c.authority->descriptor(), output);
                         const auto labels =
                             encode_frame(c.authority->descriptor(), output.frame).size();
                         bytes[labels + (projection ? 1 : 0)] ^= projection ? 1 : 252;
                         raw_write(c.a, bytes);
                         rejects([&] { c.peer->receive_classification(deadline()); },
                                 projection ? "projection" : "taxonomy");
                         poisoned(*c.peer);
                     });
        for (const bool output : {false, true})
            for (const std::size_t prefix : {std::size_t(0), std::size_t(1), std::size_t(11),
                                             std::size_t(12), std::size_t(13), SIZE_MAX})
                run_case(std::string("EOF_") + (output ? "classification_" : "frame_") +
                             std::to_string(prefix),
                         [&] {
                             Connection c(p4);
                             c.ready();
                             auto bytes = output ? encode_classification(
                                                       c.authority->descriptor(),
                                                       classification(c.authority->descriptor()))
                                                 : encode_frame(c.authority->descriptor(),
                                                                frame(c.authority->descriptor()));
                             bytes.resize(prefix == SIZE_MAX ? bytes.size() - 1 : prefix);
                             raw_write(c.b, bytes);
                             check(::shutdown(c.b, SHUT_WR) == 0, "raw EOF");
                             if (output)
                                 rejects([&] { c.authority->receive_classification(deadline()); },
                                         prefix == 0 ? "EOF" : "truncated");
                             else
                                 rejects([&] { c.authority->receive_frame(deadline()); },
                                         prefix == 0 ? "EOF" : "truncated");
                             poisoned(*c.authority);
                         });
        run_case("complete_record_then_EOF_never_terminal_success", [&] {
            Connection c(p4);
            c.ready();
            const auto input = frame(c.authority->descriptor());
            raw_write(c.b, encode_frame(c.authority->descriptor(), input));
            check(::shutdown(c.b, SHUT_WR) == 0, "EOF after complete record");
            equal_frame(input, c.authority->receive_frame(deadline()));
            rejects([&] { c.authority->receive_frame(deadline()); }, "EOF");
            poisoned(*c.authority);
        });
        run_case("absolute_receive_deadline_dribble", [&] {
            Connection c(p4);
            c.ready();
            const auto bytes =
                encode_frame(c.authority->descriptor(), frame(c.authority->descriptor()));
            auto writer = std::async(std::launch::async, [&] {
                for (const char byte : bytes) {
                    if (::send(c.b, &byte, 1, MSG_NOSIGNAL) != 1)
                        break;
                    std::this_thread::sleep_for(std::chrono::milliseconds(8));
                }
            });
            const auto started = Clock::now();
            rejects([&] { c.authority->receive_frame(started + std::chrono::milliseconds(45)); },
                    "deadline");
            const auto elapsed = Clock::now() - started;
            check(elapsed >= std::chrono::milliseconds(25) &&
                      elapsed < std::chrono::milliseconds(250),
                  "dribble cannot extend absolute deadline");
            poisoned(*c.authority);
            writer.get();
        });
        for (const bool authority : {false, true})
            run_case(std::string("absolute_send_deadline_backpressure_") +
                         (authority ? "authority" : "peer"),
                     [&] {
                         Connection c(p4);
                         c.ready();
                         const int buffer = 1024;
                         check(::setsockopt(authority ? c.a : c.b, SOL_SOCKET, SO_SNDBUF, &buffer,
                                            sizeof(buffer)) == 0,
                               "small socket send buffer");
                         const auto input = frame(c.authority->descriptor(), 900);
                         const auto started = Clock::now();
                         auto attempt = [&](auto &endpoint) {
                             rejects(
                                 [&] {
                                     endpoint.send_frame(input,
                                                         started + std::chrono::milliseconds(40));
                                 },
                                 "deadline");
                             poisoned(endpoint);
                         };
                         if (authority)
                             attempt(*c.authority);
                         else
                             attempt(*c.peer);
                         check(Clock::now() - started < std::chrono::milliseconds(250),
                               "send timeout remains absolute");
                     });
        for (unsigned fault = 0; fault < 4; ++fault)
            run_case("outgoing_validation_and_poison_" + std::to_string(fault), [&] {
                Connection c(p4);
                c.ready();
                auto input = frame(c.authority->descriptor());
                auto output = classification(c.authority->descriptor());
                if (fault == 0)
                    input.identity.map_epoch ^= 1;
                if (fault == 1)
                    output.semantic_ids.front() = 255;
                if (fault == 2)
                    output.las_codes.front() ^= 1;
                if (fault == 3)
                    output.semantic_ids.pop_back();
                if (fault == 0)
                    rejects([&] { c.authority->send_frame(input, deadline()); }, "mismatch");
                else
                    rejects([&] { c.authority->send_classification(output, deadline()); });
                poisoned(*c.authority);
                rejects([&] { c.authority->send_frame(input, deadline()); }, "permanently failed");
                rejects([&] { c.authority->send_classification(output, deadline()); },
                        "permanently failed");
            });
        run_case("broken_peer_write_poison", [&] {
            Connection c(p4);
            c.ready();
            const auto input = frame(c.authority->descriptor());
            check(::shutdown(c.b, SHUT_RDWR) == 0, "break peer");
            rejects([&] { c.authority->send_frame(input, deadline()); });
            poisoned(*c.authority);
            rejects([&] { c.authority->send_frame(input, deadline()); }, "permanently failed");
        });
        run_case("same_K_projection_and_taxonomy_id_changed_meanings_reject", [&] {
            const CpuAdmittedPackage changed(fixture(temporary.path, base, 4, true));
            Connection c(p4, &changed);
            rejects([&] { c.admit(); });
            check(c.authority->failed() && c.peer->failed(),
                  "mismatched admission poisons both endpoints");
            check(fb::init_calls == 0, "taxonomy mismatch rejects before CPU backend");
        });
        check(target_queries == 0, "no GPU target query");
        std::cout << Json{{"status", "PASS"},
                          {"checks", checks.load()},
                          {"rejections", rejections},
                          {"cases", cases},
                          {"gpu_target_queries", target_queries},
                          {"backend", "opaque CPU fixture; no neural inference or GPU execution"},
                          {"limitations",
                           {"AF_UNIX stream sockets only; no TCP network qualification",
                            "No terminal-success protocol is implemented or claimed"}}}
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
