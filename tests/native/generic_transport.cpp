// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Tuomas Pietilä
// Real CPU sockets; synthetic model bytes are opaque and never executed.
#include "synrex/generic_transport.hpp"
#include "generic_fixture_backend.hpp"
#include "synrex/generic_session.hpp"
#include <arpa/inet.h>
#include <atomic>
#include <fstream>
#include <future>
#include <iostream>
#include <sys/socket.h>
#include <thread>
#include <unistd.h>

namespace {
using namespace synrex;
using namespace synrex::generic;
const auto v3 = WireProtocol::binary_v3;
const std::string session(32, 'b');
unsigned checks{}, rejections{}, fixture_index{};
std::atomic<unsigned> target_queries{};
std::vector<std::string> cases;
std::string current_case;
void check(bool okay, const std::string &message) {
    ++checks;
    if (!okay)
        throw std::runtime_error(message);
}
template <class F> void rejects(F operation, const std::string &text = "") {
    try {
        operation();
    } catch (const std::exception &error) {
        check(text.empty() || std::string(error.what()).find(text) != std::string::npos,
              "unexpected rejection: " + std::string(error.what()) + "; expected " + text);
        ++rejections;
        return;
    }
    throw std::runtime_error("unexpected acceptance: " + text);
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
        auto pattern = (parent / "transport-fixture-XXXXXX").string();
        const auto created = ::mkdtemp(pattern.data());
        check(created != nullptr, "temporary fixture");
        path = created;
    }
    ~Temporary() {
        std::error_code error;
        fs::remove_all(path, error);
    }
};
Json taxonomy(std::size_t count) {
    Json classes = Json::array();
    const unsigned legacy[] = {1, 2, 200, 6, 9, 17, 201};
    for (std::size_t i = 0; i < count; ++i)
        classes.push_back({{"semantic_id", i},
                           {"key", "private_" + std::to_string(i)},
                           {"name", "Private meaning " + std::to_string(i)},
                           {"las_code", count == 7 ? legacy[i] : 42}});
    return {
        {"schema", "synrex.taxonomy.v1"},
        {"taxonomy_id", "example.reused.v1"},
        {"namespace", "example.cpu"},
        {"classes", classes},
        {"provenance", {{"source", "CPU fixture"}, {"author", "SynRex tests"}, {"license", "MIT"}}},
        {"unknown_semantic_id", nullptr}};
}
fs::path package_path(const fs::path &parent, const Json &base, const Json &tax) {
    const auto path = parent / ("package-" + std::to_string(++fixture_index));
    fs::create_directory(path);
    auto contract = base;
    const Taxonomy admitted(tax.dump());
    Json projection = Json::array();
    for (const auto &entry : admitted.classes())
        projection.push_back(entry.las_code);
    contract["las_classes"] = projection;
    contract["output"]["shape"][1] = admitted.classes().size();
    write(path / "model.engine", "Opaque CPU fixture: never executed");
    write(path / "MODEL_LICENSE.txt", "MIT CPU fixture");
    write(path / "taxonomy.json", tax.dump());
    write(path / "manifest.json",
          Json{{"schema", "synrex.srxmodel.v3"},
               {"contract", contract},
               {"artifacts",
                {{"engine", {{"path", "model.engine"}, {"sha256", sha256(path / "model.engine")}}},
                 {"license",
                  {{"path", "MODEL_LICENSE.txt"}, {"sha256", sha256(path / "MODEL_LICENSE.txt")}}},
                 {"taxonomy", {{"path", "taxonomy.json"}, {"sha256", admitted.sha256()}}}}},
               {"target",
                {{"architecture", "CPU-test-only"},
                 {"gpu_name", "NO GPU"},
                 {"sm", 89},
                 {"tensorrt", "10.test"},
                 {"cuda_runtime", 12010}}}}
              .dump());
    return path;
}
struct Sockets {
    int a{-1}, b{-1};
    Sockets() {
        int sockets[2];
        if (::socketpair(AF_UNIX, SOCK_STREAM, 0, sockets))
            throw std::runtime_error("socketpair");
        a = sockets[0];
        b = sockets[1];
    }
    ~Sockets() {
        ::close(a);
        ::close(b);
    }
};
void raw_send(int socket, std::string_view bytes, unsigned fragment = 0) {
    std::size_t offset{};
    while (offset < bytes.size()) {
        const auto count = fragment ? std::min<std::size_t>(fragment, bytes.size() - offset)
                                    : bytes.size() - offset;
        const auto sent = ::send(socket, bytes.data() + offset, count, MSG_NOSIGNAL);
        if (sent <= 0)
            throw std::runtime_error("raw send");
        offset += static_cast<std::size_t>(sent);
    }
}
std::string raw_read(int socket, std::size_t length) {
    std::string value(length, '\0');
    std::size_t offset{};
    while (offset < length) {
        const auto got = ::recv(socket, value.data() + offset, length - offset, 0);
        if (got <= 0)
            throw std::runtime_error("raw EOF");
        offset += static_cast<std::size_t>(got);
    }
    return value;
}
std::string envelope(unsigned kind, std::string_view bytes, std::uint32_t size_override = 0) {
    const auto size = size_override ? size_override : static_cast<std::uint32_t>(bytes.size());
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
std::string raw_message(int socket, unsigned kind) {
    const auto header = raw_read(socket, 12);
    if (static_cast<unsigned char>(header[5]) != kind)
        throw std::runtime_error("raw kind");
    std::uint32_t size{};
    for (unsigned i = 8; i < 12; ++i)
        size = (size << 8) | static_cast<unsigned char>(header[i]);
    if (!size || size > 65536)
        throw std::runtime_error("raw length bound");
    return raw_read(socket, size);
}
std::string hello(Limits limits = {}) {
    return Json{{"schema", "synrex.session.hello.v3"},
                {"limits",
                 {{"max_frame_points", limits.max_frame_points},
                  {"max_frame_bytes", limits.max_frame_bytes},
                  {"startup_timeout_ms", limits.startup_timeout_ms}}}}
        .dump();
}
// A real bidirectional stream proxy fragments every write. All threads are joined.
struct FragmentProxy {
    Sockets left, right;
    std::atomic<std::uint64_t> fragments{};
    std::thread forward, backward;
    FragmentProxy()
        : forward([this] { relay(left.b, right.a); }),
          backward([this] { relay(right.a, left.b); }) {}
    void relay(int source, int destination) {
        char bytes[257];
        for (;;) {
            const auto count = ::recv(source, bytes, sizeof(bytes), 0);
            if (count <= 0)
                break;
            for (std::size_t offset = 0; offset < static_cast<std::size_t>(count);) {
                const auto n = std::min<std::size_t>(7, static_cast<std::size_t>(count) - offset);
                const auto sent = ::send(destination, bytes + offset, n, MSG_NOSIGNAL);
                if (sent <= 0) {
                    ::shutdown(destination, SHUT_WR);
                    return;
                }
                offset += static_cast<std::size_t>(sent);
                ++fragments;
            }
        }
        ::shutdown(destination, SHUT_WR);
    }
    ~FragmentProxy() {
        ::shutdown(left.a, SHUT_RDWR);
        ::shutdown(right.b, SHUT_RDWR);
        forward.join();
        backward.join();
    }
};
void successful(int authority_socket, int peer_socket, const CpuAdmittedPackage &package,
                std::size_t count, bool size_fault = false, bool identity_fault = false) {
    AuthorityTransport authority(v3, authority_socket, package, {100, 2000, 1000}, session, 9);
    PeerTransport peer(v3, peer_socket, package, {200, 1000, 2000}, session, 9);
    check(!authority.initialization_permission(), "no initialization permission before metadata");
    auto remote = std::async(std::launch::async, [&] {
        peer.metadata_admission();
        peer.acknowledge_execution_ready();
    });
    authority.metadata_admission();
    check(authority.initialization_permission(), "metadata admits initialization");
    const auto initialized = authority.create_session();
    check(bool(initialized), "factory-owned initialization completed before ready");
    remote.get();
    const auto &d = authority.descriptor();
    check(peer.descriptor().bytes() == d.bytes(), "same complete descriptor");
    check(d.taxonomy().classes().size() == count &&
              d.taxonomy().classes().back().semantic_id == count - 1,
          "complete taxonomy including semantic ID255");
    const auto limits = authority.negotiated_limits();
    check(limits.max_frame_points == 100 && limits.max_frame_bytes == 1000 &&
              limits.startup_timeout_ms == 1000,
          "actual minimum limits negotiated");
    check(peer.negotiated_limits().max_frame_points == limits.max_frame_points,
          "peer retains negotiated limits");
    authority.authorize_frame(d.sha256(), session, 9, 100, 1000);
    peer.authorize_frame(d.sha256(), session, 9, 100, 1000);
    if (size_fault) {
        rejects([&] { authority.authorize_frame(d.sha256(), session, 9, 101, 1000); }, "limits");
        rejects([&] { peer.authorize_frame(d.sha256(), session, 9, 100, 1001); }, "limits");
        check(authority.failed() && peer.failed(), "frame limits poison both endpoints");
        rejects([&] { authority.initialization_permission(); }, "permanently failed");
        rejects([&] { peer.descriptor(); }, "permanently failed");
    } else if (identity_fault) {
        rejects([&] { authority.authorize_frame(d.sha256(), session, 10, 1, 1); }, "identity");
        rejects([&] { peer.authorize_frame(std::string(64, '0'), session, 9, 1, 1); }, "identity");
    }
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
        const auto p4 = package_path(temporary.path, base, taxonomy(4));
        const CpuAdmittedPackage package(p4);
        for (const auto count : {2u, 4u, 7u, 256u}) {
            current_case = "fragmented_K" + std::to_string(count);
            const CpuAdmittedPackage p(package_path(temporary.path, base, taxonomy(count)));
            FragmentProxy proxy;
            successful(proxy.left.a, proxy.right.b, p, count, count == 4, count == 2);
            check(proxy.fragments > 10, "actual fragmented socket writes");
            cases.push_back("fragmented_socketpair_K" + std::to_string(count));
        }
        // Real TCP loopback, independent of socketpair behavior.
        {
            current_case = "TCP_loopback";
            const int listener = ::socket(AF_INET, SOCK_STREAM, 0);
            sockaddr_in address{};
            address.sin_family = AF_INET;
            address.sin_addr.s_addr = htonl(INADDR_LOOPBACK);
            check(listener >= 0 &&
                      ::bind(listener, reinterpret_cast<sockaddr *>(&address), sizeof(address)) ==
                          0 &&
                      ::listen(listener, 1) == 0,
                  "loopback listener");
            socklen_t size = sizeof(address);
            check(::getsockname(listener, reinterpret_cast<sockaddr *>(&address), &size) == 0,
                  "loopback port");
            const int client = ::socket(AF_INET, SOCK_STREAM, 0);
            check(client >= 0 && ::connect(client, reinterpret_cast<sockaddr *>(&address),
                                           sizeof(address)) == 0,
                  "loopback connect");
            const int server = ::accept(listener, nullptr, nullptr);
            check(server >= 0, "loopback accept");
            successful(server, client, package, 4);
            ::close(client);
            ::close(server);
            ::close(listener);
            cases.push_back("TCP_loopback");
        }
        // Same K, reused taxonomy ID, exact LAS projection; changed meanings/order reject.
        for (const auto count : {4u, 7u})
            for (const bool reorder : {false, true}) {
                current_case = "taxonomy_mismatch";
                auto changed = taxonomy(count);
                if (reorder) {
                    std::swap(changed["classes"][0]["name"], changed["classes"][1]["name"]);
                    std::swap(changed["classes"][0]["key"], changed["classes"][1]["key"]);
                } else
                    changed["classes"][0]["name"] = "Changed meaning, unchanged projection";
                const CpuAdmittedPackage authoritative(
                    package_path(temporary.path, base, taxonomy(count)));
                const CpuAdmittedPackage expected(package_path(temporary.path, base, changed));
                Sockets sockets;
                AuthorityTransport authority(v3, sockets.a, authoritative, {}, session, 1);
                PeerTransport peer(v3, sockets.b, expected, {}, session, 1);
                auto remote = std::async(std::launch::async, [&] {
                    try {
                        peer.metadata_admission();
                        return false;
                    } catch (...) {
                        return peer.failed();
                    }
                });
                rejects([&] { authority.metadata_admission(); }, "EOF");
                check(remote.get() && authority.failed(), "taxonomy mismatch poisons real sockets");
                cases.push_back("sameK_sameProjection_reusedID_" + std::to_string(count) +
                                (reorder ? "_order" : "_meaning"));
            }
        // No frame authorization at fresh/admitted/initializing stages.
        for (unsigned stage = 0; stage < 2; ++stage) {
            current_case = "early_frame";
            Sockets sockets;
            AuthorityTransport authority(v3, sockets.a, package, {}, session, 1);
            PeerTransport peer(v3, sockets.b, package, {}, session, 1);
            if (stage) {
                auto remote = std::async(std::launch::async, [&] { peer.metadata_admission(); });
                authority.metadata_admission();
                remote.get();
            }
            rejects([&] { authority.authorize_frame(std::string(64, '0'), session, 1, 1, 1); },
                    "before execution-ready");
            check(authority.failed(), "early frame poisons");
            rejects([&] { authority.create_session(); }, "permanently failed");
            rejects([&] { peer.authorize_frame(std::string(64, '0'), session, 1, 1, 1); },
                    "before execution-ready");
        }
        cases.push_back("no_frames_before_readiness_fresh_and_admitted");
        // Malformed/out-of-order envelopes rejected from the fixed header alone.
        for (const auto &fault :
             {"v2", "unknown_version", "reserved", "kind", "huge", "zero", "truncated_header",
              "truncated_body", "duplicate_key", "unknown_capability"}) {
            current_case = fault;
            Sockets sockets;
            AuthorityTransport authority(v3, sockets.a, package, {}, session, 1);
            std::string bytes = envelope(1, hello());
            const std::string name(fault);
            if (name == "v2")
                bytes[3] = '2';
            if (name == "unknown_version")
                bytes[4] = 4;
            if (name == "reserved")
                bytes[7] = 1;
            if (name == "kind")
                bytes[5] = 3;
            if (name == "huge")
                bytes = envelope(1, "", UINT32_MAX);
            if (name == "zero")
                bytes = envelope(1, "");
            if (name == "truncated_header")
                bytes.resize(6);
            if (name == "truncated_body")
                bytes.resize(14);
            if (name == "duplicate_key")
                bytes = envelope(1, "{\"schema\":\"bad\",\"sch\\u0065ma\":\"synrex.session.hello."
                                    "v3\",\"limits\":{}}");
            if (name == "unknown_capability") {
                auto value = Json::parse(hello());
                value["extra"] = 1;
                bytes = envelope(1, value.dump());
            }
            raw_send(sockets.b, bytes, 1);
            ::shutdown(sockets.b, SHUT_WR);
            rejects([&] { authority.metadata_admission(); });
            check(authority.failed(), "malformed header/body poisoned");
            rejects([&] { authority.initialization_permission(); }, "permanently failed");
            cases.push_back(name);
        }
        // Invalid peer limits are bounded JSON, rejected before descriptor emission.
        for (const auto limits : {Limits{0, 1, 1}, Limits{1, 16777217, 1}, Limits{1, 1, 120001}}) {
            current_case = "invalid_peer_limits";
            Sockets sockets;
            AuthorityTransport authority(v3, sockets.a, package, {}, session, 1);
            raw_send(sockets.b, envelope(1, hello(limits)));
            rejects([&] { authority.metadata_admission(); }, "limit");
        }
        cases.push_back("invalid_peer_limits");
        // Full metadata ACK identity; duplicate/out-of-order control and startup timeout.
        for (const auto &fault : {"ack_identity", "ack_epoch", "ack_session", "early_ready_ack",
                                  "pipelined_ready_ack", "wrong_ready_challenge", "duplicate_ack",
                                  "startup_stall", "startup_truncated", "duplicate_admission"}) {
            current_case = fault;
            Sockets sockets;
            AuthorityTransport authority(v3, sockets.a, package, {100, 1000, 40}, session, 1);
            const std::string name(fault);
            auto remote = std::async(std::launch::async, [&] {
                raw_send(sockets.b, envelope(1, hello({100, 1000, 40})), 3);
                const auto descriptor = Descriptor::parse(raw_message(sockets.b, 2));
                auto ack = Json::parse(descriptor.control("metadata_ack"));
                if (name == "ack_identity")
                    ack["descriptor_sha256"] = std::string(64, '0');
                if (name == "ack_epoch")
                    ack["map_epoch"] = 2;
                if (name == "ack_session")
                    ack["session_id"] = std::string(32, '0');
                raw_send(sockets.b, envelope(name == "early_ready_ack" ? 5 : 3, ack.dump()),
                         name == "early_ready_ack" ? 0 : 2);
                if (name == "duplicate_ack")
                    raw_send(sockets.b, envelope(3, ack.dump()));
                if (name == "pipelined_ready_ack")
                    raw_send(sockets.b, envelope(5, descriptor.control("execution_ready_ack")));
                if (name == "wrong_ready_challenge") {
                    auto ready = Json::parse(raw_message(sockets.b, 4));
                    ready["kind"] = "execution_ready_ack";
                    auto challenge = ready["readiness_challenge"].get<std::string>();
                    challenge[0] = challenge[0] == '0' ? '1' : '0';
                    ready["readiness_challenge"] = challenge;
                    raw_send(sockets.b, envelope(5, ready.dump()));
                }
                if (name == "startup_truncated") {
                    raw_send(sockets.b,
                             envelope(5, descriptor.control("execution_ready_ack")).substr(0, 15),
                             1);
                    ::shutdown(sockets.b, SHUT_WR);
                }
                if (name == "startup_stall")
                    std::this_thread::sleep_for(std::chrono::milliseconds(80));
            });
            if (name == "ack_identity" || name == "ack_epoch" || name == "ack_session" ||
                name == "early_ready_ack")
                rejects([&] { authority.metadata_admission(); });
            else {
                authority.metadata_admission();
                // Ensure this ACK was actually pre-sent before initialization/readiness.
                if (name == "pipelined_ready_ack")
                    remote.wait();
                if (name == "duplicate_admission")
                    rejects([&] { authority.metadata_admission(); }, "duplicate");
                else {
                    rejects([&] { authority.create_session(); });
                }
            }
            remote.get();
            check(authority.failed(), "control failure permanently poisoned");
            cases.push_back(name);
        }
        // Peer rejects authority-selected limits above its own supported values.
        {
            current_case = "peer_excess_limits";
            Sockets sockets;
            PeerTransport peer(v3, sockets.a, package, {100, 1000, 100}, session, 1);
            auto remote = std::async(std::launch::async, [&] {
                raw_message(sockets.b, 1);
                const auto descriptor =
                    Descriptor::from_package(package, {101, 1000, 100}, session, 1);
                raw_send(sockets.b, envelope(2, descriptor.bytes()), 5);
            });
            rejects([&] { peer.metadata_admission(); }, "supported limits");
            remote.get();
            check(peer.failed(), "excess negotiated limits poison");
            cases.push_back("peer_rejects_excess_limits");
        }
        // One real five-second metadata deadline spanning delayed fragments; no reset.
        {
            current_case = "absolute_metadata_stall";
            Sockets sockets;
            AuthorityTransport authority(v3, sockets.a, package, {}, session, 1);
            auto remote = std::async(std::launch::async, [&] {
                const auto bytes = envelope(1, hello());
                raw_send(sockets.b, bytes.substr(0, 1));
                std::this_thread::sleep_for(std::chrono::milliseconds(2600));
                raw_send(sockets.b, bytes.substr(1, 1));
                std::this_thread::sleep_for(std::chrono::milliseconds(2600));
            });
            const auto start = Startup::Clock::now();
            rejects([&] { authority.metadata_admission(); }, "absolute deadline");
            const auto elapsed =
                std::chrono::duration_cast<std::chrono::milliseconds>(Startup::Clock::now() - start)
                    .count();
            check(elapsed >= 4800 && elapsed < 6000,
                  "five-second absolute deadline does not reset on fragments");
            remote.get();
            check(authority.failed(), "metadata timeout poisons");
            cases.push_back("real_5second_fragmented_metadata_stall");
        }
        check(target_queries == 0, "no GPU target calls");
        std::cout << Json{{"status", "PASS_CPU_WIRE_V3_CONTROL_ONLY"},
                          {"checks", checks},
                          {"expected_rejections", rejections},
                          {"cases", cases},
                          {"gpu_target_calls", target_queries.load()},
                          {"gpu_initializations", 0},
                          {"frame_codec_exercised", false},
                          {"service_exercised", false}}
                         .dump()
                  << '\n';
        return 0;
    } catch (const std::exception &error) {
        std::cerr << current_case << ": " << error.what() << '\n';
        return 1;
    }
}
