// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Tuomas Pietilä
// Output completion over real CPU sockets; opaque backend is never GPU execution.
#include "synrex/generic_output.hpp"
#include "generic_fixture_backend.hpp"
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
Json full_capacity_receipt;
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
        auto pattern = (parent / "output-fixture-XXXXXX").string();
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
    frame.points = {{sequence, sequence, {x, 0, 0}, {}, {}, {}}};
    const auto id = static_cast<std::uint8_t>(descriptor.taxonomy().classes().size() - 1);
    result.semantic_ids = {id};
    result.las_codes = {descriptor.taxonomy().classes().at(id).las_code};
    return result;
}
std::string receiver_finish(OutputReceiver &receiver, Startup::Time until) {
    try {
        return receiver.finish(until).dump();
    } catch (const std::exception &error) {
        return "FAILED: " + std::string(error.what());
    }
}
void poisoned(Connection &connection) {
    check(connection.authority->failed(), "authority permanently poisoned");
    check(connection.peer->failed(), "peer permanently poisoned");
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
    check(static_cast<unsigned char>(header[5]) == kind, "raw output kind");
    std::uint32_t size{};
    for (unsigned i = 8; i < 12; ++i)
        size = (size << 8) | static_cast<unsigned char>(header[i]);
    check(size > 0 && size < SemanticMap::max_snapshot_bytes, "raw output bound");
    return raw_read(socket, size);
}
Json terminal(const Descriptor &descriptor, const Classification &result, const Json &snapshot,
              const std::string &bytes) {
    return {{"schema", "synrex.output.terminal.v3"},
            {"kind", "output_complete"},
            {"descriptor_sha256", descriptor.sha256()},
            {"session_id", session_id},
            {"map_epoch", 42},
            {"taxonomy_sha256", descriptor.taxonomy().sha256()},
            {"classification_frames", 1},
            {"classified_points", 1},
            {"classification_sha256", digest(encode_classification(descriptor, result))},
            {"snapshot_sha256", digest(bytes)},
            {"snapshot_bytes", bytes.size()},
            {"snapshot_cells", snapshot.at("cells").size()},
            {"as_of_ns", 1},
            {"completion_challenge", std::string(32, 'a')}};
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
            current_case = "real_socketpair_K" + std::to_string(count);
            fb::reset();
            const CpuAdmittedPackage package(fixture(temporary.path, base, count));
            Connection connection(package);
            connection.ready();
            OutputSender sender(*connection.authority, 1.0, 2);
            OutputReceiver receiver(*connection.peer, 1.0, 2);
            for (unsigned sequence = 1; sequence <= 5; ++sequence) {
                const auto sent =
                    classification(connection.authority->descriptor(), sequence, sequence + 0.25);
                sender.send(sent, deadline());
                const auto received = receiver.receive(deadline());
                check(received.semantic_ids == sent.semantic_ids &&
                          received.las_codes == sent.las_codes &&
                          received.frame.sequence == sequence,
                      "exact K4/nonfractal7/K256 classification including ID255");
            }
            auto remote = std::async(std::launch::async,
                                     [&] { return receiver_finish(receiver, deadline()); });
            const auto receipt = sender.finish(deadline());
            const auto peer = remote.get();
            check(receipt.dump().find("PEER_VALIDATED") != std::string::npos,
                  "authority has independently validated peer receipt");
            check(peer.find("VALIDATED_ACK_SENT") != std::string::npos,
                  "peer receipt states ACK sent");
            check(receipt.at("classification_frames") == 5 &&
                      receipt.at("classified_points") == 5 && receipt.at("snapshot_cells") == 2 &&
                      receipt.at("as_of_ns") == 5,
                  "receipt binds actual nonempty classification and bounded cell counts");
            auto peer_receipt = Json::parse(peer);
            auto authority_receipt = receipt;
            peer_receipt.erase("status");
            authority_receipt.erase("status");
            check(peer_receipt == authority_receipt,
                  "receiver and authority exact digest/count/identity agreement");
            check(!connection.authority->failed() && !connection.peer->failed(),
                  "successful bounded nonempty completion leaves no failure");
            if (count == 4) {
                rejects([&] { sender.finish(deadline()); }, "already finished");
                check(connection.authority->failed(), "duplicate finish poisons authority");
            } else if (count == 7) {
                rejects([&] { receiver.finish(deadline()); }, "already finished");
                check(connection.peer->failed(), "duplicate finish poisons peer");
            } else {
                rejects(
                    [&] {
                        sender.send(classification(connection.authority->descriptor(), 6),
                                    deadline());
                    },
                    "already finished");
                check(connection.authority->failed(), "send after terminal poisons authority");
                rejects([&] { receiver.receive(deadline()); }, "already finished");
                check(connection.peer->failed(), "receive after terminal poisons peer");
            }
            cases.push_back(current_case);
        }
        const CpuAdmittedPackage package(fixture(temporary.path, base, 4));
        const CpuAdmittedPackage package256(fixture(temporary.path, base, 256));
        {
            current_case = "actual_65536_cell_snapshot_socketpair_K256_ID255";
            fb::reset();
            Connection connection(package256);
            connection.ready();
            OutputSender sender(*connection.authority, 1, SemanticMap::max_cells);
            OutputReceiver receiver(*connection.peer, 1, SemanticMap::max_cells);
            std::string escaped_source;
            for (unsigned i = 0; i < 128; ++i)
                escaped_source += "\\\"";
            const auto started = Clock::now();
            const auto until = deadline(60000);
            for (std::uint64_t sequence = 1; sequence <= 64; ++sequence) {
                auto result = classification(connection.authority->descriptor(), sequence);
                auto &frame = result.frame;
                frame.sensor_id = frame.stream_id = frame.coordinate_frame = frame.crs =
                    escaped_source;
                frame.frame_start_ns = frame.frame_end_ns = UINT64_MAX - 64 + sequence;
                frame.points.clear();
                for (std::uint64_t i = 0; i < 1024; ++i) {
                    const auto index = (sequence - 1) * 1024 + i;
                    frame.points.push_back({UINT64_MAX - index,
                                            frame.frame_end_ns,
                                            {1e9 - static_cast<double>(index), -1e9, 1e9},
                                            65535,
                                            15,
                                            15});
                }
                result.semantic_ids.assign(1024, 255);
                result.las_codes.assign(
                    1024, connection.authority->descriptor().taxonomy().classes().at(255).las_code);
                sender.send(result, until);
                const auto received = receiver.receive(until);
                check(received.frame.points.size() == 1024 &&
                          received.semantic_ids == result.semantic_ids &&
                          received.frame.sensor_id == escaped_source,
                      "full map actual escaped-source ID255 frame admitted");
            }
            auto remote = std::async(std::launch::async, [&] { return receiver.finish(until); });
            full_capacity_receipt = sender.finish(until);
            auto peer = remote.get();
            check(full_capacity_receipt.at("classification_frames") == 64 &&
                      full_capacity_receipt.at("classified_points") == 65536 &&
                      full_capacity_receipt.at("snapshot_cells") == 65536 &&
                      full_capacity_receipt.at("as_of_ns") == UINT64_MAX &&
                      full_capacity_receipt.at("snapshot_bytes").get<std::size_t>() > 6000000 &&
                      full_capacity_receipt.at("snapshot_bytes").get<std::size_t>() + 12 <=
                          SemanticMap::max_snapshot_bytes,
                  "actual full map snapshot/terminal receipt within whole envelope bound");
            auto authority = full_capacity_receipt;
            authority.erase("status");
            peer.erase("status");
            check(authority == peer, "full capacity exact independently admitted snapshot hash and "
                                     "transcript agreement");
            full_capacity_receipt["test_elapsed_ms"] =
                std::chrono::duration_cast<std::chrono::milliseconds>(Clock::now() - started)
                    .count();
            cases.push_back(current_case);
        }
        for (const std::string fault : {"oversize_snapshot_header", "prequeued_ACK",
                                        "snapshot_backpressure", "progressive_terminal_deadline"}) {
            current_case = fault;
            fb::reset();
            Connection connection(package256);
            connection.ready();
            OutputSender sender(*connection.authority, 1, 2);
            OutputReceiver receiver(*connection.peer, 1, 2);
            const auto &descriptor = connection.authority->descriptor();
            const auto result = classification(descriptor, 1);
            sender.send(result, deadline());
            receiver.receive(deadline());
            SemanticMap independent(descriptor, 1, 2);
            independent.update(result);
            const auto snapshot = independent.snapshot_bytes(1);
            auto record = terminal(descriptor, result, Json::parse(snapshot), snapshot);
            if (fault == "oversize_snapshot_header") {
                auto header = envelope(32, "");
                const auto size = SemanticMap::max_snapshot_bytes - 12 + 1;
                for (unsigned i = 0; i < 4; ++i)
                    header[8 + i] = static_cast<char>(size >> (24 - 8 * i));
                raw_write(connection.a, header);
                rejects([&] { receiver.finish(deadline()); }, "payload bound");
                check(connection.peer->failed(), "oversized snapshot rejects from header alone");
            } else if (fault == "prequeued_ACK") {
                record["kind"] = "output_complete_ack";
                raw_write(connection.b, envelope(34, record.dump()));
                ::shutdown(connection.b, SHUT_WR);
                rejects([&] { sender.finish(deadline()); }, "acknowledgement");
                check(connection.authority->failed(), "prequeued ACK cannot know fresh challenge");
            } else if (fault == "snapshot_backpressure") {
                const int buffer = 1024;
                check(::setsockopt(connection.a, SOL_SOCKET, SO_SNDBUF, &buffer, sizeof(buffer)) ==
                          0,
                      "bounded snapshot send buffer");
                const auto started = Clock::now();
                rejects([&] { sender.finish(deadline(30)); }, "deadline");
                check(Clock::now() - started < std::chrono::milliseconds(500),
                      "snapshot backpressure cannot refresh absolute deadline");
                check(connection.authority->failed(), "snapshot backpressure aborts sender");
            } else {
                raw_write(connection.a, envelope(32, snapshot));
                const auto bytes = envelope(33, record.dump());
                auto writer = std::async(std::launch::async, [&] {
                    for (const char byte : bytes) {
                        if (::send(connection.a, &byte, 1, MSG_NOSIGNAL) <= 0)
                            break;
                        std::this_thread::sleep_for(std::chrono::milliseconds(2));
                    }
                });
                const auto started = Clock::now();
                rejects([&] { receiver.finish(deadline(35)); }, "deadline");
                writer.get();
                check(Clock::now() - started < std::chrono::milliseconds(500),
                      "terminal byte progress cannot refresh absolute deadline");
                check(connection.peer->failed(), "progressive terminal deadline aborts receiver");
            }
            cases.push_back(current_case);
        }
        for (const std::string fault : {"ACK_digest", "ACK_challenge", "ACK_count", "ACK_extra",
                                        "ACK_trailing_bytes", "ACK_missing_EOF"}) {
            current_case = fault;
            fb::reset();
            Connection connection(package);
            connection.ready();
            OutputSender sender(*connection.authority, 1, 2);
            sender.send(classification(connection.authority->descriptor(), 1), deadline());
            connection.peer->receive_classification(deadline());
            const timeval timeout{1, 0};
            check(::setsockopt(connection.b, SOL_SOCKET, SO_RCVTIMEO, &timeout, sizeof(timeout)) ==
                      0,
                  "raw peer read timeout");
            auto remote = std::async(std::launch::async, [&] {
                raw_message(connection.b, 32);
                auto ack = Json::parse(raw_message(connection.b, 33));
                ack["kind"] = "output_complete_ack";
                if (fault == "ACK_digest")
                    ack["snapshot_sha256"] = std::string(64, '0');
                if (fault == "ACK_challenge")
                    ack["completion_challenge"] = std::string(32, '0');
                if (fault == "ACK_count")
                    ack["classification_frames"] = 2;
                if (fault == "ACK_extra")
                    ack["extra"] = 1;
                raw_write(connection.b, envelope(34, ack.dump()));
                if (fault == "ACK_trailing_bytes")
                    raw_write(connection.b, "x");
                if (fault != "ACK_missing_EOF")
                    ::shutdown(connection.b, SHUT_WR);
            });
            rejects([&] { sender.finish(deadline(100)); });
            remote.get();
            check(connection.authority->failed(), "hostile ACK or missing ACK EOF cannot qualify");
            cases.push_back(current_case);
        }
        // Deliberately bypass output sending to place hostile exact wire records
        // on the admitted socket. The receiver still reconstructs its own map.
        for (const std::string fault :
             {"snapshot_projection", "snapshot_source", "snapshot_schema", "snapshot_duplicate",
              "terminal_digest", "terminal_count", "terminal_challenge", "missing_terminal",
              "trailing_bytes", "missing_EOF"}) {
            current_case = fault;
            fb::reset();
            Connection connection(package);
            connection.ready();
            OutputReceiver receiver(*connection.peer, 1, 2);
            const auto &descriptor = connection.authority->descriptor();
            const auto result = classification(descriptor, 1);
            connection.authority->send_classification(result, deadline());
            receiver.receive(deadline());
            SemanticMap independent(descriptor, 1, 2);
            independent.update(result);
            auto snapshot = independent.snapshot(1);
            const auto canonical = snapshot.dump();
            auto record = terminal(descriptor, result, snapshot, canonical);
            if (fault == "snapshot_projection")
                snapshot["cells"][0]["las_code"] = 255;
            if (fault == "snapshot_source")
                snapshot["source_identity"][1] = "different";
            if (fault == "snapshot_schema")
                snapshot["schema"] = "legacy";
            std::string bytes = snapshot.dump();
            if (fault == "snapshot_duplicate")
                bytes.insert(1, "\"schema\":\"synrex.generic_map_snapshot.v1\",");
            if (fault == "terminal_digest")
                record["classification_sha256"] = std::string(64, '0');
            if (fault == "terminal_count")
                record["classified_points"] = 2;
            if (fault == "terminal_challenge")
                record["completion_challenge"] = std::string(32, 'Z');
            raw_write(connection.a, envelope(32, bytes));
            if (fault != "missing_terminal")
                raw_write(connection.a, envelope(33, record.dump()));
            if (fault == "trailing_bytes")
                raw_write(connection.a, "x");
            if (fault != "missing_EOF")
                ::shutdown(connection.a, SHUT_WR);
            rejects([&] { receiver.finish(deadline(30)); });
            check(connection.peer->failed(),
                  "hostile snapshot/terminal/EOF permanently poisons receiver");
            cases.push_back(current_case);
        }
        // Both wrapper directions perform map admission, even when a remote uses
        // the lower-level classification transport directly.
        for (bool receiver_side : {false, true})
            for (const std::string fault :
                 {"projection", "source", "sequence", "time", "coverage", "identity", "size"}) {
                current_case = (receiver_side ? "receiver_" : "sender_") + fault;
                fb::reset();
                Connection connection(package);
                connection.ready();
                OutputSender sender(*connection.authority, 1, 2);
                OutputReceiver receiver(*connection.peer, 1, 2);
                auto first = classification(connection.authority->descriptor(), 1);
                sender.send(first, deadline());
                receiver.receive(deadline());
                auto bad = classification(connection.authority->descriptor(), 2);
                if (fault == "projection")
                    bad.las_codes[0] ^= 1;
                if (fault == "source")
                    bad.frame.stream_id = "different";
                if (fault == "sequence")
                    bad.frame.sequence = 1;
                if (fault == "time") {
                    bad.frame.frame_start_ns = bad.frame.frame_end_ns = 0;
                    bad.frame.points[0].timestamp_ns = 0;
                }
                if (fault == "coverage")
                    bad.semantic_ids.clear();
                if (fault == "identity")
                    bad.frame.identity.map_epoch = 43;
                if (fault == "size")
                    bad.frame.points.resize(1025, bad.frame.points[0]);
                if (receiver_side) {
                    // Field/codec failures are already stopped by the sender
                    // transport; source and frame order reach the receiver map.
                    if (fault == "source" || fault == "sequence" || fault == "time") {
                        connection.authority->send_classification(bad, deadline());
                        rejects([&] { receiver.receive(deadline()); });
                        check(connection.peer->failed(), "receiver map failure aborts socket");
                    } else {
                        rejects(
                            [&] { connection.authority->send_classification(bad, deadline()); });
                        rejects([&] { receiver.receive(deadline()); });
                        poisoned(connection);
                    }
                } else {
                    rejects([&] { sender.send(bad, deadline()); });
                    check(connection.authority->failed(),
                          "sender validation failure aborts socket");
                    rejects([&] { receiver.receive(deadline()); });
                    poisoned(connection);
                }
                cases.push_back(current_case);
            }
        // Valid but different map settings cannot qualify an unrelated snapshot.
        for (bool capacity_mismatch : {false, true}) {
            current_case =
                capacity_mismatch ? "snapshot_capacity_mismatch" : "snapshot_resolution_mismatch";
            fb::reset();
            Connection connection(package);
            connection.ready();
            OutputSender sender(*connection.authority, 1, 2);
            OutputReceiver receiver(*connection.peer, capacity_mismatch ? 1 : 2,
                                    capacity_mismatch ? 3 : 2);
            for (unsigned sequence = 1; sequence <= 3; ++sequence) {
                sender.send(
                    classification(connection.authority->descriptor(), sequence, sequence + 0.25),
                    deadline());
                receiver.receive(deadline());
            }
            auto remote = std::async(std::launch::async,
                                     [&] { return receiver_finish(receiver, deadline()); });
            rejects([&] { sender.finish(deadline()); });
            check(remote.get().find("FAILED:") == 0, "peer rejects independently mismatched map");
            poisoned(connection);
            cases.push_back(current_case);
        }
        // Invalid construction always fails closed on the existing connection.
        for (bool receiver_side : {false, true})
            for (unsigned fault = 0; fault < 5; ++fault) {
                current_case = "invalid_config";
                fb::reset();
                Connection connection(package);
                connection.ready();
                const double resolution = fault == 0   ? 0
                                          : fault == 1 ? -1
                                          : fault == 2 ? std::numeric_limits<double>::infinity()
                                                       : 1;
                const std::size_t capacity = fault == 3 ? 0 : fault == 4 ? 65537 : 2;
                if (receiver_side) {
                    rejects(
                        [&] { OutputReceiver receiver(*connection.peer, resolution, capacity); });
                    check(connection.peer->failed(), "invalid receiver config aborts");
                } else {
                    rejects(
                        [&] { OutputSender sender(*connection.authority, resolution, capacity); });
                    check(connection.authority->failed(), "invalid sender config aborts");
                }
            }
        cases.push_back("invalid_config_both_directions");
        for (bool omit_sender : {false, true}) {
            current_case = omit_sender ? "direct_transport_bypasses_both_maps"
                                       : "direct_transport_bypasses_receiver_map";
            fb::reset();
            Connection connection(package);
            connection.ready();
            OutputSender sender(*connection.authority, 1, 2);
            OutputReceiver receiver(*connection.peer, 1, 2);
            for (unsigned sequence = 1; sequence <= 3; ++sequence) {
                const auto result = classification(connection.authority->descriptor(), sequence);
                if (sequence == 2 && omit_sender)
                    connection.authority->send_classification(result, deadline());
                else
                    sender.send(result, deadline());
                if (sequence == 2)
                    connection.peer->receive_classification(deadline());
                else
                    receiver.receive(deadline());
            }
            auto remote = std::async(std::launch::async,
                                     [&] { return receiver_finish(receiver, deadline()); });
            rejects([&] { sender.finish(deadline()); });
            check(remote.get().find("FAILED:") == 0, "omitted map observation cannot qualify");
            poisoned(connection);
            cases.push_back(current_case);
        }
        for (bool receiver_side : {false, true}) {
            current_case = "empty_finish";
            fb::reset();
            Connection connection(package);
            connection.ready();
            OutputSender sender(*connection.authority, 1, 2);
            OutputReceiver receiver(*connection.peer, 1, 2);
            if (receiver_side) {
                rejects([&] { receiver.finish(deadline()); }, "nonempty");
                check(connection.peer->failed(), "empty receiver fails closed");
            } else {
                rejects([&] { sender.finish(deadline()); }, "nonempty");
                check(connection.authority->failed(), "empty sender fails closed");
            }
        }
        cases.push_back("empty_completion_rejected_both_directions");
        for (unsigned operation = 0; operation < 4; ++operation) {
            current_case = "expired_absolute_deadline";
            fb::reset();
            Connection connection(package);
            connection.ready();
            OutputSender sender(*connection.authority, 1, 2);
            OutputReceiver receiver(*connection.peer, 1, 2);
            const auto result = classification(connection.authority->descriptor(), 1);
            const auto expired = Clock::now() - std::chrono::milliseconds(1);
            if (operation == 0)
                rejects([&] { sender.send(result, expired); }, "absolute deadline");
            if (operation == 1)
                rejects([&] { receiver.receive(expired); }, "absolute deadline");
            if (operation == 2)
                rejects([&] { sender.finish(expired); }, "absolute deadline");
            if (operation == 3)
                rejects([&] { receiver.finish(expired); }, "absolute deadline");
            check(operation % 2 ? connection.peer->failed() : connection.authority->failed(),
                  "expired wrapper operation poisons connection before acceptance");
        }
        cases.push_back("expired_deadlines_all_wrapper_operations");
        for (bool receiver_side : {false, true}) {
            current_case = "unfinished_destructor";
            fb::reset();
            Connection connection(package);
            connection.ready();
            if (receiver_side) {
                {
                    OutputReceiver receiver(*connection.peer, 1, 2);
                }
                check(connection.peer->failed(), "unfinished receiver destruction aborts");
            } else {
                { OutputSender sender(*connection.authority, 1, 2); }
                check(connection.authority->failed(), "unfinished sender destruction aborts");
            }
        }
        cases.push_back("unfinished_destruction_both_directions");
        for (bool missing_ack : {false, true}) {
            current_case = missing_ack ? "missing_ACK" : "missing_snapshot_terminal";
            fb::reset();
            Connection connection(package);
            connection.ready();
            OutputSender sender(*connection.authority, 1, 2);
            OutputReceiver receiver(*connection.peer, 1, 2);
            sender.send(classification(connection.authority->descriptor(), 1), deadline());
            receiver.receive(deadline());
            if (missing_ack) {
                rejects([&] { sender.finish(deadline(30)); });
                check(connection.authority->failed(), "missing ACK cannot qualify authority");
            } else {
                ::shutdown(connection.a, SHUT_WR);
                rejects([&] { receiver.finish(deadline(30)); });
                check(connection.peer->failed(), "EOF cannot replace snapshot/terminal");
            }
            cases.push_back(current_case);
        }
        check(target_queries == 0 && fb::runs == 0, "no GPU target calls or neural inference");
        std::cout << Json{{"status", "PASS_CPU_OUTPUT_SOCKETPAIR"},
                          {"checks", checks.load()},
                          {"expected_rejections", rejections},
                          {"cases", cases},
                          {"full_capacity_receipt", full_capacity_receipt},
                          {"gpu_target_calls", target_queries},
                          {"gpu_initializations", 0},
                          {"neural_inference_calls", 0},
                          {"backend", "opaque CPU test fixture"}}
                         .dump()
                  << '\n';
        return 0;
    } catch (const std::exception &error) {
        std::cerr << current_case << ": " << error.what() << '\n';
        return 1;
    }
}
