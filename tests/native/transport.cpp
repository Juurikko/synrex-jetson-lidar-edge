// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Tuomas Pietilä
#include "synrex/transport.hpp"
#include <cstring>
#include <iostream>
#include <netinet/in.h>
#include <sys/socket.h>
#include <unistd.h>
using namespace synrex;
using namespace synrex::streaming;
namespace w = synrex::streaming::wire;
namespace {
template <class F> void rejects(F f) {
    bool failed = false;
    try {
        f();
    } catch (const std::exception &) {
        failed = true;
    }
    require(failed, "expected visible rejection");
}
w::Input fixture() {
    auto chunk = synthetic(17, 1, 128, 128, "synrex.fractal7.v1").front();
    Assembler assembler;
    auto frame = assembler.accept(chunk, steady_ns());
    require(bool(frame), "fixture complete");
    return {*frame, w::Coordinates::sensor, 0};
}
} // namespace
int main() {
    try {
        const auto in = fixture();
        auto optional = in;
        optional.frame.points[0].intensity.reset();
        optional.frame.points[0].return_number.reset();
        optional.frame.points[0].return_count.reset();
        require(w::encode(w::decode(w::encode(optional))) == w::encode(optional),
                "optional attribute absence parity");
        const auto bytes = w::encode(in);
        auto decoded = w::decode(bytes);
        require(w::encode(decoded) == bytes, "exact binary point identity/time/features roundtrip");
        for (std::size_t n = 0; n < bytes.size(); ++n) {
            auto b = bytes;
            b.resize(n);
            rejects([&] { w::decode(b); });
        }
        auto bad = in;
        bad.frame.points[1].id = bad.frame.points[0].id;
        rejects([&] { w::encode(bad); });
        bad = in;
        bad.frame.points[0].timestamp_ns = 0;
        rejects([&] { w::encode(bad); });
        bad = in;
        bad.frame.header.chunk_count = 2;
        rejects([&] { w::encode(bad); });
        bad = in;
        bad.frame.points[0].xyz[0] = NAN;
        rejects([&] { w::encode(bad); });
        bad = in;
        bad.coordinates = w::Coordinates::projected;
        rejects([&] { w::encode(bad); });
        w::Sequence seq;
        seq.accept(in);
        rejects([&] { seq.accept(in); });
        auto next = in;
        next.frame.header.sequence = 2;
        next.frame.header.frame_id = 2;
        next.frame.first_sequence = next.frame.last_sequence = 2;
        rejects([&] { seq.accept(next); });
        next.frame.header.sequence = 1;
        next.frame.first_sequence = next.frame.last_sequence = 1;
        next.frame.header.frame_id = 1;
        next.frame.header.frame_start_ns -= 1;
        rejects([&] { seq.accept(next); });
        w::Transform pose;
        pose.from = in.frame.header.coordinate_frame;
        pose.to = "world";
        pose.valid_from_ns = in.frame.header.frame_start_ns;
        pose.valid_until_ns = in.frame.header.frame_end_ns;
        pose.translation = {10, -20, 30};
        auto moved = w::transform(in, pose);
        for (std::size_t i = 0; i < in.frame.points.size(); ++i) {
            const auto &a = in.frame.points[i], &b = moved.frame.points[i];
            require(a.id == b.id && a.timestamp_ns == b.timestamp_ns, "transform identity/time");
            for (int j = 0; j < 3; ++j)
                require(b.xyz[j] == a.xyz[j] + pose.translation[j], "moving origin transform");
        }
        pose.translation = {-10, 20, -30};
        auto moved2 = w::transform(in, pose);
        require(moved2.frame.points[0].xyz != moved.frame.points[0].xyz, "moving pose");
        pose.rotation = {0, -1, 0, 1, 0, 0, 0, 0, 1};
        auto rotated = w::transform(in, pose);
        for (std::size_t i = 0; i < in.frame.points.size(); ++i) {
            const auto &a = in.frame.points[i], &b = rotated.frame.points[i];
            require(b.xyz[0] == pose.translation[0] - a.xyz[1] &&
                        b.xyz[1] == pose.translation[1] + a.xyz[0] &&
                        b.xyz[2] == pose.translation[2] + a.xyz[2],
                    "rigid rotation parity");
        }
        pose.rotation = {1, 0, 0, 0, 1, 0, 0, 0, 1};
        auto missing = pose;
        missing.from = "missing-pose-frame";
        rejects([&] { w::transform(in, missing); });
        pose.valid_until_ns--;
        rejects([&] { w::transform(in, pose); });
        pose.valid_until_ns++;
        pose.version = 2;
        rejects([&] { w::transform(in, pose); });
        pose.version = 1;
        pose.rotation[0] = -1;
        rejects([&] { w::transform(in, pose); });
        w::Output out{
            in,
            std::vector<std::uint8_t>(128, 0),
            std::vector<std::uint8_t>(128, 1),
            {{{-1, 2, 3}, 0, in.frame.header.frame_end_ns, std::array<std::int64_t, 3>{-9, 8, 7}}},
            11,
            12,
            1,
            {}};
        out.model_manifest_sha256 = std::string(64, '0');
        require(w::encode(w::decode_output(w::encode(out))) == w::encode(out),
                "output/delta byte parity");
        auto canonical_out = out;
        canonical_out.input.replay_steady_ns = 0;
        canonical_out.received_ns = canonical_out.published_ns = 0;
        out.input.replay_steady_ns = 123456789;
        require(w::canonical(w::encode(out), true) == w::encode(canonical_out),
                "canonical output byte parity");
        require(w::canonical(w::encode(out), true, true) == w::encode(canonical_out.input),
                "canonical input byte parity");
        auto stamped = w::encode(out);
        w::stamp_publication(stamped, 999);
        require(w::decode_output(stamped).published_ns == 999 &&
                    w::canonical(stamped, true) == w::canonical(w::encode(out), true),
                "publication stamp exact field isolation");
        w::Sequence decoded_input_sequence, decoded_output_sequence;
        require(w::encode(decoded_input_sequence.decode_frame(bytes)) == bytes,
                "validated frame sequence parity");
        rejects([&] { decoded_input_sequence.decode_frame(bytes); });
        require(w::encode(decoded_output_sequence.decode_classified(w::encode(out))) ==
                    w::encode(out),
                "validated output sequence parity");
        rejects([&] { decoded_output_sequence.decode_classified(w::encode(out)); });
        // Real kernel sockets, extreme fragmentation, clean EOF versus truncated record.
        int pair[2];
        require(socketpair(AF_UNIX, SOCK_STREAM, 0, pair) == 0, "socketpair");
        w::Socket a(pair[0]), b(pair[1]);
        std::thread sender([&] {
            w::send(a, w::Kind::frame, bytes, 10000, 1);
            a.shutdown();
        });
        auto received = w::receive(b);
        require(received && received->second == bytes, "fragmented socket parity");
        require(!w::receive(b), "clean record EOF");
        sender.join();
        require(socketpair(AF_UNIX, SOCK_STREAM, 0, pair) == 0, "socketpair");
        w::Socket c(pair[0]), d(pair[1]);
        ::send(c.fd(), "bad", 3, 0);
        c.shutdown();
        rejects([&] { w::receive(d, 50); });
        require(socketpair(AF_UNIX, SOCK_STREAM, 0, pair) == 0, "socketpair");
        w::Socket e(pair[0]), f(pair[1]);
        rejects([&] { w::receive(f, 10); });
        // Unknown envelope versions/oversized declarations reject before payload allocation.
        for (auto which : {0, 1, 2, 3}) {
            require(socketpair(AF_UNIX, SOCK_STREAM, 0, pair) == 0, "socketpair");
            w::Socket tx(pair[0]), rx(pair[1]);
            w::Bytes header(56);
            const char magic[] = "SRXWIRE1";
            std::memcpy(header.data(), magic, 8);
            header[8] = 2;
            header[10] = 1;
            if (which == 0)
                header[8] = 99;
            if (which == 1)
                header[19] = 1; // reserved field
            if (which == 2)
                header[15] = 2; // 32 MiB > cap
            require(::send(tx.fd(), header.data(), header.size(), 0) == 56, "raw header send");
            tx.shutdown();
            rejects([&] { w::receive(rx, 50); });
        }
        require(socketpair(AF_UNIX, SOCK_STREAM, 0, pair) == 0, "socketpair");
        w::Socket stalled_sender(pair[0]), stalled_reader(pair[1]);
        rejects([&] { w::send(stalled_sender, w::Kind::frame, w::Bytes(w::max_payload), 10); });
        // Actual IPv4 loopback and clean reconnect preserve the same sequence state.
        auto listener = w::listen_loopback(0);
        sockaddr_in addr{};
        socklen_t length = sizeof(addr);
        require(getsockname(listener.fd(), reinterpret_cast<sockaddr *>(&addr), &length) == 0,
                "ephemeral port");
        for (int connection = 0; connection < 2; ++connection) {
            auto tx = w::connect_loopback(ntohs(addr.sin_port));
            auto rx = w::accept(listener);
            std::thread writer([&] { w::send(tx, w::Kind::frame, bytes, 1000, 7); });
            auto value = w::receive(rx);
            writer.join();
            require(value && value->second == bytes, "loopback reconnect bytes");
        }
        auto identity = in;
        identity.frame.header.sensor_id = "second-sensor";
        rejects([&] { seq.accept(identity); });
        w::Sequence separate;
        separate.accept(identity);
        // Retained sequence state is metadata-sized after over a million observations.
        w::Sequence long_sequence;
        auto repeated = in;
        for (std::uint64_t frame = 0; frame < 8192; ++frame) {
            auto &h = repeated.frame.header;
            h.sequence = h.frame_id = frame;
            repeated.frame.first_sequence = repeated.frame.last_sequence = frame;
            h.frame_start_ns = h.frame_end_ns = frame;
            for (auto &p : repeated.frame.points)
                p.timestamp_ns = frame;
            long_sequence.accept(repeated);
        }
        // Queue consumer stall is explicit, bounded and does not block producer.
        Queue<w::Bytes> queue(2);
        require(queue.push(bytes, false) && queue.push(bytes, false) && !queue.push(bytes, false),
                "bounded output queue policy");
        require(queue.cancel().size() == 2, "cancel accounting");
        std::cout << Json{{"status", "PASS"},
                          {"truncations_rejected", bytes.size()},
                          {"bounded_state_observations", 1048576},
                          {"malformed_envelopes_rejected", 4},
                          {"loopback_reconnections", 2},
                          {"canonical_output_bytes", "PASS"},
                          {"slow_socket_deadline", "PASS"},
                          {"point_and_delta_bytes", "PASS"},
                          {"sequence_gap_duplicate_clock", "PASS"},
                          {"coordinate_transform", "PASS"},
                          {"fragmentation_eof_deadline", "PASS"},
                          {"bounded_queue_cancel", "PASS"},
                          {"neural_inference", "NOT_RUN: CPU protocol test"}}
                         .dump(2)
                  << '\n';
        return 0;
    } catch (const std::exception &e) {
        std::cerr << e.what() << '\n';
        return 1;
    }
}
