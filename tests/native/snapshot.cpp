// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Tuomas Pietilä
// CPU-only serializer and independent reconstruction; no neural inference claims.
#include "synrex/map_oracle.hpp"
#include "synrex/transport.hpp"
#include <cerrno>
#include <iostream>
#include <openssl/sha.h>
#include <sys/socket.h>
using namespace synrex;
using namespace synrex::streaming;
namespace w = synrex::streaming::wire;
namespace {
template <class F> void rejects(F f, const std::string &diagnostic = "") {
    bool failed = false;
    try {
        f();
    } catch (const std::exception &error) {
        const auto found = std::string(error.what()).find(diagnostic);
        require(diagnostic.empty() || found != std::string::npos,
                "unexpected snapshot/admission error: " + std::string(error.what()));
        failed = true;
    }
    require(failed, "expected snapshot/admission rejection");
}
std::string digest(const w::Bytes &bytes) {
    unsigned char hash[32];
    SHA256(bytes.data(), bytes.size(), hash);
    static constexpr char hex[] = "0123456789abcdef";
    std::string result;
    for (const auto value : hash) {
        result += hex[value >> 4];
        result += hex[value & 15];
    }
    return result;
}
void populate(SemanticMap &map, MapOracle *oracle, std::size_t count) {
    for (std::size_t begin = 0; begin < count; begin += 4096) {
        Classification c;
        auto &h = c.frame.header;
        h.sensor_id = std::string(128, '"');
        h.stream_id = std::string(128, '\\');
        h.coordinate_frame = std::string(128, '"');
        h.crs = "LOCAL:" + std::string(122, '\\');
        h.taxonomy = "synrex.fractal7.v1";
        h.sequence = h.frame_id = begin / 4096;
        c.frame.first_sequence = c.frame.last_sequence = h.sequence;
        std::uint64_t time = 10000000000000000000ULL;
        if (begin == 0)
            time = 0;
        else if (begin + 4096 >= count)
            time = UINT64_MAX;
        h.frame_start_ns = h.frame_end_ns = time;
        h.chunk_count = 1;
        h.frame_points = std::min<std::size_t>(4096, count - begin);
        for (std::size_t i = 0; i < h.frame_points; ++i) {
            // Exercise the map API's wider +/-1e15 key contract. Live wire input is
            // narrower (+/-1e9 metres), so this is a conservative encoding stress.
            const auto x = -1000000000000000.0 + begin + i;
            c.frame.points.push_back({begin + i, time, {x, -1e15, -1e15}, {}, {}, {}});
            c.semantic_ids.push_back(6);
            c.las_codes.push_back(201);
        }
        const auto deltas = map.update(c);
        if (oracle)
            oracle->verify(c, deltas, map.size());
    }
}
} // namespace
int main(int argc, char **argv) {
    try {
        const std::string mode = argc == 1 ? "all" : argv[1];
        require(argc <= 2 && (mode == "all" || mode == "historical" || mode == "admitted"),
                "usage: snapshot-tests [all|historical|admitted]");
        w::LegacyLiveLimits limits;
        limits.validate();
        for (const auto cells : std::array<std::uint64_t, 4>{0, 65537, 262144, UINT64_MAX}) {
            limits.map_cells = cells;
            rejects([&] { limits.validate(); });
        }
        limits.map_cells = w::max_legacy_map_cells;
        limits.points = 0;
        rejects([&] { limits.validate(); });
        limits.points = 65536;
        limits.rate = 65535;
        rejects([&] { limits.validate(); });
        limits.rate = 65536;
        limits.validate();

        Json receipt{{"schema", "synrex.legacy_snapshot.cpu.v1"},
                     {"case", mode},
                     {"neural_inference", "NOT_RUN: CPU classifications"}};
        if (mode != "admitted") {
            // Recreate the original admission defect without retaining a huge fixture.
            std::cerr << "snapshot_phase=historical_262144_cells\n";
            SemanticMap historical(1, 262144);
            populate(historical, nullptr, 262144);
            const auto snapshot = historical.snapshot(UINT64_MAX);
            const auto raw = snapshot.dump();
            require(raw.size() > w::max_payload, "original oversized-map defect not reproduced");
            rejects([&] { w::json_bytes(snapshot); });
            receipt["original_262144_cells"] = {{"encoded_bytes", raw.size()},
                                                {"wire_limit_bytes", w::max_payload},
                                                {"encoded_guard", "REJECTED"}};
        }
        if (mode == "historical") {
            receipt["status"] = "PASS";
            std::cout << receipt.dump(2) << '\n';
            return 0;
        }
        SemanticMap map(1, w::max_legacy_map_cells);
        std::cerr << "snapshot_phase=empty_and_minimal\n";
        MapOracle oracle(w::max_legacy_map_cells);
        const auto empty = w::json_bytes(map.snapshot(UINT64_MAX));
        require(w::json(empty) == oracle.snapshot(UINT64_MAX), "empty snapshot parity");
        {
            SemanticMap minimal(1, 1);
            MapOracle minimal_oracle(1);
            populate(minimal, &minimal_oracle, 1);
            const auto bytes = w::json_bytes(minimal.snapshot(UINT64_MAX));
            require(w::json(bytes) == minimal_oracle.snapshot(UINT64_MAX),
                    "one-cell snapshot parity");
            receipt["minimal_snapshot_bytes"] = bytes.size();
        }
        std::cerr << "snapshot_phase=full_admitted_65536_cells\n";
        populate(map, &oracle, w::max_legacy_map_cells);
        const auto snapshot = map.snapshot(UINT64_MAX);
        const auto bytes = w::json_bytes(snapshot);
        require(bytes.size() <= w::legacy_snapshot_base_bytes +
                                    w::legacy_snapshot_cell_bytes * w::max_legacy_map_cells,
                "snapshot serialization exceeded proven bound");
        for (const auto &cell : snapshot.at("cells"))
            require(cell.dump().size() + 1 <= w::legacy_snapshot_cell_bytes,
                    "cell serialization exceeded proven bound");
        int pair[2];
        std::cerr << "snapshot_phase=socket_transfer_and_reconstruction\n";
        require(socketpair(AF_UNIX, SOCK_STREAM, 0, pair) == 0, "snapshot socketpair");
        w::Socket tx(pair[0]), rx(pair[1]);
        std::exception_ptr error;
        std::thread writer([&] {
            try {
                w::send(tx, w::Kind::snapshot, bytes, 30000, 4093);
            } catch (...) {
                error = std::current_exception();
            }
            tx.shutdown();
        });
        // Join on both success and failure so a regression emits an ordinary failure.
        std::optional<std::pair<w::Kind, w::Bytes>> message;
        try {
            message = w::receive(rx, 30000);
        } catch (...) {
            rx.shutdown();
            writer.join();
            throw;
        }
        writer.join();
        if (error)
            std::rethrow_exception(error);
        require(message && message->first == w::Kind::snapshot && message->second == bytes &&
                    digest(message->second) == digest(bytes),
                "actual full snapshot transport/digest mismatch");
        const auto reconstructed = w::json(message->second);
        require(reconstructed == oracle.snapshot(UINT64_MAX) &&
                    reconstructed.at("cells").size() == w::max_legacy_map_cells,
                "independent full snapshot/identity reconstruction mismatch");
        require(!w::receive(rx), "unexpected post-snapshot bytes");
        auto &full = receipt["full_admitted_snapshot"];
        full["cells"] = map.size();
        full["encoded_bytes"] = bytes.size();
        full["sha256"] = digest(bytes);
        full["all_cells_and_identity"] = "PASS";
        full["socket_send_receive"] = "PASS";
        // JSON string delimiters count in the final payload budget.
        std::cerr << "snapshot_phase=encoded_byte_boundary\n";
        const auto boundary = w::json_bytes(Json(std::string(w::max_payload - 2, 'x'))).size();
        require(boundary == w::max_payload, "exact encoded JSON boundary rejected");
        rejects([&] { w::json_bytes(Json(std::string(w::max_payload - 1, 'x'))); });
        // A live unused socket distinguishes admission failure from EPIPE or timeout.
        int budget_pair[2];
        require(socketpair(AF_UNIX, SOCK_STREAM, 0, budget_pair) == 0, "budget socketpair");
        w::Socket budget_tx(budget_pair[0]), budget_rx(budget_pair[1]);
        rejects([&] { w::send(budget_tx, w::Kind::snapshot, w::Bytes(w::max_payload + 1), 100); },
                "wire send budget");
        std::uint8_t unexpected{};
        require(::recv(budget_rx.fd(), &unexpected, 1, MSG_DONTWAIT) == -1 &&
                    (errno == EAGAIN || errno == EWOULDBLOCK),
                "oversized payload wrote bytes before admission");
        receipt["encoded_byte_boundary"] = "PASS";
        receipt["shared_admission"] = "PASS";
        receipt["status"] = "PASS";
        std::cout << receipt.dump(2) << '\n';
        return 0;
    } catch (const std::exception &e) {
        std::cerr << e.what() << '\n';
        return 1;
    }
}
