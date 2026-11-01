// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Tuomas Pietilä
#pragma once
#include "service.hpp"
#include <functional>
namespace synrex::streaming::wire {
using Bytes = std::vector<std::uint8_t>;
inline constexpr std::size_t max_payload = 16 * 1024 * 1024;
// Legacy terminal snapshots are single JSON payloads. See docs/streaming/SNAPSHOT_BUDGET.md.
// This transport admission limit does not constrain independent SemanticMap users.
inline constexpr std::size_t max_legacy_map_cells = 65536;
inline constexpr std::size_t legacy_snapshot_base_bytes = 8192;
inline constexpr std::size_t legacy_snapshot_cell_bytes = 224;
static_assert(legacy_snapshot_base_bytes + legacy_snapshot_cell_bytes * max_legacy_map_cells <=
              max_payload);
struct LegacyLiveLimits {
    std::size_t points{16384}, map_cells{max_legacy_map_cells};
    std::uint64_t rate{1000000};
    // Receiver and subscriber call this before sessions, exporters, or map allocation.
    void validate() const;
};
enum class Kind : std::uint16_t { frame = 1, classified = 2, snapshot = 3, end = 4 };
enum class Coordinates : std::uint8_t { sensor = 1, vehicle = 2, world = 3, projected = 4 };
struct Input {
    Frame frame;
    Coordinates coordinates{Coordinates::sensor};
    // Optional replay deadline in the receiver host's monotonic clock domain.
    // Zero means unavailable, never derived from measurement UNIX timestamps.
    std::uint64_t replay_steady_ns{};
};
struct Output {
    Input input;
    std::vector<std::uint8_t> semantic_ids, las_codes;
    std::vector<MapDelta> deltas;
    std::uint64_t received_ns{}, published_ns{}, model_executions{};
    std::string model_manifest_sha256;
};
Bytes encode(const Input &);
Input decode(const Bytes &);
Bytes encode(const Output &);
Output decode_output(const Bytes &);
// Call after decoding/validating. Clear only host clocks in an already encoded record.
Bytes canonical(const Bytes &, bool classified, bool input_only = false);
void stamp_publication(Bytes &, std::uint64_t steady_ns);
// Complete-frame sequence state persists across TCP reconnects. A different sensor
// requires a separate map/session; no implicit cross-sensor pose fusion.
class Sequence {
    std::optional<Input> previous_;
    void ordered(const Input &);

  public:
    void accept(const Input &);
    // Validate the wire exactly once, then update ordering state before returning.
    Input decode_frame(const Bytes &);
    Output decode_classified(const Bytes &);
};
struct Transform {
    std::uint32_t version{1};
    std::string from, to;
    Coordinates target{Coordinates::world};
    std::uint64_t valid_from_ns{}, valid_until_ns{};
    std::array<double, 9> rotation{1, 0, 0, 0, 1, 0, 0, 0, 1};
    std::array<double, 3> translation{};
};
Input transform(const Input &, const Transform &);
class Socket {
    int fd_{-1};

  public:
    explicit Socket(int fd = -1) : fd_(fd) {}
    ~Socket();
    Socket(const Socket &) = delete;
    Socket &operator=(const Socket &) = delete;
    Socket(Socket &&) noexcept;
    Socket &operator=(Socket &&) noexcept;
    int fd() const { return fd_; }
    void shutdown();
};
Socket listen_loopback(std::uint16_t port);
Socket connect_loopback(std::uint16_t port);
Socket accept(const Socket &, unsigned timeout_ms = 10000);
// Absolute operation deadline; clean EOF only before a new envelope, otherwise throw.
std::optional<std::pair<Kind, Bytes>> receive(const Socket &, unsigned timeout_ms = 10000,
                                              std::uint64_t *first_envelope_read_ns = nullptr);
void send(const Socket &, Kind, const Bytes &, unsigned timeout_ms = 10000,
          std::size_t fragment_bytes = 0);
Bytes json_bytes(const Json &);
Json json(const Bytes &);
} // namespace synrex::streaming::wire
