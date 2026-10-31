// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Tuomas Pietilä
#pragma once
#include "synrex/generic_map.hpp"
#include "synrex/generic_transport.hpp"

namespace synrex::generic {
// Synchronous CPU consumers of an already admitted transport. The transport must
// outlive its wrapper. All wrappers, sessions and calls sharing the connection
// have one caller; concurrent use is unsupported. Construction grants no GPU or
// execution permission. Destruction before successful finish aborts the stream.
class OutputSender final {
    AuthorityTransport &transport_;
    std::unique_ptr<SemanticMap> map_;
    std::uint64_t classifications_{}, points_{}, last_frame_end_{};
    bool finished_{}, failed_{};
    void alive() const;
    void abort() noexcept;

  public:
    OutputSender(AuthorityTransport &, double resolution, std::size_t capacity);
    ~OutputSender();
    OutputSender(const OutputSender &) = delete;
    OutputSender &operator=(const OutputSender &) = delete;
    // Complete encoded-byte, coverage, projection, source and ordering checks
    // precede transmission. Every failure permanently aborts the connection.
    void send(const Classification &, Startup::Time deadline);
    // Nonempty completion only. The peer must independently reconstruct this
    // exact map, acknowledge the terminal challenge and close its write side.
    Json finish(Startup::Time deadline);
};

class OutputReceiver final {
    PeerTransport &transport_;
    std::unique_ptr<SemanticMap> map_;
    std::uint64_t classifications_{}, points_{}, last_frame_end_{};
    bool finished_{}, failed_{};
    void alive() const;
    void abort() noexcept;

  public:
    OutputReceiver(PeerTransport &, double resolution, std::size_t capacity);
    ~OutputReceiver();
    OutputReceiver(const OutputReceiver &) = delete;
    OutputReceiver &operator=(const OutputReceiver &) = delete;
    Classification receive(Startup::Time deadline);
    // Admits the received snapshot and compares it with the local map before
    // sending an ACK. Its receipt attests ACK sent, not authority ACK receipt.
    Json finish(Startup::Time deadline);
};
} // namespace synrex::generic
