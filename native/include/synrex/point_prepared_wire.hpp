// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Tuomas Pietilä
#pragma once
#include "synrex/point_session.hpp"
#include <chrono>
#include <string_view>

namespace synrex::point::prepared_wire1 {
// Separately selected PWB1 codec. No legacy/dense protocol fallback. A binding
// captures native authority's exact descriptor and trusted retained contract.
class Binding final {
    Json descriptor_;
    PointContract contract_;
    std::string descriptor_sha256_, session_id_, worker_sha256_;
    std::uint64_t max_packet_bytes_;

  public:
    Binding(const PointSession &, const PointArtifact &, std::uint64_t max_packet_bytes = 8388608);
    const Json &descriptor() const { return descriptor_; }
    const PointContract &contract() const { return contract_; }
    const std::string &descriptor_sha256() const { return descriptor_sha256_; }
    const std::string &session_id() const { return session_id_; }
    const std::string &worker_sha256() const { return worker_sha256_; }
    std::uint64_t max_packet_bytes() const { return max_packet_bytes_; }
    void validate_source(const SourceIdentity &) const;
};
enum class Control : unsigned char {
    descriptor = 1, metadata_ack, initialization, ready_ack, source, completion, completion_ack
};
std::string encode_control(const Binding &, Control, const Json &body);
Json decode_control(const Binding &, Control, std::string_view);
std::string encode_batch(const Binding &, const PreparedBatch &);
PreparedBatch decode_batch(const Binding &, std::string_view);
std::string encode_classification(const Binding &, const Classification &);
Classification decode_classification(const Binding &, std::string_view);
// Borrowed connected SOCK_STREAM, never closed. Caller owns exactly this file
// descriptor until destruction. Single-threaded per Connection; independent
// endpoints may be driven concurrently. Absolute deadlines never refresh on
// fragments. Any operation error poisons/shuts down this endpoint. EOF throws.
// These are bounded CPU I/O primitives, not backend readiness. Applications must
// pass received controls to PointSession and call common::Session::run only after
// its two metadata and two actual-initialization ACKs. No receipt is synthesized.
class Connection final {
    int socket_;
    Binding binding_;
    bool failed_{};
    std::string receive(unsigned kind, std::chrono::steady_clock::time_point);
    void send(const std::string &, std::chrono::steady_clock::time_point);
    void fail() noexcept;

  public:
    using Clock = std::chrono::steady_clock;
    using Time = Clock::time_point;
    Connection(int borrowed_socket, Binding);
    Connection(const Connection &) = delete;
    Connection &operator=(const Connection &) = delete;
    bool failed() const { return failed_; }
    void send_control(Control, const Json &, Time);
    Json receive_control(Control, Time);
    void send_batch(const PreparedBatch &, Time);
    PreparedBatch receive_batch(Time);
    void send_classification(const Classification &, Time);
    Classification receive_classification(Time);
};
} // namespace synrex::point::prepared_wire1
