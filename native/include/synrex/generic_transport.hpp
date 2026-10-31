// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Tuomas Pietilä
#pragma once
#include "synrex/generic_startup.hpp"
#include "synrex/session.hpp"
#include <memory>

namespace synrex::generic {
class Session;
// Explicit selection only. No legacy decoder/fallback or frame socket I/O here.
enum class WireProtocol { binary_v3 };
namespace detail {
class ControlConnection;
}
// The connected SOCK_STREAM descriptor is borrowed, never closed. Failure shuts it down.
// Retains the admitted artifact independently of the package wrapper. The borrowed
// socket must outlive every produced Session/Service; it is never closed here.
// Control calls are single-threaded; do not call this object concurrently with a
// produced Session/Service. Transfer inference ownership to one worker after creation.
class AuthorityTransport {
    std::shared_ptr<detail::ControlConnection> connection_;
    std::shared_ptr<const EngineArtifact> begin_session();
    void complete_session();
    void abort_session() noexcept;
    std::function<void(std::uint64_t, std::uint64_t)> frame_authorizer() const;
    std::function<void()> session_aborter() const;

  public:
    AuthorityTransport(WireProtocol, int connected_socket, const CpuAdmittedPackage &,
                       Limits supported, std::string session_id, std::uint64_t map_epoch);
    ~AuthorityTransport();
    AuthorityTransport(const AuthorityTransport &) = delete;
    AuthorityTransport &operator=(const AuthorityTransport &) = delete;
    void metadata_admission();
    bool initialization_permission();
    // The sole authority initialization operation: consumes this connection's
    // retained engine and negotiated capacity, then exchanges execution-ready.
    // Destruction on failure precedes propagation; permission is never reusable.
    std::unique_ptr<Session> create_session(const SessionOptions &options = {});
    void authorize_frame(std::string_view descriptor_sha256, std::string_view session_id,
                         std::uint64_t map_epoch, std::uint64_t points, std::uint64_t bytes);
    const Descriptor &descriptor() const;
    Limits negotiated_limits() const;
    bool failed() const;
};
class PeerTransport {
    std::unique_ptr<detail::ControlConnection> connection_;

  public:
    PeerTransport(WireProtocol, int connected_socket, const CpuAdmittedPackage &expected_package,
                  Limits supported, std::string expected_session_id, std::uint64_t expected_epoch);
    ~PeerTransport();
    PeerTransport(const PeerTransport &) = delete;
    PeerTransport &operator=(const PeerTransport &) = delete;
    void metadata_admission();
    void acknowledge_execution_ready();
    // Identity/size authorization only, never payload allocation, decoding or validation.
    void authorize_frame(std::string_view descriptor_sha256, std::string_view session_id,
                         std::uint64_t map_epoch, std::uint64_t points, std::uint64_t bytes);
    const Descriptor &descriptor() const;
    Limits negotiated_limits() const;
    bool failed() const;
};
} // namespace synrex::generic
