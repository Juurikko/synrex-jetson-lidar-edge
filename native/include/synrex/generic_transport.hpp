// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Tuomas Pietilä
#pragma once
#include "synrex/generic_startup.hpp"
#include <memory>

namespace synrex::generic {
// Explicit selection only. There is no legacy decoder, fallback, Service or frame codec here.
enum class WireProtocol { binary_v3 };
namespace detail {
class ControlConnection;
}
// The connected SOCK_STREAM descriptor is borrowed, never closed. Failure shuts it down.
// Keep the admitted package alive through metadata_admission(). Calls are single-threaded.
class AuthorityTransport {
    std::unique_ptr<detail::ControlConnection> connection_;

  public:
    AuthorityTransport(WireProtocol, int connected_socket, const CpuAdmittedPackage &,
                       Limits supported, std::string session_id, std::uint64_t map_epoch);
    ~AuthorityTransport();
    AuthorityTransport(const AuthorityTransport &) = delete;
    AuthorityTransport &operator=(const AuthorityTransport &) = delete;
    void metadata_admission();
    bool initialization_permission();
    void begin_initialization();
    // Caller reports initialization completion; this library never initializes a GPU.
    void initialization_complete_and_wait_ready();
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
