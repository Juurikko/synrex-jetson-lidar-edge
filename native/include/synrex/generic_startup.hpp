// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Tuomas Pietilä
#pragma once
#include "synrex/engine_artifact.hpp"
#include <chrono>
#include <functional>
#include <memory>
#include <string_view>

namespace synrex::generic {
namespace detail {
class ControlConnection;
}
// CPU preparation only; companion transport/session/codec modules consume it.
struct Limits {
    std::uint64_t max_frame_points{16384}, max_frame_bytes{1048576};
    std::uint64_t startup_timeout_ms{30000};
    void validate() const;
    static Limits negotiate(const Limits &local, const Limits &peer);
};
// Admission provenance is private ownership, not a rehash of caller-mutable fields.
// The production constructor is the only ingress. No Package/Contract is exposed.
class CpuAdmittedPackage {
    const std::shared_ptr<const EngineArtifact> package_;
    friend class Descriptor;
    friend class detail::ControlConnection;
    explicit CpuAdmittedPackage(std::shared_ptr<const EngineArtifact> retained)
        : package_(std::move(retained)) {}

  public:
    explicit CpuAdmittedPackage(const fs::path &path);
    CpuAdmittedPackage(const CpuAdmittedPackage &) = delete;
    CpuAdmittedPackage &operator=(const CpuAdmittedPackage &) = delete;
};
class Descriptor {
    std::string bytes_, sha256_;
    Json spec_;
    Taxonomy taxonomy_;
    bool package_bound_{};
    explicit Descriptor(std::string bytes);
    friend class Startup;

  public:
    static constexpr std::size_t max_bytes = 65536, max_depth = 8;
    static constexpr std::uint64_t metadata_timeout_ms = 5000;
    // Only the privately owned production-admitted wrapper can create authority
    // identity. The retained captured metadata is used; no mutable Package input.
    static Descriptor from_package(const CpuAdmittedPackage &, const Limits &,
                                   const std::string &session_id, std::uint64_t map_epoch);
    static Descriptor parse(std::string_view bytes);
    const std::string &bytes() const { return bytes_; }
    const std::string &sha256() const { return sha256_; }
    const Json &spec() const { return spec_; }
    const Taxonomy &taxonomy() const { return taxonomy_; }
    std::string control(const std::string &kind) const;
};

enum class State {
    awaiting_descriptor,
    awaiting_metadata_ack,
    gpu_permitted,
    initializing,
    awaiting_execution_ready_ack,
    ready,
    failed
};
class Startup {
  public:
    using Clock = std::chrono::steady_clock;
    using Time = Clock::time_point;
    using Now = std::function<Time()>;

  private:
    Descriptor descriptor_;
    Now now_;
    Time last_, metadata_deadline_, startup_deadline_{};
    State state_{State::awaiting_descriptor};
    std::string fragments_, failure_;
    Time check_time();
    void expect(State expected);
    void check_control(const std::string &bytes, const std::string &kind);
    void poison(const std::exception &error);

  public:
    // Authority-side gate: expected must come from from_package, never solely
    // from untrusted peer parse. Peer-side parse/control encoding is separate.
    explicit Startup(Descriptor expected, Now now = [] { return Clock::now(); });
    Startup(const Startup &) = delete;
    Startup &operator=(const Startup &) = delete;
    // Fragments/repeated polls never reset either deadline. final=true parses
    // exactly those bytes and binds the entire descriptor, including its serialization.
    void descriptor_fragment(std::string_view bytes, bool final);
    void metadata_ack(const std::string &control);
    bool gpu_permission(); // Permission only; never constructs or initializes GPU resources.
    void begin_initialization();
    void initialization_complete();
    void execution_ready_ack(const std::string &control);
    void accept_frame_identity(std::string_view descriptor_sha256, std::string_view session_id,
                               std::uint64_t map_epoch);
    void poll();
    void fail(std::string reason);
    State state() const { return state_; } // Diagnostic only; not an authorization query.
    const std::string &failure() const { return failure_; }
};
} // namespace synrex::generic
