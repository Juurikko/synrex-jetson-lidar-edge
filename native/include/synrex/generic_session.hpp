// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Tuomas Pietilä
#pragma once
#include "synrex/generic_codec.hpp"
#include "synrex/session.hpp"
#include <functional>

namespace synrex::generic {
class AuthorityTransport;
// CPU admission only; disabled unless all fields are explicitly populated.
struct DiagnosticTraceOptions {
    fs::path root;
    std::uint64_t total_bytes{}, windows{};
    std::string declared_source_commit;
    bool enabled() const { return !root.empty(); }
    void validate() const;
    // Public window count is a cap; LIVE may require its exact finite frame count.
    void validate_contract(const Contract &, std::uint64_t required_windows = 1) const;
};
// Conservative backend tensor+manifest bound, checked before any diagnostic write.
std::uint64_t diagnostic_tensor_bound(const Contract &, std::uint64_t blocks,
                                      std::uint64_t memberships = 0);

// One inference worker owns this session. Only the authority's real initialization
// path can construct it; caller-built sessions/descriptors confer no readiness.
class Session final {
    std::unique_ptr<InferenceSession> backend_;
    const Descriptor descriptor_;
    const std::function<void(std::uint64_t, std::uint64_t)> authorize_;
    const std::function<void()> abort_;
    bool poisoned_{};
    const DiagnosticTraceOptions trace_;
    const GeometryMode geometry_mode_;
    const double trace_admission_seconds_;
    std::uint64_t trace_reserved_{}, trace_written_{}, trace_windows_{};

    std::optional<std::array<std::string, 4>> source_identity_;
    std::optional<std::array<std::uint64_t, 3>> last_frame_;
    friend class AuthorityTransport;
    Session(std::unique_ptr<InferenceSession>, Descriptor,
            std::function<void(std::uint64_t, std::uint64_t)>, std::function<void()>,
            DiagnosticTraceOptions, GeometryMode, double);

  public:
    Session(const Session &) = delete;
    Session &operator=(const Session &) = delete;
    Classification run(Frame frame, std::optional<std::uint64_t> receiver_frame_available_ns = {});
    // LIVE calls only after run returned and output send completed; local observations only.
    void record_output_timing(std::uint64_t window, std::uint64_t output_ready,
                              std::uint64_t send_started, std::uint64_t send_completed);
    const Descriptor &descriptor() const { return descriptor_; }
    Json identity() const;
};
} // namespace synrex::generic
