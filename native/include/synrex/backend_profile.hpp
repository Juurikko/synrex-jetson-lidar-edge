// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Tuomas Pietilä
#pragma once
#include "synrex/runtime.hpp"
#include <optional>

namespace synrex::generic {
// Fixed-size summaries. Numeric checks precede mutation, including merges.
struct ProfileSeconds {
    std::uint64_t samples{}, missing{};
    double sum{}, minimum{}, maximum{};
    void add(std::optional<double>);
    void merge(const ProfileSeconds &);
    Json receipt() const;
};
struct ProfileCounter {
    std::uint64_t samples{}, missing{}, sum{}, minimum{}, maximum{};
    void add(std::optional<std::uint64_t>);
    void merge(const ProfileCounter &);
    Json receipt() const;
};
// Consumes backend-reported values; confers no execution authority or GPU proof.
// The actual native Session identity, frozen executable and job receipts bind use.
class BackendProfile {
    std::array<ProfileSeconds, 7> phases_{};
    std::array<ProfileCounter, 3> counters_{};
    Json workload_ = nullptr;
    std::uint64_t windows_{}, points_{}, max_blocks_{}, max_points_{}, max_memberships_{},
        voxels_{};
    bool supported_{};

  public:
    static constexpr std::size_t max_receipt_bytes = 16384;
    explicit BackendProfile(const Json &session_identity);
    void observe(const Json &backend_timings, std::uint64_t returned_frame_points);
    void merge(const BackendProfile &);
    Json receipt() const;
};
} // namespace synrex::generic
