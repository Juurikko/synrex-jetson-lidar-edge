// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Tuomas Pietilä
#pragma once
#include "synrex/generic_codec.hpp"
#include "synrex/session.hpp"
#include <functional>

namespace synrex::generic {
class AuthorityTransport;
// One inference worker owns this session. Only the authority's real initialization
// path can construct it; caller-built sessions/descriptors confer no readiness.
class Session final {
    std::unique_ptr<InferenceSession> backend_;
    const Descriptor descriptor_;
    const std::function<void(std::uint64_t, std::uint64_t)> authorize_;
    const std::function<void()> abort_;
    bool poisoned_{};
    std::optional<std::array<std::string, 4>> source_identity_;
    std::optional<std::array<std::uint64_t, 3>> last_frame_;
    friend class AuthorityTransport;
    Session(std::unique_ptr<InferenceSession>, Descriptor,
            std::function<void(std::uint64_t, std::uint64_t)>, std::function<void()>);

  public:
    Session(const Session &) = delete;
    Session &operator=(const Session &) = delete;
    Classification run(Frame frame);
    const Descriptor &descriptor() const { return descriptor_; }
    Json identity() const;
};
} // namespace synrex::generic
