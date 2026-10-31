// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Tuomas Pietilä
#pragma once
#include "synrex/generic_codec.hpp"
#include "synrex/replay_reader.hpp"

namespace synrex::replay {
struct FrameOptions {
    std::size_t window_points{16384};
    std::uint64_t points_per_second{250000};
    void validate(const generic::Descriptor &) const;
};
// floor(wire_id * 1e9 / rate), with no floating-point rounding or intermediate
// overflow. Synthetic file order only: never interprets source GPS as a clock.
std::uint64_t synthetic_time_ns(std::uint64_t wire_id, std::uint64_t rate);
// CPU-only construction. Frame ID equals sequence; source stream identity is the
// admitted file SHA256. Every source starts its own descriptor/session epoch.
generic::Frame make_frame(const Catalog &, const generic::Descriptor &, const FrameOptions &,
                          const Window &, std::uint64_t sequence);

// One-source ordered coverage. Catalog and descriptor must outlive this object.
// Only counters are retained: no source cloud, frame cache, or unbounded ID set.
// Any invalid operation poisons this instance permanently. source_eof() declares
// frame coverage complete. It proves no source decoding: Reader::finish() must
// separately establish decoded header coverage and source identity where needed.
class Coverage final {
    const Catalog &catalog_;
    const generic::Descriptor &descriptor_;
    FrameOptions options_;
    std::size_t source_index_;
    std::uint64_t sequence_begin_, points_{}, windows_{}, exact_points_{}, identity_points_{};
    bool failed_{}, eof_{}, finished_{};
    void validate_next(const generic::Frame &) const;
    void advance(bool exact, std::size_t count);

  public:
    Coverage(const Catalog &, const generic::Descriptor &, FrameOptions, std::size_t source_index,
             std::uint64_t sequence_begin = 0);
    Coverage(const Coverage &) = delete;
    Coverage &operator=(const Coverage &) = delete;
    // Checks complete metadata/row identity/time/count. Makes no attribute claim.
    void accept_frame(const generic::Frame &);
    // Also checks all supported attributes against independently decoded input.
    void accept(const Window &, const generic::Frame &);
    void source_eof();
    void finish();
    Json receipt() const;
};
} // namespace synrex::replay
