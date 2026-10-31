// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Tuomas Pietilä
#pragma once
#include "synrex/replay_manifest.hpp"
#include "synrex/runtime.hpp"
#include <memory>

namespace synrex::replay {
// Label-free observations. Keep original integer/scale precision and the exact
// GPS double bits; GPS is not converted into an acquisition clock or replay time.
struct Observation {
    std::uint64_t wire_id{}, original_record_index{}, gps_time_bits{};
    std::array<std::int32_t, 3> integer_xyz{};
    std::array<double, 3> xyz{};
    std::uint16_t intensity{};
    std::uint8_t return_number{}, return_count{};
};
struct Window {
    std::size_t source_index{};
    std::vector<Observation> observations;
};
// One open source, one raw record buffer, and one caller-owned output window.
// No point-cloud preload, background worker, queue or model construction. LAZ
// decoder-internal chunk allocations are separate from the output window bound.
// The Catalog must outlive the Reader. Source labels are never exposed here.
class Reader final {
    struct Impl;
    std::unique_ptr<Impl> impl_;

  public:
    Reader(const Catalog &, std::size_t source_index, std::size_t window_points);
    ~Reader();
    Reader(const Reader &) = delete;
    Reader &operator=(const Reader &) = delete;
    bool next(Window &); // Clears the prior window; EOF returns false.
    void finish();       // Requires EOF and verifies the retained source again.
    Json metadata() const;
};
} // namespace synrex::replay
