// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Tuomas Pietilä
#pragma once
#include "synrex/runtime.hpp"
#include <memory>

namespace synrex::point {
struct SourceRecord {
    std::array<std::int32_t, 3> integer_xyz;
    std::array<double, 3> xyz;
    std::uint16_t intensity;
};
// Label-free, CPU-only source capture. Decoding and later preservation use the
// same sealed bytes. No dense Contract, model admission or backend is involved.
// Standard compressed LAS1.4 PDRF6/7/8 only; Extra Bytes and >1M remain excluded.
class PointSource final {
    struct Impl;
    std::unique_ptr<Impl> impl_;

  public:
    static constexpr std::uint64_t max_bytes = 268435456;
    static constexpr std::uint64_t max_rows = 1000000;
    explicit PointSource(const fs::path &, std::uint64_t row_limit = max_rows);
    ~PointSource();
    PointSource(const PointSource &) = delete;
    PointSource &operator=(const PointSource &) = delete;
    const std::vector<SourceRecord> &records() const;
    const std::string &sha256() const;
    const std::string &header_sha256() const;
    Json metadata() const;
    // Fails if the original named source no longer has identical bytes.
    void verify_original() const;
    // CPU writer primitive, not a session completion or inference gate. Caller
    // must supply complete original-order semantics and retained taxonomy. The
    // new bundle retains authoritative semantics even for many-to-one LAS maps.
    Json export_semantics(const fs::path &destination, const Taxonomy &,
                          const std::vector<std::uint8_t> &semantics) const;
};
} // namespace synrex::point
