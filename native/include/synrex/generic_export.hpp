// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Tuomas Pietilä
#pragma once
#include "synrex/generic_codec.hpp"
#include <memory>

namespace synrex::generic {
// Optional synchronous offline consumer, with one caller and no GPU work.
// Exactly one bounded source file (<=1M records, <=256MiB compressed) is bound
// by its SHA256. Source IDs MUST be zero-based original local record indices;
// global/replay IDs need an explicit external conversion before this API.
// No filesystem output occurs until finish. Encoding never runs in observe.
class LazExport final {
    struct Impl;
    std::unique_ptr<Impl> impl_;

  public:
    LazExport(const fs::path &source, const fs::path &model, const fs::path &destination,
              const Descriptor &);
    ~LazExport();
    LazExport(const LazExport &) = delete;
    LazExport &operator=(const LazExport &) = delete;
    // Exact sealed source capture identity for explicit upstream source binding.
    const std::string &source_sha256() const;
    // Any local row order, exactly once. Requires exact original xyz/intensity;
    // supplied return attributes must match the preserved original record.
    // Every rejected operation permanently fails this export instance.
    void observe(const Classification &);
    // Complete nonempty coverage only; atomically publishes a new bundle.
    // Receipt attests file preservation/coverage, never transport ACK or quality.
    Json finish();
};
} // namespace synrex::generic
