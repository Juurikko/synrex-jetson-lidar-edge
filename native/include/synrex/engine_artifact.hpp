// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Tuomas Pietilä
#pragma once
#include "runtime.hpp"

namespace synrex {
// CPU-only admission and owned byte capture. Construction from a pathname is the
// only ingress: caller-built mutable Package/Contract values confer no authority.
// No GPU permission or session is created by retaining these bytes.
class EngineArtifact final {
    const Package package_;
    const std::string kind_, bytes_;

  public:
    static constexpr std::uint64_t max_bytes = 2147483648ULL;
    explicit EngineArtifact(const fs::path &, PackageUse use = PackageUse::file);
    EngineArtifact(const EngineArtifact &) = delete;
    EngineArtifact &operator=(const EngineArtifact &) = delete;
    const Contract &contract() const { return package_.contract; }
    const Json &manifest() const { return package_.manifest; }
    const std::string &manifest_bytes() const { return package_.manifest_bytes; }
    const std::string &manifest_sha256() const { return package_.manifest_sha256; }
    const std::optional<Taxonomy> &taxonomy() const { return package_.taxonomy; }
    const std::string &artifact_kind() const { return kind_; }
    const std::string &model_bytes() const { return bytes_; }
    // Explicitly rejects an admitted ONNX package before the caller touches GPU.
    const std::string &engine_bytes() const;
};
} // namespace synrex
