// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Tuomas Pietilä
#pragma once
// Backend linkage only. This header is not part of the installed public API.
#include "synrex/engine_artifact.hpp"
#include "synrex/session.hpp"
namespace synrex::generic::detail {
std::unique_ptr<InferenceSession>
create_persistent_session(std::shared_ptr<const EngineArtifact> retained, std::size_t capacity,
                          const SessionOptions &options);
} // namespace synrex::generic::detail
