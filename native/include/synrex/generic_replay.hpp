// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Tuomas Pietilä
#pragma once
namespace synrex::generic {
// Manifest opt-in path of the generic CLI. Every source gets fresh admitted
// connections, descriptor epoch, inference session and semantic map.
int replay_main(int argc, char **argv);
} // namespace synrex::generic
