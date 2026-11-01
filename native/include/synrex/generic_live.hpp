// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Tuomas Pietilä
#pragma once
namespace synrex::generic {
// Explicit binary-v3, bounded finite single-file replay. The fixture executable
// invokes this same entry point with a test-only backend, never a GPU substitute.
int live_main(int argc, char **argv);
} // namespace synrex::generic
