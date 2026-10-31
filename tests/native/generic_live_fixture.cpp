// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Tuomas Pietilä
// This executable is never installed or used for GPU/quality qualification.
#include "generic_fixture_backend.hpp"
#include "synrex/generic_live.hpp"
#include <cstdlib>
#include <iostream>

namespace synrex {
Json target_info() { throw std::runtime_error("CPU CLI fixture must never query a GPU target"); }
} // namespace synrex
int main(int argc, char **argv) {
    if (const auto fault = std::getenv("SYNREX_FIXTURE_RUN"); fault && std::string(fault) == "fail")
        fixture_backend::run_fault = fixture_backend::RunFault::throw_error;
    const auto result = synrex::generic::live_main(argc, argv);
    std::cerr << synrex::Json{{"scope", "opaque CPU backend only; no neural/GPU execution"},
                              {"initializations", fixture_backend::init_calls.load()},
                              {"runs", fixture_backend::runs.load()},
                              {"live_after_return", fixture_backend::live.load()}}
                     .dump()
              << '\n';
    return result;
}
