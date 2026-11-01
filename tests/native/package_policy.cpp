// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Tuomas Pietilä
// Direct package-loading boundary probes; no inference or hardware qualification.
#include "synrex/export.hpp"
#include "synrex/session.hpp"
#include <iostream>
int main(int argc, char **argv) {
    using namespace synrex;
    try {
        require(argc == 4, "requires mode, package and unused output path");
        const std::string mode(argv[1]);
        if (mode == "file" || mode == "legacy") {
            const Package package(argv[2],
                                  mode == "file" ? PackageUse::file : PackageUse::legacy_stream);
            std::cout << package.manifest_sha256 << '\n';
        } else if (mode == "backend") {
            std::cout << (gpu_enabled() ? "gpu" : "cpu") << '\n';
        } else if (mode == "session") {
            (void)create_session(argv[2], 1);
        } else if (mode == "export") {
            streaming::AsyncLazExport exporter("absent-source.laz", argv[2],
                                               "absent-external-taxonomy.json", argv[3]);
        } else {
            require(false, "unknown package policy probe");
        }
        return 0;
    } catch (const std::exception &error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
