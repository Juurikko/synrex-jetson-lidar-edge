// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Tuomas Pietilä
#pragma once
#include "stream.hpp"
#include <thread>
namespace synrex::streaming {
// Optional offline subscriber worker. One bounded first-observation product per
// source corpus; inference never calls file encoding. Partial coverage never publishes.
class AsyncLazExport {
    fs::path source_, model_, taxonomy_, destination_;
    std::vector<Point> original_;
    std::vector<std::uint8_t> seen_, labels_, semantics_;
    std::vector<std::uint64_t> frames_, timestamps_;
    std::size_t covered_{};
    bool submitted_{};
    Json metadata_, receipt_;
    std::thread worker_;
    std::exception_ptr error_;
    void publish();

  public:
    AsyncLazExport(const fs::path &source, const fs::path &model, const fs::path &taxonomy,
                   const fs::path &destination);
    ~AsyncLazExport();
    void observe(const Classification &);
    Json finish();
};
} // namespace synrex::streaming
