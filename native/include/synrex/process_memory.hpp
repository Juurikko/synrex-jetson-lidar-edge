// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Tuomas Pietilä
#pragma once
#include <cstdint>
#include <limits>
#include <stdexcept>

namespace synrex::streaming {
// Convert resident pages from /proc/self/statm using sysconf(_SC_PAGESIZE).
// Linux ru_maxrss is measured in KiB and must not use this conversion.
inline std::uint64_t checked_rss_bytes(std::uint64_t resident_pages, long page_size) {
    if (page_size <= 0)
        throw std::invalid_argument("cannot determine host page size");
    const auto bytes_per_page = static_cast<std::uint64_t>(page_size);
    if (resident_pages > std::numeric_limits<std::uint64_t>::max() / bytes_per_page)
        throw std::overflow_error("current host RSS exceeds uint64 byte range");
    return resident_pages * bytes_per_page;
}
} // namespace synrex::streaming
