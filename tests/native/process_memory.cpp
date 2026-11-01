// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Tuomas Pietilä
#include "synrex/process_memory.hpp"
#include <iostream>
#include <string>

namespace {
void check(bool condition, const char *message) {
    if (!condition)
        throw std::runtime_error(message);
}

template <class Exception, class F> void rejects(F operation, const char *message) {
    try {
        operation();
    } catch (const Exception &error) {
        check(std::string(error.what()) == message, "unexpected RSS conversion error message");
        return;
    }
    throw std::runtime_error("expected RSS conversion rejection");
}
} // namespace

int main() {
    using synrex::streaming::checked_rss_bytes;
    try {
        constexpr auto page_message = "cannot determine host page size";
        constexpr auto overflow_message = "current host RSS exceeds uint64 byte range";
        check(checked_rss_bytes(1, 4096) == 4096, "one 4 KiB resident page");
        check(checked_rss_bytes(32768, 4096) == 134217728, "4 KiB RSS fixture");
        check(checked_rss_bytes(1, 65536) == 65536, "one 64 KiB resident page");
        check(checked_rss_bytes(65536, 65536) == 4294967296ULL, "64 KiB RSS fixture");
        check(checked_rss_bytes(0, 4096) == 0, "zero residents with 4 KiB pages");
        check(checked_rss_bytes(0, 65536) == 0, "zero residents with 64 KiB pages");

        for (const auto residents : {std::uint64_t{0}, std::uint64_t{1}}) {
            for (const long page_size : {0L, -1L}) {
                rejects<std::invalid_argument>([&] { checked_rss_bytes(residents, page_size); },
                                               page_message);
            }
        }

        const auto maximum = std::numeric_limits<std::uint64_t>::max();
        check(checked_rss_bytes(maximum, 1) == maximum, "maximum uint64 byte value");
        check(checked_rss_bytes(maximum / 3, 3) == maximum, "exact maximum product");
        rejects<std::overflow_error>([&] { checked_rss_bytes(maximum / 3 + 1, 3); },
                                     overflow_message);
        for (const long page_size : {2L, 4096L, 65536L}) {
            const auto bytes_per_page = static_cast<std::uint64_t>(page_size);
            check(checked_rss_bytes(maximum / bytes_per_page, page_size) ==
                      maximum - (bytes_per_page - 1),
                  "largest representable resident-page count");
            rejects<std::overflow_error>(
                [&] { checked_rss_bytes(maximum / bytes_per_page + 1, page_size); },
                overflow_message);
            rejects<std::overflow_error>([&] { checked_rss_bytes(maximum, page_size); },
                                         overflow_message);
        }
        std::cout << "checked_rss_bytes=PASS (4KiB/64KiB, zero, invalid page size, overflow)\n";
        return 0;
    } catch (const std::exception &error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
