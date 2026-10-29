// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Tuomas Pietilä
// Diagnostic ordinary C++ new requests; no model or GPU inference.
#include "synrex/transport.hpp"
#include <cstdlib>
#include <iostream>
#include <new>
#include <openssl/sha.h>
thread_local bool count_enabled = false;
thread_local std::uint64_t new_requests = 0;
[[gnu::noinline]] void *operator new(std::size_t n) {
    if (count_enabled)
        ++new_requests;
    if (auto p = std::malloc(n ? n : 1))
        return p;
    throw std::bad_alloc();
}
[[gnu::noinline]] void *operator new[](std::size_t n) { return ::operator new(n); }
[[gnu::noinline]] void operator delete(void *p) noexcept { std::free(p); }
[[gnu::noinline]] void operator delete[](void *p) noexcept { std::free(p); }
[[gnu::noinline]] void operator delete(void *p, std::size_t) noexcept { std::free(p); }
[[gnu::noinline]] void operator delete[](void *p, std::size_t) noexcept { std::free(p); }
using namespace synrex;
using namespace synrex::streaming;
namespace w = synrex::streaming::wire;
int main(int argc, char **argv) {
    try {
        require(argc == 2, "usage: transport-allocations NEW_RECEIPT");
        auto chunks = synthetic(91, 1, 16384, 4096, "synrex.fractal7.v1");
        Assembler assembler;
        std::optional<Frame> frame;
        for (const auto &chunk : chunks) {
            auto result = assembler.accept(chunk, steady_ns());
            if (result)
                frame = std::move(result);
        }
        require(bool(frame), "fixture complete");
        frame->header.chunk_count = 1;
        frame->header.sequence = 0;
        frame->first_sequence = frame->last_sequence = 0;
        w::Output out;
        out.input.frame = *frame;
        out.input.coordinates = w::Coordinates::sensor;
        for (std::size_t i = 0; i < frame->points.size(); ++i) {
            out.semantic_ids.push_back(i % 7);
            out.las_codes.push_back(std::vector<std::uint8_t>{1, 2, 200, 6, 9, 17, 201}[i % 7]);
        }
        out.model_manifest_sha256 = std::string(64, '0');
        const auto bytes = w::encode(out);
        std::array<unsigned char, 32> hash{};
        SHA256(bytes.data(), bytes.size(), hash.data());
        std::string digest;
        const char *hex = "0123456789abcdef";
        for (auto x : hash) {
            digest += hex[x >> 4];
            digest += hex[x & 15];
        }
        const auto start = steady_ns();
        count_enabled = true;
        std::uint64_t points = 0;
        for (unsigned repetition = 0; repetition < 128; ++repetition) {
            auto decoded = w::decode_output(bytes);
            points += decoded.input.frame.points.size();
        }
        count_enabled = false;
        const auto requests = new_requests;
        const auto seconds = (steady_ns() - start) / 1e9;
        require(w::encode(w::decode_output(bytes)) == bytes, "fixture exact decoder roundtrip");
        const Json receipt{{"schema", "synrex.streaming_r1.wire_allocation_probe.v1"},
                           {"status", "PASS"},
                           {"scope",
                            "MODEL_FREE_TEST_ONLY. Ordinary scalar/array C++ operator new requests "
                            "inside decode/destroy; excludes aligned new, direct malloc, CUDA, and "
                            "fixture setup. No neural throughput claim."},
                           {"fixture_sha256", digest},
                           {"decoded_fixture_points", points},
                           {"cpp_new_requests", requests},
                           {"decode_and_destroy_seconds", seconds}};
        write_new(argv[1], receipt.dump(2) + "\n");
        std::cout << receipt.dump(2) << '\n';
        return 0;
    } catch (const std::exception &e) {
        count_enabled = false;
        std::cerr << e.what() << '\n';
        return 1;
    }
}
