// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Tuomas Pietilä
#include <lazperf/decompressor.hpp>
#include <lazperf/readers.hpp>
#include <lazperf/streams.hpp>

#include <array>
#include <cstring>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <vector>

namespace {
void check(bool condition) {
    if (!condition)
        throw std::runtime_error("codec overlay regression assertion");
}

template <class F> void reject(F read) {
    try {
        read();
    } catch (const lazperf::error &) {
        return;
    }
    throw std::runtime_error("expected codec read exhaustion");
}

struct FakeDecoder {
    std::vector<uint32_t> symbols;
    size_t at = 0;
    uint32_t raw = 0;
    template <class T> uint32_t decodeSymbol(T &) {
        check(at < symbols.size());
        return symbols[at++];
    }
    uint32_t readBits(uint32_t) { return raw; }
    template <class T> uint32_t decodeBit(T &) { return raw; }
    uint32_t readBit() { return raw; }
};

void memory() {
    lazperf::MemoryStream stream;
    reject([&] { stream.getByte(); });
    stream.putByte(7);
    check(stream.getByte() == 7);
    reject([&] { stream.getByte(); });
    stream.idx = 0;
    unsigned char output[2] = {13, 17};
    reject([&] { stream.getBytes(output, 2); });
    check(stream.idx == 0 && output[0] == 13 && output[1] == 17);
    reject([&] { stream.getBytes(output, SIZE_MAX); });
    check(stream.idx == 0 && output[0] == 13);
    stream.getBytes(output, 1);
    check(output[0] == 7);
    stream.getBytes(output, 0);
    stream.idx = SIZE_MAX;
    reject([&] { stream.getByte(); });
    reject([&] { stream.getBytes(output, 1); });
    reject([&] { stream.getBytes(output, 0); });
}

void integer() {
    using Integer = lazperf::decompressors::integer;
    const auto decode = [](uint32_t bits, int32_t predictor, std::vector<uint32_t> symbols,
                           uint32_t raw) {
        Integer decoder(bits);
        decoder.init();
        FakeDecoder entropy{std::move(symbols), 0, raw};
        return decoder.decompress(entropy, predictor, 0);
    };
    check(lazperf::utils::signed32(0) == 0);
    check(lazperf::utils::signed32(UINT32_MAX) == -1);
    check(lazperf::utils::add32(INT32_MIN, -1) == INT32_MAX);
    check(lazperf::utils::add32(INT32_MAX, 1) == INT32_MIN);
    check(decode(32, INT32_MAX, {32}, 0) == -1);
    check(decode(32, INT32_MAX, {0}, 1) == INT32_MIN);
    check(decode(32, 0, {31, 255}, 8388607) == INT32_MIN);
    check(decode(32, 0, {31, 0}, 0) == -INT32_MAX);
    check(decode(16, 65535, {0}, 1) == 0);
}

void fixture(const std::string &mode, const char *path) {
    lazperf::reader::named_file reader(path);
    std::vector<char> point(reader.header().point_record_length);
    if (mode == "inflated") {
        reject([&] {
            for (uint64_t i = 0; i < reader.pointCount(); ++i)
                reader.readPoint(point.data());
        });
    } else {
        const std::array<int32_t, 5> expected{{INT32_MIN, INT32_MAX, -1, 0, 10000}};
        check(reader.pointCount() == 11);
        for (uint64_t i = 0; i < reader.pointCount(); ++i) {
            reader.readPoint(point.data());
            int32_t raw;
            std::memcpy(&raw, point.data(), sizeof(raw));
            check(raw == expected[i % expected.size()]);
        }
    }
}
} // namespace

int main(int argc, char **argv) {
    try {
        if (argc == 1) {
            memory();
            integer();
        } else if (argc == 3 &&
                   (std::string(argv[1]) == "extreme" || std::string(argv[1]) == "inflated")) {
            fixture(argv[1], argv[2]);
        } else {
            throw std::runtime_error("usage: codec-test [extreme|inflated fixture.laz]");
        }
        std::cout << "PASS laz-perf overlay regression\n";
        return 0;
    } catch (const std::exception &error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
