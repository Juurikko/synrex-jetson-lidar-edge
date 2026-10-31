// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Tuomas Pietilä
#include "synrex/replay_reader.hpp"
#include <algorithm>
#include <cmath>
#include <cstring>
#include <fstream>
#include <lazperf/readers.hpp>

namespace synrex::replay {
namespace {
std::uint64_t little(const unsigned char *p, unsigned width) {
    std::uint64_t value{};
    for (unsigned i = 0; i < width; ++i)
        value |= std::uint64_t(p[i]) << (8 * i);
    return value;
}
double floating(const unsigned char *p) {
    const auto bits = little(p, 8);
    double value{};
    std::memcpy(&value, &bits, sizeof(value));
    return value;
}
SourceHandle validated_source(const Catalog &catalog, std::size_t index, std::size_t points) {
    require(points > 0 && points <= 65536, "replay window outside 1..65536");
    return catalog.open_source(index);
}
} // namespace
struct Reader::Impl {
    const Catalog &catalog;
    const std::size_t source_index, window_points;
    SourceHandle source;
    std::unique_ptr<lazperf::reader::named_file> decoder;
    std::vector<unsigned char> raw;
    std::array<double, 3> scales{}, offsets{};
    std::uint64_t decoded{}, raw_decoded{}, expected{};
    unsigned format{}, global_encoding{};
    bool eof{}, finished{}, failed{};
    Impl(const Catalog &c, std::size_t index, std::size_t points)
        : catalog(c), source_index(index), window_points(points),
          source(validated_source(c, index, points)) {
        expected = catalog.sources().at(index).records;
        const auto path = source.path();
        require(validate_laz(path) == expected, "replay declared/header point count mismatch");
        std::ifstream input(path, std::ios::binary);
        std::array<unsigned char, 375> header{};
        require(bool(input.read(reinterpret_cast<char *>(header.data()), header.size())),
                "replay truncated LAS header");
        format = header[104] & 63;
        global_encoding = little(header.data() + 6, 2);
        for (unsigned axis = 0; axis < 3; ++axis) {
            scales[axis] = floating(header.data() + 131 + axis * 8);
            offsets[axis] = floating(header.data() + 155 + axis * 8);
            require(std::isfinite(scales[axis]) && scales[axis] > 0 && std::isfinite(offsets[axis]),
                    "replay invalid LAS scale/offset");
        }
        raw.resize(little(header.data() + 105, 2));
        decoder = std::make_unique<lazperf::reader::named_file>(path.string());
        require(decoder->pointCount() == expected && decoder->header().point_format_id == format &&
                    decoder->header().point_record_length == raw.size(),
                "replay decoder/header mismatch");
    }
};
Reader::Reader(const Catalog &catalog, std::size_t index, std::size_t points)
    : impl_(std::make_unique<Impl>(catalog, index, points)) {}
Reader::~Reader() = default;
bool Reader::next(Window &window) {
    auto &s = *impl_;
    try {
        require(!s.failed && !s.finished, "replay reader failed or already finished");
        window.observations.clear();
        if (window.observations.capacity() > s.window_points)
            std::vector<Observation>().swap(window.observations);
        window.source_index = s.source_index;
        if (s.decoded == s.expected) {
            s.eof = true;
            return false;
        }
        window.observations.reserve(s.window_points);
        const auto count = std::min<std::uint64_t>(s.window_points, s.expected - s.decoded);
        for (std::uint64_t i = 0; i < count; ++i) {
            s.decoder->readPoint(reinterpret_cast<char *>(s.raw.data()));
            ++s.raw_decoded;
            Observation point;
            point.original_record_index = s.decoded;
            point.wire_id = s.catalog.wire_id(s.source_index, s.decoded);
            for (unsigned axis = 0; axis < 3; ++axis) {
                const auto bits = static_cast<std::uint32_t>(little(s.raw.data() + axis * 4, 4));
                std::memcpy(&point.integer_xyz[axis], &bits, sizeof(bits));
                point.xyz[axis] = point.integer_xyz[axis] * s.scales[axis] + s.offsets[axis];
                require(std::isfinite(point.xyz[axis]) && std::abs(point.xyz[axis]) <= 1e9,
                        "replay coordinate outside generic wire domain");
            }
            point.intensity = little(s.raw.data() + 12, 2);
            point.return_number = s.raw[14] & 15;
            point.return_count = s.raw[14] >> 4;
            point.gps_time_bits = little(s.raw.data() + 22, 8);
            window.observations.push_back(point);
            ++s.decoded;
        }
        return true;
    } catch (...) {
        s.failed = true;
        window.observations.clear();
        throw;
    }
}
void Reader::finish() {
    auto &s = *impl_;
    try {
        require(!s.failed && !s.finished && s.eof && s.decoded == s.expected,
                "replay reader requires complete decoded EOF before finish");
        s.decoder.reset();
        s.source.finish();
        s.finished = true;
    } catch (...) {
        s.failed = true;
        throw;
    }
}
Json Reader::metadata() const {
    const auto &s = *impl_;
    return {{"source_index", s.source_index},
            {"source_sha256", s.catalog.sources().at(s.source_index).sha256},
            {"coordinate_frame", s.catalog.sources().at(s.source_index).coordinate_frame},
            {"crs", s.catalog.sources().at(s.source_index).crs},
            {"expected_records", s.expected},
            {"decoded_records", s.decoded},
            {"raw_records_decoded", s.raw_decoded},
            {"point_format", s.format},
            {"scale", s.scales},
            {"offset", s.offsets},
            {"global_encoding", s.global_encoding},
            {"gps_clock", "raw LAS GPS double bits; no clock conversion or acquisition claim"},
            {"window_point_limit", s.window_points},
            {"output_window_bytes", s.window_points * sizeof(Observation)},
            {"raw_record_bytes", s.raw.size()},
            {"source_verified_after_decode", s.finished},
            {"physical_eof_verified", false},
            {"count_scope",
             "all declared header records decoded; extra compressed records not audited"}};
}
} // namespace synrex::replay
