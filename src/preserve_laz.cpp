// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Tuomas Pietilä
// Developer and project lead: Tuomas Pietilä.
// AI-assisted software engineering, including OpenAI Codex.
// New public-focused staging implementation. Provenance: PUBLICATION_RIGHTS.md.
// Applies supplied uint8 labels; contains no neural model or inference implementation.
#include <lazperf/readers.hpp>
#include <lazperf/writers.hpp>

#include <algorithm>
#include <array>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

namespace fs = std::filesystem;
using Bytes = std::vector<unsigned char>;

static void require(bool ok, const char *message) {
    if (!ok)
        throw std::runtime_error(message);
}

static std::uint64_t little(const Bytes &b, std::size_t offset, std::size_t width) {
    require(width <= 8 && offset <= b.size() && width <= b.size() - offset,
            "truncated integer field");
    std::uint64_t value = 0;
    for (std::size_t i = 0; i < width; ++i)
        value |= std::uint64_t(b[offset + i]) << (8 * i);
    return value;
}

static void write_u64(std::ostream &out, std::uint64_t value) {
    std::array<char, 8> bytes{};
    for (unsigned i = 0; i < bytes.size(); ++i)
        bytes[i] = static_cast<char>((value >> (8 * i)) & 255);
    out.write(bytes.data(), bytes.size());
}

static Bytes read_bytes(std::istream &in, std::size_t count) {
    Bytes bytes(count);
    require(bool(in.read(reinterpret_cast<char *>(bytes.data()), count)), "truncated file");
    return bytes;
}

struct Vlr {
    std::size_t start;
    std::size_t length;
    bool compression;
};

struct Metadata {
    Bytes prefix;
    std::uint32_t point_offset{};
    std::uint64_t points{};
    unsigned format{}, record_length{};
    std::vector<Vlr> vlrs;
    std::size_t compression_index{};
};

static std::string user_id(const Bytes &bytes, std::size_t start) {
    std::string name(reinterpret_cast<const char *>(bytes.data() + start + 2), 16);
    while (!name.empty() && (name.back() == '\0' || name.back() == ' '))
        name.pop_back();
    return name;
}

static Metadata metadata(const fs::path &path) {
    std::ifstream in(path, std::ios::binary);
    require(bool(in), "cannot read input");
    Metadata m;
    m.prefix = read_bytes(in, 375);
    require(std::string(m.prefix.begin(), m.prefix.begin() + 4) == "LASF", "invalid LAS magic");
    require(m.prefix[24] == 1 && m.prefix[25] == 4 && little(m.prefix, 94, 2) == 375,
            "only standard LAS 1.4 public headers are supported");
    m.format = m.prefix[104] & 63;
    m.record_length = static_cast<unsigned>(little(m.prefix, 105, 2));
    const unsigned minimum = m.format == 6 ? 30 : m.format == 7 ? 36 : m.format == 8 ? 38 : 0;
    require((m.prefix[104] & 192) == 128 && minimum && m.record_length >= minimum,
            "requires compressed PDRF 6, 7 or 8 with valid record length");
    require(little(m.prefix, 227, 8) == 0 && little(m.prefix, 235, 8) == 0 &&
                little(m.prefix, 243, 4) == 0,
            "waveform data and EVLRs are outside this review utility contract");
    m.point_offset = static_cast<std::uint32_t>(little(m.prefix, 96, 4));
    m.points = little(m.prefix, 247, 8);
    require(m.points > 0, "empty point clouds are outside this review utility contract");
    require(m.point_offset >= 375 && m.point_offset <= 16 * 1024 * 1024,
            "invalid or oversized metadata prefix");
    require(fs::file_size(path) >= std::uint64_t(m.point_offset) + 8, "missing LAZ stream");
    auto tail = read_bytes(in, m.point_offset - 375);
    m.prefix.insert(m.prefix.end(), tail.begin(), tail.end());
    const auto count = little(m.prefix, 100, 4);
    require(count <= (m.point_offset - 375) / 54, "invalid VLR count");
    std::size_t offset = 375;
    unsigned compression_count = 0;
    for (std::uint64_t i = 0; i < count; ++i) {
        require(offset <= m.prefix.size() && m.prefix.size() - offset >= 54, "truncated VLR");
        const auto length = 54 + little(m.prefix, offset + 20, 2);
        require(length <= m.prefix.size() - offset, "truncated VLR payload");
        auto name = user_id(m.prefix, offset);
        require(name != "copc", "COPC spatial indexes are outside this utility contract");
        const bool compression =
            name == "laszip encoded" && little(m.prefix, offset + 18, 2) == 22204;
        if (compression) {
            ++compression_count;
            m.compression_index = m.vlrs.size();
        }
        m.vlrs.push_back({offset, static_cast<std::size_t>(length), compression});
        offset += length;
    }
    require(compression_count == 1, "requires exactly one LASzip compression VLR");
    // Remaining bytes before point_offset are retained as opaque metadata padding.
    return m;
}

class Scratch {
    fs::path path_;

  public:
    explicit Scratch(const fs::path &parent) {
        auto pattern = (parent / ".synrex-laz-review-XXXXXX").string();
        std::vector<char> buffer(pattern.begin(), pattern.end());
        buffer.push_back('\0');
        const auto p = ::mkdtemp(buffer.data());
        require(p != nullptr, "cannot create invocation-local scratch directory");
        path_ = p;
    }
    ~Scratch() {
        std::error_code ignored;
        fs::remove_all(path_, ignored);
    }
    Scratch(const Scratch &) = delete;
    Scratch &operator=(const Scratch &) = delete;
    fs::path file(const char *name) const { return path_ / name; }
};

static void stream_copy(std::istream &in, std::ostream &out) {
    std::array<char, 65536> buffer{};
    while (in.read(buffer.data(), buffer.size()) || in.gcount()) {
        out.write(buffer.data(), in.gcount());
        require(bool(out), "output stream write failed");
    }
    require(in.eof(), "input stream read failed");
}

static std::uint64_t apply(const fs::path &input, const fs::path &labels, const fs::path &output) {
    require(fs::is_regular_file(input) && fs::is_regular_file(labels),
            "requires regular input files");
    require(!fs::exists(output) && !fs::is_symlink(output), "refusing to overwrite output");
    const auto parent = output.has_parent_path() ? output.parent_path() : fs::path(".");
    require(fs::is_directory(parent), "output parent directory must exist");
    const auto source = metadata(input);
    require(fs::file_size(labels) == source.points, "label count must equal source point count");
    Scratch scratch(parent);
    const auto encoded = scratch.file("encoded.laz");
    const auto staged = scratch.file("result.laz");
    std::uint64_t changed = 0;
    {
        lazperf::reader::named_file reader(input.string());
        require(reader.pointCount() == source.points &&
                    reader.header().point_record_length == source.record_length &&
                    reader.header().point_format_id == source.format,
                "decoder/header contract mismatch");
        lazperf::writer::named_file::config config(reader.header());
        config.chunk_size = 50000;
        lazperf::writer::named_file writer(encoded.string(), config);
        std::ifstream label_stream(labels, std::ios::binary);
        std::vector<char> record(source.record_length);
        for (std::uint64_t i = 0; i < source.points; ++i) {
            reader.readPoint(record.data());
            char label{};
            require(bool(label_stream.get(label)), "label stream ended early");
            changed += record[16] != label;
            record[16] = label;
            writer.writePoint(record.data());
        }
        require(label_stream.peek() == std::char_traits<char>::eof(),
                "label stream grew during read");
        writer.close();
    }
    const auto generated = metadata(encoded);
    const auto &old_vlr = source.vlrs.at(source.compression_index);
    const auto &new_vlr = generated.vlrs.at(generated.compression_index);
    require(old_vlr.length == new_vlr.length,
            "compression VLR size changed; cannot preserve header");
    auto prefix = source.prefix;
    std::copy_n(generated.prefix.begin() + new_vlr.start, new_vlr.length,
                prefix.begin() + old_vlr.start);
    std::ifstream body(encoded, std::ios::binary);
    body.seekg(generated.point_offset);
    auto pointer = read_bytes(body, 8);
    auto table = little(pointer, 0, 8);
    require(table >= std::uint64_t(generated.point_offset) + 8 && table < fs::file_size(encoded) &&
                table <= std::uint64_t(std::numeric_limits<std::int64_t>::max()),
            "invalid generated chunk-table offset");
    const auto delta = std::int64_t(source.point_offset) - std::int64_t(generated.point_offset);
    const auto relocated = static_cast<std::int64_t>(table) + delta;
    require(relocated >= std::int64_t(source.point_offset) + 8,
            "invalid relocated chunk-table offset");
    {
        std::ofstream out(staged, std::ios::binary);
        require(bool(out), "cannot stage output");
        out.write(reinterpret_cast<const char *>(prefix.data()), prefix.size());
        write_u64(out, static_cast<std::uint64_t>(relocated));
        // Remaining compressed chunks and table are copied without reinterpretation.
        stream_copy(body, out);
        out.close();
        require(bool(out), "output close failed");
    }
    // Same-filesystem hard-link creation is atomic and cannot replace an existing pathname.
    // This is publication after write/close, not a power-loss-durability guarantee.
    fs::create_hard_link(staged, output);
    return changed;
}

// Reuse this audited MIT implementation in the native BYOM executable.
namespace synrex {
std::uint64_t validate_laz(const fs::path &input) { return metadata(input).points; }
std::uint64_t apply_labels(const fs::path &input, const fs::path &labels, const fs::path &output) {
    return apply(input, labels, output);
}
} // namespace synrex

#ifndef SYNREX_LAZ_LIBRARY
int main(int argc, char **argv) {
    if (argc != 4) {
        std::cerr << "Usage: synrex-laz-labels INPUT.laz LABELS.u8 OUTPUT.laz\n";
        return 2;
    }
    try {
        const auto changed = apply(argv[1], argv[2], argv[3]);
        std::cout << "classification_records_changed=" << changed << "\nstatus=PASS\n";
        return 0;
    } catch (const std::exception &error) {
        std::cerr << "status=FAIL reason=" << error.what() << '\n';
        return 1;
    }
}
#endif
