// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Tuomas Pietilä
// Label-free CPU I/O utility. No model admission, inference or transport API.
#include "synrex/directory_publication.hpp"
#include "synrex/point_io.hpp"
#include <cstring>
#include <fstream>
#include <iostream>
#include <unistd.h>
using namespace synrex;
namespace {
void append(std::string &s, std::uint64_t v, unsigned width) {
    for (unsigned i = 0; i < width; ++i)
        s += static_cast<char>(v >> (8 * i));
}
Json decode_source(const point::PointSource &s, const fs::path &path) {
    const auto out = fs::absolute(path);
    require(!fs::exists(out) && !fs::is_symlink(out) && fs::is_directory(out.parent_path()),
            "decode destination must be new in existing parent");
    auto pattern =
        (out.parent_path() / ("." + out.filename().string() + ".point-decode-XXXXXX")).string();
    require(::mkdtemp(pattern.data()) != nullptr, "decode stage creation failed");
    const fs::path stage = pattern;
    bool published = false;
    try {
        std::string integers, xyz, intensity;
        integers.reserve(s.records().size() * 12);
        xyz.reserve(s.records().size() * 24);
        intensity.reserve(s.records().size() * 2);
        for (const auto &r : s.records()) {
            for (unsigned a = 0; a < 3; ++a) {
                append(integers, static_cast<std::uint32_t>(r.integer_xyz[a]), 4);
                std::uint64_t bits{};
                static_assert(sizeof(double) == 8);
                std::memcpy(&bits, &r.xyz[a], 8);
                append(xyz, bits, 8);
            }
            append(intensity, r.intensity, 2);
        }
        write_new(stage / "integer_xyz.i32le", integers);
        write_new(stage / "xyz.f64le", xyz);
        write_new(stage / "intensity.u16le", intensity);
        Json result{{"schema", "synrex.point-source-arrays.v1"},
                    {"source", s.metadata()},
                    {"scope", "label-free CPU decoding only; no model execution"},
                    {"arrays", Json::array()}};
        const std::array<std::array<const char *, 3>, 3> items{
            {{{"integer_xyz.i32le", "integer_xyz", "<i4"}},
             {{"xyz.f64le", "xyz", "<f8"}},
             {{"intensity.u16le", "intensity", "<u2"}}}};
        for (const auto &a : items) {
            Json shape = Json::array({s.records().size()});
            if (std::string(a[1]) != "intensity")
                shape.push_back(3);
            result["arrays"].push_back({{"path", a[0]},
                                        {"name", a[1]},
                                        {"dtype", a[2]},
                                        {"shape", shape},
                                        {"bytes", fs::file_size(stage / a[0])},
                                        {"sha256", sha256(stage / a[0])}});
        }
        write_new(stage / "SOURCE.json", result.dump(2) + "\n");
        sync_publication_stage(stage);
        s.verify_original();
        publish_directory_noreplace(stage, out);
        published = true;
        sync_publication_parent(out);
        return result;
    } catch (...) {
        if (!published) {
            std::error_code ec;
            fs::remove_all(stage, ec);
        }
        throw;
    }
}
} // namespace
int main(int argc, char **argv) {
    try {
        require(argc >= 3, "usage: synrex-point-source inspect INPUT | decode INPUT NEW_BUNDLE | "
                           "export INPUT SEMANTICS.u8 TAXONOMY.json NEW_BUNDLE");
        const std::string mode = argv[1];
        require((mode == "inspect" && argc == 3) || (mode == "decode" && argc == 4) ||
                    (mode == "export" && argc == 6),
                "unknown point source command/argument count");
        point::PointSource source(argv[2]);
        Json result;
        if (mode == "inspect")
            result = source.metadata();
        else if (mode == "decode")
            result = decode_source(source, argv[3]);
        else {
            const fs::path labels = argv[3];
            require(fs::is_regular_file(labels) && !fs::is_symlink(labels) &&
                        fs::file_size(labels) == source.records().size(),
                    "exact semantic file length required");
            std::ifstream f(labels, std::ios::binary);
            std::vector<std::uint8_t> values(source.records().size());
            require(bool(f.read(reinterpret_cast<char *>(values.data()), values.size())) &&
                        f.peek() == EOF,
                    "semantic read changed/truncated");
            result = source.export_semantics(argv[5], Taxonomy::read(argv[4]), values);
        }
        std::cout << result.dump(2) << '\n';
        return 0;
    } catch (const std::exception &e) {
        std::cerr << e.what() << '\n';
        return 1;
    }
}
