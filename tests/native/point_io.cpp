// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Tuomas Pietilä
// CPU synthetic source/preservation tests; no model/backend execution.
#include "synrex/point_io.hpp"
#include <fstream>
#include <iostream>
#include <lazperf/readers.hpp>
#include <lazperf/utils.hpp>
#include <lazperf/writers.hpp>
#include <unistd.h>
using namespace synrex;
using namespace synrex::point;
namespace synrex {
Json target_info() { throw std::runtime_error("UNREACHABLE GPU TARGET SENTINEL"); }
} // namespace synrex
unsigned checks{};
void check(bool v, const std::string &s) {
    ++checks;
    require(v, s);
}
template <class F> void rejects(F f) {
    bool failed = false;
    try {
        f();
    } catch (const std::exception &) {
        failed = true;
    }
    check(failed, "expected rejection");
}
void source(const fs::path &p, unsigned format, unsigned extra = 0, unsigned count = 5) {
    lazperf::writer::named_file::config c;
    c.pdrf = format;
    c.minor_version = 4;
    c.extra_bytes = extra;
    c.scale = {0.01, 0.02, 0.03};
    c.offset = {1000000, -1000000, 50};
    lazperf::writer::named_file w(p.string(), c);
    const unsigned length = (format == 6 ? 30 : format == 7 ? 36 : 38) + extra;
    std::vector<char> r(length);
    for (unsigned i = 0; i < count; ++i) {
        for (unsigned a = 0; a < 3; ++a) {
            const std::int32_t n = -100 + static_cast<std::int32_t>(i * (a + 1));
            const auto u = static_cast<std::uint32_t>(n);
            for (unsigned b = 0; b < 4; ++b)
                r[a * 4 + b] = static_cast<char>(u >> (b * 8));
        }
        r[12] = static_cast<char>(255);
        r[13] = static_cast<char>(255);
        r[14] = 0x21;
        r[15] = static_cast<char>(i);
        r[16] = 9;
        for (unsigned b = 17; b < length; ++b)
            r[b] = static_cast<char>((i + b) % 256);
        lazperf::utils::pack(1000.0 + i / 1024.0, r.data() + 22);
        w.writePoint(r.data());
    }
    w.close();
}
Taxonomy taxonomy() {
    return Taxonomy(
        Json{{"schema", "synrex.taxonomy.v1"},
             {"taxonomy_id", "synthetic"},
             {"namespace", "example.point"},
             {"classes",
              Json::array(
                  {{{"semantic_id", 0}, {"key", "first"}, {"name", "First"}, {"las_code", 2}},
                   {{"semantic_id", 1}, {"key", "second"}, {"name", "Second"}, {"las_code", 2}}})},
             {"unknown_semantic_id", nullptr},
             {"provenance", {{"source", "synthetic"}, {"author", "SynRex"}, {"license", "MIT"}}}}
            .dump());
}
int main(int argc, char **argv) {
    fs::path root;
    try {
        require(argc == 2, "test build-root argument");
        auto tmp = (fs::path(argv[1]) / "point-io-XXXXXX").string();
        require(::mkdtemp(tmp.data()) != nullptr, "temporary root");
        root = tmp;
        const auto tax = taxonomy();
        for (unsigned format : {6, 7, 8}) {
            const auto input = root / ("input" + std::to_string(format) + ".laz");
            source(input, format);
            PointSource s(input);
            check(s.records().size() == 5, "source row count");
            check(s.sha256() == sha256(input) && s.header_sha256().size() == 64, "retained hashes");
            check(s.metadata()["point_format"] == format, "format identity");
            for (unsigned i = 0; i < 5; ++i) {
                const auto &r = s.records()[i];
                check(r.integer_xyz[0] == -100 + static_cast<int>(i),
                      "raw signed integer preserved");
                check(r.xyz[0] == r.integer_xyz[0] * 0.01 + 1000000 && r.intensity == 65535,
                      "double scale and full intensity");
            }
            rejects([&] { PointSource limited(input, 4); });
            rejects([&] { s.export_semantics(root / "partial", tax, {0, 1}); });
            rejects([&] { s.export_semantics(root / "invalid", tax, {0, 1, 2, 0, 1}); });
            const auto output = root / ("output" + std::to_string(format));
            const std::vector<std::uint8_t> semantics{0, 1, 0, 1, 0};
            const auto receipt = s.export_semantics(output, tax, semantics);
            check(receipt["record_preservation"] == "PASS", "writer verification");
            std::ifstream sem(output / "semantics.u8", std::ios::binary);
            const std::string bytes{std::istreambuf_iterator<char>(sem), {}};
            check(bytes == std::string(semantics.begin(), semantics.end()),
                  "ground0 and many-to-one semantics retained");
            lazperf::reader::named_file original(input.string()),
                classified((output / "classified.laz").string());
            const auto n = original.header().point_record_length;
            std::vector<char> a(n), b(n);
            for (unsigned i = 0; i < 5; ++i) {
                original.readPoint(a.data());
                classified.readPoint(b.data());
                check(b[16] == 2, "LAS projection");
                b[16] = a[16];
                check(a == b, "independent preservation");
            }
            rejects([&] { s.export_semantics(output, tax, semantics); });
            const auto extra = root / ("extra" + std::to_string(format) + ".laz");
            source(extra, format, 3);
            rejects([&] { PointSource rejected(extra); });
            auto changed = root / ("renamed" + std::to_string(format) + ".laz");
            fs::rename(input, changed);
            source(input, format, 0, 6);
            check(s.records().size() == 5, "sealed source survives pathname replacement");
            rejects([&] { s.verify_original(); });
            rejects([&] { s.export_semantics(root / "changed", tax, semantics); });
            const auto link = root / ("link" + std::to_string(format));
            fs::create_symlink(input, link);
            rejects([&] { PointSource rejected(link); });
        }
        auto big = root / "declared-big.laz";
        source(big, 7);
        {
            std::fstream f(big, std::ios::in | std::ios::out | std::ios::binary);
            f.seekp(247);
            const std::uint64_t n = 1000001;
            for (unsigned i = 0; i < 8; ++i)
                f.put(static_cast<char>(n >> (8 * i)));
        }
        rejects([&] { PointSource rejected(big); });
        auto bad = root / "bad-scale.laz";
        source(bad, 7);
        {
            std::fstream f(bad, std::ios::in | std::ios::out | std::ios::binary);
            f.seekp(131);
            for (unsigned i = 0; i < 8; ++i)
                f.put(0);
        }
        rejects([&] { PointSource rejected(bad); });
        fs::remove_all(root);
        std::cout << "point_io_checks=" << checks << " status=PASS fixture=true\n";
        return 0;
    } catch (const std::exception &e) {
        if (!root.empty()) {
            std::error_code ec;
            fs::remove_all(root, ec);
        }
        std::cerr << e.what() << '\n';
        return 1;
    }
}
