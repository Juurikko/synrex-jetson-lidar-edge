// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Tuomas Pietilä
#include "synrex/replay_manifest.hpp"
#include <algorithm>
#include <array>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <limits>
#include <nlohmann/json.hpp>
#include <openssl/evp.h>
#include <stdexcept>
#include <sys/resource.h>
#include <sys/stat.h>
#include <unistd.h>

namespace {
using Json = nlohmann::json;
using synrex::replay::Catalog;
namespace fs = std::filesystem;
void check(bool ok, const char *message) {
    if (!ok)
        throw std::runtime_error(message);
}
template <class F> void rejects(F function, const std::string &diagnostic = "") {
    try {
        function();
    } catch (const std::exception &error) {
        if (!diagnostic.empty() && std::string(error.what()).find(diagnostic) == std::string::npos)
            throw std::runtime_error("unexpected rejection: " + std::string(error.what()));
        return;
    }
    throw std::runtime_error("invalid replay manifest unexpectedly accepted");
}
struct Temporary {
    fs::path path;
    Temporary() {
        auto pattern = (fs::temp_directory_path() / "synrex-replay-manifest-XXXXXX").string();
        const auto created = ::mkdtemp(pattern.data());
        check(created, "temporary directory");
        path = created;
    }
    ~Temporary() {
        std::error_code ignored;
        fs::remove_all(path, ignored);
    }
};
void write(const fs::path &path, const std::string &bytes) {
    std::ofstream out(path, std::ios::binary);
    out.write(bytes.data(), static_cast<std::streamsize>(bytes.size()));
    out.close();
    check(bool(out), "fixture write");
}
std::string sha256(const std::string &bytes) {
    std::array<unsigned char, 32> digest{};
    unsigned length{};
    check(EVP_Digest(bytes.data(), bytes.size(), digest.data(), &length, EVP_sha256(), nullptr) ==
                  1 &&
              length == digest.size(),
          "test SHA256");
    const char *hex = "0123456789abcdef";
    std::string result;
    for (const auto byte : digest) {
        result += hex[byte >> 4];
        result += hex[byte & 15];
    }
    return result;
}
Json source(const std::string &path, const std::string &bytes, std::uint64_t count) {
    return {{"path", path},
            {"sha256", sha256(bytes)},
            {"records", count},
            {"coordinate_frame", "fixture.local.v1"},
            {"crs", "UNSPECIFIED"}};
}
Json manifest(const Json &sources) {
    return {{"schema", "synrex.distinct_replay.v1"},
            {"workload", "file_order_synthetic"},
            {"boundary_policy", "reset_each_file"},
            {"sources", sources}};
}
std::string read(const fs::path &path) {
    std::ifstream in(path, std::ios::binary);
    return std::string(std::istreambuf_iterator<char>(in), {});
}
} // namespace

int main() {
    try {
        Temporary temp;
        const auto file = temp.path / "manifest.json";
        const std::string a = "arbitrary bytes: not a decoded LAZ claim\n";
        const std::string b = "second metadata identity fixture\n";
        write(temp.path / "a.laz", a);
        write(temp.path / "b.laz", b);
        const auto good = manifest(Json::array({source("a.laz", a, 3), source("b.laz", b, 5)}));
        const auto bytes = good.dump();
        write(file, bytes);
        Catalog catalog(file);
        check(catalog.manifest_bytes() == bytes && catalog.manifest_sha256() == sha256(bytes),
              "exact manifest byte identity");
        check(catalog.total_records() == 8 && catalog.sources().size() == 2 &&
                  catalog.sources()[0].path == temp.path / "a.laz" &&
                  catalog.sources()[1].wire_begin == 3,
              "relative resolution and prefixes");
        for (std::uint64_t wire = 0; wire < catalog.total_records(); ++wire) {
            const auto location = catalog.source_for_wire_id(wire);
            check(catalog.wire_id(location.source_index, location.original_record_index) == wire,
                  "wire/source bijection");
        }
        check(catalog.source_for_wire_id(2).source_index == 0 &&
                  catalog.source_for_wire_id(3).source_index == 1 &&
                  catalog.source_for_wire_id(7).original_record_index == 4,
              "exact source boundary mapping");
        rejects([&] { catalog.wire_id(2, 0); }, "source index");
        rejects([&] { catalog.wire_id(0, 3); }, "original record index");
        rejects([&] { catalog.source_for_wire_id(8); }, "wire ID");
        rejects([&] { catalog.source_for_wire_id(UINT64_MAX); }, "wire ID");
        rejects([&] { catalog.open_source(2); }, "source index");
        {
            auto handle = catalog.open_source(0);
            check(read(handle.path()) == a && handle.fd() >= 0, "retained fd decoder ingress");
            char one{};
            check(::read(handle.fd(), &one, 1) == 1, "advance descriptor offset");
            handle.finish(); // verification is independent of descriptor offset
            auto moved = std::move(handle);
            rejects([&] { handle.finish(); }, "moved");
            moved.finish();
        }
        auto invalid = [&](auto mutate, const std::string &diagnostic = "") {
            auto value = good;
            mutate(value);
            write(file, value.dump());
            rejects([&] { Catalog ignored(file); }, diagnostic);
        };
        invalid([](Json &j) { j["extra"] = true; }, "unknown/missing");
        invalid([](Json &j) { j["sources"][0]["extra"] = true; }, "unknown/missing");
        invalid([](Json &j) { j.erase("schema"); }, "unknown/missing");
        invalid([](Json &j) { j["schema"] = "synrex.distinct_replay.v2"; }, "schema");
        invalid([](Json &j) { j["workload"] = "physical_scan"; }, "workload");
        invalid([](Json &j) { j["boundary_policy"] = "keep_state"; }, "boundary");
        invalid([](Json &j) { j["sources"] = Json::array(); }, "1..4096");
        invalid([](Json &j) { j["sources"] = Json::object(); }, "1..4096");
        invalid([](Json &j) {
            auto first = j["sources"][0];
            j["sources"] = Json::array();
            for (std::size_t i = 0; i <= Catalog::max_sources; ++i)
                j["sources"].push_back(first);
        });
        for (const Json &value :
             {Json(true), Json(-1), Json(0), Json(1.0), Json("1"), Json(nullptr)})
            invalid([&](Json &j) { j["sources"][0]["records"] = value; }, "records");
        invalid(
            [](Json &j) {
                j["sources"][0]["records"] = UINT64_MAX;
                j["sources"][1]["records"] = 1;
            },
            "overflow");
        for (const auto key : {"coordinate_frame", "crs"}) {
            for (const auto &value :
                 {std::string(), std::string(257, 'x'), std::string("bad\nID"),
                  std::string(1, '\0'), std::string(1, '\x7f'), std::string("ä")})
                invalid([&](Json &j) { j["sources"][0][key] = value; }, "identifier");
            invalid([&](Json &j) { j["sources"][0][key] = false; }, "identifier");
        }
        invalid([](Json &j) { j["sources"][0]["path"] = std::string("a\0.laz", 6); },
                "path size/NUL");
        invalid([](Json &j) { j["sources"][0]["sha256"] = std::string(64, 'A'); }, "SHA256");
        invalid([](Json &j) { j["sources"][0]["sha256"] = std::string(64, '0'); },
                "SHA256 mismatch");
        invalid([](Json &j) { j["sources"][1]["sha256"] = j["sources"][0]["sha256"]; },
                "duplicate source digest");
        write(temp.path / "copy.laz", a);
        invalid([&](Json &j) { j["sources"][1] = source("copy.laz", a, 5); },
                "duplicate source digest");
        fs::create_hard_link(temp.path / "a.laz", temp.path / "hard.laz");
        invalid([&](Json &j) { j["sources"][1] = source("hard.laz", a, 5); },
                "duplicate source digest");
        fs::remove(temp.path / "hard.laz");
        fs::create_symlink("a.laz", temp.path / "sym.laz");
        invalid([&](Json &j) { j["sources"] = Json::array({source("sym.laz", a, 1)}); },
                "non-symlink");
        invalid([](Json &j) { j["sources"][0]["path"] = "."; }, "regular");
        check(::mkfifo((temp.path / "pipe.laz").c_str(), 0600) == 0, "FIFO fixture");
        invalid([](Json &j) { j["sources"][0]["path"] = "pipe.laz"; }, "regular");
        write(temp.path / "empty.laz", "");
        invalid([&](Json &j) { j["sources"][0] = source("empty.laz", "", 1); }, "file size");
        write(temp.path / "large.laz", "x");
        fs::resize_file(temp.path / "large.laz", Catalog::max_source_bytes + 1);
        invalid([](Json &j) { j["sources"][0]["path"] = "large.laz"; }, "file size");
        fs::remove(temp.path / "large.laz");

        auto duplicate = bytes;
        duplicate.insert(1, R"("schema":"synrex.distinct_replay.v1",)");
        write(file, duplicate);
        rejects([&] { Catalog ignored(file); }, "duplicate JSON key");
        duplicate = bytes;
        duplicate.insert(duplicate.find("\"path\""), R"("path":"a.laz",)");
        write(file, duplicate);
        rejects([&] { Catalog ignored(file); }, "duplicate JSON key");
        write(file, bytes + std::string(1, '\0'));
        rejects([&] { Catalog ignored(file); }, "raw NUL");
        write(file, bytes + " true");
        rejects([&] { Catalog ignored(file); });
        write(file, std::string(Catalog::max_manifest_bytes + 1, ' '));
        rejects([&] { Catalog ignored(file); }, "file size");
        write(file, "");
        rejects([&] { Catalog ignored(file); }, "file size");
        fs::create_symlink("a.laz", temp.path / "manifest-link.json");
        rejects([&] { Catalog ignored(temp.path / "manifest-link.json"); }, "non-symlink");

        auto max_count = good;
        max_count["sources"][0]["records"] = UINT64_MAX - 5;
        write(file, max_count.dump());
        Catalog max_catalog(file);
        check(max_catalog.total_records() == UINT64_MAX &&
                  max_catalog.wire_id(1, 4) == UINT64_MAX - 1 &&
                  max_catalog.source_for_wire_id(UINT64_MAX - 1).original_record_index == 4,
              "UINT64_MAX declared-total boundary");
        rejects([&] { max_catalog.source_for_wire_id(UINT64_MAX); }, "wire ID");
        // No header or decoded-count inference: huge counts are metadata only.

        write(file, bytes + "\n");
        Catalog newline(file);
        check(newline.manifest_sha256() != catalog.manifest_sha256(), "exact serialization digest");
        // Catalog admission keeps no source descriptors; low fd limits permit
        // many sources. Every file has a distinct digest and tiny fixture bytes.
        Json many = Json::array();
        for (int i = 0; i < 80; ++i) {
            const auto name = "many-" + std::to_string(i);
            write(temp.path / name, name);
            many.push_back(source(name, name, 1));
        }
        write(file, manifest(many).dump());
        struct rlimit saved {};
        check(::getrlimit(RLIMIT_NOFILE, &saved) == 0, "read fd limit");
        auto bounded = saved;
        bounded.rlim_cur = std::min<rlim_t>(saved.rlim_cur, 48);
        check(::setrlimit(RLIMIT_NOFILE, &bounded) == 0, "set bounded fd limit");
        {
            Catalog serial(file);
            check(serial.total_records() == 80, "serial admission under fd limit");
            auto handle = serial.open_source(79);
            handle.finish();
        }
        check(::setrlimit(RLIMIT_NOFILE, &saved) == 0, "restore fd limit");

        // Replacing the pathname after admission must fail before decode.
        write(file, bytes);
        Catalog before_replace(file);
        fs::rename(temp.path / "a.laz", temp.path / "original.laz");
        write(temp.path / "a.laz", a); // same bytes, different inode
        rejects([&] { before_replace.open_source(0); }, "identity changed");
        fs::remove(temp.path / "a.laz");
        fs::rename(temp.path / "original.laz", temp.path / "a.laz");
        Catalog before_symlink(file);
        fs::rename(temp.path / "a.laz", temp.path / "original.laz");
        fs::create_symlink("original.laz", temp.path / "a.laz");
        rejects([&] { before_symlink.open_source(0); }, "non-symlink");
        fs::remove(temp.path / "a.laz");
        fs::rename(temp.path / "original.laz", temp.path / "a.laz");
        {
            Catalog pinned(file);
            auto handle = pinned.open_source(0);
            fs::rename(temp.path / "a.laz", temp.path / "original.laz");
            write(temp.path / "a.laz", "replacement bytes");
            check(read(handle.path()) == a, "decoder still consumes admitted inode");
            rejects([&] { handle.finish(); }, "identity changed");
            fs::remove(temp.path / "a.laz");
            fs::rename(temp.path / "original.laz", temp.path / "a.laz");
        }
        {
            Catalog pinned(file);
            auto handle = pinned.open_source(0);
            write(temp.path / "a.laz", std::string(a.size(), 'z'));
            rejects([&] { handle.finish(); }, "identity changed");
        }
        std::cout << "replay manifest: PASS (metadata identity, bounds, bijection, pinned-fd "
                     "mutation checks)\n";
        return 0;
    } catch (const std::exception &error) {
        std::cerr << "replay manifest: FAIL: " << error.what() << '\n';
        return 1;
    }
}
