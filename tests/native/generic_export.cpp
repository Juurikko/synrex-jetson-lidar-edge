// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Tuomas Pietilä
// CPU-only preservation/export tests. Opaque models are never executed.
#include "synrex/generic_export.hpp"
#include <algorithm>
#include <cerrno>
#include <fstream>
#include <iostream>
#include <lazperf/readers.hpp>
#include <lazperf/utils.hpp>
#include <lazperf/writers.hpp>
#include <poll.h>
#include <set>
#include <sys/inotify.h>
#include <sys/stat.h>
#include <thread>
#include <unistd.h>

namespace {
using namespace synrex;
namespace g = synrex::generic;
unsigned checks{}, rejections{}, packages{}, target_queries{};
bool fail_parent_sync{};
dev_t sync_device{};
ino_t sync_inode{};
void check(bool okay, const std::string &message) {
    ++checks;
    require(okay, message);
}
template <class F> void rejects(F f, const std::string &message = {}) {
    try {
        f();
    } catch (const std::exception &error) {
        check(message.empty() || std::string(error.what()).find(message) != std::string::npos,
              "unexpected rejection: " + std::string(error.what()));
        ++rejections;
        return;
    }
    throw std::runtime_error("unexpected acceptance: " + message);
}
void write(const fs::path &path, const std::string &bytes) {
    std::ofstream out(path, std::ios::binary);
    out.write(bytes.data(), bytes.size());
    out.close();
    check(bool(out), "fixture write");
}
std::string bytes(const fs::path &path) {
    std::ifstream in(path, std::ios::binary);
    return {std::istreambuf_iterator<char>(in), std::istreambuf_iterator<char>()};
}
struct Temporary {
    fs::path path;
    explicit Temporary(const fs::path &parent) {
        auto pattern = (parent / "generic-export-XXXXXX").string();
        const auto created = ::mkdtemp(pattern.data());
        check(created, "temporary test directory");
        path = created;
    }
    ~Temporary() {
        std::error_code ignored;
        fs::remove_all(path, ignored);
    }
};
fs::path package(const fs::path &parent, Json contract, unsigned count,
                 const std::string &meaning = "Export meaning ", bool reordered = false) {
    const auto path = parent / ("package-" + std::to_string(++packages));
    fs::create_directory(path);
    Json classes = Json::array(), projection = Json::array();
    for (unsigned i = 0; i < count; ++i) {
        const unsigned las = 40 + i % 3; // Deliberately many-to-one LAS projection.
        classes.push_back({{"semantic_id", i},
                           {"key", "private_" + std::to_string(i)},
                           {"name", meaning + std::to_string(i)},
                           {"las_code", las}});
        projection.push_back(las);
    }
    if (reordered) {
        std::swap(classes[0]["name"], classes[3]["name"]);
        std::swap(classes[0]["key"], classes[3]["key"]);
    }
    const Taxonomy taxonomy(
        Json{{"schema", "synrex.taxonomy.v1"},
             {"taxonomy_id", "example.reused.v1"},
             {"namespace", "example.export"},
             {"classes", classes},
             {"unknown_semantic_id", nullptr},
             {"provenance",
              {{"source", "CPU export fixture"}, {"author", "SynRex"}, {"license", "MIT"}}}}
            .dump());
    contract["las_classes"] = projection;
    contract["output"]["shape"][1] = count;
    write(path / "model.onnx", "Opaque model fixture; never executed");
    write(path / "MODEL_LICENSE.txt", "MIT");
    write(path / "taxonomy.json", taxonomy.raw_bytes());
    write(path / "manifest.json",
          Json{{"schema", "synrex.srxmodel.v3"},
               {"contract", contract},
               {"artifacts",
                {{"onnx", {{"path", "model.onnx"}, {"sha256", sha256(path / "model.onnx")}}},
                 {"license",
                  {{"path", "MODEL_LICENSE.txt"}, {"sha256", sha256(path / "MODEL_LICENSE.txt")}}},
                 {"taxonomy", {{"path", "taxonomy.json"}, {"sha256", taxonomy.sha256()}}}}},
               {"target", nullptr}}
              .dump());
    return path;
}
g::Descriptor descriptor(const fs::path &model) {
    const g::CpuAdmittedPackage admitted(model);
    return g::Descriptor::from_package(admitted, {65536, 16777216, 30000}, std::string(32, 'a'), 1);
}
void source(const fs::path &path, int pdrf, unsigned count = 260) {
    lazperf::writer::named_file::config config;
    config.pdrf = pdrf;
    config.minor_version = 4;
    config.extra_bytes = 3;
    config.scale = {0.01, 0.02, 0.03};
    config.offset = {1000, -1000, 50};
    lazperf::writer::named_file writer(path.string(), config);
    const auto length = (pdrf == 6 ? 30 : pdrf == 7 ? 36 : 38) + 3;
    std::vector<char> record(length);
    for (unsigned i = 0; i < count; ++i) {
        for (int axis = 0; axis < 3; ++axis) {
            const std::uint32_t value = i * (axis + 1);
            for (unsigned b = 0; b < 4; ++b)
                record[axis * 4 + b] = static_cast<char>(value >> (8 * b));
        }
        const unsigned intensity = i % 2 ? 65535 : i;
        record[12] = static_cast<char>(intensity);
        record[13] = static_cast<char>(intensity >> 8);
        record[14] = static_cast<char>(0x21 + i % 2);
        record[15] = static_cast<char>(i % 16);
        record[16] = static_cast<char>(2);
        for (int b = 17; b < length; ++b)
            record[b] = static_cast<char>((i + b) % 256);
        // PDRF6/7/8 stores GPS time at bytes22..29. Use bounded, variable,
        // binary-exact synthetic times instead of arbitrary double bit patterns.
        const double gps_time = 1000000.0 + static_cast<double>(i) / 1024.0;
        lazperf::utils::pack(gps_time, record.data() + 22);
        writer.writePoint(record.data());
    }
    writer.close();
}
g::Classification classification(const fs::path &input, const fs::path &model,
                                 const g::Descriptor &d) {
    const Package p(model);
    const auto points = decode(input, p.contract);
    g::Classification c;
    c.frame.identity = {d.sha256(), d.spec().at("session_id"), d.spec().at("map_epoch")};
    c.frame.sensor_id = "fixture-sensor";
    c.frame.stream_id = "fixture-stream";
    c.frame.coordinate_frame = "original";
    c.frame.crs = "LOCAL:fixture";
    c.frame.frame_start_ns = 100;
    c.frame.frame_end_ns = 200;
    c.frame.frame_id = 9;
    c.frame.sequence = 3;
    const auto k = d.taxonomy().classes().size();
    for (std::size_t i = 0; i < points.size(); ++i) {
        c.frame.points.push_back({i, 100 + i % 100, points[i].xyz, points[i].intensity,
                                  static_cast<std::uint8_t>(1 + i % 2), 2});
        c.semantic_ids.push_back(static_cast<std::uint8_t>(i % k));
        c.las_codes.push_back(d.taxonomy().classes()[i % k].las_code);
    }
    return c;
}
void preservation(const fs::path &input, const fs::path &output,
                  const std::vector<std::uint8_t> &labels) {
    lazperf::reader::named_file original(input.string()), classified(output.string());
    check(original.pointCount() == labels.size() && classified.pointCount() == labels.size(),
          "exact output record count");
    const auto length = original.header().point_record_length;
    check(length == classified.header().point_record_length &&
              original.header().point_format_id == classified.header().point_format_id,
          "preserved PDRF and record length");
    std::vector<char> a(length), b(length);
    for (std::size_t i = 0; i < labels.size(); ++i) {
        original.readPoint(a.data());
        classified.readPoint(b.data());
        check(static_cast<std::uint8_t>(b[16]) == labels[i], "exact projected output label");
        b[16] = a[16];
        check(a == b, "all other original point bytes preserved incl return/rgb/extra bytes");
    }
}
void no_stage(const fs::path &parent) {
    std::set<fs::path> owned;
    for (const auto &entry : fs::directory_iterator(parent)) {
        if (fs::is_symlink(entry.symlink_status())) {
            const auto target = fs::read_symlink(entry.path());
            if (!target.has_parent_path())
                owned.insert(target);
        }
    }
    for (const auto &entry : fs::directory_iterator(parent))
        check(entry.path().filename().string().find(".staging-") == std::string::npos ||
                  owned.count(entry.path().filename()),
              "no abandoned export stage; published backing stages remain owned");
}
} // namespace
extern "C" int __real_fsync(int);
extern "C" int __wrap_fsync(int fd) {
    struct stat info {};
    if (fail_parent_sync && ::fstat(fd, &info) == 0 && info.st_dev == sync_device &&
        info.st_ino == sync_inode) {
        fail_parent_sync = false;
        errno = EIO;
        return -1;
    }
    return __real_fsync(fd);
}
namespace synrex {
Json target_info() {
    ++target_queries;
    throw std::runtime_error("UNREACHABLE GPU TARGET SENTINEL");
}
} // namespace synrex
int main(int argc, char **argv) {
    try {
        check(argc == 3 || (argc == 4 && std::string(argv[3]) == "--publication-filesystem"),
              "usage: generic-export-tests CONTRACT SCRATCH_PARENT [--publication-filesystem]");
        const bool publication_filesystem = argc == 4;
        Temporary temporary(argv[2]);
        const auto base = read_json(argv[1]);
        for (const unsigned k : {4, 7, 256}) {
            const auto model = package(temporary.path, base, k);
            const auto d = descriptor(model);
            const auto input = temporary.path / ("source-" + std::to_string(k) + ".laz");
            const int pdrf = k == 4 ? 6 : k == 7 ? 7 : 8;
            source(input, pdrf);
            const auto original_sha = sha256(input);
            auto c = classification(input, model, d);
            const auto destination = temporary.path / ("complete-" + std::to_string(k));
            g::LazExport good(input, model, destination, d);
            check(good.source_sha256() == original_sha,
                  "sealed source digest available for upstream binding");
            const auto entries =
                std::distance(fs::directory_iterator(temporary.path), fs::directory_iterator());
            // Deliberately reverse local rows: sidecar output must remain in
            // original file order, preserving distinct semantics with equal LAS.
            auto reverse = c;
            std::reverse(reverse.frame.points.begin(), reverse.frame.points.end());
            std::reverse(reverse.semantic_ids.begin(), reverse.semantic_ids.end());
            std::reverse(reverse.las_codes.begin(), reverse.las_codes.end());
            good.observe(reverse);
            check(!fs::exists(destination) && std::distance(fs::directory_iterator(temporary.path),
                                                            fs::directory_iterator()) == entries,
                  "constructor/observe have no filesystem output");
            const auto receipt = good.finish();
            check(receipt.at("coverage") == 1 && receipt.at("points") == c.frame.points.size(),
                  "full single-file coverage");
            const auto sidecar = read_json(destination / "semantics.json");
            check(sidecar.at("semantic_ids").get<std::vector<std::uint8_t>>() == c.semantic_ids &&
                      bytes(destination / "labels.u8") ==
                          std::string(reinterpret_cast<const char *>(c.las_codes.data()),
                                      c.las_codes.size()),
                  "distinct original semantics and LAS projections in original row order");
            check(bytes(destination / "descriptor.json") == d.bytes() &&
                      bytes(destination / "taxonomy.json") == d.taxonomy().raw_bytes() &&
                      bytes(destination / "manifest.json") == bytes(model / "manifest.json"),
                  "exact captured descriptor/taxonomy/manifest bytes");
            for (const auto &entry : receipt.at("files").items())
                check(sha256(destination / entry.key()) == entry.value(), "bundle file SHA256");
            check(receipt.at("binding").at("source_sha256") == original_sha &&
                      receipt.at("binding").at("descriptor_sha256") == d.sha256() &&
                      sha256(input) == original_sha,
                  "exact source and full descriptor binding; unchanged original input");
            preservation(input, destination / "classified.laz", c.las_codes);
            rejects([&] { good.finish(); }, "finished");
            rejects([&] { good.observe(c); }, "failed");
            rejects([&] { g::LazExport exists(input, model, destination, d); }, "new bundle");
            if (k == 256)
                check(sidecar.at("semantic_ids").at(255) == 255,
                      "semantic ID255 remains an ordinary explicit taxonomy class");
            no_stage(temporary.path);
        }
        const auto model = package(temporary.path, base, 4);
        const auto d = descriptor(model);
        const auto input = temporary.path / "source-invalid.laz";
        source(input, 6);
        const auto c = classification(input, model, d);
        unsigned cases{};
        const auto invalid = [&](const g::Classification &bad, const std::string &message) {
            const auto path = temporary.path / ("invalid-" + std::to_string(++cases));
            g::LazExport e(input, model, path, d);
            rejects([&] { e.observe(bad); }, message);
            rejects([&] { e.observe(c); }, "failed");
            rejects([&] { e.finish(); }, "failed");
            check(!fs::exists(path), "invalid observation never publishes");
        };
        {
            auto bad = c;
            bad.las_codes[0] = 99;
            invalid(bad, "projection");
        }
        {
            auto bad = c;
            bad.semantic_ids[0] = 255;
            invalid(bad, "outside taxonomy");
        }
        {
            auto bad = c;
            bad.frame.identity.descriptor_sha256[0] = '0';
            invalid(bad, "mismatch");
        }
        {
            auto bad = c;
            bad.frame.identity.map_epoch = 2;
            invalid(bad, "mismatch");
        }
        {
            auto bad = c;
            bad.frame.points[0].id = UINT64_MAX;
            invalid(bad, "local original");
        }
        {
            auto bad = c;
            bad.frame.points[0].id = c.frame.points.size();
            invalid(bad, "local original");
        }
        {
            auto bad = c;
            bad.frame.points[0].xyz[0] += 1;
            invalid(bad, "xyz/intensity");
        }
        {
            auto bad = c;
            bad.frame.points[0].intensity.reset();
            invalid(bad, "xyz/intensity");
        }
        {
            auto bad = c;
            bad.frame.points[0].intensity = 17;
            invalid(bad, "xyz/intensity");
        }
        {
            auto bad = c;
            bad.frame.points[0].return_number = 2;
            invalid(bad, "return metadata");
        }
        {
            auto bad = c;
            bad.frame.points[1].id = bad.frame.points[0].id;
            invalid(bad, "duplicate");
        }
        {
            g::LazExport e(input, model, temporary.path / "duplicate", d);
            e.observe(c);
            rejects([&] { e.observe(c); }, "duplicate");
            rejects([&] { e.finish(); }, "failed");
            check(!fs::exists(temporary.path / "duplicate"), "duplicate poisons complete coverage");
        }
        {
            g::LazExport e(input, model, temporary.path / "partial", d);
            auto one = c;
            one.frame.points.resize(1);
            one.semantic_ids.resize(1);
            one.las_codes.resize(1);
            e.observe(one);
            rejects([&] { e.finish(); }, "partial");
            check(!fs::exists(temporary.path / "partial"), "partial cannot publish");
        }
        {
            g::LazExport e(input, model, temporary.path / "metadata", d);
            auto a = c, b = c;
            a.frame.points.resize(1);
            a.semantic_ids.resize(1);
            a.las_codes.resize(1);
            e.observe(a);
            b.frame.points.erase(b.frame.points.begin());
            b.semantic_ids.erase(b.semantic_ids.begin());
            b.las_codes.erase(b.las_codes.begin());
            b.frame.crs = "LOCAL:changed";
            rejects([&] { e.observe(b); }, "metadata identity");
        }
        for (const bool reordered : {false, true}) {
            const auto changed =
                package(temporary.path, base, 4, reordered ? "Export meaning " : "Changed meaning ",
                        reordered);
            const auto other = descriptor(changed);
            check(other.taxonomy().taxonomy_id() == d.taxonomy().taxonomy_id() &&
                      other.spec().at("las_projection") == d.spec().at("las_projection"),
                  "same K/id/LAS changed-meaning fixture");
            rejects([&] { g::LazExport e(input, changed, temporary.path / "wrong-taxonomy", d); },
                    "identity mismatch");
        }
        for (const auto *artifact :
             {"model.onnx", "taxonomy.json", "manifest.json", "MODEL_LICENSE.txt"}) {
            const auto isolated = package(temporary.path, base, 4);
            const auto exact = descriptor(isolated);
            const auto output = temporary.path / ("mutated-" + std::to_string(++cases));
            g::LazExport e(input, isolated, output, exact);
            e.observe(classification(input, isolated, exact));
            write(isolated / artifact, bytes(isolated / artifact) + " ");
            rejects([&] { e.finish(); });
            check(!fs::exists(output), "mutated package artifact never publishes");
        }
        {
            const auto mutable_source = temporary.path / "mutable-source.laz";
            fs::copy_file(input, mutable_source);
            g::LazExport e(mutable_source, model, temporary.path / "changed-source", d);
            e.observe(c);
            write(mutable_source, bytes(mutable_source) + "changed");
            rejects([&] { e.finish(); }, "source changed");
            check(!fs::exists(temporary.path / "changed-source"), "mutated source never publishes");
        }
        {
            const auto output = temporary.path / "competitor";
            g::LazExport e(input, model, output, d);
            e.observe(c);
            fs::create_directory(output);
            write(output / "sentinel", "competitor original");
            rejects([&] { e.finish(); }, "refusing replacement");
            check(bytes(output / "sentinel") == "competitor original",
                  "existing product untouched");
        }
        if (!publication_filesystem) {
            // A real source mutation after staging has started must be noticed
            // before publication. Encoding uses the sealed snapshot, so it must
            // neither consume changed source bytes nor publish stale provenance.
            const auto live = temporary.path / "late-source.laz";
            source(live, 8, 20000);
            const auto all = classification(live, model, d);
            const auto output = temporary.path / "late-source-output";
            g::LazExport e(live, model, output, d);
            e.observe(all);
            const int events = ::inotify_init1(IN_CLOEXEC);
            check(events >= 0, "source mutation event stream");
            check(::inotify_add_watch(events, temporary.path.c_str(), IN_CREATE) >= 0,
                  "source mutation watch");
            std::exception_ptr watcher_error;
            std::thread watcher([&] {
                try {
                    bool mutated = false;
                    while (!mutated) {
                        pollfd wait{events, POLLIN, 0};
                        require(::poll(&wait, 1, 5000) == 1, "late source mutation event timeout");
                        alignas(inotify_event) std::array<char, 8192> buffer{};
                        const auto size = ::read(events, buffer.data(), buffer.size());
                        require(size > 0, "late source mutation event read");
                        for (ssize_t offset = 0; offset < size;) {
                            const auto *event =
                                reinterpret_cast<const inotify_event *>(buffer.data() + offset);
                            const std::string name(event->len ? event->name : "");
                            if (name.find(".late-source-output.staging-") == 0) {
                                std::ofstream append(live, std::ios::binary | std::ios::app);
                                append << "late source mutation";
                                append.close();
                                require(bool(append), "late source mutation write");
                                mutated = true;
                            }
                            offset += sizeof(inotify_event) + event->len;
                        }
                    }
                } catch (...) {
                    watcher_error = std::current_exception();
                }
            });
            std::exception_ptr export_error;
            try {
                e.finish();
            } catch (...) {
                export_error = std::current_exception();
            }
            watcher.join();
            ::close(events);
            if (watcher_error)
                std::rethrow_exception(watcher_error);
            check(bool(export_error), "post-stage source mutation must fail");
            rejects([&] { std::rethrow_exception(export_error); }, "source changed");
            check(!fs::exists(output), "post-stage source mutation never publishes");
            no_stage(temporary.path);
        }
        {
            const auto oversized = temporary.path / "oversized-source.laz";
            write(oversized, "sparse fixture");
            fs::resize_file(oversized, 256ULL * 1024 * 1024 + 1);
            rejects([&] { g::LazExport e(oversized, model, temporary.path / "oversized", d); },
                    "<=256MiB");
            fs::remove(oversized);
            auto header = bytes(input);
            const std::uint64_t count = 1000001;
            for (unsigned i = 0; i < 8; ++i)
                header[247 + i] = static_cast<char>(count >> (8 * i));
            const auto overcount = temporary.path / "overcount-source.laz";
            write(overcount, header);
            rejects([&] { g::LazExport e(overcount, model, temporary.path / "overcount", d); },
                    "1..1000000");
        }
        {
            const auto link = temporary.path / "source-symlink.laz";
            fs::create_symlink(input, link);
            rejects([&] { g::LazExport e(link, model, temporary.path / "symlink-source", d); });
            const auto linkout = temporary.path / "output-symlink";
            fs::create_symlink(temporary.path / "nonexistent-target", linkout);
            rejects([&] { g::LazExport e(input, model, linkout, d); }, "new bundle");
        }
        {
            struct stat parent {};
            check(::stat(temporary.path.c_str(), &parent) == 0, "parent fsync injection identity");
            sync_device = parent.st_dev;
            sync_inode = parent.st_ino;
            const auto output = temporary.path / "post-publication-fsync-error";
            g::LazExport e(input, model, output, d);
            e.observe(c);
            fail_parent_sync = true;
            rejects([&] { e.finish(); }, "durability");
            check(!fail_parent_sync && fs::is_directory(output),
                  "post-publication export fsync failure preserves committed output");
            const auto receipt = read_json(output / "bundle.json");
            for (const auto &entry : receipt.at("files").items())
                check(sha256(output / entry.key()) == entry.value(),
                      "post-publication exception retains every export byte");
            if (fs::is_symlink(output))
                check(fs::is_directory(output.parent_path() / fs::read_symlink(output)),
                      "post-publication exception never deletes backing stage");

            const auto spec = temporary.path / "pack-spec.json";
            write(spec, read_json(model / "manifest.json").at("contract").dump());
            const auto v2 = temporary.path / "packed-v2";
            pack(spec, model / "model.onnx", "onnx", model / "MODEL_LICENSE.txt", v2);
            const Package packed_v2(v2);
            check(packed_v2.manifest.at("schema") == "synrex.srxmodel.v2" &&
                      sha256(packed_v2.artifact("onnx")) == sha256(model / "model.onnx"),
                  "actual filesystem v2 pack admission reads complete bytes");
            const auto v3 = temporary.path / "packed-v3";
            pack(spec, model / "model.onnx", "onnx", model / "MODEL_LICENSE.txt", v3,
                 model / "taxonomy.json");
            check(Package(v3).taxonomy->sha256() == d.taxonomy().sha256(),
                  "actual filesystem v3 pack admission reads taxonomy");
            const auto repacked = temporary.path / "repacked-v3";
            repack(v2, model / "taxonomy.json", repacked);
            check(Package(repacked).taxonomy->sha256() == d.taxonomy().sha256(),
                  "actual filesystem repack admission reads taxonomy");
            for (unsigned mode = 0; mode < 3; ++mode) {
                const auto failed_output =
                    temporary.path / ("pack-parent-sync-error-" + std::to_string(mode));
                fail_parent_sync = true;
                rejects(
                    [&] {
                        if (mode == 0)
                            pack(spec, model / "model.onnx", "onnx", model / "MODEL_LICENSE.txt",
                                 failed_output);
                        else if (mode == 1)
                            pack(spec, model / "model.onnx", "onnx", model / "MODEL_LICENSE.txt",
                                 failed_output, model / "taxonomy.json");
                        else
                            repack(v2, model / "taxonomy.json", failed_output);
                    },
                    "durability");
                check(!fail_parent_sync && fs::is_directory(failed_output) &&
                          sha256(Package(failed_output).artifact("onnx")) ==
                              sha256(model / "model.onnx"),
                      "post-publication package fsync exception retains admitted bytes");
            }
        }
        no_stage(temporary.path);
        check(target_queries == 0, "CPU export never queries GPU target");
        std::cout << Json{{"status", "PASS_CPU_GENERIC_EXPORT"},
                          {"checks", checks},
                          {"expected_rejections", rejections},
                          {"gpu_target_calls", target_queries},
                          {"class_counts", {4, 7, 256}},
                          {"pdrf", {6, 7, 8}},
                          {"qualification_mode",
                           publication_filesystem ? "PUBLICATION_FILESYSTEM" : "FULL_CPU_EXPORT"},
                          {"post_stage_mutation_inotify",
                           publication_filesystem ? "NOT_RUN_EXPLICIT_PUBLICATION_FILESYSTEM_MODE"
                                                  : "PASS"},
                          {"scope", "real preservation writer; opaque CPU fixtures; no "
                                    "quality/throughput/transport qualification"}}
                         .dump()
                  << '\n';
        return 0;
    } catch (const std::exception &error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
