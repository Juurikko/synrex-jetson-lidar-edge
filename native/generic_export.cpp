// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Tuomas Pietilä
#include "synrex/generic_export.hpp"
#include <algorithm>
#include <cerrno>
#include <fcntl.h>
#include <lazperf/readers.hpp>
#include <linux/fs.h>
#include <linux/memfd.h>
#include <openssl/evp.h>
#include <sys/stat.h>
#include <sys/syscall.h>
#include <unistd.h>

namespace synrex::generic {
namespace {
constexpr std::uint64_t max_source_bytes = 256ULL * 1024 * 1024;
struct Fd {
    int value{-1};
    explicit Fd(int fd) : value(fd) { require(fd >= 0, "export open/capture failure"); }
    ~Fd() { ::close(value); }
    Fd(const Fd &) = delete;
    Fd &operator=(const Fd &) = delete;
};
bool same(const struct stat &a, const struct stat &b) {
    return a.st_dev == b.st_dev && a.st_ino == b.st_ino && a.st_size == b.st_size &&
           a.st_mode == b.st_mode && a.st_nlink == b.st_nlink &&
           a.st_mtim.tv_sec == b.st_mtim.tv_sec && a.st_mtim.tv_nsec == b.st_mtim.tv_nsec &&
           a.st_ctim.tv_sec == b.st_ctim.tv_sec && a.st_ctim.tv_nsec == b.st_ctim.tv_nsec;
}
// Capture into a sealable anonymous file; no destination or temporary filesystem
// output. All decoding/encoding later consumes these immutable exact bytes.
// With capture=-1 this independently verifies the current original pathname.
std::string source_digest(const fs::path &path, int capture = -1) {
    Fd input(::open(path.c_str(), O_RDONLY | O_CLOEXEC | O_NOFOLLOW | O_NONBLOCK));
    struct stat initial {
    }, final_fd{}, final_path{};
    require(::fstat(input.value, &initial) == 0 && S_ISREG(initial.st_mode) &&
                initial.st_size > 0 && std::uint64_t(initial.st_size) <= max_source_bytes,
            "export source must be a regular nonempty file <=256MiB");
    std::unique_ptr<EVP_MD_CTX, decltype(&EVP_MD_CTX_free)> context(EVP_MD_CTX_new(),
                                                                    EVP_MD_CTX_free);
    require(context && EVP_DigestInit_ex(context.get(), EVP_sha256(), nullptr) == 1,
            "export source digest initialization");
    std::array<char, 65536> bytes{};
    std::uint64_t remaining = initial.st_size;
    while (remaining) {
        const auto n =
            ::read(input.value, bytes.data(), std::min<std::uint64_t>(bytes.size(), remaining));
        if (n < 0 && errno == EINTR)
            continue;
        require(n > 0, "export source read/truncation failure");
        require(EVP_DigestUpdate(context.get(), bytes.data(), n) == 1,
                "export source digest update");
        if (capture >= 0) {
            ssize_t offset{};
            while (offset < n) {
                const auto written = ::write(capture, bytes.data() + offset, n - offset);
                if (written < 0 && errno == EINTR)
                    continue;
                require(written > 0, "export source capture write failure");
                offset += written;
            }
        }
        remaining -= n;
    }
    require(::fstat(input.value, &final_fd) == 0 && ::lstat(path.c_str(), &final_path) == 0 &&
                same(initial, final_fd) && same(initial, final_path),
            "export source changed during capture/verification");
    unsigned size{};
    std::array<unsigned char, 32> hash{};
    require(EVP_DigestFinal_ex(context.get(), hash.data(), &size) == 1 && size == hash.size(),
            "export source digest final");
    const char hex[] = "0123456789abcdef";
    std::string result;
    for (const auto byte : hash) {
        result += hex[byte >> 4];
        result += hex[byte & 15];
    }
    return result;
}
void durable(const fs::path &path, bool directory = false) {
    Fd fd(::open(path.c_str(), O_RDONLY | O_CLOEXEC | O_NOFOLLOW | (directory ? O_DIRECTORY : 0)));
    require(::fsync(fd.value) == 0, "export durability failure");
}
Descriptor read_descriptor(const fs::path &model, const Descriptor &expected) {
    const auto &spec = expected.spec();
    const auto &limits = spec.at("limits");
    const CpuAdmittedPackage package(model);
    auto actual = Descriptor::from_package(package,
                                           {limits.at("max_frame_points").get<std::uint64_t>(),
                                            limits.at("max_frame_bytes").get<std::uint64_t>(),
                                            limits.at("startup_timeout_ms").get<std::uint64_t>()},
                                           spec.at("session_id"), spec.at("map_epoch"));
    // Parse permits arbitrary JSON serialization, while from_package uses the
    // production serialization. Bind both full semantic identity and exact bytes
    // retained by this consumer; a peer serialization is otherwise admissible.
    require(actual.spec() == expected.spec(), "export descriptor/package identity mismatch");
    return actual;
}
} // namespace

struct LazExport::Impl {
    const fs::path source, model, destination;
    const Descriptor descriptor;
    Fd frozen;
    fs::path frozen_path;
    std::string source_sha256, manifest_bytes;
    Json manifest;
    std::vector<Point> original;
    std::vector<std::array<std::uint8_t, 2>> returns;
    std::vector<std::uint8_t> seen, labels, semantics;
    std::vector<std::uint64_t> frames, sequences, timestamps;
    std::optional<std::array<std::string, 4>> identity;
    std::size_t covered{};
    bool failed{}, finished{};

    Impl(const fs::path &s, const fs::path &m, const fs::path &d, const Descriptor &expected)
        : source(fs::absolute(s)), model(fs::absolute(m)), destination(fs::absolute(d)),
          descriptor(Descriptor::parse(expected.bytes())),
          frozen(static_cast<int>(::syscall(SYS_memfd_create, "synrex-export-source",
                                            MFD_CLOEXEC | MFD_ALLOW_SEALING))) {
        require(!fs::exists(destination) && !fs::is_symlink(destination) &&
                    fs::is_directory(destination.parent_path()),
                "export requires new bundle in existing parent");
        read_descriptor(model, descriptor);
        const Package package(model, PackageUse::file);
        require(package.manifest_sha256 == descriptor.spec().at("manifest_sha256") &&
                    package.taxonomy &&
                    package.taxonomy->raw_bytes() == descriptor.taxonomy().raw_bytes(),
                "export package changed during admission");
        manifest_bytes = package.manifest_bytes;
        manifest = package.manifest;
        source_sha256 = source_digest(source, frozen.value);
        require(::fcntl(frozen.value, F_ADD_SEALS,
                        F_SEAL_WRITE | F_SEAL_GROW | F_SEAL_SHRINK | F_SEAL_SEAL) == 0,
                "export cannot seal source capture");
        frozen_path = "/proc/self/fd/" + std::to_string(frozen.value);
        const auto count = validate_laz(frozen_path);
        require(count > 0 && count <= 1000000, "export corpus outside 1..1000000 records");
        original = decode(frozen_path, package.contract);
        lazperf::reader::named_file reader(frozen_path.string());
        require(reader.pointCount() == original.size(), "export source return count mismatch");
        std::vector<char> record(reader.header().point_record_length);
        for (std::size_t i = 0; i < original.size(); ++i) {
            reader.readPoint(record.data());
            const auto bits = static_cast<std::uint8_t>(record.at(14));
            returns.push_back(
                {static_cast<std::uint8_t>(bits & 15), static_cast<std::uint8_t>(bits >> 4)});
        }
        verify_inputs();
        seen.resize(count);
        labels.resize(count);
        semantics.resize(count);
        frames.resize(count);
        sequences.resize(count);
        timestamps.resize(count);
    }
    void alive() const {
        require(!failed && !finished, "export already finished or permanently failed");
    }
    void verify_inputs() const {
        read_descriptor(model, descriptor);
        const Package current(model, PackageUse::file);
        require(current.manifest_bytes == manifest_bytes && current.manifest == manifest &&
                    current.taxonomy &&
                    current.taxonomy->raw_bytes() == descriptor.taxonomy().raw_bytes(),
                "export package/taxonomy changed");
        require(source_digest(source) == source_sha256, "export original source changed");
    }
    void observe(const Classification &c) {
        alive();
        validate_classification(descriptor, c);
        const std::array<std::string, 4> incoming{c.frame.sensor_id, c.frame.stream_id,
                                                  c.frame.coordinate_frame, c.frame.crs};
        require(!identity || *identity == incoming, "export source metadata identity changed");
        // Validate the entire operation before modifying coverage or labels.
        for (const auto &p : c.frame.points) {
            require(p.id < original.size(), "export requires local original source row ID");
            require(!seen[p.id], "export duplicate original source row observation");
            require(p.xyz == original[p.id].xyz && p.intensity &&
                        *p.intensity == original[p.id].intensity,
                    "export requires exact original source xyz/intensity");
            require(!p.return_number || (*p.return_number == returns[p.id][0] &&
                                         *p.return_count == returns[p.id][1]),
                    "export original source return metadata mismatch");
        }
        identity = incoming;
        for (std::size_t i = 0; i < c.frame.points.size(); ++i) {
            const auto &p = c.frame.points[i];
            seen[p.id] = 1;
            ++covered;
            labels[p.id] = c.las_codes[i];
            semantics[p.id] = c.semantic_ids[i];
            frames[p.id] = c.frame.frame_id;
            sequences[p.id] = c.frame.sequence;
            timestamps[p.id] = p.timestamp_ns;
        }
    }
    Json finish() {
        alive();
        require(covered == original.size() && identity,
                "partial original source coverage: no export bundle published");
        verify_inputs();
        require(!fs::exists(destination) && !fs::is_symlink(destination),
                "export destination exists/refusing replacement");
        auto pattern = (destination.parent_path() /
                        ("." + destination.filename().string() + ".staging-XXXXXX"))
                           .string();
        const auto created = ::mkdtemp(pattern.data());
        require(created, "export staging creation failure");
        const fs::path stage(created);
        try {
            write_new(stage / "descriptor.json", descriptor.bytes());
            write_new(stage / "taxonomy.json", descriptor.taxonomy().raw_bytes());
            write_new(stage / "manifest.json", manifest_bytes);
            write_new(stage / "labels.u8",
                      std::string(reinterpret_cast<const char *>(labels.data()), labels.size()));
            const auto changed =
                apply_labels(frozen_path, stage / "labels.u8", stage / "classified.laz");
            Json binding{{"descriptor_sha256", descriptor.sha256()},
                         {"source_sha256", source_sha256},
                         {"manifest_sha256", descriptor.spec().at("manifest_sha256")},
                         {"taxonomy_sha256", descriptor.taxonomy().sha256()},
                         {"las_projection", descriptor.spec().at("las_projection")},
                         {"las_projection_sha256", descriptor.spec().at("las_projection_sha256")},
                         {"model_artifacts", manifest.at("artifacts")},
                         {"session_id", descriptor.spec().at("session_id")},
                         {"map_epoch", descriptor.spec().at("map_epoch")}};
            Json sidecar{
                {"schema", "synrex.generic_point_semantics.v1"},
                {"binding", binding},
                {"point_count", original.size()},
                {"source_id_convention", "zero-based original local source record index"},
                {"index_order", "original source point record order"},
                {"semantic_id_definition",
                 "explicit descriptor taxonomy semantic_id; one observation per original record"},
                {"semantic_ids", semantics},
                {"mapped_labels_sha256", sha256(stage / "labels.u8")}};
            write_new(stage / "semantics.json", sidecar.dump() + "\n");
            Json provenance{
                {"schema", "synrex.generic_export_provenance.v1"},
                {"binding", binding},
                {"sensor_id", (*identity)[0]},
                {"stream_id", (*identity)[1]},
                {"coordinate_frame", (*identity)[2]},
                {"crs", (*identity)[3]},
                {"source_frame_ids", frames},
                {"source_sequences", sequences},
                {"source_timestamps_ns", timestamps},
                {"timestamp_clock", "supplied frame source clock; not independently established"},
                {"source_id_convention", sidecar.at("source_id_convention")},
                {"classification_policy",
                 "exactly one original record observation; original window contexts retained"},
                {"preservation", "original source record bytes except LAS classification; existing "
                                 "fail-closed PDRF6/7/8 writer"},
                {"scope", "offline export only; no transport completion, quality or throughput "
                          "qualification"}};
            write_new(stage / "provenance.json", provenance.dump() + "\n");
            Json receipt{{"schema", "synrex.generic_export_bundle.v1"},
                         {"status", "PASS"},
                         {"points", original.size()},
                         {"coverage", 1.0},
                         {"changed_las_classifications", changed},
                         {"binding", binding},
                         {"files", Json::object()}};
            for (const auto *name :
                 {"descriptor.json", "taxonomy.json", "manifest.json", "labels.u8",
                  "classified.laz", "semantics.json", "provenance.json"})
                receipt["files"][name] = sha256(stage / name);
            write_new(stage / "bundle.json", receipt.dump(2) + "\n");
            for (const auto &entry : fs::directory_iterator(stage))
                durable(entry.path());
            durable(stage, true);
            verify_inputs();
            require(::syscall(SYS_renameat2, AT_FDCWD, stage.c_str(), AT_FDCWD, destination.c_str(),
                              RENAME_NOREPLACE) == 0,
                    "atomic export publication failed/refusing replacement");
            durable(destination.parent_path(), true);
            finished = true;
            return receipt;
        } catch (...) {
            std::error_code ignored;
            fs::remove_all(stage, ignored);
            throw;
        }
    }
};

LazExport::LazExport(const fs::path &source, const fs::path &model, const fs::path &destination,
                     const Descriptor &descriptor)
    : impl_(std::make_unique<Impl>(source, model, destination, descriptor)) {}
LazExport::~LazExport() = default;
const std::string &LazExport::source_sha256() const { return impl_->source_sha256; }
void LazExport::observe(const Classification &classification) {
    try {
        impl_->observe(classification);
    } catch (...) {
        impl_->failed = true;
        throw;
    }
}
Json LazExport::finish() {
    try {
        return impl_->finish();
    } catch (...) {
        impl_->failed = true;
        throw;
    }
}
} // namespace synrex::generic
