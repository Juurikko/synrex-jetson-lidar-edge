// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Tuomas Pietilä
#include "synrex/point_io.hpp"
#include "synrex/directory_publication.hpp"
#include <algorithm>
#include <cerrno>
#include <cmath>
#include <cstring>
#include <fcntl.h>
#include <fstream>
#include <lazperf/readers.hpp>
#include <linux/memfd.h>
#include <openssl/evp.h>
#include <sys/stat.h>
#include <sys/syscall.h>
#include <unistd.h>
namespace synrex::point {
namespace {
struct Fd {
    int value{-1};
    explicit Fd(int fd) : value(fd) { require(fd >= 0, "point source open/capture failure"); }
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
                initial.st_size > 0 && std::uint64_t(initial.st_size) <= PointSource::max_bytes,
            "point source source must be a regular nonempty file <=256MiB");
    std::unique_ptr<EVP_MD_CTX, decltype(&EVP_MD_CTX_free)> context(EVP_MD_CTX_new(),
                                                                    EVP_MD_CTX_free);
    require(context && EVP_DigestInit_ex(context.get(), EVP_sha256(), nullptr) == 1,
            "point source source digest initialization");
    std::array<char, 65536> bytes{};
    std::uint64_t remaining = initial.st_size;
    while (remaining) {
        const auto n =
            ::read(input.value, bytes.data(), std::min<std::uint64_t>(bytes.size(), remaining));
        if (n < 0 && errno == EINTR)
            continue;
        require(n > 0, "point source source read/truncation failure");
        require(EVP_DigestUpdate(context.get(), bytes.data(), n) == 1,
                "point source source digest update");
        if (capture >= 0) {
            ssize_t offset{};
            while (offset < n) {
                const auto written = ::write(capture, bytes.data() + offset, n - offset);
                if (written < 0 && errno == EINTR)
                    continue;
                require(written > 0, "point source source capture write failure");
                offset += written;
            }
        }
        remaining -= n;
    }
    require(::fstat(input.value, &final_fd) == 0 && ::lstat(path.c_str(), &final_path) == 0 &&
                same(initial, final_fd) && same(initial, final_path),
            "point source source changed during capture/verification");
    unsigned size{};
    std::array<unsigned char, 32> hash{};
    require(EVP_DigestFinal_ex(context.get(), hash.data(), &size) == 1 && size == hash.size(),
            "point source source digest final");
    const char hex[] = "0123456789abcdef";
    std::string result;
    for (const auto byte : hash) {
        result += hex[byte >> 4];
        result += hex[byte & 15];
    }
    return result;
}
std::uint64_t little(const unsigned char *p, unsigned n) {
    std::uint64_t v{};
    for (unsigned i = 0; i < n; ++i)
        v |= std::uint64_t(p[i]) << (8 * i);
    return v;
}
double number(const unsigned char *p) {
    const auto bits = little(p, 8);
    double v{};
    std::memcpy(&v, &bits, 8);
    return v;
}
std::string digest_bytes(const unsigned char *p, std::size_t n) {
    std::array<unsigned char, 32> out{};
    unsigned size{};
    require(EVP_Digest(p, n, out.data(), &size, EVP_sha256(), nullptr) == 1 && size == 32,
            "point source header digest");
    std::string result;
    const char hex[] = "0123456789abcdef";
    for (auto v : out) {
        result += hex[v >> 4];
        result += hex[v & 15];
    }
    return result;
}
void verify_preservation(const fs::path &source, const fs::path &output,
                         const std::vector<std::uint8_t> &labels) {
    lazperf::reader::named_file a(source.string()), b(output.string());
    require(a.pointCount() == labels.size() && b.pointCount() == labels.size() &&
                a.header().point_format_id == b.header().point_format_id &&
                a.header().point_record_length == b.header().point_record_length,
            "point output layout/count mismatch");
    std::vector<char> x(a.header().point_record_length), y(x.size());
    for (std::size_t i = 0; i < labels.size(); ++i) {
        a.readPoint(x.data());
        b.readPoint(y.data());
        require(static_cast<std::uint8_t>(y[16]) == labels[i], "point output label mismatch");
        y[16] = x[16];
        require(x == y, "point output changed nonclassification record bytes");
    }
}
} // namespace
struct PointSource::Impl {
    const fs::path original;
    Fd frozen;
    fs::path captured;
    std::string digest, header_digest;
    std::array<unsigned char, 375> header{};
    std::array<double, 3> scale{}, offset{};
    unsigned format{}, record_length{};
    std::vector<SourceRecord> points;
    Impl(const fs::path &path, std::uint64_t row_limit)
        : original(fs::absolute(path)),
          frozen(static_cast<int>(::syscall(SYS_memfd_create, "synrex-point-source",
                                            MFD_CLOEXEC | MFD_ALLOW_SEALING))) {
        require(row_limit > 0 && row_limit <= PointSource::max_rows, "point source row limit");
        digest = source_digest(original, frozen.value);
        require(::fcntl(frozen.value, F_ADD_SEALS,
                        F_SEAL_WRITE | F_SEAL_GROW | F_SEAL_SHRINK | F_SEAL_SEAL) == 0,
                "point source cannot seal capture");
        captured = "/proc/self/fd/" + std::to_string(frozen.value);
        const auto rows = validate_laz(captured);
        require(rows <= row_limit, "point source exceeds admitted row limit");
        std::ifstream input(captured, std::ios::binary);
        require(bool(input.read(reinterpret_cast<char *>(header.data()), header.size())),
                "point source header truncated");
        header_digest = digest_bytes(header.data(), header.size());
        format = header[104] & 63;
        record_length = little(header.data() + 105, 2);
        const unsigned standard = format == 6 ? 30 : format == 7 ? 36 : 38;
        require(record_length == standard, "point source Extra Bytes are NOT_SUPPORTED");
        for (unsigned a = 0; a < 3; ++a) {
            scale[a] = number(header.data() + 131 + 8 * a);
            offset[a] = number(header.data() + 155 + 8 * a);
            require(std::isfinite(scale[a]) && scale[a] > 0 && std::isfinite(offset[a]),
                    "point source invalid scale/offset");
        }
        lazperf::reader::named_file reader(captured.string());
        require(reader.pointCount() == rows && reader.header().point_format_id == format &&
                    reader.header().point_record_length == record_length,
                "point source decoder/header mismatch");
        std::vector<unsigned char> raw(record_length);
        points.reserve(rows);
        for (std::uint64_t row = 0; row < rows; ++row) {
            reader.readPoint(reinterpret_cast<char *>(raw.data()));
            SourceRecord point{};
            for (unsigned a = 0; a < 3; ++a) {
                const auto bits = static_cast<std::uint32_t>(little(raw.data() + 4 * a, 4));
                std::memcpy(&point.integer_xyz[a], &bits, 4);
                point.xyz[a] = point.integer_xyz[a] * scale[a] + offset[a];
                require(std::isfinite(point.xyz[a]), "point source nonfinite coordinate");
            }
            point.intensity = little(raw.data() + 12, 2);
            points.push_back(point);
        }
        require(source_digest(original) == digest, "point source changed during decode");
    }
};
PointSource::PointSource(const fs::path &p, std::uint64_t rows)
    : impl_(std::make_unique<Impl>(p, rows)) {}
PointSource::~PointSource() = default;
const std::vector<SourceRecord> &PointSource::records() const { return impl_->points; }
const std::string &PointSource::sha256() const { return impl_->digest; }
const std::string &PointSource::header_sha256() const { return impl_->header_digest; }
void PointSource::verify_original() const {
    require(source_digest(impl_->original) == impl_->digest, "point source original bytes changed");
}
Json PointSource::metadata() const {
    const auto &s = *impl_;
    return {{"schema", "synrex.point-source.v1"},
            {"source_sha256", s.digest},
            {"header_sha256", s.header_digest},
            {"rows", s.points.size()},
            {"point_format", s.format},
            {"record_length", s.record_length},
            {"scale", s.scale},
            {"offset", s.offset},
            {"sealed", true},
            {"source_id_convention", "zero-based original local source record index"},
            {"physical_eof_verified", false},
            {"count_scope",
             "all declared header records decoded; extra compressed records not audited"}};
}
Json PointSource::export_semantics(const fs::path &path, const Taxonomy &taxonomy,
                                   const std::vector<std::uint8_t> &semantics) const {
    const auto destination = fs::absolute(path);
    require(semantics.size() == records().size(),
            "point output requires complete original-order coverage");
    std::vector<std::uint8_t> labels;
    labels.reserve(semantics.size());
    for (const auto id : semantics) {
        require(id < taxonomy.classes().size(), "point output semantic outside taxonomy");
        labels.push_back(taxonomy.classes()[id].las_code);
    }
    verify_original();
    require(!fs::exists(destination) && !fs::is_symlink(destination) &&
                fs::is_directory(destination.parent_path()),
            "point output requires new bundle");
    auto pattern =
        (destination.parent_path() / ("." + destination.filename().string() + ".point-XXXXXX"))
            .string();
    require(::mkdtemp(pattern.data()) != nullptr, "point output cannot create owned stage");
    const fs::path stage(pattern);
    bool published = false;
    try {
        write_new(stage / "labels.u8", std::string(labels.begin(), labels.end()));
        write_new(stage / "semantics.u8", std::string(semantics.begin(), semantics.end()));
        write_new(stage / "taxonomy.json", taxonomy.raw_bytes());
        const auto changed =
            apply_labels(impl_->captured, stage / "labels.u8", stage / "classified.laz");
        verify_preservation(impl_->captured, stage / "classified.laz", labels);
        Json receipt{
            {"schema", "synrex.point-preservation.v1"},
            {"scope",
             "CPU writer primitive; no learned execution, session completion or quality claim"},
            {"source", metadata()},
            {"taxonomy_sha256", taxonomy.sha256()},
            {"rows", records().size()},
            {"changed_las_classifications", changed},
            {"record_preservation", "PASS"},
            {"files", Json::object()}};
        for (const auto *name : {"labels.u8", "semantics.u8", "taxonomy.json", "classified.laz"})
            receipt["files"][name] = synrex::sha256(stage / name);
        write_new(stage / "receipt.json", receipt.dump(2) + "\n");
        sync_publication_stage(stage);
        verify_original();
        publish_directory_noreplace(stage, destination);
        published = true;
        sync_publication_parent(destination);
        return receipt;
    } catch (...) {
        if (!published) {
            std::error_code ignored;
            fs::remove_all(stage, ignored);
        }
        throw;
    }
}
} // namespace synrex::point
