// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Tuomas Pietilä
#include "synrex/replay_manifest.hpp"
#include <algorithm>
#include <array>
#include <cerrno>
#include <fcntl.h>
#include <limits>
#include <nlohmann/json.hpp>
#include <openssl/evp.h>
#include <set>
#include <stdexcept>
#include <sys/stat.h>
#include <unistd.h>

namespace synrex::replay {
namespace {
using Json = nlohmann::json;
void check(bool ok, const char *message) {
    if (!ok)
        throw std::runtime_error(message);
}
struct Descriptor {
    int value = -1;
    explicit Descriptor(int fd) : value(fd) {}
    ~Descriptor() {
        if (value >= 0)
            ::close(value);
    }
    Descriptor(const Descriptor &) = delete;
    Descriptor &operator=(const Descriptor &) = delete;
};
bool same_file(const struct stat &a, const struct stat &b) {
    return a.st_dev == b.st_dev && a.st_ino == b.st_ino && a.st_mode == b.st_mode &&
           a.st_nlink == b.st_nlink && a.st_size == b.st_size &&
           a.st_mtim.tv_sec == b.st_mtim.tv_sec && a.st_mtim.tv_nsec == b.st_mtim.tv_nsec &&
           a.st_ctim.tv_sec == b.st_ctim.tv_sec && a.st_ctim.tv_nsec == b.st_ctim.tv_nsec;
}
int open_regular(const std::filesystem::path &path) {
    const auto fd = ::open(path.c_str(), O_RDONLY | O_CLOEXEC | O_NOFOLLOW | O_NONBLOCK);
    check(fd >= 0, "replay cannot open regular non-symlink file");
    return fd;
}
struct stat identity(int fd, std::uint64_t limit) {
    struct stat value {};
    check(::fstat(fd, &value) == 0 && S_ISREG(value.st_mode), "replay file must be regular");
    check(value.st_size > 0 && static_cast<std::uint64_t>(value.st_size) <= limit,
          "replay file size outside supported range");
    return value;
}
void verify_identity(int fd, const std::filesystem::path &path, const struct stat &expected) {
    struct stat descriptor {
    }, pathname{};
    check(::fstat(fd, &descriptor) == 0 && ::lstat(path.c_str(), &pathname) == 0 &&
              same_file(expected, descriptor) && same_file(expected, pathname),
          "replay file identity changed");
}
std::string hex_digest(EVP_MD_CTX *ctx) {
    std::array<unsigned char, 32> hash{};
    unsigned length{};
    check(EVP_DigestFinal_ex(ctx, hash.data(), &length) == 1 && length == hash.size(),
          "replay SHA256 final");
    static constexpr char digits[] = "0123456789abcdef";
    std::string result;
    result.reserve(64);
    for (const auto byte : hash) {
        result += digits[byte >> 4];
        result += digits[byte & 15];
    }
    return result;
}
// pread keeps verification independent of the decoder's shared file offset.
std::string hash_fd(int fd, const struct stat &expected, std::string *capture = nullptr) {
    std::unique_ptr<EVP_MD_CTX, decltype(&EVP_MD_CTX_free)> ctx(EVP_MD_CTX_new(), EVP_MD_CTX_free);
    check(ctx && EVP_DigestInit_ex(ctx.get(), EVP_sha256(), nullptr) == 1,
          "replay SHA256 initialization");
    std::array<char, 65536> buffer{};
    off_t offset = 0;
    while (offset < expected.st_size) {
        const auto wanted = std::min<std::uint64_t>(buffer.size(), expected.st_size - offset);
        const auto count = ::pread(fd, buffer.data(), wanted, offset);
        if (count < 0 && errno == EINTR)
            continue;
        check(count > 0, "replay truncated file/read failure");
        check(EVP_DigestUpdate(ctx.get(), buffer.data(), static_cast<std::size_t>(count)) == 1,
              "replay SHA256 update");
        if (capture)
            capture->append(buffer.data(), static_cast<std::size_t>(count));
        offset += count;
    }
    return hex_digest(ctx.get());
}
void verify_source(int fd, const Source &source, const struct stat &expected) {
    verify_identity(fd, source.path, expected);
    check(hash_fd(fd, expected) == source.sha256, "replay source SHA256 mismatch");
    verify_identity(fd, source.path, expected);
}
void exact_keys(const Json &value, std::initializer_list<const char *> keys) {
    check(value.is_object() && value.size() == keys.size(), "replay unknown/missing fields");
    for (const auto key : keys)
        check(value.contains(key), "replay missing field");
}
std::string identifier(const Json &value) {
    check(value.is_string(), "replay identifier must be a string");
    const auto text = value.get<std::string>();
    check(!text.empty() && text.size() <= Catalog::max_identifier_bytes, "replay identifier size");
    check(
        std::all_of(text.begin(), text.end(), [](unsigned char c) { return c >= 32 && c <= 126; }),
        "replay identifier must be printable ASCII");
    return text;
}
std::uint64_t records(const Json &value) {
    check(value.is_number_integer(), "replay records must be an integer");
    if (value.is_number_unsigned()) {
        const auto count = value.get<std::uint64_t>();
        check(count > 0, "replay records must be positive");
        return count;
    }
    const auto count = value.get<std::int64_t>();
    check(count > 0, "replay records must be positive");
    return static_cast<std::uint64_t>(count);
}
} // namespace

struct SourceHandle::Impl {
    Descriptor descriptor;
    Source source;
    struct stat initial;
    std::filesystem::path descriptor_path;
    Impl(const Source &value, const struct stat &identity)
        : descriptor(-1), source(value), initial(identity) {}
};
struct Catalog::Impl {
    std::string bytes, digest;
    std::vector<Source> sources;
    std::vector<struct stat> identities;
    std::uint64_t total = 0;
};

SourceHandle::SourceHandle(std::unique_ptr<Impl> impl) : impl_(std::move(impl)) {}
SourceHandle::~SourceHandle() = default;
SourceHandle::SourceHandle(SourceHandle &&) noexcept = default;
SourceHandle &SourceHandle::operator=(SourceHandle &&) noexcept = default;
int SourceHandle::fd() const {
    check(bool(impl_), "replay source handle was moved");
    return impl_->descriptor.value;
}
const std::filesystem::path &SourceHandle::path() const {
    check(bool(impl_), "replay source handle was moved");
    return impl_->descriptor_path;
}
void SourceHandle::finish() const {
    check(bool(impl_), "replay source handle was moved");
    verify_source(fd(), impl_->source, impl_->initial);
}

Catalog::Catalog(const std::filesystem::path &manifest) : impl_(std::make_unique<Impl>()) {
    const auto manifest_path = std::filesystem::absolute(manifest).lexically_normal();
    const Descriptor fd(open_regular(manifest_path));
    const auto initial = identity(fd.value, max_manifest_bytes);
    impl_->bytes.reserve(static_cast<std::size_t>(initial.st_size));
    impl_->digest = hash_fd(fd.value, initial, &impl_->bytes);
    verify_identity(fd.value, manifest_path, initial);
    check(impl_->bytes.find('\0') == std::string::npos, "replay manifest contains raw NUL");
    std::vector<std::set<std::string>> keys;
    std::size_t nesting = 0;
    const auto parsed = Json::parse(impl_->bytes, [&](int, Json::parse_event_t event, Json &value) {
        if (event == Json::parse_event_t::object_start || event == Json::parse_event_t::array_start)
            check(++nesting <= 8, "replay JSON nesting limit");
        if (event == Json::parse_event_t::object_start)
            keys.emplace_back();
        if (event == Json::parse_event_t::key)
            check(keys.back().insert(value.get<std::string>()).second, "replay duplicate JSON key");
        if (event == Json::parse_event_t::object_end)
            keys.pop_back();
        if (event == Json::parse_event_t::object_end || event == Json::parse_event_t::array_end)
            --nesting;
        return true;
    });
    exact_keys(parsed, {"schema", "workload", "boundary_policy", "sources"});
    check(parsed.at("schema") == "synrex.distinct_replay.v1", "unsupported replay schema");
    check(parsed.at("workload") == "file_order_synthetic", "unsupported replay workload");
    check(parsed.at("boundary_policy") == "reset_each_file", "unsupported replay boundary policy");
    const auto &sources = parsed.at("sources");
    check(sources.is_array() && !sources.empty() && sources.size() <= max_sources,
          "replay requires 1..4096 sources");
    std::set<std::string> digests;
    for (const auto &value : sources) {
        exact_keys(value, {"path", "sha256", "records", "coordinate_frame", "crs"});
        check(value.at("path").is_string(), "replay source path must be a string");
        const auto path = value.at("path").get<std::string>();
        check(!path.empty() && path.size() <= 4096 && path.find('\0') == std::string::npos,
              "replay source path size/NUL");
        check(value.at("sha256").is_string(), "replay source digest must be a string");
        const auto digest = value.at("sha256").get<std::string>();
        check(digest.size() == 64 && std::all_of(digest.begin(), digest.end(),
                                                 [](unsigned char c) {
                                                     return (c >= '0' && c <= '9') ||
                                                            (c >= 'a' && c <= 'f');
                                                 }),
              "replay digest must be lowercase SHA256");
        check(digests.insert(digest).second, "replay duplicate source digest");
        const auto count = records(value.at("records"));
        check(count <= std::numeric_limits<std::uint64_t>::max() - impl_->total,
              "replay record total overflow");
        const auto resolved = (manifest_path.parent_path() / path).lexically_normal();
        impl_->sources.push_back({resolved, digest, count, impl_->total,
                                  identifier(value.at("coordinate_frame")),
                                  identifier(value.at("crs"))});
        impl_->total += count;
    }
    for (const auto &source : impl_->sources) {
        const Descriptor source_fd(open_regular(source.path));
        const auto source_identity = identity(source_fd.value, max_source_bytes);
        verify_source(source_fd.value, source, source_identity);
        impl_->identities.push_back(source_identity);
    }
}
Catalog::~Catalog() = default;
Catalog::Catalog(Catalog &&) noexcept = default;
Catalog &Catalog::operator=(Catalog &&) noexcept = default;
const std::string &Catalog::manifest_sha256() const { return impl_->digest; }
const std::string &Catalog::manifest_bytes() const { return impl_->bytes; }
const std::vector<Source> &Catalog::sources() const { return impl_->sources; }
std::uint64_t Catalog::total_records() const { return impl_->total; }
std::uint64_t Catalog::wire_id(std::size_t source_index, std::uint64_t row) const {
    check(source_index < sources().size(), "replay source index out of range");
    const auto &source = sources()[source_index];
    check(row < source.records, "replay original record index out of range");
    // Admission proved wire_begin + records cannot overflow.
    return source.wire_begin + row;
}
RecordLocation Catalog::source_for_wire_id(std::uint64_t wire) const {
    check(wire < total_records(), "replay wire ID out of range");
    const auto after = std::upper_bound(
        sources().begin(), sources().end(), wire,
        [](std::uint64_t id, const Source &source) { return id < source.wire_begin; });
    const auto found = std::prev(after);
    return {static_cast<std::size_t>(found - sources().begin()), wire - found->wire_begin};
}
SourceHandle Catalog::open_source(std::size_t index) const {
    check(index < sources().size(), "replay source index out of range");
    const auto &source = sources()[index];
    auto opened = std::make_unique<SourceHandle::Impl>(source, impl_->identities[index]);
    opened->descriptor.value = open_regular(source.path);
    opened->descriptor_path = "/proc/self/fd/" + std::to_string(opened->descriptor.value);
    verify_source(opened->descriptor.value, source, opened->initial);
    return SourceHandle(std::move(opened));
}
} // namespace synrex::replay
