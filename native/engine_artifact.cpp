// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Tuomas Pietilä
#include "synrex/engine_artifact.hpp"
#include <algorithm>
#include <cerrno>
#include <fcntl.h>
#include <iomanip>
#include <memory>
#include <openssl/evp.h>
#include <sstream>
#include <sys/stat.h>
#include <unistd.h>

namespace synrex {
namespace {
bool same_file(const struct stat &a, const struct stat &b) {
    return a.st_dev == b.st_dev && a.st_ino == b.st_ino && a.st_mode == b.st_mode &&
           a.st_nlink == b.st_nlink && a.st_size == b.st_size &&
           a.st_mtim.tv_sec == b.st_mtim.tv_sec && a.st_mtim.tv_nsec == b.st_mtim.tv_nsec &&
           a.st_ctim.tv_sec == b.st_ctim.tv_sec && a.st_ctim.tv_nsec == b.st_ctim.tv_nsec;
}
std::string capture(const Package &package, const std::string &kind) {
    const auto path = package.artifact(kind);
    const int fd = ::open(path.c_str(), O_RDONLY | O_CLOEXEC | O_NOFOLLOW | O_NONBLOCK);
    require(fd >= 0, "cannot open regular non-symlink model capture");
    struct Close {
        int fd;
        ~Close() { ::close(fd); }
    } close{fd};
    struct stat initial {};
    require(::fstat(fd, &initial) == 0 && S_ISREG(initial.st_mode),
            "model capture must be regular");
    require(initial.st_size > 0 &&
                static_cast<std::uint64_t>(initial.st_size) <= EngineArtifact::max_bytes,
            "model capture size outside supported range");
    const auto size = static_cast<std::size_t>(initial.st_size);
    std::string bytes(size, '\0');
    std::size_t done{};
    while (done < size) {
        const auto count =
            ::read(fd, bytes.data() + done, std::min<std::size_t>(65536, size - done));
        if (count < 0 && errno == EINTR)
            continue;
        require(count > 0, "model capture truncated/read failure");
        done += static_cast<std::size_t>(count);
    }
    struct stat final_fd {
    }, final_path{};
    require(::fstat(fd, &final_fd) == 0 && ::lstat(path.c_str(), &final_path) == 0 &&
                same_file(initial, final_fd) && same_file(initial, final_path),
            "model identity changed during capture");
    // Hash the retained bytes, never a second pathname read. Deserialization must
    // consume this same owned storage even after the source inode is modified.
    std::unique_ptr<EVP_MD_CTX, decltype(&EVP_MD_CTX_free)> ctx(EVP_MD_CTX_new(), EVP_MD_CTX_free);
    require(ctx && EVP_DigestInit_ex(ctx.get(), EVP_sha256(), nullptr) == 1 &&
                EVP_DigestUpdate(ctx.get(), bytes.data(), bytes.size()) == 1,
            "model capture SHA256 initialization/update");
    std::array<unsigned char, 32> hash{};
    unsigned count{};
    require(EVP_DigestFinal_ex(ctx.get(), hash.data(), &count) == 1 && count == hash.size(),
            "model capture SHA256 final");
    std::ostringstream encoded;
    for (const auto value : hash)
        encoded << std::hex << std::setfill('0') << std::setw(2) << unsigned(value);
    require(encoded.str() == package.manifest.at("artifacts").at(kind).at("sha256"),
            "model capture SHA256 mismatch");
    return bytes;
}
} // namespace
EngineArtifact::EngineArtifact(const fs::path &path, PackageUse use)
    : package_(path, use),
      kind_(package_.manifest.at("artifacts").contains("engine") ? "engine" : "onnx"),
      bytes_(capture(package_, kind_)) {}
const std::string &EngineArtifact::engine_bytes() const {
    require(kind_ == "engine", "execution requires an engine artifact; ONNX is build-only");
    return bytes_;
}
} // namespace synrex
