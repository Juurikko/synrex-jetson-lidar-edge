// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Tuomas Pietilä
#include "synrex/directory_publication.hpp"
#include <cerrno>
#include <fcntl.h>
#include <linux/fs.h>
#include <stdexcept>
#include <sys/stat.h>
#include <sys/syscall.h>
#include <system_error>
#include <unistd.h>

namespace synrex {
namespace {
namespace fs = std::filesystem;
struct Fd {
    int value;
    explicit Fd(int fd) : value(fd) {
        if (fd < 0)
            throw std::system_error(errno, std::generic_category(), "publication open");
    }
    ~Fd() { ::close(value); }
    Fd(const Fd &) = delete;
    Fd &operator=(const Fd &) = delete;
};
void check(bool okay, const char *message) {
    if (!okay)
        throw std::runtime_error(message);
}
fs::path parent(const fs::path &path) {
    return fs::canonical(path.has_parent_path() ? path.parent_path() : fs::path("."));
}
bool component(const fs::path &path) {
    return !path.empty() && !path.filename().empty() && path.filename() != "." &&
           path.filename() != "..";
}
int native_rename(int a, const char *source, int b, const char *destination, unsigned flags) {
    return static_cast<int>(::syscall(SYS_renameat2, a, source, b, destination, flags));
}
void sync_fd(int fd) {
    if (::fsync(fd) != 0)
        throw std::system_error(errno, std::generic_category(), "publication durability");
}
} // namespace
const char *directory_publication_name(DirectoryPublication method) noexcept {
    return method == DirectoryPublication::rename_noreplace ? "renameat2_noreplace"
                                                            : "relative_directory_symlink";
}
DirectoryPublication publish_directory_noreplace(const fs::path &stage, const fs::path &destination,
                                                 DirectoryRenameOperation operation) {
    check(component(stage) && component(destination), "publication requires named paths");
    const auto stage_name = stage.filename().string();
    const auto output_name = destination.filename().string();
    check(stage_name.front() == '.' && stage_name != output_name,
          "publication requires a distinct hidden stage");
    const auto directory = parent(stage);
    check(directory == parent(destination), "publication stage must share destination parent");
    Fd parent_fd(::open(directory.c_str(), O_RDONLY | O_CLOEXEC | O_DIRECTORY | O_NOFOLLOW));
    Fd stage_fd(::openat(parent_fd.value, stage_name.c_str(),
                         O_RDONLY | O_CLOEXEC | O_DIRECTORY | O_NOFOLLOW));
    struct stat original {
    }, current{};
    check(::fstat(stage_fd.value, &original) == 0 && S_ISDIR(original.st_mode) &&
              original.st_uid == ::geteuid(),
          "publication requires an owned real stage directory");
    const auto rename = operation ? operation : native_rename;
    if (rename(parent_fd.value, stage_name.c_str(), parent_fd.value, output_name.c_str(),
               RENAME_NOREPLACE) == 0)
        return DirectoryPublication::rename_noreplace;
    const int error = errno;
    if (error != EINVAL && error != ENOSYS && error != EOPNOTSUPP)
        throw std::system_error(error, std::generic_category(),
                                "atomic directory publication failed/refusing replacement");
    check(::fstatat(parent_fd.value, stage_name.c_str(), &current, AT_SYMLINK_NOFOLLOW) == 0 &&
              S_ISDIR(current.st_mode) && current.st_dev == original.st_dev &&
              current.st_ino == original.st_ino,
          "publication stage identity changed");
    // symlinkat creates exactly one directory entry, atomically, with EEXIST for
    // every preexisting name (including empty directories and dangling links).
    // Both names are relative to the same pinned parent; moving that tree works.
    if (::symlinkat(stage_name.c_str(), parent_fd.value, output_name.c_str()) != 0)
        throw std::system_error(errno, std::generic_category(),
                                "atomic symlink publication failed/refusing replacement");
    return DirectoryPublication::relative_symlink;
}
void sync_publication_stage(const fs::path &stage) {
    Fd directory(::open(stage.c_str(), O_RDONLY | O_CLOEXEC | O_DIRECTORY | O_NOFOLLOW));
    for (const auto &entry : fs::directory_iterator(stage)) {
        Fd member(::openat(directory.value, entry.path().filename().c_str(),
                           O_RDONLY | O_CLOEXEC | O_NOFOLLOW | O_NONBLOCK));
        struct stat info {};
        check(::fstat(member.value, &info) == 0 && S_ISREG(info.st_mode),
              "publication member must be a regular non-symlink file");
        sync_fd(member.value);
    }
    sync_fd(directory.value);
}
void sync_publication_parent(const fs::path &destination) {
    const auto directory = parent(destination);
    Fd fd(::open(directory.c_str(), O_RDONLY | O_CLOEXEC | O_DIRECTORY | O_NOFOLLOW));
    sync_fd(fd.value);
}
} // namespace synrex
