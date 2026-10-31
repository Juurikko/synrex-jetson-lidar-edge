// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Tuomas Pietilä
#pragma once
#include <filesystem>

namespace synrex {
enum class DirectoryPublication { rename_noreplace, relative_symlink };
const char *directory_publication_name(DirectoryPublication method) noexcept;
// Caller owns a completed, verified, hidden real directory beside destination.
// On success, ownership transfers to destination: never remove stage afterward,
// including when a later parent fsync fails. A symlink publication retains stage.
// The optional native operation is a narrow CPU fault-injection seam, not a
// runtime flag. It must obey renameat2's return/errno and mutation semantics.
using DirectoryRenameOperation = int (*)(int, const char *, int, const char *, unsigned);
DirectoryPublication publish_directory_noreplace(const std::filesystem::path &stage,
                                                 const std::filesystem::path &destination,
                                                 DirectoryRenameOperation operation = nullptr);
// Flush regular direct members and the stage before publication. Rejects links
// and nested directories; current package/export formats contain only files.
void sync_publication_stage(const std::filesystem::path &stage);
void sync_publication_parent(const std::filesystem::path &destination);
} // namespace synrex
