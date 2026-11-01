// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Tuomas Pietilä
#pragma once
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <memory>
#include <string>
#include <vector>

namespace synrex::replay {
struct Source {
    std::filesystem::path path;
    std::string sha256;
    std::uint64_t records, wire_begin;
    std::string coordinate_frame, crs;
};
struct RecordLocation {
    std::size_t source_index;
    std::uint64_t original_record_index;
};

// Pins one admitted inode. Pass path() to a pathname-only decoder; it names the
// retained read-only fd, never a second lookup of the original source pathname.
// Call finish() after consumption: destructors close, but cannot report errors.
class SourceHandle final {
    struct Impl;
    std::unique_ptr<Impl> impl_;
    explicit SourceHandle(std::unique_ptr<Impl>);
    friend class Catalog;

  public:
    ~SourceHandle();
    SourceHandle(SourceHandle &&) noexcept;
    SourceHandle &operator=(SourceHandle &&) noexcept;
    SourceHandle(const SourceHandle &) = delete;
    SourceHandle &operator=(const SourceHandle &) = delete;
    int fd() const;
    const std::filesystem::path &path() const;
    void finish() const;
};

// CPU-only metadata and identity admission. Declared counts are expectations,
// never proof of decoded LAZ counts. Admission opens/hashes/closes sources one at
// a time; only SourceHandle retains a source descriptor during consumption.
class Catalog final {
    struct Impl;
    std::unique_ptr<Impl> impl_;

  public:
    static constexpr std::size_t max_manifest_bytes = 1048576;
    static constexpr std::size_t max_sources = 4096;
    static constexpr std::uint64_t max_source_bytes = 268435456;
    static constexpr std::size_t max_identifier_bytes = 256;
    explicit Catalog(const std::filesystem::path &manifest);
    ~Catalog();
    Catalog(Catalog &&) noexcept;
    Catalog &operator=(Catalog &&) noexcept;
    Catalog(const Catalog &) = delete;
    Catalog &operator=(const Catalog &) = delete;
    const std::string &manifest_sha256() const;
    const std::string &manifest_bytes() const;
    const std::vector<Source> &sources() const;
    std::uint64_t total_records() const;
    std::uint64_t wire_id(std::size_t source_index, std::uint64_t original_record_index) const;
    RecordLocation source_for_wire_id(std::uint64_t) const;
    SourceHandle open_source(std::size_t source_index) const;
};
} // namespace synrex::replay
