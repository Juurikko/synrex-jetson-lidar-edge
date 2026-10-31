// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Tuomas Pietilä
#pragma once
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

namespace synrex {
struct TaxonomyClass {
    std::uint8_t semantic_id;
    std::string key, name;
    std::uint8_t las_code;
};
struct TaxonomyProvenance {
    std::string source, author, license;
};

// Validated metadata only. This object does not infer meanings or create a session.
class Taxonomy {
    std::string bytes_, taxonomy_id_, namespace_, sha256_;
    std::vector<TaxonomyClass> classes_;
    TaxonomyProvenance provenance_;
    std::optional<std::uint8_t> unknown_;

  public:
    static constexpr std::size_t max_bytes = 65536;
    static constexpr std::size_t max_depth = 8; // object/array nesting, including the root
    explicit Taxonomy(std::string exact_utf8_bytes);
    static Taxonomy read(const std::filesystem::path &path);
    const std::string &raw_bytes() const { return bytes_; }
    const std::string &taxonomy_id() const { return taxonomy_id_; }
    const std::string &taxonomy_namespace() const { return namespace_; }
    const std::string &sha256() const { return sha256_; }
    const std::vector<TaxonomyClass> &classes() const { return classes_; }
    const TaxonomyProvenance &provenance() const { return provenance_; }
    std::optional<std::uint8_t> unknown_semantic_id() const { return unknown_; }
    void validate_channels(std::size_t channels) const;
    void validate_projection(const std::vector<std::uint8_t> &las_projection) const;
    void validate_digest(std::string_view expected_sha256) const;
};
} // namespace synrex
