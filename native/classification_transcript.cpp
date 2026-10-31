// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Tuomas Pietilä
#include "synrex/classification_transcript.hpp"
#include <openssl/evp.h>

namespace synrex::generic {
namespace {
using Context = std::unique_ptr<EVP_MD_CTX, decltype(&EVP_MD_CTX_free)>;
Context context() {
    Context value(EVP_MD_CTX_new(), EVP_MD_CTX_free);
    require(value && EVP_DigestInit_ex(value.get(), EVP_sha256(), nullptr) == 1,
            "classification semantic transcript SHA256 initialization");
    return value;
}
Context copy(const Context &source) {
    auto result = context();
    require(EVP_MD_CTX_copy_ex(result.get(), source.get()) == 1,
            "classification semantic transcript SHA256 copy");
    return result;
}
std::string digest(const Context &source) {
    auto copied = copy(source);
    std::array<unsigned char, 32> bytes{};
    unsigned count{};
    require(EVP_DigestFinal_ex(copied.get(), bytes.data(), &count) == 1 && count == bytes.size(),
            "classification semantic transcript SHA256 final");
    const char hex[] = "0123456789abcdef";
    std::string result;
    for (auto byte : bytes) {
        result += hex[byte >> 4];
        result += hex[byte & 15];
    }
    return result;
}
} // namespace
struct ClassificationTranscript::Impl {
    Context digest = context();
    std::uint64_t frames{}, points{}, bytes{};
};
ClassificationTranscript::ClassificationTranscript(const Descriptor &descriptor)
    : impl_(std::make_unique<Impl>()), descriptor_(descriptor) {}
ClassificationTranscript::~ClassificationTranscript() = default;
void ClassificationTranscript::add(const Classification &result) {
    // Validate the original too: normalization must not launder invalid metrics.
    validate_classification(descriptor_, result);
    const Classification normalized{result.frame, result.semantic_ids, result.las_codes,
                                    Json::object()};
    const auto bytes = encode_classification(descriptor_, normalized);
    require(impl_->frames < UINT64_MAX &&
                result.frame.points.size() <= UINT64_MAX - impl_->points &&
                bytes.size() <= UINT64_MAX - impl_->bytes,
            "classification semantic transcript accounting overflow");
    auto next = copy(impl_->digest);
    require(EVP_DigestUpdate(next.get(), bytes.data(), bytes.size()) == 1,
            "classification semantic transcript SHA256 update");
    impl_->digest = std::move(next);
    ++impl_->frames;
    impl_->points += result.frame.points.size();
    impl_->bytes += bytes.size();
}
Json ClassificationTranscript::receipt() const {
    return {{"schema", "synrex.classification_semantic_transcript.v1"},
            {"normalization", "encode_classification_with_empty_timings_v1"},
            {"descriptor_sha256", descriptor_.sha256()},
            {"sha256", digest(impl_->digest)},
            {"classification_frames", impl_->frames},
            {"classified_points", impl_->points},
            {"encoded_bytes", impl_->bytes},
            {"scope", "ordered complete codec envelopes; only timings replaced by {}; all encoded "
                      "source/frame/point identities, attributes, timestamps, semantic IDs and LAS "
                      "codes retained"},
            {"observation", "receiver after successful send; subscriber after independent source "
                            "validation; partial local prefixes do not prove terminal completion"},
            {"storage_scope", "constant retained SHA256 state and counters; one bounded temporary "
                              "classification copy plus codec envelope; no frame history"}};
}
} // namespace synrex::generic
