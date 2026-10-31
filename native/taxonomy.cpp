// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Tuomas Pietilä
#include "synrex/taxonomy.hpp"
#include <algorithm>
#include <array>
#include <cerrno>
#include <fcntl.h>
#include <initializer_list>
#include <memory>
#include <nlohmann/json.hpp>
#include <openssl/evp.h>
#include <set>
#include <stdexcept>
#include <sys/stat.h>
#include <unistd.h>

namespace synrex {
namespace {
using Json = nlohmann::json;
void check(bool condition, const char *message) {
    if (!condition)
        throw std::runtime_error(message);
}
void exact_keys(const Json &value, std::initializer_list<const char *> keys) {
    check(value.is_object() && value.size() == keys.size(), "taxonomy unknown/missing fields");
    for (const auto key : keys)
        check(value.contains(key), "taxonomy missing field");
}
bool alphanumeric(unsigned char c) {
    return (c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') || (c >= '0' && c <= '9');
}
std::string identifier(const Json &value, std::size_t limit) {
    check(value.is_string(), "taxonomy identifier must be a string");
    const auto text = value.get<std::string>();
    check(!text.empty() && text.size() <= limit && alphanumeric(text.front()),
          "taxonomy identifier size/prefix");
    check(std::all_of(text.begin(), text.end(),
                      [](unsigned char c) {
                          return alphanumeric(c) || c == '.' || c == '_' || c == ':' || c == '-';
                      }),
          "taxonomy identifier ASCII grammar");
    return text;
}
std::string text(const Json &value, std::size_t limit) {
    check(value.is_string(), "taxonomy text must be a string");
    const auto result = value.get<std::string>();
    // The JSON parser validates UTF-8 and Unicode escapes; limits count decoded UTF-8 bytes.
    check(!result.empty() && result.size() <= limit, "taxonomy text size");
    check(std::all_of(result.begin(), result.end(),
                      [](unsigned char c) { return c >= 32 && c != 127; }),
          "taxonomy text contains C0/DEL");
    return result;
}
std::uint64_t integer(const Json &value, std::uint64_t limit) {
    check(value.is_number_integer(), "taxonomy integer required");
    if (value.is_number_unsigned()) {
        const auto result = value.get<std::uint64_t>();
        check(result <= limit, "taxonomy integer range");
        return result;
    }
    const auto result = value.get<std::int64_t>();
    check(result >= 0 && static_cast<std::uint64_t>(result) <= limit, "taxonomy integer range");
    return static_cast<std::uint64_t>(result);
}
std::string digest(const std::string &bytes) {
    std::unique_ptr<EVP_MD_CTX, decltype(&EVP_MD_CTX_free)> ctx(EVP_MD_CTX_new(), EVP_MD_CTX_free);
    check(ctx && EVP_DigestInit_ex(ctx.get(), EVP_sha256(), nullptr) == 1 &&
              EVP_DigestUpdate(ctx.get(), bytes.data(), bytes.size()) == 1,
          "taxonomy SHA256 initialization/update");
    std::array<unsigned char, 32> hash{};
    unsigned length{};
    check(EVP_DigestFinal_ex(ctx.get(), hash.data(), &length) == 1 && length == hash.size(),
          "taxonomy SHA256 final");
    static constexpr char hex[] = "0123456789abcdef";
    std::string result;
    result.reserve(64);
    for (const auto byte : hash) {
        result += hex[byte >> 4];
        result += hex[byte & 15];
    }
    return result;
}
} // namespace

Taxonomy::Taxonomy(std::string exact_utf8_bytes) : bytes_(std::move(exact_utf8_bytes)) {
    check(!bytes_.empty() && bytes_.size() <= max_bytes, "taxonomy artifact size");
    std::vector<std::set<std::string>> keys;
    std::size_t nesting = 0;
    const auto parsed = Json::parse(bytes_, [&](int, Json::parse_event_t event, Json &value) {
        if (event == Json::parse_event_t::object_start ||
            event == Json::parse_event_t::array_start) {
            check(++nesting <= max_depth, "taxonomy JSON nesting limit");
        }
        if (event == Json::parse_event_t::object_start)
            keys.emplace_back();
        if (event == Json::parse_event_t::key)
            check(keys.back().insert(value.get<std::string>()).second,
                  "taxonomy duplicate JSON key");
        if (event == Json::parse_event_t::object_end)
            keys.pop_back();
        if (event == Json::parse_event_t::object_end || event == Json::parse_event_t::array_end)
            --nesting;
        return true;
    });
    exact_keys(parsed, {"schema", "taxonomy_id", "namespace", "classes", "provenance",
                        "unknown_semantic_id"});
    check(parsed.at("schema") == "synrex.taxonomy.v1", "unsupported taxonomy schema");
    taxonomy_id_ = identifier(parsed.at("taxonomy_id"), 128);
    namespace_ = identifier(parsed.at("namespace"), 128);
    const auto &classes = parsed.at("classes");
    check(classes.is_array() && classes.size() >= 2 && classes.size() <= 256,
          "taxonomy requires 2..256 ordered classes");
    std::set<std::string> class_keys;
    for (std::size_t i = 0; i < classes.size(); ++i) {
        const auto &value = classes.at(i);
        exact_keys(value, {"semantic_id", "key", "name", "las_code"});
        check(integer(value.at("semantic_id"), 255) == i, "taxonomy semantic IDs must be ordered");
        auto key = identifier(value.at("key"), 64);
        check(class_keys.insert(key).second, "taxonomy duplicate class key");
        classes_.push_back({static_cast<std::uint8_t>(i), std::move(key),
                            text(value.at("name"), 256),
                            static_cast<std::uint8_t>(integer(value.at("las_code"), 255))});
    }
    const auto &provenance = parsed.at("provenance");
    exact_keys(provenance, {"source", "author", "license"});
    provenance_ = {text(provenance.at("source"), 512), text(provenance.at("author"), 512),
                   text(provenance.at("license"), 512)};
    if (!parsed.at("unknown_semantic_id").is_null())
        unknown_ = static_cast<std::uint8_t>(
            integer(parsed.at("unknown_semantic_id"), classes_.size() - 1));
    sha256_ = digest(bytes_);
}

Taxonomy Taxonomy::read(const std::filesystem::path &path) {
    const int fd = ::open(path.c_str(), O_RDONLY | O_CLOEXEC | O_NOFOLLOW | O_NONBLOCK);
    check(fd >= 0, "taxonomy cannot open regular artifact without following final symlink");
    struct Close {
        int fd;
        ~Close() { ::close(fd); }
    } close{fd};
    struct stat info {};
    check(::fstat(fd, &info) == 0 && S_ISREG(info.st_mode), "taxonomy artifact must be regular");
    check(info.st_size > 0 && static_cast<std::uint64_t>(info.st_size) <= max_bytes,
          "taxonomy artifact size");
    std::string bytes;
    bytes.reserve(static_cast<std::size_t>(info.st_size));
    std::array<char, 4096> buffer{};
    while (true) {
        const auto count = ::read(fd, buffer.data(), buffer.size());
        if (count < 0 && errno == EINTR)
            continue;
        check(count >= 0, "taxonomy artifact read failure");
        if (count == 0)
            break;
        check(static_cast<std::size_t>(count) <= max_bytes - bytes.size(),
              "taxonomy artifact size");
        bytes.append(buffer.data(), static_cast<std::size_t>(count));
    }
    return Taxonomy(std::move(bytes));
}
void Taxonomy::validate_channels(std::size_t channels) const {
    check(channels == classes_.size(), "taxonomy/output channel mismatch");
}
void Taxonomy::validate_projection(const std::vector<std::uint8_t> &las_projection) const {
    check(las_projection.size() == classes_.size(), "taxonomy incomplete LAS projection");
    for (std::size_t i = 0; i < classes_.size(); ++i)
        check(las_projection[i] == classes_[i].las_code, "taxonomy LAS projection mismatch");
}
void Taxonomy::validate_digest(std::string_view expected_sha256) const {
    check(expected_sha256 == sha256_, "taxonomy exact artifact digest mismatch");
}
} // namespace synrex
