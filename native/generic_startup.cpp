// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Tuomas Pietilä
#include "synrex/generic_startup.hpp"
#include <algorithm>
#include <memory>
#include <openssl/evp.h>
#include <set>
#include <stdexcept>

namespace synrex::generic {
namespace {
const std::string context_bytes =
    "{\"schema\":\"synrex.context.current_frame.v1\",\"history\":false,"
    "\"origin\":\"per_window_minimum\",\"emit\":\"current_once\"}";
void check(bool condition, const char *message) {
    if (!condition)
        throw std::runtime_error(message);
}
std::string digest(std::string_view bytes) {
    std::unique_ptr<EVP_MD_CTX, decltype(&EVP_MD_CTX_free)> ctx(EVP_MD_CTX_new(), EVP_MD_CTX_free);
    check(ctx && EVP_DigestInit_ex(ctx.get(), EVP_sha256(), nullptr) == 1 &&
              EVP_DigestUpdate(ctx.get(), bytes.data(), bytes.size()) == 1,
          "descriptor SHA256");
    unsigned size{};
    unsigned char hash[32]{};
    check(EVP_DigestFinal_ex(ctx.get(), hash, &size) == 1 && size == 32, "descriptor SHA256 final");
    const char hex[] = "0123456789abcdef";
    std::string result;
    for (const auto byte : hash) {
        result += hex[byte >> 4];
        result += hex[byte & 15];
    }
    return result;
}
bool hex_string(std::string_view text, std::size_t size) {
    return text.size() == size && std::all_of(text.begin(), text.end(), [](unsigned char c) {
               return (c >= '0' && c <= '9') || (c >= 'a' && c <= 'f');
           });
}
void keys(const Json &value, std::initializer_list<const char *> expected) {
    check(value.is_object() && value.size() == expected.size(),
          "descriptor/control unknown fields");
    for (const auto key : expected)
        check(value.contains(key), "descriptor/control missing field");
}
std::uint64_t integer(const Json &value, std::uint64_t low, std::uint64_t high) {
    check(value.is_number_integer(), "descriptor integer required");
    if (!value.is_number_unsigned())
        check(value.get<std::int64_t>() >= 0, "descriptor negative integer");
    const auto result = value.get<std::uint64_t>();
    check(result >= low && result <= high, "descriptor integer range");
    return result;
}
// SAX checks precede DOM allocation and reject duplicate decoded keys, including
// escaped aliases. Normal DOM parsing then preserves ordinary array/null semantics.
struct Guard : nlohmann::json_sax<Json> {
    std::size_t depth{};
    std::vector<std::set<std::string>> objects;
    bool null() override { return true; }
    bool boolean(bool) override { return true; }
    bool number_integer(number_integer_t) override { return true; }
    bool number_unsigned(number_unsigned_t) override { return true; }
    bool number_float(number_float_t, const string_t &) override { return true; }
    bool string(string_t &) override { return true; }
    bool binary(binary_t &) override { return false; }
    bool start_object(std::size_t) override {
        check(++depth <= Descriptor::max_depth, "descriptor nesting limit");
        objects.emplace_back();
        return true;
    }
    bool key(string_t &key) override {
        check(objects.back().insert(key).second, "descriptor duplicate key");
        return true;
    }
    bool end_object() override {
        objects.pop_back();
        --depth;
        return true;
    }
    bool start_array(std::size_t) override {
        check(++depth <= Descriptor::max_depth, "descriptor nesting limit");
        return true;
    }
    bool end_array() override {
        --depth;
        return true;
    }
    bool parse_error(std::size_t, const std::string &, const Json::exception &) override {
        return false;
    }
};
Json strict_json(const std::string &bytes, std::size_t limit) {
    check(!bytes.empty() && bytes.size() <= limit, "encoded metadata/control byte limit");
    Guard guard;
    check(Json::sax_parse(bytes, &guard), "malformed descriptor/control JSON");
    return Json::parse(bytes);
}
std::vector<std::uint8_t> projection(const Json &value, std::size_t count) {
    check(value.is_array() && value.size() == count, "descriptor full projection required");
    std::vector<std::uint8_t> result;
    for (const auto &entry : value)
        result.push_back(static_cast<std::uint8_t>(integer(entry, 0, 255)));
    return result;
}
std::string projection_digest(const std::vector<std::uint8_t> &values) {
    // Exactly K ordered raw uint8 values, not canonicalized JSON or inferred meanings.
    return digest(std::string_view(reinterpret_cast<const char *>(values.data()), values.size()));
}
Startup::Time deadline(Startup::Time start, std::chrono::milliseconds duration) {
    check(start <= Startup::Time::max() - duration, "startup deadline overflow");
    return start + duration;
}
} // namespace
CpuAdmittedPackage::CpuAdmittedPackage(const fs::path &path) : package_(path, PackageUse::file) {}
void Limits::validate() const {
    check(max_frame_points >= 1 && max_frame_points <= 20000000, "frame point limit");
    check(max_frame_bytes >= 1 && max_frame_bytes <= 16777216, "frame byte limit");
    check(startup_timeout_ms >= 1 && startup_timeout_ms <= 120000, "startup timeout limit");
}
Limits Limits::negotiate(const Limits &local, const Limits &peer) {
    local.validate();
    peer.validate();
    return {std::min(local.max_frame_points, peer.max_frame_points),
            std::min(local.max_frame_bytes, peer.max_frame_bytes),
            std::min(local.startup_timeout_ms, peer.startup_timeout_ms)};
}
Descriptor::Descriptor(std::string bytes)
    : bytes_(std::move(bytes)), spec_(strict_json(bytes_, max_bytes)),
      taxonomy_(spec_.at("taxonomy_utf8").get<std::string>()) {
    keys(spec_,
         {"schema", "binary_wire_schema", "binary_wire_version", "package_envelope",
          "dense_adapter", "text_stream_schema", "context_policy_id", "context_policy_sha256",
          "manifest_sha256", "taxonomy_utf8", "taxonomy_sha256", "class_count", "las_projection",
          "las_projection_sha256", "session_id", "map_epoch", "limits"});
    check(spec_.at("schema") == "synrex.session.descriptor.v3", "descriptor schema");
    check(spec_.at("binary_wire_schema") == "synrex.binary.v3" &&
              integer(spec_.at("binary_wire_version"), 3, 3) == 3,
          "binary wire identity");
    check(spec_.at("package_envelope") == "synrex.srxmodel.v3", "package envelope identity");
    check(spec_.at("dense_adapter") == "DENSE_VOXEL_SEGMENTATION_V1", "dense adapter identity");
    check(spec_.at("text_stream_schema").is_null(), "generic text stream unsupported");
    check(spec_.at("context_policy_id") == "synrex.context.current_frame.v1" &&
              spec_.at("context_policy_sha256") == digest(context_bytes),
          "context policy identity");
    check(hex_string(spec_.at("manifest_sha256").get<std::string>(), 64), "manifest digest format");
    check(hex_string(spec_.at("session_id").get<std::string>(), 32), "session ID format");
    integer(spec_.at("map_epoch"), 1, UINT64_MAX);
    const auto count = integer(spec_.at("class_count"), 2, 256);
    taxonomy_.validate_channels(count);
    taxonomy_.validate_digest(spec_.at("taxonomy_sha256").get<std::string>());
    const auto projected = projection(spec_.at("las_projection"), count);
    taxonomy_.validate_projection(projected);
    check(spec_.at("las_projection_sha256") == projection_digest(projected),
          "projection digest mismatch");
    const auto &limits = spec_.at("limits");
    keys(limits, {"max_metadata_bytes", "metadata_timeout_ms", "max_frame_points",
                  "max_frame_bytes", "startup_timeout_ms"});
    integer(limits.at("max_metadata_bytes"), max_bytes, max_bytes);
    integer(limits.at("metadata_timeout_ms"), metadata_timeout_ms, metadata_timeout_ms);
    Limits{integer(limits.at("max_frame_points"), 1, 20000000),
           integer(limits.at("max_frame_bytes"), 1, 16777216),
           integer(limits.at("startup_timeout_ms"), 1, 120000)}
        .validate();
    sha256_ = digest(bytes_);
}
Descriptor Descriptor::parse(std::string_view bytes) {
    check(!bytes.empty() && bytes.size() <= max_bytes, "encoded metadata/control byte limit");
    return Descriptor(std::string(bytes));
}
Descriptor Descriptor::from_package(const CpuAdmittedPackage &admitted, const Limits &limits,
                                    const std::string &session_id, std::uint64_t map_epoch) {
    limits.validate();
    const auto &package = admitted.package_;
    // This object was privately constructed by the actual Package CPU reader;
    // no caller may provide, replace or mutate its raw or derived fields.
    const auto manifest = strict_json(package.manifest_bytes(), 1048576);
    check(digest(package.manifest_bytes()) == package.manifest_sha256() &&
              manifest == package.manifest(),
          "captured manifest identity mismatch");
    check(manifest.at("schema") == "synrex.srxmodel.v3" && package.taxonomy().has_value(),
          "generic descriptor requires explicit v3 taxonomy package");
    const auto &contract = package.contract();
    check(contract.spec == manifest.at("contract"), "captured Contract mismatch");
    check(limits.max_frame_points <= contract.max_points, "frame points exceed admitted Contract");
    const auto &taxonomy = *package.taxonomy();
    taxonomy.validate_digest(
        manifest.at("artifacts").at("taxonomy").at("sha256").get<std::string>());
    taxonomy.validate_channels(contract.las_classes.size());
    taxonomy.validate_projection(contract.las_classes);
    Json value{{"schema", "synrex.session.descriptor.v3"},
               {"binary_wire_schema", "synrex.binary.v3"},
               {"binary_wire_version", 3},
               {"package_envelope", "synrex.srxmodel.v3"},
               {"dense_adapter", contract.spec.at("adapter")},
               {"text_stream_schema", nullptr},
               {"context_policy_id", "synrex.context.current_frame.v1"},
               {"context_policy_sha256", digest(context_bytes)},
               {"manifest_sha256", package.manifest_sha256()},
               {"taxonomy_utf8", taxonomy.raw_bytes()},
               {"taxonomy_sha256", taxonomy.sha256()},
               {"class_count", taxonomy.classes().size()},
               {"las_projection", contract.las_classes},
               {"las_projection_sha256", projection_digest(contract.las_classes)},
               {"session_id", session_id},
               {"map_epoch", map_epoch},
               {"limits",
                {{"max_metadata_bytes", max_bytes},
                 {"metadata_timeout_ms", metadata_timeout_ms},
                 {"max_frame_points", limits.max_frame_points},
                 {"max_frame_bytes", limits.max_frame_bytes},
                 {"startup_timeout_ms", limits.startup_timeout_ms}}}};
    // Escaping and envelope overhead count. A valid 64KiB taxonomy may not fit.
    Descriptor result(value.dump());
    result.package_bound_ = true;
    return result;
}
std::string Descriptor::control(const std::string &kind) const {
    check(kind == "metadata_ack" || kind == "execution_ready" || kind == "execution_ready_ack",
          "unsupported control kind");
    return Json{{"schema", "synrex.session.control.v3"},
                {"kind", kind},
                {"descriptor_sha256", sha256_},
                {"session_id", spec_.at("session_id")},
                {"map_epoch", spec_.at("map_epoch")}}
        .dump();
}
Startup::Startup(Descriptor expected, Now now)
    : descriptor_(std::move(expected)), now_(std::move(now)) {
    check(descriptor_.package_bound_, "startup authority requires package-bound descriptor");
    check(bool(now_), "steady clock callback required");
    last_ = now_();
    metadata_deadline_ =
        deadline(last_, std::chrono::milliseconds(Descriptor::metadata_timeout_ms));
}
Startup::Time Startup::check_time() {
    check(state_ != State::failed, "startup permanently failed");
    const auto now = now_();
    check(now >= last_, "steady clock moved backwards");
    last_ = now;
    if (state_ == State::awaiting_descriptor || state_ == State::awaiting_metadata_ack)
        check(now < metadata_deadline_, "absolute metadata deadline expired");
    else if (state_ != State::ready)
        check(now < startup_deadline_, "absolute initialization deadline expired");
    return now;
}
void Startup::expect(State expected) { check(state_ == expected, "startup event out of order"); }
void Startup::poison(const std::exception &error) {
    if (state_ != State::failed) {
        failure_ = error.what();
        state_ = State::failed;
    }
}
void Startup::descriptor_fragment(std::string_view bytes, bool final) {
    try {
        check_time();
        expect(State::awaiting_descriptor);
        check(bytes.size() <= Descriptor::max_bytes - fragments_.size(),
              "descriptor fragments byte limit");
        fragments_.append(bytes.data(), bytes.size());
        if (final) {
            const auto received = Descriptor::parse(fragments_);
            check(received.sha256() == descriptor_.sha256(),
                  "expected descriptor identity mismatch");
            state_ = State::awaiting_metadata_ack;
        }
    } catch (const std::exception &error) {
        poison(error);
        throw;
    }
}
void Startup::check_control(const std::string &bytes, const std::string &kind) {
    const auto value = strict_json(bytes, 1024);
    keys(value, {"schema", "kind", "descriptor_sha256", "session_id", "map_epoch"});
    check(value.at("schema") == "synrex.session.control.v3" && value.at("kind") == kind,
          "control schema/kind mismatch");
    check(value.at("descriptor_sha256") == descriptor_.sha256() &&
              value.at("session_id") == descriptor_.spec().at("session_id") &&
              integer(value.at("map_epoch"), 1, UINT64_MAX) ==
                  descriptor_.spec().at("map_epoch").get<std::uint64_t>(),
          "control identity mismatch");
}
void Startup::metadata_ack(const std::string &control) {
    try {
        const auto now = check_time();
        expect(State::awaiting_metadata_ack);
        check_control(control, "metadata_ack");
        const auto timeout =
            descriptor_.spec().at("limits").at("startup_timeout_ms").get<std::uint64_t>();
        startup_deadline_ = deadline(now, std::chrono::milliseconds(timeout));
        state_ = State::gpu_permitted;
    } catch (const std::exception &error) {
        poison(error);
        throw;
    }
}
bool Startup::gpu_permission() {
    try {
        check_time();
        return state_ == State::gpu_permitted || state_ == State::initializing ||
               state_ == State::awaiting_execution_ready_ack || state_ == State::ready;
    } catch (const std::exception &error) {
        poison(error);
        throw;
    }
}
void Startup::begin_initialization() {
    try {
        check_time();
        expect(State::gpu_permitted);
        state_ = State::initializing;
    } catch (const std::exception &error) {
        poison(error);
        throw;
    }
}
void Startup::initialization_complete() {
    try {
        check_time();
        expect(State::initializing);
        state_ = State::awaiting_execution_ready_ack;
    } catch (const std::exception &error) {
        poison(error);
        throw;
    }
}
void Startup::execution_ready_ack(const std::string &control) {
    try {
        check_time();
        expect(State::awaiting_execution_ready_ack);
        check_control(control, "execution_ready_ack");
        state_ = State::ready;
    } catch (const std::exception &error) {
        poison(error);
        throw;
    }
}
void Startup::accept_frame_identity(std::string_view hash, std::string_view session_id,
                                    std::uint64_t map_epoch) {
    try {
        check_time();
        expect(State::ready);
        check(hash == descriptor_.sha256() &&
                  session_id == descriptor_.spec().at("session_id").get<std::string>() &&
                  map_epoch == descriptor_.spec().at("map_epoch").get<std::uint64_t>(),
              "frame identity mismatch");
    } catch (const std::exception &error) {
        poison(error);
        throw;
    }
}
void Startup::poll() {
    try {
        check_time();
    } catch (const std::exception &error) {
        poison(error);
        throw;
    }
}
void Startup::fail(std::string reason) {
    if (state_ != State::failed) {
        failure_ = std::move(reason);
        state_ = State::failed;
    }
}
} // namespace synrex::generic
