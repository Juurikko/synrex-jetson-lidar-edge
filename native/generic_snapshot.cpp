// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Tuomas Pietilä
#include "synrex/generic_map.hpp"
#include <algorithm>
#include <cmath>
#include <set>

namespace synrex::generic {
namespace {
void keys(const Json &value, std::initializer_list<const char *> expected) {
    require(value.is_object() && value.size() == expected.size(), "generic snapshot object fields");
    for (const auto key : expected)
        require(value.contains(key), "generic snapshot missing field");
}
std::uint64_t integer(const Json &value) {
    require(value.is_number_integer(), "generic snapshot integer required");
    require(value.is_number_unsigned() || value.get<std::int64_t>() >= 0,
            "generic snapshot negative integer");
    return value.get<std::uint64_t>();
}
// Bound nesting, decoded strings, object members and array elements before DOM
// allocation; duplicate escaped aliases are rejected in every object. Four levels
// suffice for root -> cells -> cell -> key. Descriptor JSON remains an exact string.
struct SnapshotGuard : nlohmann::json_sax<Json> {
    struct Container {
        bool object;
        std::size_t count{};
        std::set<std::string> keys;
    };
    std::vector<Container> stack;
    bool value() {
        if (!stack.empty() && !stack.back().object)
            require(++stack.back().count <= SemanticMap::max_cells, "generic snapshot array bound");
        return true;
    }
    bool null() override { return value(); }
    bool boolean(bool) override { return value(); }
    bool number_integer(number_integer_t) override { return value(); }
    bool number_unsigned(number_unsigned_t) override { return value(); }
    bool number_float(number_float_t number, const string_t &) override {
        require(std::isfinite(number), "generic snapshot finite number required");
        return value();
    }
    bool string(string_t &text) override {
        require(text.size() <= Descriptor::max_bytes, "generic snapshot string bound");
        return value();
    }
    bool binary(binary_t &) override { return false; }
    bool start(bool object) {
        value();
        require(stack.size() < 4, "generic snapshot nesting limit");
        stack.push_back({object, 0, {}});
        return true;
    }
    bool start_object(std::size_t) override { return start(true); }
    bool start_array(std::size_t) override { return start(false); }
    bool key(string_t &text) override {
        require(text.size() <= 64 && stack.back().keys.size() < 16,
                "generic snapshot object bound");
        require(stack.back().keys.insert(text).second, "generic snapshot duplicate key");
        return true;
    }
    bool end_object() override {
        stack.pop_back();
        return true;
    }
    bool end_array() override {
        stack.pop_back();
        return true;
    }
    bool parse_error(std::size_t, const std::string &, const Json::exception &) override {
        return false;
    }
};
} // namespace
Json admit_snapshot(const Descriptor &descriptor, std::string_view bytes) {
    require(!bytes.empty() && bytes.size() <= SemanticMap::max_snapshot_bytes,
            "generic snapshot encoded byte limit");
    // The JSON library treats a literal NUL as end-of-input, which could hide
    // trailing bytes. JSON text cannot contain a literal NUL anywhere.
    require(bytes.find('\0') == std::string_view::npos, "generic snapshot literal NUL byte");
    SnapshotGuard guard;
    require(Json::sax_parse(bytes.begin(), bytes.end(), &guard), "generic snapshot malformed JSON");
    auto result = Json::parse(bytes);
    keys(result, {"schema", "binding", "descriptor_bytes", "source_identity", "as_of_ns", "policy",
                  "resolution_metres", "capacity", "mode", "cells"});
    require(result.at("schema") == "synrex.generic_map_snapshot.v1" &&
                result.at("mode") == "map_delta_output" &&
                result.at("policy") == "last_accepted_observation_revision_eviction_v1",
            "generic snapshot schema/mode/policy");
    const auto &bound = result.at("binding");
    keys(bound, {"descriptor_sha256", "session_id", "map_epoch", "taxonomy_sha256"});
    require(bound.at("descriptor_sha256") == descriptor.sha256() &&
                bound.at("session_id") == descriptor.spec().at("session_id") &&
                integer(bound.at("map_epoch")) ==
                    descriptor.spec().at("map_epoch").get<std::uint64_t>() &&
                bound.at("taxonomy_sha256") == descriptor.taxonomy().sha256(),
            "generic snapshot descriptor/session/epoch/taxonomy mismatch");
    require(result.at("descriptor_bytes").is_string() &&
                result.at("descriptor_bytes").get_ref<const std::string &>() == descriptor.bytes(),
            "generic snapshot exact descriptor bytes mismatch");
    const auto cells = integer(result.at("capacity"));
    require(cells > 0 && cells <= SemanticMap::max_cells, "generic snapshot capacity bound");
    require(result.at("resolution_metres").is_number(), "generic snapshot resolution numeric");
    const auto resolution = result.at("resolution_metres").get<double>();
    require(std::isfinite(resolution) && resolution >= 0.001 && resolution <= 1000,
            "generic snapshot resolution bound");
    const auto as_of = integer(result.at("as_of_ns"));
    const auto &entries = result.at("cells");
    require(entries.is_array() && entries.size() <= cells, "generic snapshot cell count bound");
    const auto &source = result.at("source_identity");
    require(source.is_null() || (source.is_array() && source.size() == 4),
            "generic snapshot source identity shape");
    require(entries.empty() == source.is_null(), "generic snapshot source/cell consistency");
    if (!source.is_null()) {
        for (const auto &part : source) {
            require(part.is_string(), "generic snapshot source identity string");
            const auto &text = part.get_ref<const std::string &>();
            require(!text.empty() && text.size() <= 256 &&
                        std::all_of(text.begin(), text.end(),
                                    [](unsigned char c) { return c >= 32 && c <= 126; }),
                    "generic snapshot source identity bound/printable ASCII");
        }
    }
    const auto low = static_cast<std::int64_t>(std::floor(-1e9 / resolution));
    const auto high = static_cast<std::int64_t>(std::floor(1e9 / resolution));
    std::optional<std::array<std::int64_t, 3>> previous;
    for (const auto &entry : entries) {
        keys(entry, {"key", "semantic_id", "timestamp_ns", "classification_age_ns", "confidence",
                     "confidence_age_ns"});
        const auto &encoded_key = entry.at("key");
        require(encoded_key.is_array() && encoded_key.size() == 3,
                "generic snapshot cell key shape");
        std::array<std::int64_t, 3> key{};
        for (std::size_t axis = 0; axis < key.size(); ++axis) {
            const auto &component = encoded_key.at(axis);
            require(component.is_number_integer(), "generic snapshot cell key integer");
            if (component.is_number_unsigned())
                require(component.get<std::uint64_t>() <= static_cast<std::uint64_t>(high),
                        "generic snapshot cell key range");
            key[axis] = component.get<std::int64_t>();
            require(key[axis] >= low && key[axis] <= high, "generic snapshot cell key range");
        }
        require(!previous || *previous < key, "generic snapshot duplicate/unordered cell key");
        previous = key;
        require(integer(entry.at("semantic_id")) < descriptor.taxonomy().classes().size(),
                "generic snapshot semantic ID outside bound taxonomy/LAS projection");
        const auto timestamp = integer(entry.at("timestamp_ns"));
        require(timestamp <= as_of &&
                    integer(entry.at("classification_age_ns")) == as_of - timestamp,
                "generic snapshot timestamp/age/as_of mismatch");
        require(entry.at("confidence").is_null() && entry.at("confidence_age_ns").is_null(),
                "generic snapshot confidence unsupported");
    }
    return result;
}
} // namespace synrex::generic
