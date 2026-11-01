// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Tuomas Pietilä
#include "synrex/point_session.hpp"
#include <algorithm>
#include <cmath>
#include <cstring>
#include <iomanip>
#include <openssl/evp.h>
#include <set>
#include <sstream>

namespace synrex::point {
namespace {
constexpr std::uint64_t json_integer_max = 9007199254740991ULL;
void keys(const Json &j, std::initializer_list<const char *> names) {
    require(j.is_object() && j.size() == names.size(), "point unknown/missing fields");
    for (const auto *name : names)
        require(j.contains(name), "point missing field");
}
bool hash(const std::string &s) {
    return s.size() == 64 && std::all_of(s.begin(), s.end(), [](unsigned char c) {
               return (c >= '0' && c <= '9') || (c >= 'a' && c <= 'f');
           });
}
std::string text(const Json &j, std::size_t limit = 256) {
    require(j.is_string(), "point string required");
    auto s = j.get<std::string>();
    require(!s.empty() && s.size() <= limit && s.find('\0') == std::string::npos,
            "point invalid string");
    return s;
}
std::uint64_t integer(const Json &j, std::uint64_t cap) {
    require(j.is_number_integer() && (!j.is_number_unsigned() ? j.get<std::int64_t>() >= 0 : true),
            "point nonnegative integer required");
    const auto n = j.get<std::uint64_t>();
    require(n <= cap, "point integer exceeds limit");
    return n;
}
std::string canonical(const Json &j) { return j.dump(-1, ' ', true); }
Json parse(const std::string &s) {
    require(!s.empty() && s.size() <= 1048576, "point oversized JSON");
    std::vector<std::set<std::string>> names;
    std::size_t nodes = 0;
    auto callback = [&](int depth, Json::parse_event_t event, Json &value) {
        require(depth <= 32 && ++nodes <= 100000, "point JSON complexity limit");
        if (event == Json::parse_event_t::object_start)
            names.emplace_back();
        if (event == Json::parse_event_t::key)
            require(names.back().insert(value.get<std::string>()).second, "point duplicate key");
        if (event == Json::parse_event_t::object_end)
            names.pop_back();
        if (event == Json::parse_event_t::value && value.is_number_float())
            require(std::isfinite(value.get<double>()), "point nonfinite JSON");
        return true;
    };
    return Json::parse(s, callback);
}
bool path(const std::string &s) {
    if (s.empty() || s.size() > 240 || s.front() == '/' || s.back() == '/')
        return false;
    if (!((s[0] >= 'A' && s[0] <= 'Z') || (s[0] >= 'a' && s[0] <= 'z') ||
          (s[0] >= '0' && s[0] <= '9') || s[0] == '_'))
        return false;
    std::istringstream parts(s);
    std::string part;
    while (std::getline(parts, part, '/'))
        if (part.empty() || part == "." || part == "..")
            return false;
    return std::all_of(s.begin(), s.end(), [](unsigned char c) {
        return (c >= 'A' && c <= 'Z') || (c >= 'a' && c <= 'z') || (c >= '0' && c <= '9') ||
               c == '_' || c == '.' || c == '/' || c == '-';
    });
}
void append(std::string &s, std::uint64_t n, unsigned bytes) {
    for (unsigned i = 0; i < bytes; ++i)
        s.push_back(static_cast<char>((n >> (i * 8)) & 255));
}
void floats(std::string &s, const std::vector<float> &v) {
    static_assert(sizeof(float) == 4 && std::numeric_limits<float>::is_iec559,
                  "point payload requires IEEE754 float32");
    for (const auto value : v) {
        require(std::isfinite(value), "point nonfinite model input");
        std::uint32_t bits{};
        std::memcpy(&bits, &value, sizeof(bits));
        append(s, bits, 4);
    }
}
unsigned peer(Peer p) {
    require(p == Peer::input || p == Peer::output, "point invalid peer");
    return p == Peer::input ? 0 : 1;
}
void source_valid(const SourceIdentity &s, std::uint64_t cap) {
    require(hash(s.sha256) && hash(s.header_sha256) && s.rows <= cap, "point invalid source");
}
bool same_source(const SourceIdentity &a, const SourceIdentity &b) {
    return a.sha256 == b.sha256 && a.header_sha256 == b.header_sha256 && a.rows == b.rows;
}
} // namespace
std::string digest_bytes(const std::string &bytes) {
    unsigned char out[32];
    unsigned count{};
    require(EVP_Digest(bytes.data(), bytes.size(), out, &count, EVP_sha256(), nullptr) == 1 &&
                count == 32,
            "point SHA256 failure");
    std::ostringstream s;
    for (auto b : out)
        s << std::hex << std::setfill('0') << std::setw(2) << unsigned(b);
    return s.str();
}
PointArtifact::PointArtifact(std::string bytes, const std::string &expected, Members members,
                             const Validator &validate) {
    require(hash(expected) && digest_bytes(bytes) == expected, "point untrusted manifest bytes");
    auto manifest = parse(bytes);
    keys(manifest, {"schema", "package_id", "backend_kind", "model_id", "execution_mode",
                    "schema_bundle_sha256", "members"});
    require(manifest.at("schema") == "synrex.srxmodel.v4" &&
                manifest.at("backend_kind") == "POINT_MODEL",
            "point package schema/backend mismatch");
    text(manifest.at("package_id"));
    text(manifest.at("model_id"));
    require(manifest.at("execution_mode") == "REFERENCE" ||
                manifest.at("execution_mode") == "HYBRID",
            "point unsupported execution mode");
    require(hash(text(manifest.at("schema_bundle_sha256"))), "point schema bundle identity");
    const auto &roster = manifest.at("members");
    require(roster.is_array() && roster.size() >= 12 && roster.size() <= 128 &&
                roster.size() == members.size(),
            "point member roster limit/mismatch");
    const std::map<std::string, std::uint64_t> caps{{"model_state", 67108864},
                                                    {"tensor_manifest", 1048576},
                                                    {"architecture", 1048576},
                                                    {"worker_source", 1048576},
                                                    {"preprocess", 1048576},
                                                    {"context_policy", 1048576},
                                                    {"precision", 1048576},
                                                    {"taxonomy", 65536},
                                                    {"target_lock", 1048576},
                                                    {"point_contract", 1048576},
                                                    {"notice", 1048576},
                                                    {"source_file", 8388608},
                                                    {"schema_document", 1048576},
                                                    {"qualification_evidence", 1048576},
                                                    {"qualification_proof", 8388608}};
    const std::set<std::string> repeated{"source_file", "schema_document", "qualification_evidence",
                                         "qualification_proof"};
    std::set<std::string> paths, singletons;
    std::uint64_t total = bytes.size();
    for (const auto &member : roster) {
        keys(member, {"path", "role", "bytes", "sha256", "document_schema"});
        const auto name = text(member.at("path"), 240), role = text(member.at("role"));
        require(path(name) && name != "MANIFEST.json" && paths.insert(name).second,
                "point unsafe/duplicate member path");
        const auto cap = caps.find(role);
        require(cap != caps.end() && (repeated.count(role) || singletons.insert(role).second),
                "point unknown/duplicate member role");
        const auto size = integer(member.at("bytes"), cap->second);
        const auto digest = text(member.at("sha256"));
        require(size > 0 && hash(digest), "point member size/hash");
        require(total <= 268435456 - size, "point aggregate package cap");
        total += size;
        auto found = members.find(name);
        require(found != members.end() && found->second.size() == size &&
                    digest_bytes(found->second) == digest,
                "point changed/missing member bytes");
        if (!member.at("document_schema").is_null()) {
            const auto schema = text(member.at("document_schema"));
            const auto document = parse(found->second);
            require(document.is_object() &&
                        document.value(role == "schema_document" ? "$id" : "schema", "") == schema,
                    "point document schema mismatch");
        }
        if (role == "point_contract")
            contract_sha256_ = digest;
        if (role == "worker_source")
            worker_sha256_ = digest;
    }
    for (const auto &[role, cap] : caps) {
        (void)cap;
        if (!repeated.count(role))
            require(singletons.count(role), "point missing singleton role");
    }
    require(bool(validate), "point requires trusted application validator");
    auto contract = validate(manifest, members);
    require(!contract.adapter_id.empty() && contract.adapter_id.size() <= 256 &&
                !contract.layout_id.empty() && contract.layout_id.size() <= 256 &&
                contract.model_id == manifest.at("model_id") &&
                contract.execution_mode == manifest.at("execution_mode") &&
                hash(contract.environment_sha256) && hash(contract.policy_sha256) &&
                (contract.selected_capability == "file" ||
                 contract.selected_capability == "buffered_same_batch" ||
                 contract.selected_capability == "causal"),
            "point validated contract identity mismatch");
    require(contract.las_projection.size() >= 2 && contract.las_projection.size() <= 256 &&
                contract.max_source_rows > 0 && contract.max_source_rows <= 1000000 &&
                contract.max_representatives > 0 && contract.max_representatives <= 65536 &&
                contract.max_input_bytes > 0 && contract.max_input_bytes <= 67108864 &&
                contract.max_output_bytes > 0 && contract.max_output_bytes <= 67108864,
            "point validated contract bounds");
    manifest_ = std::move(manifest);
    manifest_bytes_ = std::move(bytes);
    manifest_sha256_ = expected;
    members_ = std::move(members);
    contract_ = std::move(contract);
}
struct PointSession::Impl {
    struct Source {
        SourceIdentity identity;
        std::vector<std::uint8_t> accepted, emitted, semantic;
        std::uint64_t count{};
    };
    std::shared_ptr<const PointArtifact> artifact;
    Factory factory;
    SessionOptions options;
    std::unique_ptr<PointInferenceSession> backend;
    State state{State::metadata_pending};
    Json descriptor, init, completion;
    std::string descriptor_hash, init_hash, completion_hash, current;
    bool metadata[2]{}, ready[2]{}, completed[2]{};
    std::uint64_t epoch{}, request{1}, rows{}, emitted{};
    std::map<std::string, SourceIdentity> expected;
    std::map<std::string, Source> sources;
    void active() const { require(state == State::active, "point session is not active"); }
};
PointSession::PointSession(std::shared_ptr<const PointArtifact> artifact, std::string id,
                           Factory factory, SessionOptions options)
    : impl_(std::make_unique<Impl>()) {
    require(artifact && bool(factory) && id.size() == 36, "point missing retained session ingress");
    for (std::size_t i = 0; i < id.size(); ++i)
        require((i == 8 || i == 13 || i == 18 || i == 23)
                    ? id[i] == '-'
                    : ((id[i] >= '0' && id[i] <= '9') || (id[i] >= 'a' && id[i] <= 'f')),
                "point session UUID required");
    require(options.max_sources > 0 && options.max_sources <= 64 && options.max_total_rows > 0 &&
                options.max_total_rows <= 1000000 &&
                !(options.synthetic_fixture && options.qualification_only),
            "point session limits/mode");
    require(artifact->contract().capability_qualified || options.synthetic_fixture ||
                options.qualification_only,
            "point capability lacks qualification");
    auto &s = *impl_;
    s.artifact = std::move(artifact);
    s.factory = std::move(factory);
    s.options = options;
    require(!options.sources.empty() && options.sources.size() <= options.max_sources,
            "point exact source roster required");
    Json roster = Json::array();
    std::uint64_t total_rows = 0;
    for (const auto &source : options.sources) {
        source_valid(source, s.artifact->contract().max_source_rows);
        require(s.expected.emplace(source.sha256, source).second &&
                    source.rows <= options.max_total_rows - total_rows,
                "point declared source duplicate/aggregate cap");
        total_rows += source.rows;
        roster.push_back({{"source_sha256", source.sha256},
                          {"header_sha256", source.header_sha256},
                          {"rows", source.rows}});
    }
    s.descriptor = {{"schema", "synrex.point.native-session.v1"},
                    {"required_wire_version", 4},
                    {"session_id", id},
                    {"package_manifest_sha256", s.artifact->manifest_sha256()},
                    {"point_contract_sha256", s.artifact->contract_sha256()},
                    {"selected_capability", s.artifact->contract().selected_capability},
                    {"policy_sha256", s.artifact->contract().policy_sha256},
                    {"source_layout", s.artifact->contract().layout_id},
                    {"synthetic_fixture", options.synthetic_fixture},
                    {"qualification_only", options.qualification_only},
                    {"max_sources", options.max_sources},
                    {"max_total_rows", options.max_total_rows},
                    {"sources", roster}};
    s.descriptor_hash = digest_bytes(canonical(s.descriptor));
}
PointSession::~PointSession() = default;
State PointSession::state() const { return impl_->state; }
const std::string &PointSession::descriptor_sha256() const { return impl_->descriptor_hash; }
Json PointSession::descriptor() const { return impl_->descriptor; }
void PointSession::fail() {
    impl_->state = State::failed;
    impl_->backend.reset();
}
void PointSession::metadata_ack(Peer p, const std::string &digest) {
    try {
        auto &s = *impl_;
        const auto n = peer(p);
        require(s.state == State::metadata_pending && !s.metadata[n] && digest == s.descriptor_hash,
                "point metadata ACK state/identity");
        s.metadata[n] = true;
        if (s.metadata[0] && s.metadata[1])
            s.state = State::initializing;
    } catch (...) {
        fail();
        throw;
    }
}
Json PointSession::initialize() {
    try {
        auto &s = *impl_;
        require(s.state == State::initializing && !s.backend,
                "point initialization before dual metadata admission/repeated");
        s.backend = s.factory();
        require(bool(s.backend), "point null backend");
        auto receipt = s.backend->initialize(*s.artifact);
        keys(receipt, {"schema", "package_manifest_sha256", "point_contract_sha256",
                       "worker_source_sha256", "environment_sha256", "fixture", "details"});
        require(receipt.at("schema") == "synrex.point.initialization.v1" &&
                    receipt.at("package_manifest_sha256") == s.artifact->manifest_sha256() &&
                    receipt.at("point_contract_sha256") == s.artifact->contract_sha256() &&
                    receipt.at("worker_source_sha256") == s.artifact->worker_sha256() &&
                    receipt.at("environment_sha256") == s.artifact->contract().environment_sha256 &&
                    receipt.at("fixture").is_boolean() &&
                    receipt.at("fixture") == s.options.synthetic_fixture &&
                    receipt.at("details").is_object() && canonical(receipt).size() <= 65536,
                "point retained backend initialization identity");
        s.init = std::move(receipt);
        s.init_hash = digest_bytes(canonical(s.init));
        s.state = State::ready_pending;
        return {{"initialization_sha256", s.init_hash}, {"receipt", s.init}};
    } catch (...) {
        fail();
        throw;
    }
}
void PointSession::ready_ack(Peer p, const std::string &digest) {
    try {
        auto &s = *impl_;
        const auto n = peer(p);
        require(s.state == State::ready_pending && !s.ready[n] && digest == s.init_hash,
                "point ready ACK state/identity");
        s.ready[n] = true;
        if (s.ready[0] && s.ready[1]) {
            s.backend->activate();
            s.state = State::active;
        }
    } catch (...) {
        fail();
        throw;
    }
}
void PointSession::open_source(const SourceIdentity &source) {
    try {
        auto &s = *impl_;
        s.active();
        source_valid(source, s.artifact->contract().max_source_rows);
        require(s.expected.count(source.sha256) &&
                    same_source(s.expected.at(source.sha256), source) &&
                    !s.sources.count(source.sha256) && s.sources.size() < s.options.max_sources &&
                    source.rows <= s.options.max_total_rows - s.rows,
                "point source replay/aggregate cap");
        if (!s.current.empty()) {
            const auto &old = s.sources.at(s.current);
            require(old.count == old.identity.rows, "point reset discards incomplete source");
        }
        require(s.epoch < json_integer_max, "point epoch overflow");
        ++s.epoch;
        s.backend->open_source(source, s.epoch);
        Impl::Source retained{source, std::vector<std::uint8_t>(source.rows),
                              std::vector<std::uint8_t>(source.rows),
                              std::vector<std::uint8_t>(source.rows), 0};
        s.sources.emplace(source.sha256, std::move(retained));
        s.current = source.sha256;
        s.rows += source.rows;
    } catch (...) {
        fail();
        throw;
    }
}
Classification PointSession::run(const PreparedBatch &input) {
    try {
        auto &s = *impl_;
        s.active();
        const auto &c = s.artifact->contract();
        require(!s.current.empty(), "point request without source");
        auto &source = s.sources.at(s.current);
        require(same_source(input.source, source.identity), "point changed request source");
        const auto n = input.representative_source_rows.size(), count = input.current_rows.size();
        require(n > 0 && n <= c.max_representatives && count > 0 && count <= c.max_source_rows &&
                    input.inverse.size() == count && input.coord.size() == n * 3 &&
                    input.grid_coord.size() == n * 3 && input.feat.size() == n * 4 &&
                    48 * n + 8 <= c.max_input_bytes && 2 * n <= c.max_output_bytes &&
                    count <= c.max_output_bytes / 10 && input.cutoff_ns <= json_integer_max &&
                    hash(input.logical_batch_sha256),
                "point model input count/shape/byte bound");
        require(s.request < json_integer_max, "point request overflow");
        std::set<std::uint64_t> current, representatives;
        for (const auto row : input.representative_source_rows)
            require(row < source.identity.rows && representatives.insert(row).second,
                    "point representative source row range/duplicate");
        for (std::size_t i = 0; i < count; ++i)
            require(input.current_rows[i] < source.identity.rows &&
                        current.insert(input.current_rows[i]).second &&
                        !source.accepted[input.current_rows[i]] &&
                        !source.emitted[input.current_rows[i]] && input.inverse[i] < n,
                    "point current row/inverse/replay");
        // Freeze before handing control to an implementation; all row fences and
        // expansion below use this retained copy, never worker-supplied maps.
        for (auto value : input.grid_coord)
            require(value >= 0, "point negative lattice coordinate");
        const PreparedBatch batch = input;
        std::string payload;
        payload.reserve(48 * n + 8);
        floats(payload, batch.coord);
        for (auto value : batch.grid_coord)
            append(payload, static_cast<std::uint32_t>(value), 4);
        floats(payload, batch.feat);
        append(payload, n, 8);
        for (auto value : batch.representative_source_rows)
            append(payload, value, 8);
        NativeAuthority authority;
        authority.epoch = s.epoch;
        authority.request_id = s.request++;
        authority.input_payload_sha256 = digest_bytes(payload);
        authority.record = {{"schema", "synrex.point.native-authority.v1"},
                            {"descriptor_sha256", s.descriptor_hash},
                            {"epoch", authority.epoch},
                            {"request_id", authority.request_id},
                            {"source_sha256", source.identity.sha256},
                            {"source_header_sha256", source.identity.header_sha256},
                            {"source_rows", source.identity.rows},
                            {"current_rows", batch.current_rows},
                            {"inverse", batch.inverse},
                            {"representatives", n},
                            {"input_payload_sha256", authority.input_payload_sha256},
                            {"logical_batch_sha256", batch.logical_batch_sha256},
                            {"cutoff_ns", batch.cutoff_ns}};
        authority.sha256 = digest_bytes(canonical(authority.record));
        for (auto row : batch.current_rows)
            source.accepted[row] = 1;
        const auto result = s.backend->run(batch, authority);
        require(result.epoch == authority.epoch && result.request_id == authority.request_id &&
                    result.authority_sha256 == authority.sha256 &&
                    result.semantic_ids.size() == n && result.validity.size() == n,
                "point result identity/count mismatch");
        for (std::size_t i = 0; i < n; ++i)
            require(result.semantic_ids[i] < c.las_projection.size() && result.validity[i] <= 1,
                    "point invalid semantic/validity");
        Classification output{source.identity,
                              batch.current_rows,
                              {},
                              {},
                              {},
                              authority.epoch,
                              authority.request_id,
                              batch.logical_batch_sha256,
                              authority.sha256};
        for (auto index : batch.inverse) {
            require(result.validity[index] == 1, "point invalid current representative");
            output.semantic_ids.push_back(result.semantic_ids[index]);
            output.validity.push_back(1);
            output.las_codes.push_back(c.las_projection[result.semantic_ids[index]]);
        }
        // No source/output mutation happens until the entire result is checked.
        for (std::size_t i = 0; i < count; ++i) {
            const auto row = batch.current_rows[i];
            source.emitted[row] = 1;
            source.semantic[row] = output.semantic_ids[i];
        }
        source.count += count;
        s.emitted += count;
        return output;
    } catch (...) {
        fail();
        throw;
    }
}
Json PointSession::complete(const OutputCommitter &commit) {
    try {
        auto &s = *impl_;
        s.active();
        require(bool(commit) && !s.sources.empty(), "point missing output authority/source");
        require(s.rows == s.emitted && s.sources.size() == s.expected.size(),
                "point incomplete total source coverage/roster");
        std::map<std::string, std::vector<std::uint8_t>> output;
        Json coverage = Json::array();
        for (const auto &[key, source] : s.sources) {
            require(source.count == source.identity.rows &&
                        std::all_of(source.emitted.begin(), source.emitted.end(),
                                    [](auto b) { return b == 1; }),
                    "point incomplete source coverage");
            output.emplace(key, source.semantic);
            std::string bits(source.emitted.begin(), source.emitted.end());
            coverage.push_back(
                {{"source_sha256", key},
                 {"rows", source.identity.rows},
                 {"coverage_sha256", digest_bytes(bits)},
                 {"semantic_sha256",
                  digest_bytes(std::string(source.semantic.begin(), source.semantic.end()))}});
        }
        const auto durable = commit(output);
        require(hash(durable), "point invalid durable output commitment");
        s.completion = {{"schema", "synrex.point.native-completion.v1"},
                        {"descriptor_sha256", s.descriptor_hash},
                        {"source_coverage", coverage},
                        {"emitted_rows", s.emitted},
                        {"requests", s.request - 1},
                        {"last_request_id", s.request - 1},
                        {"epoch", s.epoch},
                        {"durable_output_sha256", durable},
                        {"synthetic_fixture", s.options.synthetic_fixture},
                        {"qualification_only", s.options.qualification_only}};
        s.backend->complete(s.completion);
        s.completion_hash = digest_bytes(canonical(s.completion));
        s.state = State::complete_pending;
        return {{"completion_sha256", s.completion_hash}, {"receipt", s.completion}};
    } catch (...) {
        fail();
        throw;
    }
}
void PointSession::completion_ack(Peer p, const std::string &digest) {
    try {
        auto &s = *impl_;
        const auto n = peer(p);
        require(s.state == State::complete_pending && !s.completed[n] &&
                    digest == s.completion_hash,
                "point completion ACK state/identity");
        s.completed[n] = true;
        if (s.completed[0] && s.completed[1])
            s.state = State::complete;
    } catch (...) {
        fail();
        throw;
    }
}
} // namespace synrex::point
