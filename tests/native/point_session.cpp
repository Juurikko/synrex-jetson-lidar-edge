// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Tuomas Pietilä
#include "synrex/point_session.hpp"
#include "synrex/common_session.hpp"
#include "synrex/point_prepared_wire.hpp"
#include "generic_fixture_backend.hpp"
#include <sys/socket.h>
#include <chrono>
#include <future>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <type_traits>
#include <unistd.h>

namespace synrex {
Json target_info() { throw std::runtime_error("point fixture reached device probe"); }
} // namespace synrex
using namespace synrex;
using namespace synrex::point;
static_assert(!std::is_copy_assignable<PointArtifact>::value &&
                  !std::is_move_assignable<PointArtifact>::value,
              "retained artifacts cannot be replaced behind an admitted session");
namespace {
void check(bool ok) {
    if (!ok)
        throw std::runtime_error("point fixture assertion");
}
template <class F> void refuses(F f) {
    bool rejected = false;
    try {
        f();
    } catch (const std::exception &) {
        rejected = true;
    }
    check(rejected);
}
struct PackageFixture {
    Members members;
    Json manifest;
    PointContract contract;
    PackageFixture() {
        Json roster = Json::array();
        for (const auto &role :
             {"model_state", "tensor_manifest", "architecture", "worker_source", "preprocess",
              "context_policy", "precision", "taxonomy", "target_lock", "point_contract", "notice",
              "source_file", "schema_document"}) {
            const std::string name = std::string(role) + ".bin",
                              bytes = std::string(role) + "-SYNTHETIC";
            members[name] = bytes;
            roster.push_back({{"path", name},
                              {"role", role},
                              {"bytes", bytes.size()},
                              {"sha256", digest_bytes(bytes)},
                              {"document_schema", nullptr}});
        }
        // Exercise actual JSON Schema $id and typed document schema admission.
        for (auto &entry : roster) {
            const std::string role = entry.at("role"), name = entry.at("path");
            if (role == "schema_document") {
                members[name] = Json({{"$schema", "https://json-schema.org/draft/2020-12/schema"},
                                      {"$id", "synthetic.point.v1"},
                                      {"type", "object"}})
                                    .dump();
                entry["document_schema"] = "synthetic.point.v1";
            } else if (role == "point_contract" || role == "worker_source") {
                const auto schema = "synthetic." + role + ".v1";
                members[name] = Json({{"schema", schema}, {"fixture", true}}).dump();
                entry["document_schema"] = schema;
            }
            entry["bytes"] = members[name].size();
            entry["sha256"] = digest_bytes(members[name]);
        }
        manifest = {{"schema", "synrex.srxmodel.v4"},
                    {"package_id", "synthetic-package"},
                    {"backend_kind", "POINT_MODEL"},
                    {"model_id", "synthetic-model"},
                    {"execution_mode", "REFERENCE"},
                    {"schema_bundle_sha256", digest_bytes("synthetic-schema")},
                    {"members", roster}};
        contract = {"synthetic-adapter",
                    "synthetic-model",
                    "synthetic-layout",
                    "REFERENCE",
                    digest_bytes("synthetic-environment"),
                    {2, 1, 6},
                    8,
                    8,
                    4096,
                    4096,
                    false,
                    "file",
                    digest_bytes("synthetic-policy")};
    }
    std::shared_ptr<const PointArtifact> admit() const {
        const auto bytes = manifest.dump();
        return std::make_shared<const PointArtifact>(
            bytes, digest_bytes(bytes), members,
            [this](const Json &, const Members &) { return contract; });
    }
};
struct Control {
    unsigned made{}, initialized{}, runs{}, opened{}, completed{}, retired{}, activated{};
    std::string fault;
};
struct Backend : PointInferenceSession {
    Control &control;
    std::uint64_t last_epoch{}, last_request{};
    explicit Backend(Control &c) : control(c) { ++control.made; }
    ~Backend() override { ++control.retired; }
    Json initialize(const PointArtifact &p) override {
        ++control.initialized;
        if (control.fault == "crash_init")
            throw std::runtime_error("synthetic crash");
        return {{"schema", "synrex.point.initialization.v1"},
                {"package_manifest_sha256", p.manifest_sha256()},
                {"point_contract_sha256", p.contract_sha256()},
                {"worker_source_sha256", p.worker_sha256()},
                {"environment_sha256", p.contract().environment_sha256},
                {"fixture", control.fault != "pretend_learned"},
                {"details", Json::object()}};
    }
    void activate() override {
        ++control.activated;
        if (control.fault == "activate_crash")
            throw std::runtime_error("synthetic activate crash");
    }
    void open_source(const SourceIdentity &, std::uint64_t epoch) override {
        check(epoch == last_epoch + 1);
        last_epoch = epoch;
        ++control.opened;
    }
    SemanticResult run(const PreparedBatch &b, const NativeAuthority &a) override {
        ++control.runs;
        check(a.epoch == last_epoch && a.request_id == ++last_request);
        check(digest_bytes(a.record.dump(-1, ' ', true)) == a.sha256);
        check(a.record.at("current_rows") == b.current_rows && a.record.at("inverse") == b.inverse);
        SemanticResult r{a.epoch, a.request_id, a.sha256, {}, {}};
        for (auto row : b.representative_source_rows) {
            r.semantic_ids.push_back(row % 3);
            r.validity.push_back(1);
        }
        if (control.fault == "digest")
            r.authority_sha256 = digest_bytes("stale");
        if (control.fault == "epoch")
            ++r.epoch;
        if (control.fault == "request")
            ++r.request_id;
        if (control.fault == "partial")
            r.semantic_ids.pop_back();
        if (control.fault == "class")
            r.semantic_ids.front() = 255;
        if (control.fault == "validity")
            r.validity.front() = 0;
        if (control.fault == "context_invalid")
            r.validity.back() = 0;
        if (control.fault == "crash_run")
            throw std::runtime_error("synthetic crash");
        return r;
    }
    void complete(const Json &) override {
        ++control.completed;
        if (control.fault == "completion_crash")
            throw std::runtime_error("synthetic crash");
    }
};
SourceIdentity source(std::string name = "a", std::uint64_t rows = 4) {
    return {digest_bytes("source-" + name), digest_bytes("header-" + name), rows};
}
PreparedBatch batch(SourceIdentity s = source()) {
    PreparedBatch b;
    b.source = s;
    b.cutoff_ns = 100;
    b.logical_batch_sha256 = digest_bytes("logical-batch");
    b.current_rows = {3, 0, 2, 1};
    b.representative_source_rows = {0, 1};
    b.inverse = {1, 0, 1, 1};
    b.coord = {0, 0, 0, 1, 1, 1};
    b.grid_coord = {0, 0, 0, 1, 1, 1};
    b.feat = {0, 0, 0, 0, 1, 1, 1, 1};
    return b;
}
std::unique_ptr<PointSession> session(Control &c, const PackageFixture &p = PackageFixture(),
                                      std::vector<SourceIdentity> sources = {source()}) {
    return std::make_unique<PointSession>(
        p.admit(), "00000000-0000-0000-0000-000000000001",
        [&c] { return std::make_unique<Backend>(c); },
        synrex::point::SessionOptions{true, false, 4, 8, sources});
}
void ready(PointSession &s) {
    const auto d = s.descriptor_sha256();
    s.metadata_ack(Peer::input, d);
    s.metadata_ack(Peer::output, d);
    const auto init = s.initialize().at("initialization_sha256").get<std::string>();
    s.ready_ack(Peer::input, init);
    s.ready_ack(Peer::output, init);
}
void admission() {
    PackageFixture p;
    const auto a = p.admit();
    p.members.at("model_state.bin") = "mutated";
    check(a->members().at("model_state.bin") != "mutated");
    refuses([&] { p.admit(); });
    p = PackageFixture();
    auto bytes = p.manifest.dump();
    refuses([&] { PointArtifact x(bytes, digest_bytes("wrong"), p.members, {}); });
    refuses([&] { PointArtifact x(bytes, digest_bytes(bytes), p.members, {}); });
    p.manifest["unexpected"] = 1;
    refuses([&] { p.admit(); });
    p = PackageFixture();
    p.manifest["members"][0]["path"] = "../weights";
    refuses([&] { p.admit(); });
    p = PackageFixture();
    p.manifest["members"][0]["bytes"] = 67108865;
    refuses([&] { p.admit(); });
    p = PackageFixture();
    p.manifest["members"][1]["role"] = "model_state";
    refuses([&] { p.admit(); });
    p = PackageFixture();
    p.members["extra"] = "extra";
    refuses([&] { p.admit(); });
    p = PackageFixture();
    p.contract.max_representatives = 65537;
    refuses([&] { p.admit(); });
    p = PackageFixture();
    p.contract.execution_mode = "NATIVE";
    refuses([&] { p.admit(); });
    p = PackageFixture();
    p.contract.selected_capability = "implicit";
    refuses([&] { p.admit(); });
    p = PackageFixture();
    p.contract.policy_sha256 = "unbound";
    refuses([&] { p.admit(); });
    p = PackageFixture();
    p.manifest["members"][12]["document_schema"] = "synthetic.wrong.v1";
    refuses([&] { p.admit(); });
    p = PackageFixture();
    bytes = "{\"schema\":0," + p.manifest.dump().substr(1);
    refuses([&] { PointArtifact x(bytes, digest_bytes(bytes), p.members, {}); });
    refuses([&] {
        PointSession x(p.admit(), "00000000-0000-0000-0000-000000000001",
                       [] { return std::unique_ptr<PointInferenceSession>{}; });
    });
}
void handshake() {
    Control c;
    auto s = session(c);
    refuses([&] { s->initialize(); });
    check(c.made == 0 && s->state() == State::failed);
    s = session(c);
    auto d = s->descriptor_sha256();
    auto copy = s->descriptor();
    copy["prepared_wire"]["version"] = 999;
    check(s->descriptor()["prepared_wire"]["version"] == 1);
    check(s->descriptor()["external_raw_point_wire4"] == "NOT_IMPLEMENTED");
    s->metadata_ack(Peer::input, d);
    refuses([&] { s->metadata_ack(Peer::input, d); });
    check(c.made == 0);
    s = session(c);
    d = s->descriptor_sha256();
    s->metadata_ack(Peer::input, d);
    s->metadata_ack(Peer::output, d);
    auto init = s->initialize().at("initialization_sha256").get<std::string>();
    check(c.initialized == 1);
    s->ready_ack(Peer::input, init);
    refuses([&] { s->open_source(source()); });
    c.fault = "pretend_learned";
    s = session(c);
    refuses([&] { ready(*s); });
    c.fault = "crash_init";
    s = session(c);
    refuses([&] { ready(*s); });
    c.fault = "activate_crash";
    s = session(c);
    refuses([&] { ready(*s); });
}
void reconstruction() {
    Control c;
    auto s = session(c);
    ready(*s);
    s->open_source(source());
    auto b = batch();
    const auto output = s->run(b);
    check(output.epoch == 1 && output.request_id == 1 &&
          output.logical_batch_sha256 == b.logical_batch_sha256 &&
          output.authority_sha256.size() == 64);
    check(output.rows == b.current_rows &&
          output.semantic_ids == std::vector<std::uint8_t>({1, 0, 1, 1}) &&
          output.validity == std::vector<std::uint8_t>({1, 1, 1, 1}) &&
          output.las_codes == std::vector<std::uint8_t>({1, 2, 1, 1}));
    bool committed = false;
    const auto receipt = s->complete([&](const auto &rows) {
        committed = true;
        check(rows.at(source().sha256) == std::vector<std::uint8_t>({0, 1, 1, 1}));
        return digest_bytes("synthetic-output");
    });
    check(committed && s->state() == State::complete_pending && c.completed == 1);
    auto h = receipt.at("completion_sha256").get<std::string>();
    s->completion_ack(Peer::input, h);
    check(s->state() == State::complete_pending);
    s->completion_ack(Peer::output, h);
    check(s->state() == State::complete);
    check(c.made == 1 && c.initialized == 1 && c.activated == 1 && c.runs == 1);
    refuses([&] { s->run(b); });
}
void negative_batches() {
    for (const auto &fault :
         {"digest", "epoch", "request", "partial", "class", "validity", "crash_run"}) {
        Control c;
        c.fault = fault;
        auto s = session(c);
        ready(*s);
        s->open_source(source());
        refuses([&] { s->run(batch()); });
        check(s->state() == State::failed && c.retired == 1);
        bool wrote = false;
        refuses([&] {
            s->complete([&](const auto &) {
                wrote = true;
                return digest_bytes("bad");
            });
        });
        check(!wrote);
    }
    for (unsigned which = 0; which < 11; ++which) {
        Control c;
        auto s = session(c);
        ready(*s);
        s->open_source(source());
        auto b = batch();
        if (which == 0)
            b.inverse[0] = 2;
        if (which == 1)
            b.current_rows[0] = 4;
        if (which == 2)
            b.current_rows[0] = b.current_rows[1];
        if (which == 3)
            b.representative_source_rows[1] = b.representative_source_rows[0];
        if (which == 4)
            b.coord[0] = std::numeric_limits<float>::infinity();
        if (which == 5)
            b.coord.pop_back();
        if (which == 6)
            b.cutoff_ns = 9007199254740992ULL;
        if (which == 7)
            b.source.header_sha256 = digest_bytes("foreign-header");
        if (which == 8)
            b.logical_batch_sha256 = "invalid";
        if (which == 9)
            b.representative_source_rows[0] = 4;
        if (which == 10)
            b.grid_coord[0] = -1;
        refuses([&] { s->run(b); });
        check(c.runs == 0 && s->state() == State::failed);
    }
    Control c;
    auto s = session(c);
    ready(*s);
    s->open_source(source());
    s->run(batch());
    refuses([&] { s->run(batch()); });
    check(c.runs == 1);
    c = {};
    c.fault = "context_invalid";
    s = session(c);
    ready(*s);
    s->open_source(source());
    auto b = batch();
    b.current_rows = {0, 2, 3}; // Representative row1 is context-only.
    b.inverse = {0, 0, 0};
    check(s->run(b).semantic_ids == std::vector<std::uint8_t>({0, 0, 0}));
}
void representative_self_mapping() {
    Control c;
    auto s = session(c);
    ready(*s);
    s->open_source(source());
    auto b = batch();
    // Both indices are in range, but the two current representatives select
    // each other. Refuse before backend execution, rather than silently swap.
    b.inverse = {1, 1, 1, 0};
    bool rejected_for_mapping = false;
    try {
        s->run(b);
    } catch (const std::exception &e) {
        rejected_for_mapping =
            std::string(e.what()).find("representative self-map") != std::string::npos;
    }
    check(rejected_for_mapping && c.runs == 0 && c.retired == 1 && s->state() == State::failed);
    s = session(c);
    ready(*s);
    s->open_source(source());
    b = batch();
    // Representative order may differ from source/current order when its
    // inverse is changed consistently. The original-row semantics stay fixed.
    b.representative_source_rows = {1, 0};
    b.inverse = {0, 1, 0, 0};
    const auto output = s->run(b);
    check(output.semantic_ids == std::vector<std::uint8_t>({1, 0, 1, 1}));
}
void reset_and_completion() {
    Control c;
    auto s = session(c);
    ready(*s);
    s->open_source(source());
    refuses([&] { s->open_source(source("b")); });
    check(c.opened == 1);
    s = session(c, PackageFixture(), {source(), source("b")});
    ready(*s);
    s->open_source(source());
    s->run(batch());
    s->open_source(source("b"));
    auto b = batch(source("b"));
    s->run(b);
    auto receipt = s->complete([](const auto &rows) {
        check(rows.size() == 2);
        return digest_bytes("two");
    });
    check(receipt["receipt"]["epoch"] == 2 && receipt["receipt"]["requests"] == 2);
    s = session(c);
    ready(*s);
    s->open_source(source());
    bool committed = false;
    refuses([&] {
        s->complete([&](const auto &) {
            committed = true;
            return digest_bytes("bad");
        });
    });
    check(!committed);
    s = session(c, PackageFixture(), {source("empty", 0)});
    ready(*s);
    s->open_source(source("empty", 0));
    receipt = s->complete([](const auto &rows) {
        check(rows.begin()->second.empty());
        return digest_bytes("empty");
    });
    check(receipt["receipt"]["emitted_rows"] == 0 && receipt["receipt"]["requests"] == 0);
    s = session(c);
    ready(*s);
    s->open_source(source());
    s->run(batch());
    refuses([&] { s->open_source(source()); });
    s = session(c, PackageFixture(), {source(), source("b")});
    ready(*s);
    s->open_source(source());
    s->run(batch());
    refuses([&] {
        s->complete([](const auto &) {
            throw std::runtime_error("must not commit omitted source");
            return digest_bytes("bad");
        });
    });
}
void wire_and_common() {
    namespace w = synrex::point::prepared_wire1;
    const auto deadline = [] { return w::Connection::Clock::now() + std::chrono::seconds(2); };
    PackageFixture p;
    Control c;
    auto artifact = p.admit();
    auto common = synrex::common::Session::from_point(
        artifact, "00000000-0000-0000-0000-000000000001",
        [&c] { return std::make_unique<Backend>(c); },
        synrex::point::SessionOptions{true, false, 4, 8, {source()}});
    auto &s = common.point_authority();
    w::Binding binding(s, *artifact);
    refuses([&] { synrex::generic::Descriptor::parse(s.descriptor().dump()); });
    check(common.backend() == synrex::common::Backend::point);
    refuses([&] { common.run(synrex::generic::Frame{}); });
    check(c.made == 0 && s.state() == State::metadata_pending);
    refuses([&] { common.dense_session(); });
    // Real stream sockets exercise both peers, every admission ACK, typed data,
    // native reconstruction and terminal ACKs. Only a synthetic CPU backend runs.
    int input[2], output[2];
    check(::socketpair(AF_UNIX, SOCK_STREAM, 0, input) == 0);
    check(::socketpair(AF_UNIX, SOCK_STREAM, 0, output) == 0);
    std::string classified_packet;
    {
        w::Connection authority_in(input[0], binding), peer_in(input[1], binding),
                       authority_out(output[0], binding), peer_out(output[1], binding);
        for (auto *pair : {&authority_in, &authority_out})
            pair->send_control(w::Control::descriptor, s.descriptor(), deadline());
        check(peer_in.receive_control(w::Control::descriptor, deadline()) == s.descriptor());
        check(peer_out.receive_control(w::Control::descriptor, deadline()) == s.descriptor());
        peer_in.send_control(w::Control::metadata_ack, s.descriptor_sha256(), deadline());
        s.metadata_ack(Peer::input, authority_in.receive_control(w::Control::metadata_ack, deadline()));
        check(c.made == 0);
        peer_out.send_control(w::Control::metadata_ack, s.descriptor_sha256(), deadline());
        s.metadata_ack(Peer::output, authority_out.receive_control(w::Control::metadata_ack, deadline()));
        const auto init = s.initialize();
        authority_in.send_control(w::Control::initialization, init, deadline());
        authority_out.send_control(w::Control::initialization, init, deadline());
        const auto in_init = peer_in.receive_control(w::Control::initialization, deadline());
        const auto out_init = peer_out.receive_control(w::Control::initialization, deadline());
        peer_in.send_control(w::Control::ready_ack, in_init.at("initialization_sha256"), deadline());
        peer_out.send_control(w::Control::ready_ack, out_init.at("initialization_sha256"), deadline());
        s.ready_ack(Peer::input, authority_in.receive_control(w::Control::ready_ack, deadline()));
        check(c.activated == 0);
        s.ready_ack(Peer::output, authority_out.receive_control(w::Control::ready_ack, deadline()));
        const auto identity_source = source();
        peer_in.send_control(w::Control::source,
            {{"sha256", identity_source.sha256}, {"header_sha256", identity_source.header_sha256},
             {"rows", identity_source.rows}}, deadline());
        const auto source_notice = authority_in.receive_control(w::Control::source, deadline());
        check(source_notice.at("sha256") == identity_source.sha256);
        s.open_source(identity_source);
        auto original = batch();
        original.coord[0] = -0.0F;
        original.coord[1] = std::numeric_limits<float>::denorm_min();
        original.feat[0] = std::numeric_limits<float>::max();
        peer_in.send_batch(original, deadline());
        const auto decoded = authority_in.receive_batch(deadline());
        check(decoded.coord == original.coord && decoded.feat == original.feat &&
              decoded.grid_coord == original.grid_coord && decoded.inverse == original.inverse &&
              decoded.current_rows == original.current_rows &&
              decoded.representative_source_rows == original.representative_source_rows);
        check(w::encode_batch(binding, decoded) == w::encode_batch(binding, original));
        auto classified = std::get<synrex::point::Classification>(common.run(decoded));
        classified_packet = w::encode_classification(binding, classified);
        authority_out.send_classification(classified, deadline());
        const auto observed = peer_out.receive_classification(deadline());
        check(observed.rows == classified.rows && observed.semantic_ids == classified.semantic_ids &&
              observed.authority_sha256 == classified.authority_sha256 && observed.semantic_ids[1] == 0);
        const auto completion = s.complete([](const auto &rows) {
            check(rows.at(source().sha256) == std::vector<std::uint8_t>({0, 1, 1, 1}));
            return digest_bytes("prepared_wire1 durable synthetic output");
        });
        authority_in.send_control(w::Control::completion, completion, deadline());
        authority_out.send_control(w::Control::completion, completion, deadline());
        const auto in_complete = peer_in.receive_control(w::Control::completion, deadline());
        const auto out_complete = peer_out.receive_control(w::Control::completion, deadline());
        peer_in.send_control(w::Control::completion_ack, in_complete.at("completion_sha256"), deadline());
        peer_out.send_control(w::Control::completion_ack, out_complete.at("completion_sha256"), deadline());
        s.completion_ack(Peer::input, authority_in.receive_control(w::Control::completion_ack, deadline()));
        s.completion_ack(Peer::output, authority_out.receive_control(w::Control::completion_ack, deadline()));
        check(s.state() == State::complete && c.made == 1 && c.runs == 1);
    }
    for (const auto fd : {input[0], input[1], output[0], output[1]}) ::close(fd);
    for (std::size_t i = 0; i < classified_packet.size(); ++i)
        refuses([&] { w::decode_classification(binding, std::string_view(classified_packet).substr(0, i)); });
    auto corrupt_classification = classified_packet;
    corrupt_classification.back() ^= 1;
    refuses([&] { w::decode_classification(binding, corrupt_classification); });
    corrupt_classification = classified_packet;
    corrupt_classification[corrupt_classification.size() - 8] = 0;
    refuses([&] { w::decode_classification(binding, corrupt_classification); });
    const auto encoded = w::encode_batch(binding, batch());
    check(encoded.substr(0, 4) == "PWB1" && encoded.at(4) == 1);
    // The frozen raw-point external wire4 has a different 16-byte header and
    // raw source records. Never admit it as this prepared-array channel.
    std::string external_raw_point_wire4 = "SRX4";
    external_raw_point_wire4.append({4, 0, 5, 0, 0, 0, 0, 0, 0, 0, 0, 0});
    refuses([&] { w::decode_batch(binding, external_raw_point_wire4); });
    auto falsely_versioned = encoded;
    falsely_versioned.replace(0, 4, "SRX4");
    falsely_versioned[4] = 4;
    refuses([&] { w::decode_batch(binding, falsely_versioned); });
    check(::socketpair(AF_UNIX, SOCK_STREAM, 0, input) == 0);
    {
        w::Connection receiver(input[0], binding);
        auto sender = std::async(std::launch::async, [&] {
            check(::send(input[1], encoded.data(), 6, MSG_NOSIGNAL) == 6);
            std::this_thread::sleep_for(std::chrono::milliseconds(5));
            check(::send(input[1], encoded.data() + 6, encoded.size() - 6, MSG_NOSIGNAL) ==
                  static_cast<ssize_t>(encoded.size() - 6));
        });
        check(w::encode_batch(binding, receiver.receive_batch(deadline())) == encoded);
        sender.get();
    }
    ::close(input[0]); ::close(input[1]);
    for (std::size_t i = 0; i < encoded.size(); ++i)
        refuses([&] { w::decode_batch(binding, std::string_view(encoded).substr(0, i)); });
    for (const auto index : {0U, 4U, 5U, 6U, 8U, 12U, 112U}) {
        auto bad = encoded; bad[index] ^= 1;
        refuses([&] { w::decode_batch(binding, bad); });
    }
    auto bad = encoded + "x";
    refuses([&] { w::decode_batch(binding, bad); });
    bad = encoded;
    // Current count sits after fixed identity/source/cutoff/logical digest.
    std::fill(bad.begin() + 320, bad.begin() + 328, '\xff');
    refuses([&] { w::decode_batch(binding, bad); });
    auto invalid = batch(); invalid.inverse[0] = 99;
    refuses([&] { w::encode_batch(binding, invalid); });
    invalid = batch(); invalid.feat[0] = std::numeric_limits<float>::infinity();
    refuses([&] { w::encode_batch(binding, invalid); });
    invalid = batch(); invalid.source = source("unknown");
    refuses([&] { w::encode_batch(binding, invalid); });
    refuses([&] { w::encode_control(binding, w::Control::metadata_ack, digest_bytes("wrong")); });
    auto control = w::encode_control(binding, w::Control::descriptor, s.descriptor());
    control[3] = '3';
    refuses([&] { w::decode_control(binding, w::Control::descriptor, control); });
    // An oversized advertised length rejects from its 12-byte header alone.
    check(::socketpair(AF_UNIX, SOCK_STREAM, 0, input) == 0);
    {
        w::Connection receiver(input[0], binding);
        auto header = encoded.substr(0, 12);
        std::fill(header.begin() + 8, header.end(), '\xff');
        check(::send(input[1], header.data(), header.size(), MSG_NOSIGNAL) == 12);
        refuses([&] { receiver.receive_batch(deadline()); });
        check(receiver.failed());
        refuses([&] { receiver.receive_batch(deadline()); });
    }
    ::close(input[0]); ::close(input[1]);
    check(::socketpair(AF_UNIX, SOCK_STREAM, 0, input) == 0);
    {
        w::Connection receiver(input[0], binding);
        refuses([&] { receiver.receive_batch(w::Connection::Clock::now() + std::chrono::milliseconds(10)); });
        check(receiver.failed());
    }
    ::close(input[0]); ::close(input[1]);
    check(::socketpair(AF_UNIX, SOCK_STREAM, 0, input) == 0);
    {
        w::Connection receiver(input[0], binding);
        ::shutdown(input[1], SHUT_WR);
        refuses([&] { receiver.receive_batch(deadline()); });
        check(receiver.failed());
    }
    ::close(input[0]); ::close(input[1]);
}

} // namespace
int main(int argc, char **argv) {
    try {
        check(argc == 2);
        const fs::path legacy =
            fs::path(argv[1]) / ("point-v4-legacy-refusal-" + std::to_string(::getpid()));
        check(fs::create_directory(legacy));
        write_new(legacy / "manifest.json", PackageFixture().manifest.dump());
        refuses([&] { Package p(legacy, PackageUse::file); });
        refuses([&] { Package p(legacy, PackageUse::legacy_stream); });
        check(fs::remove(legacy / "manifest.json") && fs::remove(legacy));
        admission();
        handshake();
        reconstruction();
        negative_batches();
        representative_self_mapping();
        reset_and_completion();
        wire_and_common();
        std::cout
            << "PASS synthetic point artifact/session authority; no learned/GPU qualification\n";
    } catch (const std::exception &e) {
        std::cerr << e.what() << '\n';
        return 1;
    }
}
