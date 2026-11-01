// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Tuomas Pietilä
// Independent native CPU controls; synthetic ONNX bytes are never executed.
#include "synrex/generic_startup.hpp"
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <type_traits>
#include <unistd.h>

namespace {
using namespace synrex;
using namespace synrex::generic;
// Integration-only provenance compile checks. Access violations substitute to false.
template <class T, class = void> struct HasPublicRetainedPackage : std::false_type {};
template <class T>
struct HasPublicRetainedPackage<T, std::void_t<decltype(std::declval<T &>().package_)>>
    : std::true_type {};
template <class T, class = void> struct HasPublicContract : std::false_type {};
template <class T>
struct HasPublicContract<T, std::void_t<decltype(std::declval<T &>().contract)>> : std::true_type {
};
// End integration-only provenance compile checks.
unsigned checks{}, rejections{}, target_queries{};
const std::string session(32, 'a');
void check(bool ok, const std::string &message) {
    ++checks;
    if (!ok)
        throw std::runtime_error(message);
}
template <class F> void rejects(F function, const std::string &diagnostic = "") {
    try {
        function();
    } catch (const std::exception &error) {
        check(diagnostic.empty() || std::string(error.what()).find(diagnostic) != std::string::npos,
              "unexpected rejection: " + std::string(error.what()) + "; expected: " + diagnostic);
        ++rejections;
        return;
    }
    throw std::runtime_error("invalid control unexpectedly accepted: " + diagnostic);
}
void write(const fs::path &path, const std::string &bytes) {
    std::ofstream stream(path, std::ios::binary);
    stream << bytes;
    stream.close();
    check(bool(stream), "fixture write");
}
struct Temporary {
    fs::path path;
    explicit Temporary(const fs::path &parent) {
        auto pattern = (parent / "generic-fixture-XXXXXX").string();
        const auto created = ::mkdtemp(pattern.data());
        check(created != nullptr, "temporary fixture root");
        path = created;
    }
    ~Temporary() {
        std::error_code error;
        fs::remove_all(path, error);
    }
};
Json taxonomy_json(std::size_t count, bool legacy_projection = false) {
    Json classes = Json::array();
    const unsigned legacy[] = {1, 2, 200, 6, 9, 17, 201};
    for (std::size_t i = 0; i < count; ++i)
        classes.push_back({{"semantic_id", i},
                           {"key", "private_surface_" + std::to_string(i)},
                           {"name", "Not FRACTAL meaning " + std::to_string(i)},
                           {"las_code", legacy_projection ? legacy[i] : 42}});
    return {
        {"schema", "synrex.taxonomy.v1"},
        {"taxonomy_id", "example.reused.v1"},
        {"namespace", "example.cpu"},
        {"classes", classes},
        {"provenance", {{"source", "CPU fixture"}, {"author", "SynRex tests"}, {"license", "MIT"}}},
        {"unknown_semantic_id", nullptr}};
}
fs::path write_package(const fs::path &parent, const Json &base, const std::string &tax_bytes,
                       const std::string &schema = "synrex.srxmodel.v3") {
    const auto directory = parent / ("package-" + std::to_string(checks));
    fs::create_directory(directory);
    auto contract = base;
    const Taxonomy taxonomy(tax_bytes);
    Json projected = Json::array();
    for (const auto &entry : taxonomy.classes())
        projected.push_back(entry.las_code);
    contract["las_classes"] = projected;
    contract["output"]["shape"][1] = taxonomy.classes().size();
    write(directory / "model.onnx", "CPU opaque artifact: NOT A VALID ONNX, NO EXECUTION");
    write(directory / "MODEL_LICENSE.txt", "MIT CPU fixture only");
    Json artifacts{
        {"onnx", {{"path", "model.onnx"}, {"sha256", sha256(directory / "model.onnx")}}},
        {"license",
         {{"path", "MODEL_LICENSE.txt"}, {"sha256", sha256(directory / "MODEL_LICENSE.txt")}}}};
    if (schema == "synrex.srxmodel.v3") {
        write(directory / "taxonomy.json", tax_bytes);
        artifacts["taxonomy"] = {{"path", "taxonomy.json"}, {"sha256", taxonomy.sha256()}};
    }
    write(directory / "manifest.json", Json{{"schema", schema},
                                            {"contract", contract},
                                            {"artifacts", artifacts},
                                            {"target", nullptr}}
                                               .dump() +
                                           "\n");
    return directory;
}
CpuAdmittedPackage make_package(const fs::path &parent, const Json &base, const std::string &bytes,
                                const std::string &schema = "synrex.srxmodel.v3") {
    return CpuAdmittedPackage(write_package(parent, base, bytes, schema));
}
struct TestStartup {
    Startup::Time now{};
    Startup startup;
    explicit TestStartup(const Descriptor &descriptor)
        : startup(descriptor, [this] { return now; }) {}
    void at(std::uint64_t milliseconds) {
        now = Startup::Time{} + std::chrono::milliseconds(milliseconds);
    }
};
void event(Startup &startup, const Descriptor &descriptor, unsigned index) {
    switch (index) {
    case 0:
        startup.descriptor_fragment(descriptor.bytes(), true);
        break;
    case 1:
        startup.metadata_ack(descriptor.control("metadata_ack"));
        break;
    case 2:
        startup.begin_initialization();
        break;
    case 3:
        startup.initialization_complete();
        break;
    case 4:
        startup.execution_ready_ack(descriptor.control("execution_ready_ack"));
        break;
    case 5:
        startup.accept_frame_identity(descriptor.sha256(), session, 1);
        break;
    default:
        throw std::runtime_error("invalid test event");
    }
}
void advance(Startup &startup, const Descriptor &descriptor, unsigned stage) {
    for (unsigned index = 0; index < stage; ++index)
        event(startup, descriptor, index);
}
} // namespace
namespace synrex {
Json target_info() {
    ++target_queries;
    throw std::runtime_error("UNREACHABLE GPU TARGET SENTINEL: no GPU linked or permitted");
}
} // namespace synrex
int main(int argc, char **argv) {
    try {
        check(argc == 3, "requires source fixture and existing scratch parent");
        const auto base = read_json(argv[1]);
        Temporary temporary(argv[2]);
        static_assert(!std::is_constructible_v<CpuAdmittedPackage, const Package &>);
        static_assert(!std::is_constructible_v<CpuAdmittedPackage, bool>);
        static_assert(!std::is_copy_constructible_v<CpuAdmittedPackage>);
        // Integration-only provenance assertions; runtime controls remain unchanged.
        static_assert(!std::is_constructible_v<CpuAdmittedPackage, Package &>);
        static_assert(!std::is_convertible_v<Package &, CpuAdmittedPackage>);
        static_assert(!std::is_convertible_v<CpuAdmittedPackage &, Package &>);
        static_assert(!std::is_constructible_v<CpuAdmittedPackage, const Package &, bool>);
        static_assert(!HasPublicRetainedPackage<CpuAdmittedPackage>::value);
        static_assert(!HasPublicContract<CpuAdmittedPackage>::value);
        // End integration-only provenance assertions.
        static_assert(!std::is_invocable_v<decltype(&Descriptor::from_package), const Package &,
                                           const Limits &, const std::string &, std::uint64_t>);
        const auto package_path =
            write_package(temporary.path, base, taxonomy_json(4).dump() + "\n");
        const CpuAdmittedPackage package(package_path);
        const Package independent_snapshot(package_path);
        const Limits limits{};
        const auto descriptor = Descriptor::from_package(package, limits, session, 1);
        check(descriptor.taxonomy().raw_bytes() == independent_snapshot.taxonomy->raw_bytes(),
              "exact taxonomy bytes");
        check(descriptor.spec().at("manifest_sha256") == independent_snapshot.manifest_sha256,
              "captured manifest hash");
        check(descriptor.spec().at("text_stream_schema").is_null(), "text schema unsupported");
        check(descriptor.spec().at("package_envelope") !=
                  descriptor.spec().at("binary_wire_schema"),
              "separate envelope and wire identities");
        check(Descriptor::parse(descriptor.bytes()).sha256() == descriptor.sha256(),
              "descriptor roundtrip");
        const auto peer_descriptor = Descriptor::parse(descriptor.bytes());
        check(peer_descriptor.control("metadata_ack") == descriptor.control("metadata_ack"),
              "peer can acknowledge metadata before any frame or GPU state");
        rejects([&] { Startup trust_peer_only(peer_descriptor); }, "package-bound");
        for (const auto count : {2u, 4u, 7u, 256u}) {
            const auto p =
                make_package(temporary.path, base, taxonomy_json(count, count == 7).dump());
            const auto d = Descriptor::from_package(p, limits, session, UINT64_MAX);
            check(d.taxonomy().classes().size() == count, "generic K preserved");
            check(d.taxonomy().classes().back().semantic_id == count - 1,
                  "semantic ID255 preserved");
            check(d.taxonomy().classes().front().name.find("Not FRACTAL") == 0,
                  "meanings not inferred");
        }
        auto changed = taxonomy_json(4);
        changed["classes"][0]["name"] = "Changed meaning with same ID/K/projection";
        const auto alternate = Descriptor::from_package(
            make_package(temporary.path, base, changed.dump()), limits, session, 1);
        check(descriptor.spec().at("class_count") == alternate.spec().at("class_count") &&
                  descriptor.spec().at("las_projection") == alternate.spec().at("las_projection") &&
                  descriptor.taxonomy().taxonomy_id() == alternate.taxonomy().taxonomy_id(),
              "same weak identities");
        check(descriptor.sha256() != alternate.sha256() &&
                  descriptor.taxonomy().sha256() != alternate.taxonomy().sha256(),
              "changed meaning is bound by full taxonomy digest");
        TestStartup different(descriptor);
        rejects([&] { different.startup.descriptor_fragment(alternate.bytes(), true); },
                "identity mismatch");
        check(different.startup.state() == State::failed, "meaning mismatch poisons");
        const auto seven_package =
            make_package(temporary.path, base, taxonomy_json(7, true).dump());
        const auto seven = Descriptor::from_package(seven_package, limits, session, 1);
        auto changed_seven = taxonomy_json(7, true);
        changed_seven["classes"][0]["name"] =
            "Reused ID, K7 and exact legacy projection, changed meaning";
        const auto seven_alternate = Descriptor::from_package(
            make_package(temporary.path, base, changed_seven.dump()), limits, session, 1);
        check(seven.spec().at("las_projection") == Json::array({1, 2, 200, 6, 9, 17, 201}),
              "non-FRACTAL seven preserves exact legacy projection");
        check(seven.taxonomy().taxonomy_id() == seven_alternate.taxonomy().taxonomy_id(),
              "reused K7 ID");
        TestStartup seven_mismatch(seven);
        rejects([&] { seven_mismatch.startup.descriptor_fragment(seven_alternate.bytes(), true); },
                "identity mismatch");
        auto reordered = taxonomy_json(4);
        std::swap(reordered["classes"][0]["key"], reordered["classes"][1]["key"]);
        std::swap(reordered["classes"][0]["name"], reordered["classes"][1]["name"]);
        const auto reordered_descriptor = Descriptor::from_package(
            make_package(temporary.path, base, reordered.dump()), limits, session, 1);
        TestStartup order_mismatch(descriptor);
        rejects(
            [&] { order_mismatch.startup.descriptor_fragment(reordered_descriptor.bytes(), true); },
            "identity mismatch");
        // Captured identity survives path-only mutation; no manifest reopen.
        write(package_path / "manifest.json", independent_snapshot.manifest_bytes + " ");
        check(Descriptor::from_package(package, limits, session, 1).sha256() == descriptor.sha256(),
              "descriptor uses captured manifest bytes");
        // Mutating a separate public Package cannot enter the wrapper-only API.
        // Static and negative-compilation controls prove the old call is unavailable.
        auto mutable_public_package = independent_snapshot;
        mutable_public_package.contract.las_classes[0] = 255;
        mutable_public_package.manifest["unadmitted_extra"] = true;
        mutable_public_package.taxonomy.emplace(changed.dump());
        check(Descriptor::from_package(package, limits, session, 1).sha256() == descriptor.sha256(),
              "private captured raw and derived fields unaffected by public Package mutation");
        for (const auto fault :
             {"unknown_envelope_field", "unadmitted_model_digest", "projection_conflict"}) {
            const auto malformed_path =
                write_package(temporary.path, base, taxonomy_json(4).dump());
            auto altered = read_json(malformed_path / "manifest.json");
            const std::string kind(fault);
            if (kind == "unknown_envelope_field")
                altered["extra"] = true;
            else if (kind == "unadmitted_model_digest")
                altered["artifacts"]["onnx"]["sha256"] = std::string(64, '0');
            else
                altered["contract"]["las_classes"][0] = 41;
            write(malformed_path / "manifest.json", altered.dump());
            const auto diagnostic =
                kind == "unknown_envelope_field"    ? "unknown/missing contract fields"
                : kind == "unadmitted_model_digest" ? "SHA256 mismatch: model.onnx"
                                                    : "taxonomy LAS projection mismatch";
            rejects([&] { const CpuAdmittedPackage must_reject(malformed_path); }, diagnostic);
        }
        const auto legacy =
            make_package(temporary.path, base, taxonomy_json(4).dump(), "synrex.srxmodel.v2");
        rejects([&] { Descriptor::from_package(legacy, limits, session, 1); }, "explicit v3");
        const auto negotiated = Limits::negotiate({100, 1000, 30000}, {200, 500, 10000});
        check(negotiated.max_frame_points == 100 && negotiated.max_frame_bytes == 500 &&
                  negotiated.startup_timeout_ms == 10000,
              "limits negotiated down in each dimension");
        for (const auto bad_limits :
             {Limits{0, 1, 1}, Limits{20000001, 1, 1}, Limits{1, 0, 1}, Limits{1, 16777217, 1},
              Limits{1, 1, 0}, Limits{1, 1, 120001}})
            rejects([&] { Descriptor::from_package(package, bad_limits, session, 1); }, "limit");
        rejects([&] { Descriptor::from_package(package, {1000001, 1000, 1}, session, 1); },
                "admitted Contract");
        rejects([&] { Descriptor::from_package(package, limits, "bad-session", 1); }, "session ID");
        rejects([&] { Descriptor::from_package(package, limits, session, 0); }, "integer range");
        // Independently mutate every separate identity/control field.
        for (const auto &key : {"schema", "binary_wire_schema", "package_envelope", "dense_adapter",
                                "context_policy_id", "context_policy_sha256", "taxonomy_sha256",
                                "las_projection_sha256", "manifest_sha256", "session_id"}) {
            auto value = descriptor.spec();
            value[key] = "unsupported";
            rejects([&] { Descriptor::parse(value.dump()); });
        }
        for (const auto &value : {Json(2), Json(4), Json(3.0), Json(true), Json(-1), Json("3")}) {
            auto bad = descriptor.spec();
            bad["binary_wire_version"] = value;
            rejects([&] { Descriptor::parse(bad.dump()); }, "integer");
        }
        for (const auto &value : {Json(1), Json(257), Json(4.0), Json(true), Json(-1), Json("4")}) {
            auto bad = descriptor.spec();
            bad["class_count"] = value;
            rejects([&] { Descriptor::parse(bad.dump()); });
        }
        for (const auto &value :
             {Json::array(), Json::array({42, 42, 42}), Json::array({42, 42, 42, 41}),
              Json::array({42, 42, 42, 256}), Json::array({42, 42, 42, 42.0}), Json(nullptr)}) {
            auto bad = descriptor.spec();
            bad["las_projection"] = value;
            rejects([&] { Descriptor::parse(bad.dump()); });
        }
        auto bad = descriptor.spec();
        bad["text_stream_schema"] = "synrex.text.v3";
        rejects([&] { Descriptor::parse(bad.dump()); }, "text stream unsupported");
        bad = descriptor.spec();
        bad["unknown"] = 1;
        rejects([&] { Descriptor::parse(bad.dump()); }, "unknown fields");
        bad = descriptor.spec();
        bad.erase("class_count");
        rejects([&] { Descriptor::parse(bad.dump()); });
        bad = descriptor.spec();
        bad["taxonomy_utf8"] = changed.dump();
        rejects([&] { Descriptor::parse(bad.dump()); }, "digest mismatch");
        bad = descriptor.spec();
        bad["taxonomy_utf8"] = independent_snapshot.taxonomy->raw_bytes() + "\n";
        rejects([&] { Descriptor::parse(bad.dump()); }, "digest mismatch");
        for (const auto &key : {"max_metadata_bytes", "metadata_timeout_ms", "max_frame_points",
                                "max_frame_bytes", "startup_timeout_ms"}) {
            bad = descriptor.spec();
            bad["limits"][key] = 0;
            rejects([&] { Descriptor::parse(bad.dump()); }, "integer range");
        }
        auto malformed_taxonomy = taxonomy_json(4);
        malformed_taxonomy["classes"][0]["semantic_id"] = 255;
        bad = descriptor.spec();
        bad["taxonomy_utf8"] = malformed_taxonomy.dump();
        rejects([&] { Descriptor::parse(bad.dump()); }, "ordered");
        malformed_taxonomy = taxonomy_json(4);
        malformed_taxonomy["classes"][1]["key"] = "private_surface_0";
        bad["taxonomy_utf8"] = malformed_taxonomy.dump();
        rejects([&] { Descriptor::parse(bad.dump()); }, "duplicate class key");
        for (const auto &bytes : {std::string("{}"), std::string("null"), std::string("[]"),
                                  descriptor.bytes() + "x", std::string("{")})
            rejects([&] { Descriptor::parse(bytes); });
        rejects([&] { Descriptor::parse("{\"class_count\":4," + descriptor.bytes().substr(1)); },
                "duplicate");
        rejects(
            [&] { Descriptor::parse("{\"class_co\\u0075nt\":4," + descriptor.bytes().substr(1)); },
            "duplicate");
        rejects(
            [&] {
                Descriptor::parse("{\"x\":" + std::string(9, '[') + "0" + std::string(9, ']') +
                                  "}");
            },
            "nesting");
        rejects([&] { Descriptor::parse(std::string(65537, ' ')); }, "byte limit");
        std::string invalid_utf8 = descriptor.bytes();
        invalid_utf8.insert(1, "\xFF");
        rejects([&] { Descriptor::parse(invalid_utf8); }, "malformed");
        // Legal 64KiB taxonomy and a smaller but escape-heavy taxonomy can fail the independent
        // descriptor bound.
        auto padded = taxonomy_json(4).dump();
        padded.append(65536 - padded.size(), ' ');
        check(Taxonomy(padded).raw_bytes().size() == 65536, "taxonomy maximum itself valid");
        const auto padded_package = make_package(temporary.path, base, padded);
        rejects([&] { Descriptor::from_package(padded_package, limits, session, 1); },
                "byte limit");
        auto exact_fit_taxonomy = independent_snapshot.taxonomy->raw_bytes();
        exact_fit_taxonomy.append(65536 - descriptor.bytes().size(), ' ');
        const auto exact_fit = Descriptor::from_package(
            make_package(temporary.path, base, exact_fit_taxonomy), limits, session, 1);
        check(exact_fit.bytes().size() == 65536, "exact descriptor byte ceiling accepted");
        rejects([&] { Descriptor::parse(exact_fit.bytes() + " "); }, "byte limit");
        auto escaped_json = taxonomy_json(128);
        for (auto &entry : escaped_json["classes"])
            entry["name"] = std::string(160, '"');
        const auto escaped = escaped_json.dump();
        check(escaped.size() < 65536, "escape-heavy taxonomy below artifact byte bound");
        const auto escaped_package = make_package(temporary.path, base, escaped);
        rejects([&] { Descriptor::from_package(escaped_package, limits, session, 1); },
                "byte limit");
        // CPU package validation precedes any startup object; metadata ACK is independent of
        // frames/GPU.
        TestStartup positive(descriptor);
        check(!positive.startup.gpu_permission(), "no GPU permission before descriptor");
        positive.startup.descriptor_fragment(descriptor.bytes().substr(0, 100), false);
        positive.at(4999);
        positive.startup.descriptor_fragment(descriptor.bytes().substr(100), true);
        check(!positive.startup.gpu_permission(), "no GPU permission before metadata ACK");
        event(positive.startup, descriptor, 1);
        check(positive.startup.gpu_permission() && positive.startup.state() == State::gpu_permitted,
              "metadata ACK before GPU permission or first frame");
        positive.at(5000);
        event(positive.startup, descriptor, 2);
        positive.at(34998);
        event(positive.startup, descriptor, 3);
        event(positive.startup, descriptor, 4);
        positive.at(999999);
        event(positive.startup, descriptor, 5);
        event(positive.startup, descriptor, 5);
        check(positive.startup.state() == State::ready, "startup deadline ceases once ready");
        // Every invalid ordering, including repeated transition/control calls, poisons permanently.
        for (unsigned stage = 0; stage <= 5; ++stage) {
            for (unsigned index = 0; index <= 5; ++index) {
                if (index == stage)
                    continue;
                TestStartup item(descriptor);
                advance(item.startup, descriptor, stage);
                rejects([&] { event(item.startup, descriptor, index); }, "out of order");
                check(item.startup.state() == State::failed, "out-of-order poison");
                rejects([&] { item.startup.poll(); }, "permanently failed");
            }
        }
        for (const bool ready : {false, true}) {
            const auto kind = ready ? "execution_ready_ack" : "metadata_ack";
            auto control = Json::parse(descriptor.control(kind));
            std::vector<std::string> controls{"",
                                              "{}",
                                              "null",
                                              "[]",
                                              "{",
                                              descriptor.control(kind) + "x",
                                              std::string(1025, ' ')};
            for (const auto &key :
                 {"schema", "kind", "descriptor_sha256", "session_id", "map_epoch"}) {
                auto mutated = control;
                mutated[key] = "wrong";
                controls.push_back(mutated.dump());
                mutated = control;
                mutated.erase(key);
                controls.push_back(mutated.dump());
            }
            for (const auto &epoch :
                 {Json(0), Json(-1), Json(true), Json(1.0), Json(2), Json("1")}) {
                auto mutated = control;
                mutated["map_epoch"] = epoch;
                controls.push_back(mutated.dump());
            }
            auto unknown = control;
            unknown["x"] = true;
            controls.push_back(unknown.dump());
            controls.push_back("{\"kind\":\"metadata_ack\"," + control.dump().substr(1));
            controls.push_back("{\"ki\\u006ed\":\"metadata_ack\"," + control.dump().substr(1));
            controls.push_back("{\"x\":" + std::string(9, '[') + "0" + std::string(9, ']') + "}");
            controls.push_back(std::string("{\"x\":\"") + "\xFF" + "\"}");
            for (const auto &bytes : controls) {
                TestStartup item(descriptor);
                advance(item.startup, descriptor, ready ? 4 : 1);
                rejects([&] {
                    if (ready)
                        item.startup.execution_ready_ack(bytes);
                    else
                        item.startup.metadata_ack(bytes);
                });
                check(item.startup.state() == State::failed, "bad control poison");
                rejects([&] { item.startup.gpu_permission(); }, "permanently failed");
            }
        }
        // Exact absolute deadlines: fragments/polls cannot extend metadata or initialization.
        for (unsigned stage : {0u, 1u}) {
            TestStartup item(descriptor);
            advance(item.startup, descriptor, stage);
            item.at(4999);
            item.startup.poll();
            if (stage == 0)
                item.startup.descriptor_fragment(" ", false);
            item.at(5000);
            rejects([&] { item.startup.poll(); }, "metadata deadline");
            check(item.startup.state() == State::failed, "metadata expiry poison");
        }
        for (unsigned stage : {2u, 3u, 4u}) {
            TestStartup item(descriptor);
            item.at(4999);
            advance(item.startup, descriptor, stage);
            item.at(34998);
            item.startup.poll();
            item.startup.poll();
            item.at(34999);
            rejects([&] { item.startup.poll(); }, "initialization deadline");
            check(item.startup.state() == State::failed, "initialization expiry poison");
        }
        TestStartup late_ack(descriptor);
        advance(late_ack.startup, descriptor, 1);
        late_ack.at(5000);
        rejects([&] { event(late_ack.startup, descriptor, 1); }, "metadata deadline");
        TestStartup rewind(descriptor);
        rewind.at(1);
        rewind.startup.poll();
        rewind.at(0);
        rejects([&] { rewind.startup.poll(); }, "moved backwards");
        TestStartup fragments(descriptor);
        fragments.startup.descriptor_fragment(std::string(65536, ' '), false);
        rejects([&] { fragments.startup.descriptor_fragment(" ", false); }, "fragments byte limit");
        TestStartup explicit_failure(descriptor);
        explicit_failure.startup.fail("peer rejected metadata");
        explicit_failure.startup.fail("must retain first failure");
        check(explicit_failure.startup.failure() == "peer rejected metadata",
              "first error retained");
        rejects([&] { explicit_failure.startup.poll(); }, "permanently failed");
        rejects([&] { Startup overflow(descriptor, [] { return Startup::Time::max(); }); },
                "overflow");
        auto near_maximum = Startup::Time::max() - std::chrono::milliseconds(6000);
        Startup initialization_overflow(descriptor, [&] { return near_maximum; });
        event(initialization_overflow, descriptor, 0);
        rejects([&] { event(initialization_overflow, descriptor, 1); }, "overflow");
        check(initialization_overflow.state() == State::failed,
              "startup deadline overflow poisons");
        rejects([&] { Startup no_clock(descriptor, Startup::Now{}); }, "clock callback");
        for (unsigned identity = 0; identity < 3; ++identity) {
            TestStartup item(descriptor);
            advance(item.startup, descriptor, 5);
            rejects(
                [&] {
                    item.startup.accept_frame_identity(
                        identity == 0 ? std::string(64, '0') : descriptor.sha256(),
                        identity == 1 ? std::string(32, 'b') : session, identity == 2 ? 2 : 1);
                },
                "frame identity");
            check(item.startup.state() == State::failed, "frame identity poison");
        }
        check(target_queries == 0, "no GPU target sentinel called");
        std::cout << Json{{"status", "PASS_CPU_DESCRIPTOR_STARTUP_PREPARATION_ONLY"},
                          {"checks", checks},
                          {"expected_rejections", rejections},
                          {"gpu_target_calls", target_queries},
                          {"inference_calls", 0},
                          {"laz_calls", 0},
                          {"control_transport_exercised", false},
                          {"generic_gpu_execution_qualified", false}}
                         .dump()
                  << '\n';
        return 0;
    } catch (const std::exception &error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
