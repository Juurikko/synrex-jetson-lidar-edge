// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Tuomas Pietilä
// CPU-only opaque model bytes: no valid ONNX/engine or GPU execution is claimed.
#include "synrex/engine_artifact.hpp"
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <sys/stat.h>
#include <type_traits>
#include <unistd.h>

namespace {
using namespace synrex;
unsigned checks{}, rejections{}, target_queries{};
void check(bool condition, const std::string &message) {
    ++checks;
    if (!condition)
        throw std::runtime_error(message);
}
template <class F> void rejects(F function, const std::string &diagnostic) {
    try {
        function();
    } catch (const std::exception &error) {
        check(std::string(error.what()).find(diagnostic) != std::string::npos,
              "unexpected rejection: " + std::string(error.what()));
        ++rejections;
        return;
    }
    throw std::runtime_error("invalid capture accepted: " + diagnostic);
}
void write(const fs::path &path, const std::string &bytes) {
    std::ofstream out(path, std::ios::binary);
    out << bytes;
    out.close();
    check(bool(out), "fixture write");
}
struct Temporary {
    fs::path path;
    explicit Temporary(const fs::path &parent) {
        auto pattern = (parent / "engine-capture-XXXXXX").string();
        const auto created = ::mkdtemp(pattern.data());
        check(created != nullptr, "temporary root");
        path = created;
    }
    ~Temporary() {
        std::error_code error;
        fs::remove_all(path, error);
    }
};
const std::string model = "opaque model bytes captured BEFORE any GPU access";
Json taxonomy_json() {
    return {
        {"schema", "synrex.taxonomy.v1"},
        {"taxonomy_id", "example.reused.v1"},
        {"namespace", "example.cpu"},
        {"classes",
         Json::array({{{"semantic_id", 0}, {"key", "a"}, {"name", "A"}, {"las_code", 2}},
                      {{"semantic_id", 1}, {"key", "b"}, {"name", "B"}, {"las_code", 5}},
                      {{"semantic_id", 2}, {"key", "c"}, {"name", "C"}, {"las_code", 6}}})},
        {"provenance", {{"source", "CPU fixture"}, {"author", "SynRex tests"}, {"license", "MIT"}}},
        {"unknown_semantic_id", nullptr}};
}
fs::path fixture(const fs::path &parent, const Json &contract, bool engine = true, bool v3 = true) {
    const auto path = parent / ("package-" + std::to_string(checks));
    fs::create_directory(path);
    const std::string kind = engine ? "engine" : "onnx";
    const auto basename = engine ? "model.engine" : "model.onnx";
    write(path / basename, model);
    write(path / "MODEL_LICENSE.txt", "MIT CPU fixture only");
    Json artifacts{
        {kind, {{"path", basename}, {"sha256", sha256(path / basename)}}},
        {"license",
         {{"path", "MODEL_LICENSE.txt"}, {"sha256", sha256(path / "MODEL_LICENSE.txt")}}}};
    if (v3) {
        write(path / "taxonomy.json", taxonomy_json().dump() + "\n");
        artifacts["taxonomy"] = {{"path", "taxonomy.json"},
                                 {"sha256", sha256(path / "taxonomy.json")}};
    }
    const Json target = engine ? Json{{"architecture", "CPU-test-only"},
                                      {"gpu_name", "NO GPU"},
                                      {"sm", 89},
                                      {"tensorrt", "10.test"},
                                      {"cuda_runtime", 12010}}
                               : Json(nullptr);
    write(path / "manifest.json", Json{{"schema", v3 ? "synrex.srxmodel.v3" : "synrex.srxmodel.v2"},
                                       {"contract", contract},
                                       {"artifacts", artifacts},
                                       {"target", target}}
                                          .dump() +
                                      "\n");
    return path;
}
} // namespace
namespace synrex {
Json target_info() {
    ++target_queries;
    throw std::runtime_error("CPU capture called target_info");
}
} // namespace synrex
int main(int argc, char **argv) {
    try {
        static_assert(!std::is_constructible_v<EngineArtifact, const Package &>);
        static_assert(!std::is_copy_constructible_v<EngineArtifact>);
        static_assert(std::is_same_v<decltype(std::declval<EngineArtifact &>().contract()),
                                     const Contract &>);
        static_assert(std::is_same_v<decltype(std::declval<EngineArtifact &>().model_bytes()),
                                     const std::string &>);
        check(argc == 3, "usage: engine_artifact SPEC TEMP_PARENT");
        Temporary temporary(argv[2]);
        const auto contract = read_json(argv[1]);
        const auto path = fixture(temporary.path, contract);
        const EngineArtifact admitted(path);
        const auto manifest_identity = admitted.manifest_sha256();
        const auto taxonomy_identity = admitted.taxonomy()->sha256();
        check(admitted.engine_bytes() == model && admitted.artifact_kind() == "engine",
              "exact engine capture");
        check(admitted.contract().spec == admitted.manifest().at("contract"),
              "captured Contract binding");
        admitted.taxonomy()->validate_digest(
            admitted.manifest().at("artifacts").at("taxonomy").at("sha256").get<std::string>());
        check(admitted.manifest_bytes() == read_json(path / "manifest.json").dump() + "\n",
              "exact manifest serialization retained");

        // Replace the pathname/inode after admission. The backend view remains
        // the original owned bytes, while a fresh admission rejects the digest.
        write(temporary.path / "replacement", std::string(model.size(), 'R'));
        fs::rename(temporary.path / "replacement", path / "model.engine");
        check(admitted.engine_bytes() == model, "path replacement cannot change captured bytes");
        rejects([&] { EngineArtifact ignored(path); }, "SHA256 mismatch");

        const auto inode_path = fixture(temporary.path, contract);
        const EngineArtifact same_inode(inode_path);
        struct stat before {
        }, after{};
        check(::stat((inode_path / "model.engine").c_str(), &before) == 0, "stat original inode");
        write(inode_path / "model.engine", std::string(model.size(), 'W'));
        check(::stat((inode_path / "model.engine").c_str(), &after) == 0 &&
                  before.st_ino == after.st_ino,
              "fixture overwrites same inode");
        check(same_inode.engine_bytes() == model, "same inode writes cannot change captured bytes");
        rejects([&] { EngineArtifact ignored(inode_path); }, "SHA256 mismatch");

        // Change envelope, meanings and projection consistently at the original
        // paths. A retained descriptor/backend can still use only its old capture.
        auto changed = taxonomy_json();
        changed["classes"][0]["name"] = "CHANGED MEANING";
        write(path / "taxonomy.json", changed.dump() + "\n");
        auto manifest = admitted.manifest();
        manifest["artifacts"]["taxonomy"]["sha256"] = sha256(path / "taxonomy.json");
        manifest["artifacts"]["engine"]["sha256"] = sha256(path / "model.engine");
        manifest["contract"]["geometry"]["overlap_policy"] = "first_wins";
        write(path / "manifest.json", manifest.dump() + "\n");
        const EngineArtifact fresh(path);
        check(fresh.manifest_sha256() != manifest_identity &&
                  fresh.taxonomy()->sha256() != taxonomy_identity,
              "fresh admission captures changed identities");
        check(admitted.manifest_sha256() == manifest_identity &&
                  admitted.taxonomy()->sha256() == taxonomy_identity &&
                  admitted.contract().policy == "last_wins" && admitted.engine_bytes() == model,
              "retained manifest taxonomy Contract and bytes stay consistent");
        fs::remove_all(path);
        check(admitted.engine_bytes() == model && admitted.manifest_sha256() == manifest_identity,
              "source deletion cannot revoke owned bytes");

        const auto onnx_path = fixture(temporary.path, contract, false);
        const EngineArtifact onnx(onnx_path);
        check(onnx.artifact_kind() == "onnx" && onnx.model_bytes() == model,
              "ONNX CPU descriptor capture");
        rejects([&] { (void)onnx.engine_bytes(); }, "ONNX is build-only");
        const auto v2 = fixture(temporary.path, contract, true, false);
        const EngineArtifact legacy(v2, PackageUse::legacy_stream);
        check(!legacy.taxonomy() && legacy.engine_bytes() == model, "legacy engine compatibility");
        rejects([&] { EngineArtifact ignored(onnx_path, PackageUse::legacy_stream); },
                "v3 is file-only");

        const auto linked = fixture(temporary.path, contract);
        fs::remove(linked / "model.engine");
        fs::create_symlink(v2 / "model.engine", linked / "model.engine");
        rejects([&] { EngineArtifact ignored(linked); }, "non-symlink");
        fs::create_directory_symlink(v2, temporary.path / "root-link");
        rejects([&] { EngineArtifact ignored(temporary.path / "root-link"); }, "directory package");
        const auto fifo = fixture(temporary.path, contract);
        fs::remove(fifo / "model.engine");
        check(::mkfifo((fifo / "model.engine").c_str(), 0600) == 0, "FIFO fixture");
        rejects([&] { EngineArtifact ignored(fifo); }, "regular non-symlink");
        const auto large = fixture(temporary.path, contract);
        fs::resize_file(large / "model.engine", EngineArtifact::max_bytes + 1);
        rejects([&] { EngineArtifact ignored(large); }, "size outside supported range");
        fs::resize_file(large / "model.engine", 0);
        rejects([&] { EngineArtifact ignored(large); }, "size outside supported range");
        const auto tax_bad = fixture(temporary.path, contract);
        write(tax_bad / "taxonomy.json", changed.dump() + "\n");
        rejects([&] { EngineArtifact ignored(tax_bad); }, "digest");
        check(target_queries == 0, "zero GPU queries throughout capture/admission failures");
        std::cout << Json{{"status", "PASS"},
                          {"checks", checks},
                          {"rejections", rejections},
                          {"target_queries", target_queries},
                          {"real_gpu_execution", "NOT_RUN"}}
                         .dump()
                  << '\n';
        return 0;
    } catch (const std::exception &error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
