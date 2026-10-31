// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Tuomas Pietilä
// Original generic package/contract implementation; see docs/byom/PROVENANCE.md.
#include "synrex/runtime.hpp"
#include <algorithm>
#include <cerrno>
#include <cmath>
#include <fcntl.h>
#include <fstream>
#include <iomanip>
#include <map>
#include <memory>
#include <openssl/evp.h>
#include <set>
#include <sstream>
#include <stdexcept>
#include <sys/stat.h>
#include <unistd.h>

namespace synrex {
void require(bool ok, const std::string &message) {
    if (!ok)
        throw std::runtime_error(message);
}
std::string sha256(const fs::path &path) {
    std::ifstream stream(path, std::ios::binary);
    require(bool(stream), "cannot hash artifact: " + path.filename().string());
    std::unique_ptr<EVP_MD_CTX, decltype(&EVP_MD_CTX_free)> ctx(EVP_MD_CTX_new(), EVP_MD_CTX_free);
    require(ctx && EVP_DigestInit_ex(ctx.get(), EVP_sha256(), nullptr) == 1,
            "SHA256 initialization");
    std::array<char, 65536> buffer{};
    while (stream.read(buffer.data(), buffer.size()) || stream.gcount())
        require(EVP_DigestUpdate(ctx.get(), buffer.data(), stream.gcount()) == 1, "SHA256 update");
    require(stream.eof(), "artifact read failure");
    std::array<unsigned char, EVP_MAX_MD_SIZE> digest{};
    unsigned count{};
    require(EVP_DigestFinal_ex(ctx.get(), digest.data(), &count) == 1 && count == 32,
            "SHA256 final");
    std::ostringstream out;
    for (unsigned i = 0; i < count; ++i)
        out << std::hex << std::setfill('0') << std::setw(2) << unsigned(digest[i]);
    return out.str();
}
Json read_json(const fs::path &path) {
    require(fs::is_regular_file(path) && fs::file_size(path) <= 1024 * 1024,
            "missing/oversized JSON");
    std::ifstream in(path);
    // Reject duplicate object keys rather than silently selecting one contract.
    std::vector<std::set<std::string>> keys;
    auto callback = [&keys](int, Json::parse_event_t event, Json &value) {
        if (event == Json::parse_event_t::object_start)
            keys.emplace_back();
        if (event == Json::parse_event_t::key)
            require(keys.back().insert(value.get<std::string>()).second, "duplicate JSON key");
        if (event == Json::parse_event_t::object_end)
            keys.pop_back();
        return true;
    };
    return Json::parse(in, callback);
}
void write_new(const fs::path &path, const std::string &bytes) {
    const int fd = ::open(path.c_str(), O_WRONLY | O_CREAT | O_EXCL | O_NOFOLLOW, 0600);
    require(fd >= 0,
            "cannot create output (exists or parent unavailable): " + path.filename().string());
    std::size_t done = 0;
    while (done < bytes.size()) {
        const auto n = ::write(fd, bytes.data() + done, bytes.size() - done);
        if (n <= 0) {
            ::close(fd);
            fs::remove(path);
            throw std::runtime_error("output write failed");
        }
        done += static_cast<std::size_t>(n);
    }
    if (::close(fd) != 0) {
        fs::remove(path);
        throw std::runtime_error("output close failed");
    }
}
static void exact_keys(const Json &value, std::initializer_list<const char *> expected) {
    require(value.is_object() && value.size() == expected.size(),
            "unknown/missing contract fields");
    for (const auto key : expected)
        require(value.contains(key), std::string("missing field: ") + key);
}
static std::uint64_t positive(const Json &j, std::uint64_t cap) {
    require(j.is_number_integer(), "integer required");
    const auto value = j.get<std::int64_t>();
    require(value > 0 && static_cast<std::uint64_t>(value) <= cap, "integer limit exceeded");
    return static_cast<std::uint64_t>(value);
}
Contract::Contract(const Json &j) : spec(j) {
    exact_keys(j, {"adapter", "input", "output", "geometry", "features", "las_classes", "limits",
                   "licensing"});
    require(j.at("adapter") == "DENSE_VOXEL_SEGMENTATION_V1", "unsupported adapter");
    const auto &g = j.at("geometry");
    exact_keys(
        g, {"voxel_size_xyz", "dimensions_xyz", "block_size_xyz", "overlap_xyz", "overlap_policy"});
    for (const auto key : {"voxel_size_xyz", "dimensions_xyz", "block_size_xyz", "overlap_xyz"})
        require(g.at(key).is_array() && g.at(key).size() == 3, "XYZ triples required");
    voxels = 1;
    for (int a = 0; a < 3; ++a) {
        dimensions[a] = static_cast<int>(positive(g.at("dimensions_xyz").at(a), 4096));
        size[a] = g.at("voxel_size_xyz").at(a).get<double>();
        extent[a] = g.at("block_size_xyz").at(a).get<double>();
        overlap[a] = g.at("overlap_xyz").at(a).get<double>();
        require(std::isfinite(size[a]) && size[a] > 0 && std::isfinite(extent[a]) && extent[a] > 0,
                "invalid voxel/block size");
        require(std::abs(extent[a] - size[a] * dimensions[a]) <= 1e-10 * extent[a],
                "block size must equal dimensions times voxel size");
        require(std::isfinite(overlap[a]) && overlap[a] >= 0 && overlap[a] <= 0.9,
                "overlap outside [0,0.9]");
        voxels *= static_cast<unsigned>(dimensions[a]);
        require(voxels <= 268435456, "voxel grid exceeds adapter limit");
    }
    policy = g.at("overlap_policy").get<std::string>();
    require(policy == "first_wins" || policy == "last_wins", "unsupported overlap policy");
    const auto &f = j.at("features");
    require(f.is_array() && !f.empty() && f.size() <= 32, "requires 1..32 features");
    for (const auto &value : f) {
        exact_keys(value, {"kind", "offset", "scale"});
        Feature feature{value.at("kind").get<std::string>(), value.at("offset").get<float>(),
                        value.at("scale").get<float>()};
        const std::set<std::string> supported{"occupancy", "count",   "mean_intensity",
                                              "local_x",   "local_y", "local_z"};
        require(supported.count(feature.kind) != 0 && std::isfinite(feature.offset) &&
                    std::isfinite(feature.scale) && feature.scale != 0,
                "invalid feature or normalization");
        features.push_back(feature);
    }
    const auto &classes = j.at("las_classes");
    require(classes.is_array() && classes.size() >= 2 && classes.size() <= 256,
            "requires 2..256 mapped model classes");
    for (const auto &value : classes) {
        require(value.is_number_integer() && value.get<std::int64_t>() >= 0 &&
                    value.get<std::int64_t>() <= 255,
                "LAS classes must be uint8");
        las_classes.push_back(value.get<std::uint8_t>());
    }
    for (const auto key : {"input", "output"}) {
        const auto &tensor = j.at(key);
        exact_keys(tensor, {"name", "shape", "dtype", "layout"});
        const auto name = tensor.at("name").get<std::string>();
        require(!name.empty() && name.size() <= 127 && name.find('\0') == std::string::npos,
                "invalid tensor name");
        require(tensor.at("layout") == "NCZYX", "only contiguous NCZYX is supported");
        const auto dtype = tensor.at("dtype").get<std::string>();
        require(dtype == "float32" || dtype == "float16",
                "only float32/float16 tensor I/O supported");
        const auto channels = std::string(key) == "input" ? features.size() : las_classes.size();
        require(tensor.at("shape").is_array() && tensor.at("shape").size() == 5,
                "rank-five shape required");
        for (const auto &dimension : tensor.at("shape"))
            positive(dimension, 268435456);
        require(tensor.at("shape") ==
                    Json::array({1, channels, dimensions[2], dimensions[1], dimensions[0]}),
                "tensor shape disagrees with geometry/classes/features");
        if (std::string(key) == "input") {
            input_name = name;
            input_dtype = dtype;
        } else {
            output_name = name;
            output_dtype = dtype;
        }
    }
    require(input_name != output_name, "tensor names must differ");
    const auto &limits = j.at("limits");
    exact_keys(limits, {"max_points", "max_blocks", "max_memberships", "max_tensor_bytes"});
    max_points = positive(limits.at("max_points"), 20000000);
    max_blocks = positive(limits.at("max_blocks"), 100000);
    max_memberships = positive(limits.at("max_memberships"), 100000000);
    max_tensor_bytes = positive(limits.at("max_tensor_bytes"), 2147483648ULL);
    require(voxels * (features.size() + classes.size()) * 4 <= max_tensor_bytes,
            "tensor memory budget exceeded (FP32 accounting)");
    const auto &license = j.at("licensing");
    exact_keys(license, {"spdx", "author", "source", "weights", "notice"});
    for (const auto key : {"spdx", "author", "source", "weights", "notice"})
        require(license.at(key).is_string() && !license.at(key).get<std::string>().empty(),
                "model licensing metadata required");
}
static void regular(const fs::path &p) {
    require(!fs::is_symlink(p) && fs::is_regular_file(p),
            "package member must be a regular non-symlink file");
}
namespace {
std::string hash_bytes(const std::string &bytes) {
    std::unique_ptr<EVP_MD_CTX, decltype(&EVP_MD_CTX_free)> ctx(EVP_MD_CTX_new(), EVP_MD_CTX_free);
    require(ctx && EVP_DigestInit_ex(ctx.get(), EVP_sha256(), nullptr) == 1 &&
                EVP_DigestUpdate(ctx.get(), bytes.data(), bytes.size()) == 1,
            "manifest SHA256 initialization/update");
    std::array<unsigned char, 32> hash{};
    unsigned count{};
    require(EVP_DigestFinal_ex(ctx.get(), hash.data(), &count) == 1 && count == hash.size(),
            "manifest SHA256 final");
    std::ostringstream out;
    for (const auto value : hash)
        out << std::hex << std::setfill('0') << std::setw(2) << unsigned(value);
    return out.str();
}
bool same_artifact(const struct stat &a, const struct stat &b) {
    return a.st_dev == b.st_dev && a.st_ino == b.st_ino && a.st_mode == b.st_mode &&
           a.st_nlink == b.st_nlink && a.st_size == b.st_size &&
           a.st_mtim.tv_sec == b.st_mtim.tv_sec && a.st_mtim.tv_nsec == b.st_mtim.tv_nsec &&
           a.st_ctim.tv_sec == b.st_ctim.tv_sec && a.st_ctim.tv_nsec == b.st_ctim.tv_nsec;
}
// Hash exactly the initial admitted bytes, optionally copying those same bytes.
// Never follow growth to EOF or reopen a path between hashing and copying.
std::string artifact_snapshot(const fs::path &source, const fs::path *destination = nullptr) {
    const int fd = ::open(source.c_str(), O_RDONLY | O_CLOEXEC | O_NOFOLLOW | O_NONBLOCK);
    require(fd >= 0, "cannot open regular non-symlink source artifact");
    struct Close {
        int fd;
        ~Close() { ::close(fd); }
    } close{fd};
    struct stat initial {};
    constexpr std::uint64_t limit = 2147483648ULL;
    require(::fstat(fd, &initial) == 0 && S_ISREG(initial.st_mode),
            "source artifact must be regular");
    require(initial.st_size > 0 && static_cast<std::uint64_t>(initial.st_size) <= limit,
            "artifact size outside supported range");
    std::unique_ptr<EVP_MD_CTX, decltype(&EVP_MD_CTX_free)> ctx(EVP_MD_CTX_new(), EVP_MD_CTX_free);
    require(ctx && EVP_DigestInit_ex(ctx.get(), EVP_sha256(), nullptr) == 1,
            "artifact SHA256 initialization");
    std::ofstream out;
    if (destination) {
        out.open(*destination, std::ios::binary);
        require(bool(out), "cannot create staged artifact");
    }
    std::array<char, 65536> buffer{};
    const auto size = static_cast<std::uint64_t>(initial.st_size);
    std::uint64_t total = 0;
    while (total < size) {
        const auto requested =
            static_cast<std::size_t>(std::min<std::uint64_t>(buffer.size(), size - total));
        const auto count = ::read(fd, buffer.data(), requested);
        if (count < 0 && errno == EINTR)
            continue;
        require(count > 0, "source artifact truncated/read failure during snapshot");
        total += static_cast<std::uint64_t>(count);
        if (destination) {
            out.write(buffer.data(), static_cast<std::streamsize>(count));
            require(bool(out), "staged artifact write failure");
        }
        require(EVP_DigestUpdate(ctx.get(), buffer.data(), static_cast<std::size_t>(count)) == 1,
                "artifact SHA256 update");
    }
    if (destination) {
        out.close();
        require(bool(out), "staged artifact close failure");
    }
    struct stat final_fd {
    }, final_path{};
    require(::fstat(fd, &final_fd) == 0 && ::lstat(source.c_str(), &final_path) == 0 &&
                same_artifact(initial, final_fd) && same_artifact(initial, final_path),
            "source artifact identity changed during snapshot");
    std::array<unsigned char, 32> hash{};
    unsigned count{};
    require(EVP_DigestFinal_ex(ctx.get(), hash.data(), &count) == 1 && count == hash.size(),
            "artifact SHA256 final");
    std::ostringstream encoded;
    for (const auto value : hash)
        encoded << std::hex << std::setfill('0') << std::setw(2) << unsigned(value);
    return encoded.str();
}
std::string read_manifest_bytes(const fs::path &path) {
    require(!fs::is_symlink(path) && fs::is_directory(path),
            "requires .srxmodel directory package");
    const int fd =
        ::open((path / "manifest.json").c_str(), O_RDONLY | O_CLOEXEC | O_NOFOLLOW | O_NONBLOCK);
    require(fd >= 0, "cannot open regular non-symlink package manifest");
    struct Close {
        int fd;
        ~Close() { ::close(fd); }
    } close{fd};
    constexpr std::size_t limit = 1024 * 1024;
    struct stat info {};
    require(::fstat(fd, &info) == 0 && S_ISREG(info.st_mode), "package manifest must be regular");
    require(info.st_size > 0 && static_cast<std::uint64_t>(info.st_size) <= limit,
            "package manifest size exceeds 1 MiB or is empty");
    std::string result;
    result.reserve(static_cast<std::size_t>(info.st_size));
    std::array<char, 4096> buffer{};
    while (true) {
        const auto count = ::read(fd, buffer.data(), buffer.size());
        if (count < 0 && errno == EINTR)
            continue;
        require(count >= 0, "package manifest read failure");
        if (count == 0)
            break;
        require(static_cast<std::size_t>(count) <= limit - result.size(),
                "package manifest size exceeds 1 MiB");
        result.append(buffer.data(), static_cast<std::size_t>(count));
    }
    return result;
}
// Guard before DOM construction: nlohmann callback DOM parsing can scan parent
// arrays repeatedly. This pass also bounds nesting before building a DOM.
struct ManifestGuard : nlohmann::json_sax<Json> {
    std::vector<std::set<std::string>> objects;
    std::size_t depth{};
    bool null() override { return true; }
    bool boolean(bool) override { return true; }
    bool number_integer(number_integer_t) override { return true; }
    bool number_unsigned(number_unsigned_t) override { return true; }
    bool number_float(number_float_t, const string_t &) override { return true; }
    bool string(string_t &) override { return true; }
    bool binary(binary_t &) override { return false; }
    bool start_object(std::size_t) override {
        require(++depth <= 16, "package manifest nesting limit");
        objects.emplace_back();
        return true;
    }
    bool key(string_t &key) override {
        require(objects.back().insert(key).second, "duplicate JSON key");
        return true;
    }
    bool end_object() override {
        objects.pop_back();
        --depth;
        return true;
    }
    bool start_array(std::size_t) override {
        require(++depth <= 16, "package manifest nesting limit");
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
Json parse_manifest(const std::string &bytes, PackageUse use) {
    ManifestGuard guard;
    require(Json::sax_parse(bytes, &guard), "malformed package manifest");
    auto manifest = Json::parse(bytes);
    // Check the captured envelope before Contract or artifact construction. In particular,
    // matching K/LAS projections cannot make explicit v3 meanings safe on legacy transports.
    require(use != PackageUse::legacy_stream ||
                manifest.value("schema", std::string()) != "synrex.srxmodel.v3",
            "package v3 is file-only; legacy streaming requires package v2");
    return manifest;
}
} // namespace
Package::Package(const fs::path &path) : Package(path, PackageUse::file) {}
Package::Package(const fs::path &path, PackageUse use)
    : root(path), manifest_bytes(read_manifest_bytes(path)),
      manifest_sha256(hash_bytes(manifest_bytes)), manifest(parse_manifest(manifest_bytes, use)),
      contract(manifest.at("contract")) {
    exact_keys(manifest, {"schema", "contract", "artifacts", "target"});
    const bool v3 = manifest.at("schema") == "synrex.srxmodel.v3";
    require(v3 || manifest.at("schema") == "synrex.srxmodel.v2",
            "unsupported model package version");
    const auto &artifacts = manifest.at("artifacts");
    require(artifacts.is_object() && artifacts.contains("license") &&
                (artifacts.contains("onnx") != artifacts.contains("engine")),
            "requires license and exactly one model artifact");
    require(artifacts.size() == (v3 ? 3 : 2) && artifacts.contains("taxonomy") == v3,
            "unexpected/missing package artifacts");
    if (v3) {
        const auto &entry = artifacts.at("taxonomy");
        exact_keys(entry, {"path", "sha256"});
        require(entry.at("path") == "taxonomy.json", "taxonomy fixed package basename required");
        taxonomy.emplace(Taxonomy::read(root / "taxonomy.json"));
        taxonomy->validate_digest(entry.at("sha256").get<std::string>());
        taxonomy->validate_channels(contract.las_classes.size());
        taxonomy->validate_projection(contract.las_classes);
    }
    const std::map<std::string, std::string> names{{"license", "MODEL_LICENSE.txt"},
                                                   {"onnx", "model.onnx"},
                                                   {"engine", "model.engine"},
                                                   {"taxonomy", "taxonomy.json"}};
    std::set<std::string> expected{"manifest.json"};
    for (const auto &item : artifacts.items()) {
        require(names.count(item.key()) != 0, "unsupported artifact");
        exact_keys(item.value(), {"path", "sha256"});
        const auto name = item.value().at("path").get<std::string>();
        require(name == names.at(item.key()), "artifact path must use fixed package basename");
        if (item.key() != "taxonomy") {
            regular(root / name);
            require(fs::file_size(root / name) > 0 && fs::file_size(root / name) <= 2147483648ULL,
                    "artifact size outside supported range");
            require(artifact_snapshot(root / name) == item.value().at("sha256"),
                    "SHA256 mismatch: " + name);
        }
        expected.insert(name);
    }
    for (const auto &entry : fs::directory_iterator(root))
        require(expected.erase(entry.path().filename().string()) == 1, "unexpected package entry");
    require(expected.empty(), "missing package member");
    if (artifacts.contains("onnx"))
        require(manifest.at("target").is_null(), "ONNX packages must have null target");
    else {
        const auto &t = manifest.at("target");
        exact_keys(t, {"architecture", "gpu_name", "sm", "tensorrt", "cuda_runtime"});
        for (const auto key : {"architecture", "gpu_name", "tensorrt"})
            require(t.at(key).is_string() && !t.at(key).get<std::string>().empty(),
                    "invalid target identity");
        positive(t.at("sm"), 999);
        positive(t.at("cuda_runtime"), 999999);
    }
}
fs::path Package::artifact(const std::string &name) const {
    require(manifest.at("artifacts").contains(name), "package has no " + name + " artifact");
    return root / manifest.at("artifacts").at(name).at("path").get<std::string>();
}
void pack(const fs::path &spec, const fs::path &artifact, const std::string &kind,
          const fs::path &license, const fs::path &output) {
    require(kind == "onnx" || kind == "engine", "unknown model artifact type");
    const Contract c(read_json(spec));
    require(fs::is_regular_file(artifact) && fs::is_regular_file(license),
            "model/license files required");
    require(!fs::exists(output) && !fs::is_symlink(output), "refusing package replacement");
    const Json target = kind == "engine" ? target_info() : Json(nullptr);
    // Stage privately beside output; rename through renameat2(NO_REPLACE) on Linux.
    auto pattern = (output.parent_path() / ".srxmodel-XXXXXX").string();
    std::vector<char> buffer(pattern.begin(), pattern.end());
    buffer.push_back('\0');
    const auto p = ::mkdtemp(buffer.data());
    require(p != nullptr, "cannot stage package");
    const fs::path staging(p);
    try {
        Json artifacts = Json::object();
        for (const auto &pair : std::vector<std::pair<std::string, fs::path>>{
                 {kind, artifact}, {"license", license}}) {
            const auto name = pair.first == "license" ? "MODEL_LICENSE.txt" : "model." + kind;
            fs::copy_file(pair.second, staging / name);
            artifacts[pair.first] = {{"path", name}, {"sha256", sha256(staging / name)}};
        }
        const Json manifest{{"schema", "synrex.srxmodel.v2"},
                            {"contract", c.spec},
                            {"artifacts", artifacts},
                            {"target", target}};
        write_new(staging / "manifest.json", manifest.dump(2) + "\n");
        const Package validated(staging);
        require(
            ::renameat2(AT_FDCWD, staging.c_str(), AT_FDCWD, output.c_str(), RENAME_NOREPLACE) == 0,
            "cannot publish package without replacement");
    } catch (...) {
        fs::remove_all(staging);
        throw;
    }
}
namespace {
void new_output(const fs::path &output) {
    require(!fs::exists(output) && !fs::is_symlink(output), "refusing package replacement");
}
void stage_v3(const Contract &contract, const Json &target, Json artifacts,
              const std::vector<std::pair<std::string, fs::path>> &inputs, const Taxonomy &taxonomy,
              const fs::path &output, const Package *source = nullptr,
              bool current_target = false) {
    new_output(output);
    auto pattern = (output.parent_path() / ".srxmodel-XXXXXX").string();
    std::vector<char> buffer(pattern.begin(), pattern.end());
    buffer.push_back('\0');
    const auto created = ::mkdtemp(buffer.data());
    require(created != nullptr, "cannot stage package");
    const fs::path staging(created);
    try {
        for (const auto &input : inputs) {
            auto &entry = artifacts.at(input.first);
            const auto name = entry.at("path").get<std::string>();
            const auto destination = staging / name;
            const auto copied_sha256 = artifact_snapshot(input.second, &destination);
            if (entry.contains("sha256"))
                require(copied_sha256 == entry.at("sha256"),
                        "source artifact changed while staging: " + name);
            entry["sha256"] = copied_sha256;
        }
        // Preserve the exact admitted UTF-8 bytes, including whitespace and escapes.
        write_new(staging / "taxonomy.json", taxonomy.raw_bytes());
        artifacts["taxonomy"] = {{"path", "taxonomy.json"}, {"sha256", taxonomy.sha256()}};
        const Json admitted_target = current_target ? target_info() : target;
        const Json manifest{{"schema", "synrex.srxmodel.v3"},
                            {"contract", contract.spec},
                            {"artifacts", artifacts},
                            {"target", admitted_target}};
        write_new(staging / "manifest.json", manifest.dump(2) + "\n");
        const Package validated(staging);
        require(validated.taxonomy && validated.taxonomy->sha256() == taxonomy.sha256(),
                "staged taxonomy identity mismatch");
        if (source)
            require(read_manifest_bytes(source->root) == source->manifest_bytes,
                    "source package manifest changed during repack");
        require(
            ::renameat2(AT_FDCWD, staging.c_str(), AT_FDCWD, output.c_str(), RENAME_NOREPLACE) == 0,
            "cannot publish package without replacement");
    } catch (...) {
        fs::remove_all(staging);
        throw;
    }
}
} // namespace
void pack(const fs::path &spec, const fs::path &artifact, const std::string &kind,
          const fs::path &license, const fs::path &output, const fs::path &taxonomy_path) {
    require(kind == "onnx" || kind == "engine", "unknown model artifact type");
    const Contract contract(read_json(spec));
    const Taxonomy taxonomy = Taxonomy::read(taxonomy_path);
    taxonomy.validate_channels(contract.las_classes.size());
    taxonomy.validate_projection(contract.las_classes);
    new_output(output);
    const std::vector<std::pair<std::string, fs::path>> inputs{{kind, artifact},
                                                               {"license", license}};
    Json artifacts = Json::object();
    for (const auto &input : inputs) {
        const auto name = input.first == "license" ? "MODEL_LICENSE.txt" : "model." + kind;
        artifacts[input.first] = {{"path", name}};
    }
    stage_v3(contract, Json(nullptr), artifacts, inputs, taxonomy, output, nullptr,
             kind == "engine");
}
void repack(const fs::path &source_path, const fs::path &taxonomy_path, const fs::path &output) {
    const Package source(source_path); // CPU-only validation, including engine target structure.
    const Taxonomy taxonomy = Taxonomy::read(taxonomy_path);
    taxonomy.validate_channels(source.contract.las_classes.size());
    taxonomy.validate_projection(source.contract.las_classes);
    new_output(output);
    const auto source_root = fs::weakly_canonical(source.root);
    const auto destination = fs::weakly_canonical(output);
    require(std::mismatch(source_root.begin(), source_root.end(), destination.begin(),
                          destination.end())
                    .first != source_root.end(),
            "repack output must be outside source package");
    auto artifacts = source.manifest.at("artifacts");
    artifacts.erase("taxonomy");
    std::vector<std::pair<std::string, fs::path>> inputs;
    for (const auto &entry : artifacts.items())
        inputs.emplace_back(entry.key(), source.artifact(entry.key()));
    // Preserve the admitted target verbatim. Repacking never calls target_info().
    stage_v3(source.contract, source.manifest.at("target"), artifacts, inputs, taxonomy, output,
             &source);
}
} // namespace synrex
