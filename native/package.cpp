// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Tuomas Pietilä
// Original generic package/contract implementation; see docs/byom/PROVENANCE.md.
#include "synrex/runtime.hpp"
#include <algorithm>
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
Package::Package(const fs::path &path)
    : root(path), manifest([&path]() {
          require(!fs::is_symlink(path) && fs::is_directory(path),
                  "requires .srxmodel directory package v2");
          regular(path / "manifest.json");
          return read_json(path / "manifest.json");
      }()),
      contract(manifest.at("contract")) {
    exact_keys(manifest, {"schema", "contract", "artifacts", "target"});
    require(manifest.at("schema") == "synrex.srxmodel.v2", "unsupported model package version");
    const auto &artifacts = manifest.at("artifacts");
    require(artifacts.is_object() && artifacts.contains("license") &&
                (artifacts.contains("onnx") != artifacts.contains("engine")),
            "requires license and exactly one model artifact");
    require(artifacts.size() == 2, "unexpected artifacts");
    const std::map<std::string, std::string> names{
        {"license", "MODEL_LICENSE.txt"}, {"onnx", "model.onnx"}, {"engine", "model.engine"}};
    std::set<std::string> expected{"manifest.json"};
    for (const auto &item : artifacts.items()) {
        require(names.count(item.key()) != 0, "unsupported artifact");
        exact_keys(item.value(), {"path", "sha256"});
        const auto name = item.value().at("path").get<std::string>();
        require(name == names.at(item.key()), "artifact path must use fixed package basename");
        regular(root / name);
        require(fs::file_size(root / name) > 0 && fs::file_size(root / name) <= 2147483648ULL,
                "artifact size outside supported range");
        require(sha256(root / name) == item.value().at("sha256"), "SHA256 mismatch: " + name);
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
} // namespace synrex
