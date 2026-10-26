// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Tuomas Pietilä
// Original generic TensorRT 10 backend using NVIDIA's documented public APIs.
#include "gpu.hpp"
#include "synrex/runtime.hpp"
#include <NvInfer.h>
#include <NvOnnxParser.h>
#include <algorithm>
#include <fstream>
#include <iostream>
#include <memory>
#include <sys/utsname.h>

#if NV_TENSORRT_MAJOR != 10
#error "This adapter supports TensorRT 10.x; qualify API changes separately"
#endif
namespace synrex {
static void cuda_check(cudaError_t status) {
    require(status == cudaSuccess, std::string("CUDA: ") + cudaGetErrorString(status));
}
class Logger final : public nvinfer1::ILogger {
    void log(Severity severity, const char *message) noexcept override {
        if (severity <= Severity::kWARNING)
            std::cerr << "TensorRT: " << message << '\n';
    }
};
struct Stream {
    cudaStream_t value{};
    Stream() { cuda_check(cudaStreamCreate(&value)); }
    ~Stream() {
        if (value) {
            cudaStreamSynchronize(value);
            cudaStreamDestroy(value);
        }
    }
};
class Buffer {
    void *value_{};

  public:
    explicit Buffer(std::size_t bytes) {
        require(bytes > 0, "empty device buffer");
        cuda_check(cudaMalloc(&value_, bytes));
    }
    ~Buffer() {
        if (value_)
            cudaFree(value_);
    }
    Buffer(const Buffer &) = delete;
    Buffer &operator=(const Buffer &) = delete;
    void *data() const { return value_; }
    template <typename T> T *as() const { return static_cast<T *>(value_); }
    template <typename T> void upload(const std::vector<T> &values, cudaStream_t stream) {
        cuda_check(cudaMemcpyAsync(value_, values.data(), values.size() * sizeof(T),
                                   cudaMemcpyHostToDevice, stream));
    }
};
bool gpu_enabled() { return true; }
Json target_info() {
    int device = 0;
    cuda_check(cudaGetDevice(&device));
    cudaDeviceProp prop{};
    cuda_check(cudaGetDeviceProperties(&prop, device));
    int version{};
    cuda_check(cudaRuntimeGetVersion(&version));
    struct utsname host{};
    require(::uname(&host) == 0, "cannot identify target architecture");
    return {{"architecture", host.machine},
            {"gpu_name", prop.name},
            {"sm", prop.major * 10 + prop.minor},
            {"cuda_runtime", version},
            {"tensorrt", std::to_string(getInferLibVersion())}};
}
static void check_tensor(const nvinfer1::ICudaEngine &engine, const Contract &c, bool input) {
    const auto &name = input ? c.input_name : c.output_name;
    const auto &dtype = input ? c.input_dtype : c.output_dtype;
    require(engine.getTensorIOMode(name.c_str()) ==
                (input ? nvinfer1::TensorIOMode::kINPUT : nvinfer1::TensorIOMode::kOUTPUT),
            "engine tensor name/mode mismatch");
    require(engine.getTensorDataType(name.c_str()) ==
                (dtype == "float32" ? nvinfer1::DataType::kFLOAT : nvinfer1::DataType::kHALF),
            "engine tensor dtype mismatch");
    require(engine.getTensorFormat(name.c_str()) == nvinfer1::TensorFormat::kLINEAR &&
                engine.getTensorLocation(name.c_str()) == nvinfer1::TensorLocation::kDEVICE,
            "engine I/O must be contiguous linear device tensors");
    const auto shape = engine.getTensorShape(name.c_str());
    const auto &expected = c.spec.at(input ? "input" : "output").at("shape");
    require(shape.nbDims == 5, "engine must have static rank-five tensors");
    for (int i = 0; i < 5; ++i)
        require(shape.d[i] == expected.at(i).get<std::int64_t>(),
                "engine tensor shape mismatch (dynamic shapes unsupported)");
}
static void check_engine(const nvinfer1::ICudaEngine &engine, const Contract &c) {
    require(engine.getNbIOTensors() == 2, "engine must have exactly one input and one output");
    check_tensor(engine, c, true);
    check_tensor(engine, c, false);
}
static std::string read_model(const fs::path &path) {
    const auto count = fs::file_size(path);
    require(count > 0 && count <= 2147483648ULL, "unsupported model size");
    std::string bytes(count, '\0');
    std::ifstream in(path, std::ios::binary);
    require(bool(in.read(bytes.data(), static_cast<std::streamsize>(count))), "model read failure");
    return bytes;
}
void build_engine(const Package &package, const fs::path &output) {
    const auto model = package.artifact("onnx");
    (void)
        target_info(); // Verify CUDA before building; offline compilation still needs a target GPU.
    Logger logger;
    std::unique_ptr<nvinfer1::IBuilder> builder(nvinfer1::createInferBuilder(logger));
    require(bool(builder), "cannot initialize TensorRT builder");
    std::unique_ptr<nvinfer1::INetworkDefinition> network(builder->createNetworkV2(0));
    require(bool(network), "cannot create TensorRT network");
    std::unique_ptr<nvonnxparser::IParser> parser(nvonnxparser::createParser(*network, logger));
    require(bool(parser), "cannot create ONNX parser");
    const auto bytes = read_model(model);
    // In-memory parsing intentionally provides no external-weight filesystem base path.
    require(parser->parse(bytes.data(), bytes.size()),
            "ONNX parse failed; standard self-contained graphs only");
    require(network->getNbInputs() == 1 && network->getNbOutputs() == 1,
            "ONNX must have one input and output");
    std::unique_ptr<nvinfer1::IBuilderConfig> config(builder->createBuilderConfig());
    require(bool(config), "cannot create builder config");
    config->setMemoryPoolLimit(nvinfer1::MemoryPoolType::kWORKSPACE, 256ULL * 1024 * 1024);
    std::unique_ptr<nvinfer1::IHostMemory> plan(builder->buildSerializedNetwork(*network, *config));
    require(bool(plan), "TensorRT engine build failed");
    std::unique_ptr<nvinfer1::IRuntime> runtime(nvinfer1::createInferRuntime(logger));
    require(bool(runtime), "cannot initialize TensorRT runtime");
    std::unique_ptr<nvinfer1::ICudaEngine> engine(
        runtime->deserializeCudaEngine(plan->data(), plan->size()));
    require(bool(engine), "cannot initialize built engine");
    check_engine(*engine, package.contract);
    write_new(output, std::string(static_cast<const char *>(plan->data()), plan->size()));
    std::cout << Json{{"status", "PASS"},
                      {"operation", "TensorRT_engine_build_and_initialization"},
                      {"target", target_info()},
                      {"sha256", sha256(output)}}
                     .dump(2)
              << '\n';
}
std::vector<std::uint8_t> infer(const Package &package, const std::vector<Point> &points,
                                const Plan &plan) {
    require(package.manifest.at("target") == target_info(),
            "engine target mismatch; rebuild and package on this target");
    const auto serialized = read_model(package.artifact("engine"));
    const auto &c = package.contract;
    Logger logger;
    std::unique_ptr<nvinfer1::IRuntime> runtime(nvinfer1::createInferRuntime(logger));
    require(bool(runtime), "cannot initialize TensorRT runtime");
    std::unique_ptr<nvinfer1::ICudaEngine> engine(
        runtime->deserializeCudaEngine(serialized.data(), serialized.size()));
    require(bool(engine), "TensorRT engine deserialization failed");
    check_engine(*engine, c);
    std::unique_ptr<nvinfer1::IExecutionContext> context(engine->createExecutionContext());
    require(bool(context), "TensorRT context initialization failed");
    const bool input_half = c.input_dtype == "float16", output_half = c.output_dtype == "float16";
    const auto input_bytes = c.voxels * c.features.size() * (input_half ? 2 : 4);
    const auto output_bytes = c.voxels * c.las_classes.size() * (output_half ? 2 : 4);
    std::vector<GpuPoint> gpu_points;
    gpu_points.reserve(points.size());
    for (const auto &p : points)
        gpu_points.push_back({{p.xyz[0], p.xyz[1], p.xyz[2]}, double(p.intensity)});
    const std::map<std::string, int> kinds{{"occupancy", 0}, {"count", 1},   {"mean_intensity", 2},
                                           {"local_x", 3},   {"local_y", 4}, {"local_z", 5}};
    std::vector<GpuFeature> features;
    for (const auto &f : c.features)
        features.push_back({kinds.at(f.kind), f.offset, f.scale});
    std::size_t capacity = 0;
    for (const auto &b : plan.blocks)
        capacity = std::max(capacity, b.members.size());
    require(capacity > 0, "empty inference plan");
    std::size_t free_bytes{}, total_bytes{};
    cuda_check(cudaMemGetInfo(&free_bytes, &total_bytes));
    const auto required = input_bytes + output_bytes + gpu_points.size() * sizeof(GpuPoint) +
                          points.size() * 4 + capacity * (8 + sizeof(GpuGroup)) +
                          features.size() * sizeof(GpuFeature) + 260;
    require(required < free_bytes && free_bytes - required > 64ULL * 1024 * 1024,
            "insufficient free GPU memory for bounded adapter buffers");
    Buffer input(input_bytes), output(output_bytes),
        device_points(gpu_points.size() * sizeof(GpuPoint)),
        device_features(features.size() * sizeof(GpuFeature)), mapping(c.las_classes.size()),
        labels(points.size() * sizeof(std::uint32_t)), ids(capacity * 4), voxels(capacity * 4),
        groups(capacity * sizeof(GpuGroup)), bad(sizeof(int));
    std::vector<std::uint32_t> source_ids, voxel_ids;
    std::vector<GpuGroup> group_list;
    source_ids.reserve(capacity);
    voxel_ids.reserve(capacity);
    group_list.reserve(capacity);
    // Declared last so exceptions synchronize before any device or host buffers are released.
    Stream stream;
    device_points.upload(gpu_points, stream.value);
    device_features.upload(features, stream.value);
    mapping.upload(c.las_classes, stream.value);
    cuda_check(cudaMemsetAsync(labels.data(), 255, points.size() * 4, stream.value));
    cuda_check(cudaMemsetAsync(bad.data(), 0, sizeof(int), stream.value));
    require(context->setTensorAddress(c.input_name.c_str(), input.data()) &&
                context->setTensorAddress(c.output_name.c_str(), output.data()),
            "TensorRT tensor binding failed");
    for (const auto &b : plan.blocks) {
        source_ids.clear();
        voxel_ids.clear();
        group_list.clear();
        for (std::size_t i = 0; i < b.members.size(); ++i) {
            const auto &m = b.members[i];
            source_ids.push_back(m.point);
            voxel_ids.push_back(m.voxel);
            if (group_list.empty() || group_list.back().voxel != m.voxel)
                group_list.push_back(
                    {m.voxel, static_cast<unsigned>(i), static_cast<unsigned>(i + 1)});
            else
                group_list.back().end = static_cast<unsigned>(i + 1);
        }
        ids.upload(source_ids, stream.value);
        voxels.upload(voxel_ids, stream.value);
        groups.upload(group_list, stream.value);
        cuda_check(cudaMemsetAsync(input.data(), 0, input_bytes, stream.value));
        prepare_cuda(input.data(), input_half, c.voxels, device_points.as<GpuPoint>(),
                     ids.as<std::uint32_t>(), groups.as<GpuGroup>(),
                     static_cast<unsigned>(group_list.size()), device_features.as<GpuFeature>(),
                     static_cast<unsigned>(features.size()),
                     {{b.origin[0], b.origin[1], b.origin[2]}}, bad.as<int>(), stream.value);
        cuda_check(cudaGetLastError());
        require(context->enqueueV3(stream.value), "TensorRT enqueueV3 failed");
        scatter_cuda(output.data(), output_half, c.voxels,
                     static_cast<unsigned>(c.las_classes.size()), ids.as<std::uint32_t>(),
                     voxels.as<std::uint32_t>(), static_cast<unsigned>(b.members.size()),
                     mapping.as<std::uint8_t>(), labels.as<std::uint32_t>(),
                     c.policy == "first_wins", bad.as<int>(), stream.value);
        cuda_check(cudaGetLastError());
        cuda_check(
            cudaStreamSynchronize(stream.value)); // Also owns lifetime of host upload vectors.
    }
    int invalid{};
    std::vector<std::uint32_t> result(points.size());
    cuda_check(cudaMemcpy(&invalid, bad.data(), sizeof(int), cudaMemcpyDeviceToHost));
    cuda_check(cudaMemcpy(result.data(), labels.data(), result.size() * 4, cudaMemcpyDeviceToHost));
    require(invalid == 0, "nonfinite CUDA tensor feature or occupied-voxel logits");
    std::vector<std::uint8_t> mapped;
    mapped.reserve(result.size());
    for (const auto label : result) {
        require(label <= 255, "source point was not reconstructed");
        mapped.push_back(static_cast<std::uint8_t>(label));
    }
    return mapped;
}
} // namespace synrex
