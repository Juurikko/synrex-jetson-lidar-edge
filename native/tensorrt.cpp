// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Tuomas Pietilä
// Original generic TensorRT 10 backend using NVIDIA's documented public APIs.
#include "gpu.hpp"
#include "synrex/benchmark.hpp"
#include "synrex/engine_artifact.hpp"
#include "synrex/runtime.hpp"
#include "synrex/session.hpp"
#include <NvInfer.h>
#include <NvOnnxParser.h>
#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstring>
#include <fstream>
#include <iostream>
#include <memory>
#include <mutex>
#include <sys/utsname.h>

#if NV_TENSORRT_MAJOR != 10
#error "This adapter supports TensorRT 10.x; qualify API changes separately"
#endif
namespace synrex {
static std::atomic<std::uint64_t> tracked_device_live{0}, pinned_host_live{0};
Json session_resources() {
    return {{"status", "PASS"}, {"live_tracked_device_bytes", tracked_device_live.load()},
            {"live_pinned_host_bytes", pinned_host_live.load()},
            {"scope", "Requested owned allocations only; driver context and untracked vendor caches excluded"}};
}
static void cuda_check(cudaError_t status) {
    require(status == cudaSuccess, std::string("CUDA: ") + cudaGetErrorString(status));
}
class Logger final : public nvinfer1::ILogger {
    void log(Severity severity, const char *message) noexcept override {
        if (severity <= Severity::kWARNING)
            std::cerr << "TensorRT: " << message << '\n';
    }
};
static Logger &shared_logger() {
    static Logger logger;
    return logger;
}
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
struct Event {
    cudaEvent_t value{};
    Event() { cuda_check(cudaEventCreate(&value)); }
    ~Event() {
        if (value)
            cudaEventDestroy(value);
    }
    Event(const Event &) = delete;
    Event &operator=(const Event &) = delete;
    void record(cudaStream_t stream) { cuda_check(cudaEventRecord(value, stream)); }
    double seconds_since(const Event &start) const {
        float milliseconds{};
        cuda_check(cudaEventElapsedTime(&milliseconds, start.value, value));
        return static_cast<double>(milliseconds) / 1000;
    }
};
// Track actual requested allocation lifetimes across TensorRT callbacks and adapter
// buffers. CUDA context/vendor allocations outside these callbacks are excluded.
class TrackedAllocator final : public nvinfer1::IGpuAllocator {
    mutable std::mutex mutex_;
    std::map<void *, std::uint64_t> sizes_;
    std::uint64_t live_{}, peak_{};

  public:
    void *allocate(std::uint64_t size, std::uint64_t, nvinfer1::AllocatorFlags) noexcept override {
        void *p{};
        if (!size || cudaMalloc(&p, size) != cudaSuccess)
            return nullptr;
        try {
            std::lock_guard<std::mutex> lock(mutex_);
            sizes_.emplace(p, size);
            live_ += size;
            tracked_device_live += size;
            peak_ = std::max(peak_, live_);
            return p;
        } catch (...) {
            cudaFree(p);
            return nullptr;
        }
    }
    bool deallocate(void *p) noexcept override {
        if (!p)
            return true;
        std::lock_guard<std::mutex> lock(mutex_);
        const auto entry = sizes_.find(p);
        if (entry == sizes_.end() || cudaFree(p) != cudaSuccess)
            return false;
        tracked_device_live -= entry->second;
        live_ -= entry->second;
        sizes_.erase(entry);
        return true;
    }
    void *allocateAsync(std::uint64_t size, std::uint64_t alignment, nvinfer1::AllocatorFlags flags,
                        cudaStream_t) noexcept override {
        return allocate(size, alignment, flags);
    }
    bool deallocateAsync(void *p, cudaStream_t) noexcept override { return deallocate(p); }
    std::uint64_t peak() const {
        std::lock_guard<std::mutex> lock(mutex_);
        return peak_;
    }
};
class Buffer {
    void *value_{};
    TrackedAllocator &allocator_;

  public:
    Buffer(TrackedAllocator &allocator, std::size_t bytes) : allocator_(allocator) {
        require(bytes > 0, "empty device buffer");
        value_ = allocator_.allocate(bytes, 0, 0);
        require(value_ != nullptr, "device buffer allocation failed");
    }
    ~Buffer() {
        if (value_)
            allocator_.deallocate(value_);
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
// The public legacy Package is mutable. Re-admit by pathname and compare the
// captured identity before using only the privately retained metadata/bytes.
static void same_admission(const Package &supplied, const EngineArtifact &captured) {
    require(supplied.manifest_bytes == captured.manifest_bytes() &&
                supplied.manifest_sha256 == captured.manifest_sha256() &&
                supplied.manifest == captured.manifest() &&
                supplied.contract.spec == captured.contract().spec &&
                bool(supplied.taxonomy) == bool(captured.taxonomy()),
            "package metadata changed since CPU admission");
    if (supplied.taxonomy)
        require(supplied.taxonomy->raw_bytes() == captured.taxonomy()->raw_bytes(),
                "package taxonomy changed since CPU admission");
}
void build_engine(const Package &package, const fs::path &output, const std::string &precision) {
    require(precision == "default" || precision == "fp16", "unsupported engine precision");
    const EngineArtifact captured(package.root);
    same_admission(package, captured);
    require(captured.artifact_kind() == "onnx", "engine build requires an ONNX artifact");
    const auto &bytes = captured.model_bytes();
    (void)
        target_info(); // Verify CUDA before building; offline compilation still needs a target GPU.
    auto &logger = shared_logger();
    std::unique_ptr<nvinfer1::IBuilder> builder(nvinfer1::createInferBuilder(logger));
    require(bool(builder), "cannot initialize TensorRT builder");
    std::unique_ptr<nvinfer1::INetworkDefinition> network(builder->createNetworkV2(0));
    require(bool(network), "cannot create TensorRT network");
    std::unique_ptr<nvonnxparser::IParser> parser(nvonnxparser::createParser(*network, logger));
    require(bool(parser), "cannot create ONNX parser");
    // In-memory parsing intentionally provides no external-weight filesystem base path.
    require(parser->parse(bytes.data(), bytes.size()),
            "ONNX parse failed; standard self-contained graphs only");
    require(network->getNbInputs() == 1 && network->getNbOutputs() == 1,
            "ONNX must have one input and output");
    std::unique_ptr<nvinfer1::IBuilderConfig> config(builder->createBuilderConfig());
    require(bool(config), "cannot create builder config");
    config->setMemoryPoolLimit(nvinfer1::MemoryPoolType::kWORKSPACE, 256ULL * 1024 * 1024);
    if (precision == "fp16") {
        // TensorRT 10.x weak typing supports mixed FP16 tactics; deprecated in 10.12,
        // still necessary for this explicit 10.x contract. Requalify for TensorRT 11.
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wdeprecated-declarations"
        config->setFlag(nvinfer1::BuilderFlag::kFP16);
#pragma GCC diagnostic pop
    }
    config->setProfilingVerbosity(nvinfer1::ProfilingVerbosity::kDETAILED);
    std::unique_ptr<nvinfer1::IHostMemory> plan(builder->buildSerializedNetwork(*network, *config));
    require(bool(plan), "TensorRT engine build failed");
    std::unique_ptr<nvinfer1::IRuntime> runtime(nvinfer1::createInferRuntime(logger));
    require(bool(runtime), "cannot initialize TensorRT runtime");
    std::unique_ptr<nvinfer1::ICudaEngine> engine(
        runtime->deserializeCudaEngine(plan->data(), plan->size()));
    require(bool(engine), "cannot initialize built engine");
    check_engine(*engine, captured.contract());
    std::unique_ptr<nvinfer1::IEngineInspector> inspector(engine->createEngineInspector());
    require(bool(inspector), "cannot create engine inspector");
    const auto layers = inspector->getEngineInformation(nvinfer1::LayerInformationFormat::kJSON);
    require(layers != nullptr, "cannot inspect engine");
    write_new(output, std::string(static_cast<const char *>(plan->data()), plan->size()));
    std::cout << Json{{"status", "PASS"},
                      {"operation", "TensorRT_engine_build_and_initialization"},
                      {"target", target_info()},
                      {"precision_policy", precision},
                      {"engine_layers", Json::parse(layers)},
                      {"sha256", sha256(output)}}
                     .dump(2)
              << '\n';
}
std::vector<std::uint8_t> infer(const Package &package, const std::vector<Point> &points,
                                const Plan &plan, InferenceMetrics *metrics,
                                std::vector<std::uint8_t> *semantic_ids,
                                const InferenceOptions &options) {
    require(options.iterations > 0 && options.iterations <= 100 && options.warmup <= 20,
            "persistent iteration budget exceeded");
    const EngineArtifact captured(package.root);
    same_admission(package, captured);
    const auto &serialized = captured.engine_bytes();
    const auto &c = captured.contract();
    const auto initialized_at = std::chrono::steady_clock::now();
    InferenceMetrics measured;
    std::size_t initial_free{}, initial_total{};
    cuda_check(cudaMemGetInfo(&initial_free, &initial_total));
    measured.gpu_free_before_initialization_bytes = initial_free;
    require(captured.manifest().at("target") == target_info(),
            "engine target mismatch; rebuild and package on this target");
    TrackedAllocator allocations;
    auto &logger = shared_logger();
    std::unique_ptr<nvinfer1::IRuntime> runtime(nvinfer1::createInferRuntime(logger));
    require(bool(runtime), "cannot initialize TensorRT runtime");
    runtime->setGpuAllocator(&allocations);
    std::unique_ptr<nvinfer1::ICudaEngine> engine(
        runtime->deserializeCudaEngine(serialized.data(), serialized.size()));
    require(bool(engine), "TensorRT engine deserialization failed");
    check_engine(*engine, c);
    std::unique_ptr<nvinfer1::IExecutionContext> context(engine->createExecutionContext());
    require(bool(context), "TensorRT context initialization failed");
    const bool input_half = c.input_dtype == "float16", output_half = c.output_dtype == "float16";
    const auto input_bytes = c.voxels * c.features.size() * (input_half ? 2 : 4);
    const auto output_bytes = c.voxels * c.las_classes.size() * (output_half ? 2 : 4);
    Json trace_blocks = Json::array();
    if (!options.tensor_trace.empty()) {
        require(options.iterations == 1 && options.warmup == 0,
                "tensor tracing requires a single inference cycle");
        require((input_bytes + output_bytes) * plan.blocks.size() <= 256ULL * 1024 * 1024,
                "tensor trace exceeds 256 MiB invocation budget");
        require(fs::create_directory(options.tensor_trace), "tensor trace directory must be new");
    }
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
                          points.size() * 8 + capacity * (8 + sizeof(GpuGroup)) +
                          features.size() * sizeof(GpuFeature) + 260;
    require(required < free_bytes && free_bytes - required > 64ULL * 1024 * 1024,
            "insufficient free GPU memory for bounded adapter buffers");
    Buffer input(allocations, input_bytes), output(allocations, output_bytes),
        device_points(allocations, gpu_points.size() * sizeof(GpuPoint)),
        device_features(allocations, features.size() * sizeof(GpuFeature)),
        mapping(allocations, c.las_classes.size()), labels(allocations, points.size() * 4),
        semantics(allocations, points.size() * 4), ids(allocations, capacity * 4),
        voxels(allocations, capacity * 4), groups(allocations, capacity * sizeof(GpuGroup)),
        bad(allocations, sizeof(int));
    cuda_check(cudaMemGetInfo(&free_bytes, &total_bytes));
    measured.gpu_free_after_allocation_bytes = free_bytes;
    std::vector<std::uint32_t> source_ids, voxel_ids;
    std::vector<GpuGroup> group_list;
    source_ids.reserve(capacity);
    voxel_ids.reserve(capacity);
    group_list.reserve(capacity);
    Event prepared_start, prepared_end, inference_end, postprocess_end;
    // Declared last so exceptions synchronize before any device/host buffers or events release.
    Stream stream;
    device_points.upload(gpu_points, stream.value);
    device_features.upload(features, stream.value);
    mapping.upload(c.las_classes, stream.value);
    cuda_check(cudaMemsetAsync(labels.data(), 255, points.size() * 4, stream.value));
    cuda_check(cudaMemsetAsync(bad.data(), 0, sizeof(int), stream.value));
    require(context->setTensorAddress(c.input_name.c_str(), input.data()) &&
                context->setTensorAddress(c.output_name.c_str(), output.data()),
            "TensorRT tensor binding failed");
    cuda_check(cudaStreamSynchronize(stream.value));
    measured.initialization_and_upload_seconds =
        std::chrono::duration<double>(std::chrono::steady_clock::now() - initialized_at).count();
    std::vector<std::uint32_t> previous, previous_semantics, result(points.size()),
        semantic_result(points.size());
    for (unsigned cycle = 0; cycle < options.warmup + options.iterations; ++cycle) {
        const auto cycle_start = std::chrono::steady_clock::now();
        double prepared_seconds{}, inference_seconds{}, post_seconds{};
        cuda_check(cudaMemsetAsync(labels.data(), 255, points.size() * 4, stream.value));
        cuda_check(cudaMemsetAsync(semantics.data(), 255, points.size() * 4, stream.value));
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
            prepared_start.record(stream.value);
            prepare_cuda(input.data(), input_half, c.voxels, device_points.as<GpuPoint>(),
                         ids.as<std::uint32_t>(), groups.as<GpuGroup>(),
                         static_cast<unsigned>(group_list.size()), device_features.as<GpuFeature>(),
                         static_cast<unsigned>(features.size()),
                         {{b.origin[0], b.origin[1], b.origin[2]}}, bad.as<int>(), stream.value);
            cuda_check(cudaGetLastError());
            prepared_end.record(stream.value);
            require(context->enqueueV3(stream.value), "TensorRT enqueueV3 failed");
            inference_end.record(stream.value);
            scatter_cuda(output.data(), output_half, c.voxels,
                         static_cast<unsigned>(c.las_classes.size()), ids.as<std::uint32_t>(),
                         voxels.as<std::uint32_t>(), static_cast<unsigned>(b.members.size()),
                         mapping.as<std::uint8_t>(), labels.as<std::uint32_t>(),
                         semantics.as<std::uint32_t>(), c.policy == "first_wins", bad.as<int>(),
                         stream.value);
            cuda_check(cudaGetLastError());
            postprocess_end.record(stream.value);
            cuda_check(
                cudaStreamSynchronize(stream.value)); // Also owns lifetime of host upload vectors.
            prepared_seconds += prepared_end.seconds_since(prepared_start);
            inference_seconds += inference_end.seconds_since(prepared_end);
            post_seconds += postprocess_end.seconds_since(inference_end);
            if (!options.tensor_trace.empty()) {
                const auto stem = std::to_string(trace_blocks.size());
                Json block{{"origin", b.origin},
                           {"input", stem + ".input.bin"},
                           {"logits", stem + ".logits.bin"}};
                for (const auto &entry :
                     std::vector<std::pair<std::string, std::pair<void *, std::size_t>>>{
                         {"input", {input.data(), input_bytes}},
                         {"logits", {output.data(), output_bytes}}}) {
                    std::string bytes(entry.second.second, '\0');
                    cuda_check(cudaMemcpy(bytes.data(), entry.second.first, bytes.size(),
                                          cudaMemcpyDeviceToHost));
                    const auto path =
                        options.tensor_trace / block.at(entry.first).get<std::string>();
                    write_new(path, bytes);
                    block[entry.first + "_sha256"] = sha256(path);
                }
                trace_blocks.push_back(block);
            }
        }
        cuda_check(
            cudaMemcpy(result.data(), labels.data(), result.size() * 4, cudaMemcpyDeviceToHost));
        cuda_check(cudaMemcpy(semantic_result.data(), semantics.data(), semantic_result.size() * 4,
                              cudaMemcpyDeviceToHost));
        if (!previous.empty())
            require(result == previous && semantic_result == previous_semantics,
                    "persistent predictions changed between cycles");
        previous = result;
        previous_semantics = semantic_result;
        const auto cycle_seconds =
            std::chrono::duration<double>(std::chrono::steady_clock::now() - cycle_start).count();
        measured.persistent_cycles.push_back({{"warmup", cycle < options.warmup},
                                              {"wall_seconds", cycle_seconds},
                                              {"cuda_preparation_seconds", prepared_seconds},
                                              {"tensorrt_seconds", inference_seconds},
                                              {"cuda_postprocessing_seconds", post_seconds},
                                              {"point_coverage", result.size()}});
        if (cycle >= options.warmup) {
            measured.cuda_preparation_seconds += prepared_seconds;
            measured.tensorrt_inference_seconds += inference_seconds;
            measured.cuda_postprocessing_seconds += post_seconds;
            measured.completed_inference_blocks += plan.blocks.size();
        }
    }
    int invalid{};
    cuda_check(cudaMemcpy(&invalid, bad.data(), sizeof(int), cudaMemcpyDeviceToHost));
    cuda_check(cudaMemcpy(result.data(), labels.data(), result.size() * 4, cudaMemcpyDeviceToHost));
    cuda_check(cudaMemcpy(semantic_result.data(), semantics.data(), semantic_result.size() * 4,
                          cudaMemcpyDeviceToHost));
    require(invalid == 0, "nonfinite CUDA tensor feature or occupied-voxel logits");
    std::vector<std::uint8_t> mapped;
    mapped.reserve(result.size());
    for (const auto label : result) {
        require(label <= 255, "source point was not reconstructed");
        mapped.push_back(static_cast<std::uint8_t>(label));
    }
    if (semantic_ids) {
        semantic_ids->clear();
        semantic_ids->reserve(points.size());
        for (const auto id : semantic_result) {
            require(id < c.las_classes.size(), "semantic index was not reconstructed");
            semantic_ids->push_back(static_cast<std::uint8_t>(id));
        }
    }
    measured.tracked_peak_requested_cuda_bytes = allocations.peak();
    if (!options.tensor_trace.empty()) {
        write_new(options.tensor_trace / "manifest.json",
                  Json{{"schema", "synrex.dense_tensor_trace.v1"},
                       {"tensor_contract", c.spec},
                       {"byte_order", "little-endian"},
                       {"blocks", trace_blocks},
                       {"scope", "Actual CUDA-prepared inputs and TensorRT device logits; "
                                 "diagnostic I/O included in classify total"}}
                          .dump(2) +
                      "\n");
    }
    if (metrics)
        *metrics = measured;
    return mapped;
}

// Optional page-locked staging, allocated once per session and synchronized before reuse.
class PinnedStaging {
    void *data_{};
    std::size_t bytes_{};
  public:
    explicit PinnedStaging(std::size_t bytes) : bytes_(bytes) {
        cuda_check(cudaMallocHost(&data_, bytes)); pinned_host_live += bytes_;
    }
    ~PinnedStaging() { if (data_ && cudaFreeHost(data_) == cudaSuccess) pinned_host_live -= bytes_; }
    PinnedStaging(const PinnedStaging &) = delete;
    PinnedStaging &operator=(const PinnedStaging &) = delete;
    void *data() const { return data_; }
};
// Track B: independent persistent window session; legacy infer() remains the R4 reference.
class PersistentSession final : public InferenceSession {
    const std::chrono::steady_clock::time_point started_ = std::chrono::steady_clock::now();
    // Declared before every CUDA-owning member: capture and engine-kind checks
    // must finish before their constructors can query or initialize the device.
    struct PreparedEngine {
        const EngineArtifact artifact;
        explicit PreparedEngine(const fs::path &root) : artifact(root, PackageUse::legacy_stream) {
            (void)artifact.engine_bytes();
        }
    } prepared_;
    const EngineArtifact &package_ = prepared_.artifact;
    std::size_t capacity_;
    SessionOptions options_;
    std::unique_ptr<PinnedStaging> host_points_, host_ids_, host_voxels_, host_groups_, host_labels_, host_semantics_, host_bad_;
    std::uint64_t pinned_bytes_{};
    int device_{};
    std::uint64_t windows_{}, successful_enqueues_{}, classified_source_points_{};
    bool poisoned_{};
    double initialization_seconds_{};
    std::uint64_t input_bytes_{}, output_bytes_{};
    TrackedAllocator allocations_;
    Logger &logger_ = shared_logger();
    std::unique_ptr<nvinfer1::IRuntime> runtime_;
    std::unique_ptr<nvinfer1::ICudaEngine> engine_;
    std::unique_ptr<nvinfer1::IExecutionContext> context_;
    std::unique_ptr<Buffer> input_, output_, points_, features_, mapping_, labels_, semantics_,
                            ids_, voxels_, groups_, bad_;
    Event upload_start_, upload_end_, prepare_end_, infer_end_, scatter_end_;
    Stream stream_; // Destroyed first: synchronize before any device resource is released.
  public:
    PersistentSession(const fs::path &root, std::size_t capacity, const SessionOptions &options)
        : prepared_(root), capacity_(capacity), options_(options) {
        cuda_check(cudaGetDevice(&device_));
        const auto &c = package_.contract();
        require(capacity > 0 && capacity <= c.max_points && capacity <= 1000000,
                "unsafe persistent window point budget");
        require(package_.manifest().at("target") == target_info(),
                "engine target mismatch; build a separately licensed target package");
        const auto &serialized = package_.engine_bytes();
        runtime_.reset(nvinfer1::createInferRuntime(logger_));
        require(bool(runtime_), "cannot initialize persistent TensorRT runtime");
        runtime_->setGpuAllocator(&allocations_);
        engine_.reset(runtime_->deserializeCudaEngine(serialized.data(), serialized.size()));
        require(bool(engine_), "persistent engine deserialization failed");
        check_engine(*engine_, c);
        context_.reset(engine_->createExecutionContext());
        require(bool(context_), "persistent context initialization failed");
        input_bytes_ = c.voxels * c.features.size() * (c.input_dtype == "float16" ? 2 : 4);
        output_bytes_ = c.voxels * c.las_classes.size() * (c.output_dtype == "float16" ? 2 : 4);
        const auto required = input_bytes_ + output_bytes_ +
            capacity * (sizeof(GpuPoint) + 4 * sizeof(std::uint32_t) + sizeof(GpuGroup)) +
            c.features.size() * sizeof(GpuFeature) + c.las_classes.size() + sizeof(int);
        std::size_t free{}, total{};
        cuda_check(cudaMemGetInfo(&free, &total));
        require(required <= 512ULL * 1024 * 1024 && free > required + 64ULL * 1024 * 1024,
                "persistent adapter exceeds memory budget/headroom");
        auto buffer = [&](std::size_t bytes) { return std::make_unique<Buffer>(allocations_, bytes); };
        input_ = buffer(input_bytes_); output_ = buffer(output_bytes_);
        points_ = buffer(capacity * sizeof(GpuPoint));
        features_ = buffer(c.features.size() * sizeof(GpuFeature));
        mapping_ = buffer(c.las_classes.size()); labels_ = buffer(capacity * 4);
        semantics_ = buffer(capacity * 4); ids_ = buffer(capacity * 4);
        voxels_ = buffer(capacity * 4); groups_ = buffer(capacity * sizeof(GpuGroup));
        bad_ = buffer(sizeof(int));
        if (options_.pinned_staging) {
            auto host = [&](std::size_t bytes) { pinned_bytes_ += bytes; return std::make_unique<PinnedStaging>(bytes); };
            host_points_ = host(capacity * sizeof(GpuPoint));
            host_ids_ = host(capacity * 4); host_voxels_ = host(capacity * 4);
            host_groups_ = host(capacity * sizeof(GpuGroup));
            host_labels_ = host(capacity * 4); host_semantics_ = host(capacity * 4); host_bad_ = host(sizeof(int));
        }
        const std::map<std::string, int> kinds{{"occupancy", 0}, {"count", 1}, {"mean_intensity", 2},
                                               {"local_x", 3}, {"local_y", 4}, {"local_z", 5}};
        std::vector<GpuFeature> features;
        for (const auto &f : c.features) features.push_back({kinds.at(f.kind), f.offset, f.scale});
        cuda_check(cudaMemcpy(features_->data(), features.data(), features.size() * sizeof(GpuFeature),
                              cudaMemcpyHostToDevice));
        cuda_check(cudaMemcpy(mapping_->data(), c.las_classes.data(), c.las_classes.size(),
                              cudaMemcpyHostToDevice));
        require(context_->setTensorAddress(c.input_name.c_str(), input_->data()) &&
                    context_->setTensorAddress(c.output_name.c_str(), output_->data()),
                "persistent tensor binding failed");
        initialization_seconds_ = std::chrono::duration<double>(
            std::chrono::steady_clock::now() - started_).count();
    }
    const Contract &contract() const override { return package_.contract(); }
    Json identity() const override {
        return {{"schema", "synrex.persistent_session.v1"},
                {"backend", "CUDA/TensorRT"},
                {"target", package_.manifest().at("target")},
                {"artifacts", package_.manifest().at("artifacts")},
                {"contract", package_.contract().spec},
                {"manifest_sha256", package_.manifest_sha256()},
                {"package_verifications", 1},
                {"engine_deserializations", 1},
                {"execution_contexts", 1},
                {"cuda_streams", 1},
                {"retained_model_bytes", package_.model_bytes().size()},
                {"artifact_bytes_ownership", "private_immutable_cpu_capture"},
                {"max_window_points", capacity_},
                {"completed_windows", windows_},
                {"successful_enqueue_calls", successful_enqueues_},
                {"classified_source_points", classified_source_points_},
                {"geometry_mode", geometry_name(options_.geometry)},
                {"pinned_staging", options_.pinned_staging},
                {"pinned_host_bytes", pinned_bytes_},
                {"initialization_seconds", initialization_seconds_},
                {"tracked_peak_requested_cuda_bytes", allocations_.peak()}};
    }
    Json benchmark_model(const std::vector<Point> &points, unsigned warmup, unsigned seconds) override {
        require(warmup <= 30 && seconds > 0 && seconds <= 120, "model-only duration budget");
        const auto prepared = run(points, {});
        (void)prepared;
        const auto plan = geometry(points, package_.contract(), options_.geometry);
        const auto represented = plan.blocks.back().members.size();
        streaming::LatencyHistogram batches;
        constexpr unsigned batch = 32;
        auto phase = [&](unsigned duration, bool measured) {
            const auto start = std::chrono::steady_clock::now();
            std::uint64_t calls = 0;
            double cuda_seconds = 0;
            while (std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count() < duration) {
                upload_start_.record(stream_.value);
                for (unsigned i = 0; i < batch; ++i) {
                    require(context_->enqueueV3(stream_.value), "model-only enqueueV3 failed");
                    ++successful_enqueues_;
                }
                upload_end_.record(stream_.value);
                cuda_check(cudaStreamSynchronize(stream_.value));
                const auto elapsed = upload_end_.seconds_since(upload_start_);
                calls += batch; cuda_seconds += elapsed;
                if (measured) batches.add(static_cast<std::uint64_t>(elapsed * 1e9 / batch));
            }
            const auto wall = std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
            return Json{{"enqueue_calls", calls}, {"cuda_seconds", cuda_seconds}, {"wall_seconds", wall},
                        {"inferences_per_wall_second", calls / wall}};
        };
        const auto warmed = phase(warmup, false), measured = phase(seconds, true);
        return {{"schema", "synrex.model_only_sustained.v1"}, {"status", seconds >= 30 ? "PASS" : "FAIL"},
                {"session", identity()}, {"warmup", warmed}, {"measurement", measured},
                {"batch_average_inference_latency", batches.json()}, {"batch_enqueues_per_sync", batch},
                {"source_points_represented_per_inference", represented},
                {"source_point_equivalents_per_second", represented * measured.at("inferences_per_wall_second").get<double>()},
                {"freshly_classified_emitted_source_points", 0},
                {"scope", "Actual TensorRT enqueueV3 repeatedly executes the last CUDA-prepared block. Batched timing includes stream scheduling; excludes geometry, H2D, scatter, queues and output. Point-equivalent rate is not fully classified source throughput."}};
    }
    WindowResult run(const std::vector<Point> &points, const fs::path &trace) override {
        require(!poisoned_, "persistent session is poisoned after failed execution");
        require(!points.empty() && points.size() <= capacity_, "persistent window exceeds point budget");
        cuda_check(cudaSetDevice(device_));
        const auto wall = std::chrono::steady_clock::now();
        const auto &c = package_.contract();
        GeometryProfile geometry_profile;
        const auto plan = geometry(points, c, options_.geometry, &geometry_profile);
        const auto geometry_seconds = std::chrono::duration<double>(
            std::chrono::steady_clock::now() - wall).count();
        if (!trace.empty()) {
            require((input_bytes_ + output_bytes_) * plan.blocks.size() <= 256ULL * 1024 * 1024,
                    "persistent tensor trace exceeds budget");
            require(fs::create_directory(trace), "tensor trace directory must be new");
        }
        std::vector<GpuPoint> gpu_points;
        gpu_points.reserve(points.size());
        for (const auto &p : points) gpu_points.push_back({{p.xyz[0], p.xyz[1], p.xyz[2]}, double(p.intensity)});
        std::vector<std::uint32_t> ids, voxels, labels(points.size()), semantics(points.size());
        std::vector<GpuGroup> groups;
        ids.reserve(capacity_); voxels.reserve(capacity_); groups.reserve(capacity_);
        Json trace_blocks = Json::array(), timeline_blocks = Json::array();
        std::uint64_t occupied_voxels = 0;
        double h2d{}, prep{}, inference{}, scatter{}, d2h{};
        int invalid{}; // Remains alive through the exception synchronization path.
        auto upload = [&](Buffer &device, const auto &values, PinnedStaging *host) {
            if (host) {
                const auto bytes = values.size() * sizeof(values.front());
                std::memcpy(host->data(), values.data(), bytes);
                cuda_check(cudaMemcpyAsync(device.data(), host->data(), bytes, cudaMemcpyHostToDevice, stream_.value));
            } else device.upload(values, stream_.value);
        };
        try {
            upload_start_.record(stream_.value);
            upload(*points_, gpu_points, host_points_.get());
            cuda_check(cudaMemsetAsync(labels_->data(), 255, points.size() * 4, stream_.value));
            cuda_check(cudaMemsetAsync(semantics_->data(), 255, points.size() * 4, stream_.value));
            cuda_check(cudaMemsetAsync(bad_->data(), 0, sizeof(int), stream_.value));
            upload_end_.record(stream_.value);
            cuda_check(cudaStreamSynchronize(stream_.value));
            h2d += upload_end_.seconds_since(upload_start_);
            for (const auto &b : plan.blocks) {
                const auto block_started = std::chrono::steady_clock::now();
                ids.clear(); voxels.clear(); groups.clear();
                for (std::size_t i = 0; i < b.members.size(); ++i) {
                    const auto &m = b.members[i];
                    ids.push_back(m.point); voxels.push_back(m.voxel);
                    if (groups.empty() || groups.back().voxel != m.voxel)
                        groups.push_back({m.voxel, static_cast<unsigned>(i), static_cast<unsigned>(i + 1)});
                    else groups.back().end = static_cast<unsigned>(i + 1);
                }
                occupied_voxels += groups.size();
                upload_start_.record(stream_.value);
                upload(*ids_, ids, host_ids_.get()); upload(*voxels_, voxels, host_voxels_.get());
                upload(*groups_, groups, host_groups_.get());
                cuda_check(cudaMemsetAsync(input_->data(), 0, input_bytes_, stream_.value));
                upload_end_.record(stream_.value);
                prepare_cuda(input_->data(), c.input_dtype == "float16", c.voxels,
                    points_->as<GpuPoint>(), ids_->as<std::uint32_t>(), groups_->as<GpuGroup>(),
                    static_cast<unsigned>(groups.size()), features_->as<GpuFeature>(),
                    static_cast<unsigned>(c.features.size()), {{b.origin[0], b.origin[1], b.origin[2]}},
                    bad_->as<int>(), stream_.value);
                cuda_check(cudaGetLastError());
                prepare_end_.record(stream_.value);
                require(context_->enqueueV3(stream_.value), "persistent enqueueV3 failed");
                ++successful_enqueues_;
                infer_end_.record(stream_.value);
                scatter_cuda(output_->data(), c.output_dtype == "float16", c.voxels,
                    static_cast<unsigned>(c.las_classes.size()), ids_->as<std::uint32_t>(),
                    voxels_->as<std::uint32_t>(), static_cast<unsigned>(b.members.size()),
                    mapping_->as<std::uint8_t>(), labels_->as<std::uint32_t>(),
                    semantics_->as<std::uint32_t>(), c.policy == "first_wins", bad_->as<int>(), stream_.value);
                cuda_check(cudaGetLastError());
                scatter_end_.record(stream_.value);
                cuda_check(cudaStreamSynchronize(stream_.value));
                h2d += upload_end_.seconds_since(upload_start_);
                prep += prepare_end_.seconds_since(upload_end_);
                inference += infer_end_.seconds_since(prepare_end_);
                scatter += scatter_end_.seconds_since(infer_end_);
                if (options_.timeline) timeline_blocks.push_back({
                    {"host_submission_begin_offset_seconds", std::chrono::duration<double>(block_started - wall).count()},
                    {"host_after_stream_sync_offset_seconds", std::chrono::duration<double>(std::chrono::steady_clock::now() - wall).count()},
                    {"cuda_event_h2d_zero_seconds", upload_end_.seconds_since(upload_start_)},
                    {"cuda_event_preparation_seconds", prepare_end_.seconds_since(upload_end_)},
                    {"cuda_event_tensorrt_seconds", infer_end_.seconds_since(prepare_end_)},
                    {"cuda_event_scatter_seconds", scatter_end_.seconds_since(infer_end_)}});
                if (!trace.empty()) {
                    const auto stem = std::to_string(trace_blocks.size());
                    Json block{{"origin", b.origin}, {"input", stem + ".input.bin"},
                               {"logits", stem + ".logits.bin"}};
                    for (const auto &entry : std::vector<std::pair<std::string, std::pair<void *, std::size_t>>>{
                        {"input", {input_->data(), input_bytes_}}, {"logits", {output_->data(), output_bytes_}}}) {
                        std::string bytes(entry.second.second, '\0');
                        cuda_check(cudaMemcpy(bytes.data(), entry.second.first, bytes.size(), cudaMemcpyDeviceToHost));
                        const auto path = trace / block.at(entry.first).get<std::string>();
                        write_new(path, bytes); block[entry.first + "_sha256"] = sha256(path);
                    }
                    trace_blocks.push_back(block);
                }
            }
            upload_start_.record(stream_.value);
            cuda_check(cudaMemcpyAsync(host_labels_ ? host_labels_->data() : labels.data(), labels_->data(), labels.size() * 4, cudaMemcpyDeviceToHost, stream_.value));
            cuda_check(cudaMemcpyAsync(host_semantics_ ? host_semantics_->data() : semantics.data(), semantics_->data(), semantics.size() * 4, cudaMemcpyDeviceToHost, stream_.value));
            cuda_check(cudaMemcpyAsync(host_bad_ ? host_bad_->data() : &invalid, bad_->data(), sizeof(int), cudaMemcpyDeviceToHost, stream_.value));
            upload_end_.record(stream_.value);
            cuda_check(cudaStreamSynchronize(stream_.value));
            d2h = upload_end_.seconds_since(upload_start_);
            if (options_.pinned_staging) {
                std::memcpy(labels.data(), host_labels_->data(), labels.size() * 4);
                std::memcpy(semantics.data(), host_semantics_->data(), semantics.size() * 4);
                std::memcpy(&invalid, host_bad_->data(), sizeof(int));
            }
            require(invalid == 0, "nonfinite persistent CUDA feature or occupied logits");
            WindowResult result;
            for (std::size_t i = 0; i < points.size(); ++i) {
                require(semantics[i] < c.las_classes.size() && labels[i] == c.las_classes.at(semantics[i]),
                        "persistent source point not reconstructed");
                result.semantic_ids.push_back(static_cast<std::uint8_t>(semantics[i]));
                result.las_codes.push_back(static_cast<std::uint8_t>(labels[i]));
            }
            if (!trace.empty()) write_new(trace / "manifest.json",
                Json{{"schema", "synrex.dense_tensor_trace.v1"}, {"tensor_contract", c.spec},
                     {"byte_order", "little-endian"}, {"blocks", trace_blocks}}.dump(2) + "\n");
            ++windows_; classified_source_points_ += points.size();
            result.timings = {{"cpu_geometry_seconds", geometry_seconds}, {"h2d_and_zero_seconds", h2d},
                {"cuda_preparation_seconds", prep}, {"tensorrt_seconds", inference},
                {"cuda_scatter_seconds", scatter}, {"d2h_seconds", d2h},
                {"completed_inference_blocks", plan.blocks.size()}, {"freshly_classified_points", points.size()},
                {"occupied_voxels", occupied_voxels},
                {"geometry_minimum_seconds", geometry_profile.minimum_seconds},
                {"geometry_assignment_seconds", geometry_profile.assignment_seconds},
                {"geometry_grouping_seconds", geometry_profile.grouping_seconds},
                {"geometry_record_sort_seconds", geometry_profile.record_sort_seconds},
                {"geometry_member_sort_seconds", geometry_profile.member_sort_seconds},
                {"geometry_direct_points", geometry_profile.direct_points},
                {"geometry_boundary_fallback_points", geometry_profile.boundary_fallback_points},
                {"window_started_steady_ns", std::chrono::duration_cast<std::chrono::nanoseconds>(wall.time_since_epoch()).count()},
                {"window_seconds", std::chrono::duration<double>(std::chrono::steady_clock::now() - wall).count()}};
            if (options_.timeline) result.timings["timeline_blocks"] = timeline_blocks;
            return result;
        } catch (...) {
            cudaStreamSynchronize(stream_.value); // Host upload vectors remain alive through unwind.
            poisoned_ = true;
            throw;
        }
    }
};
std::unique_ptr<InferenceSession> create_session(const fs::path &package, std::size_t capacity, const SessionOptions &options) {
    return std::make_unique<PersistentSession>(package, capacity, options);
}

} // namespace synrex
