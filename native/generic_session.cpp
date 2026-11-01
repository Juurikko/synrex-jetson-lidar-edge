// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Tuomas Pietilä
#include "synrex/generic_session.hpp"
#include "generic_backend.hpp"
#include "synrex/generic_transport.hpp"
#include <algorithm>
#include <iomanip>
#include <openssl/evp.h>
#include <sstream>

namespace synrex::generic {
namespace {
constexpr std::uint64_t trace_cap = 256ULL * 1024 * 1024;
constexpr std::uint64_t root_reserve = 512ULL * 1024, lineage_reserve = 4096, timing_reserve = 4096,
                        completion_reserve = 4096;
std::uint64_t now_ns() {
    return std::chrono::duration_cast<std::chrono::nanoseconds>(
               std::chrono::steady_clock::now().time_since_epoch())
        .count();
}
std::string digest_bytes(const std::string &bytes) {
    unsigned char result[EVP_MAX_MD_SIZE];
    unsigned size{};
    require(EVP_Digest(bytes.data(), bytes.size(), result, &size, EVP_sha256(), nullptr) == 1,
            "diagnostic input digest failed");
    std::ostringstream text;
    for (unsigned i = 0; i < size; ++i)
        text << std::hex << std::setw(2) << std::setfill('0') << unsigned(result[i]);
    return text.str();
}
Json trace_config(const DiagnosticTraceOptions &trace) {
    return {{"schema", "synrex.generic_live_trace.v1"},
            {"enabled", trace.enabled()},
            {"total_byte_limit", trace.total_bytes},
            {"window_limit", trace.windows},
            {"declared_source_commit", trace.declared_source_commit},
            {"scope", "diagnostic I/O included in service; not causal or quality qualification"}};
}
} // namespace
void DiagnosticTraceOptions::validate() const {
    if (!enabled()) {
        require(total_bytes == 0 && windows == 0 && declared_source_commit.empty(),
                "disabled diagnostic trace has nondefault options");
        return;
    }
    require(total_bytes > root_reserve && total_bytes <= trace_cap && windows > 0 && windows <= 64,
            "diagnostic trace total byte/window budget");
    require(
        (declared_source_commit.size() == 40 || declared_source_commit.size() == 64) &&
            std::all_of(declared_source_commit.begin(), declared_source_commit.end(),
                        [](char c) { return (c >= '0' && c <= '9') || (c >= 'a' && c <= 'f'); }),
        "diagnostic declared source commit required");
    const auto absolute = fs::absolute(root);
    require(absolute == absolute.lexically_normal() && root == root.lexically_normal() &&
                root.string().size() <= 1024 && absolute.filename().string().size() <= 128 &&
                absolute.filename() != "." && absolute.filename() != ".." &&
                fs::is_directory(absolute.parent_path()) &&
                fs::canonical(absolute.parent_path()) == absolute.parent_path() &&
                !fs::exists(absolute) && !fs::is_symlink(absolute),
            "diagnostic trace root must be new with an existing canonical parent");
}
std::uint64_t diagnostic_tensor_bound(const Contract &contract, std::uint64_t blocks,
                                      std::uint64_t memberships) {
    require(blocks > 0 && blocks <= contract.max_blocks && blocks <= 1000000,
            "diagnostic trace block bound");
    const auto scalars =
        contract.features.size() * (contract.input_dtype == "float16" ? 2ULL : 4ULL) +
        contract.las_classes.size() * (contract.output_dtype == "float16" ? 2ULL : 4ULL);
    require(contract.voxels <= trace_cap / scalars, "diagnostic tensor byte overflow");
    const auto per_block = contract.voxels * scalars + 4096;
    const auto metadata = 2 * contract.spec.dump(2).size() + 4096;
    require(metadata < trace_cap && blocks <= (trace_cap - metadata) / per_block,
            "diagnostic tensor total byte bound");
    const auto tensors = metadata + blocks * per_block;
    require(memberships <= (trace_cap - tensors) / 8, "diagnostic mapping total byte bound");
    return tensors + 8 * memberships;
}
void DiagnosticTraceOptions::validate_contract(const Contract &contract,
                                               std::uint64_t required_windows) const {
    if (!enabled())
        return;
    require(total_bytes > root_reserve && total_bytes <= trace_cap && windows > 0 && windows <= 64,
            "diagnostic trace contract byte/window options");
    const auto minimum_window = diagnostic_tensor_bound(contract, 1, 1) + lineage_reserve +
                                timing_reserve + completion_reserve;
    require(required_windows > 0 && required_windows <= windows &&
                required_windows <= (total_bytes - root_reserve) / minimum_window,
            "diagnostic trace cannot fit required minimum tensor windows before backend");
}
Session::Session(std::unique_ptr<InferenceSession> backend, Descriptor descriptor,
                 std::function<void(std::uint64_t, std::uint64_t)> authorize,
                 std::function<void()> abort, DiagnosticTraceOptions trace, GeometryMode mode,
                 double admission_seconds)
    : backend_(std::move(backend)), descriptor_(std::move(descriptor)),
      authorize_(std::move(authorize)), abort_(std::move(abort)), trace_(std::move(trace)),
      geometry_mode_(mode), trace_admission_seconds_(admission_seconds),
      trace_reserved_(trace_.enabled() ? root_reserve : 0) {}

std::unique_ptr<Session> AuthorityTransport::create_session(const SessionOptions &options,
                                                            const DiagnosticTraceOptions &trace) {
    try {
        require(!options.timeline, "generic wire output does not support timeline arrays");
        const auto trace_admission_started = std::chrono::steady_clock::now();
        auto admitted_trace = trace;
        if (trace.enabled())
            admitted_trace.root = fs::absolute(trace.root);
        admitted_trace.validate();
        const auto artifact = begin_session();
        const auto capacity = negotiated_limits().max_frame_points;
        admitted_trace.validate_contract(artifact->contract());
        if (admitted_trace.enabled()) {
            require(fs::create_directory(admitted_trace.root),
                    "diagnostic trace root creation failed");
            auto admission = trace_config(admitted_trace);
            admission["descriptor"] = descriptor().spec();
            admission["descriptor_sha256"] = descriptor().sha256();
            admission["executable_sha256"] = sha256("/proc/self/exe");
            admission["artifacts"] = artifact->manifest().at("artifacts");
            admission["target"] = artifact->manifest().at("target");
            admission["geometry_mode"] = geometry_name(options.geometry);
            admission["clock_scope"] =
                "receiver local steady clock; no cross-host or physical capture age claim";
            admission["source_commit_scope"] = "caller declaration; actual executable digest plus "
                                               "external source/build receipt bind provenance";
            const auto bytes = admission.dump(2) + "\n";
            require(bytes.size() <= root_reserve, "diagnostic admission metadata budget");
            write_new(admitted_trace.root / "admission.json", bytes);
        }
        const auto trace_admission_seconds =
            admitted_trace.enabled()
                ? std::chrono::duration<double>(std::chrono::steady_clock::now() -
                                                trace_admission_started)
                      .count()
                : 0.0;
        auto backend = detail::create_persistent_session(artifact, capacity, options);
        require(bool(backend), "generic backend returned no session");
        const auto identity = backend->identity();
        require(backend->contract().spec == artifact->contract().spec &&
                    identity.at("manifest_sha256") == artifact->manifest_sha256() &&
                    identity.at("artifacts") == artifact->manifest().at("artifacts") &&
                    identity.at("target") == artifact->manifest().at("target") &&
                    identity.at("max_window_points") == capacity &&
                    identity.at("backend") == "CUDA/TensorRT",
                "loaded generic backend identity mismatch");
        auto session = std::unique_ptr<Session>(
            new Session(std::move(backend), descriptor(), frame_authorizer(), session_aborter(),
                        admitted_trace, options.geometry, trace_admission_seconds));
        // Check the startup deadline after synchronous vendor construction as well
        // as before it. Expiry disqualifies the session; vendor work is not preempted.
        complete_session();
        return session;
    } catch (...) {
        abort_session();
        throw;
    }
}

std::unique_ptr<Session>
AuthorityTransport::create_session_with_output(AuthorityTransport &output,
                                               const SessionOptions &options,
                                               const DiagnosticTraceOptions &trace) {
    try {
        require(!options.timeline, "generic wire output does not support timeline arrays");
        const auto trace_admission_started = std::chrono::steady_clock::now();
        auto admitted_trace = trace;
        if (trace.enabled())
            admitted_trace.root = fs::absolute(trace.root);
        admitted_trace.validate();
        validate_output_connection(output);
        const auto artifact = begin_session();
        (void)output.begin_session();
        const auto capacity = negotiated_limits().max_frame_points;
        admitted_trace.validate_contract(artifact->contract());
        if (admitted_trace.enabled()) {
            require(fs::create_directory(admitted_trace.root),
                    "diagnostic trace root creation failed");
            auto admission = trace_config(admitted_trace);
            admission["descriptor"] = descriptor().spec();
            admission["descriptor_sha256"] = descriptor().sha256();
            admission["executable_sha256"] = sha256("/proc/self/exe");
            admission["artifacts"] = artifact->manifest().at("artifacts");
            admission["target"] = artifact->manifest().at("target");
            admission["geometry_mode"] = geometry_name(options.geometry);
            admission["clock_scope"] =
                "receiver local steady clock; no cross-host or physical capture age claim";
            admission["source_commit_scope"] = "caller declaration; actual executable digest plus "
                                               "external source/build receipt bind provenance";
            const auto bytes = admission.dump(2) + "\n";
            require(bytes.size() <= root_reserve, "diagnostic admission metadata budget");
            write_new(admitted_trace.root / "admission.json", bytes);
        }
        const auto trace_admission_seconds =
            admitted_trace.enabled()
                ? std::chrono::duration<double>(std::chrono::steady_clock::now() -
                                                trace_admission_started)
                      .count()
                : 0.0;
        auto backend = detail::create_persistent_session(artifact, capacity, options);
        require(bool(backend), "generic backend returned no session");
        const auto identity = backend->identity();
        require(backend->contract().spec == artifact->contract().spec &&
                    identity.at("manifest_sha256") == artifact->manifest_sha256() &&
                    identity.at("artifacts") == artifact->manifest().at("artifacts") &&
                    identity.at("target") == artifact->manifest().at("target") &&
                    identity.at("max_window_points") == capacity &&
                    identity.at("backend") == "CUDA/TensorRT",
                "loaded generic backend identity mismatch");
        auto session = std::unique_ptr<Session>(new Session(
            std::move(backend), descriptor(), frame_authorizer(), session_aborter(output),
            admitted_trace, options.geometry, trace_admission_seconds));
        complete_session();
        output.complete_session();
        return session;
    } catch (...) {
        abort_session();
        output.abort_session();
        throw;
    }
}

Classification Session::run(Frame frame, std::optional<std::uint64_t> receiver_frame_available_ns) {
    try {
        const auto service_started = trace_.enabled() ? std::chrono::steady_clock::now()
                                                      : std::chrono::steady_clock::time_point{};
        require(!poisoned_, "generic session permanently failed");
        // The codec performs size, full identity and source validation. No legacy
        // taxonomy allowlist or parser is used, and host clocks are not wire data.
        const auto encoded = encode_frame(descriptor_, frame);
        authorize_(frame.points.size(), encoded.size());
        const std::array<std::string, 4> source{frame.sensor_id, frame.stream_id,
                                                frame.coordinate_frame, frame.crs};
        require(!source_identity_ || *source_identity_ == source,
                "generic source identity changed within session");
        require(!last_frame_ ||
                    (frame.sequence > (*last_frame_)[0] && frame.frame_id > (*last_frame_)[1] &&
                     frame.frame_start_ns >= (*last_frame_)[2]),
                "generic frame sequence/time replay or regression");
        source_identity_ = source;
        last_frame_ = {frame.sequence, frame.frame_id, frame.frame_end_ns};
        const auto &contract = backend_->contract();
        const bool intensity =
            std::any_of(contract.features.begin(), contract.features.end(),
                        [](const auto &f) { return f.kind == "mean_intensity"; });
        std::vector<Point> points;
        points.reserve(frame.points.size());
        for (const auto &point : frame.points) {
            require(point.intensity || !intensity, "model needs missing source intensity");
            points.push_back({point.xyz, point.intensity.value_or(0)});
        }
        fs::path trace_path;
        std::uint64_t allowance{};
        if (trace_.enabled()) {
            require(trace_windows_ < trace_.windows, "diagnostic trace window budget exhausted");
            const auto plan = geometry(points, contract, geometry_mode_);
            allowance = diagnostic_tensor_bound(contract, plan.blocks.size(), plan.memberships) +
                        lineage_reserve + timing_reserve + completion_reserve;
            require(!receiver_frame_available_ns || *receiver_frame_available_ns <= now_ns(),
                    "diagnostic receive observation is in the future");
            require(trace_reserved_ <= trace_.total_bytes &&
                        allowance <= trace_.total_bytes - trace_reserved_,
                    "diagnostic trace session byte budget exhausted before inference");
            trace_reserved_ += allowance;
            trace_path = trace_.root / ("window-" + std::to_string(trace_windows_));
        }
        auto result = trace_.enabled()
                          ? backend_->run_traced(points, trace_path,
                                                 allowance - lineage_reserve - timing_reserve -
                                                     completion_reserve)
                          : backend_->run(points);
        if (trace_.enabled()) {
            const auto diagnostic_started = std::chrono::steady_clock::now();
            const auto lineage =
                Json{{"schema", "synrex.generic_window_trace.v1"},
                     {"window_index", trace_windows_},
                     {"descriptor_sha256", descriptor_.sha256()},
                     {"manifest_sha256", descriptor_.spec().at("manifest_sha256")},
                     {"context_policy_sha256", descriptor_.spec().at("context_policy_sha256")},
                     {"input_wire_sha256", digest_bytes(encoded)},
                     {"backend_manifest_sha256", sha256(trace_path / "manifest.json")},
                     {"source_stream_id", frame.stream_id},
                     {"sensor_id", frame.sensor_id},
                     {"frame_id", frame.frame_id},
                     {"sequence", frame.sequence},
                     {"points", points.size()},
                     {"first_source_id", frame.points.front().id},
                     {"last_source_id", frame.points.back().id},
                     {"declared_frame_start_ns", frame.frame_start_ns},
                     {"declared_frame_end_ns", frame.frame_end_ns},
                     {"source_identity_scope",
                      "exact encoded input; source lineage, no truth or future context"}}
                    .dump(2) +
                "\n";
            require(lineage.size() <= lineage_reserve, "diagnostic lineage metadata budget");
            write_new(trace_path / "lineage.json", lineage);
            const auto payload_published = now_ns();
            (void)encode_classification(
                descriptor_,
                Classification{frame, result.semantic_ids, result.las_codes, result.timings});
            const auto classification_validated = now_ns();
            const Json timing{
                {"schema", "synrex.generic_window_timing.v1"},
                {"clock", "receiver local steady_clock nanoseconds"},
                {"receiver_frame_available_steady_ns",
                 receiver_frame_available_ns ? Json(*receiver_frame_available_ns) : Json(nullptr)},
                {"receiver_frame_available_scope",
                 "caller observation immediately after receive_frame completed validation; null if "
                 "not supplied"},
                {"service_started_steady_ns", std::chrono::duration_cast<std::chrono::nanoseconds>(
                                                  service_started.time_since_epoch())
                                                  .count()},
                {"trace_payload_published_steady_ns", payload_published},
                {"classification_validated_steady_ns", classification_validated},
                {"boundary",
                 "backend files and lineage publication precede payload sample; this timing record "
                 "publication and final output validation follow; no physical capture claim"}};
            const auto timing_bytes = timing.dump(2) + "\n";
            require(timing_bytes.size() <= timing_reserve, "diagnostic timing metadata budget");
            write_new(trace_path / "timing.json", timing_bytes);
            std::uint64_t actual{};
            for (const auto &entry : fs::directory_iterator(trace_path)) {
                require(!entry.is_symlink() && entry.is_regular_file(),
                        "diagnostic trace unexpected entry");
                const auto bytes = entry.file_size();
                require(bytes <= allowance - completion_reserve - actual,
                        "diagnostic backend exceeded reserved byte budget");
                actual += bytes;
            }
            trace_written_ += actual;
            ++trace_windows_;
            result.timings["diagnostic_service_started_steady_ns"] =
                std::chrono::duration_cast<std::chrono::nanoseconds>(
                    service_started.time_since_epoch())
                    .count();
            result.timings["diagnostic_prevalidation_steady_ns"] =
                std::chrono::duration_cast<std::chrono::nanoseconds>(
                    std::chrono::steady_clock::now().time_since_epoch())
                    .count();
            result.timings["diagnostic_postprocess_seconds"] =
                std::chrono::duration<double>(std::chrono::steady_clock::now() - diagnostic_started)
                    .count();
            result.timings["diagnostic_service_seconds"] =
                std::chrono::duration<double>(std::chrono::steady_clock::now() - service_started)
                    .count();
        }
        Classification output{std::move(frame), std::move(result.semantic_ids),
                              std::move(result.las_codes), std::move(result.timings)};
        // Validate full coverage, ID255/K256 and exact admitted LAS projection,
        // including the actual encoded output budget before exposing results.
        (void)encode_classification(descriptor_, output);
        return output;
    } catch (...) {
        poisoned_ = true;
        abort_();
        throw;
    }
}
void Session::record_output_timing(std::uint64_t window, std::uint64_t ready, std::uint64_t started,
                                   std::uint64_t completed) {
    if (!trace_.enabled())
        return;
    try {
        require(!poisoned_ && window < trace_windows_ && ready <= started && started <= completed &&
                    completed <= now_ns(),
                "diagnostic output timing order");
        const Json timing{
            {"schema", "synrex.generic_window_output_timing.v1"},
            {"clock", "receiver local steady_clock nanoseconds"},
            {"output_ready_steady_ns", ready},
            {"output_send_started_steady_ns", started},
            {"output_send_completed_steady_ns", completed},
            {"boundary", "ready observed after Session::run and all pre-output trace publication; "
                         "send completion precedes this record publication and is not terminal "
                         "subscriber acknowledgment"}};
        const auto bytes = timing.dump(2) + "\n";
        require(bytes.size() <= completion_reserve, "diagnostic output timing byte budget");
        write_new(trace_.root / ("window-" + std::to_string(window)) / "output-timing.json", bytes);
        trace_written_ += bytes.size();
    } catch (...) {
        poisoned_ = true;
        abort_();
        throw;
    }
}
Json Session::identity() const {
    auto result = backend_->identity();
    result["generic_session"] = {{"descriptor_sha256", descriptor_.sha256()},
                                 {"descriptor", descriptor_.spec()},
                                 {"ready_exchange_completed", true},
                                 {"backend_failed", poisoned_}};
    if (trace_.enabled()) {
        result["diagnostic_trace"] = trace_config(trace_);
        result["diagnostic_trace"].update(
            {{"root", trace_.root.string()},
             {"reserved_bytes", trace_reserved_},
             {"written_window_bytes", trace_written_},
             {"completed_windows", trace_windows_},
             {"startup_trace_admission_seconds", trace_admission_seconds_}});
    }
    return result;
}
} // namespace synrex::generic
