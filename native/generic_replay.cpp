// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Tuomas Pietilä
#include "synrex/generic_replay.hpp"
#include "synrex/backend_profile.hpp"
#include "synrex/classification_transcript.hpp"
#include "synrex/generic_output.hpp"
#include "synrex/generic_session.hpp"
#include "synrex/replay_frames.hpp"
#include "synrex/replay_metrics.hpp"
#include "synrex/replay_queue.hpp"
#include "synrex/transport.hpp"
#include <algorithm>
#include <atomic>
#include <csignal>
#include <fstream>
#include <iostream>
#include <set>
#include <sys/resource.h>
#include <thread>

namespace synrex::generic {
namespace {
namespace wire = streaming::wire;
using Clock = Startup::Clock;
std::atomic<std::sig_atomic_t> interrupted{};
static_assert(decltype(interrupted)::is_always_lock_free);
void interrupt(int) { interrupted.store(1); }
void active() { require(!interrupted, "manifest replay cancelled"); }
double seconds(Clock::time_point start) {
    return std::chrono::duration<double>(Clock::now() - start).count();
}
Json latency(const replay::LatencyHistogram &histogram, const std::string &population,
             const std::string &boundary) {
    const auto quantile = [](std::optional<replay::QuantileBounds> value) -> Json {
        return value ? Json{{"lower_ns", value->lower_ns}, {"upper_ns", value->upper_ns}}
                     : Json(nullptr);
    };
    return {{"schema", "synrex.replay.latency_histogram.v1"},
            {"population", population},
            {"boundary", boundary},
            {"count", histogram.count()},
            {"sum_ns", histogram.sum_ns()},
            {"maximum_ns", histogram.max_ns() ? Json(*histogram.max_ns()) : Json(nullptr)},
            {"p50_ns", quantile(histogram.p50())},
            {"p95_ns", quantile(histogram.p95())},
            {"p99_ns", quantile(histogram.p99())},
            {"quantile_rule", "nearest rank interval; use upper_ns for conservative comparisons"},
            {"bin_width", "exact below128ns; otherwise width<=lower_ns/64"},
            {"storage_bytes", sizeof(histogram)}};
}
struct TrialMeasurement {
    replay::LatencyHistogram histogram;
    std::string population, boundary;
};
using TrialTimings = std::map<std::string, TrialMeasurement>;
Json record_latency(TrialTimings &trial, const std::string &name,
                    const replay::LatencyHistogram &histogram, const std::string &population,
                    const std::string &boundary) {
    auto &measurement = trial[name];
    if (measurement.population.empty()) {
        measurement.population = population;
        measurement.boundary = boundary;
    }
    require(measurement.population == population && measurement.boundary == boundary,
            "trial latency population/boundary mismatch");
    measurement.histogram.merge(histogram);
    return latency(histogram, population, boundary);
}
struct Args {
    std::string mode;
    std::map<std::string, std::string> values;
    Args(int argc, char **argv) {
        require(argc >= 2, "manifest replay requires send|receive|subscribe");
        mode = argv[1];
        require(mode == "send" || mode == "receive" || mode == "subscribe",
                "unknown manifest replay command");
        const std::set<std::string> allowed{
            "--wire",           "--model",       "--receipt",   "--manifest",
            "--session",        "--epoch",       "--points",    "--rate",
            "--port",           "--output-port", "--map-cells", "--queue-frames",
            "--frame-bytes",    "--startup-ms",  "--io-ms",     "--timing-instrumentation",
            "--profile-backend"};
        for (int i = 2; i < argc; i += 2)
            require(i + 1 < argc && allowed.count(argv[i]) &&
                        values.emplace(argv[i], argv[i + 1]).second,
                    "unknown/duplicate/missing manifest replay option; multi-file preservation "
                    "export is NOT_SUPPORTED");
        require(mode == "receive" || !values.count("--output-port"),
                "output-port belongs to manifest receive");
    }
    std::string get(const std::string &key, const std::string &fallback = "") const {
        const auto it = values.find(key);
        if (it != values.end())
            return it->second;
        require(!fallback.empty(), "missing " + key);
        return fallback;
    }
    std::uint64_t number(const std::string &key, std::uint64_t fallback,
                         std::uint64_t maximum) const {
        const auto text = get(key, std::to_string(fallback));
        require(
            !text.empty() && text.size() <= 20 &&
                std::all_of(text.begin(), text.end(), [](char c) { return c >= '0' && c <= '9'; }),
            "unsigned manifest replay option required: " + key);
        const auto value = std::stoull(text);
        require(value > 0 && value <= maximum, "manifest replay option budget: " + key);
        return value;
    }
};
struct Config {
    const Args &args;
    const WireProtocol protocol;
    const Limits limits;
    const replay::FrameOptions frames;
    const std::uint64_t epoch, io_ms;
    const std::uint16_t port, output_port;
    const std::size_t cells, queue_frames;
    const bool timing_enabled, profile_backend;
    explicit Config(const Args &a)
        : args(a), protocol(a.get("--wire") == "binary-source-v1" ? WireProtocol::binary_source_v1
                                                                  : WireProtocol::binary_v3),
          limits{a.number("--points", 16384, 65536),
                 a.number("--frame-bytes",
                          protocol == WireProtocol::binary_source_v1 ? 2097152 : 1048576, 16777216),
                 a.number("--startup-ms", 30000, 120000)},
          frames{static_cast<std::size_t>(limits.max_frame_points),
                 a.number("--rate", 1000000, 20000000)},
          epoch(a.number("--epoch", 1, UINT64_MAX)), io_ms(a.number("--io-ms", 10000, 120000)),
          port(a.number("--port", 39183, 65535)),
          output_port(a.mode == "receive" ? a.number("--output-port", 39184, 65535) : 0),
          cells(a.number("--map-cells", 65536, SemanticMap::max_cells)),
          queue_frames(a.number("--queue-frames", 2, 2)),
          timing_enabled(a.get("--timing-instrumentation", "enabled") == "enabled"),
          profile_backend(a.get("--profile-backend", "disabled") == "enabled") {
        require(a.get("--profile-backend", "disabled") == "enabled" ||
                    a.get("--profile-backend", "disabled") == "disabled",
                "profile-backend requires enabled or disabled");
        require(a.mode == "receive" || !a.values.count("--profile-backend"),
                "profile-backend belongs to manifest receive");
        require(a.get("--timing-instrumentation", "enabled") == "enabled" ||
                    a.get("--timing-instrumentation", "enabled") == "disabled",
                "timing-instrumentation requires enabled or disabled");
        require(a.get("--wire") == "binary-v3" || a.get("--wire") == "binary-source-v1",
                "explicit --wire binary-v3 or binary-source-v1 required");
        limits.validate();
        require(limits.max_frame_bytes >=
                    17676 + (protocol == WireProtocol::binary_source_v1 ? 391 : 0) +
                        (protocol == WireProtocol::binary_source_v1 ? 78 : 50) *
                            limits.max_frame_points,
                "manifest CLI worst-case classification exceeds frame byte budget");
        require(frames.points_per_second >= frames.window_points,
                "manifest window spans more than one synthetic replay second");
        require(a.mode != "receive" || port != output_port, "input/output ports must differ");
    }
    Clock::time_point timing_now() const {
        return timing_enabled ? Clock::now() : Clock::time_point{};
    }
    Startup::Time deadline() const {
        active();
        return Clock::now() + std::chrono::milliseconds(io_ms);
    }
};
void new_path(const fs::path &path) {
    require(!path.empty() && !fs::exists(path) && !fs::is_symlink(path) &&
                fs::is_directory(fs::absolute(path).parent_path()),
            "manifest receipt must be new in an existing parent");
}
void peer_limits(const PeerTransport &transport, const Config &config) {
    require(transport.negotiated_limits().max_frame_points == config.limits.max_frame_points,
            "manifest CLI requires identical peer point limits");
}
void identity(Json &row, const Descriptor &descriptor) {
    row["descriptor_sha256"] = descriptor.sha256();
    row["taxonomy_sha256"] = descriptor.taxonomy().sha256();
    row["map_epoch"] = descriptor.spec().at("map_epoch");
}
Json source_row(const replay::Catalog &catalog, std::size_t index) {
    const auto &source = catalog.sources().at(index);
    return {{"source_index", index},
            {"source_sha256", source.sha256},
            {"wire_begin", source.wire_begin},
            {"expected_records", source.records},
            {"coordinate_frame", source.coordinate_frame},
            {"crs", source.crs},
            {"accepted_records", 0},
            {"freshly_classified_records", 0},
            {"emitted_records", 0},
            {"independently_validated_records", 0},
            {"subscriber_snapshot_acknowledged_records", 0},
            {"frames", 0},
            {"status", "FAIL"}};
}
void preflight(const replay::Catalog &catalog, WireProtocol protocol) {
    // Header/count admission is shared by all three processes before any GPU
    // session construction. The Reader separately verifies decoded coverage.
    for (std::size_t index = 0; index < catalog.sources().size(); ++index) {
        auto handle = catalog.open_source(index);
        require(validate_laz(handle.path()) == catalog.sources().at(index).records,
                "manifest declared/header record count mismatch");
        if (protocol == WireProtocol::binary_source_v1) {
            std::array<unsigned char, 375> header{};
            std::ifstream input(handle.path(), std::ios::binary);
            require(bool(input.read(reinterpret_cast<char *>(header.data()), header.size())),
                    "source-v1 truncated public header");
            const auto format = header[104] & 63;
            const auto length = std::uint64_t(header[105]) | (std::uint64_t(header[106]) << 8);
            require(length == (format == 6   ? 30U
                               : format == 7 ? 36U
                                             : 38U),
                    "source-v1 extra record fields are NOT_SUPPORTED");
        }
        handle.finish();
    }
}
void wait_until(Clock::time_point due) {
    while (Clock::now() < due) {
        active();
        std::this_thread::sleep_until(std::min(due, Clock::now() + std::chrono::milliseconds(10)));
    }
    active();
}
void send_source(const Config &config, const CpuAdmittedPackage &admitted,
                 const replay::Catalog &catalog, std::size_t index, Clock::time_point &anchor,
                 bool &anchored, Json &row, TrialTimings &trial) {
    auto socket = wire::connect_loopback(config.port);
    PeerTransport transport(config.protocol, socket.fd(), admitted, config.limits,
                            config.args.get("--session"), config.epoch + index);
    transport.metadata_admission();
    peer_limits(transport, config);
    transport.acknowledge_execution_ready();
    identity(row, transport.descriptor());
    if (!anchored) {
        anchor = Clock::now();
        anchored = true;
    }
    replay::Coverage coverage(catalog, transport.descriptor(), config.frames, index);
    replay::WindowQueue queue(config.frames.window_points, config.queue_frames);
    Json decode;
    replay::LatencyHistogram decode_windows, assembly, sending, lag, accumulation;
    const auto report_timing = [&] {
        if (!config.timing_enabled) {
            row["timings"] = nullptr;
            return;
        }
        row["timings"] = {
            {"decode_window",
             record_latency(
                 trial, "decode_window", decode_windows, "decoded windows",
                 "Reader::next only; excludes source open/hash, EOF verification and queue wait")},
            {"assembly_and_validation",
             record_latency(trial, "assembly_and_validation", assembly, "assembled frames",
                            "make_frame plus exact coverage validation before pacing")},
            {"frame_send",
             record_latency(
                 trial, "frame_send", sending, "successfully emitted frames",
                 "send_frame serialization and socket write; no peer output acknowledgment")},
            {"sender_scheduling_lag",
             record_latency(trial, "sender_scheduling_lag", lag, "frame dispatch attempts",
                            "same sender steady clock: actual dispatch minus "
                            "scheduled frame release, clamped at zero")},
            {"synthetic_accumulation",
             record_latency(
                 trial, "synthetic_accumulation", accumulation,
                 "successfully emitted source points",
                 "scheduled end-exclusive frame release minus synthetic point timestamp; no "
                 "physical acquisition claim")}};
    };
    const auto decode_start = Clock::now();
    std::thread producer([&] {
        std::unique_ptr<replay::Reader> reader;
        try {
            reader = std::make_unique<replay::Reader>(catalog, index, config.frames.window_points);
            replay::Window window;
            for (;;) {
                const auto before_decode = config.timing_now();
                if (!reader->next(window))
                    break;
                if (config.timing_enabled)
                    decode_windows.add_ns(replay::elapsed_ns(before_decode, config.timing_now()));
                active();
                queue.push(std::move(window));
            }
            reader->finish();
            decode = reader->metadata();
            queue.close();
        } catch (...) {
            const auto error = std::current_exception();
            queue.fail(error);
            // Diagnostics must never escape a producer thread or prevent the
            // consumer from observing the original failure.
            try {
                if (reader)
                    decode = reader->metadata();
            } catch (...) {
            }
        }
    });
    std::uint64_t sequence{}, sent{};
    try {
        replay::Window window;
        while (queue.pop(window)) {
            active();
            const auto before_assembly = config.timing_now();
            auto frame = replay::make_frame(catalog, transport.descriptor(), config.frames, window,
                                            sequence);
            coverage.accept(window, frame);
            if (config.timing_enabled)
                assembly.add_ns(replay::elapsed_ns(before_assembly, config.timing_now()));
            const auto after_last = frame.points.back().id + 1;
            const auto release_ns =
                replay::synthetic_time_ns(after_last, config.frames.points_per_second);
            const auto due = anchor + std::chrono::nanoseconds(release_ns);
            wait_until(due);
            const auto before_send = config.timing_now();
            if (config.timing_enabled)
                lag.add_ns(replay::scheduling_lag_ns(due, before_send));
            transport.send_frame(frame, config.deadline());
            ++sequence;
            row["frames"] = sequence;
            sent += frame.points.size();
            row["emitted_records"] = sent;
            if (config.timing_enabled) {
                sending.add_ns(replay::elapsed_ns(before_send, config.timing_now()));
                for (const auto &point : frame.points)
                    accumulation.add_ns(release_ns - point.timestamp_ns);
            }
        }
        producer.join();
        coverage.source_eof();
        coverage.finish();
        row["input_completion"] = transport.finish_input(config.deadline());
        row["coverage"] = coverage.receipt();
    } catch (...) {
        queue.cancel();
        if (producer.joinable())
            producer.join();
        row["decoder"] = decode;
        row["decode_queue"] = queue.stats().json();
        row["coverage"] = coverage.receipt();
        row["decode_and_dispatch_seconds"] = seconds(decode_start);
        row["maximum_sender_scheduling_lag_seconds"] =
            lag.max_ns() ? Json(*lag.max_ns() / 1e9) : Json(nullptr);
        report_timing();
        throw;
    }
    row["decoder"] = decode;
    row["decode_queue"] = queue.stats().json();
    row["decode_and_dispatch_seconds"] = seconds(decode_start);
    row["maximum_sender_scheduling_lag_seconds"] =
        lag.max_ns() ? Json(*lag.max_ns() / 1e9) : Json(nullptr);
    report_timing();
    row["frames"] = sequence;
}
void receive_source(const Config &config, const CpuAdmittedPackage &admitted,
                    const replay::Catalog &catalog, std::size_t index, wire::Socket &input_listener,
                    wire::Socket &output_listener, Json &row, TrialTimings &trial,
                    std::optional<BackendProfile> &trial_profile) {
    auto input_socket = wire::accept(input_listener, config.io_ms);
    auto output_socket = wire::accept(output_listener, config.io_ms);
    AuthorityTransport input(config.protocol, input_socket.fd(), admitted, config.limits,
                             config.args.get("--session"), config.epoch + index);
    AuthorityTransport output(config.protocol, output_socket.fd(), admitted, config.limits,
                              config.args.get("--session"), config.epoch + index);
    input.metadata_admission();
    output.metadata_admission();
    auto session = input.create_session_with_output(output);
    std::optional<BackendProfile> profile;
    if (config.profile_backend) {
        const auto backend_identity = session->identity();
        profile.emplace(backend_identity);
        if (!trial_profile)
            trial_profile.emplace(backend_identity);
    }
    const auto report_profile = [&] {
        if (profile) {
            row["backend_profile"] = profile->receipt();
            trial_profile->merge(*profile);
        }
    };
    identity(row, input.descriptor());
    ClassificationTranscript semantic(input.descriptor());
    // Reset is structural: no previous Session or map survives this call.
    OutputSender sender(output, 1.0, config.cells);
    replay::Coverage coverage(catalog, input.descriptor(), config.frames, index);
    const auto &source = catalog.sources().at(index);
    std::uint64_t records{}, frames{};
    replay::LatencyHistogram ingress, service;
    const auto report_timing = [&] {
        if (!config.timing_enabled) {
            row["timings"] = nullptr;
            return;
        }
        row["timings"] = {
            {"frame_ingress",
             record_latency(trial, "frame_ingress", ingress, "received frames",
                            "same receiver steady clock: receive_frame wait, socket read "
                            "and codec validation")},
            {"receiver_service",
             record_latency(
                 trial, "receiver_service", service,
                 "successfully emitted points, weighted by frame point count",
                 "same receiver steady clock: after receive_frame validation through coverage, "
                 "inference, map, serialization and output socket write; excludes "
                 "accumulation, ingress wait and subscriber validation")}};
    };
    try {
        while (records < source.records) {
            const auto before_read = config.timing_now();
            auto frame = input.receive_frame(config.deadline());
            const auto after_read = config.timing_now();
            if (config.timing_enabled)
                ingress.add_ns(replay::elapsed_ns(before_read, after_read));
            coverage.accept_frame(frame);
            records += frame.points.size();
            ++frames;
            row["frames"] = frames;
            row["accepted_records"] = records;
            auto result = session->run(std::move(frame));
            row["freshly_classified_records"] = records;
            if (profile)
                profile->observe(result.timings, result.frame.points.size());
            sender.send(result, config.deadline());
            row["emitted_records"] = records;
            if (config.timing_enabled)
                service.add_ns(replay::elapsed_ns(after_read, config.timing_now()),
                               result.frame.points.size());
            semantic.add(result);
        }
        coverage.source_eof();
        coverage.finish();
        row["input_completion"] = input.finish_input(frames, records, config.deadline());
        row["output_completion"] = sender.finish(config.deadline());
    } catch (...) {
        row["classification_semantic_transcript"] = semantic.receipt();
        row["coverage"] = coverage.receipt();
        report_profile();
        report_timing();
        throw;
    }
    row["classification_semantic_transcript"] = semantic.receipt();
    row["coverage"] = coverage.receipt();
    row["subscriber_snapshot_acknowledged_records"] = records;
    row["session_identity"] = session->identity();
    row["frames"] = frames;
    report_profile();
    report_timing();
}
void subscribe_source(const Config &config, const CpuAdmittedPackage &admitted,
                      const replay::Catalog &catalog, std::size_t index, Json &row,
                      TrialTimings &trial) {
    // This is a second independent decode, bounded to one expected window. It
    // checks source attributes as well as the descriptor/map transcript.
    replay::Reader expected(catalog, index, config.frames.window_points);
    auto socket = wire::connect_loopback(config.port);
    PeerTransport transport(config.protocol, socket.fd(), admitted, config.limits,
                            config.args.get("--session"), config.epoch + index);
    transport.metadata_admission();
    peer_limits(transport, config);
    transport.acknowledge_execution_ready();
    identity(row, transport.descriptor());
    OutputReceiver receiver(transport, 1.0, config.cells);
    ClassificationTranscript semantic(transport.descriptor());
    replay::Coverage coverage(catalog, transport.descriptor(), config.frames, index);
    replay::Window window;
    std::uint64_t records{}, frames{};
    replay::LatencyHistogram receive_map, source_validation;
    const auto report_timing = [&] {
        if (!config.timing_enabled) {
            row["timings"] = nullptr;
            return;
        }
        row["timings"] = {
            {"classification_receive_and_map",
             record_latency(trial, "classification_receive_and_map", receive_map,
                            "received classifications",
                            "same subscriber steady clock: classification socket wait/read, "
                            "codec and independent map validation")},
            {"source_attribute_validation",
             record_latency(trial, "source_attribute_validation", source_validation,
                            "independently validated frames",
                            "coverage and attributes against independent Reader window; "
                            "excludes independent source decode")}};
    };
    try {
        while (expected.next(window)) {
            const auto before_receive = config.timing_now();
            const auto result = receiver.receive(config.deadline());
            const auto after_receive = config.timing_now();
            if (config.timing_enabled)
                receive_map.add_ns(replay::elapsed_ns(before_receive, after_receive));
            row["accepted_records"] = records + result.frame.points.size();
            coverage.accept(window, result.frame);
            records += result.frame.points.size();
            row["independently_validated_records"] = records;
            ++frames;
            row["frames"] = frames;
            if (config.timing_enabled)
                source_validation.add_ns(replay::elapsed_ns(after_receive, config.timing_now()));
            semantic.add(result);
        }
        expected.finish();
        coverage.source_eof();
        coverage.finish();
        row["output_completion"] = receiver.finish(config.deadline());
    } catch (...) {
        row["classification_semantic_transcript"] = semantic.receipt();
        row["coverage"] = coverage.receipt();
        row["decoder"] = expected.metadata();
        report_timing();
        throw;
    }
    row["classification_semantic_transcript"] = semantic.receipt();
    row["coverage"] = coverage.receipt();
    row["decoder"] = expected.metadata();
    row["frames"] = frames;
    report_timing();
}
} // namespace
int replay_main(int argc, char **argv) {
    Json receipt{{"schema", "synrex.generic_manifest_cli.v1"},
                 {"status", "FAIL"},
                 {"workload", "file_order_synthetic"},
                 {"boundary_policy", "reset_each_file"},
                 {"multi_file_preservation", "NOT_SUPPORTED"},
                 {"physical_capture_age_seconds", nullptr},
                 {"power_watts", nullptr},
                 {"sources", Json::array()},
                 {"capacity_qualification", "NOT_RUN"},
                 {"source_timestamp_transport",
                  "raw GPS remains bound by source SHA; wire timestamps are synthetic, "
                  "not raw GPS; original integer/scale/GPS roundtrip NOT_SUPPORTED"}};
    fs::path receipt_path;
    const auto start = Clock::now();
    std::optional<Clock::time_point> execution_start;
    Clock::time_point anchor{};
    bool anchored{};
    TrialTimings trial;
    std::optional<BackendProfile> trial_profile;
    try {
        interrupted = 0;
        const Args args(argc, argv);
        receipt_path = args.get("--receipt");
        new_path(receipt_path);
        const Config config(args);
        if (config.profile_backend) {
            receipt["backend_profiling"] = "enabled";
            receipt["trial_backend_profile"] = nullptr;
            receipt["backend_profile_scope"] =
                "backend values already measured by the native Session; independent of replay "
                "timing-instrumentation; no new GPU event or context";
        }
        receipt["binary_wire"] = args.get("--wire");
        receipt["timing_instrumentation"] = config.timing_enabled ? "enabled" : "disabled";
        receipt["timing_availability"] = config.timing_enabled
                                             ? "UNAVAILABLE_NO_SAMPLES"
                                             : "UNAVAILABLE_INSTRUMENTATION_DISABLED";
        receipt["timing_instrumentation_scope"] =
            "optional replay latency histograms and their clock reads only; pacing, deadlines, "
            "correctness counters, queue diagnostics, backend timing, whole-process wall/CPU/RSS "
            "and per-source histogram storage remain in both modes";
        receipt["replay_configuration"] = {{"window_points", config.frames.window_points},
                                           {"points_per_second", config.frames.points_per_second},
                                           {"map_cells", config.cells},
                                           {"queue_frames", config.queue_frames},
                                           {"io_timeout_ms", config.io_ms}};
        if (config.protocol == WireProtocol::binary_source_v1)
            receipt["source_timestamp_transport"] =
                "synrex.las14.source_attributes.v1: exact public header, integer xyz, raw GPS "
                "bits, "
                "original row identity and paired zero returns; timestamps remain synthetic";
        const replay::Catalog catalog(args.get("--manifest"));
        require(catalog.sources().size() - 1 <= UINT64_MAX - config.epoch,
                "manifest descriptor epoch overflow");
        const auto offered_ns =
            replay::synthetic_time_ns(catalog.total_records(), config.frames.points_per_second);
        require(offered_ns <= 86400000000000ULL, "manifest offered duration exceeds one day");
        preflight(catalog, config.protocol);
        const CpuAdmittedPackage admitted(args.get("--model"));
        const auto descriptor = Descriptor::from_package(
            admitted, config.limits, args.get("--session"), config.epoch, config.protocol);
        config.frames.validate(descriptor);
        receipt.update(
            {{"mode", args.mode},
             {"manifest_sha256", catalog.manifest_sha256()},
             {"declared_records", catalog.total_records()},
             {"source_count", catalog.sources().size()},
             {"offered_rate_pps", config.frames.points_per_second},
             {"offered_finite_duration_seconds", offered_ns / 1e9},
             {"admission_seconds", seconds(start)},
             {"descriptor", descriptor.spec()},
             {"scope", args.mode == "send"
                           ? "input delivery acknowledged only; receiver/subscriber "
                             "output qualification is separate"
                       : args.mode == "receive" ? "classified output and independent snapshot ACK; "
                                                  "subscriber source attribute checks are separate"
                                                : "independent source attributes, coverage and map "
                                                  "validated; terminal ACK sent"},
             {"timed_boundary",
              args.mode == "send"      ? "post-admission source reopen/hash, decode, "
                                         "assembly, paced send and per-file startup through "
                                         "input transcript ACK; output excluded"
              : args.mode == "receive" ? "post-admission peer acceptance, per-file "
                                         "startup, frame receipt, inference and "
                                         "output through snapshot ACK"
                                       : "post-admission source reopen/hash, decode, "
                                         "per-file startup, classification receipt, "
                                         "source/map validation and terminal ACK send"},
             {"queue_memory_scope",
              "up to two queued Observation windows plus one producer and one consumer "
              "window; frame/codec/socket/decoder buffers are additional"},
             {"warmup_records", 0}});
        receipt["timing_population"] =
            "trial_timings merges the same measured populations across attempted sources, "
            "including partial failures; sources[].timings retains per-source histograms; "
            "no physical capture-age verdict";
        receipt["instrumentation_overhead"] = "NOT_MEASURED_FOR_ACTUAL_PIPELINE";
        std::signal(SIGINT, interrupt);
        std::signal(SIGTERM, interrupt);
        execution_start = Clock::now();
        auto input_listener =
            args.mode == "receive" ? wire::listen_loopback(config.port) : wire::Socket{};
        auto output_listener =
            args.mode == "receive" ? wire::listen_loopback(config.output_port) : wire::Socket{};
        if (args.mode == "receive")
            std::cerr << "generic_listening=1\n" << std::flush;
        for (std::size_t index = 0; index < catalog.sources().size(); ++index) {
            active();
            auto row = source_row(catalog, index);
            if (config.profile_backend)
                row["backend_profile"] = nullptr;
            const auto source_start = Clock::now();
            try {
                if (args.mode == "send")
                    send_source(config, admitted, catalog, index, anchor, anchored, row, trial);
                else if (args.mode == "receive")
                    receive_source(config, admitted, catalog, index, input_listener,
                                   output_listener, row, trial, trial_profile);
                else
                    subscribe_source(config, admitted, catalog, index, row, trial);
                active();
                row["status"] = "PASS";
            } catch (const std::exception &error) {
                row["error"] = error.what();
            }
            row["elapsed_seconds"] = seconds(source_start);
            receipt["sources"].push_back(row);
            require(row.at("status") == "PASS", "manifest source failed; see source receipt");
        }
        receipt["status"] = "PASS";
    } catch (const std::exception &error) {
        receipt["error"] = error.what();
    }
    const auto end = Clock::now();
    receipt["elapsed_seconds"] = replay::elapsed_ns(start, end) / 1e9;
    if (execution_start)
        receipt["execution_seconds"] = replay::elapsed_ns(*execution_start, end) / 1e9;
    if (anchored)
        receipt["producer_elapsed_seconds"] = replay::elapsed_ns(anchor, end) / 1e9;
    if (trial_profile)
        receipt["trial_backend_profile"] = trial_profile->receipt();
    receipt["trial_timings"] = Json::object();
    for (const auto &[name, measurement] : trial)
        receipt["trial_timings"][name] =
            latency(measurement.histogram, measurement.population, measurement.boundary);
    if (receipt.value("timing_instrumentation", "") == "enabled" &&
        std::any_of(trial.begin(), trial.end(),
                    [](const auto &entry) { return entry.second.histogram.count() > 0; }))
        receipt["timing_availability"] = "MEASURED";
    rusage usage{};
    const bool have_usage = getrusage(RUSAGE_SELF, &usage) == 0;
    const auto cpu_seconds = [have_usage](const timeval &value) -> Json {
        return have_usage && value.tv_sec >= 0 && value.tv_usec >= 0 && value.tv_usec < 1000000
                   ? Json(static_cast<double>(value.tv_sec) + value.tv_usec / 1e6)
                   : Json(nullptr);
    };
    receipt["process_user_cpu_seconds"] = cpu_seconds(usage.ru_utime);
    receipt["process_system_cpu_seconds"] = cpu_seconds(usage.ru_stime);
    receipt["process_cpu_scope"] =
        "getrusage RUSAGE_SELF; whole process lifetime including admission, excluding children";
    receipt["process_peak_rss_bytes"] =
        have_usage && usage.ru_maxrss >= 0 &&
                static_cast<std::uint64_t>(usage.ru_maxrss) <= UINT64_MAX / 1024
            ? Json(static_cast<std::uint64_t>(usage.ru_maxrss) * 1024)
            : Json(nullptr);
    receipt["process_peak_rss_scope"] =
        "Linux ru_maxrss in KiB converted to bytes; whole process lifetime, not device memory";
    for (const auto *name :
         {"accepted_records", "freshly_classified_records", "emitted_records",
          "independently_validated_records", "subscriber_snapshot_acknowledged_records"}) {
        std::uint64_t total{};
        for (const auto &row : receipt.at("sources"))
            total += row.at(name).get<std::uint64_t>();
        receipt[name] = total;
    }
    if (anchored) {
        receipt["actual_sender_rate_pps"] = receipt.at("emitted_records").get<std::uint64_t>() /
                                            receipt.at("producer_elapsed_seconds").get<double>();
        receipt["actual_sender_rate_scope"] =
            "emitted input points per anchored sender second, including failed prefixes; "
            "not a capacity or output success verdict";
    }
    if (receipt.contains("declared_records")) {
        const auto declared = receipt.at("declared_records").get<std::uint64_t>();
        receipt["unattempted_sources"] =
            receipt.at("source_count").get<std::size_t>() - receipt.at("sources").size();
        if (receipt.at("mode") == "subscribe")
            receipt["declared_records_not_independently_validated"] =
                declared - receipt.at("independently_validated_records").get<std::uint64_t>();
        else
            receipt["declared_records_not_emitted"] =
                declared - receipt.at("emitted_records").get<std::uint64_t>();
    }
    try {
        if (!receipt_path.empty())
            write_new(receipt_path,
                      receipt.dump(2, ' ', false, Json::error_handler_t::replace) + "\n");
    } catch (const std::exception &error) {
        receipt["status"] = "FAIL";
        receipt["receipt_error"] = error.what();
    }
    std::cout << receipt.dump(2, ' ', false, Json::error_handler_t::replace) << '\n';
    return receipt.at("status") == "PASS" ? 0 : 1;
}
} // namespace synrex::generic
