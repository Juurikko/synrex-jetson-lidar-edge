// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Tuomas Pietilä
// B5 bounded native benchmark. All neural execution uses InferenceSession, never a simulator.
#include "synrex/benchmark.hpp"
#include "synrex/service.hpp"
#include "synrex/map_oracle.hpp"
#include "synrex/process_memory.hpp"
#include <openssl/evp.h>
#include <iomanip>
#include <sstream>
#include <algorithm>
#include <atomic>
#include <chrono>
#include <csignal>
#include <cstring>
#include <fstream>
#include <iostream>
#include <numeric>
#include <sys/resource.h>
#include <thread>
#include <unistd.h>
namespace {
using namespace synrex;
using namespace synrex::streaming;
volatile std::sig_atomic_t stopped{};
void stop(int) { stopped = 1; }
std::uint64_t argument_uint(const std::string &s, std::uint64_t cap, bool zero = false) {
    require(!s.empty() && std::all_of(s.begin(), s.end(), [](char c){ return c >= '0' && c <= '9'; }), "unsigned benchmark argument required");
    const auto value = std::stoull(s);
    require((zero || value > 0) && value <= cap, "benchmark argument outside budget");
    return value;
}
std::uint64_t mix(std::uint64_t hash, std::uint64_t value) { return (hash ^ value) * 1099511628211ULL; }
std::uint64_t bits(double d) { std::uint64_t v; std::memcpy(&v, &d, 8); return v; }
std::uint64_t rss() {
    struct rusage r{}; require(getrusage(RUSAGE_SELF, &r) == 0, "cannot sample host memory");
    return static_cast<std::uint64_t>(r.ru_maxrss) * 1024;
}
std::uint64_t current_rss() {
    std::ifstream in("/proc/self/statm"); std::uint64_t total{}, resident{};
    require(bool(in >> total >> resident), "cannot sample current host RSS");
    return checked_rss_bytes(resident, sysconf(_SC_PAGESIZE));
}
struct OutputStats {
    LatencyHistogram inference, source_age, queue_wait, sink_latency;
    std::uint64_t points{}, frames{}, map_updates{}, deltas_digest{1469598103934665603ULL}, digest{1469598103934665603ULL};
    std::uint64_t identity_errors{}, delayed_points{}, next_frame{}, encoded_bytes{}, all_blocks{}, measured_blocks{}, occupied_voxels{}, first_eviction_ns{};
    std::array<std::uint64_t, 7> class_counts{};
    LatencyHistogram occupied_per_window;
    Json timeline = Json::array();
    double map_oracle_seconds{}, map_seconds{}, map_validation_seconds{}, encode_seconds{}, consume_seconds{};
    Json cold_first_window;
    std::map<std::string, std::uint64_t> map_operations;
    std::map<std::string, double> stages;
};
}
int main(int argc, char **argv) {
    using namespace synrex;
    using namespace synrex::streaming;
    Json receipt{{"schema", "synrex.sustained_stream.v1"}, {"status", "FAIL"}};
    fs::path receipt_path;
    try {
        const std::set<std::string> allowed{"--model", "--source", "--receipt", "--rate", "--seconds", "--warmup", "--points", "--queue", "--map-cells", "--mode", "--jitter-us", "--latency-ms", "--schedule-ms", "--seed", "--cancel-after", "--overlap-replay", "--motion-mm", "--occlusion-percent", "--pinned", "--timeline", "--geometry", "--churn-mm", "--verify-map", "--map-policy", "--map-detail"};
        std::map<std::string, std::string> args;
        for (int i = 1; i < argc; i += 2) {
            require(i + 1 < argc && allowed.count(argv[i]) && args.emplace(argv[i], argv[i + 1]).second, "invalid benchmark option");
        }
        auto get = [&](const std::string &key, const std::string &fallback = "") {
            if (args.count(key)) return args.at(key);
            require(!fallback.empty(), "missing benchmark option: " + key); return fallback;
        };
        receipt_path = get("--receipt"); require(!fs::exists(receipt_path), "benchmark receipt must be new");
        const auto rate = argument_uint(get("--rate", "100000"), 20000000);
        const auto seconds = argument_uint(get("--seconds", "30"), 120);
        const auto warmup = argument_uint(get("--warmup", "5"), 30, true);
        const auto count = argument_uint(get("--points", "4096"), 65536);
        const auto capacity = argument_uint(get("--queue", "2"), 8);
        const auto map_cells = argument_uint(get("--map-cells", "65536"), 1000000);
        const auto jitter_us = argument_uint(get("--jitter-us", "0"), 100000, true);
        const auto latency_ns = argument_uint(get("--latency-ms", "250"), 5000) * 1000000;
        const auto schedule_ns = argument_uint(get("--schedule-ms", "50"), 1000) * 1000000;
        const auto seed = argument_uint(get("--seed", "91"), UINT32_MAX);
        const auto cancel_after = argument_uint(get("--cancel-after", "0"), 60000, true);
        const auto overlap = argument_uint(get("--overlap-replay", "0"), 75, true);
        const auto motion_mm = argument_uint(get("--motion-mm", "0"), 1000, true);
        const auto occlusion_percent = argument_uint(get("--occlusion-percent", "0"), 75, true);
        const auto churn_mm = argument_uint(get("--churn-mm", "0"), 1000000, true);
        const bool verify_map = argument_uint(get("--verify-map", "0"), 1, true) != 0;
        const auto mode = get("--mode", "points-map");
        require(mode == "points" || mode == "points-map" || mode == "json" || mode == "model-only", "unknown measured output mode");
        std::signal(SIGINT, stop); std::signal(SIGTERM, stop);
        const auto init_start = steady_ns();
        SessionOptions session_options;
        session_options.pinned_staging = argument_uint(get("--pinned", "0"), 1, true) != 0;
        session_options.timeline = argument_uint(get("--timeline", "0"), 1, true) != 0;
        session_options.geometry = geometry_mode(get("--geometry", "reference"));
        auto session = create_session(get("--model"), count, session_options);
        const auto initialization_wall = (steady_ns() - init_start) / 1e9;
        const auto &contract = session->contract();
        const auto taxonomy = contract.las_classes.size() == 7 ? "synrex.fractal7.v1" : "synrex.synthetic3.v1";
        const auto decode_start = steady_ns();
        std::vector<Point> corpus;
        const auto source = get("--source", "synthetic");
        if (source == "synthetic") {
            auto fixture = synthetic(seed, 1, count, std::min<std::uint64_t>(count, 4096), taxonomy);
            for (const auto &chunk : fixture) for (const auto &p : chunk.points) corpus.push_back({p.xyz, *p.intensity});
        } else corpus = decode(source, contract);
        require(corpus.size() >= count && count <= rate, "source corpus smaller than frame or frame duration exceeds one second");
        if (mode == "model-only") {
            std::signal(SIGTERM, SIG_DFL); std::signal(SIGINT, SIG_DFL);
            corpus.resize(count);
            receipt = session->benchmark_model(corpus, warmup, seconds);
            receipt["initialization_wall_seconds"] = initialization_wall;
            write_new(receipt_path, receipt.dump(2) + "\n");
            std::cout << receipt.dump(2) << '\n'; return receipt.at("status") == "PASS" ? 0 : 1;
        }
        std::vector<std::size_t> visible;
        if (occlusion_percent) {
            visible.resize(corpus.size()); std::iota(visible.begin(), visible.end(), 0);
            std::stable_sort(visible.begin(), visible.end(), [&](auto a, auto b) { return corpus[a].xyz[0] < corpus[b].xyz[0]; });
            visible.resize(corpus.size() * (100 - occlusion_percent) / 100);
            require(visible.size() >= count, "occluded corpus too small for unique source identities");
        }
        const auto stride = count * (100 - overlap) / 100;
        auto index_for = [&](std::uint64_t f, std::size_t i) {
            if (!visible.empty() && (f / 8) % 2) return visible[(f * stride + i) % visible.size()];
            return static_cast<std::size_t>((f * stride + i) % corpus.size());
        };
        auto motion = [&](std::uint64_t f) { return motion_mm / 1000.0 * std::sin((f % 64) * 6.283185307179586 / 64) + f * (churn_mm / 1000.0); };
        const auto decode_seconds = (steady_ns() - decode_start) / 1e9;
        const auto warmup_frames = (warmup * rate + count - 1) / count;
        const auto measured_frames = (seconds * rate + count - 1) / count;
        const auto total_frames = warmup_frames + measured_frames;
        const auto measured_duration = double(measured_frames * count) / rate;
        const auto base_unix = std::uint64_t{1700000000000000000ULL};
        const auto anchor = steady_ns() + 50000000;
        const auto measurement_start = anchor + warmup_frames * count * 1000000000ULL / rate;
        const auto planned_end = anchor + total_frames * count * 1000000000ULL / rate;
        OutputStats stats;
        SemanticMap semantic_map(1.0, map_cells, true, map_policy(get("--map-policy", "indexed")),
            argument_uint(get("--map-detail", "0"), 1, true) != 0);
        std::unique_ptr<MapOracle> oracle = verify_map ? std::make_unique<MapOracle>(map_cells) : nullptr;
        std::uint64_t snapshot_as_of = 0;
        std::atomic<std::uint64_t> completed_frames{0}, received_frames{0}, maximum_pending{0}, maximum_oldest_ns{0};
        std::atomic<bool> abort{false};
        std::string abort_reason;
        double maximum_schedule_lag = 0, generation_seconds = 0, json_frontend_seconds = 0;
        std::vector<Json> samples;
        samples.reserve(1600);
        auto consume = [&](const Classification &r) {
            const auto entered = steady_ns();
            const auto frame_index = r.frame.header.frame_id;
            const bool measured = frame_index >= warmup_frames;
            require(frame_index == stats.next_frame++, "duplicated or out-of-order output frame");
            require(r.frame.points.size() == count && r.semantic_ids.size() == count && r.las_codes.size() == count, "output coverage mismatch");
            stats.all_blocks += r.timings.at("completed_inference_blocks").get<std::uint64_t>();
            const auto source_motion = motion(frame_index);
            std::uint64_t digest = stats.digest;
            for (std::size_t i = 0; i < count; ++i) {
                const auto index = index_for(frame_index, i);
                const auto &p = r.frame.points[i]; const auto &original = corpus[index];
                const auto timestamp = base_unix + (frame_index * count + i) * 1000000000ULL / rate;
                auto expected_xyz = original.xyz; expected_xyz[0] += source_motion;
                require(p.id == index && p.timestamp_ns == timestamp && p.xyz == expected_xyz && p.intensity == original.intensity,
                        "original point identity, timestamp, geometry or intensity mismatch");
                require(r.semantic_ids[i] < contract.las_classes.size() && r.las_codes[i] == contract.las_classes.at(r.semantic_ids[i]), "semantic channel or projection mismatch");
                if (measured) {
                    ++stats.class_counts.at(r.semantic_ids[i]);
                    digest = mix(mix(mix(digest, p.id), p.timestamp_ns), r.semantic_ids[i]);
                    for (auto xyz : p.xyz) digest = mix(digest, bits(xyz));
                    digest = mix(digest, *p.intensity);
                }
            }
            if (measured) stats.consume_seconds += (steady_ns() - entered) / 1e9;
            std::uint64_t map_begin = 0, map_end = 0;
            if (mode != "points") {
                const auto validation_before = semantic_map.validation_seconds();
                const auto counts_before = semantic_map.counters().json();
                map_begin = steady_ns();
                const auto start = steady_ns();
                const auto deltas = semantic_map.update(r);
                map_end = steady_ns();
                if (!stats.first_eviction_ns && semantic_map.counters().removals) stats.first_eviction_ns = map_end;
                if (oracle) {
                    const auto begin = steady_ns(); oracle->verify(r,deltas,semantic_map.size());
                    if (measured) stats.map_oracle_seconds += (steady_ns() - begin) / 1e9;
                }
                if (measured) {
                    const auto counts_after = semantic_map.counters().json();
                    for (const auto &item : counts_after.items())
                        stats.map_operations[item.key()] += item.value().get<std::uint64_t>() - counts_before.at(item.key()).get<std::uint64_t>();
                    stats.map_validation_seconds += semantic_map.validation_seconds() - validation_before;
                    stats.map_seconds += (map_end - start) / 1e9;
                    stats.map_updates += deltas.size();
                    for (const auto &d : deltas) {
                        for (auto v : d.key) stats.deltas_digest = mix(stats.deltas_digest, static_cast<std::uint64_t>(v));
                        stats.deltas_digest = mix(mix(stats.deltas_digest, d.timestamp_ns), d.semantic_id);
                        if (d.evicted) for (auto v : *d.evicted) stats.deltas_digest = mix(stats.deltas_digest, static_cast<std::uint64_t>(v));
                    }
                }
            }
            snapshot_as_of = std::max(snapshot_as_of,r.frame.header.frame_end_ns);
            if (mode == "json") {
                const auto start = steady_ns(); const auto encoded = event(r).dump();
                if (measured) { stats.encode_seconds += (steady_ns() - start) / 1e9; stats.encoded_bytes += encoded.size(); }
            }
            const auto done = steady_ns();
            if (frame_index == 0) stats.cold_first_window = {
                {"window_seconds", r.timings.at("window_seconds")}, {"tensorrt_seconds", r.timings.at("tensorrt_seconds")},
                {"cpu_geometry_seconds", r.timings.at("cpu_geometry_seconds")},
                {"session_start_to_first_output_seconds", (done - init_start) / 1e9},
                {"first_point_to_output_seconds", (done - anchor) / 1e9}};
            if (measured) {
                ++stats.frames; stats.points += count;
                stats.measured_blocks += r.timings.at("completed_inference_blocks").get<std::uint64_t>();
                const auto occupied = r.timings.at("occupied_voxels").get<std::uint64_t>();
                stats.occupied_voxels += occupied; stats.occupied_per_window.add(occupied); stats.digest = digest;
                stats.inference.add(static_cast<std::uint64_t>(r.timings.at("window_seconds").get<double>() * 1e9));
                stats.queue_wait.add(static_cast<std::uint64_t>(r.timings.at("input_queue_wait_seconds").get<double>() * 1e9));
                stats.sink_latency.add(done - entered);
                for (const auto &p : r.frame.points) {
                    const auto projected_source_time = anchor + p.timestamp_ns - base_unix;
                    require(done >= projected_source_time, "source point classified before acquisition");
                    const auto age = done - projected_source_time;
                    stats.source_age.add(age); stats.delayed_points += age > latency_ns;
                }
                for (const auto &item : r.timings.items()) if (item.key().find("seconds") != std::string::npos && item.value().is_number())
                    stats.stages[item.key()] += item.value().get<double>();
            }
            if (session_options.timeline && measured && stats.timeline.size() < 256 && (stats.frames <= 128 || stats.frames % 100 == 0)) {
                stats.timeline.push_back({{"frame_id", frame_index}, {"source_window_begin_ns", r.frame.accepted_steady_ns},
                    {"source_window_end_ns", anchor + (frame_index + 1) * count * 1000000000ULL / rate},
                    {"submitted_ns", r.frame.submitted_steady_ns}, {"inference_begin_ns", r.timings.at("window_started_steady_ns")},
                    {"inference_complete_ns", r.timings.at("inference_complete_steady_ns")},
                    {"sink_begin_ns", entered}, {"map_begin_ns", map_begin}, {"map_end_ns", map_end}, {"sink_complete_ns", done},
                    {"cuda_blocks", r.timings.at("timeline_blocks")}});
            }
            completed_frames = frame_index + 1;
        };
        Service service(std::move(session), capacity, consume);
        auto next_sample = anchor;
        const auto initial_rss = current_rss();
        std::uint64_t measured_rss_min = UINT64_MAX, measured_rss_max = 0, measured_incoming_points = 0;
        for (std::uint64_t frame_index = 0; frame_index < total_frames && !stopped; ++frame_index) {
            const auto generation_start = steady_ns();
            Frame frame;
            auto &h = frame.header;
            h.sensor_id = source == "synthetic" ? "synthetic" : "licensed-laz-records";
            h.stream_id = "sustained-r0"; h.coordinate_frame = "source-untransformed";
            h.crs = source == "synthetic" ? "LOCAL:synthetic-metres" : "UNSPECIFIED:source-file";
            h.taxonomy = taxonomy; h.sequence = frame_index; h.frame_id = frame_index;
            h.frame_points = count; h.chunk_count = (count + 4095) / 4096;
            h.frame_start_ns = base_unix + frame_index * count * 1000000000ULL / rate;
            h.frame_end_ns = base_unix + (frame_index * count + count - 1) * 1000000000ULL / rate;
            frame.first_sequence = frame_index * h.chunk_count;
            frame.last_sequence = frame.first_sequence + h.chunk_count - 1;
            frame.accepted_steady_ns = anchor + frame_index * count * 1000000000ULL / rate;
            const auto generated_motion = motion(frame_index);
            frame.points.reserve(count);
            for (std::size_t i = 0; i < count; ++i) {
                const auto index = index_for(frame_index, i);
                const auto &p = corpus[index];
                auto moved_xyz = p.xyz; moved_xyz[0] += generated_motion;
                frame.points.push_back({index, base_unix + (frame_index * count + i) * 1000000000ULL / rate,
                                        moved_xyz, p.intensity, {}, {}});
            }
            generation_seconds += (steady_ns() - generation_start) / 1e9;
            const auto due = anchor + (frame_index + 1) * count * 1000000000ULL / rate;
            const auto jitter = jitter_us ? ((frame_index * 1103515245ULL + seed) % (jitter_us + 1)) * 1000 : 0;
            while (!stopped && steady_ns() < due + jitter) {
                const auto now = steady_ns(); if (now >= due + jitter) break;
                std::this_thread::sleep_for(std::chrono::nanoseconds(std::min<std::uint64_t>(due + jitter - now, 1000000)));
            }
            if (stopped) break;
            const auto lag = steady_ns() > due + jitter ? steady_ns() - due - jitter : 0;
            maximum_schedule_lag = std::max(maximum_schedule_lag, lag / 1e9);
            if (lag > schedule_ns) { abort_reason = "source_schedule_lag_limit"; abort = true; break; }
            if (frame_index >= warmup_frames) measured_incoming_points += count;
            if (!service.submit(std::move(frame), false)) { abort_reason = "bounded_input_queue_full"; abort = true; break; }
            received_frames = frame_index + 1;
            const auto completed = completed_frames.load();
            const auto pending = frame_index + 1 - completed;
            maximum_pending = std::max(maximum_pending.load(), pending);
            const auto oldest_start = anchor + completed * count * 1000000000ULL / rate;
            const auto oldest_age = pending && steady_ns() > oldest_start ? steady_ns() - oldest_start : 0;
            maximum_oldest_ns = std::max<std::uint64_t>(maximum_oldest_ns.load(), oldest_age);
            if (oldest_age > latency_ns * 2) { abort_reason = "oldest_pending_source_age_limit"; abort = true; break; }
            if (cancel_after && steady_ns() - anchor >= cancel_after * 1000000) { abort_reason = "requested_cancellation"; abort = true; break; }
            if (steady_ns() >= next_sample) {
                const auto memory = current_rss();
                if (steady_ns() >= measurement_start) { measured_rss_min = std::min(measured_rss_min, memory); measured_rss_max = std::max(measured_rss_max, memory); }
                auto sample = service.progress(); sample.erase("status");
                sample["elapsed_seconds"] = (steady_ns() - anchor) / 1e9;
                sample["outstanding_frames"] = pending; sample["oldest_pending_source_age_seconds"] = oldest_age / 1e9;
                sample["current_rss_bytes"] = memory;
                samples.push_back(std::move(sample)); next_sample = steady_ns() + 250000000;
                if (memory > 1024ULL * 1024 * 1024) { abort_reason = "host_rss_budget"; abort = true; break; }
            }
        }
        if (stopped) { abort_reason = "signal_cancellation"; abort = true; }
        if (abort) service.cancel();
        service.finish();
        const auto finished = steady_ns();
        auto service_receipt = service.receipt();
        const auto snapshot = semantic_map.snapshot(snapshot_as_of);
        if (oracle && mode != "points") {
            require(snapshot == oracle->snapshot(snapshot_as_of), "final full map snapshot/aging oracle mismatch");
            const auto c = semantic_map.counters();
            require(c.observations == oracle->observations && c.insertions == oracle->insertions &&
                c.existing_updates == oracle->existing && c.removals == oracle->removals &&
                c.late_rejections == oracle->late && c.peak_cells == oracle->peak, "map operation counter oracle mismatch");
        }
        const auto snapshot_bytes = snapshot.dump();
        unsigned char hash[32]; unsigned hash_size{};
        require(EVP_Digest(snapshot_bytes.data(),snapshot_bytes.size(),hash,&hash_size,EVP_sha256(),nullptr)==1 && hash_size==32, "snapshot digest failure");
        std::ostringstream snapshot_hash; snapshot_hash << std::hex << std::setfill('0');
        for (auto byte:hash) snapshot_hash << std::setw(2) << unsigned(byte);
        receipt["map_snapshot_sha256"] = snapshot_hash.str();
        receipt["map_index_profile"] = semantic_map.profile();
        receipt["map_oracle_status"] = oracle && mode != "points" ? "PASS" : "NOT_RUN";
        receipt["map_operations_measured"] = stats.map_operations;
        receipt["full_capacity_churn_seconds_including_drain"] = stats.first_eviction_ns ? (finished - stats.first_eviction_ns) / 1e9 : 0.0;
        receipt["first_eviction_replay_seconds"] = stats.first_eviction_ns ? Json((stats.first_eviction_ns - anchor) / 1e9) : Json(nullptr);
        receipt["map_operations_including_warmup"] = semantic_map.counters().json();
        receipt["churn_x_translation_metres_per_frame"] = churn_mm / 1000.0;
        receipt["geometry_mode"] = geometry_name(session_options.geometry);
        const auto drain_seconds = finished > planned_end ? (finished - planned_end) / 1e9 : 0;
        const bool complete = !abort && stats.frames == measured_frames && stats.points == measured_frames * count;
        const bool latency_pass = stats.source_age.quantile(.99) <= latency_ns / 1e9;
        const bool long_enough = measured_duration >= 30;
        const auto memory_growth = measured_rss_min == UINT64_MAX ? 0 : measured_rss_max - measured_rss_min;
        const bool memory_stable = memory_growth <= 64ULL * 1024 * 1024;
        const auto identity = service_receipt.at("session");
        const bool fresh_verified = identity.at("successful_enqueue_calls") == stats.all_blocks &&
            identity.at("classified_source_points") == service_receipt.at("freshly_classified_points") &&
            identity.at("completed_windows") == service_receipt.at("completed_windows");
        receipt["fresh_execution_count_status"] = complete && fresh_verified ? "PASS" : "FAIL";
        receipt["observed_neural_blocks_including_warmup"] = stats.all_blocks;
        receipt["measured_neural_blocks"] = stats.measured_blocks;
        receipt["semantic_histogram"] = stats.class_counts;
        receipt["occupied_voxels_total"] = stats.occupied_voxels;
        receipt["occupied_voxels_per_window"] = {{"mean", stats.frames ? double(stats.occupied_voxels) / stats.frames : 0},
            {"p50", stats.occupied_per_window.quantile(.5) * 1e9}, {"p99", stats.occupied_per_window.quantile(.99) * 1e9}};
        receipt["timeline"] = stats.timeline;
        receipt["timeline_scope"] = "Host monotonic timestamps; CUDA event durations within host submission/synchronization brackets. GPU stages within a stream are serial; worker, sink and source overlap. Stage sums are not wall-clock latency.";
        receipt["status"] = complete && fresh_verified && service_receipt.at("status") == "PASS" && latency_pass && long_enough && memory_stable ? "PASS" : "FAIL";
        receipt["failure_phase"] = receipt.at("status") == "PASS" ? "none" :
            (abort ? (received_frames <= warmup_frames ? "warmup" : "measured") : "final_gate");
        receipt["correctness_status"] = complete && service_receipt.at("status") == "PASS" ? "PASS" : "FAIL";
        receipt["sustained_duration_status"] = long_enough && complete ? "PASS" : "NOT_RUN";
        receipt["latency_status"] = latency_pass && stats.points ? "PASS" : "FAIL";
        receipt["stop_reason"] = abort_reason;
        receipt["model"] = service_receipt.at("session"); receipt["service"] = service_receipt;
        receipt["mode"] = mode; receipt["frontend"] = "versioned typed C++ Frame API; no JSON decode or file encoding in points modes";
        receipt["source"] = source == "synthetic" ? Json{{"kind", "deterministic-synthetic"}, {"seed", seed}} :
            Json{{"kind", "licensed-laz-derived"}, {"basename", fs::path(source).filename().string()}, {"sha256", sha256(source)}};
        receipt["timestamp_scope"] = "Deterministic emulated acquisition UNIX timestamps projected onto monotonic replay time; original LAS GPS-time not asserted";
        receipt["requested_rate_points_per_second"] = rate; receipt["frame_points"] = count;
        receipt["warmup_seconds"] = double(warmup_frames * count) / rate;
        receipt["measured_seconds_planned"] = measured_duration;
        receipt["measured_seconds_completed"] = complete ? measured_duration :
            (finished > measurement_start ? (finished - measurement_start) / 1e9 : 0.0);
        receipt["measurement_including_drain_seconds"] = complete ? measured_duration + drain_seconds : receipt.at("measured_seconds_completed").get<double>();
        const auto denominator = receipt.at("measurement_including_drain_seconds").get<double>();
        receipt["incoming_points_per_second"] = denominator ? measured_incoming_points / denominator : 0;
        const auto inferred_total = service_receipt.at("freshly_classified_points").get<std::uint64_t>();
        const auto inferred_warmup = std::min(inferred_total, warmup_frames * count);
        receipt["freshly_neural_classified_points_per_second"] = denominator ? (inferred_total - inferred_warmup) / denominator : 0;
        receipt["fully_classified_emitted_points_per_second"] = denominator ? stats.points / denominator : 0;
        receipt["measured_incoming_points"] = measured_incoming_points;
        receipt["semantic_map_updates_per_second"] = denominator ? stats.map_updates / denominator : 0;
        receipt["measured_source_points"] = stats.points; receipt["expected_measured_source_points"] = measured_frames * count;
        receipt["classification_coverage"] = measured_incoming_points ? double(stats.points) / measured_incoming_points : 0;
        receipt["measurement_completion_fraction"] = double(stats.points) / (measured_frames * count);
        receipt["duplicated_or_corrupt_output_points"] = stats.identity_errors;
        receipt["delayed_points_over_latency_budget"] = stats.delayed_points;
        receipt["source_to_output_latency"] = stats.source_age.json(); receipt["inference_window_latency"] = stats.inference.json();
        receipt["input_queue_wait"] = stats.queue_wait.json(); receipt["output_consumer_latency"] = stats.sink_latency.json();
        receipt["maximum_pending_frames"] = maximum_pending.load(); receipt["maximum_oldest_pending_source_age_seconds"] = maximum_oldest_ns.load() / 1e9;
        receipt["maximum_schedule_lag_seconds"] = maximum_schedule_lag;
        receipt["source_point_latency_budget_seconds"] = latency_ns / 1e9; receipt["source_schedule_lag_budget_seconds"] = schedule_ns / 1e9;
        receipt["queue_samples"] = samples;
        receipt["host_initial_rss_bytes"] = initial_rss; receipt["host_peak_rss_bytes"] = rss();
        receipt["measured_rss_range_bytes"] = memory_growth;
        receipt["memory_stability_status"] = memory_stable ? "PASS" : "FAIL";
        receipt["map_cells"] = semantic_map.size(); receipt["map_capacity"] = map_cells; receipt["map_updates"] = stats.map_updates;
        receipt["point_output_digest"] = stats.digest; receipt["map_delta_digest"] = stats.deltas_digest;
        receipt["stage_seconds_totals"] = stats.stages;
        const auto residual = stats.stages["window_seconds"] - stats.stages["cpu_geometry_seconds"] -
            stats.stages["h2d_and_zero_seconds"] - stats.stages["cuda_preparation_seconds"] -
            stats.stages["tensorrt_seconds"] - stats.stages["cuda_scatter_seconds"] - stats.stages["d2h_seconds"];
        receipt["stage_seconds_totals"]["host_adapter_orchestration_sync_residual"] = std::max(0.0, residual);
        receipt["stage_seconds_totals"]["independent_map_oracle"] = stats.map_oracle_seconds;
        receipt["stage_seconds_totals"]["semantic_map"] = stats.map_seconds;
        receipt["stage_seconds_totals"]["semantic_map_validation_subset"] = stats.map_validation_seconds;
        receipt["stage_seconds_totals"]["point_consumer_identity_digest"] = stats.consume_seconds;
        receipt["stage_seconds_totals"]["optional_json_encoding"] = stats.encode_seconds;
        receipt["source_generation_seconds_including_warmup"] = generation_seconds;
        receipt["json_frontend_seconds"] = json_frontend_seconds;
        receipt["optional_file_io"] = "NOT_RUN; JSON mode encodes outside inference worker and counts bytes without disk I/O";
        receipt["optional_encoded_bytes"] = stats.encoded_bytes;
        receipt["initialization_wall_seconds"] = initialization_wall; receipt["source_decode_seconds_before_replay"] = decode_seconds;
        receipt["cold_first_window"] = stats.cold_first_window;
        receipt["classes_reused_points"] = 0; receipt["geometry_only_points"] = 0; receipt["spatial_source_overlap_percent"] = overlap;
        receipt["arrival_jitter_maximum_microseconds"] = jitter_us;
        receipt["motion_x_amplitude_metres"] = motion_mm / 1000.0;
        receipt["occlusion_alternating_rank_x_percent"] = occlusion_percent;
        write_new(receipt_path, receipt.dump(2) + "\n");
        std::cout << Json{{"status", receipt["status"]}, {"rate", rate}, {"classified_points", stats.points}, {"stop_reason", abort_reason}}.dump() << '\n';
        return receipt.at("status") == "PASS" ? 0 : 1;
    } catch (const std::exception &e) {
        receipt["error"] = e.what();
        if (!receipt_path.empty() && !fs::exists(receipt_path)) write_new(receipt_path, receipt.dump(2) + "\n");
        std::cerr << receipt.dump() << '\n'; return 1;
    }
}
