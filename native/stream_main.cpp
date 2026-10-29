// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Tuomas Pietilä
#include "synrex/service.hpp"
#include <algorithm>
#include <cerrno>
#include <csignal>
#include <cmath>
#include <sys/resource.h>
#include <fcntl.h>
#include <fstream>
#include <iostream>
#include <unistd.h>
namespace {
using namespace synrex;
using namespace synrex::streaming;
volatile std::sig_atomic_t interrupted{};
void signal_handler(int) { interrupted = 1; }
class Output {
    int fd_;
  public:
    explicit Output(const fs::path &p) : fd_(::open(p.c_str(), O_WRONLY | O_CREAT | O_EXCL | O_NOFOLLOW, 0600)) {
        require(fd_ >= 0, "output must be new and parent must exist");
    }
    ~Output() { if (fd_ >= 0) ::close(fd_); }
    void write(const Json &j) {
        const auto bytes = j.dump() + "\n";
        std::size_t n = 0;
        while (n < bytes.size()) {
            const auto count = ::write(fd_, bytes.data() + n, bytes.size() - n);
            if (count < 0 && errno == EINTR) continue;
            require(count > 0, "event output write failed");
            n += static_cast<std::size_t>(count);
        }
    }
    void finish() {
        require(::fsync(fd_) == 0, "event output sync failed");
        const auto rc = ::close(fd_); fd_ = -1;
        require(rc == 0, "event output close failed");
    }
};
std::size_t number(const std::string &s, std::size_t cap) {
    require(!s.empty() && std::all_of(s.begin(), s.end(), [](char c) { return c >= '0' && c <= '9'; }),
            "unsigned CLI integer required");
    const auto n = std::stoull(s);
    require(n > 0 && n <= cap, "CLI integer budget exceeded");
    return n;
}
Json counters(const AssemblyCounters &c) {
    return {{"incoming_points", c.incoming_points}, {"assembled_points", c.assembled_points},
            {"dropped_points", c.dropped_points}, {"late_points", c.late_points},
            {"partial_frames", c.partial_frames}, {"rejected_chunks", c.rejected_chunks},
            {"missing_sequences", c.missing_sequences}};
}
}
int main(int argc, char **argv) {
    using namespace synrex;
    using namespace synrex::streaming;
    try {
        require(argc >= 2, "usage: synrex-stream fixture|validate|replay --key VALUE (see docs/streaming/QUICKSTART.md)");
        const std::string cmd = argv[1];
        std::map<std::string, std::string> args;
        const std::set<std::string> allowed = cmd == "fixture" ?
            std::set<std::string>{"--out", "--seed", "--frames", "--points", "--chunk", "--taxonomy"} :
            cmd == "validate" ? std::set<std::string>{"--input"} :
            cmd == "replay" ? std::set<std::string>{"--input", "--model", "--out", "--receipt", "--queue", "--map-cells", "--map-policy", "--policy", "--rate-pps"} :
            std::set<std::string>{};
        require(!allowed.empty(), "unknown stream command");
        for (int i = 2; i < argc; i += 2) {
            require(i + 1 < argc && allowed.count(argv[i]) && args.emplace(argv[i], argv[i + 1]).second,
                    "unknown/duplicate/missing CLI option");
        }
        auto get = [&](const std::string &key, const std::string &fallback = "") {
            if (args.count(key)) return args.at(key);
            require(!fallback.empty(), "required option: " + key); return fallback;
        };
        if (cmd == "fixture") {
            const auto chunks = synthetic(number(get("--seed", "7"), UINT32_MAX),
                number(get("--frames", "4"), 1000), number(get("--points", "1024"), 65536),
                number(get("--chunk", "256"), 4096), get("--taxonomy", "synrex.fractal7.v1"));
            std::string bytes;
            for (const auto &c : chunks) bytes += serialize(c) + "\n";
            write_new(get("--out"), bytes);
            std::cout << Json{{"status", "PASS"}, {"scope", "deterministic_fixture_only"},
                              {"sha256", sha256(get("--out"))}, {"chunks", chunks.size()}}.dump(2) << '\n';
            return 0;
        }
        Limits limits;
        Assembler assembler(limits);
        std::ifstream in(get("--input"));
        require(bool(in), "cannot open source stream");
        std::string line;
        if (cmd == "validate") {
            while (read_line(in, line, limits)) (void)assembler.accept(parse(line, limits), steady_ns());
            assembler.finish();
            const auto c = assembler.counters();
            const bool pass = c.incoming_points > 0 && c.dropped_points == 0 && c.partial_frames == 0 && c.missing_sequences == 0;
            std::cout << Json{{"status", pass ? "PASS" : "FAIL"}, {"scope", "CPU_protocol_only"},
                              {"assembly", counters(c)}, {"GPU_INFERENCE", "NOT_RUN"}}.dump(2) << '\n';
            return pass ? 0 : 1;
        }
        const auto receipt_path = get("--receipt");
        require(!fs::exists(receipt_path), "receipt must be new");
        const auto output_path = get("--out");
        require(fs::absolute(receipt_path).lexically_normal() != fs::absolute(output_path).lexically_normal(),
                "receipt and events must have distinct paths");
        const auto policy = get("--policy", "block");
        require(policy == "block" || policy == "reject-new", "unsupported backpressure policy");
        const auto requested_rate = args.count("--rate-pps") ? number(args.at("--rate-pps"), 1000000000) : 0;
        const auto started = steady_ns();
        auto session = create_session(get("--model"), limits.max_frame_points);
        Output output(output_path);
        SemanticMap map(1.0, number(get("--map-cells", "10000"), 1000000), false, map_policy(get("--map-policy", "indexed")));
        std::vector<double> latency;
        std::uint64_t emitted_frames = 0;
        double map_seconds = 0, parse_assembly_seconds = 0;
        std::uint64_t map_updates = 0;
        Service service(std::move(session), number(get("--queue", "2"), 1024),
            [&](const Classification &r) {
                output.write(event(r));
                const auto map_start = steady_ns();
                const auto deltas = map.update(r);
                map_seconds += (steady_ns() - map_start) / 1e9;
                for (const auto &d : deltas) {
                    Json delta{{"schema", "synrex.semantic_delta.v1"}, {"mode", "map_delta_output"},
                        {"sensor_id", r.frame.header.sensor_id}, {"coordinate_frame", r.frame.header.coordinate_frame},
                        {"crs", r.frame.header.crs}, {"stream_id", r.frame.header.stream_id}, {"taxonomy", r.frame.header.taxonomy},
                        {"key", d.key}, {"semantic_id", d.semantic_id}, {"timestamp_ns", d.timestamp_ns},
                        {"confidence", nullptr}, {"confidence_age_ns", nullptr},
                        {"classification_age_ns_at_frame_end", r.frame.header.frame_end_ns - d.timestamp_ns},
                        {"evicted", d.evicted ? Json(*d.evicted) : Json(nullptr)}};
                    output.write(delta);
                }
                map_updates += deltas.size();
                const auto sample = (steady_ns() - r.frame.accepted_steady_ns) / 1e9;
                if (latency.size() < 4096) latency.push_back(sample);
                else latency[emitted_frames % 4096] = sample;
                ++emitted_frames;
            });
        const auto replay_started = steady_ns();
        std::uint64_t scheduled_points = 0;
        double max_schedule_lag_seconds = 0;
        std::signal(SIGINT, signal_handler); std::signal(SIGTERM, signal_handler);
        std::exception_ptr error;
        try {
            while (!interrupted && read_line(in, line, limits)) {
                const auto parse_start = steady_ns();
                auto chunk = parse(line, limits);
                parse_assembly_seconds += (steady_ns() - parse_start) / 1e9;
                scheduled_points += chunk.points.size();
                if (requested_rate) {
                    const auto due = replay_started + static_cast<std::uint64_t>(scheduled_points * (1e9 / requested_rate));
                    // Short interruptible pacing intervals; no unbounded sleeping producer.
                    while (!interrupted) {
                        const auto now = steady_ns();
                        if (now >= due) break;
                        std::this_thread::sleep_for(std::chrono::nanoseconds(std::min<std::uint64_t>(due - now, 10000000)));
                    }
                    max_schedule_lag_seconds = std::max(max_schedule_lag_seconds,
                        static_cast<double>(steady_ns() > due ? steady_ns() - due : 0) / 1e9);
                }
                if (interrupted) break;
                const auto assemble_start = steady_ns();
                auto frame = assembler.accept(std::move(chunk), assemble_start);
                parse_assembly_seconds += (steady_ns() - assemble_start) / 1e9;
                if (frame) (void)service.submit(std::move(*frame), policy == "block");
            }
            assembler.finish();
            if (interrupted) service.cancel();
            service.finish();
            output.finish();
        } catch (...) { error = std::current_exception(); assembler.finish(); service.cancel(); }
        // finish always joins before reporting state or reading sink-owned metrics.
        try { service.finish(); } catch (...) { if (!error) error = std::current_exception(); }
        const auto duration = (steady_ns() - started) / 1e9;
        auto receipt = service.receipt();
        const auto c = assembler.counters();
        receipt["schema"] = "synrex.streaming_replay.v1";
        receipt["assembly"] = counters(c); receipt["backpressure_policy"] = policy;
        receipt["wall_seconds_including_initialization_and_output"] = duration;
        receipt["incoming_points_per_second"] = c.incoming_points / duration;
        receipt["freshly_classified_points_per_second"] = receipt.at("freshly_classified_points").get<double>() / duration;
        receipt["map_updates"] = map_updates; receipt["map_updates_per_second"] = map_updates / duration;
        receipt["map_resident_voxels"] = map.size();
        receipt["cpu_map_update_seconds"] = map_seconds;
        receipt["cpu_parse_assembly_seconds"] = parse_assembly_seconds;
        struct rusage usage{}; require(::getrusage(RUSAGE_SELF, &usage) == 0, "host memory accounting failed");
        receipt["host_peak_rss_bytes"] = static_cast<std::uint64_t>(usage.ru_maxrss) * 1024;
        receipt["latency_scope"] = "Nearest-rank percentiles of latest at most 4096 complete frames; monotonic first-chunk acceptance through sink writes";
        std::sort(latency.begin(), latency.end());
        auto percentile = [&](double p) -> Json {
            return latency.empty() ? Json(nullptr) : Json(latency[static_cast<std::size_t>(std::ceil(latency.size() * p)) - 1]);
        };
        receipt["acceptance_to_output_seconds"] = {{"p50", percentile(.50)}, {"p95", percentile(.95)}, {"p99", percentile(.99)}};
        receipt["source_timestamp_to_wall_age"] = "NOT_RUN: offline historical UNIX timestamps; no wall-clock replay pacing";
        receipt["requested_incoming_points_per_second"] = requested_rate ? Json(requested_rate) : Json(nullptr);
        receipt["max_replay_schedule_lag_seconds"] = max_schedule_lag_seconds;
        const auto replay_seconds = (steady_ns() - replay_started) / 1e9;
        receipt["replay_seconds_excluding_initialization"] = replay_seconds;
        receipt["replay_incoming_points_per_second"] = c.incoming_points / replay_seconds;
        receipt["replay_freshly_classified_points_per_second"] = receipt.at("freshly_classified_points").get<double>() / replay_seconds;
        receipt["replay_map_updates_per_second"] = map_updates / replay_seconds;
        receipt["rate_scope"] = requested_rate ?
            "Short point-count-paced offline replay; schedule lag explicit; not full B5 sustainable sensor qualification" :
            "Unpaced replay wall rates including initialization/output; not sustainable live sensor rates";
        if (error || c.incoming_points == 0 || c.dropped_points || c.partial_frames || c.missing_sequences ||
            c.incoming_points != receipt.at("emitted_points").get<std::uint64_t>()) receipt["status"] = "FAIL";
        receipt["input_sha256"] = sha256(get("--input")); receipt["events_sha256"] = sha256(output_path);
        write_new(receipt_path, receipt.dump(2) + "\n");
        std::cout << receipt.dump(2) << '\n';
        if (error) std::rethrow_exception(error);
        return receipt.at("status") == "PASS" ? 0 : 1;
    } catch (const std::exception &e) {
        std::cerr << Json{{"status", "FAIL"}, {"error", e.what()}}.dump() << '\n';
        return 1;
    }
}
