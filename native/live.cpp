// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Tuomas Pietilä
// Native external receiver, deterministic producer, and independent output consumer.
#include "synrex/benchmark.hpp"
#include "synrex/export.hpp"
#include "synrex/map_oracle.hpp"
#include "synrex/transport.hpp"
#include <algorithm>
#include <atomic>
#include <csignal>
#include <cstring>
#include <fstream>
#include <iostream>
#include <openssl/evp.h>
#include <sys/resource.h>
#include <unistd.h>
using namespace synrex;
using namespace synrex::streaming;
namespace w = synrex::streaming::wire;
namespace {
volatile std::sig_atomic_t stopped{};
void stop(int) { stopped = 1; }
struct Hash {
    EVP_MD_CTX *ctx = EVP_MD_CTX_new();
    Hash() { require(ctx && EVP_DigestInit_ex(ctx, EVP_sha256(), nullptr) == 1, "digest init"); }
    ~Hash() { EVP_MD_CTX_free(ctx); }
    void add(const w::Bytes &b) {
        require(EVP_DigestUpdate(ctx, b.data(), b.size()) == 1, "digest update");
    }
    std::string finish() {
        unsigned char b[32];
        unsigned n;
        require(EVP_DigestFinal_ex(ctx, b, &n) == 1 && n == 32, "digest finalize");
        static const char *hex = "0123456789abcdef";
        std::string s;
        for (auto x : b) {
            s += hex[x >> 4];
            s += hex[x & 15];
        }
        return s;
    }
};
std::uint64_t rss() {
    rusage r{};
    require(getrusage(RUSAGE_SELF, &r) == 0, "RSS query");
    return std::uint64_t(r.ru_maxrss) * 1024;
}
std::uint64_t current_rss() {
    std::ifstream f("/proc/self/statm");
    std::uint64_t total, resident;
    require(bool(f >> total >> resident), "RSS current");
    return resident * sysconf(_SC_PAGESIZE);
}
struct Args {
    std::map<std::string, std::string> a;
    Args(int argc, char **argv) {
        const std::set<std::string> allowed{"--receipt",        "--model",
                                            "--source",         "--port",
                                            "--output-port",    "--rate",
                                            "--seconds",        "--warmup",
                                            "--points",         "--map-cells",
                                            "--churn-mm",       "--geometry",
                                            "--fragment",       "--sensor",
                                            "--stream",         "--stall-ms",
                                            "--frames",         "--reconnect",
                                            "--fault",          "--export-bundle",
                                            "--taxonomy",       "--export-after-frame",
                                            "--validation-mode"};
        for (int i = 2; i < argc; i += 2)
            require(i + 1 < argc && allowed.count(argv[i]) &&
                        a.emplace(argv[i], argv[i + 1]).second,
                    "invalid live option");
    }
    std::string get(const std::string &k, const std::string &d = "") const {
        auto i = a.find(k);
        if (i != a.end())
            return i->second;
        require(!d.empty(), "missing " + k);
        return d;
    }
    std::uint64_t n(const std::string &k, std::uint64_t d, std::uint64_t max) const {
        auto s = get(k, std::to_string(d));
        require(!s.empty() &&
                    std::all_of(s.begin(), s.end(), [](char c) { return c >= '0' && c <= '9'; }),
                "unsigned live argument");
        auto x = std::stoull(s);
        require(x <= max, "live argument budget");
        return x;
    }
};
w::LegacyLiveLimits live_limits(const Args &a) {
    w::LegacyLiveLimits limits;
    limits.map_cells = a.n("--map-cells", w::max_legacy_map_cells, UINT64_MAX);
    limits.points = a.n("--points", 16384, 65536);
    limits.rate = a.n("--rate", 1000000, 20000000);
    limits.validate();
    return limits;
}
struct Replay {
    std::vector<Point> corpus;
    std::size_t count;
    std::uint64_t rate, warm, total, churn;
    std::string taxonomy, sensor, stream;
    Replay(const Args &a)
        : count(a.n("--points", 16384, 65536)), rate(a.n("--rate", 1000000, 20000000)),
          churn(a.n("--churn-mm", 0, 1000000)), sensor(a.get("--sensor", "replay")),
          stream(a.get("--stream", "r1")) {
        Package p(a.get("--model"));
        taxonomy =
            p.contract.las_classes.size() == 7 ? "synrex.fractal7.v1" : "synrex.synthetic3.v1";
        corpus = decode(a.get("--source"), p.contract);
        require(count && rate >= count && corpus.size() >= count, "replay size/rate budget");
        warm = (a.n("--warmup", 5, 30) * rate + count - 1) / count;
        const auto duration = a.n("--seconds", 60, 1800);
        require(duration > 0, "positive duration required");
        total = warm + (duration * rate + count - 1) / count;
        if (a.a.count("--frames")) {
            total = a.n("--frames", 1, 500000);
            require(total > 0, "positive frames");
            warm = 0;
        }
    }
    w::Input frame(std::uint64_t f, std::uint64_t anchor) const {
        w::Input in;
        auto &x = in.frame;
        auto &h = x.header;
        h.sensor_id = sensor;
        h.stream_id = stream;
        h.coordinate_frame = "replay_world";
        h.crs = "LOCAL:replay_world";
        h.taxonomy = taxonomy;
        h.sequence = h.frame_id = f;
        h.frame_start_ns = 1700000000000000000ULL + f * count * 1000000000ULL / rate;
        h.frame_end_ns = 1700000000000000000ULL + (f * count + count - 1) * 1000000000ULL / rate;
        h.chunk_count = 1;
        h.frame_points = count;
        x.first_sequence = x.last_sequence = f;
        in.coordinates = w::Coordinates::world;
        in.replay_steady_ns = anchor ? anchor + (f + 1) * count * 1000000000ULL / rate : 0;
        x.points.reserve(count);
        for (std::size_t i = 0; i < count; ++i) {
            const auto index = (f * count + i) % corpus.size();
            auto xyz = corpus[index].xyz;
            xyz[0] += f * (churn / 1000.0);
            x.points.push_back({index,
                                1700000000000000000ULL + (f * count + i) * 1000000000ULL / rate,
                                xyz,
                                corpus[index].intensity,
                                {},
                                {}});
        }
        return in;
    }
};
void producer(const Args &a, Json &r) {
    Replay replay(a);
    auto sock = w::connect_loopback(a.n("--port", 39181, 65535));
    const auto anchor = steady_ns() + 100000000ULL;
    double lag = 0;
    Hash hash;
    const auto fragment = a.n("--fragment", 0, 1024 * 1024),
               reconnect = a.n("--reconnect", 0, 500000);
    const auto fault = a.get("--fault", "none");
    std::uint64_t points = 0;
    double encoding = 0, send_seconds = 0;
    for (std::uint64_t f = 0; f < replay.total; ++f) {
        require(!stopped, "producer cancelled");
        const auto start = steady_ns();
        auto in = replay.frame(f, anchor);
        if (f == 2 && fault == "duplicate")
            --in.frame.header.sequence;
        if (f == 2 && fault == "gap")
            ++in.frame.header.sequence;
        if (f == 2 && fault == "clock")
            in.frame.header.frame_start_ns -= replay.count * 1000000000ULL / replay.rate;
        in.frame.first_sequence = in.frame.last_sequence = in.frame.header.sequence;
        auto bytes = w::encode(in);
        if (f == 2 && fault == "truncated")
            bytes.resize(bytes.size() / 2);
        encoding += (steady_ns() - start) / 1e9;
        std::this_thread::sleep_until(
            std::chrono::steady_clock::time_point(std::chrono::nanoseconds(in.replay_steady_ns)));
        lag = std::max(lag, (steady_ns() - in.replay_steady_ns) / 1e9);
        require(lag <= .05, "producer scheduling lag exceeded 50ms");
        if (reconnect && f == reconnect) {
            sock.shutdown();
            sock = w::connect_loopback(a.n("--port", 39181, 65535));
        }
        const auto send_start = steady_ns();
        w::send(sock, w::Kind::frame, bytes, 500, fragment);
        send_seconds += (steady_ns() - send_start) / 1e9;
        hash.add(w::canonical(bytes, false));
        points += replay.count;
    }
    w::send(sock, w::Kind::end, w::json_bytes({{"frames", replay.total}, {"points", points}}));
    r.update(
        {{"status", "PASS"},
         {"incoming_points", points},
         {"frames", replay.total},
         {"warmup_frames", replay.warm},
         {"schedule_lag_max_seconds", lag},
         {"input_sha256", hash.finish()},
         {"encode_seconds", encoding},
         {"socket_send_seconds", send_seconds},
         {"source_sha256", sha256(a.get("--source"))},
         {"note", "External native TCP replay; not physical sensor latency or physical motion."}});
}
void receiver(const Args &a, Json &r) {
    const auto limits = live_limits(a);
    const auto validation_mode = a.get("--validation-mode", "decoded");
    require(validation_mode == "reference" || validation_mode == "decoded",
            "unknown validation strategy");
    const bool decoded_validation = validation_mode == "decoded";
    const auto start = steady_ns();
    SessionOptions options;
    options.geometry = geometry_mode(a.get("--geometry", "direct"));
    options.timeline = true;
    const auto count = limits.points, cells = limits.map_cells;
    auto session = create_session(a.get("--model"), count, options);
    r["initialization_seconds"] = (steady_ns() - start) / 1e9;
    const auto model_manifest_sha256 = session->identity().at("manifest_sha256").get<std::string>();
    auto listener = w::listen_loopback(a.n("--port", 39181, 65535));
    auto output_listener = w::listen_loopback(a.n("--output-port", 39182, 65535));
    std::cout << "live_ready=1" << std::endl;
    // Subscriber connects before producer. Only the dedicated writer performs socket I/O.
    auto subscriber = w::accept(output_listener, 30000);
    struct Delivery {
        w::Kind kind;
        w::Bytes bytes;
        std::uint64_t points;
    };
    Queue<Delivery> deliveries(2);
    std::atomic<std::uint64_t> delivered{0}, failed{0}, delivered_points{0}, failed_points{0};
    std::atomic<bool> output_dead{false}, snapshot_delivered{false};
    std::string output_error;
    std::thread writer([&] {
        try {
            while (auto item = deliveries.pop()) {
                if (output_dead) {
                    ++failed;
                    failed_points += item->points;
                    continue;
                }
                try {
                    w::send(subscriber, item->kind, item->bytes, 100);
                    if (item->kind == w::Kind::snapshot)
                        snapshot_delivered = true;
                    ++delivered;
                    delivered_points += item->points;
                } catch (const std::exception &e) {
                    output_error = e.what();
                    output_dead = true;
                    ++failed;
                    failed_points += item->points;
                    subscriber.shutdown();
                }
            }
        } catch (const std::exception &e) {
            output_error = e.what();
            output_dead = true;
        }
    });
    struct JoinWriter {
        Queue<Delivery> &queue;
        std::thread &thread;
        ~JoinWriter() {
            queue.close();
            if (thread.joinable())
                thread.join();
        }
    } join_writer{deliveries, writer};
    auto enqueue = [&](w::Kind kind, w::Bytes bytes, std::uint64_t count = 0,
                       bool terminal = false) {
        if (!deliveries.push({kind, std::move(bytes), count}, terminal)) {
            ++failed;
            failed_points += count;
            output_dead = true;
            subscriber.shutdown();
        }
    };
    SemanticMap map(1, cells, true);
    w::Sequence sequence;
    Hash input_hash, output_hash;
    const auto rate = limits.rate;
    const auto warm = (a.n("--warmup", a.a.count("--rate") ? 5 : 0, 30) * rate + count - 1) / count;
    std::uint64_t frames = 0, points = 0, measured_points = 0, updates = 0, executions = 0,
                  asof = 0, first = 0, last = 0, measured_updates = 0;
    Json map_measure_start;
    std::uint64_t rss_min = UINT64_MAX, rss_max = 0;
    LatencyHistogram age, receive_age, inference, queue;
    std::map<std::string, double> stages;
    double map_seconds = 0, encode_seconds = 0;
    Json timeline = Json::array();
    auto consume = [&](const Classification &c) {
        const auto begin = steady_ns();
        if (frames == warm)
            map_measure_start = map.counters().json();
        const auto deltas = map.update(c);
        map_seconds += (steady_ns() - begin) / 1e9;
        w::Output out;
        out.input.frame = c.frame;
        out.input.coordinates = static_cast<w::Coordinates>(c.frame.coordinate_domain);
        out.input.replay_steady_ns = c.frame.replay_steady_ns;
        out.semantic_ids = c.semantic_ids;
        out.las_codes = c.las_codes;
        out.deltas = deltas;
        out.received_ns = c.frame.accepted_steady_ns;
        out.published_ns = 0;
        out.model_executions = c.timings.at("completed_inference_blocks");
        out.model_manifest_sha256 = model_manifest_sha256;
        executions += out.model_executions;
        require(out.model_executions > 0, "fresh neural execution missing");
        ++frames; // Service FIFO; source frame identifiers need not start at zero.
        points += c.frame.points.size();
        updates += deltas.size();
        asof = std::max(asof, c.frame.header.frame_end_ns);
        const auto encoded = steady_ns();
        auto bytes = w::encode(out);
        encode_seconds += (steady_ns() - encoded) / 1e9;
        output_hash.add(w::canonical(bytes, true));
        out.published_ns = steady_ns();
        w::stamp_publication(bytes, out.published_ns);
        const bool measured = frames > warm;
        if (measured) {
            measured_points += c.frame.points.size();
            measured_updates += deltas.size();
            if (!first)
                first =
                    out.input.replay_steady_ns
                        ? out.input.replay_steady_ns - c.frame.points.size() * 1000000000ULL / rate
                        : out.received_ns;
            last = out.published_ns;
            if (out.input.replay_steady_ns) {
                require(out.published_ns >= out.input.replay_steady_ns,
                        "same-host replay deadline in future");
                const auto source_age = out.published_ns - out.input.replay_steady_ns +
                                        c.frame.points.size() * 1000000000ULL / rate;
                for (std::size_t i = 0; i < c.frame.points.size(); ++i)
                    age.add(source_age);
                require(source_age <= 500000000, "oldest replay source age exceeded 500ms");
            }
            for (std::size_t i = 0; i < c.frame.points.size(); ++i)
                receive_age.add(out.published_ns - out.received_ns);
            inference.add(std::uint64_t(c.timings.at("window_seconds").get<double>() * 1e9));
            queue.add(std::uint64_t(c.timings.at("input_queue_wait_seconds").get<double>() * 1e9));
            for (auto it = c.timings.begin(); it != c.timings.end(); ++it)
                if (it.value().is_number() && it.key().find("seconds") != std::string::npos)
                    stages[it.key()] += it.value().get<double>();
            const auto mem = current_rss();
            rss_min = std::min(rss_min, mem);
            rss_max = std::max(rss_max, mem);
            require(mem <= 1024ULL * 1024 * 1024 && rss_max - rss_min <= 64ULL * 1024 * 1024,
                    "receiver host memory gate");
        }
        if (timeline.size() < 2)
            timeline.push_back(c.timings);
        enqueue(w::Kind::classified, std::move(bytes), c.frame.points.size());
    };
    Service service(std::move(session), 2, consume);
    std::string error;
    double receive_wall_seconds = 0, decode_seconds = 0, validation_seconds = 0;
    std::uint64_t received_frames = 0, received_points = 0, reconnects = 0, wire_frames = 0,
                  decoded_points = 0;
    try {
        auto producer = w::accept(listener, 30000);
        while (true) {
            require(!stopped, "receiver cancelled");
            std::uint64_t received_at = 0;
            const auto recv_start = steady_ns();
            auto message = w::receive(producer, 10000, &received_at);
            receive_wall_seconds += (steady_ns() - recv_start) / 1e9;
            if (!message) {
                require(++reconnects <= 16, "reconnect budget");
                producer = w::accept(listener, 10000);
                continue;
            }
            if (message->first == w::Kind::end) {
                require(message->second.size() <= 1024, "end marker budget");
                auto end = w::json(message->second);
                require(end.is_object() && end.size() == 2 &&
                            end.at("frames").is_number_unsigned() &&
                            end.at("points").is_number_unsigned(),
                        "invalid end marker");
                require(end.at("frames") == received_frames && end.at("points") == received_points,
                        "producer end coverage");
                break;
            }
            require(message->first == w::Kind::frame, "receiver expects frame");
            ++wire_frames;
            const auto decode_start = steady_ns();
            auto in = decoded_validation ? sequence.decode_frame(message->second)
                                         : w::decode(message->second);
            decoded_points += in.frame.points.size();
            decode_seconds += (steady_ns() - decode_start) / 1e9;
            const auto validation_start = steady_ns();
            if (!decoded_validation)
                sequence.accept(in);
            require(in.frame.points.size() <= count, "receiver configured point budget");
            in.frame.accepted_steady_ns = received_at;
            in.frame.replay_steady_ns = in.replay_steady_ns;
            in.frame.coordinate_domain = unsigned(in.coordinates);
            const auto deadline = in.replay_steady_ns;
            input_hash.add(w::canonical(message->second, false));
            require(!deadline || (in.frame.accepted_steady_ns >= deadline &&
                                  in.frame.accepted_steady_ns - deadline <= 500000000),
                    "replay scheduling/receive age gate");
            validation_seconds += (steady_ns() - validation_start) / 1e9;
            ++received_frames;
            received_points += in.frame.points.size();
            require(service.submit(std::move(in.frame), false), "bounded input queue exhausted");
        }
        service.finish();
    } catch (const std::exception &e) {
        error = e.what();
        service.cancel();
        try {
            service.finish();
        } catch (...) {
        }
    }
    r.update(
        {{"service", service.receipt()},
         {"received_frames", received_frames},
         {"received_points", received_points},
         {"wire_frame_messages", wire_frames},
         {"unaccepted_decoded_points", decoded_points - received_points},
         {"reconnects", reconnects},
         {"output_points", points},
         {"freshly_classified_measured_points", measured_points},
         {"map_updates", updates},
         {"model_executions", executions},
         {"source_to_publication", age.json()},
         {"receiver_time_definition", "first successful envelope recv syscall; includes payload "
                                      "receive/decode; not physical packet/sensor arrival"},
         {"receive_to_publication", receive_age.json()},
         {"inference", inference.json()},
         {"input_queue_wait", queue.json()},
         {"map", map.counters().json()},
         {"map_profile", map.profile()},
         {"map_seconds", map_seconds},
         {"socket_receive_wall_seconds", receive_wall_seconds},
         {"cpu_wire_decode_seconds", decode_seconds},
         {"cpu_transport_validation_seconds", validation_seconds},
         {"output_encoding_seconds", encode_seconds},
         {"measured_stage_seconds", stages},
         {"timeline_first_windows", timeline},
         {"peak_rss_bytes", rss()},
         {"measured_rss_range_bytes", rss_min == UINT64_MAX ? 0 : rss_max - rss_min},
         {"input_sha256", input_hash.finish()},
         {"output_sha256", output_hash.finish()},
         {"warmup_frames", warm},
         {"error", error}});
    const auto &identity = r["service"]["session"];
    const bool fresh_execution_match = identity.at("successful_enqueue_calls") == executions &&
                                       identity.at("classified_source_points") == points &&
                                       identity.at("completed_windows") == frames;
    r["fresh_execution_count_match"] = fresh_execution_match;
    const double seconds = last > first ? (last - first) / 1e9 : 0;
    r["measured_seconds"] = seconds;
    r["measured_map_updates"] = measured_updates;
    r["map_updates_per_second"] = seconds ? measured_updates / seconds : 0;
    auto measured_ops = map.counters().json();
    if (!map_measure_start.is_null())
        for (auto it = measured_ops.begin(); it != measured_ops.end(); ++it)
            if (it.key() != "peak_cells")
                it.value() = it.value().get<std::uint64_t>() -
                             map_measure_start.at(it.key()).get<std::uint64_t>();
    r["measured_map_operations"] = measured_ops;
    r["freshly_classified_points_per_second"] = seconds ? measured_points / seconds : 0;
    const auto snapshot = map.snapshot(asof);
    auto snapshot_bytes = w::json_bytes(snapshot);
    Hash snapshot_hash;
    snapshot_hash.add(snapshot_bytes);
    r["map_snapshot_sha256"] = snapshot_hash.finish();
    enqueue(w::Kind::snapshot, std::move(snapshot_bytes), 0, true);
    r["status"] = fresh_execution_match && error.empty() && r["service"]["status"] == "PASS" &&
                          age.quantile(.99) <= .25 && received_points == points &&
                          measured_points > 0
                      ? "PASS"
                      : "FAIL";
    enqueue(w::Kind::end, w::json_bytes(r), 0, true);
    deliveries.close();
    writer.join();
    r["output_delivery"] = {{"delivered_messages", delivered.load()},
                            {"failed_messages", failed.load()},
                            {"complete_point_records_sent", delivered_points.load()},
                            {"failed_point_deliveries", failed_points.load()},
                            {"error", output_error},
                            {"required_snapshot_sent", snapshot_delivered.load()},
                            {"validation_scope", "socket sends; independent validation is in the subscriber receipt"},
                            {"queue", deliveries.stats()}};
    if (output_dead || failed || !snapshot_delivered)
        r["status"] = "FAIL";
    r["failure_phase"] = r["status"] == "PASS" ? "none" : measured_points ? "measured" : "warmup";
    r["stop_reason"] = r["status"] == "PASS" ? ""
                       : deliveries.stats().at("rejected_pushes").get<std::uint64_t>()
                           ? "bounded_output_delivery_queue"
                       : output_dead ? "subscriber_socket_delivery"
                                     : error;
    r["resources_after_service_before_destruction"] = session_resources();
}
void subscriber(const Args &a, Json &r) {
    const auto limits = live_limits(a);
    const auto validation_mode = a.get("--validation-mode", "decoded");
    require(validation_mode == "reference" || validation_mode == "decoded",
            "unknown validation strategy");
    const bool decoded_validation = validation_mode == "decoded";
    std::map<std::string, double> cpu_stages;
    std::unique_ptr<Replay> replay;
    if (a.a.count("--source"))
        replay = std::make_unique<Replay>(a);
    std::unique_ptr<AsyncLazExport> exporter;
    if (a.a.count("--export-bundle"))
        exporter = std::make_unique<AsyncLazExport>(a.get("--source"), a.get("--model"),
                                                    a.get("--taxonomy"), a.get("--export-bundle"));
    MapOracle oracle(limits.map_cells);
    auto sock = w::connect_loopback(a.n("--output-port", 39182, 65535));
    std::cout << "subscriber_ready=1" << std::endl;
    const auto stall = a.n("--stall-ms", 0, 60000);
    if (stall)
        std::this_thread::sleep_for(std::chrono::milliseconds(stall));
    const auto expected_model_hash = sha256(fs::path(a.get("--model")) / "manifest.json");
    Hash input_hash, output_hash;
    std::uint64_t frames = 0, points = 0, updates = 0, executions = 0, last_asof = 0;
    w::Sequence sequence;
    LatencyHistogram latency, receive_latency;
    struct Prefix {
        Json &r;
        std::uint64_t &frames, &points, &updates;
        LatencyHistogram &latency;
        ~Prefix() {
            r["validated_prefix"] = {{"frames", frames},
                                     {"points", points},
                                     {"map_updates", updates},
                                     {"source_to_subscriber", latency.json()}};
        }
    } prefix{r, frames, points, updates, latency};
    std::set<std::array<std::int64_t, 3>> active;
    auto oracle_size = [&](const Classification &, const std::vector<MapDelta> &deltas) {
        for (const auto &d : deltas) {
            if (d.evicted)
                require(active.erase(*d.evicted) == 1, "subscriber eviction absent");
            active.insert(d.key);
        }
        return active.size();
    };
    bool snapshot_seen = false, end_seen = false;
    Json snapshot;
    while (auto message = w::receive(sock, 15000)) {
        if (message->first == w::Kind::snapshot) {
            require(!snapshot_seen, "duplicate snapshot");
            snapshot = w::json(message->second);
            require(snapshot == oracle.snapshot(last_asof), "subscriber full snapshot parity");
            Hash snapshot_hash;
            snapshot_hash.add(message->second);
            r["map_snapshot_sha256"] = snapshot_hash.finish();
            snapshot_seen = true;
            continue;
        }
        if (message->first == w::Kind::end) {
            r["receiver"] = w::json(message->second);
            end_seen = true;
            break;
        }
        require(!snapshot_seen && message->first == w::Kind::classified,
                "subscriber output ordering");
        auto stage_start = steady_ns();
        auto out = decoded_validation ? sequence.decode_classified(message->second)
                                      : w::decode_output(message->second);
        require(out.model_manifest_sha256 == expected_model_hash, "subscriber exact model binding");
        if (!decoded_validation)
            sequence.accept(out.input);
        cpu_stages["decode_and_sequence"] += (steady_ns() - stage_start) / 1e9;
        stage_start = steady_ns();
        require((!replay || out.input.frame.header.frame_id == frames) &&
                    out.received_ns <= out.published_ns && out.published_ns <= steady_ns(),
                "subscriber identity/clock");
        const auto deadline = out.input.replay_steady_ns;
        const auto canonical_input = w::canonical(message->second, true, true);
        if (replay)
            require(canonical_input == w::encode(replay->frame(frames, 0)),
                    "subscriber original input byte parity");
        input_hash.add(canonical_input);
        Classification c{out.input.frame, out.semantic_ids, out.las_codes, {}};
        cpu_stages["expected_source_and_class_validation"] += (steady_ns() - stage_start) / 1e9;
        stage_start = steady_ns();
        if (decoded_validation)
            oracle.verify(c, out.deltas);
        else
            oracle.verify(c, out.deltas, oracle_size(c, out.deltas));
        cpu_stages["independent_map_oracle"] += (steady_ns() - stage_start) / 1e9;
        stage_start = steady_ns();
        if (exporter && frames >= a.n("--export-after-frame", 0, 500000))
            exporter->observe(c);
        require(out.model_executions > 0, "subscriber missing fresh execution");
        executions += out.model_executions;
        output_hash.add(w::canonical(message->second, true));
        const auto now = steady_ns();
        cpu_stages["canonical_output_hash_and_export_observe"] += (now - stage_start) / 1e9;
        if (replay && frames >= replay->warm) {
            require(deadline && now >= deadline, "subscriber replay time");
            for (std::size_t i = 0; i < c.frame.points.size(); ++i) {
                latency.add(now - deadline +
                            (c.frame.points.size() - i) * 1000000000ULL / replay->rate);
                receive_latency.add(now - out.received_ns);
            }
        }
        last_asof = c.frame.header.frame_end_ns;
        ++frames;
        points += c.frame.points.size();
        updates += out.deltas.size();
    }
    if (exporter)
        r["async_laz_export"] = exporter->finish();
    require(end_seen && snapshot_seen && frames > 0 && points > 0 && executions > 0 &&
                r["receiver"]["output_points"] == points &&
                r["receiver"]["received_frames"] == frames,
            "subscriber complete output coverage");
    require(r["map_snapshot_sha256"] == r["receiver"]["map_snapshot_sha256"],
            "subscriber/receiver full snapshot digest parity");
    if (replay)
        require(frames == replay->total && points == replay->total * replay->count,
                "subscriber expected replay coverage");
    r.update({{"status",
               latency.quantile(.99) <= .25 && r["receiver"]["status"] == "PASS" ? "PASS" : "FAIL"},
              {"frames", frames},
              {"points", points},
              {"map_updates", updates},
              {"model_executions", executions},
              {"input_sha256", input_hash.finish()},
              {"output_sha256", output_hash.finish()},
              {"source_to_subscriber", latency.json()},
              {"receive_to_subscriber", receive_latency.json()},
              {"map_oracle", "PASS"},
              {"required_snapshot_validated", true},
              {"validation_mode", validation_mode},
              {"cpu_stage_seconds", cpu_stages},
              {"peak_rss_bytes", rss()},
              {"coverage", 1.0},
              {"reused_semantic_labels", 0}});
    require(r["input_sha256"] == r["receiver"]["input_sha256"] &&
                r["output_sha256"] == r["receiver"]["output_sha256"],
            "subscriber/receiver digest parity");
}
} // namespace
int main(int argc, char **argv) {
    Json r{{"schema", "synrex.streaming_r1.live.v1"}, {"status", "FAIL"}};
    fs::path receipt;
    try {
        require(argc >= 2, "usage: synrex-live receive|send|subscribe OPTIONS");
        Args a(argc, argv);
        receipt = a.get("--receipt");
        require(!fs::exists(receipt), "receipt must be new");
        std::signal(SIGINT, stop);
        std::signal(SIGTERM, stop);
        r["mode"] = argv[1];
        if (std::string(argv[1]) == "receive")
            receiver(a, r);
        else if (std::string(argv[1]) == "send")
            producer(a, r);
        else if (std::string(argv[1]) == "subscribe")
            subscriber(a, r);
        else
            require(false, "unknown live command");
    } catch (const std::exception &e) {
        r["status"] = "FAIL";
        r["error"] = e.what();
    }
    r["resources_at_exit"] = session_resources();
    r["peak_rss_bytes"] = rss();
    if (!receipt.empty() && !fs::exists(receipt))
        write_new(receipt, r.dump(2) + "\n");
    std::cout << r.dump(2) << '\n';
    return r["status"] == "PASS" ? 0 : 1;
}
