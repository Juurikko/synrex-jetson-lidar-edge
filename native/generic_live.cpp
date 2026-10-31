// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Tuomas Pietilä
#include "synrex/generic_live.hpp"
#include "synrex/generic_export.hpp"
#include "synrex/generic_output.hpp"
#include "synrex/generic_replay.hpp"
#include "synrex/generic_session.hpp"
#include "synrex/transport.hpp"
#include <algorithm>
#include <cerrno>
#include <cmath>
#include <csignal>
#include <fcntl.h>
#include <fstream>
#include <iostream>
#include <linux/memfd.h>
#include <set>
#include <sys/stat.h>
#include <sys/syscall.h>
#include <thread>
#include <unistd.h>

namespace synrex::generic {
namespace {
namespace wire = streaming::wire;
volatile std::sig_atomic_t interrupted{};
void interrupt(int) { interrupted = 1; }
void active() { require(!interrupted, "generic live cancelled"); }
constexpr std::uint64_t max_records = 1000000;
std::uint64_t steady_ns() {
    return std::chrono::duration_cast<std::chrono::nanoseconds>(
               Startup::Clock::now().time_since_epoch())
        .count();
}
struct Args {
    std::string mode;
    std::map<std::string, std::string> values;
    Args(int argc, char **argv) {
        require(argc >= 2,
                "usage: synrex-generic-live receive|send|subscribe --wire binary-v3 OPTIONS");
        mode = argv[1];
        require(mode == "receive" || mode == "send" || mode == "subscribe",
                "unknown generic live command");
        std::set<std::string> allowed{"--wire",        "--model",      "--receipt",
                                      "--session",     "--epoch",      "--points",
                                      "--frame-bytes", "--startup-ms", "--io-ms"};
        const std::set<std::string> extra =
            mode == "send" ? std::set<std::string>{"--source", "--port", "--rate"}
            : mode == "receive"
                ? std::set<std::string>{"--port",
                                        "--output-port",
                                        "--frames",
                                        "--records",
                                        "--source-sha256",
                                        "--map-cells",
                                        "--tensor-trace-root",
                                        "--tensor-trace-bytes",
                                        "--trace-source-commit"}
                : std::set<std::string>{"--port",          "--frames",    "--records",
                                        "--source-sha256", "--map-cells", "--source",
                                        "--export-bundle"};
        allowed.insert(extra.begin(), extra.end());
        for (int i = 2; i < argc; i += 2)
            require(i + 1 < argc && allowed.count(argv[i]) &&
                        values.emplace(argv[i], argv[i + 1]).second,
                    "unknown/duplicate/missing generic live option");
    }
    std::string get(const std::string &key, const std::string &fallback = "") const {
        const auto found = values.find(key);
        if (found != values.end())
            return found->second;
        require(!fallback.empty(), "missing " + key);
        return fallback;
    }
    std::uint64_t number(const std::string &key, std::uint64_t fallback,
                         std::uint64_t maximum) const {
        const auto text = get(key, std::to_string(fallback));
        require(!text.empty() && std::all_of(text.begin(), text.end(),
                                             [](char c) { return c >= '0' && c <= '9'; }),
                "unsigned generic live argument required");
        const auto value = std::stoull(text);
        require(value > 0 && value <= maximum, "generic live argument budget: " + key);
        return value;
    }
};
void new_path(const fs::path &path) {
    require(!path.empty() && !fs::exists(path) && !fs::is_symlink(path) &&
                fs::is_directory(fs::absolute(path).parent_path()),
            "output must be new in an existing parent");
}
void hex_digest(const std::string &value) {
    require(value.size() == 64 && std::all_of(value.begin(), value.end(),
                                              [](unsigned char c) {
                                                  return (c >= '0' && c <= '9') ||
                                                         (c >= 'a' && c <= 'f');
                                              }),
            "exact lowercase source SHA256 required");
}
struct Config {
    const Args &args;
    const Limits limits;
    const std::uint64_t epoch, io_ms, records, frames;
    const std::uint16_t port, output_port;
    const std::size_t cells;
    const std::string source_sha;
    DiagnosticTraceOptions trace;
    explicit Config(const Args &a)
        : args(a),
          limits{a.number("--points", 16384, 65536), a.number("--frame-bytes", 1048576, 16777216),
                 a.number("--startup-ms", 30000, 120000)},
          epoch(a.number("--epoch", 1, UINT64_MAX)), io_ms(a.number("--io-ms", 10000, 120000)),
          records(a.mode == "send" ? 0 : a.number("--records", 0, max_records)),
          frames(a.mode == "send" ? 0 : a.number("--frames", 0, max_records)),
          port(a.number("--port", 39183, 65535)),
          output_port(a.mode == "receive" ? a.number("--output-port", 39184, 65535) : 0),
          cells(a.mode == "send" ? 1 : a.number("--map-cells", 65536, SemanticMap::max_cells)),
          source_sha(a.mode == "send" ? "" : a.get("--source-sha256")) {
        require(a.get("--wire") == "binary-v3", "explicit --wire binary-v3 required");
        limits.validate();
        // Includes the maximum accepted identifier lengths, labels and flat
        // timing metadata. Admission cannot promise a frame that cannot return.
        require(limits.max_frame_bytes >= 17676 + 50 * limits.max_frame_points,
                "generic CLI worst-case classification exceeds frame byte budget");
        if (a.mode != "send") {
            hex_digest(source_sha);
            require(frames <= records &&
                        frames == (records + limits.max_frame_points - 1) / limits.max_frame_points,
                    "expected finite frame/record counts disagree");
        }
        if (a.mode == "receive")
            require(port != output_port, "input/output ports must differ");
        if (a.values.count("--tensor-trace-root")) {
            trace.root = a.get("--tensor-trace-root");
            trace.total_bytes = a.number("--tensor-trace-bytes", 0, 256ULL * 1024 * 1024);
            trace.windows = frames;
            trace.declared_source_commit = a.get("--trace-source-commit");
        } else
            require(!a.values.count("--tensor-trace-bytes") &&
                        !a.values.count("--trace-source-commit"),
                    "diagnostic trace options require root");
        trace.validate();
        if (a.values.count("--export-bundle")) {
            (void)a.get("--source");
            new_path(a.get("--export-bundle"));
        }
    }
    Startup::Time deadline() const {
        active();
        return Startup::Clock::now() + std::chrono::milliseconds(io_ms);
    }
};
struct OwnedFd {
    int value;
    explicit OwnedFd(int fd) : value(fd) { require(fd >= 0, "source open/capture failed"); }
    ~OwnedFd() { ::close(value); }
    OwnedFd(const OwnedFd &) = delete;
    OwnedFd &operator=(const OwnedFd &) = delete;
};
struct Source {
    OwnedFd frozen;
    std::vector<Point> points;
    std::string digest;
    Source(const fs::path &path, const Package &package)
        : frozen(static_cast<int>(::syscall(SYS_memfd_create, "synrex-replay-source",
                                            MFD_CLOEXEC | MFD_ALLOW_SEALING))) {
        OwnedFd input(::open(path.c_str(), O_RDONLY | O_CLOEXEC | O_NOFOLLOW | O_NONBLOCK));
        struct stat initial {
        }, final_fd{}, final_path{};
        require(::fstat(input.value, &initial) == 0 && S_ISREG(initial.st_mode) &&
                    initial.st_size > 0 && initial.st_size <= 256LL * 1024 * 1024,
                "single regular source file must be nonempty and <=256MiB");
        std::array<char, 65536> buffer{};
        auto remaining = initial.st_size;
        while (remaining) {
            const auto count = ::read(input.value, buffer.data(),
                                      std::min<std::uint64_t>(remaining, buffer.size()));
            if (count < 0 && errno == EINTR)
                continue;
            require(count > 0, "source capture truncated/read failure");
            ssize_t offset{};
            while (offset < count) {
                const auto written = ::write(frozen.value, buffer.data() + offset, count - offset);
                if (written < 0 && errno == EINTR)
                    continue;
                require(written > 0, "source capture write failure");
                offset += written;
            }
            remaining -= count;
        }
        const auto same = [&initial](const struct stat &value) {
            return initial.st_dev == value.st_dev && initial.st_ino == value.st_ino &&
                   initial.st_size == value.st_size && initial.st_mode == value.st_mode &&
                   initial.st_mtim.tv_sec == value.st_mtim.tv_sec &&
                   initial.st_mtim.tv_nsec == value.st_mtim.tv_nsec &&
                   initial.st_ctim.tv_sec == value.st_ctim.tv_sec &&
                   initial.st_ctim.tv_nsec == value.st_ctim.tv_nsec;
        };
        require(::fstat(input.value, &final_fd) == 0 && ::lstat(path.c_str(), &final_path) == 0 &&
                    same(final_fd) && same(final_path),
                "source changed during capture");
        require(::fcntl(frozen.value, F_ADD_SEALS,
                        F_SEAL_WRITE | F_SEAL_GROW | F_SEAL_SHRINK | F_SEAL_SEAL) == 0,
                "source capture sealing failed");
        const fs::path captured = "/proc/self/fd/" + std::to_string(frozen.value);
        // Hash and decode the same immutable bytes, never reopen a mutable path
        // between a hash check and the actual decoder.
        digest = sha256(captured);
        const auto count = validate_laz(captured);
        require(count > 0 && count <= max_records, "single-file replay record budget");
        points = decode(captured, package.contract);
        require(points.size() == count && sha256(path) == digest,
                "source changed during single-file preparation");
    }
};
void source_identity(const Frame &frame, const std::string &digest) {
    require(frame.sensor_id == "file-order-replay" && frame.stream_id == digest &&
                frame.coordinate_frame == "source-file" && frame.crs == "UNSPECIFIED:source-file",
            "single-file replay source identity mismatch");
}
void row_coverage(const Frame &frame, std::uint64_t &rows, const Config &config) {
    require(rows <= config.records &&
                frame.points.size() ==
                    std::min(config.limits.max_frame_points, config.records - rows),
            "single-file window/record coverage mismatch");
    for (const auto &point : frame.points)
        require(point.id == rows++, "single-file original row order/coverage mismatch");
}
void receipt_identity(Json &receipt, const Descriptor &descriptor) {
    receipt["descriptor_sha256"] = descriptor.sha256();
    receipt["descriptor"] = descriptor.spec();
    receipt["taxonomy_sha256"] = descriptor.taxonomy().sha256();
}
void send(const Config &config, const CpuAdmittedPackage &admitted, const Package &package,
          Json &receipt) {
    const auto &args = config.args;
    Source source(args.get("--source"), package);
    const auto rate = args.number("--rate", 1000000, 20000000);
    require(rate >= config.limits.max_frame_points, "frame spans more than one replay second");
    auto socket = wire::connect_loopback(config.port);
    PeerTransport transport(WireProtocol::binary_v3, socket.fd(), admitted, config.limits,
                            args.get("--session"), config.epoch);
    transport.metadata_admission();
    require(transport.negotiated_limits().max_frame_points == config.limits.max_frame_points,
            "single-file CLI requires identical peer point limits");
    transport.acknowledge_execution_ready();
    const auto anchor = Startup::Clock::now();
    receipt["replay_anchor_steady_ns"] =
        std::chrono::duration_cast<std::chrono::nanoseconds>(anchor.time_since_epoch()).count();
    receipt["replay_clock_scope"] =
        "producer local steady_clock; synthetic availability only, no physical capture claim";
    std::ifstream boot("/proc/sys/kernel/random/boot_id");
    std::string boot_id;
    std::getline(boot, boot_id);
    receipt["replay_clock_boot_id"] = boot_id;
    std::uint64_t sequence{}, rows{};
    while (rows < source.points.size()) {
        active();
        const auto end =
            std::min<std::uint64_t>(rows + config.limits.max_frame_points, source.points.size());
        Frame frame;
        frame.identity = {transport.descriptor().sha256(), args.get("--session"), config.epoch};
        frame.sensor_id = "file-order-replay";
        frame.stream_id = source.digest;
        frame.coordinate_frame = "source-file";
        frame.crs = "UNSPECIFIED:source-file";
        frame.sequence = frame.frame_id = sequence++;
        frame.frame_start_ns = rows * 1000000000ULL / rate;
        frame.frame_end_ns = (end - 1) * 1000000000ULL / rate;
        for (; rows < end; ++rows)
            frame.points.push_back({rows,
                                    rows * 1000000000ULL / rate,
                                    source.points[rows].xyz,
                                    source.points[rows].intensity,
                                    {},
                                    {}});
        // Accumulation precedes frame availability; source timestamps are declared
        // synthetic replay time, never host clocks or physical capture timestamps.
        std::this_thread::sleep_until(anchor +
                                      std::chrono::nanoseconds(end * 1000000000ULL / rate));
        if (frame.sequence == 0)
            receipt["first_scheduled_frame_available_steady_ns"] =
                receipt.at("replay_anchor_steady_ns").get<std::uint64_t>() +
                end * 1000000000ULL / rate;
        receipt["last_scheduled_frame_available_steady_ns"] =
            receipt.at("replay_anchor_steady_ns").get<std::uint64_t>() + end * 1000000000ULL / rate;
        transport.send_frame(frame, config.deadline());
    }
    receipt["input_completion"] = transport.finish_input(config.deadline());
    require(sha256(args.get("--source")) == source.digest, "replay source changed during send");
    receipt_identity(receipt, transport.descriptor());
    receipt.update({{"frames", sequence},
                    {"records", rows},
                    {"source_sha256", source.digest},
                    {"offered_rate_pps", rate},
                    {"producer_elapsed_seconds",
                     std::chrono::duration<double>(Startup::Clock::now() - anchor).count()},
                    {"scope", "input transcript acknowledged; output success is separate"}});
}
void receive(const Config &config, const CpuAdmittedPackage &admitted, Json &receipt) {
    const auto &args = config.args;
    if (config.trace.enabled()) {
        std::ifstream boot("/proc/sys/kernel/random/boot_id");
        std::string boot_id;
        std::getline(boot, boot_id);
        receipt["receiver_clock_boot_id"] = boot_id;
    }
    auto input_listener = wire::listen_loopback(config.port);
    auto output_listener = wire::listen_loopback(config.output_port);
    std::cerr << "generic_listening=1\n" << std::flush;
    // These owners outlive both borrowed transports and the inference session.
    auto input_socket = wire::accept(input_listener, config.io_ms);
    auto output_socket = wire::accept(output_listener, config.io_ms);
    AuthorityTransport input(WireProtocol::binary_v3, input_socket.fd(), admitted, config.limits,
                             args.get("--session"), config.epoch);
    AuthorityTransport output(WireProtocol::binary_v3, output_socket.fd(), admitted, config.limits,
                              args.get("--session"), config.epoch);
    input.metadata_admission();
    output.metadata_admission();
    auto session = input.create_session_with_output(output, {}, config.trace);
    OutputSender sender(output, 1.0, config.cells);
    std::uint64_t rows{};
    for (std::uint64_t sequence = 0; sequence < config.frames; ++sequence) {
        auto frame = input.receive_frame(config.deadline());
        const auto available =
            config.trace.enabled() ? std::optional<std::uint64_t>(steady_ns()) : std::nullopt;
        require(frame.sequence == sequence && frame.frame_id == sequence,
                "single-file frame sequence mismatch");
        source_identity(frame, config.source_sha);
        row_coverage(frame, rows, config);
        auto result = session->run(std::move(frame), available);
        const auto ready = config.trace.enabled() ? steady_ns() : 0;
        const auto deadline = config.deadline();
        const auto send_started = config.trace.enabled() ? steady_ns() : 0;
        sender.send(result, deadline);
        if (config.trace.enabled()) {
            const auto sent = steady_ns();
            session->record_output_timing(sequence, ready, send_started, sent);
        }
    }
    require(rows == config.records, "single-file missing input records");
    receipt["input_completion"] = input.finish_input(config.frames, rows, config.deadline());
    if (config.trace.enabled())
        receipt["input_completion_observed_steady_ns"] = steady_ns();
    receipt["output_completion"] = sender.finish(config.deadline());
    if (config.trace.enabled()) {
        receipt["output_completion_observed_steady_ns"] = steady_ns();
        receipt["completion_clock_scope"] = "receiver local steady_clock; terminal acknowledgments "
                                            "observed before samples; receipt publication follows";
    }
    receipt_identity(receipt, input.descriptor());
    receipt["session_identity"] = session->identity();
    receipt.update({{"frames", config.frames},
                    {"records", rows},
                    {"declared_source_sha256", config.source_sha},
                    {"scope", "finite single-file input and independently acknowledged output"}});
}
void subscribe(const Config &config, const CpuAdmittedPackage &admitted, const Package &package,
               Json &receipt) {
    const auto &args = config.args;
    std::unique_ptr<Source> expected;
    if (args.values.count("--source")) {
        expected = std::make_unique<Source>(args.get("--source"), package);
        require(expected->digest == config.source_sha && expected->points.size() == config.records,
                "subscriber source identity/count mismatch");
    }
    auto socket = wire::connect_loopback(config.port);
    PeerTransport transport(WireProtocol::binary_v3, socket.fd(), admitted, config.limits,
                            args.get("--session"), config.epoch);
    transport.metadata_admission();
    transport.acknowledge_execution_ready();
    OutputReceiver receiver(transport, 1.0, config.cells);
    std::unique_ptr<LazExport> exporter;
    if (args.values.count("--export-bundle")) {
        exporter = std::make_unique<LazExport>(args.get("--source"), args.get("--model"),
                                               args.get("--export-bundle"), transport.descriptor());
        require(exporter->source_sha256() == config.source_sha,
                "export source snapshot differs from admitted replay source");
    }
    std::uint64_t rows{};
    for (std::uint64_t sequence = 0; sequence < config.frames; ++sequence) {
        auto result = receiver.receive(config.deadline());
        require(result.frame.sequence == sequence && result.frame.frame_id == sequence,
                "subscriber frame sequence mismatch");
        source_identity(result.frame, config.source_sha);
        row_coverage(result.frame, rows, config);
        if (expected)
            for (const auto &point : result.frame.points)
                require(point.xyz == expected->points.at(point.id).xyz && point.intensity &&
                            *point.intensity == expected->points.at(point.id).intensity &&
                            !point.return_number && !point.return_count,
                        "subscriber source record differs");
        if (exporter)
            exporter->observe(result);
    }
    require(rows == config.records, "subscriber missing records");
    receipt["output_completion"] = receiver.finish(config.deadline());
    // Publication requires independent terminal validation. observe() never writes.
    if (exporter)
        receipt["export_bundle"] = exporter->finish();
    if (expected)
        require(sha256(args.get("--source")) == expected->digest,
                "subscriber source changed during validation");
    receipt_identity(receipt, transport.descriptor());
    receipt.update({{"frames", config.frames},
                    {"records", rows},
                    {"declared_source_sha256", config.source_sha},
                    {"source_records_independently_checked", bool(expected)},
                    {"scope", "independently reconstructed acknowledged output; ACK sent"}});
}
} // namespace
int live_main(int argc, char **argv) {
    for (int i = 2; i < argc; ++i)
        if (std::string(argv[i]) == "--manifest")
            return replay_main(argc, argv);
    Json receipt{{"schema", "synrex.generic_single_file_cli.v1"},
                 {"status", "FAIL"},
                 {"workload", "finite file-order replay with synthetic pacing"},
                 {"declared_source_clock", "synthetic nanoseconds from trial origin zero"},
                 {"physical_capture_age_seconds", nullptr},
                 {"distinct_manifest_replay", "NOT_IMPLEMENTED"}};
    fs::path receipt_path;
    try {
        interrupted = 0;
        const Args args(argc, argv);
        receipt_path = args.get("--receipt");
        new_path(receipt_path);
        const Config config(args);
        const Package package(args.get("--model"), PackageUse::file);
        const CpuAdmittedPackage admitted(args.get("--model"));
        const auto local =
            Descriptor::from_package(admitted, config.limits, args.get("--session"), config.epoch);
        require(local.spec().at("manifest_sha256") == package.manifest_sha256,
                "model changed during CPU admission");
        require(config.limits.max_frame_points <= package.contract.max_points,
                "unsafe persistent window point budget");
        config.trace.validate_contract(package.contract, config.frames);
        std::signal(SIGINT, interrupt);
        std::signal(SIGTERM, interrupt);
        receipt["mode"] = args.mode;
        if (args.mode == "send")
            send(config, admitted, package, receipt);
        else if (args.mode == "receive")
            receive(config, admitted, receipt);
        else
            subscribe(config, admitted, package, receipt);
        active();
        receipt["status"] = "PASS";
    } catch (const std::exception &error) {
        receipt["status"] = "FAIL";
        receipt["error"] = error.what();
    }
    // Diagnostics may contain raw malformed UTF8 from rejected inputs. Success
    // metadata remains exact; only the failure diagnostic uses replacement.
    const auto bytes = receipt.dump(2, ' ', false, Json::error_handler_t::replace) + "\n";
    try {
        if (!receipt_path.empty())
            write_new(receipt_path, bytes);
    } catch (const std::exception &error) {
        receipt["status"] = "FAIL";
        receipt["receipt_error"] = error.what();
    }
    std::cout << receipt.dump(2, ' ', false, Json::error_handler_t::replace) << '\n';
    return receipt.at("status") == "PASS" ? 0 : 1;
}
} // namespace synrex::generic
