// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Tuomas Pietilä
// Explicit opt-in physical GPU qualification. No model-free inference substitute.
#include "synrex/service.hpp"
#include <algorithm>
#include <cmath>
#include <cstring>
#include <cstdlib>
#include <fstream>
#include <iostream>
using namespace synrex;
using namespace synrex::streaming;
namespace {
std::string bytes(const fs::path &p) {
    require(fs::file_size(p) <= 64 * 1024 * 1024, "trace read budget");
    std::ifstream in(p, std::ios::binary);
    return std::string(std::istreambuf_iterator<char>(in), {});
}
double value(const std::string &s, std::size_t i, bool half) {
    if (!half) { float f{}; std::memcpy(&f, s.data() + i * 4, 4); return f; }
    std::uint16_t h{}; std::memcpy(&h, s.data() + i * 2, 2);
    const auto e = (h >> 10) & 31; const auto m = h & 1023;
    const auto v = e == 0 ? std::ldexp(double(m), -24) :
                   e == 31 ? (m ? NAN : INFINITY) : std::ldexp(double(1024 + m), int(e) - 25);
    return (h & 32768) ? -v : v;
}
Json compare(const fs::path &offline, const fs::path &stream) {
    const auto a = read_json(offline / "manifest.json"), b = read_json(stream / "manifest.json");
    require(a.at("tensor_contract") == b.at("tensor_contract") &&
                a.at("blocks").size() == b.at("blocks").size(), "trace contracts/boundaries mismatch");
    double max_abs = 0, max_relative = 0;
    std::uint64_t values = 0;
    for (std::size_t block = 0; block < a.at("blocks").size(); ++block) {
        const auto &aa = a.at("blocks")[block], &bb = b.at("blocks")[block];
        require(aa.at("origin") == bb.at("origin"), "context origin mismatch");
        for (const auto &kind : {"input", "logits"}) {
            const auto x = bytes(offline / aa.at(kind).get<std::string>());
            const auto y = bytes(stream / bb.at(kind).get<std::string>());
            require(x.size() == y.size(), "trace tensor size mismatch");
            if (std::string(kind) == "input") require(x == y, "CUDA input preparation differs");
            else {
                const bool half = a.at("tensor_contract").at("output").at("dtype") == "float16";
                for (std::size_t i = 0; i < x.size() / (half ? 2 : 4); ++i) {
                    const auto u = value(x, i, half), v = value(y, i, half);
                    require(std::isfinite(u) && std::isfinite(v), "nonfinite reference or stream logits");
                    const auto error = std::abs(u - v), relative = error / std::max(1e-6, std::abs(u));
                    max_abs = std::max(max_abs, error); max_relative = std::max(max_relative, relative); ++values;
                    require(error <= 0.001 + 0.001 * std::abs(u), "offline/stream logits outside numeric budget");
                }
            }
        }
    }
    return {{"status", "PASS"}, {"inputs_byte_identical", true}, {"logit_values_compared", values},
            {"max_absolute_logit_error", max_abs}, {"max_relative_logit_error", max_relative},
            {"absolute_tolerance", .001}, {"relative_tolerance", .001}};
}
Json qualify(const fs::path &model, const fs::path &out, const std::vector<fs::path> &sources = {}) {
    fs::create_directory(out);
    SessionOptions options; options.pinned_staging = std::getenv("SYNREX_SESSION_PINNED") != nullptr;
    if (const auto *mode = std::getenv("SYNREX_GEOMETRY")) options.geometry = geometry_mode(mode);
    auto session = create_session(model, 65536, options);
    const auto &c = session->contract();
    const auto taxonomy = c.las_classes.size() == 7 ? "synrex.fractal7.v1" : "synrex.synthetic3.v1";
    auto chunks = synthetic(91, 4, 777, 197, taxonomy);
    Assembler assembler;
    std::vector<Frame> frames;
    for (const auto &chunk : chunks) if (auto f = assembler.accept(chunk, steady_ns())) frames.push_back(std::move(*f));
    std::size_t genuine_points = 0;
    for (const auto &laz : sources) {
        const auto source = decode(laz, c);
        genuine_points += source.size();
        require(genuine_points <= 1000000, "qualification genuine point budget exceeded");
        // Contiguous source-record windows, with the exact same R4 per-window geometry in both paths.
        for (std::size_t offset = 0; offset < source.size(); offset += 65536) {
            auto f = frames.front(); f.points.clear();
            f.header.frame_id = frames.size();
            f.header.frame_start_ns = 1700000000000000000ULL + frames.size() * 100000000ULL;
            f.header.frame_end_ns = f.header.frame_start_ns + 99999999ULL;
            f.first_sequence = frames.size(); f.last_sequence = frames.size();
            for (auto i = offset; i < std::min(source.size(), offset + 65536); ++i)
                f.points.push_back({i, f.header.frame_start_ns, source[i].xyz, source[i].intensity, {}, {}});
            f.header.frame_points = f.points.size(); frames.push_back(std::move(f));
        }
    }
    // Window size changes batching/context, so compare each configuration to its own
    // identical-boundary offline reference, never compare labels between window sizes.
    const auto size_source = sources.empty() ? std::vector<Point>{} : decode(sources.front(), c);
    for (std::size_t n : {4096, 8192, 16384, 32768}) {
        auto f = frames.front(); f.points.clear();
        f.header.frame_id = frames.size();
        f.header.frame_start_ns = 1700000000000000000ULL + frames.size() * 100000000ULL;
        f.header.frame_end_ns = f.header.frame_start_ns + 99999999ULL;
        f.first_sequence = frames.size(); f.last_sequence = frames.size();
        if (size_source.empty()) {
            for (const auto &chunk : synthetic(91, 1, n, 4096, taxonomy))
                for (auto p : chunk.points) { p.timestamp_ns = f.header.frame_start_ns; f.points.push_back(p); }
        } else {
            require(size_source.size() >= n, "window-equivalence source too small");
            for (std::size_t i = 0; i < n; ++i)
                f.points.push_back({i, f.header.frame_start_ns, size_source[i].xyz, size_source[i].intensity, {}, {}});
        }
        f.header.frame_points = n; frames.push_back(std::move(f));
    }
    // Boundary/precision fixtures are compared through actual inference in both paths.
    for (double offset : {-6390000.123, 700000.001, 1e8}) {
        auto f=frames.front();f.points.clear();f.header.frame_id=frames.size();
        f.header.frame_start_ns=1700000000000000000ULL+frames.size()*100000000ULL;
        f.header.frame_end_ns=f.header.frame_start_ns+99999999;f.first_sequence=frames.size();f.last_sequence=frames.size();
        for (std::size_t i=0;i<1024;++i) {
            std::array<double,3> xyz{};
            for(int a=0;a<3;++a) {
                const auto edge=offset+double((i/(a+1))%3)*c.extent[a];
                xyz[a]=i%4==0?edge:i%4==1?std::nextafter(edge,-INFINITY):i%4==2?std::nextafter(edge,INFINITY):offset+double(i%250)*0.001;
            }
            f.points.push_back({i,f.header.frame_start_ns+i,xyz,static_cast<std::uint16_t>(i*61),1,2});
        }
        f.header.frame_points=f.points.size();frames.push_back(std::move(f));
    }
    // Emit an offline-emulated stream for the real threaded service, retaining original record IDs.
    std::string replay;
    std::uint64_t sequence = 0;
    for (const auto &f : frames) {
        for (std::size_t start = 0; start < f.points.size(); start += 4096) {
            auto h = f.header;
            h.chunk_index = start / 4096; h.chunk_count = (f.points.size() + 4095) / 4096;
            h.sequence = sequence++;
            h.points.assign(f.points.begin() + start, f.points.begin() + std::min(f.points.size(), start + 4096));
            replay += serialize(h) + "\n";
        }
    }
    write_new(out / "replay-input.jsonl", replay);
    // Validate both seven-channel and three-channel service bindings in this SAME executable.
    const Package reference(model);
    Json windows = Json::array();
    std::vector<std::uint64_t> histogram(c.las_classes.size(), 0);
    std::uint64_t total = 0;
    double elapsed = 0;
    for (std::size_t i = 0; i < frames.size(); ++i) {
        std::vector<Point> points;
        for (const auto &p : frames[i].points) points.push_back({p.xyz, *p.intensity});
        const auto before = steady_ns();
        const auto st = out / (std::to_string(i) + "-stream");
        auto prediction = session->run(points, st);
        const auto ot = out / (std::to_string(i) + "-offline");
        InferenceOptions options; options.tensor_trace = ot;
        std::vector<std::uint8_t> semantics;
        const auto mapped = infer(reference, points, geometry(points, c), nullptr, &semantics, options);
        require(prediction.semantic_ids == semantics && prediction.las_codes == mapped,
                "strict original-index offline/stream semantic mismatch");
        const auto numeric = compare(ot, st);
        Classification classified{frames[i], semantics, mapped, prediction.timings};
        const auto e = event(classified);
        for (std::size_t p = 0; p < points.size(); ++p) {
            require(e.at("points")[p].at("id") == frames[i].points[p].id &&
                        e.at("points")[p].at("timestamp_ns") == frames[i].points[p].timestamp_ns,
                    "source ID/timestamp mismatch");
            ++histogram.at(semantics[p]);
        }
        write_new(out / (std::to_string(i) + "-event.json"), e.dump() + "\n");
        total += points.size(); elapsed += prediction.timings.at("window_seconds").get<double>();
        windows.push_back({{"status", "PASS"}, {"points", points.size()}, {"numeric", numeric},
            {"timings", prediction.timings}, {"diagnostic_reference_and_trace_wall_seconds", (steady_ns() - before) / 1e9}});
    }
    Json receipt{{"schema", "synrex.streaming_equivalence.v1"}, {"status", "PASS"},
        {"scope", "Actual TensorRT streaming windows versus legacy R4 native offline inference; diagnostic tensor I/O enabled"},
        {"session", session->identity()}, {"semantic_histogram", histogram},
        {"freshly_classified_points", total}, {"incoming_points", total}, {"dropped_points", 0},
        {"windows", windows}, {"temporal_context_halo_frames", 0},
        {"coverage", "Every input point, including same-voxel duplicates; exact source-order semantics and LAS projection"},
        {"diagnostic_stream_window_seconds", elapsed}, {"sustainable_live_rate", "NOT_RUN"}};
    receipt["licensed_sources"] = Json::array();
    for (const auto &laz : sources) receipt["licensed_sources"].push_back(
        {{"basename", laz.filename().string()}, {"sha256", sha256(laz)}});
    receipt["source_timestamp_scope"] = "Offline-emulated UNIX frame times, not original LAS GPS-time/return decoding";
    session.reset();
    const auto resources = session_resources();
    require(resources.at("live_tracked_device_bytes") == 0 && resources.at("live_pinned_host_bytes") == 0,
            "owned session resources remain after destruction");
    receipt["owned_resources_after_session_destruction"] = resources;
    write_new(out / "receipt.json", receipt.dump(2) + "\n");
    return receipt;
}
}
int main(int argc, char **argv) {
    try {
        require(argc >= 4 && argc <= 12, "usage: synrex-stream-gpu-tests SYNTHETIC.srxmodel UNET.srxmodel NEW_OUTPUT [LICENSED_SOURCE.laz ...]");
        require(gpu_enabled(), "GPU backend required; inference NOT_RUN");
        require(fs::create_directory(argv[3]), "qualification output must be new");
        const auto binary = fs::canonical("/proc/self/exe"); const auto hash = sha256(binary);
        const auto synthetic = qualify(argv[1], fs::path(argv[3]) / "synthetic");
        std::vector<fs::path> licensed;
        for (int i = 4; i < argc; ++i) licensed.emplace_back(argv[i]);
        const auto trained = qualify(argv[2], fs::path(argv[3]) / "unet", licensed);
        require(hash == sha256(binary), "runtime binary changed between models");
        Json receipt{{"status", "PASS"}, {"schema", "synrex.streaming_gpu_qualification.v1"},
            {"binary_sha256_before_after", hash}, {"synthetic", synthetic}, {"unet", trained},
            {"AGX", "NOT_RUN"}, {"THOR", "NOT_RUN"}};
        write_new(fs::path(argv[3]) / "receipt.json", receipt.dump(2) + "\n");
        std::cout << receipt.dump(2) << '\n';
        return 0;
    } catch (const std::exception &e) {
        std::cerr << Json{{"status", "FAIL"}, {"error", e.what()}}.dump() << '\n'; return 1;
    }
}
