// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Tuomas Pietilä
// Deterministic assembler-only throughput; NO GPU, neural model or live sensor.
#include "synrex/causal_context.hpp"
#include <chrono>
#include <cstdint>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <sys/resource.h>

namespace {
using namespace synrex::generic;
constexpr std::uint64_t kFrames = 192, kFramePoints = 16384;
constexpr std::uint64_t kTotal = kFrames * kFramePoints;

Frame frame(std::uint64_t frame_no) {
    Frame out;
    out.identity = {"bench-frozen-descriptor", "bench-session", 0};
    out.sensor_id = "synthetic-als"; out.stream_id = "synthetic-source";
    out.coordinate_frame = "projected"; out.crs = "EPSG:3067";
    out.sequence = frame_no; out.frame_id = frame_no;
    out.frame_start_ns = frame_no * 1000000;
    out.frame_end_ns = (frame_no + 1) * 1000000;
    SourceHeader header{};
    header.records = kTotal; header.wire_begin = 1000000;
    out.source_header = header;
    out.points.reserve(kFramePoints);
    out.source_rows.reserve(kFramePoints);
    for(std::uint64_t i=frame_no*kFramePoints; i<(frame_no+1)*kFramePoints; ++i) {
        synrex::streaming::SourcePoint p;
        p.id = header.wire_begin + i;
        p.timestamp_ns = i;
        p.xyz = {{300000.0 + double(i%10000)*0.01,
                  6700000.0 + double((i/10000)%10000)*0.01,
                  50.0 + double(i%1000)*0.001}};
        p.intensity = static_cast<std::uint16_t>(i%60000);
        p.return_number = 1; p.return_count = 1;
        out.points.push_back(p);
        SourceRow row{};
        row.original_record_index = i;
        row.integer_xyz = {{static_cast<std::int32_t>(i%10000),
                            static_cast<std::int32_t>((i/10000)%10000), 5000}};
        out.source_rows.push_back(row);
    }
    return out;
}

std::size_t context_count(const CausalContextWindow &w) {
#ifdef SYNREX_PERF_BASELINE
    return w.context.size();
#else
    return w.context->size();
#endif
}
}
int main() {
    try {
        CausalContextPolicy p;
        p.fresh_points = kFramePoints;
        p.history_points = 114688;
        p.max_context_points = 131072;
        p.max_inflight_windows = 4;
        p.semantic_class_count = 7;
        p.fixed_world_origin_xy_m = {{300000.0, 6700000.0}};
        p.world_grid_size_m = 1.0;
        p.projected_crs = "EPSG:3067";
        p.require_exact_source_rows = true;
        CausalContextAssembler assembler(p);
        const auto t0=std::chrono::steady_clock::now();
        std::uint64_t published=0, context_processed=0, checksum=0;
        for(std::uint64_t i=0;i<kFrames;++i) {
            auto issued=assembler.ingest(frame(i));
            if (issued.size()!=1) throw std::runtime_error("unexpected context window count");
            const auto &window=issued[0];
            const auto count=context_count(window);
            std::vector<std::uint8_t> classifications(count, 1);
            auto new_only=assembler.commit_new_only(window, classifications);
            if(new_only.source_ids.size()!=kFramePoints || new_only.source_ids.front() != 1000000+i*kFramePoints)
                throw std::runtime_error("fresh point identity failure");
            published += new_only.source_ids.size();
            context_processed += count;
            checksum += new_only.source_ids.front() + new_only.source_ids.back();
        }
        if(assembler.source_eof() || assembler.failed() || assembler.accepted_points()!=kTotal)
            throw std::runtime_error("failed final coverage");
        const double elapsed=std::chrono::duration<double>(std::chrono::steady_clock::now()-t0).count();
        if(elapsed<=0.0) throw std::runtime_error("invalid timing");
        rusage usage{};
        if(getrusage(RUSAGE_SELF, &usage)!=0) throw std::runtime_error("getrusage failed");
        std::cout<<std::fixed<<std::setprecision(6);
        std::cout<<"{\"schema\":\"synrex.causal_context_perf.v1\","
                 <<"\"mode\":\""<<
#ifdef SYNREX_PERF_BASELINE
                 "sha256_baseline"
#else
                 "immutable_handle"
#endif
                 <<"\",\"model_execution\":\"NOT_RUN\","
                 <<"\"total_points\":"<<published<<",\"context_processed\":"<<context_processed
                 <<",\"windows\":"<<kFrames<<",\"elapsed_seconds\":"<<elapsed
                 <<",\"fresh_points_per_second\":"<<published/elapsed
                 <<",\"context_points_per_second\":"<<context_processed/elapsed
                 <<",\"maxrss_kib\":"<<usage.ru_maxrss<<",\"checksum\":"<<checksum<<"}\n";
        return 0;
    } catch(const std::exception& e) {
        std::cerr<<"CAUSAL_CONTEXT_PERF_FAILED "<<e.what()<<'\n';
        return 1;
    }
}
