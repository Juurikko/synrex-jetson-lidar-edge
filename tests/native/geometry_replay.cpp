// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Tuomas Pietilä
// Offline CPU oracle for every deterministic window in sustained trials.
#include "synrex/runtime.hpp"
#include <chrono>
#include <cstring>
#include <iostream>
#include <numeric>
using namespace synrex;
int main(int argc,char **argv) {
    try {
        require(argc==7,"usage: geometry-replay MODEL SOURCE FRAMES POINTS X_STEP_METRES NEW_RECEIPT");
        const Package package(argv[1]);const auto &c=package.contract;const auto corpus=decode(argv[2],c);
        const auto frames=std::stoull(argv[3]), count=std::stoull(argv[4]);const double step=std::stod(argv[5]);
        require(frames>0 && frames<=30000 && count>0 && count<=65536 && count<=corpus.size() && std::isfinite(step) && step>=0 && step<=1000,"replay oracle budget");
        std::uint64_t blocks=0,members=0,feature_values=0;const auto begin=std::chrono::steady_clock::now();
        std::vector<Point> points(count);
        // Geometry is independent of acquisition rate/timestamp. Identical frame offsets
        // cover every source window from the timed 5s+60s trials up to 3M points/s.
        for(std::size_t frame=0;frame<frames;++frame) {
            for(std::size_t i=0;i<count;++i) {points[i]=corpus[(frame*count+i)%corpus.size()];points[i].xyz[0]+=frame*step;}
            const auto reference=geometry(points,c), actual=geometry(points,c,GeometryMode::direct);
            require(reference.memberships==actual.memberships && reference.blocks.size()==actual.blocks.size(),"replay block/membership mismatch");
            for(std::size_t b=0;b<reference.blocks.size();++b) {
                const auto &x=reference.blocks[b], &y=actual.blocks[b];
                require(x.members.size()==y.members.size() && std::memcmp(x.origin.data(),y.origin.data(),24)==0,"replay origin mismatch");
                for(std::size_t i=0;i<x.members.size();++i)
                    require(x.members[i].point==y.members[i].point && x.members[i].voxel==y.members[i].voxel,"replay source/voxel/order mismatch");
                const auto left=reference_tensor(points,x,c), right=reference_tensor(points,y,c);
                require(left.size()==right.size() && std::memcmp(left.data(),right.data(),left.size()*4)==0,"replay feature bytes mismatch");
                feature_values+=left.size();++blocks;
            }
            members+=reference.memberships;
        }
        write_new(argv[6],Json{{"schema","synrex.full_window_geometry_oracle.v1"},{"status","PASS"},
            {"scope","CPU-only exact membership/source-index/voxel-index/order and full float32 feature-tensor byte comparisons. GPU kernels/engine unchanged and separately traced; no CPU neural throughput claim."},
            {"source_basename",fs::path(argv[2]).filename().string()},{"source_sha256",sha256(argv[2])},
            {"model_manifest_sha256",sha256(package.root/"manifest.json")},{"frames",frames},{"frame_points",count},
            {"translation_x_metres_per_frame",step},{"points_compared",frames*count},{"memberships_compared",members},
            {"blocks_compared",blocks},{"feature_values_byte_compared",feature_values},
            {"wall_seconds",std::chrono::duration<double>(std::chrono::steady_clock::now()-begin).count()}}.dump(2)+"\n");
        std::cout<<"full_window_geometry_oracle=PASS\n";return 0;
    }catch(const std::exception &e){std::cerr<<e.what()<<'\n';return 1;}
}
