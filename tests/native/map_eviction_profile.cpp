// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Tuomas Pietilä
// Model-free, CPU-only capacity/eviction service probe; never neural points/s.
#include "synrex/map_oracle.hpp"
#include <chrono>
#include <iostream>
using namespace synrex;
using namespace synrex::streaming;
int main(int argc,char **argv) {
    try {
        require(argc==2 || argc==3,"usage: map-eviction-profile NEW_RECEIPT [reference|indexed]");Json cases=Json::array();
        const auto policy=argc==3?map_policy(argv[2]):MapPolicy::indexed;
        for(std::size_t capacity:{1024,4096,16384,65536}) {
            SemanticMap map(1.0,capacity,true,policy,true);MapOracle oracle(capacity);
            Classification r;auto &h=r.frame.header;
            h.sensor_id="model-free";h.stream_id="eviction-profile";h.coordinate_frame="fixed-world";
            h.crs="LOCAL:metres";h.taxonomy="synrex.fractal7.v1";
            std::uint64_t time=1700000000000000000ULL;
            auto batch=[&](std::size_t start,std::size_t count) {
                r.frame.points.clear();r.semantic_ids.clear();r.las_codes.clear();h.frame_points=count;
                h.frame_start_ns=time;h.frame_end_ns=time+count;
                const std::uint8_t codes[]{1,2,200,6,9,17,201};
                for(std::size_t i=0;i<count;++i) {
                    const auto id=(start+i)%7;
                    r.frame.points.push_back({start+i,time+i,{{double(start+i)-32768.25,-12.001,0}},123,1,2});
                    r.semantic_ids.push_back(id);r.las_codes.push_back(codes[id]);
                }
                time+=count+1;
                const auto begin=std::chrono::steady_clock::now();const auto delta=map.update(r);
                const auto elapsed=std::chrono::duration<double>(std::chrono::steady_clock::now()-begin).count();
                oracle.verify(r,delta,map.size());return elapsed;
            };
            // Fill without eviction; measured new voxels then force one removal each.
            for(std::size_t start=0;start<capacity;start+=256) (void)batch(start,std::min<std::size_t>(256,capacity-start));
            double insert_seconds=0,existing_seconds=0;
            for(std::size_t i=0;i<4;++i) insert_seconds+=batch(capacity+i*256,256);
            for(std::size_t i=0;i<4;++i) existing_seconds+=batch(capacity+768,256);
            const auto snapshot=map.snapshot(time+1000);require(snapshot==oracle.snapshot(time+1000),"full snapshot/age parity");
            const auto counts=map.counters();require(counts.removals==1024 && counts.peak_cells==capacity,"full-capacity eviction coverage");
            cases.push_back({{"capacity",capacity},{"status","PASS"},{"measured_insertions_and_removals",1024},
                {"evicting_update_seconds",insert_seconds},{"existing_update_seconds",existing_seconds},
                {"model_free_evicting_updates_per_active_second",1024/insert_seconds},
                {"model_free_existing_updates_per_active_second",1024/existing_seconds},{"operations",counts.json()},{"profile",map.profile()}});
        }
        write_new(argv[1],Json{{"schema","synrex.cpu_eviction_service.v1"},{"status","PASS"},
            {"scope","CPU map-only operation service probe; actual changing coordinates and verified LRU removal/aging; dummy classes, no neural classification claim; short probes do not establish sustained stream rates"},{"cases",cases}}.dump(2)+"\n");
        std::cout<<"cpu_eviction_profile=PASS\n";return 0;
    }catch(const std::exception &e){std::cerr<<e.what()<<'\n';return 1;}
}
