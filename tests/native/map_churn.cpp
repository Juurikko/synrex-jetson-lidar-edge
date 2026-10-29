// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Tuomas Pietilä
#include "synrex/map_oracle.hpp"
#include <chrono>
#include <iostream>
using namespace synrex;
using namespace synrex::streaming;
int main(int argc,char **argv) {
    try {
        require(argc==1 || argc==2,"optional NEW_RECEIPT");Json cases=Json::array();
        for (std::size_t capacity:{1,64,1024,4096}) {
            SemanticMap map(1.0,capacity,true);MapOracle oracle(capacity);
            Classification r;auto &h=r.frame.header;
            h.sensor_id="model-free-fixture";h.stream_id="map-churn";h.coordinate_frame="fixed-world-metres";
            h.crs="LOCAL:test";h.taxonomy="synrex.fractal7.v1";
            const std::uint8_t mapping[]{1,2,200,6,9,17,201};
            const auto start=std::chrono::steady_clock::now();std::uint64_t as_of=0;
            for (std::size_t frame=0;frame<200;++frame) {
                r.frame.points.clear();r.semantic_ids.clear();r.las_codes.clear();
                h.frame_id=frame;h.frame_points=256;
                h.frame_start_ns=1700000000000000000ULL+frame*10000000;h.frame_end_ns=h.frame_start_ns+1000000;
                as_of=h.frame_end_ns;
                for (std::size_t i=0;i<256;++i) {
                    r.frame.points.push_back({i,h.frame_start_ns+i,{{frame*64.0+(i%32)/8.0-1000.125,double(i/32)-4.25,0.001*double(i%11)}},123,1,2});
                    r.semantic_ids.push_back((frame+i)%7);r.las_codes.push_back(mapping[r.semantic_ids.back()]);
                }
                auto deltas=map.update(r);oracle.verify(r,deltas,map.size());
                if (frame%17==0) require(map.snapshot(as_of+1000000)==oracle.snapshot(as_of+1000000),"full snapshot/aging parity");
            }
            // Replay an older observation of an existing cell; no mutation or delta allowed.
            r.frame.points={r.frame.points.back()};r.semantic_ids={r.semantic_ids.back()};r.las_codes={r.las_codes.back()};h.frame_points=1;
            r.frame.points[0].timestamp_ns-=1000000000;
            const auto late=map.update(r);oracle.verify(r,late,map.size());require(late.empty(),"late observation must be ignored");
            const auto snapshot=map.snapshot(as_of+1000000);require(snapshot==oracle.snapshot(as_of+1000000),"final snapshot equality");
            h.crs="UNSAFE:other-frame";bool rejected=false;
            try {(void)map.update(r);}catch(const std::exception &){rejected=true;}
            require(rejected && map.snapshot(as_of+1000000)==snapshot,"coordinate mismatch must reject without mutation");
            const auto counts=map.counters();
            require(counts.removals>0 && counts.existing_updates>0 && counts.late_rejections==1 && counts.peak_cells==capacity,"real churn counters");
            const auto elapsed=std::chrono::duration<double>(std::chrono::steady_clock::now()-start).count();
            cases.push_back({{"capacity",capacity},{"status","PASS"},{"operations",counts.json()},{"wall_seconds_including_oracle_snapshots",elapsed},
                {"model_free_observations_per_second",counts.observations/elapsed}});
        }
        const auto receipt=Json{{"schema","synrex.cpu_map_churn.v1"},{"status","PASS"},
            {"scope","Model-free map-only test. Moving coordinates, new cells, eviction, repeats, class updates, late points and aging; NOT neural classification throughput"},{"cases",cases}};
        if(argc==2)write_new(argv[1],receipt.dump(2)+"\n");
        std::cout<<"map_churn_oracle=PASS\n";return 0;
    }catch(const std::exception &e){std::cerr<<e.what()<<'\n';return 1;}
}
