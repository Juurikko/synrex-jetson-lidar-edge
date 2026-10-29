// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Tuomas Pietilä
#include "synrex/service.hpp"
#include "synrex/benchmark.hpp"
#include <future>
#include <iostream>
#include <sstream>
using namespace synrex;
using namespace synrex::streaming;
namespace {
template <typename F> void rejects(F f) {
    bool rejected = false;
    try { f(); } catch (const std::exception &) { rejected = true; }
    require(rejected, "unsafe input was accepted");
}
Frame frame(const std::vector<Chunk> &chunks) {
    Assembler a;
    std::optional<Frame> result;
    for (const auto &chunk : chunks) result = a.accept(chunk, steady_ns());
    require(bool(result), "complete frame expected");
    return std::move(*result);
}
// Model-free orchestration fixture only. These tests never claim neural/GPU inference.
class FixtureSession final : public InferenceSession {
    Contract c_;
    std::shared_future<void> release_;
    std::promise<void> *entered_{};
    std::uint64_t windows_{};
  public:
    FixtureSession(const Json &spec, std::shared_future<void> release = {}, std::promise<void> *entered = nullptr)
        : c_(spec), release_(release), entered_(entered) {}
    const Contract &contract() const override { return c_; }
    Json identity() const override { return {{"backend", "MODEL_FREE_TEST_ONLY"}, {"windows", windows_}}; }
    WindowResult run(const std::vector<Point> &p, const fs::path &) override {
        if (entered_) { entered_->set_value(); entered_ = nullptr; }
        if (release_.valid()) release_.wait();
        ++windows_;
        WindowResult r;
        for (std::size_t i = 0; i < p.size(); ++i) {
            const auto id = static_cast<std::uint8_t>(i % c_.las_classes.size());
            r.semantic_ids.push_back(id); r.las_codes.push_back(c_.las_classes[id]);
        }
        r.timings = {{"scope", "MODEL_FREE_TEST_ONLY"}};
        return r;
    }
};
}
int main(int argc, char **argv) {
    try {
        require(argc == 2, "dense spec required");
        const auto spec = read_json(argv[1]);
        { LatencyHistogram histogram;
          for (unsigned i = 0; i < 100000; ++i) histogram.add(1000000);
          const auto report = histogram.json();
          require(report["count"] == 100000 && histogram.quantile(.99) >= .001 &&
                      histogram.quantile(.99) <= .00103125, "bounded all-sample histogram accuracy");
          histogram.add(500000000); require(histogram.json()["maximum_seconds"] == .5, "tail latency preserved"); }

        const auto chunks = synthetic(7, 1, 13, 5, "synrex.synthetic3.v1");
        require(chunks.size() == 3 && serialize(chunks.front()) ==
                    serialize(synthetic(7, 1, 13, 5, "synrex.synthetic3.v1").front()), "deterministic seed");
        for (const auto &c : chunks) require(serialize(parse(serialize(c))) == serialize(c), "exact roundtrip");
        const auto encoded = serialize(chunks.front());
        auto j = Json::parse(encoded);
        require(j["points"][0]["id"].get<std::uint64_t>() > (1ULL << 53), "uint64 ID fixture");
        for (const auto &field : {"schema", "units", "clock", "taxonomy"}) {
            auto bad = j; bad[field] = "unknown"; rejects([&] { (void)parse(bad.dump()); });
        }
        { auto bad = j; bad["extension"] = 1; rejects([&] { (void)parse(bad.dump()); }); }
        { auto bad = j; bad["sequence"] = -1; rejects([&] { (void)parse(bad.dump()); }); }
        { auto bad = j; bad["sequence"] = 1.5; rejects([&] { (void)parse(bad.dump()); }); }
        { auto bad = j; bad["frame_end_ns"] = 0; rejects([&] { (void)parse(bad.dump()); }); }
        { auto bad = j; bad["points"][0]["timestamp_ns"] = 0; rejects([&] { (void)parse(bad.dump()); }); }
        { auto bad = j; bad["points"][0]["xyz"][0] = "NaN"; rejects([&] { (void)parse(bad.dump()); }); }
        { auto bad = j; bad["points"][0]["intensity"] = 65536; rejects([&] { (void)parse(bad.dump()); }); }
        { auto bad = j; bad["points"][0]["return_count"] = 0; rejects([&] { (void)parse(bad.dump()); }); }
        { auto bad = j; bad["points"][0]["return_number"] = nullptr; rejects([&] { (void)parse(bad.dump()); }); }
        { auto bad = j; bad["points"][1]["id"] = bad["points"][0]["id"]; rejects([&] { (void)parse(bad.dump()); }); }
        { auto bad = j; bad["points"][0]["intensity"] = nullptr;
          bad["points"][0]["return_number"] = nullptr; bad["points"][0]["return_count"] = nullptr;
          require(!parse(bad.dump()).points[0].intensity, "optional attributes preserved"); }
        rejects([&] { (void)parse("{\"schema\":1,\"schema\":2}"); });
        rejects([&] { (void)parse(std::string(1024 * 1024 + 1, ' ')); });
        { Limits tiny; tiny.max_line_bytes = 8; std::istringstream input("123456789"); std::string line;
          rejects([&] { (void)read_line(input, line, tiny); }); }
        { std::istringstream input(encoded + "\n" + encoded); std::string line;
          require(read_line(input, line, {}) && line == encoded && read_line(input, line, {}) &&
                      !read_line(input, line, {}), "bounded reader including EOF without newline"); }
        auto f = frame(chunks);
        require(f.points.size() == 13 && f.first_sequence == 0 && f.last_sequence == 2 &&
                    f.points.front().id == chunks.front().points.front().id &&
                    f.header.points.empty(), "frame boundaries and identity");
        { Assembler a; (void)a.accept(chunks[0], 10); a.expire(2000000010);
          require(a.counters().partial_frames == 1 && a.counters().dropped_points == 5, "deadline partial accounting"); }
        { Assembler a; (void)a.accept(chunks[0], 10); a.finish();
          require(a.counters().partial_frames == 1 && a.counters().incoming_points == a.counters().dropped_points, "EOF partial"); }
        { Assembler a; (void)a.accept(chunks[0], 10); (void)a.accept(chunks[2], 11); a.finish();
          require(a.counters().missing_sequences == 1 && a.counters().dropped_points == 8, "packet gap coverage"); }
        { Assembler a; (void)a.accept(chunks[0], 10); (void)a.accept(chunks[0], 11);
          (void)a.accept(chunks[1], 12); const auto r = a.accept(chunks[2], 13);
          require(r && a.counters().late_points == 5 && a.counters().assembled_points == 13, "duplicate packet rejected"); }
        { Assembler a; auto bad = chunks[1]; bad.points[0].id = chunks[0].points[0].id;
          (void)a.accept(chunks[0], 10); (void)a.accept(bad, 11); a.finish();
          require(a.counters().partial_frames == 1 && a.counters().dropped_points == 10, "cross-chunk duplicate IDs"); }
        { Assembler a; auto bad = chunks[1]; bad.crs = "LOCAL:other";
          (void)a.accept(chunks[0], 10); (void)a.accept(bad, 11); a.finish();
          require(a.counters().dropped_points == 10, "coordinate frame isolation"); }
        { Queue<int> q(1); require(q.push(1) && !q.push(2, false), "bounded reject-new");
          auto blocked = std::async(std::launch::async, [&] { return q.push(3); });
          require(blocked.wait_for(std::chrono::milliseconds(20)) == std::future_status::timeout, "producer backpressure");
          const auto removed = q.cancel(); require(removed.size() == 1 && !blocked.get() && !q.pop(), "cancel wakes waiters"); }
        { Queue<int> q(1); require(q.push(1), "push"); q.close();
          require(q.pop() == 1 && !q.pop() && !q.push(2), "close drains and refuses submission"); }
        std::vector<Classification> outputs;
        { Service s(std::make_unique<FixtureSession>(spec), 1,
                    [&](const auto &r) { outputs.push_back(r); });
          const auto many = synthetic(7, 4, 13, 5, "synrex.synthetic3.v1");
          Assembler a;
          for (const auto &c : many) if (auto r = a.accept(c, steady_ns())) require(s.submit(std::move(*r)), "submit");
          s.finish(); const auto receipt = s.receipt();
          require(receipt["status"] == "PASS" && receipt["emitted_points"] == 52 &&
                      receipt["freshly_classified_points"] == 52 && outputs.size() == 4,
                  "persistent orchestration complete coverage"); }
        const auto ev = event(outputs.front());
        for (std::size_t i = 0; i < f.points.size(); ++i) {
            require(ev["points"][i]["id"] == f.points[i].id && ev["points"][i]["timestamp_ns"] == f.points[i].timestamp_ns &&
                        ev["points"][i]["xyz"] == Json(f.points[i].xyz) && ev["points"][i]["intensity"] == *f.points[i].intensity,
                    "classified event metadata/uint64 preservation");
        }
        { std::promise<void> entered, release; auto gate = release.get_future().share();
          Service s(std::make_unique<FixtureSession>(spec, gate, &entered), 1, [](const auto &) {});
          require(s.submit(f), "first submit"); entered.get_future().wait(); require(s.submit(f), "queued submit");
          require(!s.submit(f, false), "full rejects frame"); s.cancel(); release.set_value(); s.finish();
          auto r = s.receipt(); require(r["status"] == "FAIL" && r["submitted_points"] == 39 &&
              r["dropped_points"] == 39 && r["emitted_points"] == 0 && r["freshly_classified_points"] == 13,
              "cancel and backpressure point accounting"); }
        { Service s(std::make_unique<FixtureSession>(spec), 1, [](const auto &) { throw std::runtime_error("sink fails"); });
          require(s.submit(f), "sink fixture submit"); rejects([&] { s.finish(); });
          require(s.receipt()["status"] == "FAIL" && s.receipt()["dropped_points"] == 13, "sink failure propagates"); }
        { auto bad = f; bad.header.taxonomy = "synrex.fractal7.v1";
          Service s(std::make_unique<FixtureSession>(spec), 1, [](const auto &) {});
          require(s.submit(bad), "bad taxonomy submit"); rejects([&] { s.finish(); });
          require(s.receipt()["freshly_classified_points"] == 0, "silent ontology remap forbidden"); }
        { auto bad = f; bad.points[0].intensity.reset();
          Service s(std::make_unique<FixtureSession>(spec), 1, [](const auto &) {});
          require(s.submit(bad), "optional intensity submit"); rejects([&] { s.finish(); }); }
        { auto r = outputs.front(); r.frame.header.taxonomy = "synrex.fractal7.v1";
          for (std::size_t i = 0; i < r.semantic_ids.size(); ++i) {
              r.semantic_ids[i] = i % 7;
              r.las_codes[i] = std::vector<std::uint8_t>{1,2,200,6,9,17,201}[i % 7];
          }
          require(event(r)["points"][6]["semantic_id"] == 6, "all seven channel IDs retained");
          r.semantic_ids[0] = 7; rejects([&] { (void)event(r); }); }
        { SemanticMap map(1.0, 2); auto r = outputs.front();
          r.frame.points.resize(1); r.frame.header.frame_points = 1; r.semantic_ids.resize(1); r.las_codes.resize(1);
          r.frame.points[0].xyz = {-0.1, 0, 0};
          auto d = map.update(r); require(d.size() == 1 && d[0].key[0] == -1, "negative map floor");
          require(map.update(r).empty(), "unchanged cell produces no delta");
          const auto timestamp = r.frame.points[0].timestamp_ns;
          --r.frame.points[0].timestamp_ns; r.semantic_ids[0] = 2; r.las_codes[0] = 6;
          require(map.update(r).empty(), "late classification cannot replace newer map state");
          r.frame.points[0].timestamp_ns = timestamp;
          require(map.update(r).size() == 1, "class change emits delta");
          r.frame.points[0].xyz[0] = 1; (void)map.update(r);
          r.frame.points[0].xyz[0] = 2; d = map.update(r);
          require(map.size() == 2 && d[0].evicted && (*d[0].evicted)[0] == -1, "bounded map LRU eviction");
          require(map.snapshot(timestamp + 5)["cells"][0]["classification_age_ns"] == 5, "map observation age");
          r.frame.header.crs = "other"; rejects([&] { (void)map.update(r); }); }
        { auto r = outputs.front(); SemanticMap map(1.0, 8);
          r.semantic_ids[0] = 255; rejects([&] { (void)map.update(r); });
          require(map.size() == 0, "scalar validation rejects invalid channels before map mutation");
          r = outputs.front(); r.las_codes[0] = 255; rejects([&] { (void)map.update(r); });
          require(map.size() == 0, "scalar validation rejects projection mismatch before map mutation"); }
        std::cout << "point_stream_contract=PASS; GPU_INFERENCE=NOT_RUN; service fixtures=MODEL_FREE_TEST_ONLY\n";
        return 0;
    } catch (const std::exception &e) { std::cerr << "point_stream_contract=FAIL: " << e.what() << '\n'; return 1; }
}
