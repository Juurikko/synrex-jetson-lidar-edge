// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Tuomas Pietilä
// Independent Track B protocol/assembly implementation using only public R4 interfaces.
#include "synrex/stream.hpp"
#include <algorithm>
#include <cmath>
#include <istream>
#include <limits>

namespace synrex::streaming {
namespace {
void keys(const Json &j, std::initializer_list<const char *> names) {
    require(j.is_object() && j.size() == names.size(), "unknown/missing stream fields");
    for (auto name : names)
        require(j.contains(name), std::string("missing stream field: ") + name);
}
std::uint64_t integer(const Json &j, std::uint64_t cap = UINT64_MAX) {
    require(j.is_number_unsigned() || (j.is_number_integer() && j.get<std::int64_t>() >= 0),
            "unsigned integer required");
    const auto n = j.get<std::uint64_t>();
    require(n <= cap, "stream integer budget exceeded");
    return n;
}
std::string identifier(const Json &j) {
    require(j.is_string(), "identifier must be string");
    const auto s = j.get<std::string>();
    require(!s.empty() && s.size() <= 128, "identifier length outside 1..128");
    for (unsigned char ch : s)
        require(ch >= 32 && ch < 127, "identifier must be printable ASCII");
    return s;
}
Json to_json(const Chunk &c) {
    Json points = Json::array();
    for (const auto &p : c.points)
        points.push_back({{"id", p.id}, {"timestamp_ns", p.timestamp_ns}, {"xyz", p.xyz},
                          {"intensity", p.intensity ? Json(*p.intensity) : Json(nullptr)},
                          {"return_number", p.return_number ? Json(*p.return_number) : Json(nullptr)},
                          {"return_count", p.return_count ? Json(*p.return_count) : Json(nullptr)}});
    return {{"schema", schema}, {"sensor_id", c.sensor_id}, {"stream_id", c.stream_id},
            {"sequence", c.sequence}, {"frame_id", c.frame_id},
            {"frame_start_ns", c.frame_start_ns}, {"frame_end_ns", c.frame_end_ns},
            {"chunk_index", c.chunk_index}, {"chunk_count", c.chunk_count},
            {"frame_points", c.frame_points}, {"units", "metres"},
            {"coordinate_frame", c.coordinate_frame}, {"crs", c.crs},
            {"clock", "unix_ns"}, {"taxonomy", c.taxonomy}, {"points", points}};
}
bool same_identity(const Chunk &a, const Chunk &b) {
    return a.sensor_id == b.sensor_id && a.stream_id == b.stream_id &&
           a.coordinate_frame == b.coordinate_frame && a.crs == b.crs && a.taxonomy == b.taxonomy;
}
}
void Limits::validate() const {
    require(max_line_bytes > 0 && max_line_bytes <= 16 * 1024 * 1024 && max_chunk_points > 0 &&
                max_chunk_points <= max_frame_points && max_frame_points <= 1000000 &&
                max_chunks > 0 && max_chunks <= 65536 && max_frame_span_ns > 0 &&
                max_frame_span_ns <= 60000000000ULL && assembly_timeout_ns > 0 &&
                assembly_timeout_ns <= 60000000000ULL, "unsafe stream limits");
}
bool read_line(std::istream &in, std::string &line, const Limits &limits) {
    limits.validate();
    line.clear();
    char ch{};
    while (in.get(ch)) {
        if (ch == '\n')
            return true;
        require(line.size() < limits.max_line_bytes, "stream line byte budget exceeded");
        line.push_back(ch);
    }
    require(in.eof(), "stream read failure");
    return !line.empty();
}
Chunk parse(const std::string &line, const Limits &limits) {
    limits.validate();
    require(line.size() <= limits.max_line_bytes, "stream line byte budget exceeded");
    std::vector<std::set<std::string>> seen;
    auto callback = [&seen](int depth, Json::parse_event_t e, Json &v) {
        require(depth <= 12, "stream nesting budget exceeded");
        if (e == Json::parse_event_t::object_start) seen.emplace_back();
        if (e == Json::parse_event_t::key)
            require(seen.back().insert(v.get<std::string>()).second, "duplicate stream JSON key");
        if (e == Json::parse_event_t::object_end) seen.pop_back();
        return true;
    };
    const auto j = Json::parse(line, callback);
    keys(j, {"schema", "sensor_id", "stream_id", "sequence", "frame_id", "frame_start_ns",
             "frame_end_ns", "chunk_index", "chunk_count", "frame_points", "units",
             "coordinate_frame", "crs", "clock", "taxonomy", "points"});
    require(j.at("schema") == schema && j.at("units") == "metres" && j.at("clock") == "unix_ns",
            "unsupported stream schema, units or clock");
    Chunk c;
    c.sensor_id = identifier(j.at("sensor_id")); c.stream_id = identifier(j.at("stream_id"));
    c.coordinate_frame = identifier(j.at("coordinate_frame")); c.crs = identifier(j.at("crs"));
    c.taxonomy = identifier(j.at("taxonomy"));
    require(c.taxonomy == "synrex.synthetic3.v1" || c.taxonomy == "synrex.fractal7.v1",
            "unsupported explicit taxonomy");
    c.sequence = integer(j.at("sequence")); c.frame_id = integer(j.at("frame_id"));
    c.frame_start_ns = integer(j.at("frame_start_ns"));
    c.frame_end_ns = integer(j.at("frame_end_ns"));
    require(c.frame_end_ns >= c.frame_start_ns &&
                c.frame_end_ns - c.frame_start_ns <= limits.max_frame_span_ns, "unsafe frame span");
    c.chunk_index = integer(j.at("chunk_index"), limits.max_chunks - 1);
    c.chunk_count = integer(j.at("chunk_count"), limits.max_chunks);
    c.frame_points = integer(j.at("frame_points"), limits.max_frame_points);
    require(c.chunk_count > 0 && c.chunk_index < c.chunk_count && c.frame_points > 0 &&
                c.chunk_count <= c.frame_points &&
                c.frame_points <= c.chunk_count * limits.max_chunk_points, "unsafe chunk/frame sizes");
    const auto &points = j.at("points");
    require(points.is_array() && !points.empty() && points.size() <= limits.max_chunk_points &&
                points.size() <= c.frame_points, "invalid chunk point count");
    std::set<std::uint64_t> ids;
    for (const auto &v : points) {
        keys(v, {"id", "timestamp_ns", "xyz", "intensity", "return_number", "return_count"});
        SourcePoint p;
        p.id = integer(v.at("id")); p.timestamp_ns = integer(v.at("timestamp_ns"));
        require(p.timestamp_ns >= c.frame_start_ns && p.timestamp_ns <= c.frame_end_ns,
                "point timestamp outside frame");
        require(ids.insert(p.id).second, "duplicate source point ID");
        const auto &xyz = v.at("xyz");
        require(xyz.is_array() && xyz.size() == 3, "XYZ triple required");
        for (int a = 0; a < 3; ++a) {
            require(xyz[a].is_number(), "numeric XYZ required");
            p.xyz[a] = xyz[a].get<double>();
            require(std::isfinite(p.xyz[a]) && std::abs(p.xyz[a]) <= 1e9,
                    "nonfinite/out-of-range XYZ");
        }
        if (!v.at("intensity").is_null()) p.intensity = integer(v.at("intensity"), 65535);
        require(v.at("return_number").is_null() == v.at("return_count").is_null(),
                "return attributes must be both present or null");
        if (!v.at("return_number").is_null()) {
            p.return_number = integer(v.at("return_number"), 15);
            p.return_count = integer(v.at("return_count"), 15);
            require(*p.return_number > 0 && *p.return_number <= *p.return_count,
                    "invalid return attributes");
        }
        c.points.push_back(p);
    }
    return c;
}
std::string serialize(const Chunk &c, const Limits &limits) {
    const auto line = to_json(c).dump();
    (void)parse(line, limits); // Also validates direct C++ callers.
    return line;
}
Assembler::Assembler(Limits limits) : limits_(limits) { limits_.validate(); }
void Assembler::discard() {
    if (active_) {
        counters_.dropped_points += active_->points.size();
        ++counters_.partial_frames;
        active_.reset(); ids_.clear();
    }
}
void Assembler::expire(std::uint64_t now) {
    if (active_) {
        require(now >= active_->accepted_steady_ns, "monotonic time moved backwards");
        if (now - active_->accepted_steady_ns >= limits_.assembly_timeout_ns) discard();
    }
}
void Assembler::finish() { discard(); }
std::optional<Frame> Assembler::accept(Chunk c, std::uint64_t now) {
    (void)serialize(c, limits_);
    counters_.incoming_points += c.points.size();
    expire(now);
    auto reject = [&](bool late) -> std::optional<Frame> {
        counters_.dropped_points += c.points.size();
        if (late) counters_.late_points += c.points.size();
        ++counters_.rejected_chunks;
        return {};
    };
    if (identity_ && !same_identity(*identity_, c)) return reject(false);
    if (last_sequence_ && c.sequence <= *last_sequence_) return reject(true);
    if (last_sequence_ && c.sequence - *last_sequence_ != 1) {
        counters_.missing_sequences += c.sequence - *last_sequence_ - 1;
        discard();
    }
    last_sequence_ = c.sequence;
    if (active_ && c.frame_id != active_->header.frame_id) discard();
    if (!active_) {
        if (c.chunk_index != 0 || (last_frame_ && c.frame_id <= *last_frame_) ||
            (last_end_ && c.frame_start_ns < *last_end_)) return reject(true);
        identity_ = c; identity_->points.clear();
        last_frame_ = c.frame_id; last_end_ = c.frame_end_ns;
        active_ = Frame{c, {}, c.sequence, c.sequence, now};
        active_->header.points.clear();
        next_chunk_ = 0;
    }
    const auto &h = active_->header;
    if (c.chunk_index != next_chunk_ || c.chunk_count != h.chunk_count ||
        c.frame_points != h.frame_points || c.frame_start_ns != h.frame_start_ns ||
        c.frame_end_ns != h.frame_end_ns ||
        active_->points.size() + c.points.size() > h.frame_points) {
        discard(); return reject(false);
    }
    for (const auto &p : c.points)
        if (!ids_.insert(p.id).second) { discard(); return reject(false); }
    active_->last_sequence = c.sequence;
    active_->points.insert(active_->points.end(), c.points.begin(), c.points.end());
    ++next_chunk_;
    if (next_chunk_ != h.chunk_count) return {};
    if (active_->points.size() != h.frame_points) { discard(); return {}; }
    auto frame = std::move(active_);
    active_.reset(); ids_.clear();
    counters_.assembled_points += frame->points.size();
    return frame;
}
void validate_classification(const Classification &r) {
    require(r.frame.points.size() == r.semantic_ids.size() && r.las_codes.size() == r.semantic_ids.size(),
            "classification coverage mismatch");
    const auto &h = r.frame.header;
    require((h.taxonomy == "synrex.fractal7.v1" || h.taxonomy == "synrex.synthetic3.v1") &&
                h.frame_points == r.frame.points.size(), "invalid classified frame/taxonomy");
    static constexpr std::array<std::uint8_t, 7> fractal{1,2,200,6,9,17,201};
    static constexpr std::array<std::uint8_t, 3> synthetic{2,5,6};
    for (std::size_t i = 0; i < r.semantic_ids.size(); ++i) {
        const auto id = r.semantic_ids[i];
        require(id < (h.taxonomy == "synrex.fractal7.v1" ? 7 : 3), "unknown semantic channel");
        require(r.las_codes[i] == (h.taxonomy == "synrex.fractal7.v1" ? fractal.at(id) : synthetic.at(id)),
                "event semantic/LAS projection mismatch");
    }
}
Json event(const Classification &r) {
    validate_classification(r);
    auto h = r.frame.header;
    h.points = r.frame.points;
    Json j = to_json(h);
    j["schema"] = "synrex.classified_frame.v1";
    j["first_sequence"] = r.frame.first_sequence; j["last_sequence"] = r.frame.last_sequence;
    j.erase("sequence"); j.erase("chunk_index"); j.erase("chunk_count");
    j["mode"] = "all_points_freshly_classified";
    for (std::size_t i = 0; i < r.semantic_ids.size(); ++i) {
        j["points"][i]["semantic_id"] = r.semantic_ids[i];
        j["points"][i]["model_las_code"] = r.las_codes[i];
        j["points"][i]["confidence"] = nullptr;
    }
    j["timings"] = r.timings;
    return j;
}
std::vector<Chunk> synthetic(std::uint64_t seed, std::size_t frames, std::size_t points,
                             std::size_t chunk_points, const std::string &taxonomy) {
    Limits limits;
    require(frames > 0 && frames <= 1000 && points > 0 && points <= limits.max_frame_points &&
                chunk_points > 0 && chunk_points <= limits.max_chunk_points &&
                frames * points <= 1000000, "synthetic fixture budget exceeded");
    const auto count = (points + chunk_points - 1) / chunk_points;
    require(count <= limits.max_chunks, "synthetic chunk budget exceeded");
    auto random = [&seed]() { seed = seed * 6364136223846793005ULL + 1442695040888963407ULL; return seed; };
    std::vector<Chunk> out;
    for (std::size_t f = 0; f < frames; ++f) {
        for (std::size_t ch = 0; ch < count; ++ch) {
            Chunk c;
            c.sensor_id = "synthetic"; c.stream_id = "deterministic-r0";
            c.coordinate_frame = "synthetic_cartesian"; c.crs = "LOCAL:synthetic-metres";
            c.taxonomy = taxonomy; c.sequence = out.size(); c.frame_id = f;
            c.frame_start_ns = 1700000000000000000ULL + f * 100000000ULL;
            c.frame_end_ns = c.frame_start_ns + 99999999ULL;
            c.chunk_index = ch; c.chunk_count = count; c.frame_points = points;
            for (auto i = ch * chunk_points; i < std::min(points, (ch + 1) * chunk_points); ++i) {
                SourcePoint p;
                p.id = (1ULL << 54) + f * points + i; // Deliberately beyond JavaScript's exact integer range.
                p.timestamp_ns = c.frame_start_ns + i * 99999999ULL / points;
                for (int a = 0; a < 3; ++a) p.xyz[a] = (random() >> 32) % 1600 / 100.0;
                p.intensity = static_cast<std::uint16_t>(random() >> 32);
                p.return_number = 1; p.return_count = 2;
                c.points.push_back(p);
            }
            out.push_back(std::move(c));
        }
    }
    return out;
}
} // namespace synrex::streaming
