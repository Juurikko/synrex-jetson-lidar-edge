// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Tuomas Pietilä
#include "synrex/transport.hpp"
#include <algorithm>
#include <cerrno>
#include <cmath>
#include <cstring>
#include <netinet/in.h>
#include <openssl/sha.h>
#include <poll.h>
#include <sys/socket.h>
#include <unistd.h>
#include <unordered_set>
namespace synrex::streaming::wire {
namespace {
// Check every condition, but construct an owning diagnostic string only on error.
// require(bool,const std::string&) otherwise allocates a temporary per scalar field.
void checked(bool ok, const char *message) {
    if (!ok)
        synrex::require(false, message);
}
struct Writer {
    Bytes b;
    void u(std::uint64_t x, unsigned n) {
        for (unsigned i = 0; i < n; ++i)
            b.push_back(x >> (i * 8));
    }
    void real(double d) {
        std::uint64_t x;
        std::memcpy(&x, &d, 8);
        u(x, 8);
    }
    void str(const std::string &s) {
        checked(!s.empty() && s.size() <= 128, "wire identifier length");
        for (unsigned char c : s)
            checked(c >= 32 && c < 127, "wire identifier ASCII");
        u(s.size(), 2);
        b.insert(b.end(), s.begin(), s.end());
    }
};
struct Reader {
    const Bytes &b;
    std::size_t at{};
    std::uint64_t u(unsigned n) {
        checked(n <= 8 && at + n <= b.size(), "truncated wire payload");
        std::uint64_t x = 0;
        for (unsigned i = 0; i < n; ++i)
            x |= std::uint64_t(b[at++]) << (8 * i);
        return x;
    }
    double real() {
        auto x = u(8);
        double d;
        std::memcpy(&d, &x, 8);
        return d;
    }
    std::string str() {
        const auto n = u(2);
        checked(n > 0 && n <= 128 && at + n <= b.size(), "wire string budget");
        std::string s(b.begin() + at, b.begin() + at + n);
        at += n;
        for (unsigned char c : s)
            checked(c >= 32 && c < 127, "wire identifier ASCII");
        return s;
    }
    void end() { checked(at == b.size(), "trailing wire bytes"); }
};
void validate(const Input &in) {
    const auto &f = in.frame;
    const auto &h = f.header;
    checked(unsigned(in.coordinates) >= 1 && unsigned(in.coordinates) <= 4,
            "unknown coordinate domain");
    checked(h.taxonomy == "synrex.fractal7.v1" || h.taxonomy == "synrex.synthetic3.v1",
            "wire taxonomy");
    checked(h.chunk_index == 0 && h.chunk_count == 1 && h.frame_points == f.points.size() &&
                !f.points.empty() && f.points.size() <= 65536,
            "only bounded complete frames accepted");
    checked(h.points.empty() && f.first_sequence == h.sequence && f.last_sequence == h.sequence,
            "wire frame metadata");
    checked(h.frame_end_ns >= h.frame_start_ns &&
                h.frame_end_ns - h.frame_start_ns <= 1000000000ULL,
            "wire frame timestamp span");
    checked(in.coordinates != Coordinates::projected || h.crs.rfind("EPSG:", 0) == 0,
            "projected coordinates require explicit EPSG CRS");
    checked(in.coordinates == Coordinates::projected || h.crs.rfind("LOCAL:", 0) == 0,
            "local/world coordinates require explicit LOCAL CRS");
    std::unordered_set<std::uint64_t> ids;
    ids.reserve(f.points.size());
    for (const auto &p : f.points) {
        checked(ids.insert(p.id).second, "duplicate source ID in frame");
        checked(p.timestamp_ns >= h.frame_start_ns && p.timestamp_ns <= h.frame_end_ns,
                "wire point time outside frame");
        for (double x : p.xyz)
            checked(std::isfinite(x) && std::abs(x) <= 1e9, "unsafe wire coordinate");
        checked(p.return_number.has_value() == p.return_count.has_value(),
                "incomplete return attributes");
        if (p.return_number)
            checked(*p.return_number > 0 && *p.return_number <= *p.return_count &&
                        *p.return_count <= 15,
                    "invalid wire returns");
    }
}
void input(Writer &w, const Input &in) {
    validate(in);
    const auto &h = in.frame.header;
    for (const auto *s : {&h.sensor_id, &h.stream_id, &h.coordinate_frame, &h.crs, &h.taxonomy})
        w.str(*s);
    w.u(unsigned(in.coordinates), 1);
    w.u(1, 1);
    w.u(1, 1);
    w.u(1, 1); // UNIX ns, metres, complete
    for (auto x : {h.sequence, h.frame_id, h.frame_start_ns, h.frame_end_ns, in.replay_steady_ns})
        w.u(x, 8);
    w.u(in.frame.points.size(), 4);
    for (const auto &p : in.frame.points) {
        w.u(p.id, 8);
        w.u(p.timestamp_ns, 8);
        for (double x : p.xyz)
            w.real(x);
        w.u(p.intensity.value_or(0), 2);
        w.u(p.return_number.value_or(0), 1);
        w.u(p.return_count.value_or(0), 1);
        w.u((p.intensity ? 1 : 0) | (p.return_number ? 2 : 0), 1);
        w.u(0, 3);
    }
}
Input input(Reader &r) {
    Input in;
    auto &f = in.frame;
    auto &h = f.header;
    h.sensor_id = r.str();
    h.stream_id = r.str();
    h.coordinate_frame = r.str();
    h.crs = r.str();
    h.taxonomy = r.str();
    in.coordinates = static_cast<Coordinates>(r.u(1));
    checked(r.u(1) == 1 && r.u(1) == 1 && r.u(1) == 1, "unsupported clock, units or completion");
    h.sequence = r.u(8);
    h.frame_id = r.u(8);
    h.frame_start_ns = r.u(8);
    h.frame_end_ns = r.u(8);
    in.replay_steady_ns = r.u(8);
    const auto n = r.u(4);
    checked(n > 0 && n <= 65536 && n * 48 <= r.b.size() - r.at, "wire point count budget");
    h.chunk_count = 1;
    h.frame_points = n;
    f.first_sequence = f.last_sequence = h.sequence;
    f.points.reserve(n);
    for (std::size_t i = 0; i < n; ++i) {
        SourcePoint p;
        p.id = r.u(8);
        p.timestamp_ns = r.u(8);
        for (auto &x : p.xyz)
            x = r.real();
        const auto intensity = r.u(2), number = r.u(1), count = r.u(1), flags = r.u(1);
        checked(flags <= 3 && r.u(3) == 0, "unknown point flags/reserved bits");
        checked((flags & 1) || !intensity, "noncanonical absent intensity");
        checked((flags & 2) || (!number && !count), "noncanonical absent returns");
        if (flags & 1)
            p.intensity = intensity;
        if (flags & 2) {
            p.return_number = number;
            p.return_count = count;
        }
        f.points.push_back(p);
    }
    validate(in);
    return in;
}
void wait(int fd, short events, std::uint64_t until) {
    while (true) {
        const auto now = steady_ns();
        checked(now < until, "socket operation deadline");
        pollfd p{fd, events, 0};
        const int n = ::poll(&p, 1, std::min<std::uint64_t>((until - now + 999999) / 1000000, 100));
        if (n < 0 && errno == EINTR)
            continue;
        checked(n >= 0, "socket poll failure");
        if (n > 0)
            return;
    }
}
bool read(int fd, std::uint8_t *p, std::size_t size, std::uint64_t until, bool eof,
          std::uint64_t *first_read = nullptr) {
    std::size_t at = 0;
    while (at < size) {
        wait(fd, POLLIN, until);
        const auto read_start = steady_ns();
        const auto n = ::recv(fd, p + at, size - at, MSG_DONTWAIT);
        if (n == 0) {
            checked(eof && at == 0, "truncated wire envelope/payload");
            return false;
        }
        if (n < 0 && (errno == EAGAIN || errno == EWOULDBLOCK || errno == EINTR))
            continue;
        checked(n > 0, "socket receive failure");
        if (at == 0 && first_read)
            *first_read = read_start;
        at += n;
    }
    return true;
}
void write(int fd, const std::uint8_t *p, std::size_t size, std::uint64_t until,
           std::size_t fragment) {
    std::size_t at = 0;
    while (at < size) {
        wait(fd, POLLOUT, until);
        const auto n = ::send(fd, p + at, std::min(size - at, fragment ? fragment : size),
                              MSG_NOSIGNAL | MSG_DONTWAIT);
        if (n < 0 && (errno == EAGAIN || errno == EWOULDBLOCK || errno == EINTR))
            continue;
        checked(n > 0, "socket send failure");
        at += n;
    }
}
Socket socket_new() {
    const auto fd = ::socket(AF_INET, SOCK_STREAM | SOCK_CLOEXEC, 0);
    checked(fd >= 0, "socket creation failed");
    Socket s(fd);
    const int one = 1;
    checked(setsockopt(fd, SOL_SOCKET, SO_REUSEADDR, &one, sizeof(one)) == 0,
            "socket option failed");
    // Explicit bounded kernel buffers, in addition to two-frame application queues.
    const int bytes = 256 * 1024;
    checked(setsockopt(fd, SOL_SOCKET, SO_SNDBUF, &bytes, sizeof(bytes)) == 0 &&
                setsockopt(fd, SOL_SOCKET, SO_RCVBUF, &bytes, sizeof(bytes)) == 0,
            "socket buffer setup");
    return s;
}
sockaddr_in address(std::uint16_t port) {
    sockaddr_in a{};
    a.sin_family = AF_INET;
    a.sin_addr.s_addr = htonl(INADDR_LOOPBACK);
    a.sin_port = htons(port);
    return a;
}
} // namespace
Bytes encode(const Input &in) {
    Writer w;
    w.b.reserve(std::min<std::size_t>(max_payload, in.frame.points.size() * 48 + 1024));
    input(w, in);
    return w.b;
}
Input decode(const Bytes &b) {
    checked(b.size() <= max_payload, "wire budget");
    Reader r{b};
    auto in = input(r);
    r.end();
    return in;
}
Bytes encode(const Output &out) {
    checked(out.input.frame.points.size() == out.semantic_ids.size() &&
                out.semantic_ids.size() == out.las_codes.size(),
            "wire classification coverage");
    Writer w;
    w.b.reserve(std::min<std::size_t>(max_payload, out.input.frame.points.size() * 114 + 2048));
    input(w, out.input);
    for (auto t : {out.received_ns, out.published_ns, out.model_executions})
        w.u(t, 8);
    checked(
        out.model_manifest_sha256.size() == 64 &&
            std::all_of(out.model_manifest_sha256.begin(), out.model_manifest_sha256.end(),
                        [](char c) { return (c >= '0' && c <= '9') || (c >= 'a' && c <= 'f'); }),
        "output model identity required");
    w.str(out.model_manifest_sha256);
    for (std::size_t i = 0; i < out.semantic_ids.size(); ++i) {
        w.u(out.semantic_ids[i], 1);
        w.u(out.las_codes[i], 1);
    }
    checked(out.deltas.size() <= out.semantic_ids.size(), "delta budget");
    w.u(out.deltas.size(), 4);
    for (const auto &d : out.deltas) {
        for (auto x : d.key)
            w.u(static_cast<std::uint64_t>(x), 8);
        w.u(d.semantic_id, 1);
        w.u(bool(d.evicted), 1);
        w.u(0, 6);
        w.u(d.timestamp_ns, 8);
        for (auto x : d.evicted.value_or(std::array<std::int64_t, 3>{}))
            w.u(static_cast<std::uint64_t>(x), 8);
    }
    checked(w.b.size() <= max_payload, "wire output budget");
    return w.b;
}
Output decode_output(const Bytes &b) {
    checked(b.size() <= max_payload, "wire output budget");
    Reader r{b};
    Output out;
    out.input = input(r);
    out.received_ns = r.u(8);
    out.published_ns = r.u(8);
    out.model_executions = r.u(8);
    out.model_manifest_sha256 = r.str();
    checked(out.model_manifest_sha256.size() == 64, "output model digest length");
    Classification c;
    c.frame = out.input.frame;
    for (std::size_t i = 0; i < c.frame.points.size(); ++i) {
        out.semantic_ids.push_back(r.u(1));
        out.las_codes.push_back(r.u(1));
    }
    c.semantic_ids = out.semantic_ids;
    c.las_codes = out.las_codes;
    validate_classification(c);
    const auto n = r.u(4);
    checked(n <= c.frame.points.size() && n * 64 <= b.size() - r.at, "wire delta count");
    for (std::size_t i = 0; i < n; ++i) {
        MapDelta d{};
        for (auto &x : d.key) {
            auto u = r.u(8);
            std::memcpy(&x, &u, 8);
        }
        d.semantic_id = r.u(1);
        const auto flags = r.u(1);
        checked(flags <= 1 && r.u(6) == 0, "delta flags");
        d.timestamp_ns = r.u(8);
        std::array<std::int64_t, 3> key{};
        for (auto &x : key) {
            auto u = r.u(8);
            std::memcpy(&x, &u, 8);
        }
        checked(d.semantic_id < (c.frame.header.taxonomy == "synrex.fractal7.v1" ? 7 : 3),
                "delta class");
        if (flags)
            d.evicted = key;
        else
            checked(key == std::array<std::int64_t, 3>{}, "noncanonical absent eviction");
        out.deltas.push_back(d);
    }
    r.end();
    return out;
}
Bytes canonical(const Bytes &bytes, bool classified, bool input_only) {
    Reader r{bytes};
    for (int i = 0; i < 5; ++i)
        (void)r.str();
    const auto replay_offset = r.at + 4 + 32;
    (void)r.u(4);
    for (int i = 0; i < 5; ++i)
        (void)r.u(8);
    const auto points = r.u(4);
    checked(points && points <= 65536 && points * 48 <= bytes.size() - r.at,
            "canonical input budget");
    const auto input_end = r.at + points * 48;
    checked(!classified || input_end + 24 <= bytes.size(), "canonical output budget");
    Bytes result(bytes.begin(), input_only ? bytes.begin() + input_end : bytes.end());
    std::fill(result.begin() + replay_offset, result.begin() + replay_offset + 8, 0);
    if (classified && !input_only)
        std::fill(result.begin() + input_end, result.begin() + input_end + 16, 0);
    return result;
}
void stamp_publication(Bytes &bytes, std::uint64_t ns) {
    Reader r{bytes};
    for (int i = 0; i < 5; ++i)
        (void)r.str();
    (void)r.u(4);
    for (int i = 0; i < 5; ++i)
        (void)r.u(8);
    const auto n = r.u(4);
    checked(n > 0 && n <= 65536 && n * 48 + 24 <= bytes.size() - r.at,
            "publication payload budget");
    const auto offset = r.at + n * 48 + 8;
    for (unsigned i = 0; i < 8; ++i)
        bytes[offset + i] = ns >> (8 * i);
}
void Sequence::accept(const Input &in) {
    validate(in);
    ordered(in);
}
Input Sequence::decode_frame(const Bytes &bytes) {
    auto in = wire::decode(bytes);
    ordered(in);
    return in;
}
Output Sequence::decode_classified(const Bytes &bytes) {
    auto out = wire::decode_output(bytes);
    ordered(out.input);
    return out;
}
void Sequence::ordered(const Input &in) {
    if (previous_) {
        const auto &a = previous_->frame.header, &b = in.frame.header;
        checked(a.sensor_id == b.sensor_id && a.stream_id == b.stream_id &&
                    a.coordinate_frame == b.coordinate_frame && a.crs == b.crs &&
                    a.taxonomy == b.taxonomy && previous_->coordinates == in.coordinates,
                "wire identity changed; separate session required");
        checked(a.sequence != UINT64_MAX && b.sequence == a.sequence + 1 && b.frame_id > a.frame_id,
                "duplicate, missing or out-of-order sequence/frame");
        checked(b.frame_start_ns >= a.frame_end_ns, "sensor clock discontinuity");
        checked(!previous_->replay_steady_ns || in.replay_steady_ns >= previous_->replay_steady_ns,
                "replay clock discontinuity");
    }
    previous_ = Input{};
    previous_->frame.header = in.frame.header;
    previous_->coordinates = in.coordinates;
    previous_->replay_steady_ns = in.replay_steady_ns;
}
Input transform(const Input &in, const Transform &t) {
    validate(in);
    checked(t.version == 1 && t.from == in.frame.header.coordinate_frame && !t.to.empty() &&
                t.to != t.from && t.target != Coordinates::projected,
            "missing/unsupported transform");
    checked(t.valid_from_ns <= in.frame.header.frame_start_ns &&
                t.valid_until_ns >= in.frame.header.frame_end_ns,
            "pose missing at measurement time");
    for (double x : t.rotation)
        checked(std::isfinite(x), "invalid rotation");
    for (double x : t.translation)
        checked(std::isfinite(x), "invalid translation");
    for (int i = 0; i < 3; ++i)
        for (int j = 0; j < 3; ++j) {
            double dot = 0;
            for (int k = 0; k < 3; ++k)
                dot += t.rotation[i * 3 + k] * t.rotation[j * 3 + k];
            checked(std::abs(dot - (i == j ? 1.0 : 0.0)) <= 1e-12, "non-rigid rotation");
        }
    const auto &m = t.rotation;
    const double determinant = m[0] * (m[4] * m[8] - m[5] * m[7]) -
                               m[1] * (m[3] * m[8] - m[5] * m[6]) +
                               m[2] * (m[3] * m[7] - m[4] * m[6]);
    checked(std::abs(determinant - 1) <= 1e-12, "reflection is not a pose");
    auto out = in;
    out.coordinates = t.target;
    out.frame.header.coordinate_frame = t.to;
    out.frame.header.crs = "LOCAL:" + t.to;
    for (auto &p : out.frame.points) {
        const auto xyz = p.xyz;
        for (int i = 0; i < 3; ++i) {
            p.xyz[i] = t.translation[i];
            for (int j = 0; j < 3; ++j)
                p.xyz[i] += t.rotation[i * 3 + j] * xyz[j];
        }
    }
    validate(out);
    return out;
}
Socket::~Socket() {
    if (fd_ >= 0)
        ::close(fd_);
}
Socket::Socket(Socket &&s) noexcept : fd_(s.fd_) { s.fd_ = -1; }
Socket &Socket::operator=(Socket &&s) noexcept {
    if (this != &s) {
        if (fd_ >= 0)
            ::close(fd_);
        fd_ = s.fd_;
        s.fd_ = -1;
    }
    return *this;
}
void Socket::shutdown() {
    if (fd_ >= 0)
        ::shutdown(fd_, SHUT_RDWR);
}
Socket listen_loopback(std::uint16_t port) {
    auto s = socket_new();
    const auto a = address(port);
    checked(::bind(s.fd(), reinterpret_cast<const sockaddr *>(&a), sizeof(a)) == 0 &&
                ::listen(s.fd(), 2) == 0,
            "loopback bind/listen failed");
    return s;
}
Socket connect_loopback(std::uint16_t port) {
    auto s = socket_new();
    const auto a = address(port);
    checked(::connect(s.fd(), reinterpret_cast<const sockaddr *>(&a), sizeof(a)) == 0,
            "loopback connect failed");
    return s;
}
Socket accept(const Socket &s, unsigned ms) {
    wait(s.fd(), POLLIN, steady_ns() + std::uint64_t(ms) * 1000000);
    const auto fd = ::accept4(s.fd(), nullptr, nullptr, SOCK_CLOEXEC);
    checked(fd >= 0, "accept failed");
    return Socket(fd);
}
std::optional<std::pair<Kind, Bytes>> receive(const Socket &s, unsigned ms,
                                              std::uint64_t *first_read) {
    const auto until = steady_ns() + std::uint64_t(ms) * 1000000;
    Bytes header(56);
    if (!read(s.fd(), header.data(), header.size(), until, true, first_read))
        return {};
    Reader r{header};
    checked(r.u(8) == 0x3145524957585253ULL && r.u(2) == 2, "unknown wire magic/version");
    const auto kind = r.u(2), n = r.u(4);
    checked(kind >= 1 && kind <= 4 && n <= max_payload && r.u(8) == 0, "wire kind/length/reserved");
    Bytes b(n);
    read(s.fd(), b.data(), b.size(), until, false);
    std::array<unsigned char, 32> digest{};
    SHA256(b.data(), b.size(), digest.data());
    checked(std::equal(digest.begin(), digest.end(), header.begin() + 24),
            "wire payload checksum mismatch");
    return std::make_pair(static_cast<Kind>(kind), std::move(b));
}
void send(const Socket &s, Kind kind, const Bytes &b, unsigned ms, std::size_t fragment) {
    checked(b.size() <= max_payload && unsigned(kind) >= 1 && unsigned(kind) <= 4,
            "wire send budget");
    Writer w;
    w.u(0x3145524957585253ULL, 8);
    w.u(2, 2);
    w.u(unsigned(kind), 2);
    w.u(b.size(), 4);
    w.u(0, 8);
    std::array<unsigned char, 32> digest{};
    SHA256(b.data(), b.size(), digest.data());
    w.b.insert(w.b.end(), digest.begin(), digest.end());
    const auto until = steady_ns() + std::uint64_t(ms) * 1000000;
    write(s.fd(), w.b.data(), w.b.size(), until, fragment);
    write(s.fd(), b.data(), b.size(), until, fragment);
}
Bytes json_bytes(const Json &j) {
    const auto s = j.dump();
    return Bytes(s.begin(), s.end());
}
Json json(const Bytes &b) {
    checked(b.size() <= max_payload, "wire JSON budget");
    std::vector<std::set<std::string>> seen;
    return Json::parse(b, [&](int depth, Json::parse_event_t event, Json &value) {
        checked(depth <= 16, "wire JSON nesting budget");
        if (event == Json::parse_event_t::object_start)
            seen.emplace_back();
        if (event == Json::parse_event_t::key)
            checked(seen.back().insert(value.get<std::string>()).second, "duplicate wire JSON key");
        if (event == Json::parse_event_t::object_end)
            seen.pop_back();
        return true;
    });
}
} // namespace synrex::streaming::wire
