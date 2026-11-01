// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Tuomas Pietilä
#include "synrex/point_wire.hpp"
#include <algorithm>
#include <cerrno>
#include <cmath>
#include <cstring>
#include <limits>
#include <poll.h>
#include <set>
#include <sys/socket.h>

namespace synrex::point::wire4 {
namespace {
constexpr std::size_t header_size = 12, identity_size = 100, source_size = 136;
constexpr std::uint64_t integer_max = 9007199254740991ULL;
constexpr unsigned batch_kind = 16, classification_kind = 17;
bool hash(std::string_view s) {
    return s.size() == 64 && std::all_of(s.begin(), s.end(), [](unsigned char c) {
        return (c >= '0' && c <= '9') || (c >= 'a' && c <= 'f');
    });
}
void put(std::string &s, std::uint64_t n, unsigned width) {
    for (unsigned i = 0; i < width; ++i)
        s.push_back(static_cast<char>(n >> (8 * i)));
}
struct Reader {
    std::string_view bytes;
    std::size_t offset{};
    std::string_view take(std::size_t n) {
        require(n <= bytes.size() - offset, "wire4 truncated payload");
        auto result = bytes.substr(offset, n);
        offset += n;
        return result;
    }
    std::uint64_t get(unsigned n) {
        std::uint64_t value{};
        unsigned i = 0;
        for (const unsigned char c : take(n))
            value |= std::uint64_t(c) << (8 * i++);
        return value;
    }
    bool done() const { return offset == bytes.size(); }
};
std::string finish(const Binding &b, unsigned kind, std::string payload) {
    require(payload.size() + header_size <= b.max_packet_bytes(), "wire4 packet cap");
    std::string out = "SRX4";
    out.push_back(4);
    out.push_back(static_cast<char>(kind));
    out.append(2, '\0');
    for (unsigned i = 4; i; --i)
        out.push_back(static_cast<char>(payload.size() >> (8 * (i - 1))));
    out += payload;
    return out;
}
std::uint32_t header(const Binding &b, unsigned kind, std::string_view bytes) {
    require(bytes.size() >= header_size && bytes.substr(0, 4) == "SRX4" &&
                static_cast<unsigned char>(bytes[4]) == 4 &&
                static_cast<unsigned char>(bytes[5]) == kind && bytes[6] == 0 && bytes[7] == 0,
            "wire4 magic/version/kind/reserved");
    std::uint32_t n{};
    for (unsigned i = 8; i < 12; ++i)
        n = (n << 8) | static_cast<unsigned char>(bytes[i]);
    require(n <= b.max_packet_bytes() - header_size, "wire4 advertised packet cap");
    return n;
}
std::string identity(const Binding &b) { return b.descriptor_sha256() + b.session_id(); }
Reader payload(const Binding &b, unsigned kind, std::string_view bytes) {
    require(header(b, kind, bytes) == bytes.size() - header_size, "wire4 length/trailing bytes");
    Reader r{bytes.substr(header_size)};
    require(r.take(identity_size) == identity(b), "wire4 descriptor/session mismatch");
    return r;
}
void put_source(std::string &out, const SourceIdentity &s) {
    out += s.sha256;
    out += s.header_sha256;
    put(out, s.rows, 8);
}
SourceIdentity get_source(const Binding &b, Reader &r) {
    SourceIdentity s{std::string(r.take(64)), std::string(r.take(64)), r.get(8)};
    b.validate_source(s);
    return s;
}
void validate_batch(const Binding &b, const PreparedBatch &v) {
    b.validate_source(v.source);
    const auto n = v.representative_source_rows.size(), count = v.current_rows.size();
    const auto &c = b.contract();
    require(n && n <= c.max_representatives && count && count <= c.max_source_rows &&
                v.inverse.size() == count && v.coord.size() == n * 3 &&
                v.grid_coord.size() == n * 3 && v.feat.size() == n * 4 &&
                48 * n + 8 <= c.max_input_bytes && count <= c.max_output_bytes / 10 &&
                2 * n <= c.max_output_bytes && v.cutoff_ns <= integer_max &&
                hash(v.logical_batch_sha256), "wire4 batch dimensions/limits");
    require(header_size + identity_size + source_size + 88 + count * 12 + n * 48 <=
                b.max_packet_bytes(), "wire4 batch packet cap");
    std::set<std::uint64_t> current, representatives;
    for (std::size_t i = 0; i < count; ++i)
        require(v.current_rows[i] < v.source.rows && current.insert(v.current_rows[i]).second &&
                    v.inverse[i] < n, "wire4 current row/inverse");
    for (const auto row : v.representative_source_rows)
        require(row < v.source.rows && representatives.insert(row).second,
                "wire4 representative row");
    for (const auto value : v.coord)
        require(std::isfinite(value), "wire4 nonfinite coordinate");
    for (const auto value : v.feat)
        require(std::isfinite(value), "wire4 nonfinite feature");
    for (const auto value : v.grid_coord)
        require(value >= 0, "wire4 negative lattice");
}
void put_float(std::string &out, float value) {
    static_assert(sizeof(float) == 4 && std::numeric_limits<float>::is_iec559,
                  "wire4 requires IEEE754 float32");
    std::uint32_t bits{};
    std::memcpy(&bits, &value, 4);
    put(out, bits, 4);
}
float get_float(Reader &r) {
    const auto bits = static_cast<std::uint32_t>(r.get(4));
    float value{};
    std::memcpy(&value, &bits, 4);
    require(std::isfinite(value), "wire4 nonfinite float");
    return value;
}
void validate_classification(const Binding &b, const Classification &v) {
    b.validate_source(v.source);
    const auto count = v.rows.size();
    require(count && count <= b.contract().max_source_rows &&
                count <= b.contract().max_output_bytes / 10 && v.semantic_ids.size() == count && v.validity.size() == count &&
                v.las_codes.size() == count && v.epoch && v.epoch <= integer_max &&
                v.request_id && v.request_id <= integer_max && hash(v.logical_batch_sha256) &&
                hash(v.authority_sha256), "wire4 classification shape/identity");
    require(header_size + identity_size + source_size + 152 + count * 11 <=
                b.max_packet_bytes(), "wire4 classification packet cap");
    std::set<std::uint64_t> rows;
    for (std::size_t i = 0; i < count; ++i)
        require(v.rows[i] < v.source.rows && rows.insert(v.rows[i]).second && v.validity[i] == 1 &&
                    v.semantic_ids[i] < b.contract().las_projection.size() &&
                    v.las_codes[i] == b.contract().las_projection[v.semantic_ids[i]],
                "wire4 classification row/semantic/projection");
}
void control_body(const Binding &b, Control kind, const Json &body) {
    const auto k = static_cast<unsigned>(kind);
    require(k >= 1 && k <= 7, "wire4 unsupported control");
    if (kind == Control::descriptor)
        require(body == b.descriptor(), "wire4 unexpected descriptor");
    else if (kind == Control::metadata_ack || kind == Control::ready_ack ||
             kind == Control::completion_ack) {
        require(body.is_string() && hash(body.get<std::string>()), "wire4 ACK hash");
        if (kind == Control::metadata_ack)
            require(body == b.descriptor_sha256(), "wire4 metadata ACK identity");
    } else if (kind == Control::source) {
        require(body.is_object() && body.size() == 3 && body.contains("sha256") &&
                    body.contains("header_sha256") && body.contains("rows") &&
                    body.at("sha256").is_string() && body.at("header_sha256").is_string() &&
                    body.at("rows").is_number_unsigned(), "wire4 source control");
        b.validate_source({body.at("sha256"), body.at("header_sha256"), body.at("rows")});
    } else {
        const auto key = kind == Control::initialization ? "initialization_sha256" : "completion_sha256";
        require(body.is_object() && body.size() == 2 && body.contains(key) &&
                    body.contains("receipt") && body.at(key).is_string() &&
                    hash(body.at(key).get<std::string>()) && body.at("receipt").is_object() &&
                    digest_bytes(body.at("receipt").dump(-1, ' ', true)) == body.at(key),
                "wire4 receipt commitment");
        const auto &receipt = body.at("receipt");
        if (kind == Control::initialization)
            require(receipt.value("schema", "") == "synrex.point.initialization.v1" &&
                        receipt.value("package_manifest_sha256", "") ==
                            b.descriptor().at("package_manifest_sha256") &&
                        receipt.value("point_contract_sha256", "") ==
                            b.descriptor().at("point_contract_sha256") &&
                        receipt.value("environment_sha256", "") == b.contract().environment_sha256 &&
                        receipt.value("worker_source_sha256", "") == b.worker_sha256() &&
                        receipt.at("fixture") == b.descriptor().at("synthetic_fixture"),
                    "wire4 initialization receipt identity");
        else
            require(receipt.value("schema", "") == "synrex.point.native-completion.v1" &&
                        receipt.value("descriptor_sha256", "") == b.descriptor_sha256() &&
                        receipt.at("synthetic_fixture") == b.descriptor().at("synthetic_fixture") &&
                        receipt.at("qualification_only") == b.descriptor().at("qualification_only"),
                    "wire4 completion receipt identity");
    }
}
void wait_socket(int fd, short event, Connection::Time deadline) {
    while (true) {
        const auto now = Connection::Clock::now();
        require(now < deadline, "wire4 deadline expired");
        const auto milliseconds = std::chrono::duration_cast<std::chrono::milliseconds>(deadline - now);
        const auto timeout = static_cast<int>(std::min<std::int64_t>(milliseconds.count() + 1, 1000));
        pollfd p{fd, event, 0};
        const auto result = ::poll(&p, 1, timeout);
        if (result < 0 && errno == EINTR)
            continue;
        require(result >= 0, "wire4 poll failure");
        if (result) {
            require(!(p.revents & POLLNVAL), "wire4 invalid socket");
            return;
        }
    }
}
void transfer(int fd, char *data, std::size_t n, bool sending, Connection::Time deadline) {
    std::size_t offset{};
    while (offset < n) {
        wait_socket(fd, sending ? POLLOUT : POLLIN, deadline);
        require(Connection::Clock::now() < deadline, "wire4 deadline expired");
        const auto result = sending ? ::send(fd, data + offset, n - offset, MSG_DONTWAIT | MSG_NOSIGNAL)
                                    : ::recv(fd, data + offset, n - offset, MSG_DONTWAIT);
        if (result < 0 && (errno == EINTR || errno == EAGAIN || errno == EWOULDBLOCK))
            continue;
        require(result > 0, result == 0 ? "wire4 EOF/truncated message" : "wire4 socket failure");
        offset += static_cast<std::size_t>(result);
    }
}
} // namespace
Binding::Binding(const PointSession &session, const PointArtifact &artifact, std::uint64_t cap)
    : descriptor_(session.descriptor()), contract_(artifact.contract()),
      descriptor_sha256_(session.descriptor_sha256()), session_id_(descriptor_.at("session_id")),
      worker_sha256_(artifact.worker_sha256()), max_packet_bytes_(cap) {
    require(cap >= 1024 && cap <= 67108864 &&
                descriptor_.at("package_manifest_sha256") == artifact.manifest_sha256() &&
                descriptor_.at("point_contract_sha256") == artifact.contract_sha256() &&
                descriptor_.at("required_wire_version") == 4 &&
                digest_bytes(descriptor_.dump(-1, ' ', true)) == descriptor_sha256_,
            "wire4 retained artifact/descriptor binding");
}
void Binding::validate_source(const SourceIdentity &s) const {
    require(hash(s.sha256) && hash(s.header_sha256) && s.rows <= contract_.max_source_rows,
            "wire4 source bounds");
    bool found = false;
    for (const auto &v : descriptor_.at("sources"))
        found |= v.at("source_sha256") == s.sha256 && v.at("header_sha256") == s.header_sha256 &&
                 v.at("rows") == s.rows;
    require(found, "wire4 source outside native roster");
}
std::string encode_control(const Binding &b, Control kind, const Json &body) {
    control_body(b, kind, body);
    const auto text = body.dump(-1, ' ', true);
    require(text.size() <= 65536, "wire4 control cap");
    return finish(b, static_cast<unsigned>(kind), identity(b) + text);
}
Json decode_control(const Binding &b, Control kind, std::string_view bytes) {
    auto r = payload(b, static_cast<unsigned>(kind), bytes);
    const auto text = r.take(r.bytes.size() - r.offset);
    require(text.size() <= 65536, "wire4 control cap");
    std::vector<std::set<std::string>> keys;
    std::size_t nodes{};
    auto guard = [&](int depth, Json::parse_event_t event, Json &value) {
        require(depth <= 32 && ++nodes <= 100000, "wire4 JSON complexity");
        if (event == Json::parse_event_t::object_start)
            keys.emplace_back();
        if (event == Json::parse_event_t::key)
            require(keys.back().insert(value.get<std::string>()).second, "wire4 duplicate key");
        if (event == Json::parse_event_t::object_end)
            keys.pop_back();
        return true;
    };
    auto body = Json::parse(text, guard);
    control_body(b, kind, body);
    return body;
}
std::string encode_batch(const Binding &b, const PreparedBatch &v) {
    validate_batch(b, v);
    auto out = identity(b);
    put_source(out, v.source);
    put(out, v.cutoff_ns, 8);
    out += v.logical_batch_sha256;
    put(out, v.current_rows.size(), 8);
    put(out, v.representative_source_rows.size(), 8);
    for (const auto x : v.current_rows) put(out, x, 8);
    for (const auto x : v.inverse) put(out, x, 4);
    for (const auto x : v.representative_source_rows) put(out, x, 8);
    for (const auto x : v.coord) put_float(out, x);
    for (const auto x : v.grid_coord) put(out, static_cast<std::uint32_t>(x), 4);
    for (const auto x : v.feat) put_float(out, x);
    return finish(b, batch_kind, std::move(out));
}
PreparedBatch decode_batch(const Binding &b, std::string_view bytes) {
    auto r = payload(b, batch_kind, bytes);
    PreparedBatch v;
    v.source = get_source(b, r);
    v.cutoff_ns = r.get(8);
    v.logical_batch_sha256 = r.take(64);
    const auto count = r.get(8), n = r.get(8);
    require(count && count <= b.contract().max_source_rows && n && n <= b.contract().max_representatives &&
                12 * count + 48 * n == r.bytes.size() - r.offset, "wire4 batch lengths before allocation");
    for (std::uint64_t i = 0; i < count; ++i) v.current_rows.push_back(r.get(8));
    for (std::uint64_t i = 0; i < count; ++i) v.inverse.push_back(r.get(4));
    for (std::uint64_t i = 0; i < n; ++i) v.representative_source_rows.push_back(r.get(8));
    for (std::uint64_t i = 0; i < n * 3; ++i) v.coord.push_back(get_float(r));
    for (std::uint64_t i = 0; i < n * 3; ++i) {
        const auto x = r.get(4);
        require(x <= 2147483647, "wire4 signed lattice overflow");
        v.grid_coord.push_back(static_cast<std::int32_t>(x));
    }
    for (std::uint64_t i = 0; i < n * 4; ++i) v.feat.push_back(get_float(r));
    require(r.done(), "wire4 trailing batch");
    validate_batch(b, v);
    return v;
}
std::string encode_classification(const Binding &b, const Classification &v) {
    validate_classification(b, v);
    auto out = identity(b);
    put_source(out, v.source);
    put(out, v.epoch, 8); put(out, v.request_id, 8);
    out += v.logical_batch_sha256; out += v.authority_sha256;
    put(out, v.rows.size(), 8);
    for (const auto x : v.rows) put(out, x, 8);
    for (const auto *values : {&v.semantic_ids, &v.validity, &v.las_codes})
        out.append(values->begin(), values->end());
    return finish(b, classification_kind, std::move(out));
}
Classification decode_classification(const Binding &b, std::string_view bytes) {
    auto r = payload(b, classification_kind, bytes);
    Classification v;
    v.source = get_source(b, r);
    v.epoch = r.get(8); v.request_id = r.get(8);
    v.logical_batch_sha256 = r.take(64); v.authority_sha256 = r.take(64);
    const auto count = r.get(8);
    require(count && count <= b.contract().max_source_rows && 11 * count == r.bytes.size() - r.offset,
            "wire4 classification lengths before allocation");
    for (std::uint64_t i = 0; i < count; ++i) v.rows.push_back(r.get(8));
    for (auto *values : {&v.semantic_ids, &v.validity, &v.las_codes}) {
        const auto bytes_part = r.take(count);
        values->assign(bytes_part.begin(), bytes_part.end());
    }
    validate_classification(b, v);
    return v;
}
Connection::Connection(int fd, Binding binding) : socket_(fd), binding_(std::move(binding)) {
    int type{}; socklen_t size = sizeof(type);
    require(fd >= 0 && ::getsockopt(fd, SOL_SOCKET, SO_TYPE, &type, &size) == 0 &&
                type == SOCK_STREAM, "wire4 connected stream socket required");
    sockaddr_storage peer{}; size = sizeof(peer);
    require(::getpeername(fd, reinterpret_cast<sockaddr *>(&peer), &size) == 0,
            "wire4 unconnected socket");
}
void Connection::fail() noexcept { failed_ = true; ::shutdown(socket_, SHUT_RDWR); }
void Connection::send(const std::string &bytes, Time deadline) {
    require(!failed_, "wire4 poisoned connection");
    transfer(socket_, const_cast<char *>(bytes.data()), bytes.size(), true, deadline);
    require(Clock::now() < deadline, "wire4 send deadline expired");
}
std::string Connection::receive(unsigned kind, Time deadline) {
    require(!failed_, "wire4 poisoned connection");
    std::string bytes(header_size, '\0');
    transfer(socket_, bytes.data(), bytes.size(), false, deadline);
    const auto n = header(binding_, kind, bytes);
    if (kind < batch_kind)
        require(n <= identity_size + 65536, "wire4 control allocation cap");
    bytes.resize(header_size + n);
    transfer(socket_, bytes.data() + header_size, n, false, deadline);
    return bytes;
}
void Connection::send_control(Control k, const Json &v, Time t) {
    try { send(encode_control(binding_, k, v), t); } catch (...) { fail(); throw; }
}
Json Connection::receive_control(Control k, Time t) {
    try { auto out = decode_control(binding_, k, receive(static_cast<unsigned>(k), t));
        require(Clock::now() < t, "wire4 decode deadline expired"); return out;
    } catch (...) { fail(); throw; }
}
void Connection::send_batch(const PreparedBatch &v, Time t) {
    try { send(encode_batch(binding_, v), t); } catch (...) { fail(); throw; }
}
PreparedBatch Connection::receive_batch(Time t) {
    try { auto out = decode_batch(binding_, receive(batch_kind, t));
        require(Clock::now() < t, "wire4 decode deadline expired"); return out;
    } catch (...) { fail(); throw; }
}
void Connection::send_classification(const Classification &v, Time t) {
    try { send(encode_classification(binding_, v), t); } catch (...) { fail(); throw; }
}
Classification Connection::receive_classification(Time t) {
    try { auto out = decode_classification(binding_, receive(classification_kind, t));
        require(Clock::now() < t, "wire4 decode deadline expired"); return out;
    } catch (...) { fail(); throw; }
}
} // namespace synrex::point::wire4
