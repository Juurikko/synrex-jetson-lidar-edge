// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Tuomas Pietilä
#include "synrex/generic_transport.hpp"
#include <algorithm>
#include <array>
#include <cerrno>
#include <climits>
#include <openssl/rand.h>
#include <poll.h>
#include <set>
#include <stdexcept>
#include <sys/socket.h>

namespace synrex::generic {
namespace {
using Clock = Startup::Clock;
using Time = Startup::Time;
enum class Message : unsigned char {
    hello = 1,
    descriptor = 2,
    metadata_ack = 3,
    execution_ready = 4,
    execution_ready_ack = 5
};
void check(bool ok, const char *message) {
    if (!ok)
        throw std::runtime_error(message);
}
Json limit_json(const Limits &limits) {
    return {{"max_frame_points", limits.max_frame_points},
            {"max_frame_bytes", limits.max_frame_bytes},
            {"startup_timeout_ms", limits.startup_timeout_ms}};
}
std::uint64_t integer(const Json &value) {
    check(value.is_number_integer() &&
              (value.is_number_unsigned() || value.get<std::int64_t>() >= 0),
          "wire-v3 limit integer required");
    return value.get<std::uint64_t>();
}
Limits read_limits(const Json &value) {
    check(value.is_object() && value.size() == 3 && value.contains("max_frame_points") &&
              value.contains("max_frame_bytes") && value.contains("startup_timeout_ms"),
          "wire-v3 unsupported limit fields");
    Limits limits{integer(value.at("max_frame_points")), integer(value.at("max_frame_bytes")),
                  integer(value.at("startup_timeout_ms"))};
    limits.validate();
    return limits;
}
// Hello is tiny, flat and has an exact schema. Reject duplicate (including escaped) keys
// and nesting during bounded DOM parsing; Descriptor has its own bounded SAX guard.
Json small_json(const std::string &bytes) {
    std::vector<std::set<std::string>> keys;
    auto callback = [&keys](int depth, Json::parse_event_t event, Json &value) {
        check(depth <= 2, "wire-v3 hello nesting limit");
        if (event == Json::parse_event_t::object_start)
            keys.emplace_back();
        if (event == Json::parse_event_t::key)
            check(keys.back().insert(value.get<std::string>()).second,
                  "wire-v3 duplicate hello key");
        if (event == Json::parse_event_t::object_end)
            keys.pop_back();
        return true;
    };
    return Json::parse(bytes, callback);
}
Json hello_json(const std::string &bytes) {
    auto value = small_json(bytes);
    check(value.is_object() && value.size() == 2 && value.contains("schema") &&
              value.contains("limits") && value.at("schema") == "synrex.session.hello.v3",
          "wire-v3 unsupported hello schema/capability");
    return value;
}
bool challenge_valid(std::string_view value) {
    return value.size() == 32 && std::all_of(value.begin(), value.end(), [](unsigned char c) {
               return (c >= '0' && c <= '9') || (c >= 'a' && c <= 'f');
           });
}
std::string new_challenge() {
    std::array<unsigned char, 16> bytes{};
    check(RAND_bytes(bytes.data(), bytes.size()) == 1,
          "wire-v3 readiness challenge generation failed");
    const char hex[] = "0123456789abcdef";
    std::string encoded;
    for (const auto byte : bytes) {
        encoded += hex[byte >> 4];
        encoded += hex[byte & 15];
    }
    return encoded;
}
std::string readiness(const Descriptor &descriptor, const std::string &kind,
                      const std::string &challenge) {
    auto control = Json::parse(descriptor.control(kind));
    control["readiness_challenge"] = challenge;
    return control.dump();
}
std::size_t bound(Message message) {
    if (message == Message::descriptor)
        return Descriptor::max_bytes;
    if (message == Message::hello)
        return 512;
    return 1024;
}
} // namespace
namespace detail {
class ControlConnection {
    int socket_;
    const CpuAdmittedPackage &package_;
    Limits supported_, negotiated_{};
    std::string session_;
    std::uint64_t epoch_;
    bool authority_, poisoned_{};
    enum class Stage { fresh, admitted, initializing, ready } stage_{Stage::fresh};
    const Time metadata_deadline_;
    Time startup_deadline_{};
    std::unique_ptr<Descriptor> descriptor_;
    std::unique_ptr<Startup> startup_;
    void alive() const { check(!poisoned_, "wire-v3 connection permanently failed"); }
    void before(Time deadline) const {
        alive();
        check(Clock::now() < deadline, "wire-v3 absolute deadline expired");
    }
    void wait(short event, Time deadline) const {
        for (;;) {
            before(deadline);
            const auto remaining = deadline - Clock::now();
            const auto millis = std::chrono::duration_cast<std::chrono::milliseconds>(remaining);
            const auto timeout = std::max<std::int64_t>(1, millis.count() + 1);
            pollfd fd{socket_, event, 0};
            const int result =
                ::poll(&fd, 1, static_cast<int>(std::min<std::int64_t>(INT_MAX, timeout)));
            if (result < 0 && errno == EINTR)
                continue;
            check(result >= 0, "wire-v3 socket poll failed");
            before(deadline);
            if (result == 0)
                continue;
            check(!(fd.revents & POLLNVAL), "wire-v3 invalid socket");
            // A readable HUP may still carry a final complete envelope. recv detects EOF.
            if (fd.revents & (event | POLLERR | POLLHUP))
                return;
        }
    }
    void transfer(char *data, std::size_t size, bool sending, Time deadline) const {
        std::size_t offset{};
        while (offset < size) {
            before(deadline);
            const auto result =
                sending ? ::send(socket_, data + offset, size - offset, MSG_DONTWAIT | MSG_NOSIGNAL)
                        : ::recv(socket_, data + offset, size - offset, MSG_DONTWAIT);
            if (result < 0 && errno == EINTR)
                continue;
            if (result < 0 && (errno == EAGAIN || errno == EWOULDBLOCK)) {
                wait(sending ? POLLOUT : POLLIN, deadline);
                continue;
            }
            check(result > 0,
                  sending ? "wire-v3 socket write failed" : "wire-v3 EOF/truncated envelope");
            offset += static_cast<std::size_t>(result);
        }
        before(deadline);
    }
    void send(Message message, const std::string &bytes, Time deadline) const {
        check(!bytes.empty() && bytes.size() <= bound(message), "wire-v3 outgoing payload bound");
        const auto size = static_cast<std::uint32_t>(bytes.size());
        std::array<char, 12> header{'S',
                                    'R',
                                    'X',
                                    '3',
                                    3,
                                    static_cast<char>(message),
                                    0,
                                    0,
                                    static_cast<char>(size >> 24),
                                    static_cast<char>(size >> 16),
                                    static_cast<char>(size >> 8),
                                    static_cast<char>(size)};
        transfer(header.data(), header.size(), true, deadline);
        // send does not mutate its buffer; transfer uses char* for recv's destination.
        transfer(const_cast<char *>(bytes.data()), bytes.size(), true, deadline);
    }
    std::string receive(Message expected, Time deadline) const {
        std::array<char, 12> header{};
        transfer(header.data(), header.size(), false, deadline);
        check(header[0] == 'S' && header[1] == 'R' && header[2] == 'X' && header[3] == '3' &&
                  header[4] == 3 && header[5] == static_cast<char>(expected) && header[6] == 0 &&
                  header[7] == 0,
              "wire-v3 unsupported/malformed/out-of-order envelope");
        std::uint32_t size{};
        for (unsigned i = 8; i < 12; ++i)
            size = (size << 8) | static_cast<unsigned char>(header[i]);
        // Header and expected-kind bounds are validated BEFORE payload allocation.
        check(size > 0 && size <= bound(expected), "wire-v3 incoming payload bound");
        std::string bytes(size, '\0');
        transfer(bytes.data(), bytes.size(), false, deadline);
        return bytes;
    }
    template <class F> decltype(auto) guarded(F operation) {
        try {
            alive();
            return operation();
        } catch (const std::exception &error) {
            poisoned_ = true;
            if (startup_)
                startup_->fail(error.what());
            ::shutdown(socket_, SHUT_RDWR);
            throw;
        }
    }
    void bind_descriptor(Descriptor value) {
        descriptor_ = std::make_unique<Descriptor>(std::move(value));
        startup_ = std::make_unique<Startup>(*descriptor_);
        startup_->descriptor_fragment(descriptor_->bytes(), true);
    }

  public:
    ControlConnection(WireProtocol protocol, int socket, const CpuAdmittedPackage &package,
                      Limits supported, std::string session, std::uint64_t epoch, bool authority)
        : socket_(socket), package_(package), supported_(supported), session_(std::move(session)),
          epoch_(epoch), authority_(authority),
          metadata_deadline_(Clock::now() +
                             std::chrono::milliseconds(Descriptor::metadata_timeout_ms)) {
        check(protocol == WireProtocol::binary_v3, "explicit binary-v3 selection required");
        // All local package/config validation happens before the first socket operation.
        Descriptor::from_package(package_, supported_, session_, epoch_);
        int type{};
        socklen_t size = sizeof(type);
        check(::getsockopt(socket_, SOL_SOCKET, SO_TYPE, &type, &size) == 0 && type == SOCK_STREAM,
              "wire-v3 connected stream socket required");
    }
    void admission() {
        guarded([&] {
            check(stage_ == Stage::fresh, "wire-v3 duplicate metadata admission");
            before(metadata_deadline_);
            if (authority_) {
                const auto hello = hello_json(receive(Message::hello, metadata_deadline_));
                negotiated_ = Limits::negotiate(supported_, read_limits(hello.at("limits")));
                bind_descriptor(Descriptor::from_package(package_, negotiated_, session_, epoch_));
                send(Message::descriptor, descriptor_->bytes(), metadata_deadline_);
                const auto ack = receive(Message::metadata_ack, metadata_deadline_);
                before(metadata_deadline_);
                startup_->metadata_ack(ack);
            } else {
                send(Message::hello,
                     Json{{"schema", "synrex.session.hello.v3"}, {"limits", limit_json(supported_)}}
                         .dump(),
                     metadata_deadline_);
                const auto bytes = receive(Message::descriptor, metadata_deadline_);
                const auto received = Descriptor::parse(bytes);
                auto limits = received.spec().at("limits");
                limits.erase("max_metadata_bytes");
                limits.erase("metadata_timeout_ms");
                negotiated_ = read_limits(limits);
                check(limit_json(Limits::negotiate(supported_, negotiated_)) ==
                          limit_json(negotiated_),
                      "wire-v3 descriptor exceeds peer supported limits");
                auto expected = Descriptor::from_package(package_, negotiated_, session_, epoch_);
                check(bytes == expected.bytes(),
                      "wire-v3 expected package/session/metadata identity mismatch");
                bind_descriptor(std::move(expected));
                const auto ack = descriptor_->control("metadata_ack");
                send(Message::metadata_ack, ack, metadata_deadline_);
                startup_->metadata_ack(ack);
            }
            before(metadata_deadline_);
            startup_deadline_ =
                Clock::now() + std::chrono::milliseconds(negotiated_.startup_timeout_ms);
            stage_ = Stage::admitted;
        });
    }
    bool permission() {
        return guarded([&] { return startup_ ? startup_->gpu_permission() : false; });
    }
    void begin() {
        guarded([&] {
            check(authority_ && stage_ == Stage::admitted, "wire-v3 initialization out of order");
            before(startup_deadline_);
            startup_->begin_initialization();
            stage_ = Stage::initializing;
        });
    }
    void ready() {
        guarded([&] {
            if (authority_) {
                check(stage_ == Stage::initializing, "wire-v3 readiness out of order");
                before(startup_deadline_);
                startup_->initialization_complete();
                // Generated only after completion: a descriptor-only, pre-sent ACK cannot
                // authorize frames before the peer has observed this readiness event.
                const auto challenge = new_challenge();
                send(Message::execution_ready,
                     readiness(*descriptor_, "execution_ready", challenge), startup_deadline_);
                const auto ack = receive(Message::execution_ready_ack, startup_deadline_);
                check(ack == readiness(*descriptor_, "execution_ready_ack", challenge),
                      "wire-v3 execution-ready acknowledgement identity/challenge mismatch");
                startup_->execution_ready_ack(descriptor_->control("execution_ready_ack"));
            } else {
                check(stage_ == Stage::admitted, "wire-v3 readiness out of order");
                const auto ready = receive(Message::execution_ready, startup_deadline_);
                const auto control = small_json(ready);
                const auto challenge = control.at("readiness_challenge").get<std::string>();
                check(challenge_valid(challenge) &&
                          ready == readiness(*descriptor_, "execution_ready", challenge),
                      "wire-v3 readiness identity/challenge mismatch");
                startup_->begin_initialization();
                startup_->initialization_complete();
                send(Message::execution_ready_ack,
                     readiness(*descriptor_, "execution_ready_ack", challenge), startup_deadline_);
                startup_->execution_ready_ack(descriptor_->control("execution_ready_ack"));
            }
            before(startup_deadline_);
            stage_ = Stage::ready;
        });
    }
    void frame(std::string_view hash, std::string_view session, std::uint64_t epoch,
               std::uint64_t points, std::uint64_t bytes) {
        guarded([&] {
            check(stage_ == Stage::ready, "wire-v3 frame before execution-ready acknowledgement");
            startup_->accept_frame_identity(hash, session, epoch);
            check(points > 0 && points <= negotiated_.max_frame_points && bytes > 0 &&
                      bytes <= negotiated_.max_frame_bytes,
                  "wire-v3 frame exceeds negotiated limits");
        });
    }
    const Descriptor &descriptor() const {
        alive();
        check(descriptor_ != nullptr && stage_ != Stage::fresh, "wire-v3 descriptor not admitted");
        return *descriptor_;
    }
    Limits limits() const {
        descriptor();
        return negotiated_;
    }
    bool failed() const { return poisoned_; }
};
} // namespace detail
AuthorityTransport::AuthorityTransport(WireProtocol protocol, int socket,
                                       const CpuAdmittedPackage &p, Limits limits,
                                       std::string session, std::uint64_t epoch)
    : connection_(std::make_unique<detail::ControlConnection>(protocol, socket, p, limits,
                                                              std::move(session), epoch, true)) {}
AuthorityTransport::~AuthorityTransport() = default;
void AuthorityTransport::metadata_admission() { connection_->admission(); }
bool AuthorityTransport::initialization_permission() { return connection_->permission(); }
void AuthorityTransport::begin_initialization() { connection_->begin(); }
void AuthorityTransport::initialization_complete_and_wait_ready() { connection_->ready(); }
void AuthorityTransport::authorize_frame(std::string_view hash, std::string_view session,
                                         std::uint64_t epoch, std::uint64_t points,
                                         std::uint64_t bytes) {
    connection_->frame(hash, session, epoch, points, bytes);
}
const Descriptor &AuthorityTransport::descriptor() const { return connection_->descriptor(); }
Limits AuthorityTransport::negotiated_limits() const { return connection_->limits(); }
bool AuthorityTransport::failed() const { return connection_->failed(); }
PeerTransport::PeerTransport(WireProtocol protocol, int socket, const CpuAdmittedPackage &p,
                             Limits limits, std::string session, std::uint64_t epoch)
    : connection_(std::make_unique<detail::ControlConnection>(protocol, socket, p, limits,
                                                              std::move(session), epoch, false)) {}
PeerTransport::~PeerTransport() = default;
void PeerTransport::metadata_admission() { connection_->admission(); }
void PeerTransport::acknowledge_execution_ready() { connection_->ready(); }
void PeerTransport::authorize_frame(std::string_view hash, std::string_view session,
                                    std::uint64_t epoch, std::uint64_t points,
                                    std::uint64_t bytes) {
    connection_->frame(hash, session, epoch, points, bytes);
}
const Descriptor &PeerTransport::descriptor() const { return connection_->descriptor(); }
Limits PeerTransport::negotiated_limits() const { return connection_->limits(); }
bool PeerTransport::failed() const { return connection_->failed(); }
} // namespace synrex::generic
