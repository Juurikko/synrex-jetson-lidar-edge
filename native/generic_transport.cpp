// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Tuomas Pietilä
#include "synrex/generic_transport.hpp"
#include "synrex/generic_map.hpp"
#include <algorithm>
#include <array>
#include <cerrno>
#include <climits>
#include <openssl/evp.h>
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
    execution_ready_ack = 5,
    frame = 16,
    classification = 17,
    snapshot = 32,
    output_complete = 33,
    output_complete_ack = 34
};
void check(bool ok, const char *message) {
    if (!ok)
        throw std::runtime_error(message);
}
using DigestContext = std::unique_ptr<EVP_MD_CTX, decltype(&EVP_MD_CTX_free)>;
DigestContext new_digest() {
    DigestContext context(EVP_MD_CTX_new(), EVP_MD_CTX_free);
    check(context && EVP_DigestInit_ex(context.get(), EVP_sha256(), nullptr) == 1,
          "wire-v3 SHA256 initialization failed");
    return context;
}
std::string digest_value(const DigestContext &context) {
    auto copy = new_digest();
    check(EVP_MD_CTX_copy_ex(copy.get(), context.get()) == 1, "wire-v3 SHA256 copy failed");
    std::array<unsigned char, 32> hash{};
    unsigned size{};
    check(EVP_DigestFinal_ex(copy.get(), hash.data(), &size) == 1 && size == hash.size(),
          "wire-v3 SHA256 final failed");
    const char hex[] = "0123456789abcdef";
    std::string result;
    for (const auto value : hash) {
        result += hex[value >> 4];
        result += hex[value & 15];
    }
    return result;
}
std::string digest_bytes(std::string_view bytes) {
    auto context = new_digest();
    check(EVP_DigestUpdate(context.get(), bytes.data(), bytes.size()) == 1,
          "wire-v3 SHA256 update failed");
    return digest_value(context);
}
// Exact complete classification envelopes in wire order; no decoded JSON or
// host-local clock is substituted. Envelope lengths make the concatenation unique.
struct Transcript {
    DigestContext context = new_digest();
    std::uint64_t frames{}, points{};
    void add(std::string_view bytes, std::size_t count) {
        check(frames < UINT64_MAX && count <= UINT64_MAX - points,
              "wire-v3 output accounting overflow");
        check(EVP_DigestUpdate(context.get(), bytes.data(), bytes.size()) == 1,
              "wire-v3 classification transcript failed");
        ++frames;
        points += count;
    }
};
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
    check(bytes.find('\0') == std::string::npos, "wire-v3 control contains a NUL byte");
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
    if (message == Message::snapshot)
        return SemanticMap::max_snapshot_bytes - 12; // Whole envelope <=16MiB.
    if (message == Message::output_complete || message == Message::output_complete_ack)
        return 2048;
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
    const CpuAdmittedPackage package_;
    Limits supported_, negotiated_{};
    std::string session_;
    std::uint64_t epoch_;
    bool authority_, poisoned_{};
    enum class Stage { fresh, admitted, initializing, ready, terminal_pending, complete };
    Stage stage_{Stage::fresh};
    Transcript sent_, received_;
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
    void transfer(char *data, std::size_t size, bool sending, Time deadline,
                  bool packet_start = false) const {
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
            if (!sending && result == 0)
                throw std::runtime_error(packet_start && offset == 0
                                             ? "wire-v3 EOF before envelope; no terminal success"
                                             : "wire-v3 truncated envelope");
            check(result > 0,
                  sending ? "wire-v3 socket write failed" : "wire-v3 socket read failed");
            offset += static_cast<std::size_t>(result);
        }
        before(deadline);
    }
    void end_writes(Time deadline) const {
        before(deadline);
        check(::shutdown(socket_, SHUT_WR) == 0, "wire-v3 terminal half-close failed");
        before(deadline);
    }
    void require_eof(Time deadline) const {
        for (;;) {
            before(deadline);
            char extra{};
            const auto result = ::recv(socket_, &extra, 1, MSG_DONTWAIT);
            if (result < 0 && errno == EINTR)
                continue;
            if (result < 0 && (errno == EAGAIN || errno == EWOULDBLOCK)) {
                wait(POLLIN, deadline);
                continue;
            }
            check(result >= 0, "wire-v3 terminal EOF read failed");
            check(result == 0, "wire-v3 trailing bytes after terminal record");
            before(deadline);
            return;
        }
    }
    Json completion_record(const std::string &kind, const Json &snapshot,
                           const std::string &snapshot_bytes, const Transcript &transcript,
                           const std::string &challenge) const {
        return {{"schema", "synrex.output.terminal.v3"},
                {"kind", kind},
                {"descriptor_sha256", descriptor_->sha256()},
                {"session_id", session_},
                {"map_epoch", epoch_},
                {"taxonomy_sha256", descriptor_->taxonomy().sha256()},
                {"classification_frames", transcript.frames},
                {"classified_points", transcript.points},
                {"classification_sha256", digest_value(transcript.context)},
                {"snapshot_sha256", digest_bytes(snapshot_bytes)},
                {"snapshot_bytes", snapshot_bytes.size()},
                {"snapshot_cells", snapshot.at("cells").size()},
                {"as_of_ns", snapshot.at("as_of_ns")},
                {"completion_challenge", challenge}};
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
    std::uint32_t payload_size(const std::array<char, 12> &header, Message expected) const {
        check(header[0] == 'S' && header[1] == 'R' && header[2] == 'X' && header[3] == '3' &&
                  header[4] == 3 && header[5] == static_cast<char>(expected) && header[6] == 0 &&
                  header[7] == 0,
              "wire-v3 unsupported/malformed/out-of-order envelope");
        std::uint32_t size{};
        for (unsigned i = 8; i < 12; ++i)
            size = (size << 8) | static_cast<unsigned char>(header[i]);
        return size;
    }
    std::string receive(Message expected, Time deadline) const {
        std::array<char, 12> header{};
        transfer(header.data(), header.size(), false, deadline, true);
        const auto size = payload_size(header, expected);
        // Header and expected-kind bounds are validated BEFORE payload allocation.
        check(size > 0 && size <= bound(expected), "wire-v3 incoming payload bound");
        std::string bytes(size, '\0');
        transfer(bytes.data(), bytes.size(), false, deadline);
        return bytes;
    }
    void ready_for_data(Time deadline) const {
        before(deadline);
        check(stage_ == Stage::ready, "wire-v3 frame before execution-ready acknowledgement");
    }
    void authorize_data(const Frame &value, std::uint64_t bytes) {
        frame(value.identity.descriptor_sha256, value.identity.session_id, value.identity.map_epoch,
              value.points.size(), bytes);
    }
    std::string receive_data(Message expected, Time deadline) const {
        ready_for_data(deadline);
        std::array<char, 12> header{};
        transfer(header.data(), header.size(), false, deadline, true);
        const auto size = payload_size(header, expected);
        const auto whole_size = static_cast<std::uint64_t>(header.size()) + size;
        check(size > 0 && whole_size <= negotiated_.max_frame_bytes,
              "wire-v3 incoming whole frame byte bound");
        // No peer-sized allocation occurs until stage, fixed header, expected kind
        // and the whole negotiated envelope budget have all been checked.
        before(deadline);
        std::string bytes(header.data(), header.size());
        bytes.resize(static_cast<std::size_t>(whole_size));
        transfer(bytes.data() + header.size(), size, false, deadline);
        return bytes;
    }
    template <class F> decltype(auto) guarded(F operation) {
        try {
            alive();
            return operation();
        } catch (const std::exception &error) {
            poisoned_ = true;
            try {
                if (startup_)
                    startup_->fail(error.what());
            } catch (...) {
            }
            ::shutdown(socket_, SHUT_RDWR);
            throw;
        } catch (...) {
            abort();
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
        : socket_(socket), package_(package.package_), supported_(supported),
          session_(std::move(session)), epoch_(epoch), authority_(authority),
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
    std::shared_ptr<const EngineArtifact> begin_session() {
        return guarded([&] {
            check(authority_ && stage_ == Stage::admitted,
                  "wire-v3 session initialization out of order");
            before(startup_deadline_);
            const auto retained = package_.package_;
            (void)retained->engine_bytes();
            // The dense persistent backend's supported capacity is smaller than
            // the generic metadata envelope. Reject before any device construction.
            check(negotiated_.max_frame_points <= 1000000 &&
                      negotiated_.max_frame_points <= retained->contract().max_points,
                  "unsafe persistent window point budget");
            begin();
            return retained;
        });
    }
    void abort() noexcept {
        poisoned_ = true;
        try {
            if (startup_)
                startup_->fail("generic session failed");
        } catch (...) {
        }
        ::shutdown(socket_, SHUT_RDWR);
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
    void send_frame(const Frame &value, Time deadline) {
        guarded([&] {
            ready_for_data(deadline);
            // Authorize readiness, identity and count before any codec allocation.
            authorize_data(value, 1);
            auto bytes = encode_frame(*descriptor_, value);
            authorize_data(value, bytes.size());
            before(deadline);
            transfer(bytes.data(), bytes.size(), true, deadline);
        });
    }
    Frame receive_frame(Time deadline) {
        return guarded([&] {
            const auto bytes = receive_data(Message::frame, deadline);
            auto value = decode_frame(*descriptor_, bytes);
            authorize_data(value, bytes.size());
            before(deadline);
            return value;
        });
    }
    void send_classification(const Classification &value, Time deadline) {
        guarded([&] {
            ready_for_data(deadline);
            authorize_data(value.frame, 1);
            auto bytes = encode_classification(*descriptor_, value);
            authorize_data(value.frame, bytes.size());
            before(deadline);
            transfer(bytes.data(), bytes.size(), true, deadline);
            sent_.add(bytes, value.frame.points.size());
            before(deadline);
        });
    }
    Classification receive_classification(Time deadline) {
        return guarded([&] {
            const auto bytes = receive_data(Message::classification, deadline);
            auto value = decode_classification(*descriptor_, bytes);
            authorize_data(value.frame, bytes.size());
            received_.add(bytes, value.frame.points.size());
            before(deadline);
            return value;
        });
    }
    Json finish_output(const std::string &expected_snapshot_bytes, std::uint64_t frames,
                       std::uint64_t points, Time deadline) {
        return guarded([&] {
            ready_for_data(deadline);
            auto &transcript = authority_ ? sent_ : received_;
            check(transcript.frames > 0 && transcript.points > 0,
                  "wire-v3 empty output cannot qualify completion");
            check(transcript.frames == frames && transcript.points == points,
                  "wire-v3 output consumer omitted classification records");
            // Only the output consumer wrappers can call this operation. Their
            // snapshot is generated from every successfully sent/received result.
            const auto snapshot = admit_snapshot(*descriptor_, expected_snapshot_bytes);
            check(expected_snapshot_bytes.size() <= bound(Message::snapshot) &&
                      snapshot.dump() == expected_snapshot_bytes && !snapshot.at("cells").empty(),
                  "wire-v3 terminal snapshot encoding/cell bound");
            before(deadline);
            stage_ = Stage::terminal_pending;
            Json terminal;
            if (authority_) {
                send(Message::snapshot, expected_snapshot_bytes, deadline);
                // Fresh only after the final snapshot write. A pre-sent ACK or
                // a receipt from a prior session cannot confer delivery success.
                const auto challenge = new_challenge();
                terminal = completion_record("output_complete", snapshot, expected_snapshot_bytes,
                                             transcript, challenge);
                send(Message::output_complete, terminal.dump(), deadline);
                end_writes(deadline);
                const auto acknowledgement = receive(Message::output_complete_ack, deadline);
                const auto expected =
                    completion_record("output_complete_ack", snapshot, expected_snapshot_bytes,
                                      transcript, challenge);
                check(
                    acknowledgement == expected.dump(),
                    "wire-v3 completion acknowledgement identity/count/digest/challenge mismatch");
                require_eof(deadline);
            } else {
                const auto received = receive(Message::snapshot, deadline);
                (void)admit_snapshot(*descriptor_, received);
                check(received == expected_snapshot_bytes,
                      "wire-v3 snapshot differs from independently reconstructed output map");
                before(deadline);
                const auto bytes = receive(Message::output_complete, deadline);
                terminal = small_json(bytes);
                const auto challenge = terminal.at("completion_challenge").get<std::string>();
                check(challenge_valid(challenge) &&
                          bytes == completion_record("output_complete", snapshot,
                                                     expected_snapshot_bytes, transcript, challenge)
                                       .dump(),
                      "wire-v3 terminal identity/count/digest/challenge mismatch");
                // A valid record followed by extra data is not a complete stream.
                // EOF fences the write direction; EOF by itself is never success.
                require_eof(deadline);
                send(Message::output_complete_ack,
                     completion_record("output_complete_ack", snapshot, expected_snapshot_bytes,
                                       transcript, challenge)
                         .dump(),
                     deadline);
                end_writes(deadline);
            }
            before(deadline);
            stage_ = Stage::complete;
            terminal.erase("kind");
            terminal["schema"] = "synrex.output.delivery.v1";
            terminal["status"] = authority_ ? "PEER_VALIDATED" : "VALIDATED_ACK_SENT";
            terminal["time_boundary"] =
                "classified_output_and_terminal_map; no_inference_quality_claim";
            return terminal;
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
    : connection_(std::make_shared<detail::ControlConnection>(protocol, socket, p, limits,
                                                              std::move(session), epoch, true)) {}
AuthorityTransport::~AuthorityTransport() = default;
Json AuthorityTransport::finish_output(const std::string &bytes, std::uint64_t frames,
                                       std::uint64_t points, Startup::Time deadline) {
    return connection_->finish_output(bytes, frames, points, deadline);
}
void AuthorityTransport::abort_output() noexcept { connection_->abort(); }
void AuthorityTransport::metadata_admission() { connection_->admission(); }
bool AuthorityTransport::initialization_permission() { return connection_->permission(); }
std::shared_ptr<const EngineArtifact> AuthorityTransport::begin_session() {
    return connection_->begin_session();
}
void AuthorityTransport::complete_session() { connection_->ready(); }
void AuthorityTransport::abort_session() noexcept { connection_->abort(); }
std::function<void(std::uint64_t, std::uint64_t)> AuthorityTransport::frame_authorizer() const {
    const auto &descriptor = connection_->descriptor();
    return [connection = connection_, hash = descriptor.sha256(),
            session = descriptor.spec().at("session_id").get<std::string>(),
            epoch = descriptor.spec().at("map_epoch").get<std::uint64_t>()](std::uint64_t points,
                                                                            std::uint64_t bytes) {
        connection->frame(hash, session, epoch, points, bytes);
    };
}
std::function<void()> AuthorityTransport::session_aborter() const {
    return [connection = connection_] { connection->abort(); };
}
void AuthorityTransport::authorize_frame(std::string_view hash, std::string_view session,
                                         std::uint64_t epoch, std::uint64_t points,
                                         std::uint64_t bytes) {
    connection_->frame(hash, session, epoch, points, bytes);
}
void AuthorityTransport::send_frame(const Frame &value, Startup::Time deadline) {
    connection_->send_frame(value, deadline);
}
Frame AuthorityTransport::receive_frame(Startup::Time deadline) {
    return connection_->receive_frame(deadline);
}
void AuthorityTransport::send_classification(const Classification &value, Startup::Time deadline) {
    connection_->send_classification(value, deadline);
}
Classification AuthorityTransport::receive_classification(Startup::Time deadline) {
    return connection_->receive_classification(deadline);
}
const Descriptor &AuthorityTransport::descriptor() const { return connection_->descriptor(); }
Limits AuthorityTransport::negotiated_limits() const { return connection_->limits(); }
bool AuthorityTransport::failed() const { return connection_->failed(); }
PeerTransport::PeerTransport(WireProtocol protocol, int socket, const CpuAdmittedPackage &p,
                             Limits limits, std::string session, std::uint64_t epoch)
    : connection_(std::make_unique<detail::ControlConnection>(protocol, socket, p, limits,
                                                              std::move(session), epoch, false)) {}
PeerTransport::~PeerTransport() = default;
Json PeerTransport::finish_output(const std::string &bytes, std::uint64_t frames,
                                  std::uint64_t points, Startup::Time deadline) {
    return connection_->finish_output(bytes, frames, points, deadline);
}
void PeerTransport::abort_output() noexcept { connection_->abort(); }
void PeerTransport::metadata_admission() { connection_->admission(); }
void PeerTransport::acknowledge_execution_ready() { connection_->ready(); }
void PeerTransport::authorize_frame(std::string_view hash, std::string_view session,
                                    std::uint64_t epoch, std::uint64_t points,
                                    std::uint64_t bytes) {
    connection_->frame(hash, session, epoch, points, bytes);
}
void PeerTransport::send_frame(const Frame &value, Startup::Time deadline) {
    connection_->send_frame(value, deadline);
}
Frame PeerTransport::receive_frame(Startup::Time deadline) {
    return connection_->receive_frame(deadline);
}
void PeerTransport::send_classification(const Classification &value, Startup::Time deadline) {
    connection_->send_classification(value, deadline);
}
Classification PeerTransport::receive_classification(Startup::Time deadline) {
    return connection_->receive_classification(deadline);
}
const Descriptor &PeerTransport::descriptor() const { return connection_->descriptor(); }
Limits PeerTransport::negotiated_limits() const { return connection_->limits(); }
bool PeerTransport::failed() const { return connection_->failed(); }
} // namespace synrex::generic
