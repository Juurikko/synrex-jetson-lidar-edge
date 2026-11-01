// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Tuomas Pietilä
#pragma once
#include "synrex/generic_transport.hpp"
#include "synrex/point_session.hpp"
#include <variant>

namespace synrex::common {
enum class Backend { dense, point };
// Closed caller seam. Dense geometry, readiness and resource ownership remain in
// the existing transport-created generic::Session. Point factories are private
// application hooks and are invoked only by PointSession's dual-admission gate.
class Session final {
    std::variant<std::unique_ptr<generic::Session>, std::unique_ptr<point::PointSession>> session_;

  public:
    using Input = std::variant<generic::Frame, point::PreparedBatch>;
    using Output = std::variant<generic::Classification, point::Classification>;
    explicit Session(std::unique_ptr<generic::Session> dense) : session_(std::move(dense)) {
        require(bool(std::get<0>(session_)), "common null dense session");
    }
    explicit Session(std::unique_ptr<point::PointSession> point) : session_(std::move(point)) {
        require(bool(std::get<1>(session_)), "common null point session");
    }
    static Session from_dense(generic::AuthorityTransport &input,
                              generic::AuthorityTransport &output,
                              const synrex::SessionOptions &options = {},
                              const generic::DiagnosticTraceOptions &trace = {}) {
        return Session(input.create_session_with_output(output, options, trace));
    }
    static Session from_point(std::shared_ptr<const point::PointArtifact> artifact,
                              std::string session_id, point::PointSession::Factory factory,
                              point::SessionOptions options = {}) {
        return Session(std::make_unique<point::PointSession>(
            std::move(artifact), std::move(session_id), std::move(factory), std::move(options)));
    }
    Session(const Session &) = delete;
    Session &operator=(const Session &) = delete;
    Backend backend() const { return session_.index() == 0 ? Backend::dense : Backend::point; }
    Output run(Input input) {
        require(input.index() == session_.index(), "common backend/input mismatch");
        if (backend() == Backend::dense)
            return std::get<0>(session_)->run(std::move(std::get<0>(input)));
        return std::get<1>(session_)->run(std::get<1>(input));
    }
    point::PointSession &point_authority() {
        require(backend() == Backend::point, "common point authority requested for dense backend");
        return *std::get<1>(session_);
    }
    const generic::Session &dense_session() const {
        require(backend() == Backend::dense, "common dense session requested for point backend");
        return *std::get<0>(session_);
    }
};
} // namespace synrex::common
