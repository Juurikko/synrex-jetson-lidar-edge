// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Tuomas Pietilä
#include "synrex/generic_output.hpp"
#include <limits>

namespace synrex::generic {
namespace {
void before(Startup::Time deadline) {
    require(Startup::Clock::now() < deadline, "generic output absolute deadline expired");
}
void countable(std::uint64_t classifications, std::uint64_t points, std::size_t added) {
    require(classifications < std::numeric_limits<std::uint64_t>::max() &&
                added <= std::numeric_limits<std::uint64_t>::max() - points,
            "generic output counter overflow");
}
} // namespace
OutputSender::OutputSender(AuthorityTransport &transport, double resolution, std::size_t capacity)
    : transport_(transport) {
    try {
        map_ = std::make_unique<SemanticMap>(transport_.descriptor(), resolution, capacity);
    } catch (...) {
        abort();
        throw;
    }
}
OutputSender::~OutputSender() {
    if (!finished_)
        abort();
}
void OutputSender::alive() const {
    require(!failed_, "generic output sender permanently failed");
    require(!finished_, "generic output sender already finished");
}
void OutputSender::abort() noexcept {
    failed_ = true;
    transport_.abort_output();
}
void OutputSender::send(const Classification &result, Startup::Time deadline) {
    try {
        alive();
        before(deadline);
        countable(classifications_, points_, result.frame.points.size());
        map_->update(result);
        before(deadline);
        transport_.send_classification(result, deadline);
        before(deadline);
        ++classifications_;
        points_ += result.frame.points.size();
        last_frame_end_ = result.frame.frame_end_ns;
    } catch (...) {
        abort();
        throw;
    }
}
Json OutputSender::finish(Startup::Time deadline) {
    try {
        alive();
        before(deadline);
        require(classifications_ > 0 && points_ > 0,
                "generic output requires nonempty classifications");
        const auto bytes = map_->snapshot_bytes(last_frame_end_);
        before(deadline);
        auto receipt = transport_.finish_output(bytes, classifications_, points_, deadline);
        before(deadline);
        finished_ = true;
        return receipt;
    } catch (...) {
        abort();
        throw;
    }
}
OutputReceiver::OutputReceiver(PeerTransport &transport, double resolution, std::size_t capacity)
    : transport_(transport) {
    try {
        map_ = std::make_unique<SemanticMap>(transport_.descriptor(), resolution, capacity);
    } catch (...) {
        abort();
        throw;
    }
}
OutputReceiver::~OutputReceiver() {
    if (!finished_)
        abort();
}
void OutputReceiver::alive() const {
    require(!failed_, "generic output receiver permanently failed");
    require(!finished_, "generic output receiver already finished");
}
void OutputReceiver::abort() noexcept {
    failed_ = true;
    transport_.abort_output();
}
Classification OutputReceiver::receive(Startup::Time deadline) {
    try {
        alive();
        before(deadline);
        auto result = transport_.receive_classification(deadline);
        countable(classifications_, points_, result.frame.points.size());
        map_->update(result);
        before(deadline);
        ++classifications_;
        points_ += result.frame.points.size();
        last_frame_end_ = result.frame.frame_end_ns;
        return result;
    } catch (...) {
        abort();
        throw;
    }
}
Json OutputReceiver::finish(Startup::Time deadline) {
    try {
        alive();
        before(deadline);
        require(classifications_ > 0 && points_ > 0,
                "generic output requires nonempty classifications");
        const auto bytes = map_->snapshot_bytes(last_frame_end_);
        before(deadline);
        auto receipt = transport_.finish_output(bytes, classifications_, points_, deadline);
        before(deadline);
        finished_ = true;
        return receipt;
    } catch (...) {
        abort();
        throw;
    }
}
} // namespace synrex::generic
