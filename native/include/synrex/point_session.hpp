// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Tuomas Pietilä
#pragma once
#include "runtime.hpp"
#include <functional>
#include <map>
#include <memory>

namespace synrex::point {
using Members = std::map<std::string, std::string>;
std::string digest_bytes(const std::string &);
// Returned only by a trusted application validator. Public core validates the
// envelope; model-specific schema, profiles, qualification and target remain the
// application's responsibility. No package-authored validator is executed.
struct PointContract {
    std::string adapter_id, model_id, layout_id, execution_mode;
    std::string environment_sha256;
    std::vector<std::uint8_t> las_projection;
    std::uint64_t max_source_rows{}, max_representatives{}, max_input_bytes{}, max_output_bytes{};
    bool capability_qualified{};
    std::string selected_capability, policy_sha256;
};
class PointArtifact final {
    Json manifest_;
    std::string manifest_bytes_, manifest_sha256_, contract_sha256_, worker_sha256_;
    Members members_;
    PointContract contract_;

  public:
    using Validator = std::function<PointContract(const Json &, const Members &)>;
    // All arguments are copied and captured. expected_manifest_sha256 comes from
    // the programme's source lock, never from an untrusted package or wire peer.
    PointArtifact(std::string manifest_bytes, const std::string &expected_manifest_sha256,
                  Members members, const Validator &);
    PointArtifact(const PointArtifact &) = delete;
    PointArtifact &operator=(const PointArtifact &) = delete;
    PointArtifact(PointArtifact &&) = delete;
    PointArtifact &operator=(PointArtifact &&) = delete;
    const PointContract &contract() const { return contract_; }
    const Json &manifest() const { return manifest_; }
    const std::string &manifest_bytes() const { return manifest_bytes_; }
    const std::string &manifest_sha256() const { return manifest_sha256_; }
    const std::string &contract_sha256() const { return contract_sha256_; }
    const std::string &worker_sha256() const { return worker_sha256_; }
    const Members &members() const { return members_; }
};
struct SourceIdentity {
    std::string sha256, header_sha256;
    std::uint64_t rows{};
};
struct PreparedBatch {
    SourceIdentity source;
    std::uint64_t cutoff_ns{};
    std::string logical_batch_sha256;
    std::vector<std::uint64_t> current_rows, representative_source_rows;
    std::vector<std::uint32_t> inverse;
    std::vector<float> coord, feat;
    std::vector<std::int32_t> grid_coord;
};
struct NativeAuthority {
    std::uint64_t epoch{}, request_id{};
    std::string sha256, input_payload_sha256;
    Json record;
};
struct SemanticResult {
    std::uint64_t epoch{}, request_id{};
    std::string authority_sha256;
    std::vector<std::uint8_t> semantic_ids, validity;
};
struct Classification {
    SourceIdentity source;
    std::vector<std::uint64_t> rows;
    std::vector<std::uint8_t> semantic_ids, validity, las_codes;
    std::uint64_t epoch{}, request_id{};
    std::string logical_batch_sha256, authority_sha256;
};
// A separate type from dense InferenceSession: no fabricated dense Contract or
// ONNX/TensorRT member. Implementations own bounded process I/O/cancellation,
// identity verification, actual initialization, deadlines and group retirement.
class PointInferenceSession {
  public:
    virtual ~PointInferenceSession() = default;
    virtual Json initialize(const PointArtifact &) = 0;
    virtual void activate() = 0;
    virtual void open_source(const SourceIdentity &, std::uint64_t epoch) = 0;
    virtual SemanticResult run(const PreparedBatch &, const NativeAuthority &) = 0;
    virtual void complete(const Json &) = 0;
};
enum class Peer { input, output };
enum class State {
    metadata_pending,
    initializing,
    ready_pending,
    active,
    complete_pending,
    complete,
    failed
};
struct SessionOptions {
    // Explicitly recorded; neither mode constitutes qualified production use.
    bool synthetic_fixture{};
    bool qualification_only{};
    std::uint64_t max_sources{64}, max_total_rows{1000000};
    // Exact native-owned roster, captured in the descriptor before metadata ACKs.
    std::vector<SourceIdentity> sources;
};
class PointSession final {
    struct Impl;
    std::unique_ptr<Impl> impl_;

  public:
    using Factory = std::function<std::unique_ptr<PointInferenceSession>()>;
    using OutputCommitter =
        std::function<std::string(const std::map<std::string, std::vector<std::uint8_t>> &)>;
    PointSession(std::shared_ptr<const PointArtifact>, std::string session_id, Factory,
                 SessionOptions = {});
    ~PointSession();
    PointSession(const PointSession &) = delete;
    PointSession &operator=(const PointSession &) = delete;
    State state() const;
    const std::string &descriptor_sha256() const;
    Json descriptor() const;
    void metadata_ack(Peer, const std::string &descriptor_sha256);
    // May create/initialize the backend only after two distinct metadata ACKs.
    Json initialize();
    void ready_ack(Peer, const std::string &initialization_sha256);
    void open_source(const SourceIdentity &);
    // Copies preparation/maps before crossing the worker boundary. Worker result
    // cannot change source, current rows, inverse, epoch or request identity.
    Classification run(const PreparedBatch &);
    // Completer is native-owned durable output authority, never a wire callback.
    // Both completion ACKs are still required before successful terminal state.
    Json complete(const OutputCommitter &);
    void completion_ack(Peer, const std::string &completion_sha256);
    void fail();
};
} // namespace synrex::point
