// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Tuomas Pietilä
#include "synrex/backend_profile.hpp"
#include <algorithm>
#include <cmath>
#include <limits>

namespace synrex::generic {
namespace {
constexpr std::array<const char *, 7> phases{
    "cpu_geometry_seconds", "h2d_and_zero_seconds", "cuda_preparation_seconds",
    "tensorrt_seconds",     "cuda_scatter_seconds", "d2h_seconds",
    "window_seconds"};
constexpr std::array<const char *, 3> counters{"completed_inference_blocks",
                                               "freshly_classified_points", "occupied_voxels"};
std::uint64_t add(std::uint64_t a, std::uint64_t b) {
    require(b <= UINT64_MAX - a, "backend profile counter overflow");
    return a + b;
}
std::optional<double> duration(const Json &values, const char *name) {
    if (!values.contains(name))
        return {};
    const auto &value = values.at(name);
    require(value.is_number(), "backend profile seconds must be numeric");
    const auto result = value.get<double>();
    require(std::isfinite(result) && result >= 0,
            "backend profile seconds must be finite nonnegative");
    return result;
}
std::optional<std::uint64_t> count(const Json &values, const char *name) {
    if (!values.contains(name))
        return {};
    const auto &value = values.at(name);
    require(value.is_number_integer() &&
                (value.is_number_unsigned() || value.get<std::int64_t>() >= 0),
            "backend profile count must be nonnegative integer");
    return value.get<std::uint64_t>();
}
} // namespace
void ProfileSeconds::add(std::optional<double> value) {
    if (!value) {
        missing = generic::add(missing, 1);
        return;
    }
    require(std::isfinite(*value) && *value >= 0 && std::isfinite(sum + *value),
            "backend profile seconds sum overflow or invalid value");
    const auto next = generic::add(samples, 1);
    minimum = samples ? std::min(minimum, *value) : *value;
    maximum = samples ? std::max(maximum, *value) : *value;
    sum += *value;
    samples = next;
}
void ProfileSeconds::merge(const ProfileSeconds &other) {
    const auto next = generic::add(samples, other.samples);
    const auto next_missing = generic::add(missing, other.missing);
    require(std::isfinite(sum + other.sum),
            "backend profile seconds sum overflow or invalid value");
    if (other.samples) {
        minimum = samples ? std::min(minimum, other.minimum) : other.minimum;
        maximum = samples ? std::max(maximum, other.maximum) : other.maximum;
    }
    samples = next;
    missing = next_missing;
    sum += other.sum;
}
Json ProfileSeconds::receipt() const {
    return {{"samples", samples},
            {"missing_windows", missing},
            {"sum_seconds", samples ? Json(sum) : Json(nullptr)},
            {"minimum_seconds", samples ? Json(minimum) : Json(nullptr)},
            {"maximum_seconds", samples ? Json(maximum) : Json(nullptr)}};
}
void ProfileCounter::add(std::optional<std::uint64_t> value) {
    if (!value) {
        missing = generic::add(missing, 1);
        return;
    }
    const auto next = generic::add(samples, 1), total = generic::add(sum, *value);
    minimum = samples ? std::min(minimum, *value) : *value;
    maximum = samples ? std::max(maximum, *value) : *value;
    sum = total;
    samples = next;
}
void ProfileCounter::merge(const ProfileCounter &other) {
    const auto next = generic::add(samples, other.samples),
               next_missing = generic::add(missing, other.missing),
               total = generic::add(sum, other.sum);
    if (other.samples) {
        minimum = samples ? std::min(minimum, other.minimum) : other.minimum;
        maximum = samples ? std::max(maximum, other.maximum) : other.maximum;
    }
    samples = next;
    missing = next_missing;
    sum = total;
}
Json ProfileCounter::receipt() const {
    return {{"samples", samples},
            {"missing_windows", missing},
            {"sum", samples ? Json(sum) : Json(nullptr)},
            {"minimum", samples ? Json(minimum) : Json(nullptr)},
            {"maximum", samples ? Json(maximum) : Json(nullptr)}};
}
BackendProfile::BackendProfile(const Json &identity) {
    supported_ = identity.value("schema", "") == "synrex.persistent_session.v1" &&
                 identity.value("backend", "") == "CUDA/TensorRT" && !identity.contains("fixture");
    if (!supported_)
        return;
    const Contract contract(identity.at("contract"));
    const auto mode = identity.at("geometry_mode").get<std::string>();
    require(mode == "reference" || mode == "direct" || mode == "flat" || mode == "radix",
            "backend profile geometry mode");
    const auto manifest = identity.at("manifest_sha256").get<std::string>();
    require(
        manifest.size() == 64 &&
            std::all_of(manifest.begin(), manifest.end(),
                        [](char c) { return (c >= '0' && c <= '9') || (c >= 'a' && c <= 'f'); }),
        "backend profile manifest identity");
    max_blocks_ = contract.max_blocks;
    max_points_ = contract.max_points;
    max_memberships_ = contract.max_memberships;
    voxels_ = contract.voxels;
    workload_ = {
        {"manifest_sha256", manifest},
        {"geometry_mode", mode},
        {"effective_geometry",
         mode == "reference" || std::any_of(contract.overlap.begin(), contract.overlap.end(),
                                            [](double v) { return v != 0; })
             ? "reference"
             : mode},
        {"geometry", contract.spec.at("geometry")},
        {"feature_count", contract.features.size()},
        {"class_count", contract.las_classes.size()},
        {"voxels_per_block", voxels_},
        {"input_dtype", contract.input_dtype},
        {"output_dtype", contract.output_dtype}};
    require(workload_.dump().size() <= 2048, "backend profile workload metadata bound");
}
void BackendProfile::observe(const Json &timings, std::uint64_t points) {
    require(points > 0, "backend profile returned frame is empty");
    auto next = *this;
    next.windows_ = add(windows_, 1);
    next.points_ = add(points_, points);
    if (supported_) {
        require(points <= max_points_ && points <= max_memberships_ && timings.is_object(),
                "backend profile frame/timing bounds");
        std::array<std::optional<std::uint64_t>, 3> values;
        for (std::size_t i = 0; i < values.size(); ++i)
            values[i] = count(timings, counters[i]);
        require(!values[0] ||
                    (*values[0] > 0 && *values[0] <= max_blocks_ && *values[0] <= max_memberships_),
                "backend profile block count");
        require(!values[1] || *values[1] == points, "backend profile classified point mismatch");
        require(!values[2] || (*values[2] > 0 && *values[2] <= max_memberships_),
                "backend profile occupied voxel count");
        require(!values[0] || !values[2] || *values[2] >= *values[0],
                "backend profile completed block must be occupied");
        require(!values[0] || !values[2] ||
                    (*values[0] <= UINT64_MAX / points && *values[2] <= points * *values[0]),
                "backend profile occupied voxels exceed returned points across blocks");
        require(!values[0] || !values[2] || *values[2] <= *values[0] * voxels_,
                "backend profile occupied voxels exceed completed grids");
        for (std::size_t i = 0; i < phases.size(); ++i)
            next.phases_[i].add(duration(timings, phases[i]));
        for (std::size_t i = 0; i < counters.size(); ++i)
            next.counters_[i].add(values[i]);
    }
    *this = std::move(next);
}
void BackendProfile::merge(const BackendProfile &other) {
    require(supported_ == other.supported_ && workload_ == other.workload_,
            "backend profile workload mismatch");
    auto next = *this;
    next.windows_ = add(windows_, other.windows_);
    next.points_ = add(points_, other.points_);
    for (std::size_t i = 0; i < phases.size(); ++i)
        next.phases_[i].merge(other.phases_[i]);
    for (std::size_t i = 0; i < counters.size(); ++i)
        next.counters_[i].merge(other.counters_[i]);
    *this = std::move(next);
}
Json BackendProfile::receipt() const {
    bool missing{};
    Json phase_data = Json::object(), counter_data = Json::object();
    for (std::size_t i = 0; i < phases.size(); ++i) {
        phase_data[phases[i]] = supported_ ? phases_[i].receipt() : Json(nullptr);
        missing = missing || phases_[i].missing > 0;
    }
    for (std::size_t i = 0; i < counters.size(); ++i) {
        counter_data[counters[i]] = supported_ ? counters_[i].receipt() : Json(nullptr);
        missing = missing || counters_[i].missing > 0;
    }
    Json result{
        {"schema", "synrex.backend_profile.v1"},
        {"availability", !supported_ ? "UNSUPPORTED_BACKEND"
                         : !windows_ ? "UNAVAILABLE_NO_SAMPLES"
                         : missing   ? "PARTIAL_MISSING_FIELDS"
                                     : "BACKEND_REPORTED"},
        {"backend", supported_ ? Json("CUDA/TensorRT") : Json(nullptr)},
        {"workload", workload_},
        {"observed_returned_windows", windows_},
        {"observed_returned_frame_points", points_},
        {"phase_seconds", phase_data},
        {"counters", counter_data},
        {"aggregation",
         "one unweighted sample per successful Session::run return; counters sum per-window "
         "backend values; retained before output map/send, including failed trial prefixes"},
        {"timing_scopes",
         {{"cpu_geometry_seconds", "receiver steady clock around CPU geometry"},
          {"h2d_and_zero_seconds",
           "CUDA events: point upload plus per-block ids/voxels/groups upload and input zeroing"},
          {"cuda_preparation_seconds", "CUDA events: feature preparation"},
          {"tensorrt_seconds", "CUDA events: TensorRT enqueue execution"},
          {"cuda_scatter_seconds", "CUDA events: occupied logits to original-point labels"},
          {"d2h_seconds", "CUDA events: final labels/semantic IDs/error download"},
          {"window_seconds",
           "receiver steady clock: CPU geometry through synchronized reconstruction and backend "
           "result creation; excludes receiver coverage/map/send and startup"}}},
        {"scope", "backend reports need external frozen source/binary and GPU job evidence; CPU "
                  "fixtures unsupported; no physical capture, quality, capacity, optimization or "
                  "overhead verdict; phase scopes may overlap and are never summed into wall"},
        {"storage_bytes", sizeof(*this)},
        {"storage_scope", "inline C++ object only; excludes bounded workload JSON heap and "
                          "serialized receipts; process RSS is separate"},
        {"maximum_receipt_bytes", max_receipt_bytes}};
    require(result.dump().size() <= max_receipt_bytes, "backend profile serialized byte bound");
    return result;
}
} // namespace synrex::generic
