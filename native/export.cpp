// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Tuomas Pietilä
#include "synrex/export.hpp"
#include <fcntl.h>
#include <linux/fs.h>
#include <sys/syscall.h>
#include <unistd.h>
namespace synrex::streaming {
namespace {
void durable(const fs::path &path, bool directory = false) {
    const int fd = ::open(path.c_str(), O_RDONLY | O_CLOEXEC | (directory ? O_DIRECTORY : 0));
    require(fd >= 0, "export sync open");
    const auto result = ::fsync(fd);
    ::close(fd);
    require(result == 0, "export sync failure");
}
} // namespace
AsyncLazExport::AsyncLazExport(const fs::path &source, const fs::path &model,
                               const fs::path &taxonomy, const fs::path &destination)
    : source_(source), model_(model), taxonomy_(taxonomy), destination_(fs::absolute(destination)) {
    require(!fs::exists(destination_) && !fs::is_symlink(destination_) &&
                fs::is_directory(destination_.parent_path()),
            "export requires new bundle in existing parent");
    const Package package(model);
    original_ = decode(source, package.contract);
    require(!original_.empty() && original_.size() <= 1000000, "bounded export corpus");
    const auto classes = read_json(taxonomy).at("classes");
    require(classes.size() == package.contract.las_classes.size(), "export taxonomy size");
    for (std::size_t i = 0; i < classes.size(); ++i)
        require(classes[i].at("semantic_id") == i &&
                    classes[i].at("output_las_code") == package.contract.las_classes[i],
                "export taxonomy mapping");
    seen_.resize(original_.size());
    labels_.resize(original_.size());
    semantics_.resize(original_.size());
    frames_.resize(original_.size());
    timestamps_.resize(original_.size());
    metadata_ = {{"source_sha256", sha256(source_)},
                 {"model_manifest_sha256", sha256(package.root / "manifest.json")},
                 {"taxonomy_sha256", sha256(taxonomy_)},
                 {"las_classes", package.contract.las_classes}};
}
AsyncLazExport::~AsyncLazExport() {
    if (worker_.joinable())
        worker_.join();
}
void AsyncLazExport::observe(const Classification &c) {
    if (submitted_)
        return;
    validate_classification(c);
    for (std::size_t i = 0; i < c.frame.points.size(); ++i) {
        const auto &p = c.frame.points[i];
        require(p.id < original_.size() && p.xyz == original_[p.id].xyz &&
                    p.intensity == original_[p.id].intensity,
                "export requires exact original source record geometry/intensity; transformed "
                "replay cannot use preservation writer");
        if (seen_[p.id])
            continue;
        seen_[p.id] = 1;
        ++covered_;
        labels_[p.id] = c.las_codes[i];
        semantics_[p.id] = c.semantic_ids[i];
        frames_[p.id] = c.frame.header.frame_id;
        timestamps_[p.id] = p.timestamp_ns;
    }
    if (covered_ != original_.size())
        return;
    metadata_["coordinate_frame"] = c.frame.header.coordinate_frame;
    metadata_["crs"] = c.frame.header.crs;
    metadata_["sensor_id"] = c.frame.header.sensor_id;
    metadata_["stream_id"] = c.frame.header.stream_id;
    metadata_["taxonomy"] = c.frame.header.taxonomy;
    submitted_ = true;
    worker_ = std::thread([this] {
        try {
            publish();
        } catch (...) {
            error_ = std::current_exception();
        }
    });
}
void AsyncLazExport::publish() {
    const auto stage = destination_.parent_path() / ("." + destination_.filename().string() +
                                                     ".staging-" + std::to_string(::getpid()));
    require(fs::create_directory(stage), "export staging exists");
    try {
        require(sha256(source_) == metadata_["source_sha256"], "export source changed");
        const Package verified(model_);
        require(sha256(verified.root / "manifest.json") == metadata_["model_manifest_sha256"],
                "export model changed");
        write_new(stage / "labels.u8",
                  std::string(reinterpret_cast<const char *>(labels_.data()), labels_.size()));
        const auto changed = apply_labels(source_, stage / "labels.u8", stage / "classified.laz");
        Json sidecar{
            {"schema", "synrex.point_semantics.v1"},
            {"source_sha256", metadata_["source_sha256"]},
            {"model_manifest_sha256", metadata_["model_manifest_sha256"]},
            {"point_count", original_.size()},
            {"index_order", "original source point record order"},
            {"semantic_id_definition",
             "zero-based model channel; first fresh observation per original source record"},
            {"semantic_ids", semantics_},
            {"las_classes", metadata_["las_classes"]},
            {"mapped_labels_sha256", sha256(stage / "labels.u8")}};
        write_new(stage / "semantics.json", sidecar.dump() + "\n");
        auto provenance = metadata_;
        provenance["schema"] = "synrex.streaming_export_provenance.v1";
        provenance["source_frame_ids"] = frames_;
        provenance["measurement_unix_ns"] = timestamps_;
        provenance["source_index_order"] = "original source record";
        provenance["classification_policy"] =
            "first fresh observation; predictions retain original window contexts; no map reuse";
        write_new(stage / "provenance.json", provenance.dump() + "\n");
        require(sha256(source_) == metadata_["source_sha256"],
                "export source changed during encoding");
        receipt_ = {{"schema", "synrex.streaming_export_bundle.v1"},
                    {"status", "PASS"},
                    {"points", original_.size()},
                    {"coverage", 1.0},
                    {"changed_las_classifications", changed},
                    {"bindings", metadata_},
                    {"files", Json::object()},
                    {"copc", "NOT_RUN: optional verified offline QGIS export"}};
        for (const auto &name :
             {"labels.u8", "classified.laz", "semantics.json", "provenance.json"})
            receipt_["files"][name] = sha256(stage / name);
        write_new(stage / "bundle.json", receipt_.dump(2) + "\n");
        for (const auto &entry : fs::directory_iterator(stage))
            durable(entry.path());
        durable(stage, true);
        require(::syscall(SYS_renameat2, AT_FDCWD, stage.c_str(), AT_FDCWD, destination_.c_str(),
                          RENAME_NOREPLACE) == 0,
                "atomic export publication failed/refusing replacement");
        durable(destination_.parent_path(), true);
    } catch (...) {
        std::error_code ignored;
        fs::remove_all(stage, ignored);
        throw;
    }
}
Json AsyncLazExport::finish() {
    if (worker_.joinable())
        worker_.join();
    if (error_)
        std::rethrow_exception(error_);
    require(submitted_ && covered_ == original_.size(),
            "partial source coverage: no export bundle published");
    return receipt_;
}
} // namespace synrex::streaming
