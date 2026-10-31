// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Tuomas Pietilä
#include "synrex/replay_reader.hpp"
#include <algorithm>
#include <chrono>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <openssl/evp.h>
#include <sstream>

namespace {
using namespace synrex;
class Digest {
    std::unique_ptr<EVP_MD_CTX, decltype(&EVP_MD_CTX_free)> context_{EVP_MD_CTX_new(),
                                                                     EVP_MD_CTX_free};

  public:
    Digest() {
        require(bool(context_) && EVP_DigestInit_ex(context_.get(), EVP_sha256(), nullptr) == 1,
                "replay digest initialization failed");
    }
    void integer(std::uint64_t value, unsigned width) {
        std::array<unsigned char, 8> bytes{};
        for (unsigned i = 0; i < width; ++i)
            bytes[i] = (value >> (8 * i)) & 255;
        require(EVP_DigestUpdate(context_.get(), bytes.data(), width) == 1,
                "replay digest update failed");
    }
    void number(double value) {
        std::uint64_t bits{};
        std::memcpy(&bits, &value, sizeof(bits));
        integer(bits, 8);
    }
    std::string finish() {
        std::array<unsigned char, 32> bytes{};
        unsigned count{};
        require(EVP_DigestFinal_ex(context_.get(), bytes.data(), &count) == 1 && count == 32,
                "replay digest finalization failed");
        std::ostringstream out;
        for (auto byte : bytes)
            out << std::hex << std::setfill('0') << std::setw(2) << unsigned(byte);
        return out.str();
    }
};
using Clock = std::chrono::steady_clock;
double elapsed(Clock::time_point start) {
    return std::chrono::duration<double>(Clock::now() - start).count();
}
} // namespace
int main(int argc, char **argv) {
    using namespace synrex;
    Json receipt{{"schema", "synrex.replay_inspection.v1"},
                 {"status", "FAIL"},
                 {"execution", "CPU_SOURCE_DECODE_ONLY"},
                 {"gpu_seconds", 0},
                 {"classification", "NOT_RUN"},
                 {"multi_file_preservation", "NOT_SUPPORTED"},
                 {"physical_capture_age_seconds", nullptr},
                 {"offered_rate_pps", nullptr},
                 {"sources", Json::array()}};
    fs::path output;
    const auto start = Clock::now();
    std::uint64_t total{}, decoded{}, raw_decoded{}, verified{}, windows{}, high_water{};
    try {
        require(
            argc == 8 && std::string(argv[1]) == "inspect" &&
                std::string(argv[2]) == "--manifest" && std::string(argv[4]) == "--window-points" &&
                std::string(argv[6]) == "--receipt",
            "usage: synrex-replay inspect --manifest FILE --window-points N --receipt NEW_FILE");
        output = argv[7];
        require(!output.empty() && !fs::exists(output) && !fs::is_symlink(output) &&
                    fs::is_directory(fs::absolute(output).parent_path()),
                "replay receipt must be new in existing parent");
        const std::string window_text = argv[5];
        require(!window_text.empty() && window_text.size() <= 5 &&
                    std::all_of(window_text.begin(), window_text.end(),
                                [](char c) { return c >= '0' && c <= '9'; }),
                "replay window must be unsigned decimal");
        const auto window_points = std::stoull(window_text);
        require(window_points > 0 && window_points <= 65536, "replay window outside 1..65536");
        const replay::Catalog catalog(argv[3]);
        receipt["manifest_sha256"] = catalog.manifest_sha256();
        receipt["declared_records"] = catalog.total_records();
        receipt["admission_seconds"] = elapsed(start);
        receipt["boundary_policy"] = "reset_each_file";
        receipt["workload"] = "file_order_synthetic";
        receipt["timed_boundary"] =
            "source-open/hash, LAZ decode, digest and post-decode verification";
        const auto decode_start = Clock::now();
        for (std::size_t index = 0; index < catalog.sources().size(); ++index) {
            const auto source_start = Clock::now();
            std::unique_ptr<replay::Reader> reader;
            replay::Window window;
            Digest features, observations;
            std::uint64_t source_windows{};
            const auto processed_before = total;
            Json row{{"source_index", index},
                     {"source_sha256", catalog.sources().at(index).sha256},
                     {"coordinate_frame", catalog.sources().at(index).coordinate_frame},
                     {"crs", catalog.sources().at(index).crs},
                     {"expected_records", catalog.sources().at(index).records},
                     {"decoded_records", 0},
                     {"raw_records_decoded", 0},
                     {"source_verified_after_decode", false},
                     {"runtime_features_sha256", nullptr},
                     {"source_observations_sha256", nullptr},
                     {"status", "FAIL"}};
            try {
                reader = std::make_unique<replay::Reader>(catalog, index, window_points);
                while (reader->next(window)) {
                    ++windows;
                    ++source_windows;
                    high_water = std::max<std::uint64_t>(high_water, window.observations.size());
                    for (const auto &point : window.observations) {
                        require(point.wire_id == total, "replay wire coverage mismatch");
                        const auto original = catalog.source_for_wire_id(total);
                        require(original.source_index == index &&
                                    original.original_record_index == point.original_record_index,
                                "replay inverse identity mismatch");
                        for (auto coordinate : point.xyz)
                            features.number(coordinate);
                        features.integer(point.intensity, 2);
                        for (auto coordinate : point.integer_xyz)
                            observations.integer(static_cast<std::uint32_t>(coordinate), 4);
                        observations.integer(point.intensity, 2);
                        observations.integer(point.return_number, 1);
                        observations.integer(point.return_count, 1);
                        observations.integer(point.gps_time_bits, 8);
                        ++total;
                    }
                }
                reader->finish();
                row.update(reader->metadata());
                row["runtime_features_sha256"] = features.finish();
                row["source_observations_sha256"] = observations.finish();
                row["status"] = "PASS";
                verified += catalog.sources().at(index).records;
            } catch (const std::exception &error) {
                if (reader)
                    row.update(reader->metadata());
                row["runtime_features_sha256"] = nullptr;
                row["source_observations_sha256"] = nullptr;
                row["error"] = error.what();
            }
            decoded += row.at("decoded_records").get<std::uint64_t>();
            raw_decoded += row.at("raw_records_decoded").get<std::uint64_t>();
            row["processed_records"] = total - processed_before;
            row["windows"] = source_windows;
            row["decode_verify_digest_seconds"] = elapsed(source_start);
            receipt["sources"].push_back(row);
            require(row.at("status") == "PASS", "replay source failed; see source receipt");
        }
        require(total == catalog.total_records(), "replay total decoded coverage mismatch");
        receipt.update({{"records", total},
                        {"windows", windows},
                        {"output_window_high_water_points", high_water},
                        {"decode_verify_digest_seconds", elapsed(decode_start)},
                        {"status", "PASS"}});
    } catch (const std::exception &error) {
        receipt["error"] = error.what();
        receipt["status"] = "FAIL";
    }
    // Processed prefixes and decoded partial windows are not verified coverage.
    receipt.update({{"processed_records", total},
                    {"decoded_records", decoded},
                    {"raw_records_decoded", raw_decoded},
                    {"verified_records", verified},
                    {"windows", windows},
                    {"output_window_high_water_points", high_water},
                    {"elapsed_seconds", elapsed(start)}});
    try {
        if (!output.empty())
            write_new(output, receipt.dump(2, ' ', false, Json::error_handler_t::replace) + "\n");
    } catch (const std::exception &error) {
        receipt["status"] = "FAIL";
        receipt["receipt_error"] = error.what();
    }
    std::cout << receipt.dump(2, ' ', false, Json::error_handler_t::replace) << '\n';
    return receipt.at("status") == "PASS" ? 0 : 1;
}
