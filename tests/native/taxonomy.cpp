// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Tuomas Pietilä
#include "synrex/taxonomy.hpp"
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <nlohmann/json.hpp>
#include <stdexcept>
#include <sys/stat.h>
#include <unistd.h>

namespace {
using Json = nlohmann::json;
using synrex::Taxonomy;
void check(bool ok, const char *message) {
    if (!ok)
        throw std::runtime_error(message);
}
template <class F> void rejects(F function, const std::string &diagnostic = "") {
    try {
        function();
    } catch (const std::exception &error) {
        check(diagnostic.empty() || std::string(error.what()).find(diagnostic) != std::string::npos,
              "taxonomy rejected through an unexpected diagnostic");
        return;
    }
    throw std::runtime_error("invalid taxonomy unexpectedly accepted");
}
Json fixture(std::size_t count) {
    Json classes = Json::array();
    for (std::size_t i = 0; i < count; ++i)
        classes.push_back({{"semantic_id", i},
                           {"key", "surface_" + std::to_string(i)},
                           {"name", "CPU surface " + std::to_string(i)},
                           {"las_code", 42}});
    return {
        {"schema", "synrex.taxonomy.v1"},
        {"taxonomy_id", "example.cpu.v1"},
        {"namespace", "example.cpu"},
        {"classes", classes},
        {"provenance", {{"source", "CPU fixture"}, {"author", "SynRex tests"}, {"license", "MIT"}}},
        {"unknown_semantic_id", nullptr}};
}
struct Temporary {
    std::filesystem::path path;
    Temporary() {
        auto pattern = (std::filesystem::temp_directory_path() / "synrex-taxonomy-XXXXXX").string();
        const auto created = ::mkdtemp(pattern.data());
        check(created != nullptr, "taxonomy temporary directory");
        path = created;
    }
    ~Temporary() {
        std::error_code ignored;
        std::filesystem::remove_all(path, ignored);
    }
};
void write(const std::filesystem::path &path, const std::string &bytes) {
    std::ofstream out(path, std::ios::binary);
    out.write(bytes.data(), static_cast<std::streamsize>(bytes.size()));
    out.close();
    check(bool(out), "taxonomy fixture write");
}
} // namespace

int main() {
    try {
        // Fixed cross-language bytes: expected SHA256 was computed with Python hashlib.
        const std::string cross_language =
            R"({"schema":"synrex.taxonomy.v1","taxonomy_id":"example.two.v1",)"
            R"("namespace":"example.cpu","classes":[{"semantic_id":0,"key":"first",)"
            R"("name":"First","las_code":42},{"semantic_id":1,"key":"second",)"
            R"("name":"Second","las_code":42}],"provenance":{"source":"CPU fixture",)"
            R"("author":"SynRex tests","license":"MIT"},"unknown_semantic_id":null})";
        const Taxonomy two(cross_language);
        check(two.raw_bytes() == cross_language && two.classes().size() == 2,
              "exact bytes/two-class taxonomy");
        two.validate_digest("a83b5890af6f255159d4764377704a0d98097d0dcc57c5ea4b90455ee23ef241");
        const Taxonomy newline(cross_language + "\n");
        check(two.sha256() != newline.sha256(), "serialization differences must change identity");
        rejects([&] { two.validate_digest(newline.sha256()); }, "exact artifact digest");
        rejects([&] { two.validate_digest(std::string(64, '0')); }, "exact artifact digest");
        rejects([&] { two.validate_digest(""); }, "exact artifact digest");

        auto four_json = fixture(4);
        const std::vector<std::uint8_t> projection{0, 2, 5, 6};
        for (std::size_t i = 0; i < projection.size(); ++i)
            four_json["classes"][i]["las_code"] = projection[i];
        four_json["classes"][1]["name"] = u8"Metsä 🌲";
        four_json["unknown_semantic_id"] = 3;
        const Taxonomy four(four_json.dump());
        four.validate_channels(4);
        four.validate_projection(projection);
        check(four.unknown_semantic_id() == 3 && four.classes()[1].name == u8"Metsä 🌲" &&
                  four.taxonomy_id() == "example.cpu.v1" &&
                  four.taxonomy_namespace() == "example.cpu" && four.provenance().license == "MIT",
              "explicit four-class metadata lost");
        rejects([&] { four.validate_channels(7); }, "channel mismatch");
        rejects([&] { four.validate_projection({0, 2, 5}); }, "incomplete LAS projection");
        rejects([&] { four.validate_projection({0, 2, 5, 5}); }, "LAS projection mismatch");

        const auto seven_json = fixture(7); // Deliberately non-FRACTAL7, duplicate LAS codes.
        const Taxonomy seven(seven_json.dump());
        seven.validate_channels(7);
        seven.validate_projection(std::vector<std::uint8_t>(7, 42));
        check(!seven.unknown_semantic_id(), "unknown class must not be inferred from K");
        for (const auto change_order : {false, true}) {
            auto changed = seven_json;
            if (change_order) {
                std::swap(changed["classes"][0], changed["classes"][1]);
                changed["classes"][0]["semantic_id"] = 0;
                changed["classes"][1]["semantic_id"] = 1;
            } else {
                changed["classes"][0]["name"] = "Changed meaning";
            }
            const Taxonomy other(changed.dump());
            check(other.taxonomy_id() == seven.taxonomy_id(), "fixture must reuse taxonomy ID");
            other.validate_channels(7);
            other.validate_projection(std::vector<std::uint8_t>(7, 42));
            rejects([&] { seven.validate_digest(other.sha256()); }, "exact artifact digest");
        }
        auto largest_json = fixture(256);
        largest_json["classes"][255]["las_code"] = 255;
        largest_json["unknown_semantic_id"] = 255;
        const Taxonomy largest(largest_json.dump());
        largest.validate_channels(256);
        check(largest.classes()[255].semantic_id == 255 && largest.classes()[255].las_code == 255 &&
                  largest.unknown_semantic_id() == 255,
              "semantic 255/256 channels must remain representable");

        auto invalid = [&](const auto &mutate, const std::string &message = "") {
            auto bad = four_json;
            mutate(bad);
            rejects([&] { Taxonomy ignored(bad.dump()); }, message);
        };
        for (const auto count : {0, 1, 257})
            rejects([&] { Taxonomy ignored(fixture(count).dump()); }, "2..256");
        invalid([](Json &j) { j["schema"] = "synrex.taxonomy.v2"; }, "unsupported taxonomy schema");
        invalid([](Json &j) { j["extra"] = 1; }, "unknown/missing fields");
        invalid([](Json &j) { j.erase("unknown_semantic_id"); }, "unknown/missing fields");
        invalid([](Json &j) { j["classes"][1]["semantic_id"] = 0; }, "IDs must be ordered");
        invalid([](Json &j) { j["classes"][0]["semantic_id"] = -1; }, "integer range");
        invalid([](Json &j) { j["classes"][0]["semantic_id"] = 256; }, "integer range");
        invalid([](Json &j) { j["classes"][0]["semantic_id"] = 0.0; }, "integer required");
        invalid([](Json &j) { j["classes"][1]["key"] = "surface_0"; }, "duplicate class key");
        invalid([](Json &j) { j["classes"][0]["las_code"] = 256; }, "integer range");
        invalid([](Json &j) { j["classes"][0]["las_code"] = -1; }, "integer range");
        invalid([](Json &j) { j["classes"][0]["las_code"] = true; }, "integer required");
        invalid([](Json &j) { j["classes"][0].erase("las_code"); }, "unknown/missing fields");
        invalid([](Json &j) { j["unknown_semantic_id"] = 4; }, "integer range");
        invalid([](Json &j) { j["unknown_semantic_id"] = false; }, "integer required");
        invalid([](Json &j) { j["taxonomy_id"] = "bad id"; }, "ASCII grammar");
        invalid([](Json &j) { j["namespace"] = std::string(129, 'x'); }, "identifier size");
        invalid([](Json &j) { j["classes"][0]["key"] = std::string(65, 'x'); }, "identifier size");
        invalid([](Json &j) { j["classes"][0]["name"] = std::string(257, 'x'); }, "text size");
        invalid([](Json &j) { j["classes"][0]["name"] = "bad\nname"; }, "C0/DEL");
        invalid([](Json &j) { j["classes"][0]["name"] = std::string(1, '\0'); }, "C0/DEL");
        invalid([](Json &j) { j["provenance"]["source"] = std::string(513, 'x'); }, "text size");
        invalid([](Json &j) { j["provenance"]["author"] = ""; }, "text size");
        invalid([](Json &j) { j["provenance"]["license"] = std::string(1, '\x7f'); }, "C0/DEL");
        auto boundaries = fixture(2);
        boundaries["taxonomy_id"] = std::string(128, 'i');
        boundaries["namespace"] = std::string(128, 'n');
        boundaries["classes"][0]["key"] = std::string(64, 'k');
        boundaries["classes"][0]["name"] = std::string(256, 'x');
        boundaries["provenance"]["source"] = std::string(512, 's');
        const Taxonomy accepted_boundaries(boundaries.dump());
        check(accepted_boundaries.classes()[0].name.size() == 256, "field bounds rejected");
        auto duplicate = cross_language;
        duplicate.insert(1, R"("schema":"synrex.taxonomy.v1",)");
        rejects([&] { Taxonomy ignored(duplicate); }, "duplicate JSON key");
        duplicate = cross_language;
        duplicate.insert(duplicate.find("\"source\""), R"("source":"duplicate",)");
        rejects([&] { Taxonomy ignored(duplicate); }, "duplicate JSON key");
        auto invalid_utf8 = cross_language;
        invalid_utf8.replace(invalid_utf8.find("First"), 5, "\xc3\x28");
        rejects([&] { Taxonomy ignored(invalid_utf8); });
        auto bad_surrogate = cross_language;
        bad_surrogate.replace(bad_surrogate.find("First"), 5, "\\uD800");
        rejects([&] { Taxonomy ignored(bad_surrogate); });
        rejects([&] { Taxonomy ignored(std::string(8, '[') + "0" + std::string(8, ']')); },
                "unknown/missing fields");
        rejects([&] { Taxonomy ignored(std::string(9, '[') + "0" + std::string(9, ']')); },
                "nesting limit");
        const auto padded =
            cross_language + std::string(Taxonomy::max_bytes - cross_language.size(), ' ');
        const Taxonomy maximum_bytes(padded);
        check(maximum_bytes.raw_bytes().size() == Taxonomy::max_bytes, "exact byte limit");
        rejects([&] { Taxonomy ignored(padded + " "); }, "artifact size");

        Temporary temporary;
        const auto artifact = temporary.path / "taxonomy.json";
        write(artifact, cross_language);
        check(Taxonomy::read(artifact).sha256() == two.sha256(), "file/raw byte identity");
        std::filesystem::create_symlink(artifact, temporary.path / "link");
        rejects([&] { Taxonomy::read(temporary.path / "link"); }, "cannot open regular artifact");
        rejects([&] { Taxonomy::read(temporary.path); }, "must be regular");
        check(::mkfifo((temporary.path / "fifo").c_str(), 0600) == 0, "fixture FIFO");
        rejects([&] { Taxonomy::read(temporary.path / "fifo"); }, "must be regular");
        write(artifact, padded + " ");
        rejects([&] { Taxonomy::read(artifact); }, "artifact size");
        write(artifact, "");
        rejects([&] { Taxonomy::read(artifact); }, "artifact size");
        std::cout << R"({"status":"PASS","taxonomy_contract":"synrex.taxonomy.v1",)"
                     R"("neural_inference":"NOT_RUN: CPU metadata fixtures"})"
                  << '\n';
        return 0;
    } catch (const std::exception &error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
