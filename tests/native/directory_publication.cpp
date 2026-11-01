// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Tuomas Pietilä
// Direct CPU helper tests; injected failures do not change production defaults.
#include "synrex/directory_publication.hpp"
#include <atomic>
#include <cerrno>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <sys/wait.h>
#include <thread>
#include <unistd.h>

namespace {
namespace fs = std::filesystem;
unsigned checks{};
void check(bool okay, const std::string &message) {
    ++checks;
    if (!okay)
        throw std::runtime_error(message);
}
template <class F> void rejects(F fn) {
    try {
        fn();
    } catch (const std::exception &) {
        ++checks;
        return;
    }
    throw std::runtime_error("publication unexpectedly accepted");
}
template <int Error> int unavailable(int, const char *, int, const char *, unsigned) {
    errno = Error;
    return -1;
}
void write(const fs::path &path, const std::string &data) {
    std::ofstream out(path, std::ios::binary);
    out << data;
    out.close();
    check(bool(out), "fixture write");
}
std::string bytes(const fs::path &path) {
    std::ifstream in(path, std::ios::binary);
    return {std::istreambuf_iterator<char>(in), std::istreambuf_iterator<char>()};
}
fs::path stage(const fs::path &parent, const std::string &data = "complete verified bytes") {
    auto pattern = (parent / ".publication-XXXXXX").string();
    const auto made = ::mkdtemp(pattern.data());
    check(made, "mkdtemp");
    const fs::path result(made);
    write(result / "payload", data);
    synrex::sync_publication_stage(result);
    return result;
}
struct Temporary {
    fs::path path;
    explicit Temporary(const fs::path &parent) {
        auto pattern = (parent / "directory-publication-XXXXXX").string();
        const auto made = ::mkdtemp(pattern.data());
        check(made, "temporary parent");
        path = made;
    }
    ~Temporary() {
        std::error_code ignored;
        fs::remove_all(path, ignored);
    }
};
} // namespace
int main(int argc, char **argv) {
    try {
        check(argc == 2, "usage: directory_publication TEMP_PARENT");
        Temporary temporary(argv[1]);
        const auto native_stage = stage(temporary.path);
        const auto native_out = temporary.path / "native";
        const auto actual = synrex::publish_directory_noreplace(native_stage, native_out);
        synrex::sync_publication_parent(native_out);
        check(bytes(native_out / "payload") == "complete verified bytes", "native bytes");
        check(fs::is_symlink(native_out) ==
                  (actual == synrex::DirectoryPublication::relative_symlink),
              "actual publication method matches directory representation");
        check(fs::exists(native_stage) == fs::is_symlink(native_out), "stage ownership transfer");

        unsigned index{};
        for (const auto operation :
             {unavailable<EINVAL>, unavailable<ENOSYS>, unavailable<EOPNOTSUPP>}) {
            const auto private_stage = stage(temporary.path);
            const auto output = temporary.path / ("unsupported-" + std::to_string(++index));
            const auto method =
                synrex::publish_directory_noreplace(private_stage, output, operation);
            synrex::sync_publication_parent(output);
            check(method == synrex::DirectoryPublication::relative_symlink &&
                      fs::read_symlink(output) == private_stage.filename(),
                  "unsupported native operation creates relative same-parent alias");
            check(fs::exists(private_stage) &&
                      bytes(output / "payload") == bytes(private_stage / "payload"),
                  "retained storage is complete");
            // Reopen from another process after all helper descriptors closed.
            const auto child = ::fork();
            check(child >= 0, "reader fork");
            if (child == 0)
                ::_exit(bytes(output / "payload") == "complete verified bytes" ? 0 : 1);
            int status{};
            check(::waitpid(child, &status, 0) == child && WIFEXITED(status) &&
                      WEXITSTATUS(status) == 0,
                  "independent process reads persistent bytes");
        }
        for (const auto operation :
             {unavailable<EACCES>, unavailable<EPERM>, unavailable<EIO>, unavailable<EXDEV>,
              unavailable<ENOSPC>, unavailable<EEXIST>, unavailable<ENOTEMPTY>}) {
            const auto private_stage = stage(temporary.path);
            const auto output = temporary.path / ("fatal-" + std::to_string(++index));
            rejects([&] { synrex::publish_directory_noreplace(private_stage, output, operation); });
            check(!fs::exists(output) && !fs::is_symlink(output) && fs::exists(private_stage),
                  "other native errors never fall back or consume stage");
        }
        // Atomic symlink creation must preserve every existing destination type.
        for (unsigned kind = 0; kind < 5; ++kind) {
            const auto output = temporary.path / ("collision-" + std::to_string(kind));
            if (kind == 0)
                write(output, "existing file");
            if (kind == 1 || kind == 2)
                fs::create_directory(output);
            if (kind == 2)
                write(output / "prior", "existing directory");
            if (kind == 3)
                fs::create_symlink("absent-target", output);
            if (kind == 4)
                fs::create_directory_symlink(native_out.filename(), output);
            const auto before = fs::symlink_status(output);
            const auto link = fs::is_symlink(before) ? fs::read_symlink(output) : fs::path();
            const auto private_stage = stage(temporary.path);
            for (auto operation : {unavailable<EINVAL>, unavailable<ENOSYS>})
                rejects(
                    [&] { synrex::publish_directory_noreplace(private_stage, output, operation); });
            check(fs::symlink_status(output).type() == before.type() && fs::exists(private_stage),
                  "collision preserves destination type and unpublished stage");
            if (kind == 0)
                check(bytes(output) == "existing file", "file collision preserved");
            if (kind == 1)
                check(fs::is_empty(output), "empty directory collision preserved");
            if (kind == 2)
                check(bytes(output / "prior") == "existing directory",
                      "directory collision preserved");
            if (kind >= 3)
                check(fs::read_symlink(output) == link, "symlink collision preserved");
        }
        for (unsigned trial = 0; trial < 12; ++trial) {
            const auto a = stage(temporary.path, "publisher A"),
                       b = stage(temporary.path, "publisher B");
            const auto output = temporary.path / ("race-" + std::to_string(trial));
            std::atomic<unsigned> ready{}, successes{};
            auto publish = [&](const fs::path &private_stage) {
                ++ready;
                while (ready.load() != 2)
                    std::this_thread::yield();
                try {
                    synrex::publish_directory_noreplace(private_stage, output, unavailable<EINVAL>);
                    ++successes;
                } catch (const std::system_error &error) {
                    if (error.code().value() != EEXIST)
                        std::terminate();
                }
            };
            std::thread first(publish, a), second(publish, b);
            first.join();
            second.join();
            check(successes == 1 && fs::exists(a) && fs::exists(b),
                  "exactly one concurrent publisher wins");
            const auto winner = fs::read_symlink(output);
            check((winner == a.filename() && bytes(output / "payload") == "publisher A") ||
                      (winner == b.filename() && bytes(output / "payload") == "publisher B"),
                  "concurrent publication exposes one whole completed stage");
        }
        const auto private_stage = stage(temporary.path);
        const auto alias_stage = temporary.path / ".stage-link";
        fs::create_directory_symlink(private_stage.filename(), alias_stage);
        rejects([&] {
            synrex::publish_directory_noreplace(alias_stage, temporary.path / "invalid",
                                                unavailable<EINVAL>);
        });
        rejects([&] {
            synrex::publish_directory_noreplace(private_stage, private_stage, unavailable<EINVAL>);
        });
        fs::create_directory(temporary.path / "elsewhere");
        rejects([&] {
            synrex::publish_directory_noreplace(private_stage, temporary.path / "elsewhere" / "out",
                                                unavailable<EINVAL>);
        });
        const auto nested = temporary.path / "relocatable";
        fs::create_directory(nested);
        const auto relocating_stage = stage(nested);
        synrex::publish_directory_noreplace(relocating_stage, nested / "output",
                                            unavailable<EINVAL>);
        fs::rename(nested, temporary.path / "moved-parent");
        check(bytes(temporary.path / "moved-parent" / "output" / "payload") ==
                  "complete verified bytes",
              "relative publication survives moving containing directory");
        for (const bool publish : {false, true}) {
            const auto abandoned_stage = stage(temporary.path, "bytes survive worker exit");
            const auto output =
                temporary.path / (publish ? "exit-after-publication" : "exit-before-publication");
            const auto child = ::fork();
            check(child >= 0, "interrupted worker fork");
            if (child == 0) {
                if (publish) {
                    synrex::publish_directory_noreplace(abandoned_stage, output,
                                                        unavailable<EINVAL>);
                    synrex::sync_publication_parent(output);
                }
                // Bypass all C++ cleanup to simulate an abruptly exiting worker.
                ::_exit(73);
            }
            int status{};
            check(::waitpid(child, &status, 0) == child && WIFEXITED(status) &&
                      WEXITSTATUS(status) == 73,
                  "worker exits without cleanup");
            check(fs::is_directory(abandoned_stage), "worker exit retains backing allocation");
            if (publish)
                check(fs::is_symlink(output) &&
                          bytes(output / "payload") == "bytes survive worker exit",
                      "restart reads complete published output after abrupt exit");
            else
                check(!fs::exists(output) && !fs::is_symlink(output),
                      "interrupted staging exposes no public output");
        }
        std::cout << "{\"status\":\"PASS\",\"checks\":" << checks << ",\"native_method\":\""
                  << synrex::directory_publication_name(actual)
                  << "\",\"concurrent_trials\":12,\"gpu_execution\":false}\n";
        return 0;
    } catch (const std::exception &error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
