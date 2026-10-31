// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Tuomas Pietilä
#include "synrex/replay_queue.hpp"
#include <iostream>
#include <thread>

namespace {
using namespace synrex::replay;
unsigned checks{};
void check(bool okay, const std::string &message) {
    ++checks;
    if (!okay)
        throw std::runtime_error(message);
}
template <class F> void rejects(F operation, const std::string &message) {
    try {
        operation();
    } catch (const std::exception &error) {
        check(std::string(error.what()).find(message) != std::string::npos,
              "unexpected exception: " + std::string(error.what()));
        return;
    }
    throw std::runtime_error("unexpected acceptance: " + message);
}
Window window(std::uint64_t first, std::size_t count, std::size_t allocation = 0) {
    Window result;
    result.source_index = static_cast<std::size_t>(first / 100);
    result.observations.reserve(allocation ? allocation : count);
    result.observations.resize(count);
    for (std::size_t i = 0; i < count; ++i) {
        result.observations[i].wire_id = first + i;
        result.observations[i].original_record_index = i;
        result.observations[i].gps_time_bits = 0x7ff8000000000123ULL;
    }
    return result;
}
// Every failure path cancels before joining, so a failed assertion cannot leave
// a blocked peer or make the test process depend on a timing-based detach.
struct Worker {
    WindowQueue &queue;
    std::exception_ptr error;
    std::thread thread;
    template <class F>
    Worker(WindowQueue &value, F operation)
        : queue(value), thread([this, operation] {
              try {
                  operation();
              } catch (...) {
                  error = std::current_exception();
              }
          }) {}
    void join() { thread.join(); }
    ~Worker() {
        if (thread.joinable()) {
            queue.cancel();
            thread.join();
        }
    }
};
template <class Predicate> void await(Predicate predicate) {
    const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(5);
    while (!predicate()) {
        if (std::chrono::steady_clock::now() >= deadline)
            throw std::runtime_error("queue synchronization deadline");
        std::this_thread::yield();
    }
}
void check_cancelled(const std::exception_ptr &error) {
    check(bool(error), "blocked peer must throw on cancellation");
    try {
        std::rethrow_exception(error);
    } catch (const Cancelled &) {
        ++checks;
    }
}
void admission() {
    rejects([] { WindowQueue queue(0); }, "1..65536");
    rejects([] { WindowQueue queue(65537); }, "1..65536");
    rejects([] { WindowQueue queue(1, 0); }, "1..2");
    rejects([] { WindowQueue queue(1, 3); }, "1..2");
    WindowQueue maximum(65536);
    check(maximum.stats().byte_limit == 2 * 65536 * sizeof(Observation), "maximum byte limit");
    WindowQueue queue(4);
    queue.push(window(0, 4));
    auto inflated = window(100, 1, 5);
    rejects([&] { queue.push(std::move(inflated)); }, "allocation");
    check(inflated.observations.size() == 1, "rejected window ownership retained");
    check(queue.stats().failed && !queue.stats().queued_allocation_bytes,
          "allocation rejection fails and clears prior storage");
    Window out;
    rejects([&] { queue.pop(out); }, "allocation");
    WindowQueue oversized(4);
    rejects([&] { oversized.push(window(0, 5)); }, "bound");
    WindowQueue empty(4);
    rejects([&] { empty.push(Window{}); }, "bound");
    WindowQueue null_failure(4);
    rejects([&] { null_failure.fail(nullptr); }, "needs an exception");
    check(!null_failure.stats().failed, "null failure rejected without changing queue");
}
void fifo_backpressure() {
    WindowQueue queue(4);
    queue.push(window(0, 4));
    queue.push(window(100, 4));
    auto third = window(200, 2, 4);
    Worker producer(queue, [&] { queue.push(std::move(third)); });
    await([&] { return queue.stats().producer_waiting; });
    auto stats = queue.stats();
    check(stats.queued_items == 2 && stats.queued_bytes == 8 * sizeof(Observation),
          "full queue retains exactly two windows");
    check(stats.queued_allocation_bytes == stats.byte_limit && stats.high_water_items == 2,
          "full queue admitted allocation bound");
    check(third.observations.size() == 2, "blocked producer has not transferred ownership");
    check(stats.producer_waits == 1 && stats.producer_wait_ns > 0 && stats.oldest_age_ns > 0,
          "blocked wait and age reported without retained samples");
    Window out;
    check(queue.pop(out) && out.observations.front().wire_id == 0, "FIFO first window");
    producer.join();
    check(!producer.error, "producer resumes after space becomes available");
    queue.close();
    check(queue.pop(out) && out.source_index == 1 && out.observations.back().wire_id == 103,
          "FIFO second window after wrap");
    check(queue.pop(out) && out.source_index == 2 && out.observations.size() == 2 &&
              out.observations.back().wire_id == 201 &&
              out.observations.back().gps_time_bits == 0x7ff8000000000123ULL,
          "partial window and exact attributes survive queue");
    check(!queue.pop(out) && out.observations.empty(), "only clean drained EOF returns false");
    stats = queue.stats();
    check(stats.closed && !stats.failed && !stats.queued_items && !stats.queued_bytes &&
              !stats.queued_allocation_bytes,
          "clean close drains all retained payload");
    check(stats.high_water_bytes <= stats.byte_limit &&
              stats.high_water_allocation_bytes <= stats.byte_limit && stats.max_age_ns > 0 &&
              !stats.producer_waiting,
          "high water and wait metrics respect bounds");
    check(stats.json().at("byte_limit") == stats.byte_limit, "receipt stats JSON");
    rejects([&] { queue.push(window(0, 1)); }, "closed");
}
void producer_cancellation() {
    WindowQueue queue(4, 1);
    queue.push(window(0, 1, 4));
    Worker producer(queue, [&] { queue.push(window(100, 1)); });
    await([&] { return queue.stats().producer_waiting; });
    queue.cancel();
    producer.join();
    check_cancelled(producer.error);
    const auto stats = queue.stats();
    check(stats.failed && stats.closed && !stats.queued_items && !stats.queued_allocation_bytes &&
              stats.high_water_items == 1 && !stats.producer_waiting,
          "cancel clears single-window queue and wakes producer");
    check(stats.discarded_items == 1 && stats.discarded_points == 1 &&
              stats.discarded_allocation_bytes == 4 * sizeof(Observation),
          "cancel retains exact discard counts and allocation bytes");
    Window out;
    rejects([&] { queue.pop(out); }, "cancelled");
}
void consumer_cancellation_and_close() {
    WindowQueue queue(4);
    Worker consumer(queue, [&] {
        Window out;
        queue.pop(out);
    });
    await([&] { return queue.stats().consumer_waiting; });
    queue.cancel();
    consumer.join();
    check_cancelled(consumer.error);
    check(!queue.stats().consumer_waiting && queue.stats().consumer_waits == 1,
          "cancel wakes blocked consumer");
    WindowQueue clean(4);
    bool eof{};
    Worker clean_consumer(clean, [&] {
        Window out;
        eof = !clean.pop(out);
    });
    await([&] { return clean.stats().consumer_waiting; });
    clean.close();
    clean_consumer.join();
    check(eof && !clean_consumer.error, "clean close wakes blocked consumer with EOF");

    WindowQueue full(4, 1);
    full.push(window(0, 1));
    Worker producer(full, [&] { full.push(window(100, 1)); });
    await([&] { return full.stats().producer_waiting; });
    full.close();
    producer.join();
    check(bool(producer.error), "clean close wakes blocked producer");
    rejects([&] { std::rethrow_exception(producer.error); }, "closed");
    Window out;
    check(full.pop(out) && out.observations[0].wire_id == 0 && !full.pop(out),
          "closing a full queue preserves previously admitted data");
}
void concurrent_fifo() {
    WindowQueue queue(4);
    Worker producer(queue, [&] {
        for (std::uint64_t i = 0; i < 4096; ++i)
            queue.push(window(i * 4, 4));
        queue.close();
    });
    Window out;
    std::uint64_t received{};
    while (queue.pop(out)) {
        check(out.observations.size() == 4, "concurrent window size");
        for (const auto &observation : out.observations)
            check(observation.wire_id == received++, "concurrent FIFO record coverage");
    }
    producer.join();
    const auto stats = queue.stats();
    check(!producer.error && received == 16384 && stats.high_water_items <= 2 &&
              stats.high_water_bytes <= stats.byte_limit &&
              stats.high_water_allocation_bytes <= stats.byte_limit && !stats.queued_items,
          "sustained two-thread FIFO exact coverage within byte bounds");
}
void late_failure() {
    WindowQueue queue(4);
    queue.push(window(0, 4));
    queue.close();
    queue.fail(std::make_exception_ptr(std::runtime_error("decoder finish verification failed")));
    queue.cancel(); // Cleanup must preserve the original decode exception.
    Window out;
    rejects([&] { queue.pop(out); }, "decoder finish");
    rejects([&] { queue.pop(out); }, "decoder finish");
    rejects([&] { queue.push(window(100, 1)); }, "decoder finish");
    check(!queue.stats().queued_items && !queue.stats().queued_allocation_bytes,
          "late failure discards queued results instead of clean drain");
    check(queue.stats().discarded_items == 1 && queue.stats().discarded_points == 4,
          "repeated cleanup never double counts discarded points");

    WindowQueue blocked(4, 1);
    blocked.push(window(0, 1));
    Worker producer(blocked, [&] { blocked.push(window(100, 1)); });
    await([&] { return blocked.stats().producer_waiting; });
    blocked.fail(std::make_exception_ptr(std::runtime_error("source changed")));
    producer.join();
    check(bool(producer.error), "failure wakes blocked producer");
    rejects([&] { std::rethrow_exception(producer.error); }, "source changed");

    WindowQueue reader_failure(4);
    Worker consumer(reader_failure, [&] {
        Window value;
        reader_failure.pop(value);
    });
    await([&] { return reader_failure.stats().consumer_waiting; });
    reader_failure.fail(std::make_exception_ptr(std::runtime_error("decode failed")));
    consumer.join();
    check(bool(consumer.error), "failure wakes blocked consumer");
    rejects([&] { std::rethrow_exception(consumer.error); }, "decode failed");
}
} // namespace
int main() {
    try {
        admission();
        fifo_backpressure();
        producer_cancellation();
        consumer_cancellation_and_close();
        concurrent_fifo();
        late_failure();
        std::cout << "replay queue checks=" << checks << '\n';
        return 0;
    } catch (const std::exception &error) {
        std::cerr << "replay queue failed: " << error.what() << '\n';
        return 1;
    }
}
