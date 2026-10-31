// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Tuomas Pietilä
#pragma once
#include "synrex/generic_codec.hpp"
#include <memory>

namespace synrex::generic {
// Constant retained digest state/counters. The descriptor must outlive this helper.
// Each add uses a bounded temporary classification copy plus one codec envelope.
// Confers no execution authority and never changes the actual wire result.
class ClassificationTranscript final {
    struct Impl;
    std::unique_ptr<Impl> impl_;
    const Descriptor &descriptor_;

  public:
    explicit ClassificationTranscript(const Descriptor &);
    ~ClassificationTranscript();
    ClassificationTranscript(const ClassificationTranscript &) = delete;
    ClassificationTranscript &operator=(const ClassificationTranscript &) = delete;
    void add(const Classification &);
    Json receipt() const;
};
} // namespace synrex::generic
