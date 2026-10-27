# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Tuomas Pietilä
# Native ARM64/SM110 configuration only. Requires an idle physical target and matching SDK.
set(SYNREX_PORT_TARGET "thor" CACHE STRING "Gated target" FORCE)
set(SYNREX_ENABLE_CUDA ON CACHE BOOL "Native CUDA backend" FORCE)
set(CMAKE_CUDA_ARCHITECTURES 110 CACHE STRING "Physical Thor SM" FORCE)
