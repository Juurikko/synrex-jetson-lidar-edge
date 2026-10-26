# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Tuomas Pietilä
# Developer and project lead: Tuomas Pietilä.
# AI-assisted software engineering, including OpenAI Codex.
cmake_minimum_required(VERSION 3.20)
project(synrex_laz_integrity LANGUAGES CXX)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(LAZPERF_SOURCE_DIR "" CACHE PATH "External laz-perf 3.4.0 source checkout")
if(NOT EXISTS "${LAZPERF_SOURCE_DIR}/cpp/lazperf/readers.hpp")
  message(FATAL_ERROR "Set LAZPERF_SOURCE_DIR to an external laz-perf 3.4.0 checkout")
endif()
# Compile only the documented codec library; no upstream datasets/tests/tools are fetched.
file(GLOB LAZPERF_SOURCES CONFIGURE_DEPENDS
  "${LAZPERF_SOURCE_DIR}/cpp/lazperf/*.cpp"
  "${LAZPERF_SOURCE_DIR}/cpp/lazperf/detail/*.cpp")
add_library(lazperf_review STATIC ${LAZPERF_SOURCES})
target_include_directories(lazperf_review SYSTEM PUBLIC "${LAZPERF_SOURCE_DIR}/cpp")
target_compile_definitions(lazperf_review PUBLIC LAZPERF_VENDORED)
add_executable(synrex-laz-labels src/preserve_laz.cpp)
target_link_libraries(synrex-laz-labels PRIVATE lazperf_review)
target_compile_options(synrex-laz-labels PRIVATE -Wall -Wextra -Wpedantic -Werror)
option(SYNREX_SANITIZE "Address/undefined-behavior sanitizer build" OFF)
if(SYNREX_SANITIZE)
  foreach(t lazperf_review synrex-laz-labels)
    target_compile_options(${t} PRIVATE -fsanitize=address,undefined -fno-omit-frame-pointer)
  endforeach()
  target_link_options(synrex-laz-labels PRIVATE -fsanitize=address,undefined)
endif()
