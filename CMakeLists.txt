# Placeholder build. Stage 0 only needs `pip install -e .` to produce an importable
# `learning_llamas` package, so this file declares the project and builds nothing.
#
# S0-03 replaces it with the real build: add_subdirectory(vendor/llama.cpp) with the
# CPU-only backend, plus the `learningllamas` C shim in csrc/ compiled against llama.cpp's
# private src/ internals. The vendored tree is pinned by the git submodule (ADR-0001).

cmake_minimum_required(VERSION 3.21)

project(learning_llamas LANGUAGES C CXX)

set(LL_VENDOR_LLAMA_CPP "${CMAKE_CURRENT_SOURCE_DIR}/vendor/llama.cpp")

if(NOT EXISTS "${LL_VENDOR_LLAMA_CPP}/CMakeLists.txt")
    message(FATAL_ERROR
        "vendor/llama.cpp is not initialized.\n"
        "  run: git submodule update --init --recursive")
endif()

message(STATUS "learning-llamas: placeholder CMake build (no native targets until S0-03)")
message(STATUS "learning-llamas: vendored llama.cpp found at ${LL_VENDOR_LLAMA_CPP}")
