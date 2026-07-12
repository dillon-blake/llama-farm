# Placeholder build. S0-01 only needs `pip install -e .` to produce an importable
# `learning_llamas` package, so this file declares the project and builds nothing.
#
# S0-03 replaces it with the real build: add_subdirectory(vendor/llama.cpp) with the
# CPU-only backend, plus the `learningllamas` C shim in csrc/ compiled against llama.cpp's
# private src/ internals.

cmake_minimum_required(VERSION 3.21)

project(learning_llamas LANGUAGES C CXX)

message(STATUS "learning-llamas: placeholder CMake build (no native targets until S0-03)")
