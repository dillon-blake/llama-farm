# learning-llamas native build.
#
# Two things get built:
#   1. vendored llama.cpp (CPU backend only at stage 0 -- the CPU backend is the
#      correctness oracle for the whole kernel plan, ROADMAP §4);
#   2. `learningllamas`, the C shim in csrc/, compiled against llama.cpp's *private*
#      src/ internals and exposing a flat C ABI (csrc/farm_api.h).
#
# scikit-build-core drives this from pyproject.toml. Only the `learningllamas` install
# component is packaged into the wheel (see pyproject.toml install.components), so
# llama.cpp's own install rules -- headers, CMake config, pkgconfig -- stay out of it.

cmake_minimum_required(VERSION 3.21)

project(learning_llamas LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "" FORCE)
endif()

# ccache/sccache are honored if present but never required (keeps S0-07's CI caching
# working with no changes here).
find_program(LL_COMPILER_LAUNCHER NAMES ccache sccache)
if(LL_COMPILER_LAUNCHER AND NOT CMAKE_C_COMPILER_LAUNCHER)
    set(CMAKE_C_COMPILER_LAUNCHER "${LL_COMPILER_LAUNCHER}")
    set(CMAKE_CXX_COMPILER_LAUNCHER "${LL_COMPILER_LAUNCHER}")
    message(STATUS "learning-llamas: using compiler launcher ${LL_COMPILER_LAUNCHER}")
endif()

# ---------------------------------------------------------------------------
# Vendored llama.cpp (pinned submodule -- ADR-0001)
# ---------------------------------------------------------------------------

set(LL_VENDOR_LLAMA_CPP "${CMAKE_CURRENT_SOURCE_DIR}/vendor/llama.cpp")

if(NOT EXISTS "${LL_VENDOR_LLAMA_CPP}/CMakeLists.txt")
    message(FATAL_ERROR
        "vendor/llama.cpp is not initialized.\n"
        "  run: git submodule update --init --recursive")
endif()

# Shared libraries, so the documented load order (csrc/README.md) is a real contract and
# not an accident of static linking.
set(BUILD_SHARED_LIBS ON CACHE BOOL "" FORCE)

# Hide by default, everywhere -- so that "it builds on Linux" MEANS "it links on Windows".
#
# The shim links against llama.cpp's internal C++ classes on purpose (ADR-0001). ELF exports every
# symbol of a shared library by default, so on Linux and macOS that costs nothing and nobody has to
# think about it. A PE DLL exports nothing it did not declare, so on Windows each of those internals
# must be marked LLAMA_API_INTERNAL -- and if one is not, the shim fails to link with an unresolved
# external that reads like a broken build rather than a missing declaration.
#
# S1-33 found exactly that, the expensive way: eight internals, discovered by a Windows CI run,
# after the Linux lane had been green for thirty PRs. Forcing hidden visibility here makes the two
# platforms agree -- an unmarked internal is now a build failure on the machine the developer is
# actually sitting at.
set(CMAKE_C_VISIBILITY_PRESET     hidden)
set(CMAKE_CXX_VISIBILITY_PRESET   hidden)
set(CMAKE_VISIBILITY_INLINES_HIDDEN ON)

# Stage 0 is CPU-only. Stages 2-4 flip these on behind their own CI lanes.
set(GGML_METAL  OFF CACHE BOOL "" FORCE)
set(GGML_CUDA   OFF CACHE BOOL "" FORCE)
set(GGML_VULKAN OFF CACHE BOOL "" FORCE)

# We build a library, not a CLI: no examples, no server, no embedded web UI.
# LLAMA_BUILD_TESTS is left overridable -- CI (S0-07) turns it on to build the vendored
# test-backend-ops harness. LLAMA_BUILD_TOOLS likewise: S0-06's Q4_K fixture path has a
# documented fallback to the llama-quantize binary.
set(LLAMA_BUILD_EXAMPLES OFF CACHE BOOL "")
set(LLAMA_BUILD_SERVER   OFF CACHE BOOL "")
set(LLAMA_BUILD_APP      OFF CACHE BOOL "")
set(LLAMA_BUILD_UI       OFF CACHE BOOL "")
set(LLAMA_BUILD_TESTS    OFF CACHE BOOL "")
set(LLAMA_BUILD_TOOLS    OFF CACHE BOOL "")

add_subdirectory("${LL_VENDOR_LLAMA_CPP}" vendor-llama.cpp EXCLUDE_FROM_ALL)

# The vendored build produces, with GGML_BACKEND_DL off and BUILD_SHARED_LIBS on:
#   libggml-base   (contains ggml-opt.cpp -- the training driver lives here)
#   libggml        (backend registry)
#   libllama
# plus one shared library per enabled *backend*, each linked PUBLIC into libggml.
#
# Do not hardcode that backend list. It is platform-dependent in ways that are easy to
# miss: on macOS, ggml enables the BLAS backend (Accelerate) by default, so an Apple
# build produces libggml-blas in addition to libggml-cpu -- and a wheel missing it fails
# at dlopen, not at build time. ggml already publishes the list it actually built, as a
# cache variable set by ggml_add_backend_library(), so use that. It also means stages 2-4
# pick up libggml-metal / libggml-cuda / libggml-vulkan with no change here.
set(LL_VENDOR_LIBS ggml-base ggml llama ${GGML_AVAILABLE_BACKENDS})

message(STATUS "learning-llamas: packaging backends: ${GGML_AVAILABLE_BACKENDS}")

# Strip VERSION/SOVERSION from the vendored libraries. Inside the wheel we want exactly one
# file per library with a predictable name (libggml.so, not libggml.so.0.9.4 plus two
# symlinks): ctypes.CDLL in _ffi/loader.py names them literally, and wheels do not carry
# symlinks portably.
#
# These must be *unset*, not set to "": an empty SOVERSION is still a version, and CMake
# dutifully emits `libggml.so.` with a trailing dot.
foreach(lib IN LISTS LL_VENDOR_LIBS)
    set_property(TARGET ${lib} PROPERTY VERSION)
    set_property(TARGET ${lib} PROPERTY SOVERSION)
endforeach()

add_subdirectory(csrc)

# ---------------------------------------------------------------------------
# Wheel install: everything into learning_llamas/lib/, resolving each other via RPATH.
# ---------------------------------------------------------------------------

set(LL_INSTALL_LIBDIR "learning_llamas/lib")

if(APPLE)
    set(LL_RPATH "@loader_path")
else()
    set(LL_RPATH "$ORIGIN")
endif()

set_target_properties(${LL_VENDOR_LIBS} learningllamas PROPERTIES
    INSTALL_RPATH "${LL_RPATH}"
    BUILD_WITH_INSTALL_RPATH FALSE)

# MSVC (S1-33).
#
# /utf-8 because MSVC otherwise reads a source file in the system codepage, and this codebase has
# UTF-8 in its comments. Without it the build does not fail -- it mis-decodes, and you get warnings
# about characters that are fine.
#
# The DLLs install to the same lib/ directory as everything else (RUNTIME DESTINATION, below), which
# is what lets the loader's os.add_dll_directory find each one's dependencies. The .lib import
# libraries are deliberately kept OUT of the packaged component (see the ARCHIVE line below):
# nothing links against the shim at build time, ctypes opens it at run time.
if(MSVC)
    add_compile_options(/utf-8)
endif()

# The import libraries were ALREADY not installed, and this says so out loud rather than relying on
# it. `cmake --help-command install`, TARGETS: "For shared libraries on DLL platforms, if neither
# RUNTIME nor ARCHIVE destinations are specified, both [...] are installed to their default
# destinations. If either a RUNTIME or ARCHIVE destination is specified, the component is installed
# to that destination, and the other component is not installed." Every target here is a shared
# library (BUILD_SHARED_LIBS is forced ON above), and the rule already named a RUNTIME destination
# and no ARCHIVE one -- so on Windows the .libs generated no install rule at all. No wheel ever
# grew a top-level `lib/*.lib`, and none of this is a bug fix.
#
# What the explicit ARCHIVE clause buys is that the guarantee no longer rests on a defaulting rule
# that has to be read out of the CMake manual to see. The .libs get a component the wheel never
# asks for (pyproject: install.components = ["learningllamas"]), plus EXCLUDE_FROM_ALL so a plain
# `cmake --install` does not ship them either, and the intent survives someone later adding a
# static target or dropping the RUNTIME destination. Net wheel contents: unchanged.
install(TARGETS ${LL_VENDOR_LIBS} learningllamas
        LIBRARY DESTINATION "${LL_INSTALL_LIBDIR}" COMPONENT learningllamas
        RUNTIME DESTINATION "${LL_INSTALL_LIBDIR}" COMPONENT learningllamas
        ARCHIVE DESTINATION "${LL_INSTALL_LIBDIR}" COMPONENT learningllamas_import_libs
                EXCLUDE_FROM_ALL)
