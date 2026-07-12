# liblearningllamas -- the C shim (Layer 1).

# The version string is single-sourced from the Python package so a stale native build is
# detectable: ll_version() must equal learning_llamas.__version__.
file(READ "${CMAKE_SOURCE_DIR}/src/learning_llamas/__init__.py" LL_INIT_PY)
string(REGEX MATCH "__version__ = \"([^\"]+)\"" _ "${LL_INIT_PY}")
set(LL_VERSION "${CMAKE_MATCH_1}")
if(NOT LL_VERSION)
    message(FATAL_ERROR "could not read __version__ from src/learning_llamas/__init__.py")
endif()

# The vendored commit is baked in at configure time; it is what ll_probe() returns and what
# the S0-04 version lock compares against.
find_package(Git QUIET REQUIRED)
execute_process(
    COMMAND "${GIT_EXECUTABLE}" rev-parse HEAD
    WORKING_DIRECTORY "${LL_VENDOR_LLAMA_CPP}"
    OUTPUT_VARIABLE LL_LLAMA_CPP_COMMIT
    OUTPUT_STRIP_TRAILING_WHITESPACE
    RESULT_VARIABLE LL_GIT_RESULT)
if(NOT LL_GIT_RESULT EQUAL 0)
    message(FATAL_ERROR
        "could not read the vendor/llama.cpp commit.\n"
        "  run: git submodule update --init --recursive")
endif()

message(STATUS "learning-llamas: version ${LL_VERSION}, vendored llama.cpp ${LL_LLAMA_CPP_COMMIT}")

configure_file(farm_version.h.in "${CMAKE_CURRENT_BINARY_DIR}/farm_version.h" @ONLY)

# The Python half of the version lock (S0-04). Generated into the *source* tree rather than the
# install tree on purpose: under an editable install `learning_llamas._ffi` resolves out of
# src/, so a copy installed into site-packages would either be invisible or -- worse -- shadow
# the whole subpackage. It is gitignored and rewritten on every configure.
configure_file(
    "${CMAKE_SOURCE_DIR}/src/learning_llamas/_ffi/_version_lock.py.in"
    "${CMAKE_SOURCE_DIR}/src/learning_llamas/_ffi/_version_lock.py"
    @ONLY)

add_library(learningllamas
    farm_api.h
    farm_probe.cpp
    farm_train.cpp
    farm_internals.cpp)

target_include_directories(learningllamas
    PUBLIC  "${CMAKE_CURRENT_SOURCE_DIR}"
    PRIVATE "${CMAKE_CURRENT_BINARY_DIR}"
            # llama.cpp's *private* headers. This is the whole point of the shim: the
            # training loop's dependencies (llm_graph_params, llm_graph_result, the batch
            # allocator, the memory module) live here and nowhere else.
            "${LL_VENDOR_LLAMA_CPP}/src")

target_link_libraries(learningllamas PRIVATE llama ggml)

target_compile_definitions(learningllamas PRIVATE LL_BUILD)

set_target_properties(learningllamas PROPERTIES
    C_VISIBILITY_PRESET hidden
    CXX_VISIBILITY_PRESET hidden
    VISIBILITY_INLINES_HIDDEN ON)
