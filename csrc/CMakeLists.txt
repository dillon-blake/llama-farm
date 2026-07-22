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
#
# Git is authoritative WHENEVER IT CAN ANSWER, and in a git checkout it always can: the gitlink is
# the one thing about the vendored tree that is unambiguously true, and the whole point of the lock
# is that the gitlink, the native build and the Python package agree.
#
# But an unpacked sdist is not a git checkout -- there is no .git in vendor/llama.cpp and none at
# the top level either -- and `python -m build` with no arguments builds the wheel FROM the sdist,
# as does `pip install <source tarball>`. Those paths used to die here on a hard FATAL_ERROR whose
# advice ("run: git submodule update --init --recursive") cannot possibly work in a tree that was
# never a checkout. CI never noticed because CI only ever builds --wheel.
#
# So when there is NO GIT METADATA to answer from, fall back to the commit RECORDED IN THE SDIST:
# the _version_lock.py generated below is written back into the source tree, and pyproject.toml's
# sdist.cmake + sdist.include put that generated file in the tarball. Same string, produced by this
# same rule, at the moment the sdist was cut.
#
# "No metadata" is the ONLY thing that opens the fallback. A vendor/llama.cpp that has a .git stays
# on the git path or fails hard -- whether git could not read HEAD, or was not installed at all
# (both below). What must never happen is a build inventing a commit or reusing a stale one, so
# with neither source available this is still a hard error.
find_package(Git QUIET)
set(LL_LLAMA_CPP_COMMIT "")

if(Git_FOUND AND EXISTS "${LL_VENDOR_LLAMA_CPP}/.git")
    execute_process(
        COMMAND "${GIT_EXECUTABLE}" rev-parse HEAD
        WORKING_DIRECTORY "${LL_VENDOR_LLAMA_CPP}"
        OUTPUT_VARIABLE LL_LLAMA_CPP_COMMIT
        OUTPUT_STRIP_TRAILING_WHITESPACE
        RESULT_VARIABLE LL_GIT_RESULT)
    if(NOT LL_GIT_RESULT EQUAL 0)
        # .git is there and git still could not read HEAD: a half-initialised submodule, not an
        # sdist. Say so rather than falling through to a stale recorded value.
        message(FATAL_ERROR
            "vendor/llama.cpp has a .git but its HEAD could not be read.\n"
            "  run: git submodule update --init --recursive")
    endif()
endif()

# The OTHER way the fallback can be reached from a checkout, and it has to be just as hard an
# error. The guard above needs BOTH Git_FOUND and a .git, so relaxing `find_package(Git REQUIRED)`
# to a soft find opened a path where the metadata IS there and only the executable is missing --
# and that path silently baked whatever commit `_version_lock.py` happened to hold. That file is
# generated and gitignored, so it survives a branch switch and a submodule bump; the native half
# and the Python half are configured from this same variable, so ll_probe() and VENDORED_COMMIT
# would agree on the stale string and _verify_commit would pass. The lock's whole job is to catch
# exactly that, so it is not allowed to be defeated quietly.
if(EXISTS "${LL_VENDOR_LLAMA_CPP}/.git" AND NOT Git_FOUND)
    message(FATAL_ERROR
        "vendor/llama.cpp is a git checkout but no git executable was found, so the vendored\n"
        "commit cannot be read from the one source that is authoritative for it.\n"
        "  install git, or build from an sdist (which carries the commit recorded at sdist time)")
endif()

set(LL_RECORDED_COMMIT_FILE "${CMAKE_SOURCE_DIR}/src/learning_llamas/_ffi/_version_lock.py")
if(NOT LL_LLAMA_CPP_COMMIT AND EXISTS "${LL_RECORDED_COMMIT_FILE}")
    file(READ "${LL_RECORDED_COMMIT_FILE}" LL_RECORDED_LOCK)
    string(REGEX MATCH "VENDORED_COMMIT = \"([0-9a-fA-F]+)\"" _ "${LL_RECORDED_LOCK}")
    set(LL_LLAMA_CPP_COMMIT "${CMAKE_MATCH_1}")
    if(LL_LLAMA_CPP_COMMIT)
        # "no git metadata", not "from an sdist": the only way to reach this line is with no
        # vendor/llama.cpp/.git at all (the guard above makes the has-metadata-but-no-git case
        # fatal), but the recorded file itself carries no provenance, so the message says what was
        # observed rather than inferring where the value came from.
        message(STATUS
            "learning-llamas: no git metadata for vendor/llama.cpp -- using the commit recorded "
            "in _version_lock.py (${LL_LLAMA_CPP_COMMIT})")
    endif()
endif()

if(NOT LL_LLAMA_CPP_COMMIT)
    message(FATAL_ERROR
        "could not determine the vendored llama.cpp commit.\n"
        "  in a git checkout:    git submodule update --init --recursive\n"
        "  in an unpacked sdist: the recorded commit is missing, which means the sdist was\n"
        "                        built without [tool.scikit-build] sdist.cmake -- rebuild it")
endif()

message(STATUS "learning-llamas: version ${LL_VERSION}, vendored llama.cpp ${LL_LLAMA_CPP_COMMIT}")

configure_file(farm_version.h.in "${CMAKE_CURRENT_BINARY_DIR}/farm_version.h" @ONLY)

# The Python half of the version lock (S0-04).
#
# Generated into the *source* tree, because that is where an editable install resolves
# `learning_llamas._ffi` from -- a copy that existed only in the install tree would be invisible to
# every developer and every test.
#
# ...and then ALSO installed, because the source-tree copy is gitignored, and scikit-build-core
# honours gitignore when it packs `wheel.packages`. So the wheel shipped the *template* and not the
# generated file, and `import learning_llamas` raised ModuleNotFoundError on `._ffi._version_lock`
# for anyone who installed one. Every lane was green throughout: the editable installs the tests run
# under read the file straight out of src/, and nothing ever installed a wheel and imported it from
# somewhere else. `ci-windows` does now, and so does `ci-cpu` (S1-33).
#
# One configure_file, two destinations, so the two copies cannot disagree.
configure_file(
    "${CMAKE_SOURCE_DIR}/src/learning_llamas/_ffi/_version_lock.py.in"
    "${CMAKE_SOURCE_DIR}/src/learning_llamas/_ffi/_version_lock.py"
    @ONLY)

install(FILES "${CMAKE_SOURCE_DIR}/src/learning_llamas/_ffi/_version_lock.py"
        COMPONENT learningllamas
        DESTINATION "learning_llamas/_ffi")

add_library(learningllamas
    farm_api.h
    farm_probe.cpp
    farm_train.cpp
    farm_adapter.cpp
    farm_preflight.cpp
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

# Fail at LINK time on an unresolved symbol, not at dlopen.
#
# ELF lets a shared library ship with dangling undefined symbols and hope they turn up at load. So
# a shim that reaches for an internal llama.cpp never marked LLAMA_API_INTERNAL still *links* --
# and then dies inside `import learning_llamas` with a mangled name and no hint of the cause. A
# Windows linker would have refused it outright, which is the more useful behaviour, so ask for it.
#
# Together with the hidden visibility forced in the top-level CMakeLists, this is the guard: the
# Linux build now rejects exactly what the Windows build would reject, at the same step.
if(CMAKE_SYSTEM_NAME STREQUAL "Linux")
    target_link_options(learningllamas PRIVATE "LINKER:--no-undefined")
endif()
