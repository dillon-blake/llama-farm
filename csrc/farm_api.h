// farm_api.h -- the flat C ABI of liblearningllamas.
//
// Layer 1 of the architecture (BLUEPRINT §3). Everything Python needs from llama.cpp's
// private C++ internals crosses into ctypes through this header, and only through this
// header: no C++ types, no exceptions, no ownership subtleties.
//
// At S0-03 the surface is just two probe functions, which exist to prove the toolchain,
// the include paths, and the packaging. The real entry points arrive later:
//
//   S1-01  ll_opt_init_lora  -- ggml_set_param on the adapter A/B tensors
//   S1-02  ll_train_step     -- the forked per-ubatch loop with a pluggable loss
//
// This file is original learning-llamas code. Files here that copy or adapt llama.cpp code
// carry a provenance header instead; see docs/PROVENANCE.md.

#ifndef LEARNING_LLAMAS_FARM_API_H
#define LEARNING_LLAMAS_FARM_API_H

#ifdef __cplusplus
extern "C" {
#endif

#if defined(_WIN32)
#if defined(LL_BUILD)
#define LL_API __declspec(dllexport)
#else
#define LL_API __declspec(dllimport)
#endif
#else
#define LL_API __attribute__((visibility("default")))
#endif

// The learning-llamas version this library was built from -- the same string as
// learning_llamas.__version__. A mismatch means a stale native build is on the path.
//
// Returns a static, NUL-terminated string; the caller must not free it.
LL_API const char * ll_version(void);

// The full 40-character git commit hash of the vendored llama.cpp tree this library was
// compiled against, baked in at configure time.
//
// This is the anchor of the version lock (S0-04). The ctypes layer mirrors llama.cpp
// struct layouts by hand, so it is only correct against one exact commit: struct drift
// produces silent memory corruption, not a link error. _ffi compares this against the
// commit recorded when the Python package was generated and refuses to run on a mismatch.
//
// Returns a static, NUL-terminated string; the caller must not free it.
LL_API const char * ll_probe(void);

#ifdef __cplusplus
}
#endif

#endif // LEARNING_LLAMAS_FARM_API_H
