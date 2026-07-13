// Private-internals access check.
//
// The whole reason liblearningllamas exists is that the per-ubatch training loop --
// llama_context::opt_epoch_iter, declared at vendor/llama.cpp/src/llama-context.h:207 --
// depends on private C++ types (llm_graph_params, llm_graph_result, llama_batch_allocr,
// llama_memory_i) that are not in the public headers, and the public llama_opt_epoch
// hardcodes the wrong loss with no masking support. S1-01 and S1-02 fork that loop.
//
// This translation unit compiles nothing useful. It exists so that "the shim can reach
// llama.cpp's private src/ headers" is a *build-time* fact from stage 0 onward, rather
// than a discovery made in the middle of S1-02. If an upstream rebase moves these types
// behind a wall, this file stops compiling and the ci-cpu lane says so immediately.
//
// It exports no symbols: the assertions are all static.

#include "llama-context.h"
#include "llama-graph.h"

#include <cstddef>

namespace {

// Private types, reachable only from vendor/llama.cpp/src.
static_assert(sizeof(llama_context) > 0, "llama_context must be a complete type here");
static_assert(sizeof(llm_graph_params) > 0, "llm_graph_params must be a complete type here");
static_assert(sizeof(llm_graph_result) > 0, "llm_graph_result must be a complete type here");

// The optimizer-params struct crosses the ABI by value, which is the one ctypes hazard the
// binding layer is designed around (BLUEPRINT §3): _ffi never registers a Python callback
// returning it, and passes ggml_opt_get_constant_optimizer_params with a Python-owned
// struct as userdata instead. Pin its size here so a vendor bump that changes the layout
// breaks the build rather than the training run.
static_assert(sizeof(ggml_opt_optimizer_params) == 7 * sizeof(float),
              "ggml_opt_optimizer_params layout changed -- regenerate the _ffi mirrors");

// ggml_opt_params likewise crosses by value, and it is the more dangerous of the two: it holds
// FUNCTION POINTERS, so a mirror that disagrees about the layout does not read a wrong number --
// it calls a wrong address.
//
// S1-10 inserted `grad_clip` into the middle of it (between opt_period and get_opt_pars), and for
// a while the ctypes mirror did not declare it AT ALL. That was harmless purely by accident: a
// float at offset 44 lands in the padding the mirror already had between an int32 at 40 and an
// 8-aligned pointer at 48, so every later field still aligned and sizeof still came to 72. The
// next field added anywhere before get_opt_pars would have silently shifted the callback pointer.
//
// These numbers are mirrored in src/learning_llamas/_ffi/ggml_opt.py, and pinned again there by
// test_the_opt_params_mirror_matches_the_C_layout. Change one, and the other breaks -- which is
// the point.
static_assert(sizeof(ggml_opt_params) == 72, "ggml_opt_params size changed -- update the _ffi mirror");
static_assert(offsetof(ggml_opt_params, opt_period) == 40, "ggml_opt_params layout changed");
static_assert(offsetof(ggml_opt_params, grad_clip) == 44, "ggml_opt_params layout changed");
static_assert(offsetof(ggml_opt_params, get_opt_pars) == 48, "ggml_opt_params layout changed");
static_assert(offsetof(ggml_opt_params, optimizer) == 64, "ggml_opt_params layout changed");

} // namespace
