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

#include <stddef.h>
#include <stdint.h>

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

// ---------------------------------------------------------------------------
// Training (S1-01)
// ---------------------------------------------------------------------------

struct llama_context;
struct llama_model;
struct llama_adapter_lora;

// Negative return codes. Every entry point below returns a non-negative value on success.
#define LL_OK 0
#define LL_ERR_INVALID_ARG -1     // a null or nonsensical argument
#define LL_ERR_NO_ADAPTERS -2     // n_adapters == 0, or an adapter has no A/B tensors
#define LL_ERR_ALREADY_INIT -3    // ll_opt_init_lora called twice on one context
#define LL_ERR_TENSOR_NOT_F32 -4  // an adapter tensor is not F32; LoRA A/B must be
#define LL_ERR_TENSOR_NOT_LEAF -5 // an adapter tensor is not a leaf (op != GGML_OP_NONE)
#define LL_ERR_NOT_INITIALIZED -6 // no training state for this context

// A base tensor's buffer type cannot run OUT_PROD, so its gradient node would be
// unschedulable. Load the model with use_extra_bufts=false.
#define LL_ERR_BASE_BUFT_NO_BACKWARD -7

#define LL_ERR_STEP_FAILED -8 // the training step itself failed

// n_tokens changed between steps. ggml-opt indexes its optimizer state by graph node index and
// sizes it from the first graph it sees, so every step must have the same shape. Pad instead.
#define LL_ERR_SHAPE_MISMATCH -9

// AdamW hyperparameters, owned by the caller and read afresh on every optimizer step.
//
// Python keeps this struct alive and mutates it between steps, which is how a learning-rate
// schedule works with no callback: the shim hands ggml the exported
// ggml_opt_get_constant_optimizer_params with a pointer to this struct as its userdata.
struct ll_opt_params {
    float alpha; // learning rate
    float beta1;
    float beta2;
    float eps;
    float wd; // weight decay; 0 to disable
};

// Mark the LoRA adapter's A/B tensors as the ONLY trainable parameters, and put the context
// into training mode.
//
// This is the whole trick (BLUEPRINT D2). ggml's autograd gives gradients and optimizer state
// only to leaf tensors flagged with ggml_set_param. `build_lora_mm` already injects the adapter
// A/B matmuls into every architecture's forward graph, so flagging those tensors is *all* that
// is missing -- ggml_build_backward_expand does the rest, with zero per-architecture code.
//
// Nobody had wired this: llama.cpp's own llama_set_param helper iterates base-model tensors only
// and never offers adapter tensors to the filter, so today they ride through training graphs as
// inert constants.
//
// Two orderings are enforced, not merely documented:
//   - adapters must already be attached to `ctx` (llama_set_adapters_lora), because this call
//     takes the same handles;
//   - this must run BEFORE the first training step, because PARAM-flagged leaves are promoted
//     into graph nodes at build time and ggml-opt's gradient/momentum allocation scans only
//     graph nodes. Flagging afterwards would silently train nothing.
//
// Because only the adapter trains, the base model is never written and may stay quantized AND
// memory-mapped (use_mmap=true) -- unlike full fine-tuning, which must disable mmap because the
// optimizer writes weights back in place through what is a read-only mapping.
//
// Perf note for stages 2-4: an adapter tensor whose base tensor uses an extra/repacked buffer
// type falls back to a CPU buffer at load (llama-adapter.cpp:337-350). Once GPU backends exist,
// that costs cross-backend gradient traffic. Harmless on the CPU-only stage-1 build.
//
// Args:
//   ctx:        the context the adapters are attached to.
//   model:      the model the adapters were loaded against.
//   adapters:   the same handles passed to llama_set_adapters_lora.
//   n_adapters: how many.
//   opt_period: how many ll_train_step calls make one optimizer step. 1 = step every call.
//               Anything greater accumulates gradients across that many calls and steps on the
//               last -- which is what gradient accumulation IS. Must be >= 1.
//   grad_clip:  clip the gradients to this GLOBAL norm before every optimizer step. 0 disables it.
//               Global, i.e. one norm over every trainable tensor jointly -- which bounds the
//               length of the update without rotating it.
//
//               Applied as nodes in the graph, because it cannot be applied anywhere else: the
//               optimizer step is fused into the backward graph, so by the time this shim regains
//               control the weights have already moved.
//   params:     AdamW hyperparameters.
//
//               THE SHIM STORES THIS POINTER AND RE-READS THE STRUCT ON EVERY STEP. That is the
//               feature -- a learning-rate schedule is `params.alpha = ...` between steps, with no
//               callback back into the caller on the hot path -- and it is also the sharpest edge
//               in this header, so it is spelled out:
//
//               The caller MUST keep the struct alive until ll_opt_free. If it does not, every
//               subsequent step reads its hyperparameters out of freed memory, and what happens
//               then depends on what the allocator put there: NaN weights, a learning rate of
//               zero, an abort on ggml's `GGML_ASSERT(alpha > 0)`, or a segfault -- differing from
//               run to run. There is no way for the shim to detect it.
//
//               From Python, do not call this directly: `_ffi.opt_init_lora` holds the reference
//               for you (src/learning_llamas/_ffi/farm.py). A bare `h, _ = init(...)` followed by
//               `for _ in range(n)` is enough to free the struct, and it has already happened once.
//
// Returns the number of tensors flagged (2 x the number of adapted base tensors), or a negative
// LL_ERR_* code.
LL_API int32_t ll_opt_init_lora(struct llama_context * ctx, struct llama_model * model,
                                struct llama_adapter_lora ** adapters, size_t n_adapters,
                                struct ll_opt_params * params, int32_t opt_period, float grad_clip);

// Release the shim's training state for `ctx`. Idempotent. The context itself is not freed.
LL_API int32_t ll_opt_free(struct llama_context * ctx);

// The number of tensors ll_opt_init_lora flagged, or LL_ERR_NOT_INITIALIZED.
LL_API int32_t ll_opt_n_params(struct llama_context * ctx);

// ---------------------------------------------------------------------------
// Training step (S1-02)
// ---------------------------------------------------------------------------

// Run one training step over `tokens`, with a per-token weighted cross-entropy loss.
//
// The stock training path (llama_opt_epoch) hardcodes an unmasked cross-entropy over dense
// one-hot labels. That cannot train an instruction-tuned model, where the prompt tokens must
// contribute NOTHING to the loss -- which is the single most common thing anyone wants to do,
// and the reason BLUEPRINT D1 forks the loop.
//
// Here the loss is
//
//     L = -sum_i w_i * log_softmax(logits_i)[target_i] / sum_i w_i
//
// so a token with w_i = 0 contributes exactly zero, and the result is the mean over the tokens
// that actually count -- not over all of them. Setting every w_i = 1 recovers the standard
// next-token loss.
//
// Args:
//   ctx:      a context that has been through ll_opt_init_lora.
//   tokens:   the input token ids, length n_tokens.
//   targets:  the target token id for each position, length n_tokens. Usually tokens shifted
//             left by one.
//   weights:  the per-token loss weight, length n_tokens. 0 masks a token out.
//   n_tokens: how many. Must not exceed the context's n_batch.
//   train:    true to backpropagate and step the optimizer; false for a forward-only eval.
//   loss_out: receives the loss. May be NULL.
//
// seq_ids and positions are what make PACKING work (S1-07). Several independent samples share one
// batch; llama.cpp derives the attention mask from (seq_id, pos), and the training path masks
// across sequences exactly as the cached path does (`if (s0 != s1) continue`, llama-graph.cpp), so
// giving each packed sample its own seq_id and restarting its positions at 0 makes them invisible
// to one another. Pass NULL for both to get the unpacked default: one sequence, positions 0..n-1.
//
// Args:
//   tokens:    [n_tokens] the input token at each position.
//   targets:   [n_tokens] the token that position is asked to predict.
//   weights:   [n_tokens] how much that prediction counts. 0 masks it out entirely.
//   seq_ids:   [n_tokens] which sequence each position belongs to, or NULL for "all one".
//   positions: [n_tokens] each position's index WITHIN its sequence, or NULL for 0..n-1.
//   n_tokens:  the batch size. Must be the same on every step -- see LL_ERR_SHAPE_MISMATCH.
//   train:     false runs the forward pass only: no gradient, no optimizer state, nothing written.
//   loss_out:  the loss, per valid token.
//
// Returns LL_OK, or a negative LL_ERR_* code.
LL_API int32_t ll_train_step(struct llama_context * ctx, const int32_t * tokens, const int32_t * targets,
                             const float * weights, const int32_t * seq_ids, const int32_t * positions,
                             int32_t n_tokens, bool train, float * loss_out);

// ---------------------------------------------------------------------------
// Adapter enumeration (S1-08) -- read the trained tensors back out
// ---------------------------------------------------------------------------
//
// Training writes the adapter's A/B tensors in place. To SAVE the result something has to read them
// back, and llama.cpp offers no way to: `ab_map` lives in a private header, so the tensors are
// unreachable from outside the vendored tree.
//
// Tensors are addressed by INDEX, and the index is the position in the adapter's base-tensor names
// sorted lexicographically -- not ab_map's iteration order, which is an unordered_map's and is not
// promised to be stable between builds.
//
// Unlike the ll_debug_* accessors below, these need no training context: an adapter handle is
// enough, so saving does not require the model to still be in training mode.

// How many base tensors the adapter adapts (i.e. how many A/B pairs), or LL_ERR_INVALID_ARG.
LL_API int32_t ll_adapter_n_tensors(struct llama_adapter_lora * adapter);

// The index-th pair: its base tensor's name, and the ggml `ne` of its A and B.
//
// Args:
//   name_out:      receives the base tensor name, NUL-terminated.
//   name_capacity: its size. LL_ERR_INVALID_ARG if the name does not fit.
//   ne_a, ne_b:    receive GGML_MAX_DIMS (4) elements each.
LL_API int32_t ll_adapter_tensor_info(struct llama_adapter_lora * adapter, int32_t index,
                                      char * name_out, int32_t name_capacity,
                                      int64_t * ne_a, int64_t * ne_b);

// Copy the index-th pair's A (is_b=false) or B (is_b=true) out. Returns the element count, or
// a negative LL_ERR_* code.
//
// Pass out=NULL with n_max=0 to ask only how many elements there are, then call again with a buffer
// -- the same two-call convention as llama_tokenize.
LL_API int64_t ll_adapter_get(struct llama_adapter_lora * adapter, int32_t index, bool is_b,
                              float * out, int64_t n_max);

// ---------------------------------------------------------------------------
// Debug accessors (S1-03)
// ---------------------------------------------------------------------------
//
// Deliberately named ll_debug_*, and deliberately outside any API-stability promise. They exist
// so a test can reach into a live adapter tensor and its gradient, which is the only way to run
// a finite-difference check on the WHOLE graph rather than on one op. S1-08 supersedes them with
// a real enumeration/save API.
//
// Tensors are addressed by their base tensor name (the ab_map key, e.g. "blk.0.attn_q.weight")
// plus which of the pair is wanted.

// Number of elements in an adapter tensor, or a negative LL_ERR_* code.
LL_API int64_t ll_debug_n_elements(struct llama_context * ctx, const char * base_name, bool is_b);

// Copy an adapter tensor's values out. Returns the number of elements written, or LL_ERR_*.
LL_API int64_t ll_debug_get_tensor(struct llama_context * ctx, const char * base_name, bool is_b,
                                   float * out, int64_t n_max);

// Overwrite an adapter tensor's values. Returns the number of elements written, or LL_ERR_*.
LL_API int64_t ll_debug_set_tensor(struct llama_context * ctx, const char * base_name, bool is_b,
                                   const float * data, int64_t n);

// Copy an adapter tensor's GRADIENT out, from ggml-opt's accumulator.
//
// Only valid after a training step has built the backward graph; before that there is no
// accumulator and this returns LL_ERR_NOT_INITIALIZED.
LL_API int64_t ll_debug_grad(struct llama_context * ctx, const char * base_name, bool is_b,
                             float * out, int64_t n_max);

#ifdef __cplusplus
}
#endif

#endif // LEARNING_LLAMAS_FARM_API_H
