// ll_opt_init_lora -- flag the adapter's A/B tensors as the only trainable parameters.
//
// See the doc comment in farm_api.h for why this is the whole trick. This file is where the
// shim earns its existence: llama_adapter_lora's `ab_map` lives in a private header
// (src/llama-adapter.h), so nothing outside the vendored tree can reach the tensors that need
// flagging.

#include "farm_api.h"

#include "llama-adapter.h"
#include "llama-impl.h"
#include "llama-model.h"
#include "llama-context.h"

#include "ggml-backend.h"
#include "ggml-opt.h"
#include "ggml.h"
#include "llama.h"

#include <memory>
#include <unordered_map>
#include <vector>

namespace {

// The shim's per-context training state. Keyed by context because the C ABI is flat and
// llama_context has no room for us; the alternative -- handing Python an opaque handle it must
// thread through every call -- buys nothing and invites use-after-free.
struct ll_train_state {
    // The optimizer hyperparameters ggml reads on every step. This points at CALLER-owned
    // memory (Python's struct), which is what makes a learning-rate schedule a plain attribute
    // assignment rather than a callback.
    ll_opt_params * params = nullptr;

    // Mirrors `params` into the layout ggml expects. ggml_opt_get_constant_optimizer_params
    // casts its userdata straight to this type, so the layouts must agree; refreshed before
    // every step by ll_train_step (S1-02).
    ggml_opt_optimizer_params opt_pars = {};

    // Every A/B tensor we flagged, in flagging order. S1-08 saves them; S1-09 checkpoints their
    // optimizer state.
    std::vector<ggml_tensor *> param_tensors;
};

std::unordered_map<llama_context *, std::unique_ptr<ll_train_state>> & train_states() {
    static std::unordered_map<llama_context *, std::unique_ptr<ll_train_state>> states;
    return states;
}

// A param filter that says no to everything.
//
// llama_opt_init does two things we want -- it puts the context into training mode (so attention
// bypasses the KV cache, S1-00) and it creates the ggml_opt context -- and one thing we do not:
// it walks the BASE model's tensors and offers them to this filter. Rejecting all of them means
// the base model stays frozen, which is the entire point of LoRA: the base can remain quantized
// and memory-mapped, because nothing ever writes to it.
//
// We then flag the adapter tensors ourselves, which llama.cpp never offers to the filter at all.
bool reject_every_base_tensor(const ggml_tensor * tensor, void * userdata) {
    (void)tensor;
    (void)userdata;
    return false;
}

// Can the backward pass actually differentiate through this base tensor?
//
// This is not a theoretical question. ggml-cpu has "extra buffer types" -- repacked weight
// layouts (q4_K_8x8 and friends) that make MUL_MAT much faster. When a base tensor lands in one,
// ggml_backend_cpu_device_supports_op RETURNS EARLY and delegates the whole decision to that
// buffer type's handler (ggml-cpu.cpp:430-439), which implements MUL_MAT and MUL_MAT_ID -- and
// not OUT_PROD.
//
// But OUT_PROD is exactly what the backward pass needs: the gradient of MUL_MAT with respect to
// its activations is ggml_out_prod(W, grad^T) (ggml.c:6594-6630). So a repacked base tensor
// makes its own gradient node unschedulable, and ggml_backend_sched aborts with the singularly
// unhelpful:
//
//     ggml-backend.cpp:1242: GGML_ASSERT(*cur_backend_id != -1) failed
//
// which names neither the op nor the tensor. This bit Q4_K but not Q8_0, purely because a
// q4_K_8x8 repack variant exists and a q8_0 one does not -- i.e. it is silent, quant-dependent,
// and would have surfaced as "Q4_K training is broken" with no clue why.
//
// So ask the question directly, at init, while we still have something useful to say: build the
// exact OUT_PROD node the backward would build, and ask the device that owns the tensor whether
// it can run it. The cure is to load the model with use_extra_bufts=false.
bool base_tensor_supports_backward(ggml_tensor * base) {
    if (base == nullptr || base->buffer == nullptr) {
        return true; // not our business; other validation will catch it
    }

    ggml_backend_buffer_type_t buft = ggml_backend_buffer_get_type(base->buffer);
    ggml_backend_dev_t dev = ggml_backend_buft_get_device(buft);
    if (dev == nullptr) {
        return true;
    }

    // A scratch context: we build the node to ask about it, never to run it.
    ggml_init_params ip = {};
    ip.mem_size = ggml_tensor_overhead() * 4;
    ip.mem_buffer = nullptr;
    ip.no_alloc = true;

    ggml_context * ctx = ggml_init(ip);
    if (ctx == nullptr) {
        return true;
    }

    // ggml_out_prod(a, b) requires a->ne[1] == b->ne[1]. In the MUL_MAT backward, `a` is the
    // weight and `b` is the transposed gradient, so give `b` the shape that satisfies it.
    ggml_tensor * grad_t = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 1, base->ne[1]);
    ggml_tensor * node = ggml_out_prod(ctx, base, grad_t);

    const bool supported = ggml_backend_dev_supports_op(dev, node);

    ggml_free(ctx);

    return supported;
}

// Reject a tensor ggml_set_param would abort on, with an error code instead of a crash.
int32_t validate_adapter_tensor(const ggml_tensor * tensor) {
    if (tensor == nullptr) {
        return LL_ERR_INVALID_ARG;
    }
    if (tensor->type != GGML_TYPE_F32) {
        // LoRA A/B are F32 by construction (S0-05 writes them that way and the loader validates
        // it). A quantized adapter tensor cannot receive a gradient.
        return LL_ERR_TENSOR_NOT_F32;
    }
    if (tensor->op != GGML_OP_NONE) {
        // ggml_set_param asserts this (ggml.c:7678). Adapter tensors are ggml_dup_tensor copies
        // made at load time, so they are leaves -- but check rather than trust, because the
        // failure mode is an abort deep inside graph construction.
        return LL_ERR_TENSOR_NOT_LEAF;
    }
    return LL_OK;
}

} // namespace

int32_t ll_opt_init_lora(llama_context * ctx, llama_model * model, llama_adapter_lora ** adapters, size_t n_adapters,
                         ll_opt_params * params) {
    if (ctx == nullptr || model == nullptr || adapters == nullptr || params == nullptr) {
        return LL_ERR_INVALID_ARG;
    }
    if (n_adapters == 0) {
        return LL_ERR_NO_ADAPTERS;
    }
    if (train_states().count(ctx) != 0) {
        return LL_ERR_ALREADY_INIT;
    }

    // Validate everything BEFORE mutating anything. A half-flagged adapter is worse than a
    // rejected one: it would train a subset of the tensors and look like it worked.
    size_t n_tensors = 0;
    for (size_t i = 0; i < n_adapters; ++i) {
        if (adapters[i] == nullptr) {
            return LL_ERR_INVALID_ARG;
        }
        for (const auto & [name, weight] : adapters[i]->ab_map) {
            const int32_t err_a = validate_adapter_tensor(weight.a);
            if (err_a != LL_OK) {
                return err_a;
            }
            const int32_t err_b = validate_adapter_tensor(weight.b);
            if (err_b != LL_OK) {
                return err_b;
            }

            // The base tensor this adapter hangs off must be differentiable, or the backward
            // graph will not schedule. See base_tensor_supports_backward.
            ggml_tensor * base = const_cast<ggml_tensor *>(model->get_tensor(name.c_str()));
            if (!base_tensor_supports_backward(base)) {
                LLAMA_LOG_ERROR(
                    "%s: base tensor '%s' is in buffer type '%s', which cannot run OUT_PROD -- "
                    "so its gradient node would be unschedulable. Load the model with "
                    "use_extra_bufts=false to train it.\n",
                    __func__, name.c_str(),
                    base && base->buffer ? ggml_backend_buft_name(ggml_backend_buffer_get_type(base->buffer)) : "?");
                return LL_ERR_BASE_BUFT_NO_BACKWARD;
            }

            n_tensors += 2;
        }
    }
    if (n_tensors == 0) {
        return LL_ERR_NO_ADAPTERS;
    }

    auto state = std::make_unique<ll_train_state>();
    state->params = params;

    // Put the context into training mode and create its ggml_opt context. The reject-all filter
    // is what keeps the base model frozen: llama_opt_init would otherwise flag every F32 base
    // tensor and we would be doing a full fine-tune with extra steps.
    llama_opt_params lopt = {};
    lopt.n_ctx_train = 0; // use the context's n_ctx
    lopt.param_filter = reject_every_base_tensor;
    lopt.param_filter_ud = nullptr;
    lopt.get_opt_pars = ggml_opt_get_constant_optimizer_params;
    lopt.get_opt_pars_ud = &state->opt_pars;
    lopt.optimizer_type = GGML_OPT_OPTIMIZER_TYPE_ADAMW;

    state->opt_pars.adamw.alpha = params->alpha;
    state->opt_pars.adamw.beta1 = params->beta1;
    state->opt_pars.adamw.beta2 = params->beta2;
    state->opt_pars.adamw.eps = params->eps;
    state->opt_pars.adamw.wd = params->wd;
    state->opt_pars.sgd.alpha = params->alpha;
    state->opt_pars.sgd.wd = params->wd;

    llama_opt_init(ctx, model, lopt);

    // THE trick: flag the adapter's A/B tensors. Everything else -- gradients, the AdamW
    // momenta, the backward graph -- falls out of ggml_build_backward_expand with no
    // architecture-specific code at all, because build_lora_mm already put these tensors in the
    // forward graph.
    for (size_t i = 0; i < n_adapters; ++i) {
        for (const auto & [name, weight] : adapters[i]->ab_map) {
            ggml_set_param(weight.a);
            ggml_set_param(weight.b);
            state->param_tensors.push_back(weight.a);
            state->param_tensors.push_back(weight.b);
        }
    }

    train_states()[ctx] = std::move(state);

    return (int32_t)n_tensors;
}

int32_t ll_opt_free(llama_context * ctx) {
    if (ctx == nullptr) {
        return LL_ERR_INVALID_ARG;
    }
    train_states().erase(ctx);
    return LL_OK;
}

int32_t ll_opt_n_params(llama_context * ctx) {
    if (ctx == nullptr) {
        return LL_ERR_INVALID_ARG;
    }
    const auto it = train_states().find(ctx);
    if (it == train_states().end()) {
        return LL_ERR_NOT_INITIALIZED;
    }
    return (int32_t)it->second->param_tensors.size();
}
