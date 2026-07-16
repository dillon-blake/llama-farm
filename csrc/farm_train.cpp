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

#include <algorithm>
#include <cinttypes>
#include <cstring>
#include <cmath>
#include <memory>
#include <unordered_map>
#include <cstddef>
#include <string>
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

    // Layers per gradient-checkpointing segment; 0 = off (S1-17). Mirrored here only so that
    // ll_set_grad_checkpointing can refuse a mid-run change.
    int32_t grad_ckpt_segment = 0;

    // Tokens per attention query chunk; 0 = off (S1-24). Same reason, and a harder one: chunking
    // changes the graph's node COUNT, and ggml-opt indexes optimizer state by node index.
    int32_t attn_chunk_q = 0;

    // The gradient accumulator of each param tensor, keyed by the tensor's name.
    //
    // Captured once, from inside the first training step, because that is the only window in
    // which ggml_opt will tell us: with dynamic graphs, ggml_opt_eval nulls the graphs it looked
    // them up in. The tensors themselves persist in opt_ctx's static context.
    std::unordered_map<std::string, ggml_tensor *> grad_accs;

    // The AdamW moments of each param tensor, keyed the same way and captured in the same window
    // and for the same reason. These are what a checkpoint has to save: resuming from the weights
    // alone restarts the optimizer from a standing start, and the loss curve jumps.
    std::unordered_map<std::string, ggml_tensor *> grad_m;
    std::unordered_map<std::string, ggml_tensor *> grad_v;

    // The n_tokens of the first step. Every later step must match it -- see ll_train_step.
    int32_t n_tokens_seen = 0;

    // And the OBJECTIVE of the first step, for exactly the same reason and with a worse failure.
    //
    // Different objectives build different numbers of nodes -- the DPO tail adds a scale, an add and
    // a softplus that the SFT loss does not have. ggml-opt sizes grad_accs from the FIRST graph's
    // node count and indexes it by node index thereafter, so switching objectives on one context
    // makes the backward pass index past the end of that vector. It does not assert; it reads
    // whatever is next in memory and calls it a gradient tensor. (Found as a segfault.)
    int32_t loss_kind_seen = -1;

    // The backend scheduler ggml_opt_init was handed. It captures the POINTER and holds it forever;
    // llama_context is free to replace it. See LL_ERR_SCHED_INVALIDATED.
    ggml_backend_sched_t sched = nullptr;

    // OUR ggml_opt context, not llama_context's.
    //
    // llama_opt_init builds one with GGML_OPT_LOSS_TYPE_CROSS_ENTROPY hardcoded, and keeps it
    // private. We need GGML_OPT_LOSS_TYPE_SUM instead -- because with SUM, ggml-opt simply sums
    // whatever node it is handed as `outputs`, and summing a scalar is the identity. So our own
    // loss node *becomes* the loss, and the loss is pluggable with no ggml change at all.
    ggml_opt_context_t opt_ctx = nullptr;

    ~ll_train_state() {
        if (opt_ctx) {
            ggml_opt_free(opt_ctx);
        }
    }
};

std::unordered_map<llama_context *, std::unique_ptr<ll_train_state>> & train_states() {
    static std::unordered_map<llama_context *, std::unique_ptr<ll_train_state>> states;
    return states;
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

// Flag one BASE model weight as trainable, if it can be (S1-48, full fine-tuning).
//
// Skips, silently and by design: a null slot (most llama_layer fields are unused by any one arch),
// a non-F32 tensor (a quantized base weight cannot receive a gradient), a non-leaf, the precomputed
// rope_freqs table (a constant, not a learnable), a tensor whose buffer cannot run its backward
// (repacked -- avoided entirely by full_finetune's use_extra_bufts=false), and a tensor already
// flagged (tied embeddings alias output == tok_embd, and ggml_set_param must not run twice).
void try_flag_base_tensor(ggml_tensor * t, ll_train_state * state) {
    if (t == nullptr || t->type != GGML_TYPE_F32 || t->op != GGML_OP_NONE) {
        return;
    }
    if (std::strcmp(ggml_get_name(t), "rope_freqs.weight") == 0) {
        return;
    }
    if (!base_tensor_supports_backward(t)) {
        return;
    }
    for (ggml_tensor * seen : state->param_tensors) {
        if (seen == t) {
            return;
        }
    }
    ggml_set_param(t);
    state->param_tensors.push_back(t);
}

} // namespace

int32_t ll_opt_init_lora(llama_context * ctx, llama_model * model, llama_adapter_lora ** adapters, size_t n_adapters,
                         ll_opt_params * params, int32_t opt_period, float grad_clip) {
    if (ctx == nullptr || model == nullptr || adapters == nullptr || params == nullptr) {
        return LL_ERR_INVALID_ARG;
    }
    if (opt_period < 1 || grad_clip < 0.0f) {
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

    state->opt_pars.adamw.alpha = params->alpha;
    state->opt_pars.adamw.beta1 = params->beta1;
    state->opt_pars.adamw.beta2 = params->beta2;
    state->opt_pars.adamw.eps = params->eps;
    state->opt_pars.adamw.wd = params->wd;
    state->opt_pars.sgd.alpha = params->alpha;
    state->opt_pars.sgd.wd = params->wd;

    // Training mode: attention must bypass the KV cache or the backward graph cannot be built at
    // all (S1-00). This also disables flash attention, which has no backward rule.
    //
    // We deliberately do NOT call llama_opt_init. It would build a ggml_opt context with the loss
    // type hardcoded to cross-entropy and keep it private, and it would walk the base model's
    // tensors offering them to a param filter. We want neither -- see ll_train_state::opt_ctx.
    ctx->set_training(true);

    ggml_opt_params opt_params = ggml_opt_default_params(ctx->get_sched(), GGML_OPT_LOSS_TYPE_SUM);

    // opt_period comes from the CALLER, and means "how many ll_train_step calls per optimizer
    // step". It is not inferred from the context's batch geometry.
    //
    // llama_opt_init infers it, as n_batch / n_ubatch, and that is wrong in a way that hides.
    // ggml_opt steps only on every opt_period-th ggml_opt_eval, and opt_step_custom evals once per
    // ubatch. With n_batch=64 and n_ubatch=32 that gives opt_period=2 -- so a caller passing 32
    // tokens gets ONE ubatch, opt_i never wraps, and the optimizer steps on every OTHER call to
    // ll_train_step. The loss still falls, at half the rate, and nothing says why. (It also leaves
    // opt_ctx->gb_opt NULL on the steps that do not wrap, which is how it was found: as a segfault
    // inside ggml_opt_grad_acc.)
    //
    // So gradient accumulation is the trainer's decision, stated explicitly, and one ll_train_step
    // is always exactly one micro-batch.
    opt_params.opt_period = opt_period;

    // Global-norm gradient clipping, applied inside the graph (see ggml_opt_build). 0 disables it
    // and inserts no nodes, so an unclipped run's graph is exactly what it always was.
    //
    // It cannot be done from here, and that is worth knowing rather than rediscovering: the AdamW
    // step is fused into the backward graph, so by the time ll_train_step regains control the
    // weights have already moved. A host-side clip could only ever see the gradients of the
    // micro-steps BEFORE the one that triggers the step.
    opt_params.grad_clip = grad_clip;

    opt_params.get_opt_pars = ggml_opt_get_constant_optimizer_params;
    opt_params.get_opt_pars_ud = &state->opt_pars;
    opt_params.optimizer = GGML_OPT_OPTIMIZER_TYPE_ADAMW;

    state->sched = ctx->get_sched();
    state->opt_ctx = ggml_opt_init(opt_params);

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

int32_t ll_opt_init_full(llama_context * ctx, llama_model * model, ll_opt_params * params, int32_t opt_period,
                         float grad_clip) {
    if (ctx == nullptr || model == nullptr || params == nullptr) {
        return LL_ERR_INVALID_ARG;
    }
    if (opt_period < 1 || grad_clip < 0.0f) {
        return LL_ERR_INVALID_ARG;
    }
    if (train_states().count(ctx) != 0) {
        return LL_ERR_ALREADY_INIT;
    }

    auto state = std::make_unique<ll_train_state>();
    state->params = params;

    state->opt_pars.adamw.alpha = params->alpha;
    state->opt_pars.adamw.beta1 = params->beta1;
    state->opt_pars.adamw.beta2 = params->beta2;
    state->opt_pars.adamw.eps = params->eps;
    state->opt_pars.adamw.wd = params->wd;
    state->opt_pars.sgd.alpha = params->alpha;
    state->opt_pars.sgd.wd = params->wd;

    // Everything below mirrors ll_opt_init_lora exactly -- the training flag, the SUM-loss opt
    // context, the caller-owned hyperparameters, the graph-fused clip -- because full fine-tuning
    // differs from LoRA in one place only: WHICH leaves are flagged. See ll_train_state::opt_ctx
    // for why the loss type is SUM rather than llama_opt_init's hardcoded cross-entropy.
    ctx->set_training(true);

    ggml_opt_params opt_params = ggml_opt_default_params(ctx->get_sched(), GGML_OPT_LOSS_TYPE_SUM);
    opt_params.opt_period = opt_period;
    opt_params.grad_clip = grad_clip;
    opt_params.get_opt_pars = ggml_opt_get_constant_optimizer_params;
    opt_params.get_opt_pars_ud = &state->opt_pars;
    opt_params.optimizer = GGML_OPT_OPTIMIZER_TYPE_ADAMW;

    state->sched = ctx->get_sched();
    state->opt_ctx = ggml_opt_init(opt_params);

    // Flag the base weights. This traversal mirrors llama_context::opt_init's (llama-context.cpp),
    // with two deliberate differences: it flags the token embedding (whose gradient is GET_ROWS's
    // VJP, which this fork implements), and it builds our own SUM opt context so the masked-CE
    // ll_train_step works unchanged. The whole-struct reinterpret over each layer is llama.cpp's
    // own idiom (llama_layer is a flat run of ggml_tensor * fields), so any arch's per-block
    // weights are covered without an arch-specific list.
    try_flag_base_tensor(model->tok_embd, state.get());
    try_flag_base_tensor(model->output_norm, state.get());
    try_flag_base_tensor(model->output, state.get());
    for (llama_layer & layer : model->layers) {
        const size_t n_fields = sizeof(layer) / sizeof(ggml_tensor *);
        for (size_t i = 0; i < n_fields; ++i) {
            try_flag_base_tensor(reinterpret_cast<ggml_tensor **>(&layer)[i], state.get());
        }
    }

    if (state->param_tensors.empty()) {
        // Nothing trainable -- an all-quantized base, or a model with no F32 leaves. Refuse rather
        // than stand up an optimizer that would step on nothing.
        return LL_ERR_INVALID_ARG;
    }

    const int32_t n_tensors = (int32_t)state->param_tensors.size();
    train_states()[ctx] = std::move(state);

    return n_tensors;
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

int64_t ll_compute_buffer_bytes(llama_context * ctx) {
    if (ctx == nullptr) {
        return LL_ERR_INVALID_ARG;
    }

    ggml_backend_sched_t sched = ctx->get_sched();
    if (sched == nullptr) {
        return LL_ERR_NOT_INITIALIZED;
    }

    size_t total = 0;
    for (int i = 0; i < ggml_backend_sched_get_n_backends(sched); ++i) {
        total += ggml_backend_sched_get_buffer_size(sched, ggml_backend_sched_get_backend(sched, i));
    }

    return (int64_t)total;
}

int32_t ll_set_grad_checkpointing(llama_context * ctx, int32_t segment_len) {
    if (ctx == nullptr || segment_len < 0) {
        return LL_ERR_INVALID_ARG;
    }

    const auto it = train_states().find(ctx);
    if (it == train_states().end()) {
        return LL_ERR_NOT_INITIALIZED;
    }

    // ggml-opt sizes its gradient accumulators and AdamW momenta from the FIRST graph it is shown
    // and indexes them by node index forever after (BLUEPRINT D1). Checkpointing does not disturb
    // that -- it changes only the backward graph, and the forward prefix is preserved for exactly
    // this reason -- but it does change the step's memory and speed profile completely, and a run
    // whose steps are not comparable is not a run. So: before the first step, or not at all.
    if (it->second->n_tokens_seen > 0) {
        return LL_ERR_ALREADY_INIT;
    }

    ctx->set_grad_checkpointing((uint32_t)segment_len);
    it->second->grad_ckpt_segment = segment_len;

    return LL_OK;
}

int32_t ll_set_chunked_attention(llama_context * ctx, int32_t chunk_q) {
    if (ctx == nullptr || chunk_q < 0) {
        return LL_ERR_INVALID_ARG;
    }

    const auto it = train_states().find(ctx);
    if (it == train_states().end()) {
        return LL_ERR_NOT_INITIALIZED;
    }

    // Same reason as ll_set_grad_checkpointing, and a harder one. Chunking changes the graph's
    // TOPOLOGY -- it emits n_chunks softmaxes and n_chunks-1 concats per layer where the naive
    // path emits one softmax -- and ggml-opt keys its gradient accumulators and AdamW momenta by
    // NODE INDEX, from the first graph it is shown (BLUEPRINT D1). Change the chunk factor
    // mid-run and one parameter's momentum lands on another. It would not crash.
    if (it->second->n_tokens_seen > 0) {
        return LL_ERR_ALREADY_INIT;
    }

    ctx->set_attn_chunk_q((uint32_t)chunk_q);
    it->second->attn_chunk_q = chunk_q;

    return LL_OK;
}

// ---------------------------------------------------------------------------
// ll_train_step (S1-02) -- one training step with a MASKED cross-entropy loss.
// ---------------------------------------------------------------------------

namespace {

// Cache the gradient accumulator of every parameter, once, from INSIDE a training step.
//
// This has to happen here and nowhere else, and the reason is worth stating plainly because the
// alternative looks like it should work and segfaults instead.
//
// ggml_opt_grad_acc(opt_ctx, t) looks the accumulator up in opt_ctx->gb_opt -- the backward graph.
// With dynamic graphs (which is what a llama.cpp context uses: the graph is rebuilt every step,
// because the ubatch shape can change) ggml_opt_eval sets gf, gb_grad and gb_opt back to NULL when
// it returns, because the caller is about to free the compute context those graphs were built in.
// So ggml_opt_grad_acc is only meaningful in the window BETWEEN ggml_opt_alloc and ggml_opt_eval --
// and the set-loss-inputs hook is the one callback that runs inside exactly that window.
//
// The accumulator TENSORS, by contrast, live in opt_ctx's own static context. They persist across
// steps and hold the gradient from the last backward pass. So caching their addresses once is
// enough; only the graph that indexes them is transient.
void capture_grad_accs(ll_train_state * state) {
    if (state == nullptr || !state->grad_accs.empty()) {
        return;
    }

    for (ggml_tensor * t : state->param_tensors) {
        const std::string name = ggml_get_name(t);

        ggml_tensor * grad = ggml_opt_grad_acc(state->opt_ctx, t);
        if (grad != nullptr) {
            state->grad_accs[name] = grad;
        }

        // The AdamW moments, from the same window and for the same reason (S1-09).
        ggml_tensor * m = ggml_opt_grad_m(state->opt_ctx, t);
        ggml_tensor * v = ggml_opt_grad_v(state->opt_ctx, t);
        if (m != nullptr && v != nullptr) {
            state->grad_m[name] = m;
            state->grad_v[name] = v;
        }
    }
}

// Everything the loss builder needs, threaded through llama_context::opt_step_custom as userdata.
// Which objective the loss builder should build.
enum ll_loss_kind {
    LL_LOSS_SFT = 0,  // masked mean cross-entropy, normalized by the weight actually carried
    LL_LOSS_DPO = 1,  // -log sigmoid(beta * (logratio - ref_logratio))
    LL_LOSS_LOGP = 2, // sum_i w_i * logp_i, unnormalized -- for precomputing reference logratios
    LL_LOSS_GRPO = 3, // clipped importance-ratio surrogate + k3 KL, weighted by per-token advantage
};

struct loss_ctx {
    ll_train_state * state = nullptr; // so the alloc-time hook can capture the gradient accumulators
    bool train = false;

    ll_loss_kind kind = LL_LOSS_SFT;

    // DPO only. beta scales the implicit reward; ref_delta is the REFERENCE model's log-ratio
    // logp(chosen) - logp(rejected), precomputed with the adapter off and fed in as a constant.
    float beta = 0.1f;
    float ref_delta = 0.0f;

    // GRPO only. Caller-owned arrays, [n_tokens], indexed exactly like `weights`.
    const ll_grpo_inputs * grpo = nullptr;
    float clip_eps = 0.2f;

    const int32_t * targets = nullptr; // [n_tokens] target id per position
    const float * weights = nullptr;   // [n_tokens] loss weight per position; 0 masks out
    int32_t n_tokens = 0;

    // Set by build_masked_ce, filled by upload_masked_ce once ggml_opt_alloc has given it memory.
    ggml_tensor * labels = nullptr;     // I32 [n_ubatch] -- the target token of each position
    ggml_tensor * ce_weights = nullptr; // F32 [n_ubatch] -- its loss weight, pre-normalized
    ggml_tensor * dpo_ref = nullptr;    // F32 [1] -- beta * ref_delta, a constant (DPO only)

    // GRPO only. All F32 [n_ubatch], all constants: none of them carries a gradient.
    ggml_tensor * grpo_adv = nullptr;      // advantage_i * mask_i * norm
    ggml_tensor * grpo_logp_old = nullptr; // the behaviour policy's logp of target_i
    ggml_tensor * grpo_logp_ref = nullptr; // the reference policy's logp of target_i
    ggml_tensor * grpo_kl_w = nullptr;     // kl_coef * mask_i * norm

    int32_t pos = 0; // this ubatch's offset within the batch
    int32_t n_ubatch = 0;
    int64_t n_vocab = 0;
};

// Build  L = sum_i w_i * (logsumexp_j(x_ij) - x_i[target_i]) / sum_i w_i  -- the masked mean CE.
//
// This is S1-04's ggml_cross_entropy_loss_sparse: it takes ONE integer target and ONE weight per
// token and returns the per-token loss unreduced, so the caller reduces it however it likes. The
// reduction here is ggml_sum, and the 1/sum(w) normalization is folded into the weights on the
// host (see upload_masked_ce) so that the graph itself stays a plain weighted sum.
//
// It replaces a stopgap that reused the DENSE ggml_cross_entropy_loss with a "weighted one-hot"
// label matrix -- w_i at row i's target column. That trick has a correct forward and a WRONG
// BACKWARD, which is exactly the kind of bug this ticket exists to catch:
//
//   L = -(1/nr) * sum_ij y_ij * log_softmax(x)_ij
//   dL/dx_ik    =  (1/nr) * (softmax(x)_ik * S_i - y_ik)      where S_i = sum_j y_ij
//
// but the kernel (ggml-cpu/ops.cpp, cross_entropy_loss_back_f32) computes
//
//   dL/dx_ik    =  (1/nr) * (softmax(x)_ik - y_ik)
//
// i.e. it hardcodes S_i == 1, which the op is entitled to do -- it is documented as taking one-hot
// labels. A weighted one-hot has S_i = w_i, so every row whose weight is not exactly 1 gets the
// wrong gradient, and a MASKED row (w_i = 0) gets softmax(x)_i/nr rather than zero. The loss still
// falls, because the wrong gradient is correlated with the right one; it just converges somewhere
// else. Nothing short of a finite-difference check would have noticed.
//
// The sparse op derives its backward from its own forward and weights it per token, so a masked
// token contributes a bitwise zero (ADR-0003) and a fractional weight scales the gradient exactly.
ggml_tensor * build_masked_ce(ggml_context * ctx_compute, ggml_cgraph * gf, ggml_tensor * logits, int32_t pos,
                              int32_t n_ubatch, void * userdata) {
    auto * lc = (loss_ctx *)userdata;

    lc->pos = pos;
    lc->n_ubatch = n_ubatch;
    lc->n_vocab = logits->ne[0];

    lc->labels = ggml_new_tensor_1d(ctx_compute, GGML_TYPE_I32, n_ubatch);
    ggml_set_input(lc->labels);
    ggml_set_name(lc->labels, "ll_ce_labels");

    lc->ce_weights = ggml_new_tensor_1d(ctx_compute, GGML_TYPE_F32, n_ubatch);
    ggml_set_input(lc->ce_weights);
    ggml_set_name(lc->ce_weights, "ll_ce_weights");

    if (lc->kind == LL_LOSS_GRPO) {
        // Four constants, one per token. None of them is differentiated: ggml only propagates
        // gradients from tensors flagged PARAM or LOSS, so a pure input never gets an accumulator.
        lc->grpo_adv = ggml_new_tensor_1d(ctx_compute, GGML_TYPE_F32, n_ubatch);
        ggml_set_input(lc->grpo_adv);
        ggml_set_name(lc->grpo_adv, "ll_grpo_adv");

        lc->grpo_logp_old = ggml_new_tensor_1d(ctx_compute, GGML_TYPE_F32, n_ubatch);
        ggml_set_input(lc->grpo_logp_old);
        ggml_set_name(lc->grpo_logp_old, "ll_grpo_logp_old");

        lc->grpo_logp_ref = ggml_new_tensor_1d(ctx_compute, GGML_TYPE_F32, n_ubatch);
        ggml_set_input(lc->grpo_logp_ref);
        ggml_set_name(lc->grpo_logp_ref, "ll_grpo_logp_ref");

        lc->grpo_kl_w = ggml_new_tensor_1d(ctx_compute, GGML_TYPE_F32, n_ubatch);
        ggml_set_input(lc->grpo_kl_w);
        ggml_set_name(lc->grpo_kl_w, "ll_grpo_kl_w");
    }

    // logit_scale = 1.0 (off), softcap = 0.0 (off): a plain llama arch applies neither. Gemma-2
    // and friends do, and S1-11 wires them through -- the op already takes them.
    ggml_tensor * per_token =
        ggml_cross_entropy_loss_sparse(ctx_compute, logits, lc->labels, lc->ce_weights, 1.0f, 0.0f);
    ggml_set_name(per_token, "ll_ce_per_token");

    // ce_sparse gives  per_token[i] = w_i * (logsumexp(x_i) - x_i[target_i]) = -w_i * logp_i.
    // Summing it is therefore  -sum_i w_i * logp_i  -- the negative weighted log-probability.
    ggml_tensor * loss = ggml_sum(ctx_compute, per_token);
    ggml_set_name(loss, "ll_ce_loss");

    if (lc->kind == LL_LOSS_GRPO) {
        // GRPO. The clipped importance-ratio surrogate, weighted by each token's group advantage.
        //
        //     logp_new = -per_token                      (ce_sparse already carries the mask)
        //     r        = exp(logp_new - logp_old)        the importance ratio
        //     L_i      = min( r_i * A_i , clip(r_i, 1-eps, 1+eps) * A_i )
        //     loss     = -sum_i L_i  +  sum_i kl_w_i * k3_i
        //
        // logp_old and the advantages are CONSTANTS -- named inputs, no gradient path. The gradient
        // is the policy's alone, which is what makes this an off-policy correction rather than a
        // second model.
        //
        // THE CLIP IS BUILT FROM RELU, and that is the interesting part. The obvious ggml_clamp
        // does now have a backward rule, but the relu composite is the reference form the ticket
        // pins, it is topology-stable, and it needs nothing that was not already there:
        //
        //     clip(r, lo, hi) = lo + relu(r - lo) - relu(r - hi)
        //     min(a, b)       = a - relu(a - b)
        //
        // Check the clip on its three regions: r < lo gives lo + 0 - 0; lo <= r <= hi gives
        // lo + (r - lo) - 0 = r; r > hi gives lo + (r - lo) - (r - hi) = hi. And min is exact
        // whichever way round a and b fall -- which matters, because a NEGATIVE advantage swaps
        // them, and PPO's asymmetry between "made a good token likelier" and "made a bad token
        // likelier" lives entirely in that swap.
        //
        // The min's ARGUMENT ORDER is load-bearing and is NOT the form the blueprint prescribed.
        // See the comment at the min itself, below.
        const float lo = 1.0f - lc->clip_eps;
        const float hi = 1.0f + lc->clip_eps;

        ggml_tensor * logp_new = ggml_scale(ctx_compute, per_token, -1.0f);
        ggml_set_name(logp_new, "ll_grpo_logp_new");

        ggml_tensor * ratio = ggml_exp(ctx_compute, ggml_sub(ctx_compute, logp_new, lc->grpo_logp_old));
        ggml_set_name(ratio, "ll_grpo_ratio");

        // clip(r, lo, hi), from relu identities. ggml_scale_bias(x, 1, -lo) is x - lo in one node,
        // and its backward correctly ignores the bias (d/dx of s*x + b is s).
        ggml_tensor * clipped = ggml_scale_bias(
            ctx_compute,
            ggml_sub(ctx_compute, ggml_relu(ctx_compute, ggml_scale_bias(ctx_compute, ratio, 1.0f, -lo)),
                     ggml_relu(ctx_compute, ggml_scale_bias(ctx_compute, ratio, 1.0f, -hi))),
            1.0f, lo);
        ggml_set_name(clipped, "ll_grpo_clipped");

        ggml_tensor * unclipped_obj = ggml_mul(ctx_compute, ratio, lc->grpo_adv);
        ggml_tensor * clipped_obj = ggml_mul(ctx_compute, clipped, lc->grpo_adv);

        // min(a, b) = a - relu(a - b), and NOT the equally-true b - relu(b - a).
        //
        // The two are algebraically identical and give the same forward value. They differ in where
        // the gradient goes AT A TIE, and that is not a nicety.
        //
        // ggml_step(0) == 0. So when r lands exactly on lo, relu(r - lo) is fed exactly zero, its
        // backward is step(0) = 0, and d(clipped)/dr is 0. But clipped *evaluates* to lo == r, so
        // clipped_obj == unclipped_obj bit for bit -- a tie. And `b - relu(b - a)` at a tie routes
        // the whole gradient into b, which is the branch whose derivative was just computed as zero.
        //
        // Net: dL/dr = 0. For a POSITIVE advantage that is simply wrong. min(rA, clip(r)A) equals rA
        // on BOTH sides of lo -- the clip does not bind from below when A > 0 -- so the objective is
        // smooth there, with slope A. There is no kink to pick a subgradient at. The token's entire
        // policy gradient is dropped. Measured on the fixture:
        //
        //   r a few ulps below lo:  max|dL/dB| = 6.465078e-02
        //   r == lo exactly:        max|dL/dB| = 0            <-- and its neighbours agree with each
        //   r a few ulps above lo:  max|dL/dB| = 6.465083e-02     other, so it is not a kink
        //
        // `a - relu(a - b)` routes a tie into `a`, the UNCLIPPED branch, whose derivative is A. That
        // is correct at r == lo, and still a legal subgradient at r == hi (a genuine kink, where
        // either 0 or A is defensible). Same forward, same node count, right derivative.
        ggml_tensor * surrogate = ggml_sub(ctx_compute, unclipped_obj,
                                           ggml_relu(ctx_compute, ggml_sub(ctx_compute, unclipped_obj, clipped_obj)));
        ggml_set_name(surrogate, "ll_grpo_surrogate");

        // The k3 KL estimator: exp(d) - d - 1, with d = logp_ref - logp_new. Written as
        // expm1(d) - d, which is the same number and one node fewer -- and, crucially, EXACT near
        // d = 0, which is exactly where a GRPO run starts. exp(d) - 1 in F32 at d ~ 1e-4 loses
        // every significant digit to cancellation; expm1 does not lose any.
        //
        // The KL nodes are built even when the KL is off, and the coefficient is folded into kl_w
        // host-side. Building them conditionally would change the graph's node COUNT between runs,
        // and ggml-opt indexes its optimizer state by node index off the first graph it sees.
        ggml_tensor * d_ref = ggml_sub(ctx_compute, lc->grpo_logp_ref, logp_new);
        ggml_tensor * k3 = ggml_sub(ctx_compute, ggml_expm1(ctx_compute, d_ref), d_ref);
        ggml_tensor * kl = ggml_mul(ctx_compute, k3, lc->grpo_kl_w);
        ggml_set_name(kl, "ll_grpo_kl");

        loss = ggml_add(ctx_compute, ggml_scale(ctx_compute, ggml_sum(ctx_compute, surrogate), -1.0f),
                        ggml_sum(ctx_compute, kl));
        ggml_set_name(loss, "ll_grpo_loss");
    }

    if (lc->kind == LL_LOSS_DPO) {
        // DPO. With weights of +1 on the chosen completion's tokens and -1 on the rejected's,
        //
        //     loss (so far) = -(logp_chosen - logp_rejected) = -delta
        //
        // and the objective is
        //
        //     L = -log sigmoid( beta * (delta - ref_delta) )
        //       =  softplus( -beta * (delta - ref_delta) )          [ -log sigmoid(x) = softplus(-x) ]
        //       =  softplus( beta * loss + beta * ref_delta )       [ since loss = -delta ]
        //
        // The softplus identity is not a stylistic choice. SIGMOID's backward falls into the unary
        // default and ABORTS, so -log(sigmoid(x)) cannot be built that way at all -- and softplus is
        // one node and numerically stable at large |x|, where log(1 + exp(-x)) computed naively is
        // not.
        //
        // ref_delta is the REFERENCE model's log-ratio, precomputed with the adapter off and handed
        // in as a constant. It carries no gradient, which is the whole point: DPO's gradient is the
        // policy's, shaped by how far it has moved from a reference that does not move.
        lc->dpo_ref = ggml_new_tensor_1d(ctx_compute, GGML_TYPE_F32, 1);
        ggml_set_input(lc->dpo_ref);
        ggml_set_name(lc->dpo_ref, "ll_dpo_ref");

        ggml_tensor * z = ggml_scale(ctx_compute, loss, lc->beta);
        z = ggml_add(ctx_compute, z, lc->dpo_ref);
        ggml_set_name(z, "ll_dpo_logit");

        loss = ggml_softplus(ctx_compute, z);
        ggml_set_name(loss, "ll_dpo_loss");
    }

    ggml_build_forward_expand(gf, loss);

    return loss;
}

// Fill the targets and weights. Runs after ggml_opt_alloc, because until then they have no memory.
//
// It is also the only moment at which ggml_opt will hand over the gradient accumulators, so the
// shim grabs them here on the first training step. See capture_grad_accs.
void upload_masked_ce(void * userdata) {
    auto * lc = (loss_ctx *)userdata;

    if (lc->train && lc->state != nullptr) {
        capture_grad_accs(lc->state);
    }

    const int64_t n_vocab = lc->n_vocab;
    const int32_t n_ubatch = lc->n_ubatch;

    // Normalize by the weight actually carried, not by the token count: a batch that is 90% prompt
    // must not have its loss (and so its gradient) scaled down 10x relative to one that is 10%
    // prompt. Folding 1/sum(w) into the weights here keeps the graph a plain sum. A batch with no
    // unmasked token at all has zero loss and zero gradient, which is the honest answer.
    //
    // DPO does NOT normalize, and must not. Its weights are +1 on the chosen completion's tokens and
    // -1 on the rejected's, so sum(w) is the DIFFERENCE in their lengths -- a number with no meaning
    // at all as a denominator, and zero whenever the two happen to be the same length. What DPO
    // wants is the plain sum of per-token log-probabilities, which is what an unnormalized weighted
    // sum with +/-1 weights is.
    float scale = 1.0f;

    if (lc->kind == LL_LOSS_SFT) {
        float sum_w = 0.0f;
        for (int32_t i = 0; i < n_ubatch; ++i) {
            sum_w += lc->weights[lc->pos + i];
        }
        scale = sum_w > 0.0f ? 1.0f / sum_w : 0.0f;
    }

    std::vector<int32_t> labels(n_ubatch);
    std::vector<float> weights(n_ubatch);

    for (int32_t i = 0; i < n_ubatch; ++i) {
        const int32_t target = lc->targets[lc->pos + i];
        const bool in_range = target >= 0 && target < n_vocab;

        // An out-of-range target is masked out rather than allowed to index off the end of a row.
        // Label 0 is then arbitrary but never read, because its weight is zero.
        labels[i] = in_range ? target : 0;
        weights[i] = in_range ? lc->weights[lc->pos + i] * scale : 0.0f;
    }

    ggml_backend_tensor_set(lc->labels, labels.data(), 0, labels.size() * sizeof(int32_t));
    ggml_backend_tensor_set(lc->ce_weights, weights.data(), 0, weights.size() * sizeof(float));

    if (lc->dpo_ref != nullptr) {
        const float value = lc->beta * lc->ref_delta;
        ggml_backend_tensor_set(lc->dpo_ref, &value, 0, sizeof(float));
    }

    if (lc->kind == LL_LOSS_GRPO) {
        const ll_grpo_inputs * g = lc->grpo;

        std::vector<float> adv(n_ubatch);
        std::vector<float> logp_old(n_ubatch);
        std::vector<float> logp_ref(n_ubatch);
        std::vector<float> kl_w(n_ubatch);

        for (int32_t i = 0; i < n_ubatch; ++i) {
            const int32_t j = lc->pos + i;

            // A masked position gets zeros in EVERY array, and the shim enforces that rather than
            // trusting the caller -- because getting it wrong produces NaN, not an error.
            //
            // On a masked token ce_sparse multiplies by w = 0, so logp_new is exactly 0. A nonzero
            // logp_ref there would then make d = logp_ref - 0 a large number, expm1(d) would be
            // +inf, and inf * kl_w = inf * 0 = NaN. One stray value in a padding slot would take
            // the whole batch's loss and every gradient with it, in silence.
            //
            // With zeros: logp_new = 0, logp_old = 0, so the ratio is exp(0) = 1; the advantage is
            // 0, so the surrogate is 0; d = 0, so k3 = expm1(0) - 0 = 0. The token contributes
            // nothing to the loss and nothing to the gradient, which is what "masked" means.
            const bool live = weights[i] != 0.0f;

            adv[i] = live ? g->adv[j] : 0.0f;
            logp_old[i] = live ? g->logp_old[j] : 0.0f;
            logp_ref[i] = live ? (g->logp_ref ? g->logp_ref[j] : g->logp_old[j]) : 0.0f;
            kl_w[i] = live && g->kl_w ? g->kl_w[j] : 0.0f;
        }

        ggml_backend_tensor_set(lc->grpo_adv, adv.data(), 0, adv.size() * sizeof(float));
        ggml_backend_tensor_set(lc->grpo_logp_old, logp_old.data(), 0, logp_old.size() * sizeof(float));
        ggml_backend_tensor_set(lc->grpo_logp_ref, logp_ref.data(), 0, logp_ref.size() * sizeof(float));
        ggml_backend_tensor_set(lc->grpo_kl_w, kl_w.data(), 0, kl_w.size() * sizeof(float));
    }
}

} // namespace

namespace {

// The one training step, with the objective as a parameter. ll_train_step and ll_train_step_dpo are
// both this; ll_logp_delta is this with train=false and no reduction.
int32_t train_step_impl(llama_context * ctx, const int32_t * tokens, const int32_t * targets, const float * weights,
                        const int32_t * seq_ids, const int32_t * positions, int32_t n_tokens, ll_loss_kind kind,
                        float beta, float ref_delta, const ll_grpo_inputs * grpo, bool train, float * loss_out) {
    if (ctx == nullptr || tokens == nullptr || targets == nullptr || weights == nullptr || n_tokens <= 0) {
        return LL_ERR_INVALID_ARG;
    }

    const auto it = train_states().find(ctx);
    if (it == train_states().end()) {
        return LL_ERR_NOT_INITIALIZED;
    }
    ll_train_state * state = it->second.get();

    // One ubatch per step -- see the opt_period comment in ll_opt_init_lora. A batch that spans
    // several ubatches would take several optimizer steps, which is not what "one training step"
    // means.
    if ((uint32_t)n_tokens > ctx->n_ubatch()) {
        LLAMA_LOG_ERROR("%s: n_tokens (%d) exceeds n_ubatch (%u): a training step must fit in one "
                        "ubatch. Accumulate gradients across steps instead.\n",
                        __func__, n_tokens, ctx->n_ubatch());
        return LL_ERR_INVALID_ARG;
    }

    // Every step must have the same n_tokens as the first.
    //
    // This is not a limitation ggml announces, and it does not currently crash -- a 32-token step
    // followed by a 16-token step runs, and the loss keeps falling. It is guarded anyway, because
    // what makes it work is a coincidence:
    //
    // ggml_opt_build allocates opt_ctx->grad_accs ONCE, sized to the FIRST graph's node count
    // (`if (opt_ctx->grad_accs.empty())`, ggml-opt.cpp), and thereafter ggml_build_backward_expand
    // indexes that array by node index on every rebuilt graph. A transformer's node COUNT happens
    // not to depend on the ubatch size, so the indices line up. Nothing enforces that -- a fused-op
    // choice that keys off shape, or a future graph tweak, changes the node count for one shape and
    // not another, and then the backward reads past the end of the vector. That is a silent
    // out-of-bounds, not an assert: it corrupts gradients rather than stopping.
    //
    // So the shape is pinned to the first step's, and a mismatch is a clear error. Fixed-shape
    // batches are what the trainer produces anyway (it pads with weight-0 tokens); this just means
    // a caller cannot get it wrong quietly.
    if (state->loss_kind_seen < 0) {
        state->loss_kind_seen = (int32_t)kind;
    } else if (state->loss_kind_seen != (int32_t)kind) {
        LLAMA_LOG_ERROR("%s: the objective changed from %d to %d on one context. Different "
                        "objectives build different numbers of graph nodes, and ggml-opt indexes its "
                        "optimizer state by node index off the FIRST graph -- so this would read "
                        "past the end of that state rather than fail. Use a separate context.\n",
                        __func__, state->loss_kind_seen, (int32_t)kind);
        return LL_ERR_SHAPE_MISMATCH;
    }

    if (state->n_tokens_seen == 0) {
        state->n_tokens_seen = n_tokens;
    } else if (state->n_tokens_seen != n_tokens) {
        LLAMA_LOG_ERROR("%s: n_tokens changed from %d to %d. Every step must have the same shape: "
                        "ggml-opt indexes its optimizer state by graph node index, and sizes it "
                        "from the first graph it sees. Pad the batch instead (weight 0).\n",
                        __func__, state->n_tokens_seen, n_tokens);
        return LL_ERR_SHAPE_MISMATCH;
    }

    // Has llama_context swapped its scheduler out from under us?
    //
    // ggml_opt_init captured the pointer and holds it for the life of the optimizer context.
    // llama_context does `sched.reset(ggml_backend_sched_new(...))` whenever sched_need_reserve is
    // set -- and llama_decode and llama_set_adapters_lora BOTH set it. The old scheduler is freed
    // and the optimizer is left holding it.
    //
    // The next step then aborts inside ggml-backend on an index that is no longer in range, or
    // corrupts memory quietly. Neither says anything about what actually happened, so check.
    if (ctx->get_sched() != state->sched) {
        LLAMA_LOG_ERROR("%s: llama_context has replaced its backend scheduler since ll_opt_init_lora "
                        "-- the optimizer is holding a freed one. Something called llama_decode or "
                        "llama_set_adapters_lora on this context after training was set up; both "
                        "force a re-reserve, which frees the scheduler. Do inference on a separate "
                        "context, or before ll_opt_init_lora.\n",
                        __func__);
        return LL_ERR_SCHED_INVALIDATED;
    }

    // Packing (S1-07): several independent samples in one batch, told apart by seq_id.
    //
    // Two things must hold for that to be correct, and both fail SILENTLY, so both are checked
    // rather than documented.
    //
    // 1. The batch must not be REORDERED on its way to the graph. llama.cpp picks its ubatch
    //    splitter from the KV cache's stream count -- `n_stream == 1 ? split_simple : split_equal`
    //    (llama-kv-cache.cpp), and n_stream is `unified ? 1 : n_seq_max`. split_simple hands out
    //    contiguous slices in the original order, which is exactly what the loss builder assumes
    //    when it reads targets[pos + i]. split_equal REGROUPS THE TOKENS BY SEQUENCE. With
    //    kv_unified off, targets and weights would then pair with the wrong tokens -- and the loss
    //    would still look perfectly reasonable, because it IS a valid loss, just of the wrong thing.
    //
    // 2. Every seq_id must fit in the context's n_seq_max, or the batch allocator rejects the batch
    //    with a message about sequence ids and nothing about packing.
    if (seq_ids != nullptr) {
        if (!ctx->get_cparams().kv_unified) {
            LLAMA_LOG_ERROR("%s: packing (seq_ids) needs a context created with kv_unified=true. "
                            "Without it, llama.cpp regroups the batch by sequence and the loss "
                            "targets would silently pair with the wrong tokens.\n",
                            __func__);
            return LL_ERR_INVALID_ARG;
        }

        const int32_t n_seq_max = (int32_t)ctx->n_seq_max();
        for (int32_t i = 0; i < n_tokens; ++i) {
            if (seq_ids[i] < 0 || seq_ids[i] >= n_seq_max) {
                LLAMA_LOG_ERROR("%s: seq_id %d at position %d is outside n_seq_max (%d). Create the "
                                "context with n_seq_max >= the most samples you will pack.\n",
                                __func__, seq_ids[i], i, n_seq_max);
                return LL_ERR_INVALID_ARG;
            }
        }
    }

    // Refresh the optimizer hyperparameters from the caller's struct. This is what makes a
    // learning-rate schedule a plain attribute assignment in Python.
    state->opt_pars.adamw.alpha = state->params->alpha;
    state->opt_pars.adamw.beta1 = state->params->beta1;
    state->opt_pars.adamw.beta2 = state->params->beta2;
    state->opt_pars.adamw.eps = state->params->eps;
    state->opt_pars.adamw.wd = state->params->wd;

    llama_batch batch = llama_batch_init(n_tokens, 0, 1);
    batch.n_tokens = n_tokens;
    for (int32_t i = 0; i < n_tokens; ++i) {
        batch.token[i] = tokens[i];

        // NULL means "one sequence, in order" -- the unpacked case. When they are given, each
        // packed sample gets its own seq_id and restarts its positions at 0, and llama.cpp's mask
        // then makes the samples invisible to one another (S1-07).
        batch.pos[i] = positions ? positions[i] : i;
        batch.n_seq_id[i] = 1;
        batch.seq_id[i][0] = seq_ids ? seq_ids[i] : 0;

        batch.logits[i] = true; // every position needs logits: every one may carry loss
    }

    loss_ctx lc;
    lc.state = state;
    lc.train = train;
    lc.kind = kind;
    lc.beta = beta;
    lc.ref_delta = ref_delta;
    lc.grpo = grpo;
    lc.clip_eps = grpo != nullptr ? grpo->clip_eps : 0.2f;
    lc.targets = targets;
    lc.weights = weights;
    lc.n_tokens = n_tokens;

    ggml_opt_result_t result = ggml_opt_result_init();

    const int32_t status =
        ctx->opt_step_custom(batch, state->opt_ctx, result, build_masked_ce, upload_masked_ce, &lc, train);

    if (status == 0 && loss_out != nullptr) {
        double loss = 0.0;
        ggml_opt_result_loss(result, &loss, nullptr);
        *loss_out = (float)loss;
    }

    ggml_opt_result_free(result);
    llama_batch_free(batch);

    return status == 0 ? LL_OK : LL_ERR_STEP_FAILED;
}

} // namespace

int32_t ll_train_step(llama_context * ctx, const int32_t * tokens, const int32_t * targets, const float * weights,
                      const int32_t * seq_ids, const int32_t * positions, int32_t n_tokens, bool train,
                      float * loss_out) {
    return train_step_impl(ctx, tokens, targets, weights, seq_ids, positions, n_tokens, LL_LOSS_SFT,
                           /*beta =*/0.0f, /*ref_delta =*/0.0f, /*grpo =*/nullptr, train, loss_out);
}

int32_t ll_train_step_dpo(llama_context * ctx, const int32_t * tokens, const int32_t * targets, const float * weights,
                          const int32_t * seq_ids, const int32_t * positions, int32_t n_tokens, float beta,
                          float ref_delta, bool train, float * loss_out) {
    if (beta <= 0.0f) {
        return LL_ERR_INVALID_ARG;
    }

    return train_step_impl(ctx, tokens, targets, weights, seq_ids, positions, n_tokens, LL_LOSS_DPO, beta, ref_delta,
                           /*grpo =*/nullptr, train, loss_out);
}

int32_t ll_train_step_grpo(llama_context * ctx, const int32_t * tokens, const int32_t * targets, const float * weights,
                           const int32_t * seq_ids, const int32_t * positions, int32_t n_tokens,
                           const ll_grpo_inputs * grpo, bool train, float * loss_out) {
    if (grpo == nullptr || grpo->adv == nullptr || grpo->logp_old == nullptr) {
        return LL_ERR_INVALID_ARG;
    }

    // An epsilon outside (0, 1) is not a clip. At 0 the clipped branch is the constant 1 and the
    // ratio's gradient vanishes; at 1 the lower bound is 0 and the clip never binds below.
    if (grpo->clip_eps <= 0.0f || grpo->clip_eps >= 1.0f) {
        return LL_ERR_INVALID_ARG;
    }

    // The weights ARE the completion mask, and they have to be exactly 0 or 1.
    //
    // This is not fastidiousness. logp_new is -ce_sparse(...), and ce_sparse multiplies by w_i --
    // so a weight of 0.5 does not down-weight this token's contribution, it HALVES the log
    // probability that goes into the importance ratio. exp(0.5*logp - logp_old) is not a ratio of
    // anything. Nothing would fail; the run would just optimize a different objective.
    //
    // Every normalization GRPO needs belongs in `adv` and `kl_w`, where it scales the loss without
    // touching the logprob.
    for (int32_t i = 0; i < n_tokens; ++i) {
        if (weights[i] != 0.0f && weights[i] != 1.0f) {
            return LL_ERR_INVALID_ARG;
        }
    }

    return train_step_impl(ctx, tokens, targets, weights, seq_ids, positions, n_tokens, LL_LOSS_GRPO,
                           /*beta =*/0.0f, /*ref_delta =*/0.0f, grpo, train, loss_out);
}

int32_t ll_logp_delta(llama_context * ctx, const int32_t * tokens, const int32_t * targets, const float * weights,
                      const int32_t * seq_ids, const int32_t * positions, int32_t n_tokens, float * out) {
    if (ctx == nullptr || tokens == nullptr || targets == nullptr || weights == nullptr || out == nullptr ||
        n_tokens <= 0) {
        return LL_ERR_INVALID_ARG;
    }

    // Its OWN, throwaway, forward-only optimizer context. Three things force that, and each one bit.
    //
    // 1. It cannot share the training context's optimizer. ggml-opt sizes its optimizer state from
    //    the node count of the FIRST graph it sees and indexes that state by node index forever
    //    after. This graph has no DPO tail, so it has fewer nodes than a training step's -- and the
    //    training step would then index past the end of that state and call whatever it found a
    //    gradient tensor. It does not assert. (Found as a segfault.)
    //
    // 2. It cannot use llama_decode. A plain decode on a TRAINING-mode context crashes on the second
    //    call -- reproduced in isolation: four decodes are fine with cparams.training false, and the
    //    second one segfaults with it true. It also sets sched_need_reserve, which makes
    //    llama_context replace its scheduler; anything holding the old one is left with a freed
    //    pointer (see LL_ERR_SCHED_INVALIDATED).
    //
    // 3. And it must use the TRAINING graph, not the inference one. The two use different kernels --
    //    KV cache, flash attention, repacked buffer types -- and on a quantized base they disagree at
    //    the 1e-4 level. Measured: the DPO loss at initialization came out at 0.69252 rather than
    //    log 2 = 0.693147, which looks like a rounding error and is actually two different models.
    //
    // A forward-only build type means ggml_opt_build stops after the forward graph: no backward, no
    // gradient accumulators, and -- the reason this works at all -- no requirement that the graph
    // contain any trainable parameters.
    ctx->set_training(true);

    ggml_opt_params opt_params = ggml_opt_default_params(ctx->get_sched(), GGML_OPT_LOSS_TYPE_SUM);
    opt_params.build_type = GGML_OPT_BUILD_TYPE_FORWARD;
    opt_params.opt_period = 1;

    ggml_opt_context_t opt_ctx = ggml_opt_init(opt_params);

    llama_batch batch = llama_batch_init(n_tokens, 0, 1);
    batch.n_tokens = n_tokens;
    for (int32_t i = 0; i < n_tokens; ++i) {
        batch.token[i] = tokens[i];
        batch.pos[i] = positions ? positions[i] : i;
        batch.n_seq_id[i] = 1;
        batch.seq_id[i][0] = seq_ids ? seq_ids[i] : 0;
        batch.logits[i] = true;
    }

    loss_ctx lc;
    lc.state = nullptr; // no training state: nothing to capture, nothing to accumulate
    lc.train = false;
    lc.kind = LL_LOSS_LOGP;
    lc.targets = targets;
    lc.weights = weights;
    lc.n_tokens = n_tokens;

    ggml_opt_result_t result = ggml_opt_result_init();

    const int32_t status =
        ctx->opt_step_custom(batch, opt_ctx, result, build_masked_ce, upload_masked_ce, &lc, /*train =*/false);

    double loss = 0.0;
    if (status == 0) {
        double value = 0.0;
        double unc = 0.0;
        ggml_opt_result_loss(result, &value, &unc);
        loss = value;
    }

    ggml_opt_result_free(result);
    llama_batch_free(batch);
    ggml_opt_free(opt_ctx);

    if (status != 0) {
        return LL_ERR_STEP_FAILED;
    }

    // ce_sparse is -logp, so the graph's sum is -sum_i w_i * logp_i. Negate it once, here, so that
    // no caller is left to remember the sign of a cross-entropy.
    *out = (float)-loss;

    return LL_OK;
}

// ---------------------------------------------------------------------------
// Debug accessors (S1-03) -- see farm_api.h for why these exist and why ll_debug_*.
// ---------------------------------------------------------------------------

namespace {

// The A or B tensor of one adapted base tensor, or nullptr.
ggml_tensor * find_adapter_tensor(ll_train_state * state, const char * base_name, bool is_b) {
    // param_tensors was filled in ab_map order, A then B for each entry. Rather than re-derive
    // that ordering, look the tensor up by its ggml name -- which llama.cpp set from the adapter
    // GGUF, so it is exactly "<base>.lora_a" / "<base>.lora_b".
    const std::string want = std::string(base_name) + (is_b ? ".lora_b" : ".lora_a");

    for (ggml_tensor * t : state->param_tensors) {
        if (want == ggml_get_name(t)) {
            return t;
        }
    }
    return nullptr;
}

ll_train_state * state_for(llama_context * ctx) {
    const auto it = train_states().find(ctx);
    return it == train_states().end() ? nullptr : it->second.get();
}

// A flagged parameter tensor addressed by its EXACT ggml name -- the full-finetune analogue of
// find_adapter_tensor, which instead builds "<base>.lora_a/b". Base weights carry their GGUF name
// unchanged ("blk.0.attn_q.weight", "output.weight", "token_embd.weight"), so an exact match is all
// that is needed.
ggml_tensor * find_param_by_name(ll_train_state * state, const char * name) {
    for (ggml_tensor * t : state->param_tensors) {
        if (std::strcmp(ggml_get_name(t), name) == 0) {
            return t;
        }
    }
    return nullptr;
}

} // namespace

int64_t ll_debug_n_elements(llama_context * ctx, const char * base_name, bool is_b) {
    if (ctx == nullptr || base_name == nullptr) {
        return LL_ERR_INVALID_ARG;
    }
    ll_train_state * state = state_for(ctx);
    if (state == nullptr) {
        return LL_ERR_NOT_INITIALIZED;
    }

    ggml_tensor * t = find_adapter_tensor(state, base_name, is_b);
    if (t == nullptr) {
        return LL_ERR_INVALID_ARG;
    }
    return ggml_nelements(t);
}

int64_t ll_debug_get_tensor(llama_context * ctx, const char * base_name, bool is_b, float * out, int64_t n_max) {
    if (ctx == nullptr || base_name == nullptr || out == nullptr) {
        return LL_ERR_INVALID_ARG;
    }
    ll_train_state * state = state_for(ctx);
    if (state == nullptr) {
        return LL_ERR_NOT_INITIALIZED;
    }

    ggml_tensor * t = find_adapter_tensor(state, base_name, is_b);
    if (t == nullptr) {
        return LL_ERR_INVALID_ARG;
    }

    const int64_t n = ggml_nelements(t);
    if (n > n_max) {
        return LL_ERR_INVALID_ARG;
    }

    ggml_backend_tensor_get(t, out, 0, n * sizeof(float));
    return n;
}

int64_t ll_debug_set_tensor(llama_context * ctx, const char * base_name, bool is_b, const float * data, int64_t n) {
    if (ctx == nullptr || base_name == nullptr || data == nullptr) {
        return LL_ERR_INVALID_ARG;
    }
    ll_train_state * state = state_for(ctx);
    if (state == nullptr) {
        return LL_ERR_NOT_INITIALIZED;
    }

    ggml_tensor * t = find_adapter_tensor(state, base_name, is_b);
    if (t == nullptr || ggml_nelements(t) != n) {
        return LL_ERR_INVALID_ARG;
    }

    ggml_backend_tensor_set(t, data, 0, n * sizeof(float));
    return n;
}

int64_t ll_debug_grad(llama_context * ctx, const char * base_name, bool is_b, float * out, int64_t n_max) {
    if (ctx == nullptr || base_name == nullptr || out == nullptr) {
        return LL_ERR_INVALID_ARG;
    }
    ll_train_state * state = state_for(ctx);
    if (state == nullptr || state->opt_ctx == nullptr) {
        return LL_ERR_NOT_INITIALIZED;
    }

    ggml_tensor * t = find_adapter_tensor(state, base_name, is_b);
    if (t == nullptr) {
        return LL_ERR_INVALID_ARG;
    }

    // Read the CACHED accumulator, not ggml_opt_grad_acc.
    //
    // ggml_opt_grad_acc looks the accumulator up in opt_ctx->gb_opt -- and with dynamic graphs
    // ggml_opt_eval sets gb_opt (and gf, and gb_grad) back to NULL when it finishes, because the
    // caller is expected to free the compute context those graphs live in. So calling
    // ggml_opt_grad_acc after a step dereferences a null graph. It is only usable BETWEEN
    // ggml_opt_alloc and ggml_opt_eval -- which is where the shim captures it (see capture_grads).
    //
    // The accumulator TENSORS themselves live in opt_ctx's static context and persist, holding
    // the gradient from the last backward pass. Caching their pointers once is all that is needed.
    const auto git = state->grad_accs.find(std::string(ggml_get_name(t)));
    if (git == state->grad_accs.end()) {
        return LL_ERR_NOT_INITIALIZED;
    }
    ggml_tensor * grad = git->second;
    if (grad == nullptr) {
        return LL_ERR_NOT_INITIALIZED;
    }

    const int64_t n = ggml_nelements(grad);
    if (n > n_max) {
        return LL_ERR_INVALID_ARG;
    }

    ggml_backend_tensor_get(grad, out, 0, n * sizeof(float));
    return n;
}

int64_t ll_debug_base_n_elements(llama_context * ctx, const char * name) {
    if (ctx == nullptr || name == nullptr) {
        return LL_ERR_INVALID_ARG;
    }
    ll_train_state * state = state_for(ctx);
    if (state == nullptr) {
        return LL_ERR_NOT_INITIALIZED;
    }

    ggml_tensor * t = find_param_by_name(state, name);
    if (t == nullptr) {
        return LL_ERR_INVALID_ARG;
    }
    return ggml_nelements(t);
}

int64_t ll_debug_base_grad(llama_context * ctx, const char * name, float * out, int64_t n_max) {
    if (ctx == nullptr || name == nullptr || out == nullptr) {
        return LL_ERR_INVALID_ARG;
    }
    ll_train_state * state = state_for(ctx);
    if (state == nullptr || state->opt_ctx == nullptr) {
        return LL_ERR_NOT_INITIALIZED;
    }

    ggml_tensor * t = find_param_by_name(state, name);
    if (t == nullptr) {
        return LL_ERR_INVALID_ARG;
    }

    // The cached accumulator, keyed by the base tensor's own name -- same window and same reason as
    // ll_debug_grad (see capture_grad_accs).
    const auto git = state->grad_accs.find(std::string(name));
    if (git == state->grad_accs.end() || git->second == nullptr) {
        return LL_ERR_NOT_INITIALIZED;
    }
    ggml_tensor * grad = git->second;

    const int64_t n = ggml_nelements(grad);
    if (n > n_max) {
        return LL_ERR_INVALID_ARG;
    }

    ggml_backend_tensor_get(grad, out, 0, n * sizeof(float));
    return n;
}

int32_t ll_grad_norms(llama_context * ctx, float * pre_out, float * post_out) {
    if (ctx == nullptr || pre_out == nullptr || post_out == nullptr) {
        return LL_ERR_INVALID_ARG;
    }
    ll_train_state * state = state_for(ctx);
    if (state == nullptr || state->opt_ctx == nullptr) {
        return LL_ERR_NOT_INITIALIZED;
    }
    // NaN unless the last eval was a clipping OPT step -- ggml-opt caches these out of the graph.
    *pre_out = ggml_opt_grad_norm_pre(state->opt_ctx);
    *post_out = ggml_opt_grad_norm_post(state->opt_ctx);
    return LL_OK;
}

// ---------------------------------------------------------------------------
// Optimizer-state checkpoint / resume (S1-09)
// ---------------------------------------------------------------------------
//
// In farm_train.cpp rather than a farm_checkpoint.cpp as the ticket suggests, because the state
// these read -- ll_train_state and its captured moment tensors -- lives in this file's anonymous
// namespace. Moving them out to share it would widen the shim's internal surface for no gain.

namespace {

// The moment tensors, in a STABLE order: sorted by parameter name.
//
// The map they come from is an unordered_map, whose iteration order is not promised to be the same
// between two builds. An index that meant a different tensor on a different machine is a fine way
// to write a checkpoint that resumes into the wrong optimizer state -- and it would not crash, it
// would just train slightly wrong.
std::vector<std::string> sorted_param_names(ll_train_state * state) {
    std::vector<std::string> names;
    names.reserve(state->grad_m.size());

    for (const auto & [name, tensor] : state->grad_m) {
        names.push_back(name);
    }

    std::sort(names.begin(), names.end());

    return names;
}

// The m or v of one parameter, or nullptr.
ggml_tensor * find_moment(ll_train_state * state, const char * param_name, bool is_v) {
    const auto & map = is_v ? state->grad_v : state->grad_m;
    const auto it = map.find(std::string(param_name));
    return it == map.end() ? nullptr : it->second;
}

} // namespace

int32_t ll_opt_state_count(llama_context * ctx) {
    ll_train_state * state = ctx ? state_for(ctx) : nullptr;
    if (state == nullptr) {
        return LL_ERR_NOT_INITIALIZED;
    }
    if (state->grad_m.empty()) {
        // The moments do not exist until ggml-opt has built an optimizer graph, which happens on
        // the first TRAINING step. Checkpointing before then is not "an empty checkpoint", it is a
        // question with no answer yet.
        return LL_ERR_NOT_INITIALIZED;
    }

    return (int32_t)(2 * state->grad_m.size()); // an m and a v for each parameter
}

int32_t ll_opt_state_info(llama_context * ctx, int32_t index, char * name_out, int32_t name_capacity, bool * is_v,
                          int64_t * n_elements) {
    if (ctx == nullptr || name_out == nullptr || is_v == nullptr || n_elements == nullptr) {
        return LL_ERR_INVALID_ARG;
    }

    ll_train_state * state = state_for(ctx);
    if (state == nullptr || state->grad_m.empty()) {
        return LL_ERR_NOT_INITIALIZED;
    }

    const std::vector<std::string> names = sorted_param_names(state);

    // Laid out m-then-v per parameter, so index/2 is the parameter and index%2 the role.
    const size_t param = (size_t)index / 2;
    const bool want_v = (index % 2) == 1;

    if (index < 0 || param >= names.size()) {
        return LL_ERR_INVALID_ARG;
    }

    const std::string & name = names[param];

    if ((int32_t)name.size() + 1 > name_capacity) {
        return LL_ERR_INVALID_ARG;
    }

    std::memcpy(name_out, name.c_str(), name.size() + 1);

    ggml_tensor * tensor = find_moment(state, name.c_str(), want_v);
    if (tensor == nullptr) {
        return LL_ERR_NOT_INITIALIZED;
    }

    *is_v = want_v;
    *n_elements = ggml_nelements(tensor);

    return LL_OK;
}

int64_t ll_opt_state_get(llama_context * ctx, const char * param_name, bool is_v, float * out, int64_t n_max) {
    if (ctx == nullptr || param_name == nullptr || n_max < 0 || (out == nullptr && n_max != 0)) {
        return LL_ERR_INVALID_ARG;
    }

    ll_train_state * state = state_for(ctx);
    if (state == nullptr || state->grad_m.empty()) {
        return LL_ERR_NOT_INITIALIZED;
    }

    ggml_tensor * tensor = find_moment(state, param_name, is_v);
    if (tensor == nullptr) {
        return LL_ERR_INVALID_ARG;
    }

    const int64_t n = ggml_nelements(tensor);

    if (out == nullptr) {
        return n; // the sizing call
    }
    if (n > n_max) {
        return LL_ERR_INVALID_ARG;
    }

    ggml_backend_tensor_get(tensor, out, 0, n * sizeof(float));

    return n;
}

int64_t ll_opt_state_set(llama_context * ctx, const char * param_name, bool is_v, const float * data, int64_t n) {
    if (ctx == nullptr || param_name == nullptr || data == nullptr || n < 0) {
        return LL_ERR_INVALID_ARG;
    }

    ll_train_state * state = state_for(ctx);
    if (state == nullptr || state->grad_m.empty()) {
        return LL_ERR_NOT_INITIALIZED;
    }

    ggml_tensor * tensor = find_moment(state, param_name, is_v);
    if (tensor == nullptr) {
        return LL_ERR_INVALID_ARG;
    }

    // The shape must match EXACTLY. A checkpoint whose rank differs from the adapter it is being
    // restored into is not a checkpoint of this run, and quietly restoring the overlap would give
    // an optimizer state that is part one run and part another.
    if (ggml_nelements(tensor) != n) {
        LLAMA_LOG_ERROR("%s: %s has %" PRId64 " elements but the checkpoint has %" PRId64
                        ". The checkpoint is not of this adapter.\n",
                        __func__, param_name, ggml_nelements(tensor), n);
        return LL_ERR_SHAPE_MISMATCH;
    }

    ggml_backend_tensor_set(tensor, data, 0, n * sizeof(float));

    return n;
}

int64_t ll_opt_get_iter(llama_context * ctx) {
    ll_train_state * state = ctx ? state_for(ctx) : nullptr;
    if (state == nullptr || state->opt_ctx == nullptr) {
        return LL_ERR_NOT_INITIALIZED;
    }

    return ggml_opt_get_iter(state->opt_ctx);
}

int32_t ll_opt_set_iter(llama_context * ctx, int64_t iter) {
    if (iter < 1) {
        // ggml asserts this, and the reason is worth knowing: iteration 0 would make AdamW's bias
        // correction divide by 1 - beta^0 == 0.
        return LL_ERR_INVALID_ARG;
    }

    ll_train_state * state = ctx ? state_for(ctx) : nullptr;
    if (state == nullptr || state->opt_ctx == nullptr) {
        return LL_ERR_NOT_INITIALIZED;
    }

    ggml_opt_set_iter(state->opt_ctx, iter);

    return LL_OK;
}

// ---------------------------------------------------------------------------
// Trainability preflight (S1-11) -- the entry point; the walker is farm_preflight.cpp
// ---------------------------------------------------------------------------

namespace {

// What the preflight callback needs to get its results back out.
struct preflight_ctx {
    ll_train_state * state = nullptr;

    ll_preflight_entry * out = nullptr;
    int32_t max_entries = 0;

    int32_t n_entries = 0;
    int32_t n_blocked = 0;
};

// Runs at the exact moment the forward graph is complete and nothing has been executed.
//
// llama_context::opt_step_custom hands the loss builder the finished forward graph so it can append
// a loss to it. That is also the only place anything outside the vendored tree ever SEES that graph
// -- so it is where the walk happens. The loss returned is a throwaway scalar; with train=false no
// backward is built and nothing is optimized.
ggml_tensor * build_preflight(ggml_context * ctx_compute, ggml_cgraph * gf, ggml_tensor * logits, int32_t pos,
                              int32_t n_ubatch, void * userdata) {
    (void)pos;
    (void)n_ubatch;

    auto * pc = (preflight_ctx *)userdata;

    if (pc->n_entries == 0 && pc->n_blocked == 0) { // first ubatch only; they are all the same shape
        const int32_t written =
            ll_preflight_walk(gf, pc->state->param_tensors.data(), (int32_t)pc->state->param_tensors.size(), pc->out,
                              pc->max_entries, &pc->n_blocked);

        pc->n_entries = written > 0 ? written : 0;

        // The bypass warning, and it is the one nothing else catches.
        //
        // An adapter tensor that appears in NO node of the graph means its target projection never
        // went through build_lora_mm -- the architecture builds that matmul some other way. The
        // tensor is flagged trainable, ggml dutifully allocates a gradient for it, and the gradient
        // is always zero. So it sits at its initial value forever while the loss falls perfectly
        // well, because the OTHER adapter tensors are learning. Nothing fails. You just trained a
        // smaller adapter than you asked for.
        std::set<const ggml_tensor *> in_graph;
        for (int i = 0; i < ggml_graph_n_nodes(gf); ++i) {
            ggml_tensor * node = ggml_graph_node(gf, i);
            in_graph.insert(node);
            for (int s = 0; s < GGML_MAX_SRC; ++s) {
                if (node->src[s]) {
                    in_graph.insert(node->src[s]);
                }
            }
        }

        for (ggml_tensor * t : pc->state->param_tensors) {
            if (in_graph.count(t)) {
                continue;
            }

            if (pc->out != nullptr && pc->n_entries < pc->max_entries) {
                ll_preflight_entry & e = pc->out[pc->n_entries++];

                std::snprintf(e.node, sizeof(e.node), "%s", ggml_get_name(t));
                std::snprintf(e.op, sizeof(e.op), "%s", "LORA");
                e.status = LL_PREFLIGHT_WARN;
                std::snprintf(e.detail, sizeof(e.detail),
                              "this adapter tensor is in no graph node: its target projection does "
                              "not go through build_lora_mm on this architecture. Its gradient will "
                              "be zero forever, and the loss will fall anyway -- on the other "
                              "tensors. You would be training a smaller adapter than you asked for.");
            }
        }
    }

    // A throwaway scalar. With train=false nothing is differentiated, so it is never used for
    // anything; ggml-opt just needs a loss node to exist.
    ggml_tensor * loss = ggml_sum(ctx_compute, logits);
    ggml_set_name(loss, "ll_preflight_loss");
    ggml_build_forward_expand(gf, loss);

    return loss;
}

void preflight_no_inputs(void * userdata) {
    (void)userdata; // the preflight loss has no inputs to fill
}

} // namespace

int32_t ll_preflight(llama_context * ctx, const int32_t * tokens, int32_t n_tokens, ll_preflight_entry * out,
                     int32_t max_entries, int32_t * n_blocked) {
    if (ctx == nullptr || tokens == nullptr || n_tokens <= 0 || n_blocked == nullptr) {
        return LL_ERR_INVALID_ARG;
    }

    ll_train_state * state = state_for(ctx);
    if (state == nullptr) {
        // The walk is seeded from the trainable tensors, so there has to be a set of them.
        return LL_ERR_NOT_INITIALIZED;
    }

    llama_batch batch = llama_batch_init(n_tokens, 0, 1);
    batch.n_tokens = n_tokens;
    for (int32_t i = 0; i < n_tokens; ++i) {
        batch.token[i] = tokens[i];
        batch.pos[i] = i;
        batch.n_seq_id[i] = 1;
        batch.seq_id[i][0] = 0;
        batch.logits[i] = true;
    }

    preflight_ctx pc;
    pc.state = state;
    pc.out = out;
    pc.max_entries = max_entries;

    ggml_opt_result_t result = ggml_opt_result_init();

    // train=false: build the forward graph, run it, build no backward. The walk happens at build
    // time, so the forward's result is discarded -- but running it does prove the graph is
    // SCHEDULABLE, which a pure graph build would not.
    const int32_t status = ctx->opt_step_custom(batch, state->opt_ctx, result, build_preflight, preflight_no_inputs,
                                                &pc, /*train =*/false);

    ggml_opt_result_free(result);
    llama_batch_free(batch);

    if (status != 0) {
        return LL_ERR_STEP_FAILED;
    }

    *n_blocked = pc.n_blocked;

    return pc.n_entries;
}
