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

    // The gradient accumulator of each param tensor, keyed by the tensor's name.
    //
    // Captured once, from inside the first training step, because that is the only window in
    // which ggml_opt will tell us: with dynamic graphs, ggml_opt_eval nulls the graphs it looked
    // them up in. The tensors themselves persist in opt_ctx's static context.
    std::unordered_map<std::string, ggml_tensor *> grad_accs;

    // The n_tokens of the first step. Every later step must match it -- see ll_train_step.
    int32_t n_tokens_seen = 0;

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
        ggml_tensor * grad = ggml_opt_grad_acc(state->opt_ctx, t);
        if (grad != nullptr) {
            state->grad_accs[std::string(ggml_get_name(t))] = grad;
        }
    }
}

// Everything the loss builder needs, threaded through llama_context::opt_step_custom as userdata.
struct loss_ctx {
    ll_train_state * state = nullptr; // so the alloc-time hook can capture the gradient accumulators
    bool train = false;

    const int32_t * targets = nullptr; // [n_tokens] target id per position
    const float * weights = nullptr;   // [n_tokens] loss weight per position; 0 masks out
    int32_t n_tokens = 0;

    // Set by build_masked_ce, filled by upload_masked_ce once ggml_opt_alloc has given it memory.
    ggml_tensor * labels = nullptr;    // I32 [n_ubatch] -- the target token of each position
    ggml_tensor * ce_weights = nullptr; // F32 [n_ubatch] -- its loss weight, pre-normalized
    int32_t pos = 0;                   // this ubatch's offset within the batch
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

    // logit_scale = 1.0 (off), softcap = 0.0 (off): a plain llama arch applies neither. Gemma-2
    // and friends do, and S1-11 wires them through -- the op already takes them.
    ggml_tensor * per_token =
        ggml_cross_entropy_loss_sparse(ctx_compute, logits, lc->labels, lc->ce_weights, 1.0f, 0.0f);
    ggml_set_name(per_token, "ll_ce_per_token");

    ggml_tensor * loss = ggml_sum(ctx_compute, per_token);
    ggml_set_name(loss, "ll_ce_loss");

    ggml_build_forward_expand(gf, loss);

    return loss;
}

// Fill the targets and weights. Runs after ggml_opt_alloc, because until then they have no memory.
//
// It is also the only moment at which ggml_opt will hand over the gradient accumulators, so the
// shim grabs them here on the first training step. See capture_grad_accs.
void upload_masked_ce(void * userdata) {
    auto * lc = (loss_ctx *)userdata;

    if (lc->train) {
        capture_grad_accs(lc->state);
    }

    const int64_t n_vocab = lc->n_vocab;
    const int32_t n_ubatch = lc->n_ubatch;

    // Normalize by the weight actually carried, not by the token count: a batch that is 90% prompt
    // must not have its loss (and so its gradient) scaled down 10x relative to one that is 10%
    // prompt. Folding 1/sum(w) into the weights here keeps the graph a plain sum. A batch with no
    // unmasked token at all has zero loss and zero gradient, which is the honest answer.
    float sum_w = 0.0f;
    for (int32_t i = 0; i < n_ubatch; ++i) {
        sum_w += lc->weights[lc->pos + i];
    }
    const float scale = sum_w > 0.0f ? 1.0f / sum_w : 0.0f;

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
}

} // namespace

int32_t ll_train_step(llama_context * ctx, const int32_t * tokens, const int32_t * targets, const float * weights,
                      const int32_t * seq_ids, const int32_t * positions, int32_t n_tokens, bool train,
                      float * loss_out) {
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
    if ((uint32_t) n_tokens > ctx->n_ubatch()) {
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
    if (state->n_tokens_seen == 0) {
        state->n_tokens_seen = n_tokens;
    } else if (state->n_tokens_seen != n_tokens) {
        LLAMA_LOG_ERROR("%s: n_tokens changed from %d to %d. Every step must have the same shape: "
                        "ggml-opt indexes its optimizer state by graph node index, and sizes it "
                        "from the first graph it sees. Pad the batch instead (weight 0).\n",
                        __func__, state->n_tokens_seen, n_tokens);
        return LL_ERR_SHAPE_MISMATCH;
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

        const int32_t n_seq_max = (int32_t) ctx->n_seq_max();
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

int64_t ll_debug_get_tensor(llama_context * ctx, const char * base_name, bool is_b, float * out,
                            int64_t n_max) {
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

    ggml_backend_tensor_get(t, out, 0, n*sizeof(float));
    return n;
}

int64_t ll_debug_set_tensor(llama_context * ctx, const char * base_name, bool is_b,
                            const float * data, int64_t n) {
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

    ggml_backend_tensor_set(t, data, 0, n*sizeof(float));
    return n;
}

int64_t ll_debug_grad(llama_context * ctx, const char * base_name, bool is_b, float * out,
                      int64_t n_max) {
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

    ggml_backend_tensor_get(grad, out, 0, n*sizeof(float));
    return n;
}
