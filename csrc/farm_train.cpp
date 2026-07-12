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
    opt_params.opt_period = (int32_t)(ctx->n_batch() / ctx->n_ubatch());
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

// Everything the loss builder needs, threaded through llama_context::opt_step_custom as userdata.
struct loss_ctx {
    const int32_t * targets = nullptr; // [n_tokens] target id per position
    const float * weights = nullptr;   // [n_tokens] loss weight per position; 0 masks out
    int32_t n_tokens = 0;

    // Set by build_masked_ce, filled by upload_masked_ce once ggml_opt_alloc has given it memory.
    ggml_tensor * labels = nullptr; // [n_vocab, n_ubatch] weighted one-hot
    int32_t pos = 0;                // this ubatch's offset within the batch
    int32_t n_ubatch = 0;
    int64_t n_vocab = 0;
};

// Build  L = -sum_i w_i * logp_i[target_i] / sum_i w_i  out of the ops that exist today.
//
// ggml_cross_entropy_loss(logits, labels) computes  -(1/nr) * sum_ij labels_ij * log_softmax(logits)_ij.
// It is documented as taking a one-hot label matrix, but nothing requires the entries to be 1:
// putting w_i at row i's target column makes it compute exactly
//
//     -(1/nr) * sum_i w_i * logp_i[target_i]
//
// which is the masked loss, up to the 1/nr where nr counts ALL rows rather than the unmasked
// ones. That is corrected on the host by pre-scaling the weights by nr / sum(w) -- see
// upload_masked_ce. So no new op is needed for a masked loss.
//
// The cost is the dense [n_vocab, n_ubatch] label matrix, which is pure waste: every row is zero
// except one entry. That is precisely what S1-04's sparse cross-entropy removes, and it is why
// this is called a stopgap.
ggml_tensor * build_masked_ce(ggml_context * ctx_compute, ggml_cgraph * gf, ggml_tensor * logits, int32_t pos,
                              int32_t n_ubatch, void * userdata) {
    auto * lc = (loss_ctx *)userdata;

    lc->pos = pos;
    lc->n_ubatch = n_ubatch;
    lc->n_vocab = logits->ne[0];

    lc->labels = ggml_new_tensor_2d(ctx_compute, GGML_TYPE_F32, logits->ne[0], n_ubatch);
    ggml_set_input(lc->labels);
    ggml_set_name(lc->labels, "ll_masked_ce_labels");

    ggml_tensor * loss = ggml_cross_entropy_loss(ctx_compute, logits, lc->labels);
    ggml_set_name(loss, "ll_masked_ce_loss");

    ggml_build_forward_expand(gf, loss);

    return loss;
}

// Fill the label matrix. Runs after ggml_opt_alloc, because until then `labels` has no memory.
void upload_masked_ce(void * userdata) {
    auto * lc = (loss_ctx *)userdata;

    const int64_t n_vocab = lc->n_vocab;
    const int32_t n_ubatch = lc->n_ubatch;

    // ggml_cross_entropy_loss divides by the row count -- ALL rows, including the masked ones.
    // We want the mean over the tokens that count. Pre-scale the weights by n_rows / sum(w) so
    // the two cancel. If every weight is zero the loss is zero and there is nothing to scale.
    float sum_w = 0.0f;
    for (int32_t i = 0; i < n_ubatch; ++i) {
        sum_w += lc->weights[lc->pos + i];
    }
    const float scale = sum_w > 0.0f ? (float)n_ubatch / sum_w : 0.0f;

    std::vector<float> labels((size_t)n_vocab * n_ubatch, 0.0f);
    for (int32_t i = 0; i < n_ubatch; ++i) {
        const int32_t target = lc->targets[lc->pos + i];
        if (target < 0 || target >= n_vocab) {
            continue; // out-of-range target: treat as masked rather than corrupt memory
        }
        labels[(size_t)i * n_vocab + target] = lc->weights[lc->pos + i] * scale;
    }

    ggml_backend_tensor_set(lc->labels, labels.data(), 0, labels.size() * sizeof(float));
}

} // namespace

int32_t ll_train_step(llama_context * ctx, const int32_t * tokens, const int32_t * targets, const float * weights,
                      int32_t n_tokens, bool train, float * loss_out) {
    if (ctx == nullptr || tokens == nullptr || targets == nullptr || weights == nullptr || n_tokens <= 0) {
        return LL_ERR_INVALID_ARG;
    }

    const auto it = train_states().find(ctx);
    if (it == train_states().end()) {
        return LL_ERR_NOT_INITIALIZED;
    }
    ll_train_state * state = it->second.get();

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
        batch.pos[i] = i;
        batch.n_seq_id[i] = 1;
        batch.seq_id[i][0] = 0;
        batch.logits[i] = true; // every position needs logits: every one may carry loss
    }

    loss_ctx lc;
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
