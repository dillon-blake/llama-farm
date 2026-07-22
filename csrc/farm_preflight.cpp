// Will this model train, and if not, what exactly stops it? (S1-11)
//
// Without this, the answer to "can I LoRA-tune Mixtral?" is: try it, and watch ggml abort inside
// ggml_compute_backward with a message naming an op enum and nothing else -- not the tensor, not
// the layer, not what to do about it. And that abort only happens on the first backward pass, i.e.
// after the model has loaded, the data has tokenized, and the user has waited.
//
// So: walk the forward graph before training, work out which nodes would actually NEED a gradient,
// and check each of those against the set of ops ggml can differentiate. Nodes off the gradient
// path are never blockers, which matters -- a model is full of ops with no backward rule that the
// backward pass never reaches, and reporting them would be noise that trains people to ignore this.
//
// The "needs a gradient" propagation mirrors ggml_build_backward_expand: seed the flagged
// parameters, then a node needs a gradient if any of its sources does. Integer tensors never do
// (ggml.c skips them), which is why token ids and position indices do not drag half the graph onto
// the gradient path.

#include "farm_api.h"

#include "llama-context.h"

#include "ggml.h"

#include <cstring>
#include <set>
#include <string>
#include <unordered_set>
#include <vector>

namespace {

// The ops ggml_compute_backward can differentiate, read off its switch rather than remembered.
//
// If this drifts from ggml.c the preflight lies in the most damaging direction -- it says a model
// will train and then the abort happens anyway -- so it is worth re-deriving on a vendor bump:
//
//   awk '/^static void ggml_compute_backward/,/^}$/' ggml/src/ggml.c | grep -oE 'case GGML_OP_[A-Z_0-9]+'
//
// And it no longer relies on anyone remembering to: tests/test_preflight_freshness.py parses that
// same switch out of the vendored source and fails if this table disagrees with it in EITHER
// direction (S1-42) -- a case gained upstream that is missing here (blocks a model that would
// train), or a case listed here that upstream dropped (promises a run that then aborts).
bool op_has_backward(ggml_op op) {
    switch (op) {
    case GGML_OP_NONE:
    case GGML_OP_DUP:
    case GGML_OP_ADD:
    case GGML_OP_ADD1:
    case GGML_OP_ACC:
    case GGML_OP_SUB:
    case GGML_OP_MUL:
    case GGML_OP_DIV:
    case GGML_OP_SQR:
    case GGML_OP_SQRT:
    case GGML_OP_LOG:
    case GGML_OP_SIN:
    case GGML_OP_COS:
    case GGML_OP_SUM:
    case GGML_OP_SUM_ROWS:
    case GGML_OP_MEAN:
    case GGML_OP_REPEAT:
    case GGML_OP_REPEAT_BACK:
    case GGML_OP_RMS_NORM:
    case GGML_OP_MUL_MAT:
    case GGML_OP_SCALE:
    case GGML_OP_SET:
    case GGML_OP_CPY:
    case GGML_OP_CONT:
    case GGML_OP_RESHAPE:
    case GGML_OP_VIEW:
    case GGML_OP_PERMUTE:
    case GGML_OP_TRANSPOSE:
    case GGML_OP_GET_ROWS:
    case GGML_OP_DIAG_MASK_INF:
    case GGML_OP_DIAG_MASK_ZERO:
    case GGML_OP_SOFT_MAX:
    case GGML_OP_ROPE:
    case GGML_OP_IM2COL:
    case GGML_OP_POOL_2D:
    case GGML_OP_WIN_PART:
    case GGML_OP_WIN_UNPART:
    case GGML_OP_CONCAT: // S1-29
    case GGML_OP_UNARY:
    case GGML_OP_CLAMP:
    case GGML_OP_CROSS_ENTROPY_LOSS:
    case GGML_OP_CROSS_ENTROPY_LOSS_SPARSE:
    case GGML_OP_GLU:
    // learning-llamas: MoE (S1-25 wiring + S1-26/S1-27 kernels). MUL_MAT_ID is the expert matmul
    // and ADD_ID the per-expert bias; both have backwards now, which is what makes a Mixtral
    // trainable. The OUT_PROD_ID* ops are what MUL_MAT_ID's backward EMITS -- they never appear on
    // a forward graph, but listing them costs nothing and stops a future graph-shape surprise from
    // reading as "no backward rule".
    case GGML_OP_MUL_MAT_ID:
    case GGML_OP_ADD_ID:
    case GGML_OP_OUT_PROD_ID:
    case GGML_OP_OUT_PROD_ID_GRP:
    case GGML_OP_GLU_BACK:
    // learning-llamas: Mamba / state-space (S1-29b..S1-31). ggml_compute_backward now differentiates
    // both the causal conv (SSM_CONV -> ssm_conv_back) and the selective scan (SSM_SCAN ->
    // ssm_scan_back), so a Mamba adapter's gradient path no longer aborts here. These belonged on the
    // BLOCKED side until the fork landed those VJPs, and nothing flipped them when it did -- the exact
    // silent-drift the freshness guard now forbids. B-10 then lifted the scan's n_group == 1 refusal
    // -- SSM_SCAN's backward differentiates Mamba-2 / Falcon-H1 group routing too, given only that
    // n_head divides evenly among the groups -- so the remaining scope limit is the A-matrix and ids
    // gradients, which that case still asserts on. That is a numerical scope question rather than an
    // abort risk either way: the preflight only predicts whether the backward pass COMPLETES.
    case GGML_OP_SSM_CONV:
    case GGML_OP_SSM_SCAN:
        return true;
    default:
        return false;
    }
}

// GLU is a family, not an op, and only ONE member of it has a backward.
//
// ggml_compute_backward's GLU case implements split SWIGLU and nothing else: a fused SWIGLU trips
// `GGML_ASSERT(src1 && "backward pass only implemented for split swiglu")`, and REGLU / GEGLU /
// GEGLU_ERF / GEGLU_QUICK / SWIGLU_OAI hit `GGML_ABORT("unsupported glu op for backward pass")`.
//
// Saying `case GGML_OP_GLU: return true;` therefore told every Gemma (GEGLU) and gpt-oss
// (SWIGLU_OAI) user that their model trains -- and then aborted in the backward build, which is
// exactly the outcome this preflight exists to predict. The whole point of the report is that it
// is believed. S1-28 adds the missing VJPs and flips these on.
bool glu_has_backward(const ggml_tensor * node) {
    // S1-28 gave the whole family a VJP (GGML_OP_GLU_BACK), including the fused form and the
    // GEGLU/REGLU/SWIGLU_OAI variants that used to GGML_ABORT. Gemma and gpt-oss train now.
    switch (ggml_get_glu_op(node)) {
    case GGML_GLU_OP_REGLU:
    case GGML_GLU_OP_GEGLU:
    case GGML_GLU_OP_SWIGLU:
    case GGML_GLU_OP_SWIGLU_OAI:
    case GGML_GLU_OP_GEGLU_ERF:
    case GGML_GLU_OP_GEGLU_QUICK:
        return true;
    default:
        return false;
    }
}

// The unary sub-switch has its own coverage, and its default aborts just like the outer one.
bool unary_has_backward(ggml_unary_op op) {
    switch (op) {
    case GGML_UNARY_OP_ABS:
    case GGML_UNARY_OP_SGN:
    case GGML_UNARY_OP_NEG:
    case GGML_UNARY_OP_STEP:
    case GGML_UNARY_OP_RELU:
    case GGML_UNARY_OP_SILU:
    case GGML_UNARY_OP_EXP:
    case GGML_UNARY_OP_EXPM1:
    case GGML_UNARY_OP_SOFTPLUS:
    case GGML_UNARY_OP_TANH:
    case GGML_UNARY_OP_SIGMOID:
        return true;
    default:
        return false;
    }
}

// What unblocks each op we know about. A message that says only "no backward rule" tells a user
// their model does not train; one that names the ticket tells them whether to wait or to give up.
const char * blocker_detail(ggml_op op) {
    switch (op) {
    case GGML_OP_FLASH_ATTN_EXT:
        return "flash attention has no backward. Training mode disables it "
               "(llama_context::set_training), so seeing it here means something re-enabled it. "
               "Unblocked by S1-21..S1-24.";
    case GGML_OP_GLU:
        return "only SPLIT SwiGLU has a backward. Fused SwiGLU, and the GEGLU / REGLU / "
               "SWIGLU_OAI variants (Gemma, gpt-oss), have none. Unblocked by S1-28.";
    case GGML_OP_OUT_PROD:
        return "OUT_PROD is a backward-only op; seeing it on the forward graph is unexpected.";
    // SSM_CONV / SSM_SCAN are no longer here on purpose: S1-30/S1-31 gave them backwards, op_has_backward
    // returns true for both, and so neither can reach this table. A message claiming "no backward yet"
    // would now be a lie -- exactly the kind the freshness guard exists to prevent.
    case GGML_OP_ARGSORT:
    case GGML_OP_TOP_K:
    case GGML_OP_ARGMAX:
        return "this op is not differentiable, by nature. If it is on the gradient path, the "
               "graph is doing something unexpected.";
    default:
        return "ggml_compute_backward has no rule for this op, so the backward pass would "
               "abort here.";
    }
}

// I32/I64 tensors never carry gradients -- ggml_build_backward_expand skips them (ggml.c) -- which
// is what keeps token ids and position indices from dragging the whole graph onto the gradient path.
bool can_carry_grad(const ggml_tensor * t) {
    return t->type == GGML_TYPE_F32 || t->type == GGML_TYPE_F16 || t->type == GGML_TYPE_BF16;
}

} // namespace

int32_t ll_preflight_walk(ggml_cgraph * gf, ggml_tensor ** params, int32_t n_params, ll_preflight_entry * out,
                          int32_t max_entries, int32_t * n_blocked_out) {
    if (gf == nullptr || (params == nullptr && n_params > 0) || n_blocked_out == nullptr) {
        return LL_ERR_INVALID_ARG;
    }

    // Seed: the trainable tensors themselves.
    std::unordered_set<const ggml_tensor *> needs_grad;
    for (int32_t i = 0; i < n_params; ++i) {
        if (params[i] != nullptr) {
            needs_grad.insert(params[i]);
        }
    }

    int32_t n_entries = 0;
    int32_t n_blocked = 0;

    // One forward pass over the nodes is enough: ggml graphs are topologically ordered, so every
    // source of a node appears before it.
    for (int i = 0; i < ggml_graph_n_nodes(gf); ++i) {
        ggml_tensor * node = ggml_graph_node(gf, i);

        bool on_grad_path = false;
        for (int s = 0; s < GGML_MAX_SRC; ++s) {
            const ggml_tensor * src = node->src[s];
            if (src != nullptr && can_carry_grad(src) && needs_grad.count(src)) {
                on_grad_path = true;
                break;
            }
        }

        if (!on_grad_path) {
            continue; // off the gradient path: the backward never reaches it, so it cannot block
        }

        if (!can_carry_grad(node)) {
            // An I32 output TERMINATES the gradient path -- ggml_build_backward_expand skips such
            // nodes outright (`if (node->type == GGML_TYPE_I32) continue;`), so it never asks for
            // their VJP and they cannot block anything.
            //
            // Without this, a MoE model reported ARGSORT as a blocker: its source (the router
            // probabilities) is on the gradient path, so the node looked reachable -- but its
            // output is the I32 expert selection, which is never differentiated. The preflight was
            // right that argsort has no gradient and wrong that it mattered, which is the failure
            // mode that trains people to ignore a report.
            continue;
        }

        needs_grad.insert(node);

        const bool supported = op_has_backward(node->op) &&
                               (node->op != GGML_OP_UNARY || unary_has_backward(ggml_get_unary_op(node))) &&
                               (node->op != GGML_OP_GLU || glu_has_backward(node));

        if (supported) {
            continue;
        }

        ++n_blocked;

        if (out != nullptr && n_entries < max_entries) {
            ll_preflight_entry & e = out[n_entries++];

            std::snprintf(e.node, sizeof(e.node), "%s", ggml_get_name(node));

            const char * op_name =
                node->op == GGML_OP_UNARY ? ggml_unary_op_name(ggml_get_unary_op(node)) : ggml_op_name(node->op);
            std::snprintf(e.op, sizeof(e.op), "%s", op_name);

            e.status = LL_PREFLIGHT_BLOCKED;
            std::snprintf(e.detail, sizeof(e.detail), "%s", blocker_detail(node->op));
        }
    }

    *n_blocked_out = n_blocked;

    return n_entries;
}
