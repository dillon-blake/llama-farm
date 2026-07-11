---
id: B-09
title: "Full-FT extras: MoE E8 + SSM S5 (non-LoRA parameter gradients)"
stage: backlog
track: kernels
size: L
deps: [S1-27, S1-31]
status: open
pr: null
---

# B-09 — Full-FT extras: MoE E8 + SSM S5 (non-LoRA parameter gradients)

**One-line outcome:** **DEFERRED** — the gradient paths LoRA training never needs:
MoE per-expert bias grads / router full training (ROADMAP §9 E8) and SSM
dA/dD/conv-weight/dt_bias grads plus cross-ubatch BPTT (ROADMAP §10 S5).

**Activation trigger:** full fine-tuning (training non-adapter parameters) enters
product scope. This project's target configuration — frozen quantized base, F32
LoRA A/B as the only trainable parameters — never exercises these paths, and
quantized-expert full FT is an explicit non-goal (ROADMAP §9 E8).

## Why (context)

Stage-1 MoE and SSM tickets (S1-25/26/27, S1-29/30/31) scoped their backward work
to what LoRA training requires: activation gradients *through* frozen ops and
weight gradients for the LoRA operands only. The remainders were deferred by
design, and this stub is their named owner so the deferral pointers in those
tickets do not dangle:

- **MoE E8:** per-expert bias grads (scatter-add over expert assignment), router
  full training beyond the top-k-weights path, quantized-expert full FT
  (permanent non-goal — would require quantized weight updates).
- **SSM S5:** parameter grads for A, D, the conv weight, and dt_bias; cross-ubatch
  BPTT is out of scope by the documented truncation-at-ubatch semantics (S1-31) —
  revisiting that truncation belongs here too.

## What to do

Split into concrete tickets at activation (one per op family, per the S1-25/S1-29
op inventory); this stub tracks scope, not a single PR. Reverse-scan parameter
grads reuse the S1-31 chunk-recompute skeleton; bias scatter-adds must honor the
gate G-B determinism default (segmented accumulation, atomics opt-in).

## Out of scope

- Anything LoRA training needs — already owned by stage 1–4 tickets.
- Embedding/lm_head training — B-05.
- Linear-attention family (RWKV/delta-net) — B-06.

## Acceptance criteria

- [ ] At activation: sub-tickets filed covering the affected ops, each with
      MODE_GRAD acceptance vs the CPU oracle per ADR-0002.
- [ ] Until then: this stub is the valid deferral target referenced by
      S1-25/S1-27/S1-29 (and the backend MoE/SSM port tickets).

## Testing & verification

Defined per sub-ticket at activation; the harness is the vendored
`tests/test-backend-ops` MODE_GRAD suite plus e2e convergence variants.

## PR notes

- Branch (at activation, per sub-ticket): `ticket/B-09x-<slug>`.
- Two-repo flow per S0-02; upstreaming disposition: **upstream-later**, riding the
  MoE/SSM op-family RFCs (ROADMAP §11 triage b).
