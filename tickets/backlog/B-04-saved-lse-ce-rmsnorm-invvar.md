---
id: B-04
title: "Saved-LSE CE exploitation + RMS_NORM_BACK saved-inv-var variant"
stage: backlog
track: kernels
size: M
deps: [S1-04]
status: open
pr: null
---

# B-04 — Saved-LSE CE exploitation + RMS_NORM_BACK saved-inv-var variant

**One-line outcome:** **DEFERRED** — CE backward consumes the per-row `lse` reserved in the gate
G-A ABI instead of re-reducing the vocab row; optionally RMS_NORM saves one F32 `inv_var` per row
for a pure two-load backward.

**Activation trigger:** a milestone profile shows CE-backward vocab re-reduction or
RMS_NORM_BACK's statistic recomputation as a measurable backward share, **and** the
microbenchmark mandated by ROADMAP §13 item 6 confirms the win — microbenchmark before committing
any ABI change.

## Why (context)

S1-04 settled the sparse-CE cross-backend ABI at decide-first gate G-A (recorded as ADR-0003):
whether the forward stashes the per-row `lse` and whether backward may alias the logits buffer.
ROADMAP §11 K5 assumes the ABI slot was **reserved** in K0 even if unexploited. Two cases: if
ADR-0003 chose the stash, wire every backend's CE backward to consume it, halving vocab-row reads;
if it chose recompute, this ticket **re-opens the decision with data** before touching any
implementation. Unsloth's Apache-2.0 CE kernel is the design reference (math only): a
logsumexp-only forward saving one F32 per row with the chunked-lse decomposition for wide vocabs
(`unsloth/kernels/cross_entropy_loss.py:87-150`), and a backward
`dloss·w·(exp(x−lse) − onehot)` consuming the stash with in-place logit grads
(`cross_entropy_loss.py:260-276`).

The second half applies the same idea to RMS_NORM: today's backward recomputes the row statistic
from the saved forward *input* (e.g. CUDA `rms_norm_back_f32`,
`vendor/llama.cpp/ggml/src/ggml-cuda/norm.cu:158`). Stashing one F32 `inv_var` per row makes the
backward a pure two-load formula; the worked math, including the Gemma `W+1` convention, is on
unsloth's Apache side (`unsloth/kernels/rms_layernorm.py:51-111` — `inv_var` stashed at `:55`;
their self-test uses the same 0.05 threshold as ADR-0002, `rms_layernorm.py:326`). Optional
op-ABI change, same measure-first bar (ROADMAP §13 items 1/6).

## What to do

1. Microbenchmark both candidates at realistic shapes (128k-vocab CE rows; `n_embd`-wide RMS_NORM
   rows) per backend; commit the report; proceed per item only where the win is material.
2. **CE:** if ADR-0003 chose the stash — implement consumption in the CPU oracle first (S1-04
   kernel; row-structure pattern `vendor/llama.cpp/ggml/src/ggml-cpu/ops.cpp:11158`), then the
   S2-07/S3-03/S4-04 ports; if it chose recompute — amend ADR-0003 with the new data first.
3. **RMS_NORM saved-inv-var:** optional extra forward output + a backward variant consuming it,
   CPU first, then CUDA (`norm.cu:158`), Vulkan (`rms_norm_back.comp`), Metal (S2-03); honor the
   Gemma `W+1` convention (math reference `rms_layernorm.py:51-111`).
4. MODE_GRAD parity for both ops on all four backends; verify the graph allocator handles the
   extra outputs (the G-A aliasing prototype from S1-04 is the precedent).
5. Enable per backend only where the microbenchmark won; record the ABI deltas as an ADR-0003
   amendment.

## Out of scope

- LayerNorm (`NORM`) backward and its mean+inv_var stash — ROADMAP §13 item 7, owned by B-08.
- Chunked lm_head/CE host patterns — S1-13.
- Any change to CE loss semantics, masking, or softcap/scale op-params (fixed by ADR-0003).

## Acceptance criteria

- [ ] Microbenchmark report in `benches/` (or ticket closed with numbers showing no material
      win — a valid outcome).
- [ ] `test-backend-ops` MODE_GRAD passes for sparse CE and RMS_NORM(_BACK) with saved-stat paths
      forced, vs the CPU oracle, within the ADR-0002 tolerance, on all four backends.
- [ ] Paired test: saved-stat vs recompute paths agree within ADR-0002 per-op tolerance.
- [ ] ADR-0003 amendment (or confirmation record) committed with the implementation.
- [ ] All four CI lanes green on the targeted sweeps.

## Testing & verification

Vendored `tests/test-backend-ops` MODE_GRAD vs the CPU oracle (ADR-0002) for both ops: targeted
per-PR on all four lanes, full sweeps nightly. Microbenchmarks in `benches/` on the stage VMs;
the S1-12 convergence gate re-runs nightly on `ci-cpu` to catch loss-curve drift.

## PR notes

- Branch: `ticket/B-04-saved-lse-ce-rmsnorm-invvar`.
- Two-repo flow per S0-02: fork PR + llama-farm submodule bump.
- Upstreaming disposition: **upstream-later** — extends fork-local ABIs (sparse-CE outputs,
  RMS_NORM extra dst); rides the sparse-CE RFC once proven (ROADMAP §11 triage b).
- Provenance per S0-01: headers name `unsloth/kernels/cross_entropy_loss.py` and
  `unsloth/kernels/rms_layernorm.py` (Apache-2.0, headers verified) — math only, no code.
