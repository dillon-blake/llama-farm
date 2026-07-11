---
id: S1-31
title: "SSM: SSM_SCAN_BACK CPU (chunk-recompute) + tiny-Mamba e2e"
stage: 1
track: kernels
size: L
deps: ["S1-29", "S0-09"]
status: open
pr: null
---

# S1-31 — SSM: SSM_SCAN_BACK CPU (chunk-recompute) + tiny-Mamba e2e

**One-line outcome:** the `SSM_SCAN_BACK` CPU kernel exists —
`(s0, x, dt, A, B, C, ids, dy) -> {dx, ddt, dB, dC}` with checkpoint-every-K-tokens state
recomputation — MODE_GRAD is green for `SSM_SCAN`, and a tiny Mamba model's SFT loss falls
on CPU.

## Why (context)

This is the flagship SSM item (ROADMAP §10 S3): with S1-29's wiring and S1-30's conv
backward in place, `SSM_SCAN_BACK` is the last op between the Mamba-1/2 family and CPU
LoRA training. The four trainable projections are plain `MUL_MAT`
(`vendor/llama.cpp/src/models/mamba-base.cpp:45,86,104,140`); `A`, `D`, conv weight, and
`dt_bias` stay frozen, so exactly four gradients are needed: `dx`, `ddt`, `dB`, `dC`.

The structural difficulty is that the forward
(`ggml_compute_forward_ssm_scan_f32`, `vendor/llama.cpp/ggml/src/ggml-cpu/ops.cpp:9562-9773`)
overwrites intermediate recurrent states in place: each token iteration writes the updated
state into the same per-sequence buffer and then aliases it as the next source
(`s0 = s;`, `ops.cpp:9770`; the per-element store at `:9762`), so by the time backward
runs, only the final state exists. The backward must therefore **recompute states**:
checkpoint every K tokens during a re-run of the forward recurrence, then per chunk re-run
forward from the checkpoint and reverse-scan (ROADMAP §10 S3). A store-all variant
(materialize all `n_t` states) is acceptable ONLY as the first-cut CPU reference — it is
~1 GiB/layer for Mamba-2-class configs at 512 tokens. Do NOT attempt the algebraic state
inverse (dividing out the decay): `dA = exp(dt_softplus·A)` underflows and the
reconstruction is numerically hopeless.

Two more forward facts shape the kernel. First, `dt` passes through softplus before use
(`dt_soft_plus = ggml_compute_softplus_f32(dt[h])`, `ops.cpp:9623`; Mamba-1 branch at
`:9720`), so `ddt` chains through `sigmoid(dt)`. Second, `B`/`C` are shared across a
GQA-style head group (`g = h / (nh/ng)` repeat_interleave, `ops.cpp:9625`), so `dB`/`dC`
must reduce over the heads in each group. Boundary terms vanish for single-ubatch
training: the initial state is a recurrent-cache constant (no `ds0` output), and the
final-state copy back into the cache (`vendor/llama.cpp/src/models/mamba-base.cpp:124-132`)
has no loss dependency, so the packed dst grad's state region is zero — this is
truncation-at-ubatch BPTT, and it must be documented as such (cross-ubatch BPTT is the
deferred S5 item).

## What to do

All code changes land in the vendored llama.cpp fork via the S0-02 two-repo flow.

1. **Kernel:** implement `ggml_compute_forward_ssm_scan_back_f32` in
   `vendor/llama.cpp/ggml/src/ggml-cpu/ops.cpp` next to the forward (`:9562-9773`),
   consuming the S1-29 constructor's srcs and writing the packed `dx‖ddt‖dB‖dC` dst.
   Reverse-pass math per ROADMAP §10 S3, per token t (working state grad `ds`, same shape
   as the recurrent state):
   `ds += C ⊗ dy` · `dC += Σᵢ dyᵢ·sᵢⱼ` (needs the recomputed state s at t) ·
   `dB += Σᵢ dsᵢⱼ·x_dtᵢ` · `dx = dt_softplus · Σⱼ dsⱼ·Bⱼ` ·
   `ddt = (∂/∂dt_softplus terms) · sigmoid(dt)` (chain through softplus, forward `:9623`) ·
   `ds_{t-1} = dA · ds_t` with `dA = exp(dt_softplus·A)` recomputed per token (forward
   `:9624`). Handle both the Mamba-2 scalar-decay branch (`A->ne[0] == 1`) and the Mamba-1
   per-state branch, mirroring the forward's two paths. All accumulation in F32
   (ADR-0002).
2. **State recomputation:** first land the store-all reference (all `n_t` states in
   scratch), then the checkpoint-every-K variant: pass 1 re-runs the forward recurrence
   storing every K-th state; pass 2 walks chunks in reverse, re-running the forward inside
   the chunk to materialize its K states, then reverse-scanning them. K is an op/config
   parameter with a documented default; the trade is checkpoint memory vs ~2x scan FLOPs
   (ROADMAP §12 Q7). Scratch lives in the CPU work buffer — extend the `n_tasks`/work-size
   planning in `vendor/llama.cpp/ggml/src/ggml-cpu/ggml-cpu.c` (SSM cases at `:2011`,
   `:2381`) to size it (Q7's buffer-placement question resolved for CPU as wdata).
3. **Determinism (gate G-B, S0-09):** partition threads by head as the forward does
   (`ops.cpp:9596-9601`) for `dx`/`ddt`; for `dB`/`dC` (per-group, shared by `nh/ng`
   heads) either assign group ownership to threads or reduce per-thread partials with a
   fixed-order tree — no atomics by default. Document the scheme in a comment.
4. **MODE_GRAD tests:** `test_ssm_scan` exists
   (`vendor/llama.cpp/tests/test-backend-ops.cpp:3819`; Mamba-1/Mamba-2/Falcon-H1 shapes at
   `:8507-8509`) but sets no params. Add grad-enabled cases that `ggml_set_param` only
   `x`, `dt`, `B`, `C` (convention at `:1984`), covering: Mamba-1 and Mamba-2 shapes,
   `n_group > 1` (exercises the group reduction), multi-sequence, and `n_seq_tokens` both
   below and above K (exercises chunking). Verify store-all and chunked variants agree
   bitwise, and both pass finite differences within the ADR-0002 tolerance (`exp`-heavy
   recurrences may need the harness's tighter-eps knobs — justify any per-case eps in the
   PR).
5. **Flip the S1-29 blocked assertions:** the mamba backward-build test now executes
   end-to-end.
6. **Tiny-Mamba e2e:** extend the S1-12 convergence-gate fixtures with a tiny Mamba-arch
   GGUF (2 layers, small `d_inner`/`d_state`); a pytest in llama-farm (e.g.
   `tests/test_ssm_training.py`) runs a short CPU SFT loop and asserts the loss falls
   (loss-decrease gate; full PEFT-parity curves stay owned by S1-12). Verify the S1-11
   preflight now reports mamba-family archs trainable — its graph walk should flip
   automatically once the backward cases exist; update its expected-arch fixtures.
7. **Submodule bump PR** in llama-farm referencing this ticket, per S0-02.

## Out of scope

- `ds0`, `dA`, `dD`, conv-weight and `dt_bias` grads; cross-ubatch BPTT (ROADMAP §10 S5,
  deferred — this ticket documents truncation-at-ubatch semantics only).
- Metal/CUDA/Vulkan `SSM_SCAN_BACK` ports (stage 2-4 tickets; this kernel is their
  MODE_GRAD oracle — the CUDA reversed-loop register-pressure question in ROADMAP §12 Q7
  belongs to the CUDA port).
- RWKV/`GATED_DELTA_NET` linear-attention backward (ROADMAP §10 "Others", not scheduled).
- SIMD optimization of the backward — scalar-first per ROADMAP §4 P3.

## Acceptance criteria

- [ ] Fork branch: `test-backend-ops` mode `grad` passes on CPU for all new `SSM_SCAN`
      grad cases (Mamba-1, Mamba-2, `n_group>1`, multi-seq, chunked-vs-store-all) within
      the ADR-0002 tolerance.
- [ ] Store-all and checkpoint-every-K paths produce bitwise identical grads on the same
      inputs; two thread counts produce bitwise identical grads (gate G-B determinism).
- [ ] The S1-29 mamba backward-build test executes end-to-end (no blocked markers remain).
- [ ] `tests/test_ssm_training.py`: tiny-Mamba CPU SFT loss falls over the scripted run;
      the S1-11 preflight report lists the mamba-family test arch as trainable.
- [ ] llama-farm submodule-bump PR is green in `ci-cpu`; the tiny-Mamba e2e runs in the
      nightly `ci-cpu` job (per-PR runs the MODE_GRAD cases only).

## Testing & verification

Primary harness: vendored `tests/test-backend-ops` MODE_GRAD on CPU (finite differences vs
analytic backward, ADR-0002 tolerance, S0-09) — per-PR in `ci-cpu` after the submodule
bump. The tiny-Mamba e2e joins the S1-12 gate family and runs in nightly `ci-cpu`
(per ROADMAP §11's e2e tier). Determinism checks (thread-count and chunking invariance)
run as fork-side unit tests in the same lane.

## PR notes

- Branch: `ticket/S1-31-ssm-scan-back-cpu-tiny-mamba`.
- Two-repo flow per S0-02: implementation PR against the fork's `llama-farm-base` branch
  (may be split fork-side into store-all reference + chunked variant commits for review),
  plus one llama-farm submodule-bump PR carrying the e2e pytest and fixtures, all
  referencing this ticket ID.
- Upstreaming disposition: **fork-local first, upstream-later** — one RFC for the
  `SSM_*_BACK` op family with S1-29/S1-30 once the CPU oracle (and ideally one GPU port)
  proves the design (ROADMAP §11 triage b).
- No copied external code; reverse-pass math comes from ROADMAP §10 S3 (derived in this
  project's research, not translated from any external implementation).
