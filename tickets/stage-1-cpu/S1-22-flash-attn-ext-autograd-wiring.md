---
id: S1-22
title: "FA2: FLASH_ATTN_EXT autograd wiring in ggml_compute_backward"
stage: 1
track: kernels
size: S
deps: ["S1-21"]
status: deferred
pr: null
---

# S1-22 — FA2: FLASH_ATTN_EXT autograd wiring in ggml_compute_backward

> **⏸️ DEFERRED OUT OF STAGE 1 — project decision, 2026-07.**
>
> This ticket does **not** change how learning-llamas trains. Flash attention is force-disabled for
> training (`llama_context::set_training` logs *"disabling flash attention for training (no backward
> pass)"*), and **none of S1-21 / S1-22 / S1-23 turns it back on** — all three place that explicitly
> out of scope. Completing the whole family, ~6-8 weeks, would leave the training path byte-for-byte
> identical.
>
> Its real product is a **CPU correctness oracle for GPU flash-attention backward kernels** (FA5/6/7,
> stages 2-4). That payoff is entirely deferred on a CPU-only target, so this family moves to the
> front of whichever stage first starts a GPU backend.
>
> The memory argument does not rescue it either: S1-17 (gradient checkpointing) already removed the
> `x n_layers` factor, and **S1-24** — re-scoped as forward Q-chunking over S1-17's existing recompute
> — removes the `n_ctx^2` factor within a layer, using ops that already have backward rules. S1-24
> stays in stage 1; S1-21/22/23 do not.
>
> **A landmine for whoever picks this up.** The tiled FA forward path never updates `M[tq]` when a
> sink raises the running max (`ggml-cpu/ops.cpp`, the tiled sink fold — compare the one-chunk path,
> which does `M = s;`). Harmless today, because the forward normalizes by `S` and throws `M` away.
> **Silently wrong the moment anyone computes `lse = M + log(S)`** — which is exactly what S1-21
> exists to do. Fix it first, and test sinks x tiled, or the oracle ships wrong to every GPU backend.

**One-line outcome:** training graphs containing `FLASH_ATTN_EXT` build their backward
automatically: `ggml_compute_backward` emits the S1-21 `ggml_flash_attn_ext_back` op,
routes the packed dq/dk/dv views into the q/k/v gradients, and marks mask/sinks
`ignore_src`.

## Why (context)

S1-21 (FA1) delivers the flash-attention training ABI — the `emit_lse` flag on
`FLASH_ATTN_EXT` (packed `O‖LSE` dst with accessor views) and the new
`ggml_flash_attn_ext_back(q, k, v, mask, sinks, o, dO, lse, …) → dq‖dk‖dv` op replacing
the aborted legacy `ggml_flash_attn_back` (`vendor/llama.cpp/ggml/src/ggml.c:5470`).
But autograd still cannot see it: `ggml_compute_backward`
(`vendor/llama.cpp/ggml/src/ggml.c:6430`) has no `GGML_OP_FLASH_ATTN_EXT` case, so any
training graph containing an FA node dies in the switch default — "unsupported ggml op
for backward pass" (`vendor/llama.cpp/ggml/src/ggml.c:6904-6907`). This ticket is the
glue between the FA1 ABI and the FA3 CPU kernel (S1-23): after it, ROADMAP §8 FA2 is
done and the only missing piece for CPU-correct FA training is the kernel itself.

Two inputs of `FLASH_ATTN_EXT` must never receive gradients: the additive mask
(`src[3]`, set in the constructor at `vendor/llama.cpp/ggml/src/ggml.c:5421`) and the
attention sinks (`src[4]`, attached by `ggml_flash_attn_ext_add_sinks`,
`vendor/llama.cpp/ggml/src/ggml.c:5445-5458`). Both are F16/F32 tensors, so the
automatic I32 exclusion in `ggml_build_backward_expand`
(`vendor/llama.cpp/ggml/src/ggml.c:7049-7051`) does not cover them — they need explicit
`ignore_src` marks, exactly like ROPE's positions
(`vendor/llama.cpp/ggml/src/ggml.c:7069-7076`).

One more path matters for training graphs specifically: **after S1-00** they bypass the KV
cache (before it, backward-graph construction aborts outright), so K and
V arrive as F32→F16 casts (`vendor/llama.cpp/src/llama-graph.cpp:2416-2422`;
`ggml_cast` emits a `GGML_OP_CPY` node, `vendor/llama.cpp/ggml/src/ggml.c:3527`). The
dK/dV produced by the back op must therefore flow through the existing CPY backward
case (`vendor/llama.cpp/ggml/src/ggml.c:6665-6676`) to reach the F32 pre-cast tensors —
this needs verification in a test, not new code.

## What to do

All code changes land in the vendored llama.cpp fork via the S0-02 two-repo flow.

1. **Backward case** in `ggml_compute_backward`
   (`vendor/llama.cpp/ggml/src/ggml.c:6430`): add `case GGML_OP_FLASH_ATTN_EXT`.
   Construct **exactly one** `ggml_flash_attn_ext_back` node per FA node even when
   several of q/k/v need grads (unlike the per-src emissions of the MUL_MAT case,
   `vendor/llama.cpp/ggml/src/ggml.c:6578-6630`, share a local across the three
   branches). Inputs: q/k/v/mask/sinks from the node's srcs; `o` and `lse` via the FA1
   packed-dst accessor views; `dO` as the O-region view of the incoming packed-dst
   `grad` (the LSE region is a training-internal output that no loss path consumes);
   scale/max_bias/softcap copied from the forward node's op-params. Unpack dq/dk/dv via
   the FA1 accessors and route each into the src grads with `ggml_add_or_set`
   (`vendor/llama.cpp/ggml/src/ggml.c:6361`).
2. **Guard rails inside the case:** `GGML_ASSERT` with a clear message that the forward
   node has `emit_lse` set (backward is impossible without the stored LSE; training
   graph builders must request it — the naive path remains the learning-llamas default until
   a later milestone flips FA on). Assert mask/sinks were not requested as grad
   targets, pattern of the SOFT_MAX mask assert
   (`vendor/llama.cpp/ggml/src/ggml.c:6772`).
3. **`ignore_src` marks** in `ggml_build_backward_expand`
   (`vendor/llama.cpp/ggml/src/ggml.c:7020`): add a `GGML_OP_FLASH_ATTN_EXT` case
   setting `ignore_src[3]` (mask) and `ignore_src[4]` (sinks), following the ROPE
   positions precedent (`vendor/llama.cpp/ggml/src/ggml.c:7069-7076`).
4. **Graph-construction test** (new small fork-side test, e.g.
   `tests/test-fa-backward-build.cpp` registered in `tests/CMakeLists.txt`): build a
   tiny fwd+bwd graph with a `FLASH_ATTN_EXT(emit_lse)` node, q/k/v as params, K/V fed
   through F32→F16 casts as in training graphs. Assert: backward build does not abort;
   exactly one back-op node exists with the expected srcs; grads exist for q and for
   the **pre-cast F32** K/V tensors (CPY backward route,
   `vendor/llama.cpp/ggml/src/ggml.c:6665-6676`); no grad tensors exist for mask or
   sinks.
5. **MODE_GRAD cases, marked dependent on S1-23:** enable gradient checking on small
   `test_flash_attn_ext` cases (`vendor/llama.cpp/tests/test-backend-ops.cpp:6612`).
   Until S1-23 wires the CPU kernel into supports_op, these cases must **skip cleanly
   as not-supported** (the grad harness checks supports_op per graph tensor,
   `vendor/llama.cpp/tests/test-backend-ops.cpp:1723`) — never abort. S1-23
   flips them to executing; note this in the test comments with the ticket ID.
6. **Submodule bump PR** in learning-llamas referencing this ticket, per S0-02, so `ci-cpu`
   builds and runs the vendored tests.

## Out of scope

- The CPU backward kernel and its supports_op flip — S1-23 (this ticket's graphs build;
  execution of the back op stays unsupported until then).
- GPU FA backward and forward-LSE ports (FA4-FA7) — stage 2-4 tickets.
- The kernel-free chunked-attention fallback — S1-24.
- Enabling FA in learning-llamas training graphs by default, and preflight reporting —
  owned by the later FA-integration milestone; naive attention remains the default.
- Sink gradients — sinks are frozen in LoRA training (ROADMAP §8 FA5 scope note).

## Acceptance criteria

- [ ] Fork branch: the new graph-construction test passes — backward for a
      `FLASH_ATTN_EXT(emit_lse)` graph builds without hitting the
      `ggml_compute_backward` default abort, emits exactly one back op, grads reach the
      pre-cast F32 K/V tensors, and mask/sinks have no grads.
- [ ] `test-backend-ops` mode `grad` completes on CPU with the new
      `test_flash_attn_ext` grad cases reported as not-supported skips (no abort, no
      failure) while S1-23 is unlanded.
- [ ] The `emit_lse` requirement and mask/sinks asserts exist with descriptive
      messages (grep-verifiable in the fork diff).
- [ ] The `ignore_src` case for `GGML_OP_FLASH_ATTN_EXT` covers src[3] and src[4]
      (grep-verifiable).
- [ ] learning-llamas submodule-bump PR is green in `ci-cpu` (per-PR lane), which runs the
      vendored test-backend-ops grad mode and the new build test.

## Testing & verification

Fork-side: the new `tests/test-fa-backward-build.cpp` graph-construction test plus
`test-backend-ops` MODE_GRAD (cases added here, skipping until S1-23; full
finite-difference parity vs the CPU oracle within ADR-0002 tolerances is S1-23's
acceptance, running per-PR in `ci-cpu` from then on). learning-llamas side: after the
submodule bump, `ci-cpu / test` (S0-07) runs both per-PR; nightly `ci-cpu` re-runs the
full suite.

## PR notes

- Branch: `ticket/S1-22-flash-attn-ext-autograd-wiring`.
- Two-repo flow per S0-02: fork PR against `learning-llamas-base` with the ticket ID in the
  title, then a trivial learning-llamas submodule-bump PR referencing the same ID.
- Upstreaming disposition: **upstream-later** — this wiring is part of the FA-training
  op family (FA1 ABI + FA2 + FA3), fork-local first and proposed upstream as one RFC
  once the CPU oracle plus one GPU backend prove the design (ROADMAP §11 triage b;
  fallback if rejected: fork-local op flag with a small rebase patch).
- No copied external code expected; in-tree pattern sources (ROPE ignore_src, SOFT_MAX
  assert) are MIT and already carry their notices.
