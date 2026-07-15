# The testing strategy — and how backend lanes inherit it

Written at the close of the 2026-07-15/16 Stage 0+1 audit (`audit-2026-07-15.md`). This is the
recipe, extracted from what actually caught bugs, for Stage 2 (Metal), Stage 3 (CUDA) and
Stage 4 (Vulkan) to reuse rather than rediscover.

## The one-sentence version

A green test proves nothing until you know what would make it red — so every claim gets a
**numerical oracle**, an **equivalence witness**, or a **mutation-proofed guard**, and every
tolerance is a **measurement with the observed number quoted beside it**.

## The five layers

| layer | instrument | catches | example |
|---|---|---|---|
| per-kernel | `test-backend-ops` MODE_GRAD (FD), driven through the vacuity-guarded wrapper | a wrong kernel in isolation | `tests/test_backend_ops_grad.py` + `tests/project_ops.py` (ONE registry) |
| whole-graph | central FD of the real graph | a wrong composition with correct parts | `tests/test_p0_gradient.py` |
| oracle | float64 reference, self-audited, per-tensor + per-step | anything numerically wrong that still trains | `reference_llama/moe/mamba/dpo/grpo.py` |
| independent | recorded PEFT curve, identity-hashed | the graph and the oracle wrong *together* | `tests/convergence/` |
| equivalence | bitwise A-vs-B where the claim IS the equivalence | silent divergence between paths | chunked-vs-naive attention, resume-vs-uninterrupted, threads 1/2/4, packed-vs-unpacked |

## The rules that earned their keep

1. **The oracle is independent and audited.** Written from the math, never transcribed from the
   graph ("a reference that shares the graph's structure shares its bugs"), and its own backward
   is FD-checked against its own forward before it judges anything.
2. **Tolerances are measurements.** Quote the observed number beside the band. When a band is
   wide, name the mechanism (DPO's z-cancellation floor; alpha=2.5 trajectory conditioning; the
   Q8_0 quantization band) — a band you can't explain is a bug you're hiding.
3. **Every guard must be able to fail, provably.** Variants assert their separation from the
   recorded trajectory; guards get mutation tests (`test_the_vacuity_guard_can_actually_detect_
   vacuity`, the poisoned-logp fallback test, the freshness guard's reconcile tests). The suite's
   worst historical failures were all green-and-worthless.
4. **Run OFF the happy path.** alpha≠rank, user_scale≠1, wd at ggml's max, rank≠4, grad_accum>1
   (`test_convergence_variants.py`): the recorded config's coincidences (scale exactly 1.0, decay
   exactly 0) hid a whole bug class. New features must ask "what coincidence is my default config
   sitting on?"
5. **Per-op green does not compose.** All three live bugs this audit caught — SOFT_MAX_BACK
   aliasing (MoE router), SSM initial-state aliasing, the Mamba backward abort — were invisible
   to MODE_GRAD because the *kernel* was right and the *graph* (allocator aliasing, in-place
   caches, view backwards) was wrong. The graph-level oracle is not optional.
6. **When oracle and graph disagree, FD of the real graph is the referee.** Applied three times;
   sided with the oracle three times. Diagnose before touching either side.
7. **Positive assertions only.** "The device we asked for did something," never "the word
   'Skipping' is absent" — absence-guards fired false on healthy runs twice.
8. **One registry.** Op lists, hyperparameters, tolerances: single source of truth, imported by
   CI and tests alike, with a freshness guard where a table mirrors external ground truth.
9. **Never pipe a test run through anything that eats the exit code.** It happened twice in this
   audit alone; the memory of "green" runs that never ran is the founding trauma of this suite.
10. **Rebuild the shim after any fork/csrc change** — `tests/test_probe.py` exists because this
    trap fired seven times.

## What a new backend lane (S2/S3/S4) actually does

The gate is already backend-parameterized; a lane does not fork the tests, it points them at its
device:

1. `pytest tests/test_backend_ops_grad.py --device <metal|cuda|vulkan>` — per-kernel FD on the
   real device, with the registry and vacuity guards intact (`ggml_device` fixture resolves the
   REAL registered name — `MTL0`, `CUDA0`, `Vulkan0`; never trust a guessed `-b`).
2. `pytest tests/test_convergence.py tests/test_convergence_variants.py --device <d>` — the full
   oracle gate on-device. The float64 references are device-independent; only the ggml side moves.
3. The equivalence witnesses (`test_determinism.py`, chunked attention, checkpointing) re-run
   per-device — a backend whose reductions are nondeterministic will fail the bit-identity test
   and must document its own claim instead.
4. The sched fallback report (`sched_fallback_report.py`) makes CPU-fallback visible: acceptable
   but reported, never hidden. Kernel-complete milestones flip ops to fallback-forbidden.
5. Tolerance bands may be re-measured per backend (SIMD/warp reduction order moves the noise
   floor) — but re-measured means *measured*, with the number quoted, not widened until green.

## Current verified state (2026-07-16)

SFT, DPO, GRPO, MoE, Mamba-1, full-finetune, quantized-base LoRA: all oracle-verified on CPU
(worst observed one-step gradient deviations 1e-6..4e-6; trajectories 5e-7..1e-5 with mechanisms
named). Known boundaries: SSM `n_group>1` refused loudly (B-10); Mamba-2 e2e deferred (B-10);
throughput audit deferred (B-11). 369 tests, ~75 s on a 4-core N100.
