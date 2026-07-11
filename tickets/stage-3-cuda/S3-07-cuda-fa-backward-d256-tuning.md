---
id: S3-07
title: "CUDA FA5 (ext): head size 256 + occupancy tuning"
stage: 3
track: kernels
size: L
deps: ["S3-06"]
status: open
pr: null
---

# S3-07 — CUDA FA5 (ext): head size 256 + occupancy tuning

**One-line outcome:** the S3-06 FA backward extended to D=256 with measured (not
assumed) shared-memory occupancy, per-arch tile-shape config entries, MODE_GRAD-green
at D=256, and a published perf snapshot vs the FA8 fallback and the naive path at 4k
context.

## Why (context)

S3-06 deliberately shipped FA backward for D=64/128 only, because D=256 was pre-scoped
as a distinct risk: **shared-memory pressure** (ROADMAP §12 Q5). The backward's pass-2
working set (a K/V tile plus a Q/dO tile plus the P/dS scratch) grows linearly with the
head size, so tile shapes that reach healthy occupancy at D=128 can drop to one block
per SM — or exceed the smem limit outright — at D=256. ROADMAP §12 Q5 therefore
mandates prototyping occupancy **before committing kernel shapes**, with tile
nbatch-splitting as the pre-scoped mitigation. D=256 matters: gemma-class and other
wide-head models are unreachable for FA training without it.

The forward tile family shows both the problem and the mitigation pattern. Its config
tables carry explicit D=256 entries
(`vendor/llama.cpp/ggml/src/ggml-cuda/fattn-tile.cuh:71-75` in the NVIDIA-fp16 table)
and encode per-(DKQ, DV, ncols) tuples of nthreads/occupancy/nbatch_fa/nbatch_K packed
by the macro at `vendor/llama.cpp/ggml/src/ggml-cuda/fattn-tile.cuh:13-19` — nbatch_K
is exactly the "split the head-dimension loads into chunks" knob the backward needs at
D=256. Config selection dispatches per vendor/arch class
(`vendor/llama.cpp/ggml/src/ggml-cuda/fattn-tile.cuh:316-340`) with typed accessors
(`:345-375`); S3-06's backward config table follows this pattern, so this ticket's
deliverable is largely **new table entries plus whatever kernel generalization the
prototype proves necessary** (e.g. staging K/V tiles in nbatch_K-sized slices, or
splitting pass 2 over head-dim chunks).

Tuning is per-arch: the ROADMAP §11 CI matrix names sm_70 and sm_90 class targets, and
S3-01's compile matrix builds both; the GPU lane tests whatever silicon the project VM
offers (documented in the S3-01/S0-08 playbook). Entries for arch classes without CI
hardware are best-effort extrapolations and must be marked as such in the table
comments. Numerics and determinism are unchanged from S3-06 (ADR-0002: deterministic
exclusive-write passes, MODE_GRAD vs the S1-23 CPU oracle).

## What to do

All code lands in the vendored llama.cpp fork via the S0-02 two-repo flow.

1. **Prototype first (ROADMAP §12 Q5):** before changing kernel shapes, build a small
   occupancy study over candidate D=256 pass-2/pass-3 tile shapes — vary
   ncols/nthreads/nbatch_fa/nbatch_K-style splits — recording per shape: smem bytes,
   registers/thread, achieved occupancy, and runtime at training-representative shapes
   (n_kv 2k/4k, GQA 4). Use the S3-06 kernels with the config table as the only
   variable. Land the study as `docs/perf/fa-backward-d256-occupancy.md` (tables +
   the chosen shapes + rejected alternatives).
2. **Kernel generalization as proven necessary:** implement nbatch splitting for the
   D=256 working set in `vendor/llama.cpp/ggml/src/ggml-cuda/fattn-back.cuh` (pattern:
   the forward's nbatch_K head-dim chunking, `fattn-tile.cuh:13-19` semantics) so the
   per-block smem footprint is bounded independently of D. Keep the three-pass
   exclusive-write structure — no determinism regression (gate G-B/ADR-0002).
3. **Per-arch config table entries:** add D=256 rows (and revisit the S3-06 D=64/128
   rows with the same measurement harness) for the sm_70/sm_80/sm_90 classes as
   available on CI — sm_70 and sm_90 are S3-01 compile targets, tuned numbers for the
   CI GPU's actual arch (sm_80-class if that is what the VM offers), commented
   best-effort values elsewhere — following the `fattn-tile.cuh:316-340` dispatch +
   `:345-375` accessor pattern.
4. **Flip `supports_op`** to accept DKQ == DV == 256 (F16 K/V), leaving the other
   S3-06 gates (no MLA-576, no quantized KV) intact.
5. **MODE_GRAD at D=256:** extend the S3-06 grad matrix
   (`vendor/llama.cpp/tests/test-backend-ops.cpp:6612` cases) with D=256 variants —
   GQA, mask/ALiBi, softcap, sinks — vs the CPU oracle within ADR-0002 tolerances;
   re-run the determinism assertion at D=256.
6. **Perf snapshot:** on the ci-cuda GPU runner, measure the FA-backward training step
   vs (a) the FA8 graph-level chunked fallback (S1-24) and (b) the naive attention
   path, at 4k ctx (plus 2k for trend) on a tiny model config with D=128 and a
   wide-head config exercising D=256: tok/s and peak allocation. Publish as
   `docs/perf/fa-backward-snapshot.md` + a CI artifact; this is the number that
   justifies the XL spend and feeds S3-10's platform table.
7. **Submodule bump PR** in llama-farm per S0-02.

## Out of scope

Explicit non-goals (manifest): **mma-family backward**, **atomic-dQ** single-pass,
**quantized-KV** backward, **MLA DKQ=576**, and **sink gradients** — all deferred to
backlog B-02 (perf phase, ROADMAP §11 K5) or excluded by construction. Also out:
Vulkan/Metal FA backward (FA6/FA7), the fallback-forbidden CI flip and convergence
gate (S3-10), and any re-tuning of the *forward* tile tables (upstream owns those).

## Acceptance criteria

- [ ] `docs/perf/fa-backward-d256-occupancy.md` exists with the measured shape study
      and names the chosen config-table entries (prototype-before-shapes evidence,
      ROADMAP §12 Q5).
- [ ] Fork branch: `test-backend-ops grad -b CUDA0 -o FLASH_ATTN_BACK` passes with the
      D=256 matrix included, within ADR-0002 tolerances vs the CPU oracle.
- [ ] Determinism test passes at D=256 (bitwise-identical grads across two runs; still
      no `atomicAdd` in `fattn-back.cu*`, grep-verifiable).
- [ ] `supports_op` accepts D=256 F16-K/V; MLA-576 remains CPU-fallback (sched report
      confirms).
- [ ] `docs/perf/fa-backward-snapshot.md` + CI artifact exist with tok/s and
      peak-memory for FA-backward vs FA8 vs naive at 2k/4k ctx, including a D=256
      config.
- [ ] Config table entries exist for the sm_70 and sm_90 compile classes (measured
      where CI hardware allows, marked best-effort otherwise); `ci-cuda / compile` is
      green on both matrix entries.
- [ ] llama-farm submodule-bump PR is green in `ci-cuda` (compile + GPU lanes) and
      `ci-cpu`.

## Testing & verification

Primary harness: vendored `tests/test-backend-ops` MODE_GRAD on CUDA vs the S1-23 CPU
oracle per ADR-0002 — per-PR in llama-farm's `ci-cuda` GPU lane (kernel-gated, targeted
`-o FLASH_ATTN_BACK`) and in the nightly full sweep (S3-01). The occupancy study and
perf snapshot run on the ci-cuda GPU runner (locally on the CUDA VM per the S0-08
playbook while iterating); their outputs are committed docs plus nightly artifacts, so
regressions are diff-visible. The snapshot's FA8 comparison reuses S1-24's fallback
path unchanged.

## PR notes

- Branch: `ticket/S3-07-cuda-fa-backward-d256-tuning`.
- Two-repo flow per S0-02: fork PR against `llama-farm-base` (ticket ID in title) plus
  a trivial llama-farm submodule-bump PR referencing the same ID; the perf/occupancy
  docs land on the llama-farm side.
- Size L — stage commits: (1) occupancy study + doc, (2) nbatch splitting + config
  entries + supports_op, (3) test matrix + perf snapshot.
- Upstreaming disposition: **upstream-later** — rides the FA-training op-family RFC
  with S3-06 (ROADMAP §11 triage class b); the per-arch table format intentionally
  matches upstream's forward tile tables to ease that review.
- Provenance per S0-01 policy: config-table/nbatch patterns adapted from
  `ggml/src/ggml-cuda/fattn-tile.cuh` (MIT, commit `4f37f51`).
