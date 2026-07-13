---
id: S1-11
title: "Trainability preflight: graph walk vs supported-backward op set + arch report"
stage: 1
track: shim
size: M
deps: ["S1-02"]
status: pr-open
pr: https://github.com/dillon-blake/llama-farm/pull/25
---

# S1-11 — Trainability preflight: graph walk vs supported-backward op set + arch report

**One-line outcome:** a preflight that builds the forward graph once at load, walks its
nodes against the supported-backward op set, and produces an actionable report (trainable /
blocked-by-op-X / warn: raw `mul_mat` bypasses LoRA) — plus the auto-generated mirror of
`LLM_TENSOR_INFOS` into `src/learning_llamas/arch.py`.

## Why (context)

BLUEPRINT D5 is explicit: trainability is decided by a **graph walk**, never an arch
whitelist. llama.cpp carries ~133 architectures whose forward graphs (including LoRA
injection via `build_lora_mm`, `vendor/llama.cpp/src/llama-graph.cpp:1382-1449`) come free
by construction (BLUEPRINT §1.4); an arch whitelist would rot, while a walk of the actual
built graph is precise and robust to new archs. The consequence D5 draws is the two-tier
support model: tier 1 = verified-fast configs, tier 2 = anything that passes preflight runs
on the generic path — "new arch" must mean *unoptimized*, never *unsupported*.

The preflight exists because failure today is catastrophic, not graceful: backward coverage
lives only as the case labels of `ggml_compute_backward`
(`vendor/llama.cpp/ggml/src/ggml.c:6430-6913`), and any op that needs grads without a case
hits `GGML_ABORT` (`:6905-6907`) — the process dies mid-step. Concrete examples the report
must turn into actionable errors: TANH has no VJP (the unary sub-switch's default aborts,
`vendor/llama.cpp/ggml/src/ggml.c:6872-6876`), which blocks gemma2 and gemma3-with-softcap
GGUFs until the small-VJPs ticket S1-19 lands (BLUEPRINT §8 model coverage); `MUL_MAT_ID`
has no backward case (falls to the default abort at
`vendor/llama.cpp/ggml/src/ggml.c:6906`), which blocks all MoE archs. Two refinements make
the walk precise rather than alarmist: I32 tensors are automatically excluded from
gradients (`vendor/llama.cpp/ggml/src/ggml.c:7049-7051` — router top-k indices, positions),
and an op lacking backward only matters if it sits on a path from a trainable param to the
loss.

Gap G13 motivates the second deliverable: llama.cpp has no introspection APIs, and
`LLM_TENSOR_INFOS` (`vendor/llama.cpp/src/llama-arch.cpp:618-857`) — the table that
mechanically classifies every weight by its consuming op (`GGML_OP_MUL_MAT` →
LoRA-targetable, `MUL_MAT_ID` → MoE expert, norms → not targetable) — is internal-only.
D5 says to mirror it into Python at vendor-bump time so target selection works from the
base GGUF's tensor list alone. Finally, D5 mandates a *warning* (not failure) for
projections that bypass `build_lora_mm`: ~75 raw `ggml_mul_mat` call sites in exotic archs
(DeepSeek-MLA, RWKV, gemma3n — BLUEPRINT §1.3) silently receive no LoRA injection.

## What to do

1. **`csrc/farm_preflight.cpp` — the walker core.** A pure function over
   `(ggml_cgraph *, param tensor set)` so tests can feed synthetic graphs: replicate the
   grads-needed propagation of `ggml_build_backward_expand` — seed the adapter A/B params,
   propagate "needs grads" forward (skipping I32 tensors per
   `vendor/llama.cpp/ggml/src/ggml.c:7049-7051` and mirroring per-op `ignore_src`
   exclusions), and classify every node that needs grads against a **supported-backward op
   table** mirroring `ggml_compute_backward` coverage
   (`vendor/llama.cpp/ggml/src/ggml.c:6430-6913`), including per-op caveats (unary
   sub-switch coverage; `SOFT_MAX_BACK`'s `max_bias == 0` restriction). Nodes off the grad
   path are never blockers.
2. **`ll_preflight` entry point:** build the forward graph once at model+adapter load via
   the S1-02 graph-build path (`llama_model::build_graph`,
   `vendor/llama.cpp/src/llama-model.h:673`) without executing a training step, run the
   walker, and return a structured report over the flat C ABI in `csrc/farm_api.h`:
   entries of `{node name, op, status ∈ {ok, blocked, warn}, detail string}` plus an
   overall verdict.
3. **Actionable blocker messages:** each blocked op names the op, the first offending node,
   and the ticket that unlocks it (e.g. "TANH (final_logit_softcapping): blocked until
   S1-19"; "MUL_MAT_ID: MoE backward not yet implemented"). Message table lives in one
   place so later tickets flip entries.
4. **Bypass warning:** after graph build with the adapter attached, any adapter A/B tensor
   that appears in no graph node means its target projection bypassed `build_lora_mm`
   (raw `ggml_mul_mat` site, BLUEPRINT §1.3) — emit `warn`, not failure, naming the target
   tensor.
5. **Mirror generator `tools/gen_arch_mirror.py`:** parse `LLM_TENSOR_INFOS`
   (`vendor/llama.cpp/src/llama-arch.cpp:618-857`) from the vendored source text and
   generate `src/learning_llamas/arch.py` (tensor-name → consuming-op classification +
   LoRA-targetability, per BLUEPRINT §1.4/D5), with a generated-file header naming the
   vendor commit. Also generate the walker's supported-backward op table (parse the `case
   GGML_OP_*` labels of `ggml_compute_backward`) so both mirrors share one freshness
   mechanism at vendor-bump time.
6. **CI freshness check:** a `ci-cpu` step re-runs the generator and fails on any diff
   against the committed mirrors — the S0-02 vendor-bump checklist gains "re-run
   `gen_arch_mirror.py`".
7. **Python surface:** `FarmModel.preflight()` in `src/learning_llamas/model.py` (BLUEPRINT §4)
   returning the report as structured data with a human-readable rendering; trainers call
   it before the first step and raise on `blocked`.
8. **Two-tier support doc `docs/support-tiers.md`:** tier-1 verified-fast (the fixture- and
   convergence-verified archs, initially tiny-llama class) vs tier-2 passes-preflight
   (D5); states that preflight, not a whitelist, is the gate.

## Out of scope

- Adding any missing VJP (TANH etc. — S1-19; MoE/SSM backward — later-stage/backlog
  tickets). This ticket only *reports* them.
- Backend-coverage probing (which device an op runs on) — the sched's CPU fallback makes
  placement a performance concern reported by S1-12's gate, not a trainability one
  (ROADMAP §11 scheduler note).
- Target-preset selection UX and adapter creation (S0-05 owns creation; `arch.py` data
  feeds it).
- Norm-vector / non-`MUL_MAT` adapter targets (adapter format ignores them today,
  BLUEPRINT §1.3).

## Acceptance criteria

- [ ] `pytest tests/test_preflight.py` passes on the Linux CPU VM: the tiny llama-arch
      fixture (with attached zero-init adapter) reports **trainable** with zero blockers.
- [ ] Synthetic-graph tests exercise the walker core directly: a graph with TANH on the
      param→loss path reports `blocked` naming TANH and S1-19; the same graph with TANH
      off the grad path reports trainable; an I32-input op (e.g. GET_ROWS indices) is not
      flagged.
- [ ] A bypass-warning test: an adapter containing an A/B pair whose target never appears
      in the built graph yields status `warn` naming that tensor, and preflight still
      returns trainable overall.
- [ ] `src/learning_llamas/arch.py` is generated (header names commit `4f37f51`), and the CI
      freshness step fails when a mirror is stale (verified once by mutating a copy in the
      PR's CI run or a unit test of the diff logic).
- [ ] `docs/support-tiers.md` exists and defines tier 1 vs tier 2 per D5.
- [ ] `ci-cpu / test` green per-PR including the new tests and the freshness step.

## Testing & verification

`tests/test_preflight.py` (walker unit tests on synthetic ggml graphs built via `_ffi`,
plus end-to-end fixture-model preflight) in the S0-06 harness, per-PR in `ci-cpu / test`;
the generator freshness check runs in the same lane. No new ggml ops and no kernels, so no
`test-backend-ops` MODE_GRAD cases belong here. Manual: paste the rendered preflight report
for the fixture model into the PR description.

## PR notes

- Branch: `ticket/S1-11-trainability-preflight-arch-report`.
- Single learning-llamas PR (shim + generator + Python + docs); no vendored llama.cpp changes
  expected, so no two-repo flow. The walker reads vendored *source text* at generation
  time only.
- Upstreaming disposition: **fork-local** (BLUEPRINT §10 notes introspection APIs might be
  welcome upstream eventually; not pursued in v1).
- Soft coordination: S1-19 flips the TANH blocker entry when it lands; S0-05's target
  presets should consume `arch.py` once this merges (prose contract, not a dep).
