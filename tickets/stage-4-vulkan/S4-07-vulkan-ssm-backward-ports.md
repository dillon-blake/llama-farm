---
id: S4-07
title: "Vulkan SSM: SSM_CONV_BACK + SSM_SCAN_BACK ports"
stage: 4
track: kernels
size: L
deps: [S4-01, S1-30, S1-31]
status: open
pr: null
---

# S4-07 — Vulkan SSM: SSM_CONV_BACK + SSM_SCAN_BACK ports

**One-line outcome:** Mamba-family LoRA training is GPU-resident on Vulkan —
`SSM_CONV_BACK` and `SSM_SCAN_BACK` run as sibling shaders of the existing forwards,
MODE_GRAD-parity-checked against the S1-30/S1-31 CPU oracles, and a tiny-Mamba model
trains end-to-end with `--device vulkan`.

## Why (context)

All four LoRA-able Mamba projections are plain `MUL_MAT`, already covered by the base
`OUT_PROD` work (S4-02/S4-03); what blocks GPU-resident SSM training on Vulkan is
gradient flow *through* the SSM ops (ROADMAP §10 S2/S3). S1-29 wired the backward switch
and S1-30/S1-31 delivered the CPU kernels; until this ticket lands, every mamba backward
node falls back to CPU via `ggml_backend_sched`. Vulkan already has both SSM **forwards**
(`ssm_conv.comp`, `ssm_scan.comp` under
`vendor/llama.cpp/ggml/src/ggml-vulkan/vulkan-shaders/`), so each backward is a
same-shape sibling shader (ROADMAP §10).

The forward structures to mirror: `ssm_conv.comp` is a 60-line row-parallel shader — one
thread per `d_inner` channel row over a token block, pipelines (including fused
silu/bias variants) created at
`vendor/llama.cpp/ggml/src/ggml-vulkan/ggml-vulkan.cpp:5499-5501`. `ssm_scan.comp`
already ships **two reduction variants** behind the `USE_SUBGROUP_ADD` compile-time flag
(extension gate `ssm_scan.comp:5-7`; `subgroupAdd` vs shared-memory tree at `:98-112`;
generator entries `vulkan-shaders-gen.cpp:1106-1107`), selected at pipeline-creation time
by `device->subgroup_arithmetic && device->subgroup_require_full_support`
(`ggml-vulkan.cpp:5491-5498`) — exactly the ROADMAP §10 S3 note that the subgroup-add
variant is already scaffolded, and the project portability rule (keep a no-subgroup
fallback; subgroup arithmetic is force-disabled on MoltenVK+AMD, `:5983-5994`) is
satisfied by replicating that dual-variant pattern in the backward.

The hard part is `SSM_SCAN_BACK`: the forward overwrites recurrent states in place (CPU
reference `s0 = s`, `vendor/llama.cpp/ggml/src/ggml-cpu/ops.cpp:9770`), so the backward
must recompute states — checkpoint every K tokens, re-run the forward inside each chunk,
then reverse-scan — following the chunk-recompute schedule and reverse-pass math the
S1-31 CPU reference established (`ddt` chains through `sigmoid(dt)` since the forward
applies softplus, `ops.cpp:9623`; `dB`/`dC` reduce over GQA-style head groups, `:9625`).
One Vulkan-specific honesty note: the forward `supports_op` accepts **Mamba-2 shapes
only** — the `is_mamba2` gate plus `d_state ∈ {128, 256}`, `head_dim % 16 == 0`, a shmem
check, and `subgroup_basic` (`ggml-vulkan.cpp:17647-17683`; `SSM_CONV` is F32-only,
`:17685-17686`). The backward mirrors those gates, so Mamba-1 backward keeps falling
back to the CPU oracle just as Mamba-1 forward does today. Determinism is gate G-B
(S0-09/ADR-0002): fixed-order reductions, no atomics.

## What to do

All ggml changes land in the vendored llama.cpp fork via the S0-02 two-repo flow.

1. **`ssm_conv_back.comp`:** sibling of `ssm_conv.comp` implementing the S1-30 contract —
   only `d_sx` (flipped-window correlation; conv weight frozen). Keep the forward's
   thread-per-channel-row layout so each thread owns disjoint `d_sx` rows: deterministic
   with no atomics for free. No fused silu/bias variants (the forward's fusion cases
   stay forward-only). Pipeline + getter case next to the forward's
   (`ggml-vulkan.cpp:11044-11050`), F32-only `supports_op` mirroring `:17685-17686`.
2. **`ssm_scan_back.comp`:** patterned on `ssm_scan.comp`, consuming the S1-29
   constructor's srcs and writing the packed `dx‖ddt‖dB‖dC` dst. Generate **both**
   variants via `USE_SUBGROUP_ADD` exactly as the forward does
   (`vulkan-shaders-gen.cpp:1106-1107`; pipeline-creation condition
   `ggml-vulkan.cpp:5491-5498`) — the no-subgroup shared-memory tree path is mandatory
   per the portability rule. Per chunk: pass 1 re-runs the forward recurrence from the
   previous checkpoint to materialize the chunk's states; pass 2 walks the chunk's
   tokens in reverse applying the S1-31 reverse-pass math with F32 accumulation
   throughout (ADR-0002). `dB`/`dC` head-group reductions use fixed-order
   shared-memory trees (or `subgroupAdd` in the subgroup variant) — no atomics
   (gate G-B).
3. **Checkpoint buffer (ROADMAP §12 Q7, Vulkan half):** store every-K-th state in a
   transient device buffer (the `prealloc_*` pattern,
   `ggml-vulkan.cpp:2102`) sized `ceil(n_t/K) × state`, unless an allocator constraint
   forces an extra-dst alternative; record the decision, the default K, and the shmem
   budget of the reversed loop in a code comment and the fork PR.
4. **`supports_op` for both `*_BACK` ops:** mirror the forward gates exactly
   (`:17647-17683`, `:17685-17686`) so the backward never accepts a shape its sibling
   forward rejects; Mamba-1 and other rejected shapes fall back to the S1-30/S1-31 CPU
   oracles via sched — documented, not silent.
5. **Plumbing:** the six mechanical touch points per new op (ROADMAP §7) for both ops.
6. **Tests:** re-run the S1-30/S1-31 MODE_GRAD case lists on Vulkan — grad-enabled
   `test_ssm_conv` (`vendor/llama.cpp/tests/test-backend-ops.cpp:3753`, shapes at
   `:8477-8482`) and `test_ssm_scan` (`:3819`, shapes at `:8507-8510`: Mamba-1, Mamba-2,
   Falcon-H1) plus S1-31's added cases (`n_group > 1`, multi-sequence, `n_seq_tokens`
   below/above K) — vs the CPU oracles within the ADR-0002 tolerance, on lavapipe
   per-PR and the native lane nightly/label-gated (S4-01). Mamba-1 cases are expected
   to report not-supported on Vulkan (forward parity), and the run log must show that,
   not a wrong answer. Exercise **both** shader variants: the native NVIDIA driver
   takes the subgroup path; force the fallback variant in one lavapipe job via a
   `GGML_VK_DISABLE_SUBGROUP_ARITHMETIC`-style env knob added alongside the existing
   `GGML_VK_DISABLE_*` toggles (`ggml-vulkan.cpp:5816-5841`) if no CI driver lacks
   subgroup arithmetic naturally. Add a determinism check: two identical runs produce
   bitwise-identical packed grads.
7. **e2e:** run S1-31's `tests/test_ssm_training.py` tiny-Mamba SFT with
   `--device vulkan` (fixture must be a Mamba-2-class config so the scan stays
   GPU-resident); the fallback report must show the SSM backward ops on Vulkan.
   Lavapipe correctness; perf numbers only from the native lane.
8. **Submodule bump PR** in learning-llamas per S0-02, appending both ops to the ci-vulkan
   targeted-op defaults.

## Out of scope

- Metal/CUDA ports — S2-12 / S3-09 (same contracts, same CPU oracles).
- Extending Vulkan SSM coverage beyond the forward's gates (Mamba-1 `d_state == 16`,
  odd head dims) — a forward+backward pair of gate relaxations is separate work; the
  backward only mirrors what the forward supports today.
- `ds0`, `dA`, `dD`, conv-weight and `dt_bias` grads; cross-ubatch BPTT — ROADMAP §10
  S5, deferred (truncation-at-ubatch semantics documented in S1-31).
- RWKV6/7, `GATED_LINEAR_ATTN`, `GATED_DELTA_NET` backward — ROADMAP §10 "Others",
  backlog.
- Perf tuning beyond the Q7 shmem/K recording; atomics-based variants — opt-in later
  per gate G-B, backlog.

## Acceptance criteria

- [ ] Fork branch: `test-backend-ops grad -b Vulkan0` passes for all grad-enabled
      `SSM_CONV` cases within the ADR-0002 tolerance vs the S1-30 CPU oracle.
- [ ] Fork branch: `test-backend-ops grad -b Vulkan0` passes for the Mamba-2-class
      `SSM_SCAN` grad cases (`n_group > 1`, multi-seq, chunk boundaries) within the
      ADR-0002 tolerance vs the S1-31 CPU oracle (≤ 0.05 max-abs @ fp16); Mamba-1 cases
      show as not-supported (CPU fallback), never as failures.
- [ ] Both `ssm_scan_back` variants (subgroup-add and shared-memory fallback) are built
      and each is exercised by at least one CI job (run logs identify which variant ran).
- [ ] Determinism: two identical Vulkan runs produce bitwise-identical `dx‖ddt‖dB‖dC`.
- [ ] The Q7 record exists in the fork PR: checkpoint-buffer decision, default K, and
      the reversed-loop shmem budget.
- [ ] `supports_op` for both `*_BACK` ops mirrors the forward gates; a shape the forward
      rejects is rejected by the backward (unit-asserted).
- [ ] `tests/test_ssm_training.py --device vulkan` passes on the lavapipe lane; the
      native-lane fallback report shows `SSM_CONV_BACK`/`SSM_SCAN_BACK` executing on
      Vulkan.
- [ ] learning-llamas submodule-bump PR is green in `ci-vulkan / lavapipe` (per-PR),
      `ci-vulkan / gpu` (label-gated), and `ci-cpu`.

## Testing & verification

Primary harness: vendored `tests/test-backend-ops` MODE_GRAD on the Vulkan backend vs
the S1-30/S1-31 CPU oracles under the ADR-0002 tolerances (S0-09), plus the fork-side
determinism check. Runs per-PR on `ci-vulkan / lavapipe` (targeted op list; one job
forcing the no-subgroup variant) with the full sweep nightly, and on the label-gated
`ci-vulkan / gpu` native lane (S4-01). The tiny-Mamba e2e joins the nightly ci-vulkan
e2e job; S4-09 later folds the SSM ops into the fallback-forbidden set.

## PR notes

- Branch: `ticket/S4-07-vulkan-ssm-backward-ports`.
- Two-repo flow per S0-02: implementation PR against the fork's `learning-llamas-base` branch
  with the ticket ID in the title (may be split fork-side into conv-back and scan-back
  commits for review), plus a trivial learning-llamas submodule-bump PR referencing the same
  ticket ID.
- Upstreaming disposition: **fork-local first, upstream-later** — the `SSM_*_BACK` op
  enums ride the op-family RFC with S1-29/30/31 once the CPU oracle and one GPU backend
  prove the design (ROADMAP §11 triage b); coordinate with S3-09/S2-12 on which port
  anchors the RFC.
- Provenance: shaders adapt the in-tree MIT forward structures (`ssm_conv.comp`,
  `ssm_scan.comp`, llama.cpp `4f37f51`); reverse-pass math comes from this project's
  research (ROADMAP §10 S3), not from any external implementation.
