---
id: S1-29
title: "SSM: CONCAT backward + SSM backward-switch wiring (S1/S4)"
stage: 1
track: kernels
size: S
deps: ["S0-02"]
status: open
pr: null
---

# S1-29 — SSM: CONCAT backward + SSM backward-switch wiring (S1/S4)

**One-line outcome:** `CONCAT` has a backward case (dst grad routed into two per-src grad
views — graph-level, no kernel), and `ggml_compute_backward` gains `SSM_CONV`/`SSM_SCAN`
cases emitting the new `SSM_CONV_BACK`/`SSM_SCAN_BACK` ops, so Mamba training graphs build
backward end-to-end once the S1-30/S1-31 CPU kernels land.

## Why (context)

The Mamba-1/2 family is one of the two architecture classes still blocked on CPU after the
dense work (ROADMAP §10). The four LoRA-able projections (`ssm_in`, `ssm_x`, `ssm_dt`,
`ssm_out`) are plain `MUL_MAT` through `build_lora_mm`
(`vendor/llama.cpp/src/models/mamba-base.cpp:45,86,104,140`), already covered by the
existing `OUT_PROD` backward — what is missing is gradient flow *through* the SSM-specific
ops. Today `CONCAT`, `SSM_CONV`, and `SSM_SCAN` all have no case in the
`ggml_compute_backward` switch (`vendor/llama.cpp/ggml/src/ggml.c:6430-6913`) and hit the
default `GGML_ABORT("unsupported ggml op for backward pass")` (`:6906`) — a training graph
containing any mamba layer kills the process at backward-build time.

This ticket is the graph/ABI half (ROADMAP §10 items S1 and S4): the `CONCAT` VJP, the two
new op enums with their constructors, and the backward-switch cases that emit them. The
compute kernels are deliberately split out — `SSM_CONV_BACK` is S1-30 and `SSM_SCAN_BACK`
is S1-31 — so this ticket's emission wiring builds graphs that cannot yet *execute*; tests
must be marked accordingly.

Frozen-tensor scoping (ROADMAP §10) bounds the work sharply: `A`, `D`, the conv weight, and
`dt_bias` stay frozen in LoRA training — no parameter grads are needed through the SSM ops,
only activation grads. `CONCAT` appears in the mamba graph as the conv-state concat
`ggml_concat(conv, transpose(x), 0)` (`vendor/llama.cpp/src/models/mamba-base.cpp:55`;
mamba-2 twin at `:199`), where the cache-side src (`conv`) is a recurrent-state constant
needing no grad — the backward case must therefore handle each src independently.

## What to do

All code changes land in the vendored llama.cpp fork via the S0-02 two-repo flow.

1. **`CONCAT` backward case** in `ggml_compute_backward`
   (`vendor/llama.cpp/ggml/src/ggml.c:6430-6913`): read the concat dim from op_params
   (set at `ggml.c:2621` in the `ggml_concat` constructor, `:2601-2628`); for each of
   src0/src1 that needs grads, take a view of `grad` covering that src's slab along the
   concat dim (src0 at offset 0, src1 offset by `src0->ne[dim]`), `ggml_cont` it to the
   src's shape, and accumulate with `ggml_add_or_set` (`ggml.c:6361-6375`). Precedent for
   grad-subregion routing: the `GGML_OP_VIEW` case (`ggml.c:6693-6719`) with
   `ggml_acc_or_set` (`:6377-6396`). Guard each side with
   `src0_needs_grads`/`src1_needs_grads` — the mamba conv-state concat only ever needs the
   src1 side.
2. **New op enums + constructors:** add `GGML_OP_SSM_CONV_BACK` and `GGML_OP_SSM_SCAN_BACK`
   at the op-enum tail (existing SSM entries: `vendor/llama.cpp/ggml/include/ggml.h:562-563`;
   tail placement minimizes fork rebase conflicts per ROADMAP §11), update the op
   name/symbol tables and their static asserts in `ggml.c`, and add constructors
   `ggml_ssm_conv_back(sx, c, dy) -> d_sx` next to `ggml_ssm_conv`
   (`vendor/llama.cpp/ggml/src/ggml.c:5534-5558`) and
   `ggml_ssm_scan_back(s0, x, dt, A, B, C, ids, dy) -> packed {dx‖ddt‖dB‖dC}` next to
   `ggml_ssm_scan` (`:5562-5624`). For the packed multi-grad dst follow the legacy
   `ggml_flash_attn_back` layout (aligned `offs_q/offs_k/offs_v` concatenation,
   `ggml.c:5501-5514`); `ggml_ssm_scan`'s own dst is already a packed 1-D `y‖states`
   tensor (`:5611-5612`), so packed outputs are established practice for this op family.
3. **Backward-switch case for `SSM_CONV`:** emit `SSM_CONV_BACK` for the input grad
   (`d_sx`, src0) only; `GGML_ASSERT(!src1_needs_grads)` with a comment pointing at the
   ROADMAP §10 S5 deferral (conv weight frozen).
4. **Backward-switch case for `SSM_SCAN`:** extract the `dy` view from the packed dst grad
   (first `ggml_nelements(x)` elements — the y region), emit `SSM_SCAN_BACK`, and unpack
   its packed dst into grad views for `x`, `dt`, `B`, `C` (reshaped to each src's shape)
   via `ggml_add_or_set`. Assert no grads are requested for `s0` (cache constant) or `A`
   (frozen); `ids` is I32 and automatically excluded from gradient propagation
   (`vendor/llama.cpp/ggml/src/ggml.c:7049-7051`; the `ignore_src` mechanism at
   `:7054-7076` is the pattern if an explicit exclusion is needed). The dst-grad state
   region (final states) carries no loss dependency in single-ubatch training — document
   the truncation-at-ubatch BPTT semantics in a comment (full semantics recorded in S1-31).
5. **MODE_GRAD for CONCAT:** `test_concat` exists
   (`vendor/llama.cpp/tests/test-backend-ops.cpp:5551`, cases generated at `:9015-9018`)
   but opts into gradient checking only via `ggml_set_param` calls in `build_graph`
   (harness convention documented at `:1984`) — add F32 grad-enabled cases covering all
   four dims and the non-contiguous-view variants (`v` = 1, 2, 3).
6. **Graph-build test for mamba backward emission:** a small test (fork-side C++ test or
   learning-llamas pytest via the bindings) that builds a mamba-style subgraph
   (concat → ssm_conv → ssm_scan with an F32 param upstream), calls
   `ggml_build_backward_expand`, and asserts the backward graph contains
   `SSM_CONV_BACK`/`SSM_SCAN_BACK` nodes. Build-only: execution is blocked until S1-30 and
   S1-31 land — mark the execution assertion skipped/xfail with a reference to those
   ticket IDs.
7. **Submodule bump PR** in learning-llamas referencing this ticket, per S0-02.

## Out of scope

- The `SSM_CONV_BACK` CPU kernel (S1-30) and the `SSM_SCAN_BACK` CPU kernel + tiny-Mamba
  e2e (S1-31) — this ticket only makes backward graphs *build*.
- Parameter grads for `A`, `D`, conv weight, `dt_bias`, and cross-ubatch BPTT (ROADMAP §10
  S5, deferred; B-09 owns activation if SSM full-FT enters scope).
- Metal/CUDA/Vulkan `SSM_*_BACK` ports (S2-12, S3-09, S4-07 per the stage plans).
- RWKV/gated-delta-net linear-attention ops (ROADMAP §10 "Others" — inventory only, not
  scheduled).

## Acceptance criteria

- [ ] Fork branch: `test-backend-ops` mode `grad` passes on CPU for the new grad-enabled
      `CONCAT` cases (all dims, non-contiguous variants) within the ADR-0002 tolerance.
- [ ] Fork branch: the mamba-subgraph backward-build test passes — backward graph builds
      without abort and contains `SSM_CONV_BACK` and `SSM_SCAN_BACK` nodes; its execution
      assertion is present but marked blocked-on-S1-30/S1-31.
- [ ] `GGML_OP_SSM_CONV_BACK`/`GGML_OP_SSM_SCAN_BACK` enums, name-table entries, and
      constructors exist; op name/symbol static asserts pass (build is green).
- [ ] Grad requests on frozen/cache srcs (`s0`, `A`, conv weight) fail with an explicit
      assert message, not the generic backward abort (grep-verifiable in the fork diff).
- [ ] learning-llamas submodule-bump PR is green in `ci-cpu` (per-PR lane).

## Testing & verification

Primary harness: vendored `tests/test-backend-ops` MODE_GRAD for `CONCAT` (CPU is the
finite-difference oracle per S0-09/ADR-0002), plus the new mamba backward-build test, both
running in learning-llamas's `ci-cpu` lane per-PR after the submodule bump; nightly `ci-cpu`
re-runs the full suite. `SSM_*_BACK` MODE_GRAD execution cases are owned by S1-30/S1-31,
which flip this ticket's blocked assertions on.

## PR notes

- Branch: `ticket/S1-29-concat-backward-ssm-switch-wiring`.
- Two-repo flow per S0-02: implementation PR against the fork's `learning-llamas-base` branch
  with the ticket ID in the title, plus a trivial learning-llamas submodule-bump PR referencing
  the same ticket ID.
- Upstreaming disposition: split — the `CONCAT` VJP is **upstream-early** (pure gap-fill,
  mainline training benefits, tests included); the `SSM_*_BACK` op enums and switch wiring
  are **fork-local first, upstream-later** as one RFC with the S1-30/S1-31 kernels once the
  CPU oracle proves the design (ROADMAP §11 triage b — new op enums live at the enum tail
  to minimize rebase conflicts).
- No copied external code; in-tree pattern reuse only.
