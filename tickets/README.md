# learning-llamas implementation tickets — agent guide

This directory is the full implementation backlog for learning-llamas: a Python library
that trains LoRA adapters (SFT / DPO / GRPO) on **frozen, quantized GGUF models**
using llama.cpp's ggml backend, with the training step **GPU-resident when a GPU
is available** on exactly four backends: **CPU, Metal, CUDA, Vulkan**.

Every ticket is scoped to become **exactly one pull request**. If you are an agent
picking up work: read this file top to bottom once, then work one ticket at a time.

The two design documents that every ticket cites live in `../docs/`:

- `docs/GGUF-LORA-TRAINING-BLUEPRINT.md` ("BLUEPRINT") — library architecture,
  design decisions D1–D7, gap analysis G1–G15, training methods, phases P0–P4.
- `docs/KERNEL-ROADMAP.md` ("ROADMAP") — the kernel-change plan per backend,
  items K-*/C*/M*/V*/FA*/E*/S*, delivery phases K0–K5, decide-first gates G-A/G-B.

Both were adversarially fact-checked against the pinned llama.cpp commit
(`4f37f51`); their `file:line` citations are reliable at that commit.

## Stage ordering

Work proceeds in stages, **in this order**:

| Stage | Directory | Theme | Exit criterion |
|---|---|---|---|
| 0 | `stage-0-groundwork/` | Repo, build, bindings, fixtures, CPU CI, policies | S0-07 CI green; no-op adapter smoke test passes |
| 1 | `stage-1-cpu/` | Complete, **correct** training on CPU (the oracle): **S1-00 first** (training graph must bypass the KV cache or backward aborts), then shim, losses, SFT/DPO/GRPO, all new ops' CPU reference kernels, FA fallback, MoE + SSM on CPU | S1-12 convergence gate green on CPU; MoE + SSM tiny models train |
| 2 | `stage-2-metal/` | Metal kernel suite → GPU-resident training on Apple Silicon | S2-10 milestone: convergence gate on `--device metal`, zero CPU fallback (dense) |
| 3 | `stage-3-cuda/` | CUDA ports + flash-attention backward flagship | S3-10 milestone: fully GPU-resident incl. FA |
| 4 | `stage-4-vulkan/` | Vulkan ports → NVIDIA/AMD/Intel coverage | S4-09 milestone: cross-backend parity report |
| — | `backlog/` | Deferred work with explicit activation triggers | not scheduled |

Stage N+1 work should not start before stage N's milestone ticket is done, with
one deliberate exception: **each stage's CI ticket (S2-01, S3-01, S4-01) may land
any time after S0-07**, so backend CI exists before backend kernels arrive.

Within a stage, tickets are ordered by their `deps`. Anything whose deps are all
`done` is claimable, and independent tickets are safe to work in parallel — the
dependency graph, not the numbering, is the truth.

Note for readers of the ROADMAP: its K-phases assumed CUDA-first delivery. This
project deliberately schedules **Metal before CUDA** (project decision); the
per-item content of the ROADMAP is unchanged, only the ordering differs.

## How to pick up a ticket

1. Choose a ticket whose frontmatter `deps` are all `status: done` and whose
   stage is active. Prefer lower stage, then smaller size, then the stage's
   critical path (called out in each milestone ticket).
2. Claim it: set `status: in-progress` in the ticket frontmatter in a tiny PR
   (or the first commit of your branch) so two agents never work the same ticket.
3. Branch: `ticket/<id>-<slug>` (e.g. `ticket/S1-04-sparse-ce-cpu-oracle`).
4. Implement exactly the ticket's "What to do". If you discover necessary work
   outside "What to do", check "Out of scope" — if it's excluded, file a note in
   the PR and leave it; if it's genuinely missing from the plan, add a new ticket
   file in the same PR and link it.
5. Run the ticket's "Testing & verification" locally (on the stage's VM — see
   `docs/dev/vm-playbooks.md`) before opening the PR.
6. Open the PR with the ticket ID in the title (`[S1-04] …`), set `pr:` in the
   frontmatter, `status: pr-open`. After merge: `status: done`.

**Two-repo flow for kernel tickets.** Tickets that modify vendored llama.cpp code
(track `kernels`) implement in the project's llama.cpp fork (see S0-02): open the
real PR against the fork's `learning-llamas-base` branch, then a trivial PR here that
bumps the `vendor/llama.cpp` submodule and flips the ticket status. The fork PR
carries the ticket ID in its title too.

## Definition of done (all tickets)

- All acceptance-criteria checkboxes in the ticket are true and demonstrated in
  the PR (test output, CI links, or committed artifacts).
- CI is green on every lane the ticket names — at minimum `ci-cpu`; kernel
  tickets also the lane of their backend stage.
- New ops/kernels: `test-backend-ops` MODE_GRAD parity vs the CPU oracle within
  the ADR-0002 tolerance (max-abs gradient error ≤ 0.05 at fp16), deterministic
  by default (gate G-B) — atomics only as measured, opt-in variants.
- Copied/adapted code carries per-file provenance headers (source path, commit,
  license). Unsloth: **math and design only** — never code from AGPL/LGPL files
  (see ROADMAP §13 for the license map).
- The ticket file's `status`/`pr` fields are updated.

## Correctness policy (read before writing any kernel)

Two ADRs bind every ticket; read them before starting kernel work:

- **ADR-0002 (S0-09): numerics + determinism.** F32 accumulation for all gradient
  matmuls (CUDA `CUBLAS_COMPUTE_32F` forced, Vulkan f16acc variants forbidden on
  grad paths); F32 row stats/LSE/loss; deterministic backward kernels by default.
- **ADR-0003 (decided inside S1-04): sparse-CE cross-backend ABI (gate G-A)** —
  LSE stash vs recompute, logit-buffer aliasing, softcap/scale op-params. Every
  backend CE port (S2-07, S3-03, S4-04) implements this ABI exactly.

The CPU implementation of every op is the oracle. GPU ports never redefine
semantics; they match the oracle within tolerance.

## Testing model: VMs + GitHub Actions

Development and pre-PR testing happen on per-backend VMs (playbooks:
`docs/dev/vm-playbooks.md`). GitHub Actions runs the detailed test suites per
backend on every PR (quick lane) and nightly (full lane):

| Lane | Runner | Per-PR | Nightly |
|---|---|---|---|
| `ci-cpu` | ubuntu-latest + macos-14 (CPU) | build, pytest, targeted MODE_GRAD | full MODE_GRAD + convergence gate |
| `ci-metal` | GH macOS arm64 (paravirt GPU); self-hosted Apple Silicon fallback | build + targeted Metal ops | full Metal sweep + gate `--device metal` |
| `ci-cuda` | compile lane on ubuntu-latest; GPU lane on GH GPU runner or self-hosted CUDA VM | compile + targeted GPU subset | full CUDA sweep + gate `--device cuda` |
| `ci-vulkan` | ubuntu-latest + Mesa lavapipe (software Vulkan); self-hosted native GPU lane | lavapipe correctness | full sweep + gate, native GPU |

Until a backend's ops land, `ggml_backend_sched` transparently falls back to CPU
for unsupported nodes — training *works* everywhere from stage 1 onward; each
backend stage moves ops from "CPU fallback" to "GPU-resident". CI reports which
ops ran where; milestone tickets flip their lane to **fallback-forbidden**.

## Sizing legend

**S** ≤ 2 days · **M** ≤ 1 week · **L** 2–3 weeks · **XL** 4+ weeks
(per engineer/agent already familiar with the backend).

## Ticket index

<!-- BEGIN GENERATED INDEX -->

### Stage 0 — Groundwork (9 tickets)

| ID | Title | Track | Size | Depends on |
|---|---|---|---|---|
| S0-01 | Repo scaffolding, license, packaging skeleton, provenance policy | infra | S | — |
| S0-02 | Vendor llama.cpp as pinned submodule + fork + patch queue | infra | S | S0-01 |
| S0-03 | CMake + scikit-build-core build: vendored llama.cpp (CPU) + shim skeleton liblearningllamas | infra | M | S0-02 |
| S0-04 | ctypes binding layer (_ffi): load order, struct mirrors, version lock | python | M | S0-03 |
| S0-05 | adapter.py: zero-init LoRA adapter GGUF via gguf-py (create/read/enumerate) | python | M | S0-01 |
| S0-06 | Test harness: pytest + tiny fixture GGUF models + no-op adapter smoke test | python | M | S0-04, S0-05 |
| S0-07 | CPU CI: GitHub Actions build + test on Linux x86 and macOS arm (CPU-only) | infra | M | S0-06 |
| S0-08 | Developer docs: build guide + per-backend VM playbooks | docs | S | S0-01 |
| S0-09 | ADR: numerics policy + determinism default (gate G-B) + parity criterion | docs | S | S0-01 |

### Stage 1 — CPU training core (34 tickets)

| ID | Title | Track | Size | Depends on |
|---|---|---|---|---|
| S1-00 | Training attention path: bypass the KV cache so gradients reach K/V (unblocks all backward) | kernels | M | S0-02, S0-03 |
| S1-01 | Shim: ll_opt_init_lora — ggml_set_param on adapter A/B tensors | shim | M | S1-00, S0-03, S0-04 |
| S1-02 | Shim: ll_train_step — forked opt_epoch_iter with pluggable loss + extra inputs | shim | L | S1-01 |
| S1-03 | P0 proof-of-gradient: stopgap composite CE, loss falls, FD check, llama-cli loads adapter | python | M | S1-02, S0-06 |
| S1-04 | New ggml op: ggml_cross_entropy_loss_sparse — ABI (gate G-A) + CPU oracle fwd/bwd | kernels | M | S0-09 |
| S1-05 | SFT trainer on sparse CE + masked validation eval | python | M | S1-03, S1-04, S1-06 |
| S1-06 | Data layer: chat templating, tokenization, loss-mask round-trip | python | M | S0-04 |
| S1-07 | Data layer: sample packing with seq_ids, boundary masking, fixed-shape collators | python | M | S1-06 |
| S1-08 | Adapter save-from-training + merged-model export with re-quantize | python | M | S1-01, S0-05 |
| S1-09 | Optimizer-state sidecar checkpoint/resume (name-keyed AdamW m/v) | shim | M | S1-02 |
| S1-10 | Gradient clipping | shim | S | S1-02 |
| S1-11 | Trainability preflight: graph walk vs supported-backward op set + arch report | shim | M | S1-02 |
| S1-12 | Convergence gate: tiny-model SFT vs recorded PEFT reference | python | M | S1-05 |
| S1-13 | Chunked lm_head / selective-logprob host pattern for no-grad passes | python | M | S1-04 |
| S1-14 | DPO trainer | python | M | S1-05, S1-13 |
| S1-15 | GRPO rollout engine: generation, logp_old capture, rewards, advantages | python | M | S1-05 |
| S1-16 | GRPO training step: clip-via-relu graph, k3 KL, self-verification harness | python | M | S1-15, S1-13 |
| S1-17 | Gradient checkpointing: layer-segmented recompute (+ optional CPU-offloaded boundaries) | shim | L | S1-02 |
| S1-18 | K-F16OP: F16/BF16 CPU out_prod (replace abort with to_float row path) | kernels | S | S0-02 |
| S1-19 | Small VJPs: TANH, SIGMOID, CLAMP (composite backward rules) | kernels | S | S0-02 |
| S1-20 | K-SMB: SOFT_MAX_BACK max_bias>0 — add the missing test, lift the CPU assert | kernels | S | S0-02 |
| S1-21 | FA1: emit_lse ABI on FLASH_ATTN_EXT + ggml_flash_attn_ext_back op + CPU forward LSE | kernels | M | S1-00, S0-09 |
| S1-22 | FA2: FLASH_ATTN_EXT autograd wiring in ggml_compute_backward | kernels | S | S1-21 |
| S1-23 | FA3: CPU flash-attention backward (modernize legacy kernel — the GPU oracle) | kernels | L | S1-21, S1-22 |
| S1-24 | FA8: graph-level chunked-attention backward fallback (kernel-free long-context path) | shim | M | S1-00, S1-19, S1-20 |
| S1-25 | MoE: MUL_MAT_ID + ADD_ID backward wiring (E1/E4) | kernels | S | S0-02 |
| S1-26 | MoE: OUT_PROD_ID CPU reference (activation grads through quantized experts) | kernels | M | S1-25, S0-09 |
| S1-27 | MoE: OUT_PROD_ID_GRP CPU reference (grouped expert outer product for LoRA A/B grads) | kernels | M | S1-25, S0-09 |
| S1-28 | MoE: GLU-family backward (SWIGLU_OAI, GEGLU exact/tanh, REGLU) + tiny-MoE e2e | kernels | M | S1-26, S1-27, S1-19 |
| S1-29 | SSM: CONCAT backward + SSM backward-switch wiring (S1/S4) | kernels | S | S0-02 |
| S1-30 | SSM: SSM_CONV_BACK CPU | kernels | S | S1-29 |
| S1-31 | SSM: SSM_SCAN_BACK CPU (chunk-recompute) + tiny-Mamba e2e | kernels | L | S1-29, S0-09 |
| S1-32 | CPU throughput audit + published tok/s sizing (P4) | docs | S | S1-12 |
| S1-33 | Windows/MSVC build of liblearningllamas + ci-windows lane (CPU) | infra | M | S0-03, S0-07 |

### Stage 2 — Metal (13 tickets)

| ID | Title | Track | Size | Depends on |
|---|---|---|---|---|
| S2-01 | Metal CI: GitHub Actions lane on macOS arm64 (MODE_GRAD + e2e) | infra | M | S0-07 |
| S2-02 | Metal M3: SOFT_MAX_BACK kernel | kernels | M | S2-01, S1-20 |
| S2-03 | Metal M4: RMS_NORM_BACK kernel | kernels | M | S2-01 |
| S2-04 | Metal M5: SILU_BACK kernel | kernels | S | S2-01 |
| S2-05 | Metal M1: OUT_PROD F32 kernel | kernels | M | S2-01 |
| S2-06 | Metal M2: OUT_PROD quantized-src0 kernel (critical path) | kernels | L | S2-05 |
| S2-07 | Metal M7/M8: sparse CE forward + backward kernels | kernels | M | S2-01, S1-04 |
| S2-08 | Metal M6: REPEAT_BACK kernel | kernels | M | S2-01 |
| S2-09 | Metal M10: ADD1 + DIAG_MASK_ZERO (+ graph-dump confirmation) | kernels | S | S2-01 |
| S2-10 | Metal milestone: GPU-resident dense-LoRA training on Apple Silicon | python | M | S2-02, S2-03, S2-04, S2-05, S2-06, S2-07, S2-08, S2-09, S1-12 |
| S2-11 | Metal MoE: OUT_PROD_ID + OUT_PROD_ID_GRP ports | kernels | L | S2-06, S1-26, S1-27 |
| S2-12 | Metal SSM: SSM_CONV_BACK + SSM_SCAN_BACK ports | kernels | L | S2-01, S1-30, S1-31 |
| S2-13 | Metal FA7: flash-attention forward LSE + backward | kernels | XL | S1-23, S2-02, S2-03, S2-04, S2-05, S2-06 |

### Stage 3 — CUDA (10 tickets)

| ID | Title | Track | Size | Depends on |
|---|---|---|---|---|
| S3-01 | CUDA CI: GPU lane (hosted GPU runner or self-hosted VM) + per-PR compile lane | infra | M | S0-07 |
| S3-02 | CUDA C1+C2: quantized + F16/BF16 OUT_PROD via dequant + cublasGemmEx (F32 accumulate) | kernels | M | S0-09, S3-01 |
| S3-03 | CUDA C3: sparse CE forward + backward | kernels | M | S1-04, S3-01 |
| S3-04 | CUDA C4: SOFT_MAX_BACK ALiBi gate lift | kernels | S | S1-20, S3-01 |
| S3-05 | CUDA FA4: flash-attention forward LSE emission | kernels | S | S1-21, S3-01 |
| S3-06 | CUDA FA5 (core): flash-attention backward, head sizes 64/128 | kernels | XL | S3-05, S1-23, S0-09 |
| S3-07 | CUDA FA5 (ext): head size 256 + occupancy tuning | kernels | L | S3-06 |
| S3-08 | CUDA MoE: OUT_PROD_ID + OUT_PROD_ID_GRP ports | kernels | L | S3-02, S1-26, S1-27 |
| S3-09 | CUDA SSM: SSM_CONV_BACK + SSM_SCAN_BACK ports | kernels | L | S3-01, S1-30, S1-31 |
| S3-10 | CUDA milestone: fully GPU-resident training including FA | python | M | S3-02, S3-03, S3-04, S3-06, S3-08, S3-09, S1-12 |

### Stage 4 — Vulkan (9 tickets)

| ID | Title | Track | Size | Depends on |
|---|---|---|---|---|
| S4-01 | Vulkan CI: lavapipe software lane (hosted) + native GPU lane (self-hosted) | infra | M | S0-07 |
| S4-02 | Vulkan V1: OUT_PROD F32 shader | kernels | M | S4-01 |
| S4-03 | Vulkan V2: OUT_PROD quantized via dequant + transpose-copy + mul_mm reformulation | kernels | M | S4-02 |
| S4-04 | Vulkan V3: sparse CE forward + backward shaders | kernels | M | S1-04, S4-01 |
| S4-05 | Vulkan V4+V5: DIAG_MASK_ZERO + backward-op constraint audit (incl. max_bias validation) | kernels | M | S4-01, S1-20 |
| S4-06 | Vulkan MoE: OUT_PROD_ID + OUT_PROD_ID_GRP ports | kernels | L | S4-03, S1-26, S1-27 |
| S4-07 | Vulkan SSM: SSM_CONV_BACK + SSM_SCAN_BACK ports | kernels | L | S4-01, S1-30, S1-31 |
| S4-08 | Vulkan FA6: flash-attention forward LSE + backward | kernels | XL | S1-23, S4-02, S4-03 |
| S4-09 | Vulkan milestone: GPU-resident training on NVIDIA/AMD/Intel via Vulkan | python | M | S4-02, S4-03, S4-04, S4-05, S4-06, S4-07, S1-12 |

### Backlog — deferred (9 stubs)

| ID | Title | Track | Size | Depends on |
|---|---|---|---|---|
| B-01 | Fused quantized OUT_PROD tile kernels (CUDA/Vulkan) — only if profiling triggers | kernels | XL | S3-02, S4-03 |
| B-02 | CUDA FA backward perf tier: mma family + opt-in atomic-dQ + quantized-KV dequant + MLA/sink grads | kernels | XL | S3-06, S3-07 |
| B-03 | Fused GLU backward op (GGML_OP_GLU_BACK) — bandwidth win | kernels | M | S1-28 |
| B-04 | Saved-LSE CE exploitation + RMS_NORM_BACK saved-inv-var variant | kernels | M | S1-04 |
| B-05 | GET_ROWS_BACK generalization + embedding/lm_head training | kernels | L | S3-02 |
| B-06 | Linear-attention family backward: RWKV6/7, GATED_LINEAR_ATTN, GATED_DELTA_NET | kernels | XL | S1-31 |
| B-07 | Adapter/product extensions: DoRA + rsLoRA, MoltenVK stopgap decision, multi-GPU training | python | L | S2-10 |
| B-08 | GELU-family + NORM (LayerNorm) VJPs | kernels | M | S1-19 |
| B-09 | Full-FT extras: MoE E8 + SSM S5 (non-LoRA parameter gradients) | kernels | L | S1-27, S1-31 |

<!-- END GENERATED INDEX -->
