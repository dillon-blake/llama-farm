---
id: B-07
title: "Adapter/product extensions: DoRA + rsLoRA, MoltenVK stopgap decision, multi-GPU training"
stage: backlog
track: python
size: L
deps: [S2-10]
status: open
pr: null
---

# B-07 — Adapter/product extensions: DoRA + rsLoRA, MoltenVK stopgap decision, multi-GPU training

**One-line outcome:** **DEFERRED, tracking-only umbrella** — this ticket is never implemented
as one PR: at activation each sub-item below is split into its own sized ticket (B-07a
rsLoRA/DoRA, B-07b MoltenVK decision, B-07c multi-GPU, …) and this umbrella's job is done when
those tickets are filed. It tracks: DoRA/rsLoRA adapter variants (needing upstream
`get_scale`/`build_lora_mm` changes), the one-benchmark MoltenVK-as-interim-mac-path decision,
multi-GPU layer-split training with gradient placement, and the norm-vector / multi-adapter
training questions. The `size: L` field estimates the tracked scope in aggregate, not this
file's own PR.

**Activation trigger:** per sub-item, after the S2-10 Metal milestone — DoRA/rsLoRA on user demand
(BLUEPRINT §9 P4); the MoltenVK benchmark as soon as Metal stage numbers exist to compare against
(ROADMAP §12 Q10 — one benchmark, cheap); multi-GPU on demand for models that do not fit one
device (BLUEPRINT §10 q6).

## Why (context)

**DoRA/rsLoRA** are upstream-coordination items: the LoRA scale is computed inside llama.cpp —
`llama_adapter_lora_weight::get_scale` hardcodes `adapter_scale · alpha / rank`
(`vendor/llama.cpp/src/llama-adapter.h:53-57`) — and the injection graph is built by
`build_lora_mm` (`vendor/llama.cpp/src/llama-graph.cpp:1382`). rsLoRA needs an `alpha/√rank`
scale mode; DoRA additionally needs a per-column magnitude vector, a weight-norm node in the
injection graph, and adapter-GGUF extensions that stock llama.cpp must load or explicitly
reject — hence "needs upstream `get_scale`/`build_lora_mm` changes" (BLUEPRINT §9 P4). Design
upstream-first, or trained adapters stop being interchange-clean (BLUEPRINT D3).

**MoltenVK** works today as a functional Vulkan-on-macOS path but only via scalar `mul_mm` — no
KHR coopmat, and subgroup arithmetic is force-disabled on MoltenVK+AMD
(`vendor/llama.cpp/ggml/src/ggml-vulkan/ggml-vulkan.cpp:5983-5994`). ROADMAP §7/§12 Q10 scope the
decision to exactly one benchmark against the native Metal stage outcome plus a recorded keep/skip
decision. **Multi-GPU** is explicitly deferred by BLUEPRINT §3: v1 is one compute device + CPU
host because ggml-opt allocates grads and optimizer moments on sched backend 0; layer-split
training requires ggml-opt state-placement work (grads/moments on the backend owning each split's
params). Two BLUEPRINT §10 q6 questions are tracked here so they have a home: norm-vector adapter
training (the loader ignores norm vectors — `vendor/llama.cpp/src/llama-adapter.cpp:287-288`,
"TODO: add support for norm vector") and multi-adapter/mixed-task training.

## What to do

Split into sub-tickets at activation; scope per item:

1. **rsLoRA:** propose the scale-mode KV + `get_scale` change upstream; learning-llamas side is
   adapter-creation metadata (S0-05 writer) plus a convergence A/B run. Small — do first.
2. **DoRA:** upstream design (magnitude tensor in the adapter GGUF, `build_lora_mm` norm epilogue,
   loader acceptance); then training wiring (magnitude as an extra `ggml_set_param` target) and a
   PEFT-DoRA parity check on the S1-12 gate model.
3. **MoltenVK:** run the S1-12/S2-10 benchmark config on MoltenVK (scalar `mul_mm`, no coopmat);
   compare tok/s and correctness vs the recorded Metal outcome; commit the keep/skip decision
   record — if "skip", wire nothing.
4. **Multi-GPU:** design doc for layer-split training — graph splits per device, grad/moment
   placement per split, cross-device grad-traffic audit, the concrete ggml-opt changes needed —
   plus a 2-GPU CUDA prototype; no productization inside this ticket.
5. **Tracked questions:** norm-vector adapter training and multi-adapter training get
   problem-statement notes until promoted to their own tickets.

## Out of scope

- Any kernel work (backend tickets own kernels).
- 8-bit optimizer states (BLUEPRINT §10 — LoRA moments are tiny; probably never).
- Serving-side multi-GPU inference (upstream llama.cpp owns that).

## Acceptance criteria

Umbrella acceptance: **sub-tickets filed at activation**, each carrying the per-item criteria
below into its own file. (The per-item criteria are recorded here so they transfer verbatim.)

- [ ] rsLoRA: upstream PR (or fork patch + upstream issue link) for the scale mode; a trained
      rsLoRA adapter GGUF loads in stock `llama-cli --lora`.
- [ ] DoRA: upstream design accepted-or-recorded; parity vs PEFT-DoRA on the S1-12 tiny-model
      gate within its loss-curve tolerance.
- [ ] MoltenVK: benchmark artifact + committed keep/skip decision record referencing ROADMAP §12
      Q10 (a "skip" with numbers fully satisfies this item).
- [ ] Multi-GPU: design doc committed; 2-GPU prototype trains the S1-12 config with loss matching
      single-GPU within tolerance (prototype-quality).
- [ ] Norm-vector + multi-adapter problem statements committed under `docs/` or as new tickets.

## Testing & verification

Adapter variants: learning-llamas `tests/` round-trip + convergence checks on `ci-cpu` per-PR, S1-12
gate config for parity (nightly). MoltenVK benchmark on the macOS VM (`ci-metal` infrastructure,
manual/nightly — lavapipe cannot stand in). Multi-GPU prototype on a self-hosted 2-GPU CUDA
runner (manual/nightly; not a per-PR lane).

## PR notes

- Branch: `ticket/B-07-dora-rslora-moltenvk-multigpu` (sub-tickets branch per item at activation).
- No vendored-kernel changes expected; DoRA/rsLoRA graph changes landing via the fork first follow
  the S0-02 two-repo flow.
- Upstreaming disposition: **upstream-early** for the llama.cpp-side DoRA/rsLoRA changes
  (coordinate before building — the adapter format is the interchange contract, BLUEPRINT D3);
  the Python-layer work is learning-llamas-local.
