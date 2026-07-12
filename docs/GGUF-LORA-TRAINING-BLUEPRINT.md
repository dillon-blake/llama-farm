# llama-farm — Blueprint for a Python LoRA-Training Library on the ggml Backend

**Goal:** a Python library that trains LoRA adapters on top of **frozen, quantized GGUF models**, running locally on llama.cpp's ggml backend, supporting **SFT, DPO, and GRPO**, inheriting new model architectures from llama.cpp with near-zero per-model work, and adopting unsloth-style efficiency techniques where they transfer.

**Status:** design blueprint (no implementation). Grounded in a deep read of both repos in this directory: `llama.cpp` (HEAD 4f37f51, 2026-07-10) and `unsloth`. File:line references below point into those checkouts so each section can be expanded into a concrete plan.

---

## 0. Feasibility verdict (TL;DR)

**This is very buildable, and most of the hard machinery already exists in llama.cpp.** The load-bearing facts:

1. **ggml already has autograd + optimizers.** `ggml_build_backward_expand` / `ggml_compute_backward` (`ggml/src/ggml.c:6430-7117`) plus a driver layer `ggml-opt` (`ggml/include/ggml-opt.h`) with AdamW/SGD as graph ops, gradient accumulation, and loss construction. All exported C API.
2. **Frozen-quantized-base + trainable-F32-LoRA is architecturally supported by ggml today.** Only leaf tensors flagged `ggml_set_param` get gradients/optimizer state; quantized weights are skipped by grads-needed propagation; activation gradients flow *through* quantized matmuls via a dequantizing `GGML_OP_OUT_PROD` (CPU: `ggml/src/ggml-cpu/ops.cpp:4363-4501`). Nobody has wired LoRA tensors into this — that's the gap, not the engine.
3. **llama.cpp trains with the same graph it infers with.** `llama_context::opt_epoch_iter` (`src/llama-context.cpp:3257-3364`) calls `model.build_graph(...)` — the identical per-arch builder used for inference. And every arch's projections route through `build_lora_mm` (`src/llama-graph.cpp:1382-1449`), which already injects `W·x + scale·B(A·x)` as ordinary differentiable nodes. **Mark the adapter A/B tensors as params and gradients flow to them with zero per-arch code** — *once G16 is fixed* (see §2): today the causal-arch training graph routes K/V through the KV cache, which severs the autodiff edge and makes backward-graph construction abort. The fix is one place in `llm_graph_context`, and the zero-per-arch-code property survives it. New architectures inherit trainability by construction.
4. **The LoRA adapter GGUF format is a complete, stable interchange contract** (loader: `src/llama-adapter.cpp:149-418`; converter: `convert_lora_to_gguf.py`). Our trained adapters, written in that format, load directly into llama.cpp, llama-server, and ollama.
5. **gguf-py alone can create a zero-initialized adapter GGUF from the base model's metadata** — no C code needed for adapter initialization (recipe in §6.1).
6. **What's genuinely missing** (and is the work of this project): LoRA param wiring + adapter create/save, a per-step training API with custom losses (the current one hardcodes next-token cross-entropy), one new ggml op (fused sparse-label CE — masking on the existing CE op is *mathematically wrong*, see D4 in §5), a handful of GPU kernels (quantized `OUT_PROD` on CUDA above all), and the whole Python layer.

**Biggest risks:** (a) GPU backward through quantized weights currently falls back to CPU (CUDA `OUT_PROD` is F32-only, `ggml-cuda.cu:4689`) — usable but slow until we write one kernel; (b) no flash-attention backward → training runs the memory-hungry softmax attention path; (c) MoE / SSM / RWKV architectures are untrainable until backward ops for `MUL_MAT_ID` / SSM ops are added (forward-only inference is unaffected).

---

## 1. Foundation inventory — what llama.cpp already provides

### 1.1 ggml autograd + ggml-opt (reuse as-is)

| Piece | Where | Notes |
|---|---|---|
| Backward graph builder | `ggml.c:7020-7117` (`ggml_build_backward_expand`) | Params = leaf tensors flagged `ggml_set_param`; F32/F16 gate applies only to tensors that *need* grads, so frozen quantized weights pass through fine |
| Per-op VJPs | `ggml.c:6430-6913` (`ggml_compute_backward`) | Covered: MUL_MAT, RMS_NORM, SOFT_MAX, ROPE, GET_ROWS, split-SWIGLU, SILU/RELU/EXP/SOFTPLUS, CE, MUL/ADD/SUB/SCALE/SUM/LOG, views/reshapes. Missing: see §2 |
| Optimizer steps | `ggml_opt_step_adamw/sgd` (graph ops, `ggml.c:6155-6196`) | F32 params only; CUDA/Vulkan kernels exist; per-step hyperparams via callback → Python-driven LR schedules for free |
| Driver layer | `ggml-opt.h` / `ggml-opt.cpp` | Loss types MEAN/SUM/CE/MSE; grad accumulation (`opt_period`); dynamic-graph mode (rebuild per ubatch) is exactly how llama.cpp uses it; header explicitly blesses copying the high-level parts (`ggml-opt.h:191-194`) |
| Quantized backward (CPU) | `ops.cpp:4363-4501` (`out_prod` with dequant) | all block-quant types supported as src0 (Q4_0…Q8_0, Q2_K…Q6_K, IQ*, TQ*, MXFP4, NVFP4); **F16 aborts** (why finetune.cpp forces F32 KV cache) |
| Grad/optimizer state | F32 per param (grads + AdamW m/v) | Tiny for LoRA ranks; allocated on sched backend 0 |

**Custom-loss escape hatch (officially sanctioned):** build your own loss expression in the forward graph, make it the `outputs` tensor, and use `GGML_OPT_LOSS_TYPE_SUM`/`MEAN` to reduce it (`ggml-opt.h:28-29`). Extra `GGML_TENSOR_FLAG_LOSS` nodes are rejected (`ggml-opt.cpp:343`) — everything must fold into one scalar. This is how all three training methods attach their objectives (§6).

### 1.2 The llama_opt_* training path (copy as template, don't call)

- Public C API: `llama_opt_init` / `llama_opt_epoch` + `llama_opt_param_filter` (`include/llama.h:1560-1591`), all `LLAMA_API`-exported and ctypes-drivable today.
- Reality of the implementation (`src/llama-context.cpp:3200-3412`):
  - Loss **hardcoded** to `GGML_OPT_LOSS_TYPE_CROSS_ENTROPY` (line 3224); labels built as **dense one-hot F32 `[n_vocab, n_ubatch]`**, one 1.0 per position — no masking possible (3344-3354).
  - `llama_set_param` silently skips every non-F32 tensor (3201) and only iterates **base-model** tensors (3234-3254). **LoRA adapter tensors are never offered to the filter** — they ride through the training graph as inert constants.
  - `token_embd.weight` and `rope_freqs.weight` hard-excluded ("FIXME", 3207-3212).
  - `opt_epoch_iter` (3257-3364) is the ~100-line loop worth copying verbatim: clear KV → build `llama_batch` → ubatch split via the normal memory module → `model.build_graph(gparams)` → `ggml_opt_prepare_alloc(opt_ctx, ..., res->get_inp_tokens(), res->get_logits())` → fill labels → `ggml_opt_eval`. Everything it touches (`graph_params`, `llm_graph_result::get_logits()`, `balloc`, `memory`) is **private C++** — a shim compiled against `src/` internals is mandatory.
- The existing `examples/training/finetune.cpp` is full-parameter F32-only finetuning (README: "very much WIP"), forces `use_mmap=false` and F32 KV cache. Useful as a smoke-test reference only.

### 1.3 The LoRA adapter subsystem (adopt wholesale)

- `llama_adapter_lora` holds `ab_map: name → {a, b}` keyed by **base tensor name**; scale = `user_scale * alpha / rank`, rank = `b->ne[0]` (`src/llama-adapter.h:48-88`); caveat: `alpha == 0` in metadata silently drops the `alpha/rank` factor (scale = user scale only) — always write a real alpha.
- `build_lora_mm` / `build_lora_mm_id` (`src/llama-graph.cpp:1382-1449`) inject `res = W·x + scale·B(A·x)` at graph build; ~263 call sites across `src/models/*.cpp` cover attn q/k/v/qkv/o, ffn up/gate/down, MoE router+experts, lm_head, token_embd (embedding path uses efficient `get_rows` on A, `llama-graph.cpp:2151+`).
- **Adapter tensors are live graph leaves, never merged** — they can be updated in place and are exactly the right objects to `ggml_set_param`.
- GGUF adapter format (the on-disk contract we adopt for checkpoints):
  - KV: `general.type="adapter"`, `general.architecture=<must equal base arch>`, `adapter.type="lora"`, `adapter.lora.alpha=<f32>`; optional task_name / prompt_prefix / aLoRA invocation tokens (aLoRA = "activated LoRA": the adapter engages only after a trigger token sequence).
  - Tensors: `<base_name>.lora_a` (ne `[n_in, r]` = numpy `(r, n_in)`) and `<base_name>.lora_b` (ne `[r, n_out]` = numpy `(n_out, r)`) — identical layout to PEFT; `token_embd.weight` uses a flipped convention (`llama-adapter.cpp:356-368`).
- Known holes: no create-from-memory, no save, no in-place update API, no tensor enumeration in the C API; norm-vector adapter tensors ignored at load; ~75 raw `ggml_mul_mat` call sites in exotic archs (DeepSeek MLA, RWKV, gemma3n, …) silently bypass LoRA.
- Related tools to copy from: `convert_lora_to_gguf.py` (format writer), `tools/export-lora` (merge into base; extend to re-quantize instead of forcing F16).

### 1.4 Architecture registry — why "new model support" is nearly free

- ~133 architectures; each is a `llama_model_base` subclass with exactly three overrides (`load_arch_hparams`, `load_arch_tensors`, `build_arch_graph`); dispatch is data-driven from the GGUF's `general.architecture` (`src/llama-model.cpp:39-309`).
- Tensor **naming is global**, not per-arch (`LLM_TENSOR_NAMES`, `src/llama-arch.cpp:371-606`), and `LLM_TENSOR_INFOS` (`llama-arch.cpp:618-857`) mechanically classifies every weight by its consuming op (`GGML_OP_MUL_MAT` → LoRA-targetable; `MUL_MAT_ID` → MoE expert; norms/scans → not targetable). This table is internal-only — we mirror it (§5.5).
- What a training lib inherits per new arch, for free, by linking libllama: forward graph incl. LoRA injection, tokenizer, chat template (embedded Jinja via `llama_model_chat_template`), KV/recurrent/hybrid memory selection, hparams accessors.
- What it still needs per arch: **nothing hand-written** — LoRA target names + shapes are derivable from the base GGUF's tensor list (gguf-py) plus the mirrored op table; trainability is decided by a *graph walk* against the supported-backward op set (D5 in §5), not an arch whitelist.

### 1.5 gguf-py (the Python foundation)

- `GGUFReader` (mmap, zero-copy), `GGUFWriter` (incl. adapter typing and `.lora_a/.lora_b`-aware param counting), pure-numpy quantize (F32/F16/BF16/Q4_0/Q4_1/Q5_0/Q5_1/Q8_0/TQ/MXFP4) and dequantize (those + all K-quants/IQ/NVFP4), the `MODEL_ARCH`/`MODEL_TENSORS`/`TENSOR_NAMES` registries, `tensor_mapping.py` for HF↔GGUF names.
- In-repo precedent for ctypes over libggml: `gguf-py/tests/test_quants.py` (calls `ggml_quantize_chunk` for K-quant quantization from Python).
- Everything needed to **create a zero-init adapter GGUF from base metadata alone** exists here today.

---

## 2. Gap analysis — what must be built

Ordered roughly by how fundamental each item is. "Layer" refers to the architecture in §3.

| # | Gap | Evidence | Layer that fills it |
|---|---|---|---|
| **G16** | **The KV cache severs the gradient path to K/V, and backward-graph construction aborts.** `build_attn` stores `k_cur` via `cpy_k` = `ggml_set_rows` and then reads attention's K from a *view of the cache leaf* (`get_k`) — no autodiff edge from `k_cur` to attention. `ggml_set_rows` returns a view with `op = SET_ROWS`, which `ggml_build_backward_expand` refuses (`ggml.c:7093`), so the graph **hard-aborts** as soon as anything upstream of K/V needs a gradient — i.e. every multi-layer LoRA config. Verified by running upstream `llama-finetune`, which aborts before printing a loss; `examples/training/` has no CI. **Blocks everything below.** | `llama-graph.cpp:2667-2677`; `llama-kv-cache.cpp:1243,1329`; `ggml.c:3917,7093` | vendored fork: training graph bypasses the cache (ticket S1-00) |
| G1 | No LoRA training wiring: adapter tensors never `ggml_set_param`'d; no adapter-only param filter | `llama-context.cpp:3234-3254` | C shim |
| G2 | No adapter create-from-config / save / in-place-update / enumeration APIs | loader is file-path-only, `llama-adapter.cpp:420`; no writer in C++ | C shim + gguf-py |
| G3 | Loss hardcoded to unmasked next-token CE; no custom-loss hook; per-ubatch loop is private C++ | `llama-context.cpp:3224, 3344-3354` | C shim (forked `opt_epoch_iter` + loss-graph builder) |
| G4 | **Masking on existing CE is mathematically wrong**: backward computes `(softmax − labels)·d/nr` unconditionally, so all-zero label rows still emit `softmax·d/nr ≠ 0` gradients, and `1/nr` counts masked rows | verified: `ops.cpp:11284-11307`, `cross-entropy-loss.cu:88-90` | **one new ggml op** (D4) |
| G5 | Dense one-hot labels are O(n_vocab) per token (128k vocab × 512 ubatch ≈ 262 MB F32) | `llama-context.cpp:3344-3354` | same new op kills this |
| G6 | CUDA `OUT_PROD` F32-only → backward through every frozen quantized matmul falls back to CPU; Metal/Vulkan lack `OUT_PROD`/CE entirely | `ggml-cuda.cu:4689-4690`; `ggml-metal-device.m`; `ggml-vulkan.cpp` | kernel roadmap (§7) |
| G7 | No flash-attention backward (`ggml_flash_attn_back` aborts, `ggml.c:5470`); `soft_max_back` requires `max_bias==0` (no ALiBi) | `ggml.c:6904-6907`, `ops.cpp:5539` | constraint in v1; kernel roadmap later |
| G8 | Missing VJPs: `MUL_MAT_ID` (all MoE), `NORM` (LayerNorm archs), GELU-family, TANH (gemma softcapping), fused GLU variants, SSM_CONV/SSM_SCAN, RWKV/delta-net (linear-attention family) ops, CONCAT, ARGSORT, SIGMOID, CLAMP | `ggml.c:6430-6913` coverage list | preflight + upstreamable patches |
| G9 | No gradient checkpointing anywhere in ggml-opt (full fwd+bwd graph held; README reports 24 GB for 1B F32 @ n_ctx 512) | `ggml-opt.cpp` (absent) | C shim (layer-segmented training), later |
| G10 | `ggml_opt_dataset` is a single fixed-shape tensor pair; no masks/pairs/advantages/packing; varying batch shapes assert | `ggml-opt.cpp:86-130, 851` | Python data layer (bypass entirely) |
| G11 | No DPO/GRPO anything: no pairwise batching, no logprob-gather op, no ref-model plumbing, no advantage weighting | — | Python trainers + shim loss graphs (§6) |
| G12 | No Python bindings for training; llama-cpp-python has never bound `llama_opt_*` and this fork's API has drifted from it | — | binding layer (§3, Layer 2) |
| G13 | No introspection APIs: tensor enumeration, "is this weight LoRA-hookable", "is this graph differentiable" | `llama.h` (absent) | C shim + mirrored tables |
| G14 | No optimizer-state serialization: grads exposed (`ggml_opt_grad_acc`) but AdamW m/v are internal and keyed by fragile forward-graph node index | `ggml-opt.h:156`; `ggml-opt.cpp:458-486` | C shim: enumerate/extract/restore m/v **by param tensor name**; sidecar checkpoint format (see D3) |
| G15 | No gradient clipping anywhere in ggml-opt (optimizer step is fused into the backward graph) | `ggml-opt.h` (absent) | C shim: host clip pass over grad accumulators between accumulation and step (`opt_period > 1`), or a scale-by-global-norm node inserted before `opt_step_adamw` |

---

## 3. Proposed architecture

Four layers; native code is deliberately thin and mostly *copied* from llama.cpp per its own invitation (ggml-opt header: "can be copied to and adapted for user code").

```
┌────────────────────────────────────────────────────────────────────┐
│ Layer 3 — Python library  (the product)                            │
│  adapters (create/save/load) · datasets/collators (chat template,  │
│  masking, packing) · trainers (SFT/DPO/GRPO) · arch registry &     │
│  trainability preflight · rollout engine (GRPO) · eval/logging     │
├────────────────────────────────────────────────────────────────────┤
│ Layer 2 — Binding                                                  │
│  ctypes first (in-repo precedent: test_quants.py); promote the     │
│  shim edge to nanobind when zero-copy views / GIL-release matter   │
├────────────────────────────────────────────────────────────────────┤
│ Layer 1 — C shim ("libllamafarm")  ← the key new native code       │
│  compiled against vendored llama.cpp src/ internals; flat C ABI:   │
│   • adapter: create_zero / from_file / tensors / set-get / save    │
│   • opt_init_lora (ggml_set_param on A/B only)                     │
│   • train_step(batch, aux inputs, loss_spec) → forked              │
│     opt_epoch_iter + custom loss epilogue on res->get_logits()     │
│   • preflight: walk built graph, report ops lacking backward       │
│   • new ggml op: fused sparse-label CE (fwd+bwd, CPU+CUDA)         │
├────────────────────────────────────────────────────────────────────┤
│ Layer 0 — vendored llama.cpp (git submodule, pinned commit)        │
│  libllama + libggml-base unmodified where possible; small patch    │
│  set upstreamed when accepted (clamp/sigmoid VJPs, quantized       │
│  OUT_PROD kernels, mul_mat_id backward, …)                         │
└────────────────────────────────────────────────────────────────────┘
```

**Why a shim instead of pure ctypes on the public API:** the per-ubatch loop's dependencies (`graph_params`, `llm_graph_result`, `balloc`, `memory->init_batch`) are private C++ (`src/llama-context.h:207-213`); the public `llama_opt_epoch` bakes in the wrong loss and no masking. The shim is small (~1-2 kLOC: forked `opt_epoch_iter` + adapter lifecycle + loss builders) and everything else stays upstream.

**Why ctypes first:** every symbol needed is already exported (`LLAMA_API`/`GGML_API`; ggml-opt lives in `libggml-base.so`). Precedent exists in-repo. The one hazard — a struct-by-value callback return for optimizer params — is avoidable by passing the exported `ggml_opt_get_constant_optimizer_params` with a Python-owned params struct as userdata, mutated between steps for LR schedules.

**Packaging:** steal llama-cpp-python's build approach (scikit-build-core + CMake over the vendored submodule) but do not depend on llama-cpp-python (no training coverage; API drift). Pin llama.cpp commit + gguf-py + struct mirrors as one atomic version. Load order: `libggml-base` → `libggml` (RTLD_GLOBAL, honors dlopen'd backends) → `libllama` → shim.

**Which llama.cpp is Layer 0?** The checkout in this directory is a fork that has drifted from upstream (it carries `src/llama-ext.h` staging API and API differences like the batch `llama_set_adapters_lora`). Decide and record which lineage the submodule pins — the fork or upstream — since it changes the upstreaming path and the ctypes struct mirrors; every "this fork" note in this document refers to this checkout.

**Device placement & parallelism (v1 scoping):** v1 targets **one compute device + CPU host**. ggml-opt allocates grads/momenta on sched backend 0; partial `n_gpu_layers` offload works mechanically (the sched splits graphs) but is not a supported v1 config — recommended configs are full-offload GPU or pure CPU, since quantized-weight backward falls to CPU anyway until kernel #2 (§7) lands. Multi-GPU training (layer-split, grad placement per split) is explicitly deferred (open question, §10).

**Distribution:** v1 = source wheels + one prebuilt CPU wheel; CUDA wheels via a versioned extra index (cibuildwheel matrix) — this is precisely where llama-cpp-python causes user pain, so budget for it. `GGML_BACKEND_DL` (dlopen'd backends) is the eventual path to one thin wheel + per-backend packages; the load order above already accommodates it.

---

## 4. Proposed repo structure

```
llama-farm/
├── vendor/llama.cpp/                # git submodule, pinned; small patch queue in patches/
├── csrc/                            # Layer 1 shim
│   ├── farm_adapter.cpp             #   create_zero/from_file/save/enumerate/set-get (copies
│   │                                #   llama_adapter_lora_init_impl minus the file I/O)
│   ├── farm_train.cpp               #   forked opt_epoch_iter → lf_train_step(); opt_init_lora;
│   │                                #   loss-graph epilogues (sft_ce / dpo / grpo)
│   ├── farm_preflight.cpp           #   graph walk vs supported-backward op set; backend probes
│   ├── farm_api.h                   #   flat C ABI (everything Python sees)
│   └── ggml_ext/ce_sparse.{c,cu}    #   new op: fused sparse-label cross-entropy (fwd+bwd)
├── src/llama_farm/                  # Layer 3
│   ├── _ffi/                        #   Layer 2: ctypes mirrors of structs/functions (generated
│   │                                #   where possible; version-locked to the submodule commit)
│   ├── adapter.py                   #   LoraAdapter: create/save/load/merge; gguf-py backed
│   ├── model.py                     #   FarmModel: load GGUF, attach adapters, preflight report
│   ├── arch.py                      #   mirrored LLM_TENSOR_INFOS + target presets; auto-generated
│   │                                #   from llama-arch.cpp at vendor-bump time
│   ├── data/                        #   chat templating (jinja2 on GGUF-embedded template),
│   │                                #   tokenization, loss-mask computation, packing collators
│   ├── train/
│   │   ├── sft.py  ├── dpo.py  ├── grpo.py
│   │   ├── loop.py                  #   step loop, grad-accum, LR schedules, checkpointing
│   │   └── rollout.py               #   GRPO generation via normal decode; logp_old capture
│   ├── quant.py                     #   (de)quant helpers; ctypes ggml_quantize_chunk fallback
│   └── eval.py / callbacks.py / logging.py
├── tests/                           #   incl. finite-difference grad checks (copy test-backend-ops
│                                    #   MODE_GRAD harness) and the adapter round-trip test (§10)
├── benches/
└── pyproject.toml                   # scikit-build-core; builds vendor + csrc into wheels
```

---

## 5. Core design decisions

### D1 — Fork the training loop; never call `llama_opt_epoch`
(ggml-opt — the layer *below* — is still reused via its public API; only the `llama_opt_*` wrapper gets forked.)
Copy `opt_epoch_iter` into the shim and parameterize it: (a) expose the forward graph's logits tensor for loss attachment, (b) accept extra named input tensors (loss mask, sampled ids, advantages, ref/old logprobs), (c) let the host choose the loss epilogue, reduced to a scalar via `GGML_OPT_LOSS_TYPE_SUM`. Keep the proven mechanics verbatim (memory clear, ubatch split, dynamic `ggml_opt_prepare_alloc`). Constraint to respect: dynamic-graph mode keys grad/optimizer state by forward-graph node index (`ggml-opt.cpp:458-486`), so graph topology must be identical across steps → pad batches to fixed ubatch shapes (Python side).

### D2 — LoRA params via `ggml_set_param` on adapter tensors; that's the whole trick
(**Precondition: G16.** The training graph must first stop routing K/V through the KV cache — otherwise `ggml_build_backward_expand` aborts and no gradient reaches anything. See §2 G16 and ticket S1-00.)
After adapters are attached to the context (`llama_set_adapters_lora`), flag every `ab_map` A/B tensor (F32) as a param and pass a filter that rejects everything else. `build_lora_mm` already put the A/B matmuls in the graph; `ggml_build_backward_expand` does the rest. Two preconditions (verified, non-blocking): `ggml_set_param` requires leaves (`op == GGML_OP_NONE` — adapter tensors qualify, they are `ggml_dup_tensor` copies) and must run **after adapter attach but before the first opt-graph build** — PARAM-flagged leaves are promoted into graph nodes at build time and ggml-opt's grad/momentum allocation scans only graph nodes. Perf note: adapter tensors that fell back to CPU bufts (repacked base weights, `llama-adapter.cpp:337-350`) incur cross-backend gradient traffic. Base weights stay quantized, frozen, **mmap-able** (unlike full FT, LoRA training never writes base weights — keep `use_mmap=true`; verify early).

### D3 — Adapter GGUF format = checkpoint format
Trained adapters must load in stock llama.cpp/llama-server/ollama with zero conversion. Zero-init B ⇒ step-0 adapter is a provable no-op — ship that as a smoke test (logits with adapter@scale=1 == logits without). Also provide merged-model export (copy `export-lora`'s graph; extend to re-quantize to the base's original quant types). Scope note: the adapter GGUF is the *interchange* format, not the *resume* format — mid-run checkpoints pair it with a sidecar holding AdamW m/v tensors keyed by param tensor name, step count, LR-schedule state, and data cursor/RNG (G14).

### D4 — One new ggml op serves all three methods: `ggml_cross_entropy_loss_sparse(logits, i32_labels, f32_weights)`
Returns per-token `−w_t · log_softmax(logits)[y_t]` (no reduction) with backward `w_t · (softmax − onehot)` — exactly zero where `w_t = 0`. Pattern the CPU kernel on `ops.cpp:11158` (which already uses `ggml_vec_log_soft_max_f32`) and CUDA on `cross-entropy-loss.cu`. This one op provides:
- **SFT prompt masking** (weights = mask) — fixing the wrong-gradient problem (G4),
- **elimination of dense one-hot labels** (G5) and of the NaN-prone `log∘softmax∘select` composite,
- **DPO/GRPO per-token logprob gather** (negate; weights select completion tokens),
- and it *is* the unsloth fused-CE idea, expressed as a ggml op.

### D5 — Architecture support: inherit by construction, gate by graph walk
Never re-implement per-arch graphs. Target selection = base GGUF tensor list (gguf-py) ∩ default preset {attn_q,k,v,qkv,output; ffn_up,gate,down; optional output/token_embd} classified via a **mirrored `LLM_TENSOR_INFOS`** table auto-generated from `llama-arch.cpp` at vendor-bump time. Trainability = build the forward graph once at load and walk its nodes against the supported-backward op set — precise, arch-agnostic, and robust to new archs (an arch whitelist would rot). Two-tier support (the key unsloth structural lesson): tier 1 = verified fast configs; tier 2 = anything that passes preflight runs on the generic path — "new arch" must mean *unoptimized*, never *unsupported*. Warn (don't fail) on projections that bypass `build_lora_mm` (raw `ggml_mul_mat` call sites in DeepSeek-MLA/RWKV/gemma3n/etc.).

### D6 — Reference model = same weights, adapter disabled; never a second model
For DPO/GRPO, reference logprobs come from the identical frozen base with LoRA off (`llama_set_adapters_lora` with empty set / scale 0) — zero extra weight memory. Prefer **precomputing ref logprobs offline** (normal `llama_decode` + logits) and feeding them as constant input tensors, so the training graph needs no second forward at all. (Unsloth arrives at the same design via a PEFT `disable_adapter()` hack; llama.cpp's per-context adapter sets make it clean.)

### D7 — Data pipeline entirely in Python; bypass `ggml_opt_dataset`
Chat templating (jinja2 over the GGUF-embedded template), tokenization (llama.cpp tokenizer via C API), loss-mask boundaries computed at tokenization time, sample packing with distinct `seq_id`s (llama.cpp's attention masking already isolates sequences — copy unsloth's boundary-label-masking semantics), padding to fixed ubatch shapes. Write batches straight into input tensors via `ggml_backend_tensor_set`.

---

## 6. Training methods

### 6.1 SFT (v1 core)
- Loss: `sum(ce_sparse(logits, labels, mask))`, `LOSS_TYPE_SUM`, host-normalized by valid-token count.
- Masking/packing/templating per D7. Until the new op lands, a stopgap composite exists (softmax → one-hot mul → sum_rows → log, select-then-log order to avoid `0·(−inf)` NaNs) — all constituent ops have VJPs — but it materializes one-hot tensors; treat as bring-up only.
- Adapter init recipe (pure gguf-py, no C): read base GGUF → enumerate targets → write `A ~ N(0,σ) (r, n_in)`, `B = 0 (n_out, r)` F32 + the four adapter KVs (`token_embd` targets use the flipped, A-transposed convention — `llama-adapter.cpp:356-360`). Loadable by stock `llama_adapter_lora_init` immediately.
- Evaluation: masked validation loss on a held-out split via the same forward graph with backward disabled (forward-only `ggml_opt_prepare_alloc`); optional generation-based eval later reuses the GRPO rollout engine.
- Optimizer/accum/LR schedules come free from ggml-opt (per-step params callback).

### 6.2 DPO
| Requirement | Status |
|---|---|
| Policy forward + per-token logprobs of realized tokens | `ce_sparse` (negated) on the training graph's logits |
| Reference logprobs | precomputed offline (D6), fed as constant F32 inputs |
| Per-sequence sums with prompt mask | `mul` by mask + `sum_rows` — VJPs exist |
| `−log σ(β·Δ)` loss | **`SIGMOID` has no backward** — use the identity `−log σ(x) = softplus(−x)`; `SOFTPLUS` VJP exists (`ggml.c:6867`) |
| Chosen/rejected pairing | Python batcher packs pairs into the batch dim; per-pair loss values become the `outputs` tensor under `LOSS_TYPE_SUM` |

### 6.3 GRPO
- **Rollouts are ordinary llama.cpp inference** (KV cache, samplers, parallel sequences; adapter active) — the generation side needs zero new native code. Capture per-token `logp_old` at sample time; compute rewards + group-normalized advantages in numpy.
- Training forward: `logp_new` via `ce_sparse`; importance ratio `exp(logp_new − logp_old_const)` (EXP VJP exists); **PPO clip composed from RELU identities** — `clip(r,lo,hi) = lo + relu(r−lo) − relu(r−hi)`, `min(a,b) = b − relu(b−a)` — because `CLAMP` has no backward (a ~10-line upstream VJP would simplify; do both); per-token weighting = `mul` by `advantage·mask` input; optional KL penalty to the reference policy — the low-variance "k3" estimator `exp(Δ) − Δ − 1` — from precomputed ref logprobs via EXP/SUB.
- Three-pass step shape (copy from unsloth): (1) generate with adapter on, (2) no-grad chunked logp passes for old/ref, (3) grad pass with the loss above. Copy their *self-verification* discipline: any exactness-preserving optimization (packing, prefix sharing) first-use-compares against the naive path and permanently falls back on mismatch.

---

## 7. Kernel & efficiency roadmap (unsloth-informed, ranked by impact)

Unsloth's code is Triton/PyTorch — **ideas transfer, code does not**. Notably, unsloth *dequantizes* nf4 to 16-bit for every matmul; ggml's native quantized matmuls are already better on that axis. What to build, in order:

1. **Fused sparse-label CE op** (D4) — biggest memory win (no `[tokens, vocab]` one-hot or full logit-grad materialization; unsloth's "fused CE" equivalent). CPU + CUDA first.
2. **Quantized `OUT_PROD` for CUDA** (dequant-on-the-fly rank-k update, mirroring the CPU path; or reformulate `dX = Wᵀ·dY` as a mul_mat against the existing quantized kernels) — unblocks GPU-resident backward through frozen quantized weights (G6). *The single highest-value performance item for GPU LoRA on quantized GGUFs.* Metal/Vulkan follow.
3. **Chunked-CE / chunked selective-log-softmax host pattern** — even with op #1, chunk the lm_head matmul over row-chunks for long contexts and for GRPO/DPO no-grad logp passes (all existing ops; copy the `autotune_batch_and_chunks` interface).
4. **Gradient checkpointing** (layer-boundary recompute, optionally CPU-offloaded boundary states) — built at the shim's graph-construction layer; per-layer builds already exist in llama.cpp; ggml gives no help (G9). Unsloth's crossover heuristic: offload only for seq ≥ ~512.
5. **Small upstream VJPs**: CLAMP, SIGMOID, TANH (unlocks gemma2/3 logit softcapping), GELU-family, fused-GLU forms, `NORM` (LayerNorm) — each ~10-50 lines following existing patterns; each unlocks arch families (GELU/NORM → GPT-2/Phi/BERT-style).
6. **`MUL_MAT_ID` backward** — unlocks all ~30 MoE archs; substantial kernel work.
7. **Flash-attention backward** — large project; until then train with FA off (softmax path, `max_bias==0` only).
8. **Fused LoRA epilogue** (collapse the 6-nodes-per-weight pattern; fuse `W·x + scale·B(A·x)`) — bandwidth win, after correctness.
9. Fused elementwise SwiGLU/GeGLU backward (one pass producing `(h, df, de)`, activation recomputed not stored) — copy unsloth's math directly into a ggml op.

Not copying: Q-GaLore (full-FT oriented), MoE grouped-GEMM kernels (AGPLv3).

**Licensing & provenance.** Unsloth is Apache-2.0 at the top level, but AGPLv3 markers also appear on specific functions *outside* the MoE kernels — notably GRPO per-token-logp code in `models/rl_replacements.py` (line ~1191). The GRPO chunking/autotune items above must therefore be **clean-room reimplementations of the idea** (interfaces and math re-derived), never translations of that code. Copied llama.cpp code is MIT: retain copyright notices, add per-file provenance headers in `csrc/`, ship a NOTICE file. gguf-py (MIT, vendored/pinned) is a hard dependency. Pick MIT or Apache-2.0 for llama-farm itself so downstream llama.cpp upstreaming stays frictionless.

---

## 8. Hard constraints to enforce/document (v1)

| Constraint | Reason |
|---|---|
| Flash attention OFF during training | no FA backward (`ggml.c:5470`) |
| No ALiBi models | `soft_max_back` requires `max_bias==0` |
| KV cache F32 | CPU F16 `OUT_PROD` aborts (`ops.cpp:4487-4491`) |
| LoRA A/B tensors F32 | `ggml_set_param` path + optimizer kernels are F32-only |
| Fixed ubatch shapes across steps | dynamic-graph state keyed by node index; varying batch asserts (`ggml-opt.cpp:851`) |
| Expect CPU fallback for quantized backward on GPU until kernel #2 | CUDA `OUT_PROD` F32-only |
| MoE / SSM / RWKV / LayerNorm archs: blocked at preflight with actionable errors | missing VJPs (G8) |
| NVFP4 (4-bit float quant) bases with per-tensor scales: refuse LoRA | asserts in `build_ffn` (`llama-graph.cpp:1595-1600`) |
| Single compute device + CPU host in v1 | multi-GPU/grad-placement deferred (§3, §10) |
| No new special tokens / vocab resize in v1 | base embeddings frozen & quantized; embedding training deferred to P4 — adapters must use the base vocab. Loss-mask boundaries must be computed against the model's actual template-token behavior (round-trip test in data layer) |
| token_embd/lm_head LoRA: supported for inference-format adapters; training them needs the embedding FIXMEs revisited | `llama-context.cpp:3207` |

**v1 model coverage:** dense RMS-norm transformers (llama, qwen2/3, phi3, mistral, granite, olmo2, smollm3, …) are fully trainable today given the constraints above — their graphs use only backward-covered ops. **Not** gemma2 (nor gemma3 GGUFs with `final_logit_softcapping` set): softcapping applies `ggml_tanh`, which has **no backward** (unary default aborts, `ggml.c:6872-6876`) — a trivial upstream VJP (`grad · (1 − tanh²)`, item #5 in §7) unlocks them; until then the D5 graph-walk preflight rejects them with a clear error. GELU-MLP variants likewise wait on VJP item #5.

**Platform status (v1):**

| Platform | Status | Bottleneck / fix |
|---|---|---|
| Linux CUDA | works | backward through quantized weights falls to CPU until kernel #2 (P2) |
| CPU (Linux/macOS/Windows) | fully works | slowest; the P0 baseline |
| macOS Metal | forward on GPU only | Metal has no `OUT_PROD`/CE kernels → backward is CPU-bound; Metal kernels post-P2 |
| Windows | expected to work, untested | shim builds against private C++ internals — make MSVC a CI target from P1 |

---

## 9. Phased roadmap

- **P0 — Proof of gradient (1 milestone):** ctypes bring-up; zero-init adapter via gguf-py; shim `opt_init_lora` + forked step loop; SFT with stopgap composite CE on a small Q4_K llama-arch model, CPU. Exit test: loss falls; trained adapter loads in stock `llama-cli --lora`; finite-difference grad check on A/B passes.
- **P1 — SFT for real:** `ce_sparse` op (CPU+CUDA); masking/packing/chat-template data layer; optimizer-state checkpoint/resume (sidecar format, G14); grad clipping (G15); adapter save/merge; trainability preflight + arch report; masked validation-loss eval; tiny-model convergence test with loss-curve tolerance vs a recorded PEFT reference; benches vs `examples/training/finetune`; Windows/MSVC CI.
- **P2 — Performance:** quantized `OUT_PROD` CUDA kernel; chunked lm_head/CE; grad checkpointing; mmap-preserving LoRA path; Metal/Vulkan triage.
- **P3 — Preference methods:** DPO (offline ref logprobs → softplus loss), then GRPO (rollout engine, advantage plumbing, clip-via-relu graph, self-verification harness).
- **P4 — Breadth:** upstream small VJPs (CLAMP/SIGMOID/GELU/NORM); MUL_MAT_ID backward (MoE); embedding/lm_head training; DoRA/rsLoRA (needs `get_scale`/`build_lora_mm` changes upstream); FA backward.

---

## 10. Key risks & open questions

1. **Fork-vs-upstream tension.** The shim compiles against `src/` internals, which move. Mitigations: pin submodule; keep the shim surface minimal; upstream everything upstreamable (VJPs, kernels, maybe `llama_opt_init_lora` itself — likely welcome given the FIXMEs). `src/llama-ext.h` shows the vendored checkout already stages experimental API exactly this way (see the Layer-0 lineage note in §3).
2. **Backward-through-quantized correctness/precision** across quant types (IQ*, K-quants) is engine-supported but lightly exercised — validate early with the P0 finite-difference test per quant type and backend.
3. **Dynamic-graph state binding** (node-index keyed) is fragile; any conditional graph structure silently misbinds optimizer state. Enforce fixed shapes; consider upstreaming name-keyed state.
4. **GRPO throughput** hinges on rollout speed (llama.cpp is strong here) and the graph-rebuild cost when toggling adapters for ref passes (adapter-set changes force graph rebuild — batch ref passes; prefer offline/precomputed logprobs).
5. **Memory ceiling without checkpointing** limits context length in P0/P1 (README reports 24 GB for a 1B F32 full-FT at n_ctx 512; LoRA on a quantized base is far lighter but activations still dominate at long ctx). Back-of-envelope for sizing docs: total ≈ mmap'd quantized base + activations (~`n_layers × n_ubatch × (c₁·n_embd + n_head·n_ctx)` × 4 B under softmax attention — the `n_head·n_ctx` attention-matrix term is why no-FA hurts) + logits fwd+grad (`2 × n_vocab × n_ubatch × 4 B` until chunked CE) + LoRA params × 12 B (grad + AdamW m + v) + F32 KV cache. Produce a worked example per P0/P1/P2 config when specifying milestones.
6. **Open:** multi-GPU training (layer-split grad placement — deferred, §3)? Train norm vectors as an option (adapter format ignores them today)? Multi-adapter/mixed-task training? 8-bit optimizer states (moments are tiny for LoRA — probably never needed)? Windows story for the shim?

---

## Appendix A — What each existing component contributes (copy/reuse/edit map)

| llama.cpp component | Use |
|---|---|
| `ggml-opt.{h,cpp}` | reuse via API; copy high-level loop pieces into shim (header-sanctioned) |
| `llama_context::opt_epoch_iter` | copy into shim, parameterize loss + inputs |
| `llama-adapter.{h,cpp}` | reuse loader; copy `init_impl` as basis of `create_zero`; expose enumerate/get/set |
| `build_lora_mm` / `build_lora_mm_id` | reuse untouched — the heart of arch inheritance |
| `convert_lora_to_gguf.py` | copy format-writing logic into `adapter.py` (gguf-py based) |
| `tools/export-lora` | copy merge graph; extend to re-quantize output |
| `gguf-py` | depend on directly (pin to vendored commit) |
| `common_opt_dataset_init` | reimplement in numpy (10 lines) |
| `tests/test-backend-ops` MODE_GRAD | copy finite-difference harness for new VJPs/kernels |
| `tests/test-lora-conversion-inference.sh` | mirror as end-to-end adapter fidelity test |

| unsloth idea | ggml translation |
|---|---|
| Fused/chunked CE from hidden states | `ce_sparse` op + chunked lm_head host loop |
| Offloaded gradient checkpointing | shim-level per-layer graph segmentation + pinned host ring buffer |
| Fused LoRA backward (rank-r chains, in-place dX) | graph design (automatic) + optional fused epilogue op |
| Padding-free packing + boundary masking | llama.cpp seq_ids natively + Python collator |
| GRPO chunked selective log-softmax; hidden-states protocol | existing ggml ops, host loop |
| Ref model via adapter-disable | `llama_set_adapters_lora` scale toggle / offline precompute |
| Two-tier arch support; strict-eligibility fast path with graceful fallback | preflight + tier model (D5) |
| Self-verification of exactness-preserving optimizations | first-use numeric compare + persistent fallback |
