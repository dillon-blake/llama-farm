# Support tiers

BLUEPRINT D5 draws a two-tier support model, and the distinction it insists on is that **trainability
is decided by a graph walk, never by an architecture whitelist**. llama.cpp carries ~133
architectures whose forward graphs — including LoRA injection via `build_lora_mm` — come free by
construction. A whitelist of "supported" archs would rot the moment upstream adds one; a walk of the
*actual built graph* against the set of ops that have a backward rule is precise and survives new
archs automatically. The consequence D5 draws, and the one this document exists to state plainly:

> **"New arch" must mean _unoptimized_, never _unsupported_.**

The gate is the preflight, not this document. The preflight (`csrc/farm_preflight.cpp`,
`src/learning_llamas/preflight.py`, exposed as `FarmModel.preflight()`) builds the forward graph once
at load, walks its nodes against the supported-backward op table, and returns a per-node report of
`ok` / `blocked` / `warn`. Trainers call it before the first step and raise on any `blocked` node
(`src/learning_llamas/train/loop.py`). This file only names the two tiers; the preflight decides
which tier a given model falls into.

## Tier 1 — verified-fast

Configurations that are exercised by a committed fixture **and** pinned by a numerical oracle or the
convergence gate: their gradient is known correct per tensor, not merely "the loss falls". These are
the archs the project actively regression-tests every PR.

| Family | Fixture | What verifies it |
|---|---|---|
| Dense llama (SwiGLU FFN) | `gen_tiny_llama` (F32 / Q8_0 / Q4_K) | S1-12 convergence gate vs a recorded PEFT reference + a float64 numpy oracle; S1-03 full-graph finite-difference gate |
| Mixtral-style MoE | `gen_tiny_moe` | S1-41 MoE gradient oracle — all expert/router LoRA gradients matched to a self-audited float64 reference |
| Mamba-1 (`n_group == 1`, `head_dim == 1`, per-state `A`) | `gen_tiny_mamba` | S1-47 SSM training oracle — every LoRA gradient matched to float64 at ~1e-6, e2e loss falls |

Tier 1 grows one fixture + oracle at a time; adding a family here is a ticket, not a config flag.

## Tier 2 — passes-preflight

Anything whose forward graph walks clean — every op that sits on a path from a trainable parameter to
the loss has a backward rule — runs on the **generic** training path. It is supported: it will train.
It is simply not (yet) fixture-verified or perf-tuned, and it may run some ops on the CPU fallback
even on a backend build. A tier-2 model is a first-class training target; the only thing it lacks is a
tier-1 arch's committed proof that its specific composition is numerically pinned.

The preflight can also return `warn` (not `blocked`): a projection that bypasses `build_lora_mm` (one
of the ~75 raw `ggml_mul_mat` sites in exotic archs — DeepSeek-MLA, RWKV, gemma3n) silently receives
no LoRA injection. The model still trains; the warning names the target tensor whose adapter is inert.

## Blocked (neither tier)

A model is `blocked` when some op on its grad path has no backward rule. The preflight names the op,
the first offending node, and the ticket that unlocks it — e.g. an `n_group > 1` SSM arch (Mamba-2 /
Falcon-H1) aborts loudly today because its group-index routing has no gradient oracle (B-10). Blocked
is a *reported* state, never a silent `GGML_ABORT` mid-step, and it flips to tier-2 the moment the
unlocking ticket lands, with no change here.
