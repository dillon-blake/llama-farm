# The carried llama.cpp diff

ADR-0001 commits this project to rebasing its llama.cpp fork against upstream on a **monthly**
cadence. Until now that rebase had no map: eighteen merged fork PRs, and no single document saying
what they change, why, or which of them belong upstream rather than here.

This is that map. Regenerate its facts with:

```bash
cd vendor/llama.cpp
git log --oneline 4f37f51..HEAD          # the carried commits
git diff --stat 4f37f51..HEAD            # the carried diff
```

## What is carried

| | |
|---|---|
| Upstream base (pinned) | `4f37f51` |
| Fork | `dillon-blake/llama.cpp`, branch `learning-llamas-base` |
| Merge-base with the pin | **exactly `4f37f51`** — a clean linear stack, no upstream merges mixed in |
| Carried commits | **18** substantive (one squashed commit per fork PR #1–#18) |
| Diff | **18 files changed, +2342 / −52** |
| New files | **zero** |

That last row is the finding. ADR-0001 §2 states its own rebase-hygiene rule — *"new functionality
goes in **new files** wherever it plausibly can"* — and **the fork has not followed it once**. The
sparse-CE kernels (202 lines) went into `ggml/src/ggml-cpu/ops.cpp`, a file upstream has touched 66
times in the last twelve months, rather than into a new `ops-ce-sparse.cpp` that would rebase for
free. Every one of the 18 files is a modification to a file upstream already owns.

The rule that *is* being followed: the new op enums (`GGML_OP_CROSS_ENTROPY_LOSS_SPARSE` and its
`_BACK`) are appended at the enum tail immediately before `GGML_OP_COUNT`, with an in-code comment
citing the reason.

## The commits

Disposition is assessed **per commit**, not inherited from the ticket. That distinction matters:
six of these are upstream bugs found *incidentally* while building a product feature, so the
ticket's declared disposition describes its deliverable and not the fix.

| Hash | Ticket | What it changes, and why | Disposition | Rebase risk |
|---|---|---|---|---|
| `c47889b9c` | S1-16 | **`op_expm1` was `expf(x) - 1.0f`** — precisely the catastrophic cancellation the op exists to avoid. At `x=5e-5` the k3 KL estimator goes **negative** (−5.13e-8 vs a true +1.25e-9), so a KL penalty starts *rewarding* divergence — and `x≈0` is exactly where on-policy RL lives. Now `expm1f`, which was already used twice in the same file. | **upstream-early** | LOW |
| `79e37988c` | S1-03 | With dynamic graphs, ggml-opt forces accumulation on but **nothing zeroed the accumulators** — every step summed onto the last and the optimizer descended on a running total. Also affects upstream's own `llama-finetune` whenever `n_batch == n_ubatch`. | **upstream-early** | LOW |
| `94b04d328` | S1-05 | `ggml_opt_eval` advanced `opt_i` on *forward-only* evals, so "train, validate, train" at `opt_period=2` left the window misaligned and **the optimizer never stepped at all**. | **upstream-early** | LOW |
| `ce78dc2ec` | S1-20 | Deletes the `SOFT_MAX_BACK max_bias == 0` assert. ALiBi's bias is additive, so it drops out of the Jacobian and the kernel never read `max_bias` anyway — the assert guarded nothing and made every ALiBi model untrainable. | **upstream-early** | LOW–MED |
| `2126ae199` | S1-18 | Routes F16/BF16 `out_prod` through the existing `_q_f32` path instead of `GGML_ABORT`. The F16 case had **no `break` after its abort**, so the naive fix would have silently read F16 bytes as F32. | **upstream-early** | MED |
| `eee8346a7` | S1-19 | TANH/SIGMOID/CLAMP VJPs — **and fixes `ggml_clamp` itself**, which returned `ggml_view_tensor(a)`, so any clamp whose gradient was requested *aborted*. | **upstream-early** | MED |
| `5608bb9aa` | S1-29 | CONCAT VJP. Its grad test was vacuous three independent ways (no `set_param`; shapes above `grad_nmax`; and routing ops have a constant gradient under a `sum(out)` objective). | **upstream-early** | MED–HIGH |
| `245f8c275` | S1-34 | A `grad_loss` hook on `test_case`, because **MODE_GRAD differentiates `sum(out)` and a softmax's rows sum to 1 by construction** — `d(sum(out))/dx ≡ 0`, so the SOFT_MAX grad test had never exercised `SOFT_MAX_BACK` at all. Also makes the missing `dL/d(sinks)` rule abort loudly instead of silently returning zero. | **upstream-early** | **HIGH** |
| `063661f95` | S0-10 | Skips the view+param inplace cases `ggml_build_backward_expand` asserts on, so `test-backend-ops grad` completes instead of hard-aborting mid-sweep. | **upstream-early** | **HIGH** |
| `039dd4ff2` | S1-04 | `ggml_cross_entropy_loss_sparse`: I32 labels + F32 per-token weights → an unreduced per-token loss vector. Replaces a 1 GB one-hot matrix and is what makes `w=0` masking expressible. | upstream-later | MED–HIGH |
| `c1bfd273f` | S1-10 | In-graph global-norm gradient clipping. It **must** be in-graph: the optimizer step is fused into the backward, so by the time a host callback could run, the weights have already moved. Adds `grad_clip` to the public `ggml_opt_params` — see the ABI note below. | upstream-later | LOW |
| `9c2c8188c` | S1-10 | Reorders the clip to `factor = clip/clamp(norm,clip,INF)` so a clip *above* the norm is a bit-for-bit identity. Previously a clip of 1e6 ("off") perturbed real LoRA weights in the 7th decimal. | upstream-later | LOW |
| `ef82649e3` | S1-09 | `ggml_opt_grad_m/_v` and `ggml_opt_get/set_iter` — the AdamW moments and the bias-correction counter a resume has to restore. Pure additions. | upstream-later | LOW |
| `fd4933986` | S1-17 | **Two independent changes.** (a) `ggml_build_backward_expand_checkpointed` restores gradient checkpointing, which ggml *had* and lost in the 2024 backward refactor. (b) `graph_max_nodes() *= 4` when training, and `set_training()` re-creates the graph buffers and scheduler — a training graph is forward *and* backward, and the inference-derived bound aborts at 12 layers. **(b) was a live bug: deep models could not train at all, with checkpointing off.** | (a) upstream-later, (b) **upstream-early** | **HIGH** |
| `f0fad422e` | S1-00 | Adds `cparams.training`; `build_attn` then attends over `k_cur`/`v_cur` directly rather than through the KV cache, whose `ggml_set_rows` result is a *view* that severs the autodiff edge to K/V and aborts the backward. **This is the root of the whole project** — upstream's own `llama-finetune` aborts before printing a loss. | **disputed** — see below | **HIGH** |
| `107a016eb` | S1-02 | `opt_step_custom`: a fork of `opt_epoch_iter` where the caller owns the `ggml_opt` context and builds the loss node, exploiting `LOSS_TYPE_SUM` (summing a scalar is the identity) so the loss becomes pluggable with no ggml change. | fork-local | **HIGH** |
| `78c3e18bc` | S1-14 | `opt_step_custom` left `gf_res_prev` (a one-entry graph cache) pointing at nodes in `ctx_compute_opt`, which is freed each step — so a later `llama_decode` on the same context walked freed memory. Fixes fork-local code. | fork-local | LOW |
| `ffb7d06e7` | S1-33 | `LLAMA_API_INTERNAL` on the nine internals the shim links against. A PE DLL exports nothing it was not told to, so the same source that links fine on ELF produced eight `LNK2019`s on Windows. | fork-local | LOW–MED |

**Nine of eighteen are upstream-early**: general bug fixes upstream would plausibly want, six of
them found incidentally. ADR-0001's own rule is *"PR to mainline **first**"*, and it has been
followed **zero times**. That is real value sitting in a private fork, and it is permanent rebase
cost paid monthly for defects that are not this project's to own.

> ⚠️ **Upstreaming has a constraint ADR-0001 does not account for.** llama.cpp's `AGENTS.md` states:
> *"This project does **not** accept pull requests that are fully or predominantly AI-generated."*
> Contributors must understand their code fully and be able to explain any change to a reviewer
> without AI assistance. **Private forks are explicitly exempt**, so everything on
> `learning-llamas-base` is fine as-is — but an upstream PR is a human commitment, not a
> cherry-pick. Budget for that, and start with `c47889b9c` (`op_expm1` → `expm1f`): it is one hunk,
> its correctness argument fits in a paragraph, and it is a plain numerical bug fix.

## Where a rebase will hurt

Risk is upstream churn (commits touching the file in the last 12 months) against the fork's
footprint in it.

| | Files |
|---|---|
| **HIGH** | `tests/test-backend-ops.cpp` (churn **221**, 16 hunks) · `src/llama-context.cpp` (111, 4) · `src/llama-graph.cpp` (103, 6) · `ggml/src/ggml-cpu/ops.cpp` (66, 4) · `ggml/src/ggml.c` (51, 13) |
| **MED** | `src/llama-model.h` (63) · `src/llama-graph.h` (48) · `ggml/src/ggml-cpu/ggml-cpu.c` (44) · `ggml/include/ggml.h` (43) · `src/llama-context.h` (31) · `src/llama-cparams.h` (16, **a shared struct**) |
| **LOW** | `ops.h` (12) · `ggml-cpu.cpp` (11) · `llama-impl.h` (8) · `test-opt.cpp` (5) · `unary-ops.cpp` (4) · `ggml-opt.cpp` (**2**) · `ggml-opt.h` (**2**) |

The counterintuitive one: **`ggml-opt.cpp` carries the fork's single largest footprint — 17 hunks,
209 lines — and is its *lowest* rebase risk.** Upstream touched it twice in a year. ggml-opt is
close to unmaintained, which is precisely why the fork found four real bugs in it.

Conversely `tests/test-backend-ops.cpp` is the most rebase-exposed file by a factor of two, and the
`grad_loss` hook is an invasive change to the harness's `test_case` base class. That is where a
rebase will hurt first, and it is a *test* file — so it will hurt without the compiler's help.

## An ABI landmine, found and defused

`c1bfd273f` inserted `float grad_clip` into the **middle** of the public `struct ggml_opt_params`,
between `opt_period` and `get_opt_pars`. The ctypes mirror in `_ffi/ggml_opt.py` did not declare
it — and that was harmless **purely by accident**:

```
                mirror   fork
opt_period        @40    @40
grad_clip       (absent) @44   <- lands exactly in the mirror's alignment padding
get_opt_pars      @48    @48
optimizer         @64    @64
sizeof             72     72
```

A float at offset 44 falls into the padding already sitting between an int32 at 40 and an 8-aligned
pointer at 48. Every later field still aligned; `sizeof` still came to 72; ctypes zero-fills, and
`0.0f` happens to mean "clipping disabled". Nothing was wrong, and nothing was guarding it.

The next field added anywhere before `get_opt_pars` would have shifted `get_opt_pars` by four
bytes — and that field is a **function pointer**. The failure mode of a misaligned function pointer
is not a wrong number, it is a call to a wrong address: exactly the *"silent memory corruption, not
a link error"* that ADR-0001's Context section says the ctypes-mirror discipline exists to prevent.

Now: the field is declared, `csrc/farm_internals.cpp` pins the C offsets with `static_assert` and
`offsetof`, and `test_the_opt_params_mirror_matches_the_C_layout` pins the Python ones. Break either
side and the other complains.

**The general rule this implies:** a fork change that adds a field to a public ggml or llama struct
must add it to the `_ffi` mirror *in the same PR*, and pin it. Appending at the tail is safer than
inserting, for the same reason the enum rule exists.

## Two contradictions the docs have not resolved

**1. S1-00's disposition.** Two authoritative documents disagree about the most rebase-exposed
change in the fork:

- `tickets/stage-1-cpu/S1-00-...md:173` — *"**upstream-early** — this is a straight bug fix to a
  broken upstream example, and carrying it fork-local indefinitely fights every rebase."*
- `docs/adr/ADR-0001-vendor-lineage.md` §2 — lists S1-00 under **`fork-local`**: *"The
  training-graph KV-cache bypass (S1-00) **in its project-specific form**."*

These may be reconcilable — the *defect* is upstream's, while the *shape of the fix* (a `cparams`
flag plus a rewired `build_attn`) is this project's. But "may be reconcilable" is not a decision,
and this is the change that will cost the most at every rebase. **It needs one.**

**2. The disposition vocabulary has two incompatible spellings.** ADR-0001's table calls the middle
class **`in-fork-first`**. The ticket template and ~31 tickets call it **`upstream-later`**. Both
share `upstream-early` and `fork-local`. Nothing anywhere defines both.

Recommendation: standardize on **`upstream-early` / `upstream-later` / `fork-local`** — the spelling
the tickets overwhelmingly already use — and amend ADR-0001, noting `in-fork-first` as its former
name.
