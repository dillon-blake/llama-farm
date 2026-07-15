# The carried llama.cpp diff

ADR-0001 commits this project to rebasing its llama.cpp fork against upstream on a **monthly**
cadence. Until now that rebase had no map: a stack of carried fork commits, and no single document
saying what they change, why, or which of them belong upstream rather than here.

This is that map. Regenerate — and *verify* — its facts with the commands below. Run each one; each
reproduces exactly one number cited in the next section, so the doc audits itself:

```bash
cd vendor/llama.cpp
git merge-base HEAD 4f37f51                             # 4f37f519722a...  (must equal the pin)
git log --oneline --no-merges 4f37f51..HEAD | wc -l     # 31   substantive commits
git log --oneline           4f37f51..HEAD | wc -l       # 49   = 31 + 18 early-PR merge commits
git diff --stat        4f37f51..HEAD | tail -1          # 22 files changed, +4900 / -97
git diff --name-status 4f37f51..HEAD | grep -c '^A'     # 2    new files
```

> **The count depends on `--no-merges`, and the old regenerate recipe left it off.** The first
> eighteen fork changes each landed through a squash-merged PR, so the plain `git log 4f37f51..HEAD`
> carries **18 merge commits** on top of the substantive ones; the thirteen kernel-era commits
> (S1-24 … S1-47) were committed straight onto the base branch and have no merge commit. So the bare
> log prints **49**, not the 31 substantive changes the inventory is about — which is precisely how
> the previous version of this doc came to disagree with its own regenerate command. Count with
> `--no-merges`.

## What is carried

| | |
|---|---|
| Upstream base (pinned) | `4f37f51` |
| Fork | `dillon-blake/llama.cpp`, branch `learning-llamas-base` (HEAD `555c446` on `ticket/S1-24-chunked-attention`) |
| Merge-base with the pin | **exactly `4f37f51`** — a clean linear stack, no upstream merges mixed in |
| Carried commits | **31** substantive (18 via early squash-merged PRs #1–#18, then 13 committed straight onto the branch as the kernel work landed) |
| Diff | **22 files changed, +4900 / −97** |
| New files | **2** — both are *test* files: `tests/test-glu-back.cpp` (S1-28) and `tests/test-soft-max-back-inplace.cpp` (S1-41) |

That last row is still most of the finding. ADR-0001 §2 states its own rebase-hygiene rule — *"new
functionality goes in **new files** wherever it plausibly can"* — and for **kernels** the fork has
not followed it once: `ggml-cpu/ops.cpp` alone has grown by **+1020 lines** (sparse-CE, `OUT_PROD_ID`,
`OUT_PROD_ID_GRP`, `GLU_BACK`, `SSM_CONV_BACK`, `SSM_SCAN_BACK`), a file upstream has touched 66 times
in the last twelve months, rather than into new `ops-*.cpp` files that would rebase for free. The two
new files that *do* exist are both regression tests, not kernels — so the rule is honoured only where
it costs nothing and broken everywhere it would actually save rebase work. Every carried *kernel* is a
modification to a file upstream already owns.

The rule that *is* being followed: the seven new op enums the fork appends —
`GGML_OP_CROSS_ENTROPY_LOSS_SPARSE` and its `_BACK`, `GGML_OP_OUT_PROD_ID`, `GGML_OP_OUT_PROD_ID_GRP`,
`GGML_OP_GLU_BACK`, `GGML_OP_SSM_CONV_BACK`, `GGML_OP_SSM_SCAN_BACK` — all sit at the enum tail
immediately before `GGML_OP_COUNT` (ggml.h:592–605), with in-code comments citing the reason, so a
rebase never has to renumber an upstream op.

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

The thirteen kernel-era commits (committed straight onto the branch, no PR merge commit):

| Hash | Ticket | What it changes, and why | Disposition | Rebase risk |
|---|---|---|---|---|
| `c8165770a` | S1-25 | **`MUL_MAT_ID` / `ADD_ID` backward wiring.** MoE's expert matmul and its per-expert bias-add had no VJP, so any MoE arch fell through `ggml_compute_backward`'s `default:` abort. Adds the backward rules and the `OUT_PROD_ID` / `OUT_PROD_ID_GRP` op enums at the tail. | fork-local | MED |
| `0d0afcb91` | S1-26 | **`OUT_PROD_ID` CPU kernel** — the activation half of `MUL_MAT_ID`'s backward (dL/d expert inputs). New op; scalar reference kernel. | fork-local (upstream-later) | MED–HIGH |
| `7b700fe19` | S1-27 | **`OUT_PROD_ID_GRP` CPU kernel** — the weight half (dL/d expert weights), reduced per expert. New op. | fork-local (upstream-later) | MED–HIGH |
| `b8e007507` | S1-26/27 | Both new kernels read `grad` and `ids` through a hard-coded 4-byte stride; a TRANSPOSE-node grad has `nb[0]==8` and `ggml_mul_mat_id` puts no contiguity constraint on `ids` — 24/36 elements wrong, no assert fires. Now read through `nb[0]`. Fixes fork-local code. | fork-local | LOW |
| `74a1db10b` | S1-28 | **`GLU_BACK`** — one VJP covering every gated-linear-unit variant (`REGLU`/`GEGLU`/`GEGLU_ERF`/`GEGLU_QUICK`/`SWIGLU_OAI`); only plain `SWIGLU` had a backward, so every other gated-FFN arch aborted. New op enum, and the fork's first genuinely-new source file, `tests/test-glu-back.cpp`. | fork-local (upstream-later) | MED |
| `8857bc8d4` | S1-28 | **Two more upstream ggml bugs that made every MoE model untrainable**, found incidentally while wiring `GLU_BACK`. Plain correctness fixes to existing code. | upstream-early | MED |
| `eda3a57c0` | S1-37 | Fixes `mean_abs_asymm`'s **signed** denominator (`(a−b)/(a+b)`, which sends any near-zero-gradient element to ±∞) and the `nvalid==0` NaN free-pass it was hiding in the MODE_GRAD harness. A test-harness correctness fix. | upstream-early | MED |
| `cc0116f03` | S1-29b | The `SSM_CONV_BACK` / `SSM_SCAN_BACK` op enums and backward-switch wiring that S1-29's squashed commit was supposed to carry and did not. | fork-local (upstream-later) | MED |
| `5fdae0c36` | S1-30 | **`SSM_CONV_BACK` CPU kernel** — the causal-conv backward for Mamba. New op. | fork-local (upstream-later) | MED–HIGH |
| `2c8091797` | S1-31 | **`SSM_SCAN_BACK` CPU kernel** — the selective-scan backward; Mamba has a backward at last. New op. | fork-local (upstream-later) | MED–HIGH |
| `2a276de8f` | S1-24 | **Chunked attention** — a kernel-free long-context path (2.17x less backward memory at n_ctx 4096, bit-identical to naive), built entirely in shim-side graph construction (`llama-graph.cpp` + a `cparams` flag), no new kernel. | fork-local | HIGH |
| `41141dd4f` | S1-41 | **`ggml_compute_forward_soft_max_ext_back_f32` was not in-place-safe onto `src1`** (it overwrites `dst` before its last read of `src1`). `SOFT_MAX_BACK` is on `ggml_op_can_inplace`, and gallocr aliases `dst` onto the softmax output `y` exactly when the back-op is `y`'s **sole** gradient consumer — never in attention, **always** in a Mixtral router — so `d_logits≈0` and every LoRA upstream of an MoE block trained on a gradient wrong by 5–25 %, loss falling the whole time. The S1-41 MoE oracle caught it; `tests/test-soft-max-back-inplace.cpp` forces the alias (0.22 error pre-fix, 9.6e-9 post). | **upstream-early** | **HIGH** |
| `555c44643` | S1-47 | **Two Mamba-training fixes.** (a) `ggml_ssm_scan`'s initial-state input is a *view of the recurrent-state cache*, which the forward overwrites in place; the scan backward recomputes intermediate states from it and so started from the **final** state — every Mamba layer but the last trained on a corrupted scan gradient. Fix: `ggml_cont` the initial state. (b) the VIEW-backward handed `ggml_acc` a non-contiguous grad (`nb[0]!=sizeof(float)`), aborting **every** Mamba training graph at backward — `ggml_cont` before `acc_or_set`, matching the neighbouring `RESHAPE` guard. Also adds the loud `n_group>1` refusal (unproven routing — see B-10). | fork-local (a); **upstream-early** (b) | **HIGH** |

**Twelve of the thirty-one are upstream-early**: general bug fixes upstream would plausibly want,
most of them found incidentally while building a product feature. ADR-0001's own rule is *"PR to
mainline **first**"*, and it has been followed **zero times**. That is real value sitting in a
private fork, and it is permanent rebase cost paid monthly for defects that are not this project's to
own. The kernel-era additions (`OUT_PROD_ID*`, `GLU_BACK`, `SSM_*_BACK`) are dispositioned
*fork-local for now, upstream-later*: they fill real gaps in ggml's autodiff, but they carry new op
enums and want to be proposed to upstream as one RFC once the whole MoE/SSM backward set is stable,
not cherry-picked piecemeal.

> ⚠️ **Upstreaming has a constraint ADR-0001 does not account for.** llama.cpp's `AGENTS.md` states:
> *"This project does **not** accept pull requests that are fully or predominantly AI-generated."*
> Contributors must understand their code fully and be able to explain any change to a reviewer
> without AI assistance. **Private forks are explicitly exempt**, so everything on
> `learning-llamas-base` is fine as-is — but an upstream PR is a human commitment, not a
> cherry-pick. Budget for that, and start with `c47889b9c` (`op_expm1` → `expm1f`): it is one hunk,
> its correctness argument fits in a paragraph, and it is a plain numerical bug fix.

## Where a rebase will hurt

Risk is upstream churn (commits touching the file in the last 12 months) against the fork's
footprint in it (lines / hunks from `git diff 4f37f51..HEAD`). Churn is measured against the pin and
so is unchanged since S1-36; the footprints have roughly doubled with the kernel work.

| | Files (churn · fork footprint) |
|---|---|
| **HIGH** | `tests/test-backend-ops.cpp` (churn **221** · **+1065, 37 hunks**) · `ggml/src/ggml-cpu/ops.cpp` (66 · **+1020, 8**) · `ggml/src/ggml.c` (51 · +849, 17) · `src/llama-context.cpp` (111 · +268, 4) · `src/llama-graph.cpp` (103 · +280, 7) |
| **MED** | `ggml/include/ggml.h` (43 · +179, 7) · `src/llama-model.h` (63 · +3) · `src/llama-graph.h` (48 · +3) · `ggml/src/ggml-cpu/ggml-cpu.c` (44 · +81, 8) · `src/llama-context.h` (31 · +83) · `src/llama-cparams.h` (16 · +25, **a shared struct**) |
| **LOW** | `ggml-opt.cpp` (**2** · +209, 11) · `test-opt.cpp` (**5** · +526, 3) · `ggml-cpu.cpp` (11 · +53) · `unary-ops.cpp` (4 · +21) · `llama-impl.h` (8 · +22) · `ops.h` (12 · +7) · `ggml-opt.h` (**2** · +44) · `models/mamba-base.cpp` (small · +14) · `tests/test-glu-back.cpp`, `tests/test-soft-max-back-inplace.cpp` (**new files, churn 0**) |

The counterintuitive ones still hold, and there are now two of them. **`ggml-opt.cpp` (+209, 11 hunks)
and `test-opt.cpp` (+526) are both large footprints at the *lowest* rebase risk** — upstream touched
each about twice in a year. ggml-opt is close to unmaintained, which is precisely why the fork found
four real bugs in it. And the fork's two brand-new files (`test-glu-back.cpp`,
`test-soft-max-back-inplace.cpp`) rebase for free by construction — the argument for the new-files
rule the kernels ignore.

Conversely `tests/test-backend-ops.cpp` (+1065) and `ggml-cpu/ops.cpp` (+1020) are now the two
largest footprints *and* two of the highest-churn files — test-backend-ops the most rebase-exposed
file in the tree by a wide margin, with the `grad_loss` hook an invasive change to the harness's
`test_case` base class, and ops.cpp carrying six new kernels in a file upstream reworks constantly.
That is where a rebase will hurt first, and for the test harness it will hurt without the compiler's
help.

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
