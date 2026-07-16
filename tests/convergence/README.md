# The convergence gate

**What it asserts:** that the whole training stack — collation, the mask shift, the graph, the
backward, AdamW — computes the *right* thing, not merely a thing whose loss falls.

Run it:

```bash
pytest tests/test_convergence.py                 # the gate: reference audit, one step, 40-step curves
pytest tests/test_convergence_variants.py        # the same gate, off the recorded config (S1-38)
pytest tests/test_determinism.py                 # thread-count bit-identity
pytest tests/test_convergence.py --device cuda   # a backend lane (S2-01/S3-01/S4-01)
```

(Nothing here is `slow`-marked: the three files together run in ~13 s, and a gate that only runs
nightly tells you which of the day's twenty PRs broke it.)

Nothing here needs `torch`. The PEFT curve is **recorded once** and committed as
`reference_curve.json`; the gate only reads it.

## Two oracles, answering different questions

| | what it is | what only it can catch |
|---|---|---|
| **`tests/reference_llama.py`** | the same architecture and weights in **float64**, backward derived by hand from the maths | a gradient wrong by a *fraction of a percent* — which still makes the loss fall, and which no tolerance band can see |
| **`reference_curve.json`** | **transformers + peft**, recorded once on a checkpoint proved to be the same model | the graph and the float64 reference being wrong **together** — a shared misunderstanding of what LoRA SFT is |

The float64 reference audits *itself* first
(`test_the_reference_backward_is_itself_correct`): its analytic backward is compared against a
central finite difference of its own forward. A hand-derived VJP is exactly as likely to be wrong
as the one it is auditing, and a wrong oracle does not fail loudly — it silently redefines
"correct".

## The bands, and why they are so much tighter than the ticket expected

S1-12 assumed per-step exact match was impossible and prescribed windowed tolerance bands. It is
not impossible. Measured over 40 optimizer steps:

| comparison | worst per-step deviation |
|---|---|
| llama.cpp (F32) vs the float64 reference | **5.3e-07** |
| llama.cpp (F32) vs PEFT | **9.5e-07** |
| the float64 reference vs PEFT | **7.9e-07** |
| llama.cpp (Q8_0) vs the F32 reference | 4.1e-02 |

Three independent implementations agree to ~1e-6. The gate is therefore **per-step**, not
windowed — which matters, because a band wide enough to absorb kernel-numerics drift is wide
enough to hide a real gradient bug, and that is the exact class of bug this project keeps finding.

| tolerance | value | observed | margin |
|---|---|---|---|
| `CURVE_TOL` (F32 vs float64) | 1e-4 | 5.3e-7 | ~190× |
| `PEFT_TOL` (F32 vs PEFT) | 2e-4 | 9.5e-7 | ~210× |
| `GRAD_TOL` (per LoRA tensor) | 1e-3 | 2.0e-6 | ~500× |
| `QUANT_STEP_TOL` (Q8_0, per step) | 0.15 | 4.1e-2 | ~3.6× |
| `QUANT_WINDOW_TOL` (Q8_0, windowed mean) | 0.03 | 3.5e-3 | ~8.5× |

The margins exist for **one** reason: a different host's SIMD width changes the float32 reduction
order (ADR-0002), and that cannot be measured from here. It is *not* thread count — the curve is
**bit-identical across 1, 2 and 4 threads** on this host, and since S1-38 that claim is asserted
(`tests/test_determinism.py`), not just stated.

## Off the recorded config: the variants (S1-38)

The recorded config has three built-in coincidences — `alpha == rank` (effective scale exactly
1.0), `weight_decay == 0`, and a single rank — that make it numerically blind to a dropped or
doubled `alpha/rank` factor, an ignored `user_scale`, an unplumbed decay term, and rank-dependent
shape errors. `tests/test_convergence_variants.py` re-runs the gate with one knob at a time moved
off the recorded value (`RunSpec` in `config.py`); each variant gets the one-step all-gradients
check plus a curve band, and must prove its own separation from the recorded trajectory so it
cannot silently become decorative.

One measured surprise worth knowing: at effective scale 2.5 the f32-vs-f64 divergence compounds to
**5.7e-04 by step 35 with exactly-correct gradients** (verified at 6.2e-06 by the one-step check).
Trajectory conditioning depends on the config, which is why each variant's band is its own
measurement rather than `CURVE_TOL`, and why the sharp per-variant claim rests on the one-step
gradients — the same division of labor as the Q8_0 case.

**Q8_0 is looser on purpose and for a different reason.** Quantizing the base perturbs the logits,
so a Q8_0 run is a genuinely different model whose trajectory diverges in the small; PEFT cannot
run a GGUF at all, so there is no quantized reference to record. What the Q8_0 case asserts is that
a quantized base trains along the *same trajectory*, not that it lands on the same number. The
gradient's correctness is established by the F32 case, at 1e-4.

### The gate is mutation-tested

A gate that passes proves nothing until it can fail. Every bug class it exists to catch was
injected into the reference; all seven were caught:

| injected bug | caught by |
|---|---|
| RoPE: interleaved → split-half (HF's convention) | 5 tests |
| RMSNorm backward: drop the through-the-norm term | 4 tests |
| loss normalized by token count instead of `sum(w)` | 3 tests |
| `silu'` replaced by `sigmoid` (plausible, and wrong) | 3 tests |
| **one LoRA gradient off by 0.1%** | 3 tests |
| AdamW: `eps` inside the `sqrt` instead of outside | the 40-step curve |
| AdamW: bias-correction counter starts at 0, not 1 | the 40-step curve |

The last two are caught *only* by the curve, and correctly so — a single step at `lr = 1e-30` does
not exercise the optimizer's trajectory.

## The four things that had to be twinned

Each of these silently makes the reference a *different model*, which shows up as a plausible drift
that invites you to widen the tolerance until it passes.

1. **The RoPE convention.** GGUF's llama arch rotates **interleaved adjacent** pairs
   `(x[2k], x[2k+1])`; HF's `rotate_half` rotates **split-half** pairs `(x[k], x[k+d/2])`. These
   are different functions of the same weights. `hf_twin.py` applies the inverse of
   `convert_hf_to_gguf.py`'s row permutation — and `hf_twin.verify()` then *checks* it, by
   comparing the twin's logits against the float64 reference before a single step is recorded. A
   wrong permutation lands in the ones, not the millionths. (The recorded twin deviation is
   **8.0e-07**.)

   The LoRA tensors need **no** permutation: `A` acts on the un-permuted input side, and `B` is
   zero at init — its gradient stays permutation-related thereafter, and AdamW is elementwise, so
   the two runs have *identical losses* while their `B` tensors differ by a row permutation.

2. **The LoRA scale.** llama.cpp:
   `scale = alpha ? user_scale * alpha / rank : user_scale` (`llama-adapter.h:52-57`) — note that
   `alpha == 0` does not mean "scale by zero", it means the `alpha/rank` factor is *dropped*.
   PEFT: `scaling = lora_alpha / r`. Setting `alpha = rank` makes both exactly 1.0 and sidesteps
   the branch.

3. **The LoRA A init.** PEFT uses `kaiming_uniform(a=sqrt(5))`; learning-llamas uses
   `normal(0, 1/sqrt(r))`. Neither is wrong and they are not the same, so the recorder
   **overwrites** PEFT's A with the fixture adapter's own A. (`B` is zero on both sides by
   construction — which is what makes a fresh adapter an exact no-op.)

4. **The loss normalization.** The shim folds `1/sum(w)` into the per-token weights host-side, so
   the loss is a weighted *mean over the tokens that carry weight*. Not a mean over all tokens, and
   not a sum. `lora_dropout` is 0 for the same family of reason: a nonzero dropout makes the
   reference non-deterministic and no seeding fixes that.

## Regenerating `reference_curve.json`

The curve embeds an **identity hash** over the fixture generator, the config and the dataset. If
any of them changes, the gate fails loudly with "recorded against a different experiment" rather
than comparing against a curve that describes something else. That is when — and the only time —
you regenerate.

`torch` and `peft` are **not** dependencies of this project, and must not become any. Record in a
throwaway environment:

```bash
uv venv /tmp/recorder
VIRTUAL_ENV=/tmp/recorder uv pip install --index-url https://download.pytorch.org/whl/cpu torch
VIRTUAL_ENV=/tmp/recorder uv pip install peft transformers gguf jinja2

PYTHONPATH=$PWD:$PWD/src /tmp/recorder/bin/python -m tests.convergence.record_reference
```

It prints the twin's logit deviation, refuses to record if the twin is not the same model, and
writes the JSON. Commit it. In CI, the `record-reference` `workflow_dispatch` job does the same and
uploads the result as an artifact.

There is a **second** recorded curve, `reference_curve_wd.json`, identical to the first but with
`weight_decay = 1.0` — the only PEFT curve that exercises AdamW's decoupled decay term (the
recorded config has `weight_decay = 0`, so its decay multiplies by exactly 1). It has its own
identity hash (`config.variant_identity(config.WD_SPEC, ...)`) and its own gate,
`tests/test_convergence_variants.py::test_the_weight_decay_curve_matches_the_recorded_peft_reference`.
Regenerate it with the `--wd` flag; it writes only that file and never touches `reference_curve.json`:

```bash
PYTHONPATH=$PWD:$PWD/src /tmp/recorder/bin/python -m tests.convergence.record_reference --wd
```

The recorder needs **no compiled shim** — it only builds an F32 fixture and a zero adapter, both
pure `gguf-py` — so the project source on `PYTHONPATH` is enough.

## `--device`

ROADMAP §11 makes this gate the phase-exit criterion for every backend stage, so it is
backend-parameterized from day one and S2-01/S3-01/S4-01 reuse it with one flag rather than forking
it. A device the build does not have **skips** — it does not fail, and it does not quietly run on
the CPU while reporting success. `test_the_report_records_the_device_it_actually_ran_on` writes
`tests/.report/convergence-<device>.json` with the device requested and the devices ggml actually
registered.

There is deliberately **no sched-split fallback count yet**. On a CPU-only build every op runs on
the CPU by definition, so the count is identically zero and reports nothing; emitting it would be
worse than omitting it. S2-01 adds the accessor when there is a second backend for it to be about.
