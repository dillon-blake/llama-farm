---
id: S1-50
title: "Minors sweep: actually close the 15 parked minor findings from the 2026-07-15 audit"
stage: 1
track: quality
size: M
deps: [S1-10, S1-04, S1-37, S1-12, S1-06, S1-07, S1-14, S1-17, S1-24, S1-31, S1-41, S1-47]
status: done
pr: null
---

# S1-50 — Minor-findings sweep

**One-line outcome:** the 15 findings S1-49 *parked* are now each in one of exactly two states —
IMPLEMENTED (fix/test/feature, with the repo's oracle-and-mutation discipline) or ACCEPTED
DEVIATION (argued, with what-would-change-it) — so nothing is left in limbo.

## Why (context)

S1-49 dispositioned the 2026-07-15 audit's 23 `minor/`-tagged findings: 8 fixed, 15 parked. "Parked"
was a holding state, not a resolution. This ticket takes each of the 15 to a real outcome. The
authoritative list is the "Minor-findings triage (2026-07-16)" section of
`docs/dev/audit-2026-07-15.md` (entries 9-23), now updated in place from "parked" to
implemented/accepted per item.

## Disposition (15)

Numbered as in the task; the audit-triage entry is in parentheses.

1. **(#9, L73) pre/post-clip norm getter — IMPLEMENTED.** ggml-opt now computes both global grad
   norms in the backward graph: pre-clip as `sqrt(sum sq)` of the unclipped grads (the existing
   `grad_norm` node), post-clip *measured off the clipped grads* it steps on (a new `grad_norm_post`
   reduction, not `pre*factor`). `ggml_opt_grad_norm_pre/post` read them back like the loss;
   `ll_grad_norms(ctx, pre, post)` surfaces them (csrc + `_ffi`). Tests: pre pinned to an independent
   host-side norm of the accumulators (the anti-tautology), `post <= pre`, `post == clip` when it
   binds, `post == pre` bit-identical when slack, both nonzero, and NaN when unclipped (no norm node
   built). Observed: pre matches host to ~1e-7; post lands on 0.01 as `0.00999999`.
2. **(#10, L93) S1-04 sparse-vs-dense forward parity — IMPLEMENTED.** `tests/test_ce_parity.py`
   builds a raw ggml graph computing both `ggml_cross_entropy_loss_sparse` (summed) and the dense
   `ggml_cross_entropy_loss` on the same logits, runs it on the CPU backend, and checks both against
   a float64 numpy reference within F32 round-off (observed ~1e-6 at n_vocab 4096), plus the two ggml
   ops against each other. A `test-backend-ops` case is the wrong venue: its value comparison is
   backend-vs-CPU, i.e. CPU-vs-CPU/vacuous on a CPU-only build. Mutation guard: unit weights make the
   summed loss `n_tokens x` the dense mean, proving the comparison discriminates.
3. **(#11, L95) `mean_abs_asymm` `nvalid==0` NaN free-pass — IMPLEMENTED.** The vendored harness now
   returns `+inf` when the `grad_expect` filter discards every element, so a grad-check that compared
   nothing FAILS loudly instead of passing on `0.0/0 = NaN` (`NaN > tol` is false). Protective: CLAMP
   still passes 16884/16884 (no registered case currently triggers `nvalid==0`), and the guard closes
   the same class S1-37 closed for the per-element `0/0`.
4. **(#12, L99) convergence AC#6 nightly-slow deviation — ACCEPTED DEVIATION.** The 40-step gate runs
   per-PR (unmarked), not nightly-`slow` as the AC lists. It is the *only* check that catches AdamW's
   trajectory numerics + the PEFT oracle, runs in ~7 s, and moving it nightly would report those bugs
   a day late against an unknown PR — worse than the ticket. Written up in
   `S1-12-convergence-gate-peft-reference.md`; would move to nightly if it ever became expensive.
5. **(#13, L101) weight-decay only cross-checked ref-vs-ggml — IMPLEMENTED.** Recorded a SECOND PEFT
   curve at `weight_decay=1.0` (`reference_curve_wd.json`, its own `variant_identity`, recorded on
   this box in a throwaway CPU-torch venv via `record_reference.py --wd`; the existing
   `reference_curve.json` is untouched). New gate
   `test_the_weight_decay_curve_matches_the_recorded_peft_reference` runs the ggml stack at `wd=1.0`
   and matches the PEFT-with-decay curve within `PEFT_TOL` — so AdamW's decoupled decay is finally
   cross-checked against the independent oracle, not just the float64 reference that shares the
   codebase. Twin deviation 8.0e-07; curves agree well within the band.
6. **(#14, L109) template-less GGUF error path — IMPLEMENTED.** `tests/test_template_less.py` builds a
   template-less model by copying the F32 fixture minus the single `tokenizer.chat_template` key
   (reusing `export`'s byte-exact tensor copy), loads it, and drives `ChatTemplate.from_model`'s NULL
   branch to the documented `ValueError`. Editing `gen_tiny_llama.py` was off the table (its
   `cache_key` is baked into the convergence identity); post-processing a copy sidesteps that. Mutation
   guard: the original fixture still yields a template.
7. **(#15, L111) S1-07 throughput counters — IMPLEMENTED.** `Batch` now carries `pad_count` and
   `n_samples` (the packer and the SFT/DPO collators set them; nothing downstream can recover a pad
   from a masked prompt, both are weight 0). `StepMetrics` gains `n_tokens`/`pad_tokens`/`n_samples`
   and derived `pad_fraction`/`valid_token_fraction`/`samples_per_pack`, threaded through
   `Trainer.record` into the logging-hook payload. Tests: the counts ride the packed Batch, and the
   hook receives them matching the batch (non-vacuous: the batch is genuinely mostly padding).
8. **(#16, L113) special-token one-token test — IMPLEMENTED (already in-tree).** The finding was stale:
   `test_special_token_text_is_parsed_as_one_token` already tests the fixture's real BOS `<s>` (not
   the absent `<|im_start|>`), so `parse_special` has a real token to collapse, with a sibling
   mutation guard. Verified passing; triage relabelled.
9. **(#17, L115) `mask.py` per-message prefix validation — IMPLEMENTED.** `build_masked_sample` now
   checks the token-prefix property at EVERY message boundary (each incremental prefix, the
   pre-completion prefix, the full conversation), not only the prompt/completion boundary — matching
   its own docstring and S1-06 item 4. Mutation guard: a synthetic merge injected at the *first*
   message boundary (loss boundary left clean) is caught and named, the exact case the old
   single-boundary check missed.
10. **(#18, L123) DPO invariance tests — IMPLEMENTED.** Two new tests: a fully-masked pair contributes
    a bitwise-zero gradient and sits at exactly `log 2` (S1-14 §6d), and a DPO step moves only the
    adapter — base tensors have no gradient accumulator in LoRA mode (`ll_debug_base_grad` errors),
    while A/B do move (§6c). "Base weights byte-identical" is already covered by the sibling
    reference-unchanged test (the adapter-off forward is byte-stable across a run).
11. **(#19, L127) grad-checkpointing core tests @slow — IMPLEMENTED (per-PR smoke added).** The full
    segment-length sweep stays nightly, but a new per-PR (non-`slow`) smoke runs ONE on/off comparison
    on the deep 12-layer fixture, asserting bitwise loss+weight equality AND `peak_on < peak_off`. The
    12 layers are load-bearing: they make the memory claim non-vacuous (a no-op passes equality but not
    the reduction). ~2.7 s per-PR; observed off 53 MiB, on 12 MiB.
12. **(#20, L129) chunked 2x@4096 not asserted — IMPLEMENTED (nightly).** A new `@slow` test runs the
    actual acceptance context (n_ctx 4096) and asserts `naive/both >= 2x`, so the 2.17x headline is
    reproduced by CI rather than a hand-measured table; the per-PR 1.5x@1024 smoke is retained.
    Measured on this box: naive ~904 MiB, ckpt+chunk ~414 MiB -> 2.18x (fits the 3.5 GB box, hence
    verifiable here; kept nightly because the naive arm alone is ~0.9 GiB).
13. **(#21, L131) chunked softcap/ALiBi no fixture — ACCEPTED DEVIATION.** Neither branch can be
    toggled on a `llama` fixture (`attn_soft_cap` is hardcoded in `gemma2.cpp`; the `llama` loader
    never reads `attention.max_alibi_bias` into `f_max_alibi_bias`), and the constituent ops are
    already grad-checked (`test_softcap`, `test_soft_max` `max_bias in {0,8}`, `test_flash_attn_ext`
    `softcap x max_bias`). Authoring a gemma2/ALiBi training fixture is disproportionate. Written up in
    `S1-24-chunked-attention-backward-fallback.md`.
14. **(#22, L141) MoE/SSM thread-determinism — IMPLEMENTED.** Copied `test_determinism.py`'s pattern:
    a MoE loss-curve determinism test (exercises `OUT_PROD_ID`/`OUT_PROD_ID_GRP` across 1/2/4 threads)
    and a Mamba-1 loss-curve determinism test (`SSM_CONV_BACK`/`SSM_SCAN_BACK`), both asserting the
    curve is bit-identical across thread counts. Complements B-10's existing single-step *gradient*
    determinism check on Mamba-2 `n_group>1` (not duplicated: different observable, different fixture).
    The SSM_SCAN_BACK checkpoint-K variant remains a vendor kernel feature carried elsewhere.
15. **(#23, L149/L184) `std::random_device` non-reproducible MODE_GRAD — IMPLEMENTED.** A
    `GGML_TEST_SEED` env var makes the three shared float-init primitives (`init_tensor_uniform`,
    `init_tensor_kq_mask`, `init_tensor_tril`) deterministic run to run; unset, behavior is
    byte-for-byte unchanged (`std::random_device`). Verified: two seeded `grad` runs of a project op
    are byte-identical output. The per-class discrete-label RNGs (integer labels/masks) are left
    random by default — they do not drive the tolerance-sensitive float magnitudes the finding named.

**Tally: implemented 13, accepted 2.**

## Verification

- Full suite green: 390 tests (384 non-`slow` + 6 `slow`; baseline 373 + 17 new), captured with the
  exit code checked. `test-backend-ops` rebuilt; CLAMP grad and the 22-test grad harness green.
- `pre-commit run --all-files` clean.
- Two commits: the fork (`ggml-opt.cpp/.h` norm capture, `test-backend-ops.cpp` `nvalid` guard +
  seed) on `ticket/S1-24-chunked-attention-backward-fallback`; the outer repo on
  `ticket/S1-50-minors-sweep`.

## PR notes

- Branch: `ticket/S1-50-minors-sweep` (outer); fork on `ticket/S1-24-chunked-attention-backward-fallback`.
- Upstreaming disposition: **fork-local** — the ggml-opt norm capture extends the fork's existing
  grad-clip feature; the `test-backend-ops` changes are test-harness hardening.
