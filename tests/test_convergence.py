"""The convergence gate (S1-12): does the whole training stack compute the *right* thing?

Every other test in this suite checks a piece. `test-backend-ops grad` checks one kernel against a
finite difference. `test_p0_gradient` checks the composition against a finite difference of the
whole graph. Neither can say that *training* — graph build, adapter wiring, the masked CE, the
optimizer, forty steps of it — arrives where it should.

Two oracles, and they answer different questions.

**The float64 reference** (``tests/reference_llama.py``) is the tight one: the same architecture and
the same weights, in float64, with a backward derived from the maths rather than transcribed from
the graph. It is compared per step, and it can name the tensor that disagrees. It catches a
gradient that is wrong by a *fraction of a percent* — the class of bug that still makes a loss curve
fall, converges somewhere slightly else, and is invisible to any band.

**The recorded PEFT curve** (``tests/convergence/``) is the independent one: transformers + peft,
written by other people, on a checkpoint that is provably the same model. It cannot be as tight,
and it does not need to be — it is there to catch a *shared misunderstanding*, the case where the
graph and the float64 reference agree with each other and both disagree with what LoRA SFT is.

The ticket expected these to be loose ("per-step exact match is impossible") and prescribed windowed
tolerance bands. Measured, all three implementations agree to **~1e-6 over 40 optimizer steps**, so
the gate is per-step at 1e-4 instead. A band wide enough to absorb kernel-numerics drift would have
been wide enough to hide a real gradient bug; it turns out no such band is needed.
"""

from __future__ import annotations

import json
import pathlib

import numpy as np

from learning_llamas import Model, create_zero_adapter

from . import reference_llama as ref
from .convergence import config, harness
from .fixtures import gen_tiny_llama

# ---------------------------------------------------------------------------
# Tolerances. Every one of these is a measurement, not a guess -- the number that was observed is
# quoted beside it, and the margin is the difference.
# ---------------------------------------------------------------------------

# llama.cpp (F32) vs the float64 reference, per step, over the whole 40-step run.
#
#   observed: 5.3e-07 absolute (1.1e-07 relative), and BIT-IDENTICAL across 1, 2 and 4 threads.
#
# The margin is for a different host's SIMD width changing the reduction order (ADR-0002), which
# cannot be measured from here. 1e-4 is ~190x the observed gap and still catches a gradient that is
# wrong in its fourth significant figure.
CURVE_TOL = 1e-4

# Same, against PEFT. Slightly looser because torch's kernels are a third numerics.
#   observed: 9.5e-07 absolute.
PEFT_TOL = 2e-4

# A Q8_0 base is a *different model* -- quantizing the weights perturbs the logits, and PEFT cannot
# run the GGUF at all -- so this compares against the same F32 reference with a band that reflects
# the quantization, not the gradient.
#   observed: 4.1e-02 worst per-step, 3.5e-03 worst windowed-mean.
QUANT_STEP_TOL = 0.15
QUANT_WINDOW_TOL = 0.03

# The gradient of one LoRA tensor, ggml vs the float64 reference.
#   observed: 2.0e-06 worst, over all 28 tensors.
GRAD_TOL = 1e-3

WINDOWS = ((0, 10), (10, 25), (25, 40))

# The run/reference helpers live in tests/convergence/harness.py, shared with the off-unit
# variants (test_convergence_variants.py) and the determinism check (test_determinism.py). The
# `conv_data` fixture is in conftest.py for the same reason. Everything here runs the RECORDED
# spec — the one reference_curve.json holds.


# ---------------------------------------------------------------------------
# The oracle audits itself first.
# ---------------------------------------------------------------------------


def test_the_reference_backward_is_itself_correct(tiny_f32, tmp_path, conv_data) -> None:
    """The float64 backward, against a central finite difference of the float64 *forward*.

    An oracle nobody audited is not an oracle. This VJP was derived by hand, which makes it exactly
    as likely to be wrong as the graph it is auditing — and a wrong reference does not fail loudly,
    it silently redefines "correct" and then agrees with whatever it is compared to.

    ``h = 1e-4``, deliberately. The FD's roundoff floor is ``ulp(L) / 2h``; at ``h = 1e-6`` and a
    loss of ~6 that is 4.4e-10, which is 6e-5 *relative* to the smaller gradients here — enough to
    look like a failure when nothing is wrong. The same trap, one order of magnitude down, is what
    makes ``test-backend-ops grad -o SILU`` report FAIL on an exact kernel.
    """
    tokens, targets, weights = conv_data
    adapter = tmp_path / "a.gguf"
    create_zero_adapter(
        tiny_f32, adapter, r=config.RANK, alpha=config.ALPHA, seed=config.ADAPTER_SEED
    )

    base, hp = ref.load_model(tiny_f32)
    loras = harness.load_loras(tiny_f32, adapter)
    scale = ref.lora_scale(config.ALPHA, config.RANK, 1.0)

    # B is zero at init, and dA is proportional to B — so with a fresh adapter every dA is exactly
    # zero and half the backward would be audited against nothing at all. Wake it up.
    rng = np.random.default_rng(11)
    for lora in loras.values():
        lora.b += rng.normal(0.0, 0.02, size=lora.b.shape)

    logits, cache = ref.forward(base, hp, loras, scale, tokens[0])
    _, dlogits = ref.loss_from_logits(logits, targets[0], weights[0])
    grads = ref.backward(base, hp, loras, scale, cache, dlogits)

    def loss_at(name: str, is_b: bool, idx: tuple, delta: float) -> float:
        lora = loras[name]
        arr = lora.b if is_b else lora.a
        original = arr[idx]
        arr[idx] = original + delta
        lg, _ = ref.forward(base, hp, loras, scale, tokens[0])
        value, _ = ref.loss_from_logits(lg, targets[0], weights[0])
        arr[idx] = original
        return value

    h = 1e-4
    worst = 0.0
    for name in loras:
        for is_b in (False, True):
            arr = grads[f"{name}.lora_{'b' if is_b else 'a'}"]
            idx = tuple(int(rng.integers(0, s)) for s in arr.shape)
            numeric = (loss_at(name, is_b, idx, h) - loss_at(name, is_b, idx, -h)) / (2 * h)
            analytic = float(arr[idx])
            worst = max(worst, abs(analytic - numeric) / max(abs(analytic), abs(numeric), 1e-12))

    assert worst < 1e-5, (
        f"the float64 reference's own backward disagrees with its own forward: {worst:.2e}"
    )

    # S1-48: the SAME audit for the base-weight gradients the full-finetune backward now emits.
    # An oracle nobody audited is not an oracle, and the base-weight VJPs (the RMSNorm weight sums,
    # the dy.T @ x at each projection, the output head, and the embedding scatter) were derived by
    # hand exactly like the LoRA ones. A sample of each *kind* is finite-differenced against this
    # file's own forward, at the same h and the same tolerance. The LoRA assertion above stands
    # untouched.
    base_grads = ref.backward(base, hp, loras, scale, cache, dlogits, base_grads=True)

    def base_loss_at(name: str, idx: tuple, delta: float) -> float:
        original = base[name][idx]
        base[name][idx] = original + delta
        lg, _ = ref.forward(base, hp, loras, scale, tokens[0])
        value, _ = ref.loss_from_logits(lg, targets[0], weights[0])
        base[name][idx] = original
        return value

    # One of every kind: the embedding scatter, the output head, a trainable norm, an attention
    # projection, and an FFN projection.
    sampled = [
        "token_embd.weight",
        "output.weight",
        "output_norm.weight",
        "blk.0.attn_norm.weight",
        "blk.0.attn_q.weight",
        "blk.1.attn_v.weight",
        "blk.0.ffn_gate.weight",
        "blk.1.ffn_down.weight",
    ]
    base_worst = 0.0
    for name in sampled:
        arr = base_grads[name]
        if name == "token_embd.weight":
            # A random vocab row is almost never one this 32-token sample looked up, so its
            # gradient — and the FD — would both be a vacuous zero. Pick a row a token selected.
            idx = (int(tokens[0][3]), int(rng.integers(0, arr.shape[1])))
        else:
            idx = tuple(int(rng.integers(0, s)) for s in arr.shape)
        numeric = (base_loss_at(name, idx, h) - base_loss_at(name, idx, -h)) / (2 * h)
        analytic = float(arr[idx])
        rel = abs(analytic - numeric) / max(abs(analytic), abs(numeric), 1e-12)
        base_worst = max(base_worst, rel)

    assert base_worst < 1e-5, (
        f"the reference's base-weight backward disagrees with its own forward: {base_worst:.2e}"
    )


# ---------------------------------------------------------------------------
# One step: the loss, and every gradient. This is the strongest check in the suite.
# ---------------------------------------------------------------------------


def test_one_step_matches_the_reference_exactly(tiny_f32, tmp_path, libs, conv_data) -> None:
    """The loss and **every one of the 28 LoRA gradients**, against float64.

    A loss that matches proves the forward. It says nothing about the backward: the S1-03 bug (a
    weighted one-hot CE whose kernel hardcodes ``sum(labels) == 1``) had a perfectly correct
    forward and a wrong gradient on every masked row. So the gradients are compared tensor by
    tensor, and a failure names the tensor.

    ``lr`` is 1e-30 so the optimizer cannot move the weights before the gradient is read back.
    (The mechanics live in ``harness.one_step_grad_errors``, shared with the off-unit variants.)
    """
    loss_rel, by_tensor = harness.one_step_grad_errors(
        libs, tiny_f32, tmp_path / "a.gguf", conv_data
    )

    assert loss_rel < 1e-5, (
        f"the training loss disagrees with the float64 reference by {loss_rel:.2e} relative"
    )

    assert len(by_tensor) == 28, f"expected 28 LoRA tensors, compared {len(by_tensor)}"
    worst = max(by_tensor, key=by_tensor.__getitem__)
    assert by_tensor[worst] < GRAD_TOL, (
        f"{worst}: ggml's gradient disagrees with the float64 reference by "
        f"{by_tensor[worst]:.2e} (relative to the tensor's largest element)"
    )


# ---------------------------------------------------------------------------
# Forty steps.
# ---------------------------------------------------------------------------


# The 40-step checks are NOT marked `slow`, and that is a deliberate reversal of the ticket.
#
# S1-12 assumed the gate would be expensive and put it in the nightly tier. The whole file runs in
# ~7 seconds. Leaving it nightly-only would mean per-PR CI never exercises ggml's AdamW numerics at
# all -- a single step at lr=1e-30 does not move the optimizer -- and never runs the independent
# (PEFT) oracle. Both AdamW mutations in the mutation table are caught ONLY by these three tests.
# A gate that runs once a night is a gate that tells you which of the day's twenty PRs broke it.
def test_the_loss_curve_matches_the_float64_reference(tiny_f32, tmp_path, libs, conv_data) -> None:
    """Forty optimizer steps of ``train_sft``, per step, against float64.

    This is the whole stack: collation, the mask shift, the graph, the backward, AdamW's bias
    correction, forty times over. Anything that is wrong by a fraction of a percent compounds here
    and nowhere else.
    """
    adapter = tmp_path / "a.gguf"
    got = harness.train(libs, tiny_f32, adapter, conv_data)
    want = harness.reference_curve(tiny_f32, adapter, conv_data)

    assert len(got) == len(want) == config.EPOCHS * config.N_SAMPLES

    diff = np.abs(np.array(got) - np.array(want))
    worst = int(diff.argmax())
    assert diff.max() < CURVE_TOL, (
        f"the loss curve left the float64 reference at step {worst}: "
        f"{got[worst]:.8f} vs {want[worst]:.8f} (worst |diff| {diff.max():.2e} > {CURVE_TOL:.0e})"
    )


def test_the_loss_curve_matches_the_recorded_peft_reference(
    tiny_f32, tmp_path, libs, conv_data
) -> None:
    """The same forty steps against transformers + peft, recorded once and committed.

    An implementation nobody here wrote, on a checkpoint proven to be the same model (the recorder
    refuses to record unless the HF twin's logits match the float64 reference — see
    ``tests/convergence/hf_twin.py``, where the RoPE permutation lives).

    This is the only check in the project that can catch the graph and the float64 reference being
    wrong *together*.
    """
    recorded = json.loads(config.CURVE_PATH.read_text())

    assert recorded["identity"] == config.identity(gen_tiny_llama.HPARAMS.n_vocab), (
        "reference_curve.json was recorded against a different fixture, dataset or config. "
        "It describes a different experiment and cannot be compared against — regenerate it "
        "(see tests/convergence/README.md)."
    )

    adapter = tmp_path / "a.gguf"
    got = np.array(harness.train(libs, tiny_f32, adapter, conv_data))
    want = np.array(recorded["curve"])

    diff = np.abs(got - want)
    worst = int(diff.argmax())
    assert diff.max() < PEFT_TOL, (
        f"the loss curve left the PEFT reference at step {worst}: "
        f"{got[worst]:.8f} vs {want[worst]:.8f} (worst |diff| {diff.max():.2e} > {PEFT_TOL:.0e})"
    )


def test_a_quantized_base_trains_within_its_band(tiny_q8_0, tmp_path, libs, conv_data) -> None:
    """A Q8_0 base, against the same F32 reference — with a band that reflects the quantization.

    Quantizing the base perturbs the logits, so the trajectory genuinely diverges *in the small*
    and no tight per-step bound is meaningful. PEFT cannot run the GGUF at all, so there is no
    quantized reference to record; the F32 one is what there is.

    The band is therefore documented rather than derived: worst per-step deviation measured at
    4.1e-2, worst windowed-mean at 3.5e-3. What is actually being asserted is that a quantized
    base trains along the *same trajectory*, not that it lands on the same number.
    """
    adapter = tmp_path / "a.gguf"
    got = np.array(harness.train(libs, tiny_q8_0, adapter, conv_data))

    f32_path, _ = gen_tiny_llama.build("f32", tiny_q8_0.parent)
    create_zero_adapter(
        f32_path, adapter, r=config.RANK, alpha=config.ALPHA, seed=config.ADAPTER_SEED
    )
    want = np.array(harness.reference_curve(f32_path, adapter, conv_data))

    assert np.abs(got - want).max() < QUANT_STEP_TOL, (
        f"the Q8_0 curve left its per-step band: worst |diff| {np.abs(got - want).max():.3e}"
    )

    for lo, hi in WINDOWS:
        drift = abs(got[lo:hi].mean() - want[lo:hi].mean())
        assert drift < QUANT_WINDOW_TOL, (
            f"the Q8_0 windowed mean over steps {lo + 1}-{hi} drifted by {drift:.3e}: "
            f"{got[lo:hi].mean():.6f} vs {want[lo:hi].mean():.6f}"
        )

    assert got[-1] < got[0], "the quantized model did not learn at all"


# ---------------------------------------------------------------------------
# What the gate ran on, and one finding it pins.
# ---------------------------------------------------------------------------


def test_the_report_records_the_device_it_actually_ran_on(
    tiny_f32, tmp_path, libs, device, conv_data
) -> None:
    """Write the JSON report artifact ROADMAP §11 wants: what was asked for, what was available.

    "Acceptable but reported, never hidden": a backend stage whose kernels are incomplete runs the
    gaps on the CPU via ``ggml_backend_sched`` and still passes, which is fine — but a green tick
    must never be mistaken for "this ran GPU-resident".

    There is no sched-split count here yet, and emitting a zero would be worse than omitting it: on
    a CPU-only build every op runs on the CPU *by definition*, so a fallback count is identically
    zero and reports nothing. S2-01 adds the accessor when there is a second backend for it to be
    about.
    """
    from .conftest import available_devices

    report_dir = pathlib.Path(__file__).parent / ".report"
    report_dir.mkdir(exist_ok=True)

    adapter = tmp_path / "a.gguf"
    curve = harness.train(libs, tiny_f32, adapter, conv_data)

    report = {
        "device_requested": device,
        "devices_available": sorted(available_devices(libs)),
        "identity": config.identity(gen_tiny_llama.HPARAMS.n_vocab),
        "final_loss": curve[-1],
        "curve": curve,
    }
    (report_dir / f"convergence-{device}.json").write_text(json.dumps(report, indent=2) + "\n")

    assert device in report["devices_available"]


def test_inference_and_training_differ_only_by_the_f16_kv_cache(tiny_f32, libs, conv_data) -> None:
    """``model.logits()`` and the training forward are **not** the same function, and here is why.

    llama.cpp's KV cache is F16 by default (``llama_context_default_params``: ``type_k`` /
    ``type_v`` = ``GGML_TYPE_F16``). The **training** path bypasses the cache entirely (S1-00) and
    keeps K/V in F32. So the inference path quantizes K and V to ~5e-4 relative before attention
    ever runs, and the two paths land measurably apart:

        inference vs the float64 reference:  ~8e-4   (F16's epsilon is 2^-11 = 4.9e-4)
        training  vs the float64 reference:  ~5e-7   (float32 rounding)

    Three orders of magnitude, from one default. This cost real time during S1-12 — the reference
    looked wrong until the gap was attributed — so it is pinned here rather than rediscovered. It
    is also the honest answer to "why do my training logits differ from my inference logits".
    """
    tokens, _, _ = conv_data
    base, hp = ref.load_model(tiny_f32)

    model = Model(tiny_f32, libs=libs, n_ctx=harness.N_CTX, n_ubatch=config.SEQ_LEN, n_threads=2)
    try:
        inference = np.asarray(model.logits([int(t) for t in tokens[0]]), dtype=np.float64)
    finally:
        model.close()

    exact, _ = ref.forward(base, hp, {}, 1.0, tokens[0])
    exact_last = exact[-1]
    spread = float(exact_last.max() - exact_last.min())

    relative = float(np.abs(inference - exact_last).max()) / spread

    # It is the F16 KV cache and nothing worse: comfortably above float32 noise (5e-7) and
    # comfortably below anything a structural bug would produce (O(1)).
    assert 1e-5 < relative < 5e-3, (
        f"the inference path deviates from the float64 reference by {relative:.2e} relative. "
        "~5e-4 is the F16 KV cache. Far more means something is actually broken; far less would "
        "mean the KV cache stopped being F16, and this test's premise needs rechecking."
    )
