"""The convergence gate, off the recorded config's happy path (S1-38).

Everything in ``test_convergence.py`` runs the recorded config, and the recorded config has three
coincidences that hide whole classes of bug:

* **``alpha == rank``**, so the effective LoRA scale is exactly 1.0. Chosen deliberately — it makes
  the PEFT comparison sidestep llama.cpp's ``alpha == 0`` trapdoor (``tests/convergence/README.md``)
  — but it means a stack that *dropped* the ``alpha/rank`` factor, applied it twice, or ignored the
  adapter's ``user_scale`` would sail through the entire gate: every power of 1.0 is 1.0. The
  finite-difference checks cannot catch it either; they prove the backward is consistent with the
  forward, not that either applies the scale the GGUF asked for.
* **``weight_decay == 0``**, so ggml's decoupled decay (``w *= 1 - lr*wd``) multiplies by exactly
  1 — and nothing else in the suite turns it on, so the term was never exercised at all. (ggml
  asserts ``wd <= 1.0`` — ``ggml-opt.cpp:984`` — so 1.0 is also the hardest it can be pushed.)
* **``rank == 4`` everywhere**, so a rank-dependent shape or indexing error has never seen a
  second data point.

Each variant moves ONE knob off the recorded value (``rank-8`` moves ``alpha`` with it, keeping the
effective scale at 1.0 so the *shape* change is the only thing under test) and asks two questions:

1. **One step, all 28 gradients, vs float64** (``harness.one_step_grad_errors``). The sharp
   instrument: a scale misapplied by even 0.1% moves every gradient by 0.1%, ~160x above the
   measured noise floor. Observed worst at alpha=10: 6.2e-06.
2. **The full curve, per step, within a measured band.** The band is variant-specific because
   conditioning is: at effective scale 2.5 the f32-vs-f64 rounding divergence compounds to 5.7e-04
   by step 35 *with exactly-correct gradients* (verified by check 1), so alpha's band is wider and
   the sharp claim rests on check 1, mirroring how the gate treats Q8_0.

There is no PEFT curve for these, on purpose: PEFT's job — catching a shared misunderstanding of
the *convention* — was done once, at scale 1.0, where the conventions provably coincide; the
float64 reference is the tighter oracle for whether the stack *implements* the convention.

And each variant proves it can fail: its reference curve must sit at least ``20x`` its band away
from the recorded config's reference curve, so a stack that silently ignored the knob (exactly the
bug class under test) is caught by the same assertion that checks the numerics. The separation is
asserted, not assumed.
"""

from __future__ import annotations

import dataclasses

import numpy as np
import pytest

from learning_llamas import create_zero_adapter

from .convergence import config, harness
from .test_convergence import GRAD_TOL

# (spec, curve band). Every band is a measurement with the observed number beside it.
VARIANTS = (
    # observed drift 5.7e-04 at step 35 — conditioning at scale 2.5, gradients exact (6.2e-06).
    pytest.param(config.RunSpec(alpha=10.0), 5e-3, id="alpha-2.5x-rank"),
    # observed drift 5.8e-07 — scale 0.5 contracts the trajectory, no amplification.
    pytest.param(config.RunSpec(user_scale=0.5), 1e-4, id="user-scale-half"),
    # observed drift 8.5e-07; dropping the decay entirely lands at 4.0e-02, 400x the band.
    pytest.param(config.RunSpec(weight_decay=1.0), 1e-4, id="weight-decay-at-ggml-max"),
    # observed drift 1.1e-06; same effective scale (8/8 == 1.0), different shapes: the rank axis
    # gets its second data point, and ignoring it lands 6.0e-01 away.
    pytest.param(config.RunSpec(rank=8, alpha=8.0), 1e-4, id="rank-8"),
    # observed drift 7.2e-07 against a reference that SUMS the window's gradients (ggml-opt's
    # opt_period semantics). A stack that averaged instead, or stepped every micro-batch anyway,
    # lands ≥ 3.1e-01 away. This is the mean-vs-sum normalization bug class, pinned.
    pytest.param(config.RunSpec(grad_accum=2), 1e-4, id="grad-accum-2"),
)

SEPARATION_MARGIN = 20


@pytest.fixture(scope="module")
def recorded_reference(tiny_f32, tmp_path_factory, conv_data) -> np.ndarray:
    """The float64 reference curve for the RECORDED spec — the separation baseline."""
    adapter = tmp_path_factory.mktemp("recorded") / "a.gguf"
    create_zero_adapter(
        tiny_f32, adapter, r=config.RANK, alpha=config.ALPHA, seed=config.ADAPTER_SEED
    )
    return np.array(harness.reference_curve(tiny_f32, adapter, conv_data))


@pytest.mark.parametrize(("spec", "band"), VARIANTS)
def test_one_step_gradients_are_exact_off_the_recorded_config(
    spec: config.RunSpec, band: float, tiny_f32, tmp_path, libs, conv_data
) -> None:
    """All 28 LoRA gradients at the variant spec. This is where a wrong scale actually fails."""
    loss_rel, by_tensor = harness.one_step_grad_errors(
        libs, tiny_f32, tmp_path / "a.gguf", conv_data, spec=spec
    )

    assert loss_rel < 1e-5, (
        f"the loss at {spec} disagrees with the float64 reference by {loss_rel:.2e} relative — "
        f"the forward is already applying this spec differently than llama-adapter.h says it does"
    )

    worst = max(by_tensor, key=by_tensor.__getitem__)
    assert by_tensor[worst] < GRAD_TOL, (
        f"{worst}: ggml's gradient at {spec} disagrees with the float64 reference by "
        f"{by_tensor[worst]:.2e} (relative to the tensor's largest element). The recorded config "
        f"cannot see this class of bug; this variant exists precisely for it."
    )


@pytest.mark.parametrize(("spec", "band"), VARIANTS)
def test_the_curve_stays_in_its_band_off_the_recorded_config(
    spec: config.RunSpec,
    band: float,
    tiny_f32,
    tmp_path,
    libs,
    conv_data,
    recorded_reference,
) -> None:
    """Forty steps of ``train_sft`` at the variant spec, per step, within the measured band."""
    adapter = tmp_path / "a.gguf"
    got = np.array(harness.train(libs, tiny_f32, adapter, conv_data, spec=spec))
    want = np.array(harness.reference_curve(tiny_f32, adapter, conv_data, spec=spec))

    # The knob must MATTER: if this variant's reference curve is indistinguishable from the
    # recorded config's, the band below could not catch a stack that ignored the knob, and the
    # variant is decorative.
    separation = np.abs(want - recorded_reference).max()
    assert separation > SEPARATION_MARGIN * band, (
        f"this variant's reference curve sits only {separation:.2e} from the recorded config's — "
        f"less than {SEPARATION_MARGIN}x its band ({band:.0e}). Size the knob so ignoring it "
        f"is detectable."
    )

    diff = np.abs(got - want)
    worst = int(diff.argmax())
    moved = ", ".join(
        f"{f.name}={getattr(spec, f.name)!r}"
        for f in dataclasses.fields(spec)
        if getattr(spec, f.name) != getattr(config.RECORDED, f.name)
    )
    assert diff.max() < band, (
        f"the loss curve left the float64 reference at step {worst}: "
        f"{got[worst]:.8f} vs {want[worst]:.8f} (worst |diff| {diff.max():.2e} > {band:.0e}). "
        f"The knob(s) under test: {moved}."
    )
