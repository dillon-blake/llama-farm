"""The Mamba composition's gradients, against float64 (S1-47).

The SSM backward kernels are checked op-by-op (``SSM_CONV_BACK``/``SSM_SCAN_BACK`` MODE_GRAD), and
the e2e test here proves a Mamba model trains -- the loss falls, the SSM adapters move. The
2026-07-15 audit's ``moe-ssm`` major: nothing proved the **composed** gradient (ssm_in -> causal
conv -> selective scan -> D skip -> SiLU gate -> ssm_out, with LoRA on all four projections) is
*right*. Wrong by a constant, missing the ``dA`` path of ``ddt``, or a conv window shifted by one,
it still trains -- somewhere slightly and silently else. The MoE side got exactly this gate in
S1-41; this is its Mamba twin.

Four layers, the same shape as the MoE gate:

1. the oracle audits itself (central finite difference of its own forward, every LoRA tensor);
2. one step, **every LoRA gradient tensor** -- all four Mamba projections, both layers -- vs
   ``ll_debug_grad``, at effective LoRA scale 2.0 (``alpha != rank``, the S1-38 lesson);
3. a 24-step ``train_sft`` loss trajectory on the fixture, per step, against the reference;
4. the e2e gate: loss falls on the real stack, and the S1-11 preflight calls the arch trainable.

The support boundary this exercises is Mamba-1: per-state ``A`` (``A->ne[0] == d_state``),
``n_group == 1`` (llama.cpp's Mamba-1 always builds it so). ``n_group > 1`` (real Mamba-2 /
Falcon-H1) has no gradient oracle and the fork's backward now refuses it loudly; see the S1-47
ticket and tickets/backlog.
"""

from __future__ import annotations

import ctypes
import pathlib

import numpy as np
import pytest

from learning_llamas import Model, _ffi, create_zero_adapter, enumerate_targets
from learning_llamas.adapter import _index_by_name
from learning_llamas.data import MaskedSample
from learning_llamas.preflight import preflight_adapter
from learning_llamas.train import SFTConfig, TrainConfig, Trainer, train_sft
from learning_llamas.train.loop import Batch

from . import reference_llama as ref_llama
from . import reference_mamba as ref_mamba
from .fixtures import gen_tiny_mamba

RANK = 4
ALPHA = 8.0  # effective scale 2.0: the SSM path gets off-unit coverage too (S1-38)
ADAPTER_SEED = 3
SEQ_LEN = 16
N_CTX = 64
LR = 1e-3
EPOCHS = 4
N_SAMPLES = 6

# A fresh adapter has B == 0, so the LoRA delta is exactly zero and the SSM sits at the base
# fixture's near-silent operating point (0.02-scale weights): the dt pathway's gradient there is
# ~1e-11, below any finite difference's resolution, and cross-checking it would be checking noise.
# So the gradient oracles wake BOTH A and B to a non-trivial trained-ish state -- which is also the
# operating point that actually matters. The trajectory test deliberately does NOT wake up: it
# trains from the no-op init, exactly as a real run does, and compares losses (not gradients).
WAKE_SIGMA = 0.3

# One step, per LoRA tensor, relative to the tensor's largest element.
#   observed: 3.7e-06 worst over all 16 tensors (loss agrees at 3.9e-08). The Mamba forward is a
#   length-SEQ_LEN recurrence per layer, deeper in f32 reductions than the dense gate, and scale 2.0
#   doubles the deltas. The band is generous over the observation on purpose.
GRAD_TOL = 1e-3

# The 24-step curve. observed: 4.5e-07 worst per step.
CURVE_TOL = 1e-4


@pytest.fixture(scope="session")
def tiny_mamba_f32(libs: _ffi.Libraries, fixture_cache_dir: pathlib.Path) -> pathlib.Path:
    path, _ = gen_tiny_mamba.build("f32", fixture_cache_dir)
    return path


def _dataset() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """A fixed dataset in the convergence-config shape, sized for the Mamba vocab."""
    rng = np.random.default_rng(20260715)
    raw = rng.integers(3, gen_tiny_mamba.HPARAMS.n_vocab, size=(N_SAMPLES, SEQ_LEN + 1))
    tokens, targets = raw[:, :SEQ_LEN], raw[:, 1 : SEQ_LEN + 1]
    weights = np.ones((N_SAMPLES, SEQ_LEN), dtype=np.float64)
    weights[:, : SEQ_LEN // 2] = 0.0
    return tokens, targets, weights


def _load_loras(base_path: pathlib.Path, adapter_path: pathlib.Path) -> dict[str, ref_llama.Lora]:
    """The adapter's A/B in float64 for every enumerated target (the four Mamba projections)."""
    import gguf

    reader = gguf.GGUFReader(str(adapter_path), "r")
    data = {t.name: np.array(t.data, dtype=np.float64) for t in reader.tensors}
    return {
        t.name: ref_llama.Lora(a=data[f"{t.name}.lora_a"].copy(), b=data[f"{t.name}.lora_b"].copy())
        for t in enumerate_targets(base_path)
    }


def _wake_up_reference(loras: dict[str, ref_llama.Lora]) -> None:
    """Perturb A and B off init to a non-trivial state, f32-rounded (see :data:`WAKE_SIGMA`)."""
    rng = np.random.default_rng(11)

    def wake(arr: np.ndarray) -> np.ndarray:
        return (
            (arr + rng.normal(0.0, WAKE_SIGMA, size=arr.shape))
            .astype(np.float32)
            .astype(np.float64)
        )

    for lora in loras.values():
        lora.a = wake(lora.a)
        lora.b = wake(lora.b)


def _wake_up(libs: _ffi.Libraries, model: Model, loras: dict[str, ref_llama.Lora]) -> None:
    """Wake A and B on BOTH sides to the *same* f32 bits, so ggml and the reference agree."""
    _wake_up_reference(loras)  # same seed/order -> identical draws
    index = _index_by_name(libs, model.adapter)
    for name, lora in loras.items():
        for is_b, arr in ((False, lora.a), (True, lora.b)):
            flat = np.ascontiguousarray(arr.astype(np.float32).reshape(-1))
            n = libs.farm.ll_adapter_set(
                model.adapter,
                index[name],
                is_b,
                flat.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
                flat.size,
            )
            assert n == flat.size


def test_the_mamba_reference_backward_is_itself_correct(tiny_mamba_f32, tmp_path) -> None:
    """The float64 Mamba backward vs a central finite difference of its own forward.

    Samples every LoRA tensor. The FD reruns the whole forward -- conv, scan, gate and all -- so a
    backward that got the conv window direction wrong, or dropped the scan's dA path of ddt,
    disagrees here before ggml ever enters the picture.
    """
    tokens, targets, weights = _dataset()
    adapter = tmp_path / "a.gguf"
    create_zero_adapter(tiny_mamba_f32, adapter, r=RANK, alpha=ALPHA, seed=ADAPTER_SEED)

    base, hp = ref_mamba.load_model(tiny_mamba_f32)
    loras = _load_loras(tiny_mamba_f32, adapter)
    scale = ref_llama.lora_scale(ALPHA, RANK, 1.0)

    _wake_up_reference(loras)
    rng = np.random.default_rng(101)

    logits, cache = ref_mamba.forward(base, hp, loras, scale, tokens[0])
    _, dlogits = ref_llama.loss_from_logits(logits, targets[0], weights[0])
    grads = ref_mamba.backward(base, hp, loras, scale, cache, dlogits)

    def loss_at(name: str, is_b: bool, idx: tuple, delta: float) -> float:
        arr = loras[name].b if is_b else loras[name].a
        original = arr[idx]
        arr[idx] = original + delta
        lg, _ = ref_mamba.forward(base, hp, loras, scale, tokens[0])
        value, _ = ref_llama.loss_from_logits(lg, targets[0], weights[0])
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
        f"the float64 Mamba reference's own backward disagrees with its own forward: {worst:.2e}"
    )


def test_one_step_matches_the_mamba_reference(tiny_mamba_f32, tmp_path, libs) -> None:
    """The loss and every LoRA gradient -- the four Mamba projections -- against float64."""
    tokens, targets, weights = _dataset()
    adapter = tmp_path / "a.gguf"
    create_zero_adapter(tiny_mamba_f32, adapter, r=RANK, alpha=ALPHA, seed=ADAPTER_SEED)

    base, hp = ref_mamba.load_model(tiny_mamba_f32)
    loras = _load_loras(tiny_mamba_f32, adapter)
    scale = ref_llama.lora_scale(ALPHA, RANK, 1.0)

    model = Model(
        tiny_mamba_f32, libs=libs, n_ctx=N_CTX, n_ubatch=SEQ_LEN, training=True, n_threads=2
    )
    try:
        model.attach_adapter(adapter, scale=1.0)
        _wake_up(libs, model, loras)

        with Trainer(libs, model, TrainConfig(lr=1e-30)) as trainer:
            metrics = trainer.step(
                Batch(
                    tokens=[int(t) for t in tokens[0]],
                    targets=[int(t) for t in targets[0]],
                    weights=[float(w) for w in weights[0]],
                )
            )

            logits, cache = ref_mamba.forward(base, hp, loras, scale, tokens[0])
            loss, dlogits = ref_llama.loss_from_logits(logits, targets[0], weights[0])
            grads = ref_mamba.backward(base, hp, loras, scale, cache, dlogits)

            assert abs(metrics.loss - loss) / abs(loss) < 1e-5, (
                f"the Mamba training loss disagrees with float64: {metrics.loss:.8f} vs {loss:.8f}"
            )

            n_compared = 0
            for name in loras:
                for is_b in (False, True):
                    expected = grads[f"{name}.lora_{'b' if is_b else 'a'}"]
                    buf = (ctypes.c_float * expected.size)()
                    got = libs.farm.ll_debug_grad(
                        model.ctx, name.encode(), is_b, buf, expected.size
                    )
                    assert got == expected.size, f"ll_debug_grad({name}) returned {got}"

                    actual = np.frombuffer(buf, dtype=np.float32, count=got).reshape(expected.shape)
                    rel = np.abs(actual.astype(np.float64) - expected).max() / max(
                        np.abs(expected).max(), 1e-30
                    )
                    assert rel < GRAD_TOL, (
                        f"{name}.lora_{'b' if is_b else 'a'}: ggml's Mamba gradient disagrees with "
                        f"float64 by {rel:.2e} -- suspect the scan reverse-recurrence (ddt's dA "
                        f"path) or the conv window direction first"
                    )
                    n_compared += 1
            # 8 LoRA tensors on the 2-layer Mamba fixture (per layer: ssm_in/ssm_x/ssm_dt/ssm_out),
            # each compared for lora_a and lora_b -> 16 comparisons.
            assert n_compared == 16, (
                f"expected 16 LoRA comparisons on the Mamba fixture, {n_compared}"
            )
    finally:
        model.close()


def test_the_mamba_loss_curve_matches_the_reference(tiny_mamba_f32, tmp_path, libs) -> None:
    """24 steps of the real ``train_sft`` on the Mamba fixture, per step, against float64."""
    tokens, targets, weights = _dataset()
    adapter = tmp_path / "a.gguf"

    create_zero_adapter(tiny_mamba_f32, adapter, r=RANK, alpha=ALPHA, seed=ADAPTER_SEED)
    model = Model(
        tiny_mamba_f32, libs=libs, n_ctx=N_CTX, n_ubatch=SEQ_LEN, training=True, n_threads=2
    )
    try:
        model.attach_adapter(adapter, scale=1.0)
        samples = [
            MaskedSample(
                tokens=[int(t) for t in tokens[i]] + [int(targets[i][-1])],
                weights=[0.0] + [float(w) for w in weights[i]],
            )
            for i in range(N_SAMPLES)
        ]
        result = train_sft(
            libs,
            model,
            samples,
            SFTConfig(lr=LR, seq_len=SEQ_LEN, epochs=EPOCHS, shuffle=False, schedule="constant"),
        )
        got = np.array([s.loss for s in result.steps])
    finally:
        model.close()

    base, hp = ref_mamba.load_model(tiny_mamba_f32)
    loras = _load_loras(tiny_mamba_f32, adapter)
    scale = ref_llama.lora_scale(ALPHA, RANK, 1.0)

    params: dict[str, np.ndarray] = {}
    for name, lora in loras.items():
        params[f"{name}.lora_a"] = lora.a
        params[f"{name}.lora_b"] = lora.b
    opt = ref_llama.AdamW(lr=LR)

    want = []
    for _epoch in range(EPOCHS):
        for i in range(N_SAMPLES):
            logits, cache = ref_mamba.forward(base, hp, loras, scale, tokens[i])
            loss, dlogits = ref_llama.loss_from_logits(logits, targets[i], weights[i])
            grads = ref_mamba.backward(base, hp, loras, scale, cache, dlogits)
            opt.step(params, grads)
            want.append(loss)
    want = np.array(want)

    assert len(got) == len(want) == EPOCHS * N_SAMPLES
    diff = np.abs(got - want)
    worst = int(diff.argmax())
    assert diff.max() < CURVE_TOL, (
        f"the Mamba loss curve left the float64 reference at step {worst}: "
        f"{got[worst]:.8f} vs {want[worst]:.8f} (worst |diff| {diff.max():.2e})"
    )


def test_a_mamba_model_trains_on_the_real_stack(tiny_mamba_f32, tmp_path, libs) -> None:
    """The e2e gate: preflight calls the arch trainable, and a short SFT loop's loss falls.

    This is the check S1-31 promised and never shipped: a tiny-Mamba CPU SFT loss falls, and the
    S1-11 preflight reports the mamba-family arch trainable (the graph walk flips automatically once
    the SSM backward cases exist -- which they now do).
    """
    tokens, targets, weights = _dataset()
    adapter = tmp_path / "a.gguf"
    create_zero_adapter(tiny_mamba_f32, adapter, r=RANK, alpha=ALPHA, seed=ADAPTER_SEED)

    model = Model(
        tiny_mamba_f32, libs=libs, n_ctx=N_CTX, n_ubatch=SEQ_LEN, training=True, n_threads=2
    )
    try:
        model.attach_adapter(adapter, scale=1.0)

        report = preflight_adapter(libs, model.ctx, model.model, [model.adapter])
        assert report.trainable, report.summary()

        samples = [
            MaskedSample(
                tokens=[int(t) for t in tokens[i]] + [int(targets[i][-1])],
                weights=[0.0] + [float(w) for w in weights[i]],
            )
            for i in range(N_SAMPLES)
        ]
        result = train_sft(
            libs,
            model,
            samples,
            SFTConfig(lr=1e-2, seq_len=SEQ_LEN, epochs=EPOCHS, shuffle=False, schedule="constant"),
        )
        losses = [s.loss for s in result.steps]
    finally:
        model.close()

    assert losses[-1] < losses[0], (
        f"Mamba SFT loss did not fall: {losses[0]:.5f} -> {losses[-1]:.5f}"
    )
