"""The MoE composition's gradients, against float64 (S1-41).

``test_moe.py`` proves a MoE model trains — the loss falls, the expert adapters move. The audit's
confirmed-major point: nothing proved the **composed** gradient (router softmax → top-k weights →
renormalization → expert gather → 3D LoRA) is *right*. Wrong by a constant factor, scattered to
the wrong expert slice, or missing the router-coupling term, it still trains — somewhere slightly
else, silently.

Three layers, same shape as the dense gate:

1. the oracle audits itself (finite difference of its own forward — including through the router,
   whose top-k makes this the one place an FD *can* legitimately blip: a perturbation that flips a
   selection lands in the ones, not the millionths, and would be caught);
2. one step, **every one of the 28 LoRA gradient comparisons** — 14 targets (per layer:
   attn_q/k/v/output plus the three ``*_exps`` stacks), each checked for its A and its B — so 2D
   attention pairs and 3D expert stacks alike, at effective LoRA scale 2.0, which gets the MoE path
   the scale coverage the dense gate only got in S1-38;
3. a 24-step loss trajectory of the real ``train_sft`` on the MoE fixture, per step.
"""

from __future__ import annotations

import ctypes
import pathlib

import numpy as np
import pytest

from learning_llamas import Model, _ffi, create_zero_adapter, enumerate_targets
from learning_llamas.adapter import _index_by_name
from learning_llamas.data import MaskedSample
from learning_llamas.train import SFTConfig, TrainConfig, Trainer, train_sft
from learning_llamas.train.loop import Batch

from . import reference_llama as ref_llama
from . import reference_moe as ref_moe
from .fixtures import gen_tiny_moe

RANK = 4
ALPHA = 8.0  # effective scale 2.0: the MoE path gets the off-unit coverage of S1-38 too
ADAPTER_SEED = 3
SEQ_LEN = 24
N_CTX = 64
LR = 1e-3
EPOCHS = 4
N_SAMPLES = 6

# One step, per LoRA tensor, relative to the tensor's largest element.
#   observed: 1.5e-05 worst over all 28 comparisons (the dense gate observes 2.0e-06 over its own
#   28; the MoE forward is deeper in reductions, and scale 2.0 doubles the deltas).
GRAD_TOL = 1e-3

# The 24-step curve.  observed: 2.7e-06 worst per step.
CURVE_TOL = 1e-4


@pytest.fixture(scope="session")
def tiny_moe_f32(libs: _ffi.Libraries, fixture_cache_dir: pathlib.Path) -> pathlib.Path:
    path, _ = gen_tiny_moe.build("f32", fixture_cache_dir)
    return path


def _dataset() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """A fixed dataset in the shape the convergence config uses, sized for the MoE vocab."""
    rng = np.random.default_rng(20260715)
    raw = rng.integers(3, gen_tiny_moe.HPARAMS.n_vocab, size=(N_SAMPLES, SEQ_LEN + 1))
    tokens, targets = raw[:, :SEQ_LEN], raw[:, 1 : SEQ_LEN + 1]
    weights = np.ones((N_SAMPLES, SEQ_LEN), dtype=np.float64)
    weights[:, : SEQ_LEN // 2] = 0.0
    return tokens, targets, weights


def _load_loras(base_path: pathlib.Path, adapter_path: pathlib.Path) -> dict[str, ref_llama.Lora]:
    """The adapter's A/B in float64 — 2D attention pairs and 3D expert stacks alike."""
    import gguf

    reader = gguf.GGUFReader(str(adapter_path), "r")
    data = {t.name: np.array(t.data, dtype=np.float64) for t in reader.tensors}
    return {
        t.name: ref_llama.Lora(a=data[f"{t.name}.lora_a"].copy(), b=data[f"{t.name}.lora_b"].copy())
        for t in enumerate_targets(base_path)
    }


def _wake_up_b(libs: _ffi.Libraries, model: Model, loras: dict[str, ref_llama.Lora]) -> None:
    """Perturb B off zero on BOTH sides identically, so dA (∝ B) is live everywhere."""
    rng = np.random.default_rng(11)
    index = _index_by_name(libs, model.adapter)
    for name, lora in loras.items():
        lora.b += rng.normal(0.0, 0.02, size=lora.b.shape)
        flat = np.ascontiguousarray(lora.b.astype(np.float32).reshape(-1))
        n = libs.farm.ll_adapter_set(
            model.adapter,
            index[name],
            True,
            flat.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            flat.size,
        )
        assert n == flat.size


def test_the_moe_reference_backward_is_itself_correct(tiny_moe_f32, tmp_path) -> None:
    """The float64 MoE backward vs a central finite difference of its own forward.

    Samples every LoRA tensor — expert slices included — plus, implicitly, the router path: the
    FD reruns the whole forward, top-k and all, so a backward missing the router-coupling term
    disagrees here before ggml ever enters the picture.
    """
    tokens, targets, weights = _dataset()
    adapter = tmp_path / "a.gguf"
    create_zero_adapter(tiny_moe_f32, adapter, r=RANK, alpha=ALPHA, seed=ADAPTER_SEED)

    base, hp = ref_moe.load_model(tiny_moe_f32)
    loras = _load_loras(tiny_moe_f32, adapter)
    scale = ref_llama.lora_scale(ALPHA, RANK, 1.0)

    rng = np.random.default_rng(11)
    for lora in loras.values():
        lora.b += rng.normal(0.0, 0.02, size=lora.b.shape)

    logits, cache = ref_moe.forward(base, hp, loras, scale, tokens[0])
    _, dlogits = ref_llama.loss_from_logits(logits, targets[0], weights[0])
    grads = ref_moe.backward(base, hp, loras, scale, cache, dlogits)

    def loss_at(name: str, is_b: bool, idx: tuple, delta: float) -> float:
        arr = loras[name].b if is_b else loras[name].a
        original = arr[idx]
        arr[idx] = original + delta
        lg, _ = ref_moe.forward(base, hp, loras, scale, tokens[0])
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
        f"the float64 MoE reference's own backward disagrees with its own forward: {worst:.2e}"
    )


def test_one_step_matches_the_moe_reference(tiny_moe_f32, tmp_path, libs) -> None:
    """The loss and every LoRA gradient — 3D expert stacks included — against float64."""
    tokens, targets, weights = _dataset()
    adapter = tmp_path / "a.gguf"
    create_zero_adapter(tiny_moe_f32, adapter, r=RANK, alpha=ALPHA, seed=ADAPTER_SEED)

    base, hp = ref_moe.load_model(tiny_moe_f32)
    loras = _load_loras(tiny_moe_f32, adapter)
    scale = ref_llama.lora_scale(ALPHA, RANK, 1.0)

    model = Model(
        tiny_moe_f32, libs=libs, n_ctx=N_CTX, n_ubatch=SEQ_LEN, training=True, n_threads=2
    )
    try:
        model.attach_adapter(adapter, scale=1.0)
        _wake_up_b(libs, model, loras)

        with Trainer(libs, model, TrainConfig(lr=1e-30)) as trainer:
            metrics = trainer.step(
                Batch(
                    tokens=[int(t) for t in tokens[0]],
                    targets=[int(t) for t in targets[0]],
                    weights=[float(w) for w in weights[0]],
                )
            )

            logits, cache = ref_moe.forward(base, hp, loras, scale, tokens[0])
            loss, dlogits = ref_llama.loss_from_logits(logits, targets[0], weights[0])
            grads = ref_moe.backward(base, hp, loras, scale, cache, dlogits)

            assert abs(metrics.loss - loss) / abs(loss) < 1e-5, (
                f"the MoE training loss disagrees with float64: {metrics.loss:.8f} vs {loss:.8f}"
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
                        f"{name}.lora_{'b' if is_b else 'a'}: ggml's MoE gradient disagrees with "
                        f"float64 by {rel:.2e} — if this is an _exps tensor, suspect the expert "
                        f"scatter or the router-coupling term first"
                    )
                    n_compared += 1
            # 14 LoRA tensors on the 2-layer MoE fixture (per layer: attn_q/k/v/output +
            # ffn_{gate,up,down}_exps), each compared for lora_a and lora_b -> 28 comparisons.
            assert n_compared == 28, (
                f"expected 28 LoRA comparisons on the MoE fixture, {n_compared}"
            )
    finally:
        model.close()


def test_the_moe_loss_curve_matches_the_reference(tiny_moe_f32, tmp_path, libs) -> None:
    """24 steps of the real ``train_sft`` on the MoE fixture, per step, against float64.

    Selections may legitimately CHANGE as the adapter moves the router logits; f32 and f64 could
    in principle disagree about a near-tie and fork the trajectory. With this seed they do not
    (measured); if this ever fails with a sudden step-function divergence, check for a routing
    flip near a tie before suspecting the gradient.
    """
    tokens, targets, weights = _dataset()
    adapter = tmp_path / "a.gguf"

    create_zero_adapter(tiny_moe_f32, adapter, r=RANK, alpha=ALPHA, seed=ADAPTER_SEED)
    model = Model(
        tiny_moe_f32, libs=libs, n_ctx=N_CTX, n_ubatch=SEQ_LEN, training=True, n_threads=2
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

    base, hp = ref_moe.load_model(tiny_moe_f32)
    loras = _load_loras(tiny_moe_f32, adapter)
    scale = ref_llama.lora_scale(ALPHA, RANK, 1.0)

    params: dict[str, np.ndarray] = {}
    for name, lora in loras.items():
        params[f"{name}.lora_a"] = lora.a
        params[f"{name}.lora_b"] = lora.b
    opt = ref_llama.AdamW(lr=LR)

    want = []
    for _epoch in range(EPOCHS):
        for i in range(N_SAMPLES):
            logits, cache = ref_moe.forward(base, hp, loras, scale, tokens[i])
            loss, dlogits = ref_llama.loss_from_logits(logits, targets[i], weights[i])
            grads = ref_moe.backward(base, hp, loras, scale, cache, dlogits)
            # A tensor whose experts were all unselected this step has no grad entry; AdamW still
            # wants one (ggml's accumulators are zero there, and zero moves m/v toward zero).
            for pname in params:
                grads.setdefault(pname, np.zeros_like(params[pname]))
            opt.step(params, grads)
            want.append(loss)
    want = np.array(want)

    assert len(got) == len(want) == EPOCHS * N_SAMPLES
    diff = np.abs(got - want)
    worst = int(diff.argmax())
    assert diff.max() < CURVE_TOL, (
        f"the MoE loss curve left the float64 reference at step {worst}: "
        f"{got[worst]:.8f} vs {want[worst]:.8f} (worst |diff| {diff.max():.2e})"
    )


def _moe_curve(libs, fixture, adapter, n_threads: int) -> list[float]:
    """A MoE training curve at a fixed thread count, for the determinism check below."""
    tokens, targets, weights = _dataset()
    create_zero_adapter(fixture, adapter, r=RANK, alpha=ALPHA, seed=ADAPTER_SEED)
    model = Model(
        fixture, libs=libs, n_ctx=N_CTX, n_ubatch=SEQ_LEN, training=True, n_threads=n_threads
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
        return [s.loss for s in result.steps]
    finally:
        model.close()


def test_the_moe_curve_is_bit_identical_across_thread_counts(tiny_moe_f32, tmp_path, libs) -> None:
    """Same host, same inputs, 1 vs 2 vs 4 threads: exactly equal (S1-50, extending S1-38).

    ``test_determinism.py`` pins this for the dense llama fixture; nothing pinned it for the MoE
    backward, whose expert-gather gradient rides on ``OUT_PROD_ID`` / ``OUT_PROD_ID_GRP`` -- a
    reduction over the tokens routed to each expert. If that sum's result came to depend on how the
    tokens were split across threads (a store-order or accumulation-order bug the tolerance-based
    gates would never see), this goes red. The Metal/CUDA/Vulkan lanes inherit the same claim on
    their own hosts, which is why the assertion is bitwise, not banded.
    """
    curves = {n: _moe_curve(libs, tiny_moe_f32, tmp_path / f"moe-{n}.gguf", n) for n in (1, 2, 4)}
    assert curves[1] == curves[2] == curves[4], (
        "the MoE loss curve depends on the thread count: an expert-gather reduction "
        "(OUT_PROD_ID/OUT_PROD_ID_GRP) now depends on how the tokens were split across threads, "
        "which breaks ADR-0002's same-host determinism claim for the MoE backward."
    )
