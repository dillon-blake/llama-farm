"""Run the same experiment through llama.cpp and through the float64 reference.

One harness, three consumers: the gate (``tests/test_convergence.py``, which runs the recorded
spec and also compares against the committed PEFT curve), the variants
(``tests/test_convergence_variants.py``, which move one parameter at a time off the recorded
values), and the determinism check (``tests/test_determinism.py``). Future backend lanes drive the
same harness via ``--device`` rather than forking it.
"""

from __future__ import annotations

import ctypes
import pathlib

import gguf
import numpy as np

from learning_llamas import Model, _ffi, create_zero_adapter, enumerate_targets
from learning_llamas.adapter import _index_by_name
from learning_llamas.data import MaskedSample
from learning_llamas.train import SFTConfig, TrainConfig, Trainer, train_sft
from learning_llamas.train.loop import Batch

from .. import reference_llama as ref
from . import config

N_CTX = 64

ConvData = tuple[np.ndarray, np.ndarray, np.ndarray]


def samples(conv_data: ConvData) -> list[MaskedSample]:
    """The dataset as S1-05 wants it.

    ``to_batch`` re-derives the targets by shifting, so a sample carries ``SEQ_LEN + 1`` tokens and
    its weights are shifted onto the *prediction*: ``weights[i]`` grades the guess at
    ``tokens[i + 1]``. Handing it the already-shifted arrays would train the model one step out of
    phase, and the loss would still fall.
    """
    tokens, targets, weights = conv_data
    return [
        MaskedSample(
            tokens=[int(t) for t in tokens[i]] + [int(targets[i][-1])],
            weights=[0.0] + [float(w) for w in weights[i]],
        )
        for i in range(config.N_SAMPLES)
    ]


def sft_config(spec: config.RunSpec = config.RECORDED) -> SFTConfig:
    return SFTConfig(
        lr=spec.lr,
        betas=spec.betas,
        eps=spec.eps,
        weight_decay=spec.weight_decay,
        grad_accum=spec.grad_accum,
        seq_len=config.SEQ_LEN,
        pad_id=config.PAD_ID,
        epochs=spec.epochs,
        shuffle=config.SHUFFLE,
        schedule="constant",
    )


def load_loras(base_path: pathlib.Path, adapter_path: pathlib.Path) -> dict[str, ref.Lora]:
    """The same A/B the GGUF adapter holds, as float64 — so both sides start identical."""
    reader = gguf.GGUFReader(str(adapter_path), "r")
    data = {t.name: np.array(t.data, dtype=np.float64) for t in reader.tensors}
    return {
        t.name: ref.Lora(
            a=data[f"{t.name}.lora_a"].copy(),
            b=data[f"{t.name}.lora_b"].copy(),
        )
        for t in enumerate_targets(base_path)
    }


def reference_curve(
    base_path: pathlib.Path,
    adapter_path: pathlib.Path,
    conv_data: ConvData,
    spec: config.RunSpec = config.RECORDED,
) -> list[float]:
    """Run the float64 reference for the same steps: forward, backward, AdamW."""
    tokens, targets, weights = conv_data
    base, hp = ref.load_model(base_path)
    loras = load_loras(base_path, adapter_path)
    scale = ref.lora_scale(spec.alpha, spec.rank, spec.user_scale)

    params: dict[str, np.ndarray] = {}
    for name, lora in loras.items():
        params[f"{name}.lora_a"] = lora.a
        params[f"{name}.lora_b"] = lora.b

    opt = ref.AdamW(lr=spec.lr, betas=spec.betas, eps=spec.eps, weight_decay=spec.weight_decay)

    # Gradient accumulation is a SUM over the window, and the optimizer steps on the window's last
    # micro-batch — ggml-opt's opt_period semantics, restated in TrainConfig.grad_accum's
    # docstring. Not a mean: a stack that averaged would show up here as a factor-of-k gradient.
    curve: list[float] = []
    pending: dict[str, np.ndarray] = {}
    in_window = 0
    for _epoch in range(spec.epochs):
        for i in range(config.N_SAMPLES):
            logits, cache = ref.forward(base, hp, loras, scale, tokens[i])
            loss, dlogits = ref.loss_from_logits(logits, targets[i], weights[i])
            grads = ref.backward(base, hp, loras, scale, cache, dlogits)
            for name, g in grads.items():
                pending[name] = pending.get(name, 0.0) + g
            in_window += 1
            if in_window == spec.grad_accum:
                opt.step(params, pending)
                pending, in_window = {}, 0
            curve.append(loss)

    return curve


def one_step_grad_errors(
    libs: _ffi.Libraries,
    base_path: pathlib.Path,
    adapter_path: pathlib.Path,
    conv_data: ConvData,
    spec: config.RunSpec = config.RECORDED,
) -> tuple[float, dict[str, float]]:
    """One training step at ``lr = 1e-30``: the loss and every LoRA gradient, vs float64.

    Returns the loss's relative error and, per LoRA tensor, the max element error relative to the
    tensor's largest element — so a failure can name the tensor.

    B is perturbed off zero (identically on both sides) before the step, because dA is proportional
    to B: with a fresh adapter every dA is exactly zero and half the comparison would be vacuous.
    ``lr = 1e-30`` keeps the optimizer from moving the weights before the gradient is read back.
    """
    tokens, targets, weights = conv_data
    create_zero_adapter(
        base_path, adapter_path, r=spec.rank, alpha=spec.alpha, seed=spec.adapter_seed
    )

    base, hp = ref.load_model(base_path)
    loras = load_loras(base_path, adapter_path)
    scale = ref.lora_scale(spec.alpha, spec.rank, spec.user_scale)

    model = Model(
        base_path, libs=libs, n_ctx=N_CTX, n_ubatch=config.SEQ_LEN, training=True, n_threads=2
    )
    try:
        model.attach_adapter(adapter_path, scale=spec.user_scale)

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

        with Trainer(libs, model, TrainConfig(lr=1e-30)) as trainer:
            metrics = trainer.step(
                Batch(
                    tokens=[int(t) for t in tokens[0]],
                    targets=[int(t) for t in targets[0]],
                    weights=[float(w) for w in weights[0]],
                )
            )

            logits, cache = ref.forward(base, hp, loras, scale, tokens[0])
            loss, dlogits = ref.loss_from_logits(logits, targets[0], weights[0])
            grads = ref.backward(base, hp, loras, scale, cache, dlogits)

            loss_rel = abs(metrics.loss - loss) / abs(loss)

            by_tensor: dict[str, float] = {}
            for name in loras:
                for suffix in ("lora_a", "lora_b"):
                    expected = grads[f"{name}.{suffix}"]
                    buf = (ctypes.c_float * expected.size)()
                    got = libs.farm.ll_debug_grad(
                        model.ctx, name.encode(), suffix == "lora_b", buf, expected.size
                    )
                    assert got == expected.size, f"ll_debug_grad({name}) returned {got}"

                    actual = np.frombuffer(buf, dtype=np.float32, count=got).reshape(expected.shape)
                    by_tensor[f"{name}.{suffix}"] = float(
                        np.abs(actual.astype(np.float64) - expected).max()
                        / max(np.abs(expected).max(), 1e-30)
                    )
            return loss_rel, by_tensor
    finally:
        model.close()


def train(
    libs: _ffi.Libraries,
    base_path: pathlib.Path,
    adapter_path: pathlib.Path,
    conv_data: ConvData,
    spec: config.RunSpec = config.RECORDED,
    n_threads: int = 2,
) -> list[float]:
    """Run the real stack: create the adapter, attach it, and ``train_sft``."""
    create_zero_adapter(
        base_path, adapter_path, r=spec.rank, alpha=spec.alpha, seed=spec.adapter_seed
    )
    model = Model(
        base_path,
        libs=libs,
        n_ctx=N_CTX,
        n_ubatch=config.SEQ_LEN,
        training=True,
        n_threads=n_threads,
    )
    try:
        model.attach_adapter(adapter_path, scale=spec.user_scale)
        result = train_sft(libs, model, samples(conv_data), sft_config(spec))
        return [step.loss for step in result.steps]
    finally:
        model.close()
