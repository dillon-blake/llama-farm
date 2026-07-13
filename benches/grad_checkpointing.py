"""What does gradient checkpointing actually buy, and what does it cost? (S1-17)

Checkpointing is a trade, and both sides of it are measurable, so measure both. The forward pass's
activations are kept only at layer boundaries and everything between is recomputed in the backward:
that is less memory and more arithmetic, and the exchange rate is what this prints.

The third column is the one that makes it a trade rather than a gamble: the trained weights are
compared bit-for-bit against the unchecked run. Recomputing an activation is running the same ops on
the same inputs, so it must produce the same bits, so the gradients must be the same bits. If that
column ever says anything but `identical`, the memory saving is not a saving — it is a bug.

Run it:

    python benches/grad_checkpointing.py --n-layer 12 --seq 512
"""

from __future__ import annotations

import argparse
import ctypes
import pathlib
import sys
import time

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from learning_llamas import Model, _ffi  # noqa: E402
from learning_llamas.adapter import create_zero_adapter  # noqa: E402
from learning_llamas.train import TrainConfig, Trainer  # noqa: E402
from learning_llamas.train.loop import Batch  # noqa: E402
from tests.fixtures import gen_tiny_llama  # noqa: E402
from tests.fixtures.gen_tiny_llama import TinyLlamaHParams  # noqa: E402


def _run(libs, base, adapter, batch, n_ctx, seq_len, segment_len, n_steps):
    """One run: its losses, its trained weights, its peak activation memory, its time per step."""
    model = Model(base, libs=libs, n_ctx=n_ctx, n_ubatch=seq_len, training=True)
    model.attach_adapter(adapter, scale=1.0)

    with Trainer(libs, model, TrainConfig(lr=1e-2)) as trainer:
        if segment_len:
            _ffi.check(
                libs.farm.ll_set_grad_checkpointing(model.ctx, segment_len),
                "ll_set_grad_checkpointing",
            )

        start = time.perf_counter()
        losses = [trainer.step(batch).loss for _ in range(n_steps)]
        per_step = (time.perf_counter() - start) / n_steps

        peak = libs.farm.ll_compute_buffer_bytes(model.ctx)

    weights = []
    for i in range(libs.farm.ll_adapter_n_tensors(model.adapter)):
        for is_b in (False, True):
            n = libs.farm.ll_adapter_get(model.adapter, i, is_b, None, 0)
            buf = (ctypes.c_float * n)()
            libs.farm.ll_adapter_get(model.adapter, i, is_b, buf, n)
            weights.append(np.frombuffer(buf, dtype=np.float32, count=n).copy())

    model.close()

    return losses, np.concatenate(weights), peak, per_step


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-layer", type=int, default=12)
    parser.add_argument("--seq", type=int, default=512)
    parser.add_argument("--rank", type=int, default=4)
    parser.add_argument("--steps", type=int, default=2)
    parser.add_argument("--variant", default="q4_k", choices=gen_tiny_llama.VARIANTS)
    parser.add_argument("--segments", type=int, nargs="+", default=[4, 2, 1])
    args = parser.parse_args()

    libs = _ffi.load()
    libs.llama.llama_backend_init()

    cache = pathlib.Path(".pytest-fixtures")
    hp = TinyLlamaHParams(n_layer=args.n_layer, n_ctx_train=max(1024, 2 * args.seq))
    base, _ = gen_tiny_llama.build(args.variant, cache, hp=hp)

    adapter = cache / f"bench-ckpt-r{args.rank}-l{args.n_layer}.gguf"
    if not adapter.exists():
        create_zero_adapter(base, adapter, r=args.rank, seed=7)

    rng = np.random.default_rng(1)
    tokens = [int(x) for x in rng.integers(1, 400, size=args.seq)]
    batch = Batch(
        tokens=tokens,
        targets=tokens[1:] + [tokens[0]],
        weights=[0.0] * 16 + [1.0] * (args.seq - 16),
    )

    n_ctx = max(1024, 2 * args.seq)
    common = (
        libs,
        base,
        adapter,
        batch,
        n_ctx,
        args.seq,
    )

    print(
        f"\ntiny-llama n_layer={args.n_layer} n_embd={hp.n_embd}, {args.variant.upper()} base, "
        f"LoRA r={args.rank}, seq {args.seq}\n"
    )
    print(f"{'segment':>8} {'peak activations':>18} {'vs off':>8} {'step':>9}  gradients")

    off_losses, off_w, off_peak, off_t = _run(*common, 0, args.steps)
    print(f"{'off':>8} {off_peak / 1e6:>15.2f} MB {'1.00x':>8} {off_t * 1e3:>6.0f} ms  --")

    for segment in args.segments:
        losses, weights, peak, per_step = _run(*common, segment, args.steps)

        same = losses == off_losses and np.array_equal(weights, off_w)

        print(
            f"{segment:>8} {peak / 1e6:>15.2f} MB {peak / off_peak:>7.2f}x "
            f"{per_step * 1e3:>6.0f} ms  {'identical' if same else '*** DIFFERS ***'}"
        )


if __name__ == "__main__":
    main()
