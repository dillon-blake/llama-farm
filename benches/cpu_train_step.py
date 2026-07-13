"""What does one CPU LoRA training step actually cost? (S1-32)

The point of this is not a leaderboard. It is that "can I fine-tune on my laptop?" has an answer,
and the honest form of that answer is a number with the assumptions attached — thread count, batch
shape, rank, and which quantization the base is in.

Two things this reports that a single tokens/s number hides:

**Scaling with threads is not linear, and the point where it stops is the useful fact.** Beyond it,
adding threads costs power and buys nothing; below it, the machine is idle. The sweep finds it.

**The backward is not free, and its cost is not a fixed multiple of the forward.** It is dominated
by ``OUT_PROD``, which dequantizes the base weight on every call — so the backward's share *rises*
as the base gets more compressed. The forward/backward split is measured rather than assumed, by
timing a forward-only step against a full one.

Run:  python benches/cpu_train_step.py [--model PATH] [--threads 1,2,4,8] [--steps 20]
"""

from __future__ import annotations

import argparse
import ctypes
import os
import pathlib
import statistics
import sys
import time

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "tests"))

from learning_llamas import _ffi  # noqa: E402
from learning_llamas.adapter import create_zero_adapter  # noqa: E402
from learning_llamas.train import Batch, TrainConfig, Trainer  # noqa: E402


class Model:
    """A trainable model, built the way the trainer needs it."""

    def __init__(self, libs, path, n_ctx, n_ubatch, n_threads):
        mp = libs.llama.llama_model_default_params()
        mp.n_gpu_layers = 0
        mp.use_extra_bufts = False  # the backward's OUT_PROD is unschedulable on repacked weights

        self.model = libs.llama.llama_model_load_from_file(str(path).encode(), mp)
        if not self.model:
            raise RuntimeError(f"failed to load {path}")

        cp = libs.llama.llama_context_default_params()
        cp.n_ctx = cp.n_batch = n_ctx
        cp.n_ubatch = n_ubatch
        cp.n_threads = cp.n_threads_batch = n_threads

        self.ctx = libs.llama.llama_init_from_model(self.model, cp)
        self.adapter = None
        self._libs = libs

    def attach(self, path):
        self.adapter = self._libs.llama.llama_adapter_lora_init(self.model, str(path).encode())
        arr = (ctypes.c_void_p * 1)(self.adapter)
        self._libs.llama.llama_set_adapters_lora(self.ctx, arr, 1, (ctypes.c_float * 1)(1.0))

    def close(self):
        self._libs.llama.llama_free(self.ctx)
        self._libs.llama.llama_model_free(self.model)


def _batch(n: int) -> Batch:
    tokens = [7, 11, 13, 17] * (n // 4)
    return Batch(
        tokens=tokens,
        targets=tokens[1:] + [tokens[0]],
        weights=[1.0] * n,
    )


def measure(libs, model_path, adapter_path, *, n_ctx, n_ubatch, threads, rank, steps, warmup):
    """Median seconds per step, forward-only and full, at one thread count."""
    model = Model(libs, model_path, n_ctx, n_ubatch, threads)
    model.attach(adapter_path)

    batch = _batch(n_ubatch)

    try:
        with Trainer(libs, model, TrainConfig(lr=1e-4)) as trainer:
            for _ in range(warmup):
                trainer.step(batch)

            full = []
            for _ in range(steps):
                t = time.perf_counter()
                trainer.step(batch)
                full.append(time.perf_counter() - t)

            fwd = []
            for _ in range(steps):
                t = time.perf_counter()
                trainer.evaluate([batch])
                fwd.append(time.perf_counter() - t)
    finally:
        model.close()

    return statistics.median(full), statistics.median(fwd)


def main() -> int:
    default = pathlib.Path("tests/.fixtures")
    fixtures = sorted(default.glob("*/tiny-llama-q4_k.gguf"))

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=pathlib.Path, default=fixtures[0] if fixtures else None)
    parser.add_argument("--threads", default="1,2,4")
    parser.add_argument("--n-ubatch", type=int, default=128)
    parser.add_argument("--rank", type=int, default=8)
    parser.add_argument("--steps", type=int, default=15)
    parser.add_argument("--warmup", type=int, default=3)
    args = parser.parse_args()

    if args.model is None or not args.model.exists():
        print("no model: pass --model, or run the test suite once to build the fixtures")
        return 1

    libs = _ffi.load()
    libs.llama.llama_backend_init()

    adapter = pathlib.Path(os.environ.get("TMPDIR", "/tmp")) / "bench-adapter.gguf"
    create_zero_adapter(args.model, adapter, r=args.rank, seed=0)

    print(f"model      {args.model}")
    print(
        f"n_ubatch   {args.n_ubatch}   rank {args.rank}   "
        f"steps {args.steps} (warmup {args.warmup})"
    )
    print()
    print(
        f"  {'threads':>7}  {'tok/s':>9}  {'step (ms)':>10}  {'fwd (ms)':>9}  "
        f"{'bwd share':>9}  {'scaling':>8}"
    )
    print(f"  {'-' * 7}  {'-' * 9}  {'-' * 10}  {'-' * 9}  {'-' * 9}  {'-' * 8}")

    base = None

    for threads in [int(t) for t in args.threads.split(",")]:
        full, fwd = measure(
            libs,
            args.model,
            adapter,
            n_ctx=max(256, args.n_ubatch),
            n_ubatch=args.n_ubatch,
            threads=threads,
            rank=args.rank,
            steps=args.steps,
            warmup=args.warmup,
        )

        toks = args.n_ubatch / full
        base = base or toks

        # The backward's share of the step. It is not a fixed multiple of the forward: OUT_PROD
        # dequantizes the base weight on every call, so the more compressed the base, the more of
        # the step the backward is.
        bwd_share = (full - fwd) / full

        print(
            f"  {threads:>7}  {toks:>9.0f}  {full * 1000:>10.2f}  {fwd * 1000:>9.2f}  "
            f"{bwd_share:>8.0%}  {toks / base:>7.2f}x"
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
