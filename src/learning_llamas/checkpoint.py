"""Checkpoint and resume a training run — optimizer state included.

The adapter file is not a checkpoint. It holds the weights, and a training run is more than its
weights: it is the weights **plus the optimizer's memory of how it got there**. AdamW carries a
first moment (a running gradient average), a second moment (a running squared-gradient average),
and an iteration counter that corrects both for bias.

Resume from the adapter alone and all three are gone. `m` and `v` start at zero, the bias correction
starts over at iteration 1, and the first steps after the restart are *much larger* than the ones
they follow — because the correction ``1/(1 - beta^t)`` is enormous at small ``t``. The loss curve
jumps at every resume. It is not obviously a bug; it looks like the model got worse, and it happens
only on the runs you restarted, which are the long ones.

So a checkpoint here is two files:

* the **adapter GGUF** (S1-08), which is a real adapter — stock llama.cpp loads it, and it is the
  thing you keep when the run is done;
* an **optimizer sidecar GGUF**, which is deliberately *not* a real adapter. Its ``general.type`` is
  ``learning-llamas-checkpoint``, so llama.cpp's loader refuses it rather than loading it as a
  strangely-shaped LoRA.

Keeping them apart matters: the artefact you ship and the artefact you resume from have different
lifetimes, and merging them would mean either shipping optimizer moments to users or making the
adapter unloadable.

Resume is bit-exact — but only if the weights and the moments both go back, and only in the right
ORDER. A run stopped at step N and resumed produces the same weights as one that never stopped, not
approximately but exactly, provided the step-N weights are written back *after* the priming step
that brings the moments into existence. That last part is not a file, and today it needs the debug
shim; :func:`restore_checkpoint` spells out why, and what the two-file recipe alone gets you
instead. Putting back the sidecar alone is not a resume either; there is a test for both halves.
"""

from __future__ import annotations

import ctypes
import pathlib
from dataclasses import dataclass, field

import numpy as np

from learning_llamas import _ffi

try:
    import gguf
except ImportError as exc:  # pragma: no cover - environment problem, not a code path
    raise ImportError("learning-llamas needs the vendored gguf-py; see adapter.py") from exc

# Bumped when the sidecar's layout changes in a way an older reader would misread. A reader that
# finds a version it does not know refuses, rather than guessing at the difference.
CHECKPOINT_VERSION = 1

# NOT "adapter". llama.cpp's loader keys off general.type, so this is what makes stock llama.cpp
# refuse the sidecar instead of trying to load optimizer moments as LoRA weights.
CHECKPOINT_TYPE = "learning-llamas-checkpoint"

KEY_VERSION = "farm.checkpoint.version"
KEY_ITER = "farm.checkpoint.iter"
KEY_OPT_PERIOD = "farm.checkpoint.opt_period"
KEY_MICRO_STEP = "farm.checkpoint.micro_step"
KEY_EPOCH = "farm.checkpoint.epoch"
KEY_SAMPLE = "farm.checkpoint.sample"


@dataclass
class TrainingPosition:
    """Where in the data a run had got to — so a resume does not re-read what it already trained on.

    Attributes:
        micro_step: How many batches have been pushed through.
        epoch: Which pass over the data.
        sample: The index within that pass.
    """

    micro_step: int = 0
    epoch: int = 0
    sample: int = 0


@dataclass
class Checkpoint:
    """The optimizer state of a run, as read from a sidecar.

    Attributes:
        iter: AdamW's iteration counter. Restoring the moments without this corrects them for the
            wrong iteration, which is its own quiet bug.
        opt_period: The gradient-accumulation period the run was using. Resuming into a different
            one is refused: the moments were accumulated over windows of a different size.
        position: Where in the data the run had got to.
        moments: ``{(param_name, is_v): values}``.
    """

    iter: int = 1
    opt_period: int = 1
    position: TrainingPosition = field(default_factory=TrainingPosition)
    moments: dict[tuple[str, bool], np.ndarray] = field(default_factory=dict)


def save_checkpoint(
    libs: _ffi.Libraries,
    ctx: int,
    out_path: str | pathlib.Path,
    opt_period: int = 1,
    position: TrainingPosition | None = None,
) -> int:
    """Write the optimizer sidecar of a live training context.

    Args:
        libs: The loaded native libraries.
        ctx: A ``llama_context *`` that has taken **at least one training step** — the moments do
            not exist before ggml-opt has built an optimizer graph.
        out_path: Where to write the sidecar.
        opt_period: The run's gradient-accumulation period, recorded so a resume can refuse a
            mismatch.
        position: Where in the data the run had got to.

    Returns:
        How many moment tensors were written (two per trainable tensor).

    Raises:
        RuntimeError: If the context has no optimizer state yet.
    """
    position = position or TrainingPosition()

    n = _ffi.check(libs.farm.ll_opt_state_count(ctx), "ll_opt_state_count")
    iteration = _ffi.check(libs.farm.ll_opt_get_iter(ctx), "ll_opt_get_iter")

    writer = gguf.GGUFWriter(str(out_path), arch=CHECKPOINT_TYPE)
    writer.add_type(CHECKPOINT_TYPE)

    writer.add_uint32(KEY_VERSION, CHECKPOINT_VERSION)
    writer.add_uint64(KEY_ITER, int(iteration))
    writer.add_uint32(KEY_OPT_PERIOD, int(opt_period))
    writer.add_uint64(KEY_MICRO_STEP, int(position.micro_step))
    writer.add_uint32(KEY_EPOCH, int(position.epoch))
    writer.add_uint64(KEY_SAMPLE, int(position.sample))

    for i in range(n):
        name_buf = ctypes.create_string_buffer(256)
        is_v = ctypes.c_bool()
        n_elements = ctypes.c_int64()

        _ffi.check(
            libs.farm.ll_opt_state_info(
                ctx, i, name_buf, 256, ctypes.byref(is_v), ctypes.byref(n_elements)
            ),
            "ll_opt_state_info",
        )

        name = name_buf.value.decode()
        values = _read_moment(libs, ctx, name, is_v.value)

        writer.add_tensor(f"{name}.adamw_{'v' if is_v.value else 'm'}", values)

    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()

    return n


def read_checkpoint(path: str | pathlib.Path) -> Checkpoint:
    """Read a sidecar.

    Args:
        path: The sidecar GGUF.

    Returns:
        Its optimizer state.

    Raises:
        ValueError: If it is not a checkpoint, or is a version this code does not know.
    """
    reader = gguf.GGUFReader(str(path), "r")

    kind = reader.get_field(gguf.Keys.General.TYPE)
    if kind is None or str(kind.contents()) != CHECKPOINT_TYPE:
        raise ValueError(
            f"{path} is not a learning-llamas checkpoint (general.type is "
            f"{kind.contents() if kind else None!r}, expected {CHECKPOINT_TYPE!r})"
        )

    version_field = reader.get_field(KEY_VERSION)
    version = int(version_field.contents()) if version_field else 0
    if version != CHECKPOINT_VERSION:
        raise ValueError(
            f"{path} is checkpoint version {version}; this build reads version "
            f"{CHECKPOINT_VERSION}. Refusing rather than guessing at the difference."
        )

    def kv(key: str, default: int = 0) -> int:
        field_ = reader.get_field(key)
        return int(field_.contents()) if field_ else default

    result = Checkpoint(
        iter=kv(KEY_ITER, 1),
        opt_period=kv(KEY_OPT_PERIOD, 1),
        position=TrainingPosition(
            micro_step=kv(KEY_MICRO_STEP),
            epoch=kv(KEY_EPOCH),
            sample=kv(KEY_SAMPLE),
        ),
    )

    for tensor in reader.tensors:
        for suffix, is_v in ((".adamw_m", False), (".adamw_v", True)):
            if tensor.name.endswith(suffix):
                param = tensor.name[: -len(suffix)]
                result.moments[(param, is_v)] = np.array(tensor.data, dtype=np.float32).reshape(-1)

    if not result.moments:
        raise ValueError(f"{path} holds no optimizer moments")

    return result


def restore_checkpoint(
    libs: _ffi.Libraries, ctx: int, checkpoint: Checkpoint, opt_period: int = 1
) -> int:
    """Put a sidecar's optimizer state back into a live training context.

    This restores the moments and AdamW's iteration counter, and **nothing else**. The adapter's
    weights are the other half of a checkpoint and they are not touched here — they travel in the
    adapter GGUF (:func:`~learning_llamas.adapter.save_adapter`).

    The context must already have taken **one training step**, because that is when ggml-opt builds
    the optimizer graph and the moments come into existence. That priming step is taken with zeroed
    moments and a bias correction at iteration 1, so it *moves the weights*, and nothing in this
    module moves them back: the restore below overwrites ``m``, ``v`` and ``iter``, not the
    parameters. So a resume is:

    1. Start from the step-N weights: attach the adapter GGUF that was saved alongside this
       sidecar.
    2. Take the priming step, so that the optimizer graph — and with it the moments — exists. It
       moves the weights off step N.
    3. Write the step-N weights back over what the priming step did, then call this.

    Necessary and sufficient are not the same thing here, and the difference is worth stating.
    Parts 1 and 2 are all the public names can do, and that resume is **not** exact: it diverges by
    exactly the priming step's AdamW update, which at a realistic learning rate is the largest
    single update the run will ever take (``tests/test_checkpoint.py`` measures 5.5e-2 relative).
    Part 3 is what makes it exact — and because it overwrites every parameter, it also makes part 1
    redundant. That is why the bit-exactness test starts from a *zero-init* adapter, never attaches
    the step-N one, and still matches the uninterrupted run element for element.

    Part 3 has no public API today: the test does it through the debug shim
    (``ll_debug_set_tensor``). Until it has one, "attach the adapter, step, restore, carry on" — the
    only recipe expressible in public names — is a resume that diverges on its first step, which is
    why the protocol is written out here rather than left as "one step, restore, carry on".

    Args:
        libs: The loaded native libraries.
        ctx: A ``llama_context *`` that has taken at least one training step.
        checkpoint: What to restore.
        opt_period: The gradient-accumulation period the resumed run will use.

    Returns:
        How many moments were restored.

    Raises:
        ValueError: If the checkpoint's ``opt_period`` differs, or it does not cover exactly the
            parameters this context has.
        RuntimeError: If the context has no optimizer state yet.
    """
    if checkpoint.opt_period != opt_period:
        raise ValueError(
            f"the checkpoint was written with opt_period={checkpoint.opt_period} but this run uses "
            f"{opt_period}. Its moments were accumulated over windows of a different size, so "
            f"resuming into a different period would silently change what they mean."
        )

    live = _live_moments(libs, ctx)

    missing = live - set(checkpoint.moments)
    extra = set(checkpoint.moments) - live

    if missing or extra:
        raise ValueError(
            f"the checkpoint does not match this adapter: "
            f"{len(missing)} moment(s) missing, {len(extra)} unexpected. "
            f"It is a checkpoint of a different run."
        )

    for (name, is_v), values in checkpoint.moments.items():
        buf = (ctypes.c_float * len(values))(*values.tolist())
        _ffi.check(
            libs.farm.ll_opt_state_set(ctx, name.encode(), is_v, buf, len(values)),
            "ll_opt_state_set",
        )

    _ffi.check(libs.farm.ll_opt_set_iter(ctx, checkpoint.iter), "ll_opt_set_iter")

    return len(checkpoint.moments)


def _live_moments(libs: _ffi.Libraries, ctx: int) -> set[tuple[str, bool]]:
    """Every ``(param_name, is_v)`` the context actually has."""
    n = _ffi.check(libs.farm.ll_opt_state_count(ctx), "ll_opt_state_count")

    out: set[tuple[str, bool]] = set()
    for i in range(n):
        name_buf = ctypes.create_string_buffer(256)
        is_v = ctypes.c_bool()
        n_elements = ctypes.c_int64()

        _ffi.check(
            libs.farm.ll_opt_state_info(
                ctx, i, name_buf, 256, ctypes.byref(is_v), ctypes.byref(n_elements)
            ),
            "ll_opt_state_info",
        )
        out.add((name_buf.value.decode(), is_v.value))

    return out


def _read_moment(libs: _ffi.Libraries, ctx: int, name: str, is_v: bool) -> np.ndarray:
    n = _ffi.check(
        libs.farm.ll_opt_state_get(ctx, name.encode(), is_v, None, 0), "ll_opt_state_get"
    )

    buf = (ctypes.c_float * n)()
    got = _ffi.check(
        libs.farm.ll_opt_state_get(ctx, name.encode(), is_v, buf, n), "ll_opt_state_get"
    )

    return np.frombuffer(buf, dtype=np.float32, count=got).copy()
