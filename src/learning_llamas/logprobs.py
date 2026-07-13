"""S1-13 — per-token logprobs without ever building the logits.

DPO and GRPO ask the model one question over and over: *what probability did you assign to the
token that was actually there?* That is `n_tokens` floats. The obvious way to get them is to run
the lm_head, materialize `[n_tokens, n_vocab]` logits, log-softmax, and gather — and then throw
away everything except the diagonal.

At a 128k vocab that discarded tensor is **512 KB per token row**: 4096 tokens is 2 GB of F32,
for 16 KB of answer. On the machines this project exists for, that is the difference between
running and not.

So don't build it. Decode with embeddings output on — llama.cpp then stops one op short and hands
back the post-final-norm hidden states, never running the lm_head at all — and do the projection
here, a chunk of token rows at a time:

    logits_chunk = lm_head @ hidden_chunk        # [n_vocab, chunk_rows], transient
    logp_chunk   = -ce_sparse(logits_chunk, labels, weights)

`ce_sparse` (S1-04, ADR-0003) returns `w_i * (logsumexp(z_i) - z_i[label_i])` per token, unreduced
— which *is* the negated logprob of the realized token, with the caller's mask already applied and
softcap/logit-scale already handled in stable-lse math. So the log-softmax never materializes
either. Peak transient is one chunk of logits, and the answer is `n_tokens` floats.

The lm_head itself is **never copied**: it is read straight out of the GGUF's mmap, still
quantized, and wrapped as a ggml buffer in place (`ggml_backend_cpu_buffer_from_ptr`). Copying a
128k-vocab lm_head to compute logprobs would cost more memory than the full-logits path this
module exists to avoid.

The graph is two ops and touches no llama_context, so it is sched-agnostic by construction:
stages 2-4 move it to a GPU by changing which device `ggml_backend_init_by_type` is asked for.

Provenance (ROADMAP §13, S0-01)
-------------------------------
The chunked-logsumexp *math* is Apache-2.0 (unsloth `kernels/cross_entropy_loss.py`, including the
softcap and logit-scale handling) and was already imported into the `ce_sparse` kernel by S1-04.
Everything at the orchestration level in this module — the chunk loop, the masking policy, the
tied-embedding fallback, and `choose_chunk_rows` — is **original work, re-derived from the idea
only**. unsloth's own chunked-logprob orchestration is AGPL-marked
(`models/rl_replacements.py:1191`) and its autotune helper comes from the unaudited `unsloth_zoo`
package; neither was consulted. No AGPL code was read while writing this file.
"""

from __future__ import annotations

import ctypes
import pathlib
from dataclasses import dataclass

import numpy as np

from . import _ffi

# ggml.h
GGML_TYPE_F32 = 0
GGML_TYPE_I32 = 26

# ggml-backend.h
GGML_BACKEND_DEVICE_TYPE_CPU = 0
GGML_STATUS_SUCCESS = 0

# A ggml graph node costs a fixed-size struct plus hash-table slots. The chunk graph has at most
# eight nodes (mul_mat, ce_sparse, negate, plus the LoRA delta's three), so the default graph size
# is ample; this is just the metadata arena for the tensor structs themselves.
_GRAPH_ARENA_BYTES = 16 * 1024 * 1024

#: What a chunk's logits transient is allowed to cost, when the caller does not say.
DEFAULT_MEM_BUDGET = 128 * 1024 * 1024


@dataclass(frozen=True)
class LmHead:
    """The output projection, still quantized, viewed in place in the GGUF's mmap.

    Attributes:
        data: The raw tensor bytes — a view into the mmap, not a copy.
        ggml_type: The ggml type enum the bytes are encoded in (Q4_K, F32, ...).
        n_embd: Rows of the projection input, i.e. ``ne[0]``.
        n_vocab: Rows of the projection output, i.e. ``ne[1]``.
        tied: True if this came from ``token_embd.weight`` because the model has no
            ``output.weight`` — the model ties its input and output embeddings.
    """

    data: np.ndarray
    ggml_type: int
    n_embd: int
    n_vocab: int
    tied: bool


def load_lm_head(path: pathlib.Path) -> LmHead:
    """Read a model's output projection from its GGUF, without copying it.

    ``GGUFReader`` mmaps the file, so the returned array is a *view*: no dequantization, no
    allocation proportional to the lm_head's size.

    A model with no ``output.weight`` ties its embeddings and projects with ``token_embd.weight``
    instead. Getting this wrong is silent — you get logits, they are just the wrong logits — so the
    fallback is explicit and the result records which one it took.

    Args:
        path: The base model's GGUF.

    Returns:
        The lm_head, as a view into the mmap.

    Raises:
        KeyError: If the GGUF has neither an output projection nor a token embedding.
    """
    from gguf import GGUFReader  # noqa: PLC0415 - a heavy import, and only this function needs it

    reader = GGUFReader(str(path), "r")
    by_name = {t.name: t for t in reader.tensors}

    tied = "output.weight" not in by_name
    name = "token_embd.weight" if tied else "output.weight"

    if name not in by_name:
        raise KeyError(
            f"{path.name} has neither 'output.weight' nor 'token_embd.weight': there is no "
            f"output projection to read."
        )

    tensor = by_name[name]

    # GGUF stores shapes reversed relative to ggml: `shape` is (n_embd, n_vocab) already in ggml's
    # ne order, i.e. ne[0] is the contiguous dimension.
    n_embd, n_vocab = (int(x) for x in tensor.shape[:2])

    return LmHead(
        data=tensor.data,
        ggml_type=int(tensor.tensor_type),
        n_embd=n_embd,
        n_vocab=n_vocab,
        tied=tied,
    )


def choose_chunk_rows(
    n_tokens: int,
    n_vocab: int,
    *,
    mem_budget: int = DEFAULT_MEM_BUDGET,
    override: int | None = None,
) -> int:
    """Pick the largest chunk whose logits transient fits the budget.

    The transient is ``chunk_rows * n_vocab * 4`` bytes of F32 logits, and it is the only thing in
    the pass that scales with the chunk. So the choice is a division, and the reason it is a
    function rather than a constant is that the two numbers it depends on — the vocab and the
    caller's memory — vary by three orders of magnitude across the models this has to serve.

    There is always a floor of one row: a single row that busts the budget is still the smallest
    thing that can be computed, and failing here would be worse than being slow.

    Args:
        n_tokens: Tokens in the pass. The chunk is never larger than this.
        n_vocab: The model's vocabulary size.
        mem_budget: Bytes the logits transient may occupy.
        override: Use this chunk size instead, clamped to ``[1, n_tokens]``.

    Returns:
        Rows per chunk, in ``[1, n_tokens]``.

    Raises:
        ValueError: If ``override`` is not positive, or the budget is not positive.
    """
    if n_tokens <= 0:
        return 1
    if override is not None:
        if override <= 0:
            raise ValueError(f"chunk_rows override must be positive, got {override}")
        return min(override, n_tokens)
    if mem_budget <= 0:
        raise ValueError(f"mem_budget must be positive, got {mem_budget}")

    row_bytes = n_vocab * 4
    rows = mem_budget // row_bytes

    return max(1, min(int(rows), n_tokens))


@dataclass(frozen=True)
class LoRADelta:
    """A LoRA that targets the output projection: ``W_eff = W + scale * B @ A``.

    Shapes are **numpy row-major**, which is the reverse of ggml's ``ne`` order — the same
    convention the GGUF writer uses, so an array read out of an adapter needs no transposing.

    Attributes:
        a: ``(r, n_embd)`` F32.
        b: ``(n_vocab, r)`` F32.
        scale: The *effective* scale llama.cpp applies — ``user_scale * alpha / r``, or just
            ``user_scale`` when alpha is 0 (``llama-adapter.h:55``). :func:`output_lora` works it
            out; passing the user scale here instead would silently score a differently-weighted
            adapter than the one the model is running.
    """

    a: np.ndarray
    b: np.ndarray
    scale: float


def output_lora(
    libs: _ffi.Libraries,
    adapter: int,
    adapter_path: pathlib.Path,
    *,
    scale: float = 1.0,
) -> LoRADelta | None:
    """The delta the adapter applies to the output projection, or None if it targets something else.

    A/B are read from the **live** adapter rather than off disk, on purpose: an online algorithm
    scores the policy it is currently training, and that policy is in memory, not in a file. The
    alpha and rank come from the GGUF, which does not change as the adapter trains.

    Returning None is a real answer, not a failure: most adapters target the attention and MLP
    projections and leave ``output.weight`` alone, and for those the base weight *is* the whole
    lm_head. Passing None into :func:`chunked_token_logprobs` is then exactly right.

    Args:
        libs: The loaded native libraries.
        adapter: A ``llama_adapter_lora *`` — the handle from ``llama_adapter_lora_init``, not a
            context.
        adapter_path: The adapter GGUF — for its alpha and rank.
        scale: The user scale the adapter was attached with.

    Returns:
        The delta, or None if the adapter does not target the output projection.
    """
    from .adapter import read_adapter  # noqa: PLC0415 - avoids a cycle at import time

    info = read_adapter(adapter_path)
    if "output.weight" not in info.ranks:
        return None

    index, ne_a, ne_b = _adapter_index_of(libs, adapter, "output.weight")
    rank = info.ranks["output.weight"]

    # ne is ggml's order, so numpy's shape is its reverse -- the same flip adapter.py makes when it
    # reads an adapter back (adapter.py:444).
    a = _read_adapter_tensor(libs, adapter, index, is_b=False).reshape(int(ne_a[1]), int(ne_a[0]))
    b = _read_adapter_tensor(libs, adapter, index, is_b=True).reshape(int(ne_b[1]), int(ne_b[0]))

    # llama.cpp's rule, exactly: alpha == 0 does not mean "no adapter", it means the alpha/rank
    # factor is dropped (llama-adapter.h:55).
    effective = scale * info.alpha / rank if info.alpha else scale

    return LoRADelta(a=a, b=b, scale=effective)


def _adapter_index_of(
    libs: _ffi.Libraries, adapter: int, base_name: str
) -> tuple[int, ctypes.Array, ctypes.Array]:
    """The adapter's index for a base tensor, with the A and B ``ne``. Raises if it is not there."""
    for i in range(_ffi.check(libs.farm.ll_adapter_n_tensors(adapter), "ll_adapter_n_tensors")):
        name = ctypes.create_string_buffer(256)
        # GGML_MAX_DIMS elements each, not one: the shim writes four int64s through these, and a
        # scalar here is a stack smash that shows up as a segfault somewhere else entirely.
        ne_a = (ctypes.c_int64 * 4)()
        ne_b = (ctypes.c_int64 * 4)()
        _ffi.check(
            libs.farm.ll_adapter_tensor_info(adapter, i, name, 256, ne_a, ne_b),
            "ll_adapter_tensor_info",
        )
        if name.value.decode() == base_name:
            return i, ne_a, ne_b

    raise KeyError(f"the adapter does not target {base_name!r}")


def _read_adapter_tensor(
    libs: _ffi.Libraries, adapter: int, index: int, *, is_b: bool
) -> np.ndarray:
    n = _ffi.check(libs.farm.ll_adapter_get(adapter, index, is_b, None, 0), "ll_adapter_get")
    buf = (ctypes.c_float * n)()
    got = _ffi.check(libs.farm.ll_adapter_get(adapter, index, is_b, buf, n), "ll_adapter_get")
    return np.frombuffer(buf, dtype=np.float32, count=got).copy()


class _Arena:
    """A ggml context plus the buffers hung off it, freed together."""

    def __init__(self, libs: _ffi.Libraries, backend: int) -> None:
        self.libs = libs
        self.backend = backend
        self.buffers: list[int] = []
        params = _ffi.ggml_init_params(mem_size=_GRAPH_ARENA_BYTES, mem_buffer=None, no_alloc=True)
        self.ctx = libs.ggml_base.ggml_init(params)
        if not self.ctx:
            raise RuntimeError("ggml_init failed for the chunked-logprob graph")

    def close(self) -> None:
        for buf in self.buffers:
            self.libs.ggml_base.ggml_backend_buffer_free(buf)
        self.libs.ggml_base.ggml_free(self.ctx)


def chunked_token_logprobs(
    libs: _ffi.Libraries,
    hidden: np.ndarray,
    lm_head: LmHead,
    labels: np.ndarray,
    weights: np.ndarray | None = None,
    *,
    chunk_rows: int | None = None,
    logit_scale: float = 1.0,
    softcap: float = 0.0,
    lora: LoRADelta | None = None,
) -> np.ndarray:
    """Per-token logprobs of the realized tokens, one chunk of rows at a time.

    Never allocates ``[n_tokens, n_vocab]``. The peak transient is one chunk of logits, and it is
    the caller's ``chunk_rows`` that sets it.

    Args:
        libs: The loaded native libraries.
        hidden: ``[n_tokens, n_embd]`` F32 post-final-norm hidden states, one row per token, as
            :func:`sequence_logprobs` gathers them.
        lm_head: The output projection, from :func:`load_lm_head`.
        labels: ``[n_tokens]`` int32 — the token each position is being scored against.
        weights: ``[n_tokens]`` F32 mask. A 0 zeroes that token's logprob without computing it into
            the answer, which is how prompt tokens and pads are excluded. Defaults to all ones.
        chunk_rows: Rows per chunk. Defaults to :func:`choose_chunk_rows`.
        logit_scale: Multiplies the logits before the softmax. 1.0 is off.
        softcap: ``softcap * tanh(u / softcap)`` on the logits. 0.0 is off.
        lora: An adapter targeting the output projection, or None to project with the base weight
            alone. Passing None when the attached adapter *does* target ``output`` silently scores
            the wrong model — :func:`sequence_logprobs` works this out for you.

    Returns:
        ``[n_tokens]`` F32: ``weights[i] * logp(labels[i])``. Masked-out tokens are exactly 0.

    Raises:
        ValueError: If the shapes disagree with each other or with the lm_head.
        RuntimeError: If the graph fails to compute.
    """
    hidden = np.ascontiguousarray(hidden, dtype=np.float32)
    labels = np.ascontiguousarray(labels, dtype=np.int32)

    n_tokens = hidden.shape[0]
    if hidden.ndim != 2 or hidden.shape[1] != lm_head.n_embd:
        raise ValueError(f"hidden must be [n_tokens, {lm_head.n_embd}], got {tuple(hidden.shape)}")
    if labels.shape != (n_tokens,):
        raise ValueError(f"labels must be [{n_tokens}], got {tuple(labels.shape)}")

    if weights is None:
        weights = np.ones(n_tokens, dtype=np.float32)
    weights = np.ascontiguousarray(weights, dtype=np.float32)
    if weights.shape != (n_tokens,):
        raise ValueError(f"weights must be [{n_tokens}], got {tuple(weights.shape)}")

    if n_tokens == 0:
        return np.zeros(0, dtype=np.float32)

    rows = choose_chunk_rows(n_tokens, lm_head.n_vocab, override=chunk_rows)

    backend = libs.ggml.ggml_backend_init_by_type(GGML_BACKEND_DEVICE_TYPE_CPU, None)
    if not backend:
        raise RuntimeError("no CPU backend: ggml_backend_init_by_type returned NULL")

    out = np.empty(n_tokens, dtype=np.float32)
    try:
        for start in range(0, n_tokens, rows):
            stop = min(start + rows, n_tokens)
            out[start:stop] = _one_chunk(
                libs,
                backend,
                hidden[start:stop],
                lm_head,
                labels[start:stop],
                weights[start:stop],
                logit_scale=logit_scale,
                softcap=softcap,
                lora=lora,
            )
    finally:
        libs.ggml_base.ggml_backend_free(backend)

    return out


def _one_chunk(  # noqa: PLR0913 - the graph's inputs; grouping them would only hide them
    libs: _ffi.Libraries,
    backend: int,
    hidden: np.ndarray,
    lm_head: LmHead,
    labels: np.ndarray,
    weights: np.ndarray,
    *,
    logit_scale: float,
    softcap: float,
    lora: LoRADelta | None,
) -> np.ndarray:
    """Project one chunk of hidden states and score it. Returns ``[chunk]`` F32 logprobs.

    The last chunk is short rather than padded: ggml is happy to build the graph at whatever
    ``ne[1]`` the chunk actually has, and a ragged final chunk costs one extra graph build — while
    padding would cost a branch in every consumer to strip the pad rows back out.
    """
    ggml = libs.ggml_base
    n_rows = hidden.shape[0]

    arena = _Arena(libs, backend)
    try:
        # The lm_head: a tensor whose bytes are the mmap's bytes. No copy, no dequantization.
        w = ggml.ggml_new_tensor_2d(arena.ctx, lm_head.ggml_type, lm_head.n_embd, lm_head.n_vocab)
        ggml.ggml_set_name(w, b"lm_head")

        w_bytes = lm_head.data.nbytes
        w_ptr = lm_head.data.ctypes.data
        w_buf = libs.ggml_base.ggml_backend_cpu_buffer_from_ptr(w_ptr, w_bytes)
        if not w_buf:
            raise RuntimeError("ggml_backend_cpu_buffer_from_ptr failed for the lm_head")
        arena.buffers.append(w_buf)
        _check(ggml.ggml_backend_tensor_alloc(w_buf, w, w_ptr), "alloc lm_head")

        # The inputs. These do get a real buffer -- they are small, and they are ours.
        h = ggml.ggml_new_tensor_2d(arena.ctx, GGML_TYPE_F32, lm_head.n_embd, n_rows)
        lab = ggml.ggml_new_tensor_1d(arena.ctx, GGML_TYPE_I32, n_rows)
        wgt = ggml.ggml_new_tensor_1d(arena.ctx, GGML_TYPE_F32, n_rows)

        a = b = None
        if lora is not None:
            # ne is the reverse of numpy's shape: a numpy (r, n_embd) array is ggml ne=[n_embd, r],
            # because ne[0] is the contiguous dimension and numpy's last axis is. Getting this
            # backwards builds a graph of the right *shape* out of transposed data, which computes
            # cleanly and returns nonsense.
            a = ggml.ggml_new_tensor_2d(arena.ctx, GGML_TYPE_F32, lora.a.shape[1], lora.a.shape[0])
            b = ggml.ggml_new_tensor_2d(arena.ctx, GGML_TYPE_F32, lora.b.shape[1], lora.b.shape[0])

        # Build the ops BEFORE allocating. `ggml_backend_alloc_ctx_tensors` gives memory to every
        # tensor in the context that has not got any -- and an op's result tensor is created by the
        # call that builds the op. Allocate first and the intermediates, logits included, are still
        # NULL when the graph runs. It does not fail; it writes through a null pointer.
        #
        # logits = W @ h -- [n_vocab, n_rows]. The only tensor here that scales with the vocab, and
        # it lives exactly as long as this graph.
        logits = ggml.ggml_mul_mat(arena.ctx, w, h)

        if lora is not None:
            # ...plus the adapter's rank-r correction: scale * B @ (A @ h). Two matmuls through an
            # r-wide bottleneck, so this costs nothing next to the projection itself.
            delta = ggml.ggml_mul_mat(arena.ctx, b, ggml.ggml_mul_mat(arena.ctx, a, h))
            logits = ggml.ggml_add(
                arena.ctx, logits, ggml.ggml_scale(arena.ctx, delta, ctypes.c_float(lora.scale))
            )

        # ce_sparse gives w_i * (logsumexp - z[label]) = -w_i * logp_i, per token, unreduced.
        loss = ggml.ggml_cross_entropy_loss_sparse(
            arena.ctx, logits, lab, wgt, ctypes.c_float(logit_scale), ctypes.c_float(softcap)
        )
        logp = ggml.ggml_scale(arena.ctx, loss, ctypes.c_float(-1.0))
        ggml.ggml_set_name(logp, b"logp")

        gf = ggml.ggml_new_graph(arena.ctx)
        ggml.ggml_build_forward_expand(gf, logp)

        # Now everything exists, so now it can all get memory. W is skipped: it already has a
        # buffer, the one wrapping the mmap.
        buf = ggml.ggml_backend_alloc_ctx_tensors(arena.ctx, backend)
        if not buf:
            raise RuntimeError("ggml_backend_alloc_ctx_tensors failed for the chunk graph")
        arena.buffers.append(buf)

        # ...and only now can the inputs be filled: before the allocation they had nowhere to go.
        _upload(libs, h, np.ascontiguousarray(hidden))
        _upload(libs, lab, np.ascontiguousarray(labels))
        _upload(libs, wgt, np.ascontiguousarray(weights))
        if lora is not None:
            _upload(libs, a, np.ascontiguousarray(lora.a, dtype=np.float32))
            _upload(libs, b, np.ascontiguousarray(lora.b, dtype=np.float32))

        _check(libs.ggml_base.ggml_backend_graph_compute(backend, gf), "compute the chunk graph")

        out = np.empty(n_rows, dtype=np.float32)
        libs.ggml_base.ggml_backend_tensor_get(
            logp, out.ctypes.data_as(ctypes.c_void_p), 0, out.nbytes
        )
        return out
    finally:
        arena.close()


def _upload(libs: _ffi.Libraries, tensor: int, array: np.ndarray) -> None:
    libs.ggml_base.ggml_backend_tensor_set(
        tensor, array.ctypes.data_as(ctypes.c_void_p), 0, array.nbytes
    )


def _check(status: int, what: str) -> None:
    if status != GGML_STATUS_SUCCESS:
        raise RuntimeError(f"ggml failed to {what}: status {status}")


# ---------------------------------------------------------------------------------------------
# The high-level pass: decode, gather the hidden states, score them.
# ---------------------------------------------------------------------------------------------


def hidden_states(
    libs: _ffi.Libraries,
    ctx: int,
    tokens: list[int],
    positions: list[int] | None = None,
    seq_ids: list[int] | None = None,
) -> np.ndarray:
    """Decode a batch and return its post-final-norm hidden states, ``[n_tokens, n_embd]`` F32.

    Turns embeddings output on for the duration, which is what makes llama.cpp stop before the
    lm_head. The flag is restored afterwards: a context is a shared thing, and a caller who decodes
    again expecting logits should get logits.

    The KV cache is cleared first. These are absolute-position scores of a fixed sequence, not a
    continuation of whatever the context was last doing.

    Args:
        libs: The loaded native libraries.
        ctx: The ``llama_context``.
        tokens: The batch's tokens.
        positions: Position of each token. Defaults to ``0..n-1``.
        seq_ids: Sequence each token belongs to. Defaults to all-zero.

    Returns:
        ``[n_tokens, n_embd]`` F32, one row per input token, in batch order.

    Raises:
        RuntimeError: If the decode fails.
    """
    n = len(tokens)
    if n == 0:
        return np.zeros((0, 0), dtype=np.float32)

    positions = list(range(n)) if positions is None else positions
    seq_ids = [0] * n if seq_ids is None else seq_ids

    n_embd = libs.llama.llama_model_n_embd(libs.llama.llama_get_model(ctx))

    libs.llama.llama_memory_clear(libs.llama.llama_get_memory(ctx), True)
    libs.llama.llama_set_embeddings(ctx, True)
    try:
        tok = (_ffi.llama_token * n)(*tokens)
        pos = (_ffi.llama_pos * n)(*positions)
        n_seq_id = (ctypes.c_int32 * n)(*([1] * n))
        seq_arrays = [(_ffi.llama_seq_id * 1)(s) for s in seq_ids]
        seq_id = (ctypes.POINTER(_ffi.llama_seq_id) * n)(*seq_arrays)

        # Every position, not just the last: each token is being scored on its own.
        want_output = (ctypes.c_int8 * n)(*([1] * n))

        batch = _ffi.llama_batch(
            n_tokens=n,
            token=tok,
            embd=None,
            pos=pos,
            n_seq_id=n_seq_id,
            seq_id=seq_id,
            logits=want_output,
        )

        status = libs.llama.llama_decode(ctx, batch)
        if status != 0:
            raise RuntimeError(f"llama_decode failed with status {status} in a logprob pass")

        out = np.empty((n, n_embd), dtype=np.float32)
        for i in range(n):
            row = libs.llama.llama_get_embeddings_ith(ctx, i)
            if not row:
                raise RuntimeError(f"no hidden state at position {i}: embeddings output is off?")
            out[i] = np.ctypeslib.as_array(row, shape=(n_embd,))
    finally:
        libs.llama.llama_set_embeddings(ctx, False)

    return out


def sequence_logprobs(  # noqa: PLR0913 - a batch's five parallel arrays, as everywhere else
    libs: _ffi.Libraries,
    ctx: int,
    lm_head: LmHead,
    tokens: list[int],
    targets: list[int],
    weights: list[float],
    seq_ids: list[int] | None = None,
    positions: list[int] | None = None,
    *,
    chunk_rows: int | None = None,
    logit_scale: float = 1.0,
    softcap: float = 0.0,
    lora: LoRADelta | None = None,
) -> np.ndarray:
    """Score a batch: one decode, then the chunked projection.

    The batch shape is :class:`~learning_llamas.train.loop.Batch`'s, so a batch built for training
    can be scored with no translation — which is the point, since DPO's reference pass and GRPO's
    old/ref passes score exactly the batches they are about to train on.

    Args:
        libs: The loaded native libraries.
        ctx: The ``llama_context``. Whether an adapter is attached is the caller's choice, and it
            is what makes this a policy pass or a reference pass.
        lm_head: The output projection, from :func:`load_lm_head`.
        tokens: The batch's input tokens.
        targets: What each position is scored against, i.e. ``tokens[i + 1]``.
        weights: Per-token mask. 0 excludes a position from the answer.
        seq_ids: Sequence each token belongs to. Defaults to all-zero.
        positions: Position of each token. Defaults to ``0..n-1``.
        chunk_rows: Rows per chunk; see :func:`choose_chunk_rows`.
        logit_scale: Multiplies the logits before the softmax. 1.0 is off.
        softcap: ``softcap * tanh(u / softcap)`` on the logits. 0.0 is off.
        lora: An adapter targeting the output projection; None to score the base weight alone.

    Returns:
        ``[n_tokens]`` F32: ``weights[i] * logp(targets[i])``, zero where masked.
    """
    hidden = hidden_states(libs, ctx, tokens, positions, seq_ids)

    return chunked_token_logprobs(
        libs,
        hidden,
        lm_head,
        np.asarray(targets, dtype=np.int32),
        np.asarray(weights, dtype=np.float32),
        chunk_rows=chunk_rows,
        logit_scale=logit_scale,
        softcap=softcap,
        lora=lora,
    )
