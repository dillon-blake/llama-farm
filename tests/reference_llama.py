"""A float64 numpy llama: forward, LoRA backward, and AdamW. The oracle the gate is built on.

Why this exists, when S1-12 asked for a PEFT reference
------------------------------------------------------
The ticket's plan is to record a PEFT/transformers loss curve once and compare against it with
**tolerance bands per step window**. That is a real check — an independent implementation, written
by other people, that we cannot have accidentally agreed with. It is kept (see
``tests/convergence/``), and it catches the one thing this file cannot: a shared *misunderstanding*
of what LoRA SFT is supposed to be.

But it is a weak *numerical* oracle, and the ticket concedes as much when it says per-step exact
match is impossible. The bands have to be wide enough to absorb kernel-numerics drift between
llama.cpp and torch — and a band wide enough to absorb that is wide enough to hide a real gradient
bug. A gradient that is wrong by 10% still produces a loss curve that falls, inside any honest
band, and converges somewhere slightly different, and nothing complains. That is precisely the
class of bug this project keeps finding (the weighted-one-hot CE backward in S1-03; the transposed
`grad` in the MoE kernels; `mean_abs_asymm` itself).

So the *tight* oracle is here: the same architecture, the same weights, the same A/B init, in
**float64**, with an analytic backward derived from the maths rather than transcribed from the
graph. It is compared per step, at ~1e-5, and it can say *which tensor* disagrees rather than
merely that the curve drifted.

Deliberately not a translation of the graph
-------------------------------------------
Following ``tests/reference_grpo.py``: *"a reference that shares the graph's structure shares its
bugs."* Nothing here is transcribed from ``csrc/farm_train.cpp`` or ``llama-graph.cpp``. The
forward is written from the architecture, the backward is derived by hand from the forward, and the
backward is then checked against a finite difference **of this file's own forward** (see
``tests/test_convergence.py::test_the_reference_backward_is_itself_correct``) — because a
hand-derived VJP is exactly as likely to be wrong as the one it is auditing, and an oracle nobody
audited is not an oracle.

The four things that must be twinned, and are
---------------------------------------------
Any of these silently makes the reference a *different model*, which shows up as a plausible-looking
drift that invites you to widen the tolerance until it passes:

1. **RoPE pairing.** GGUF's llama arch is ``LLAMA_ROPE_TYPE_NORM`` — ggml mode 0 — which rotates
   **interleaved adjacent** pairs ``(x[2k], x[2k+1])``. HF's ``rotate_half`` rotates **split-half**
   pairs ``(x[k], x[k + d/2])``. They are *not* the same function; `convert_hf_to_gguf.py` permutes
   the Q and K weight rows at conversion time to make them agree. This file implements the ggml
   convention because it is reading a GGUF. (The HF twin in ``tests/convergence/`` applies the
   inverse permutation — see its README, because that is where getting it wrong is invisible.)
2. **The LoRA scale.** llama.cpp computes ``scale = alpha ? user_scale * alpha / rank : user_scale``
   (``llama-adapter.h:52-57``) — so ``alpha == 0`` does not mean scale zero, it means the
   ``alpha/rank`` factor is *dropped*. PEFT's is ``alpha / r``. :func:`lora_scale` implements
   llama.cpp's, including the trapdoor.
3. **The loss normalization.** The shim folds ``1 / sum(w)`` into the per-token weights host-side
   (``farm_train.cpp``'s ``upload_masked_ce``), so the loss is a weighted *mean* over the tokens
   that carry weight — not a mean over all tokens, and not a sum. A reference that summed instead
   would be off by a factor of ``n_valid`` and the loss curve would still fall.
4. **eps and theta.** ``f_norm_rms_eps`` and ``rope.freq_base`` are read from the **GGUF itself**,
   not typed in here, so there is exactly one source of truth and the reference cannot drift from
   the fixture it is checking.

Everything is float64. The fixture is F32 on disk and ggml computes in F32, so a residual
disagreement of order 1e-6 relative is expected and is the *point* — that is the gap the
tolerances are sized against, and it is four orders of magnitude below any real gradient bug.
"""

from __future__ import annotations

import dataclasses
import pathlib

import gguf
import numpy as np

# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------


@dataclasses.dataclass(frozen=True)
class RefHParams:
    """Hyperparameters, read from the GGUF rather than assumed.

    Attributes:
        n_layer: Transformer blocks.
        n_embd: Residual stream width.
        n_head: Query heads.
        n_head_kv: Key/value heads. Fewer than ``n_head`` means grouped-query attention.
        n_vocab: Vocabulary size.
        rms_eps: RMSNorm epsilon, *inside* the square root.
        rope_freq_base: RoPE's theta base.
    """

    n_layer: int
    n_embd: int
    n_head: int
    n_head_kv: int
    n_vocab: int
    rms_eps: float
    rope_freq_base: float

    @property
    def head_dim(self) -> int:
        return self.n_embd // self.n_head

    @property
    def n_rep(self) -> int:
        """How many query heads share one key/value head."""
        return self.n_head // self.n_head_kv


def load_model(path: str | pathlib.Path) -> tuple[dict[str, np.ndarray], RefHParams]:
    """Read an F32 base GGUF into float64 numpy arrays.

    A GGUF weight of ``ne = [n_in, n_out]`` arrives from ``GGUFReader`` as a numpy array of shape
    ``(n_out, n_in)`` — the ordinary ``nn.Linear`` convention, so ``y = x @ W.T``.

    Args:
        path: The base model GGUF. Must be F32: a quantized base cannot be dequantized here
            without reimplementing ggml's quantizer, and the reference is deliberately the *smooth*
            model that the backward differentiates anyway (see ``tests/test_p0_gradient.py``).

    Returns:
        A ``(tensors, hparams)`` pair; every tensor is float64.

    Raises:
        ValueError: If any 2-D tensor is not F32.
    """
    reader = gguf.GGUFReader(str(path), "r")

    def kv(key: str) -> object:
        field = reader.get_field(key)
        if field is None:
            raise ValueError(f"{path} has no {key}")
        return field.contents()

    arch = str(kv("general.architecture"))
    hp = RefHParams(
        n_layer=int(kv(f"{arch}.block_count")),
        n_embd=int(kv(f"{arch}.embedding_length")),
        n_head=int(kv(f"{arch}.attention.head_count")),
        n_head_kv=int(kv(f"{arch}.attention.head_count_kv")),
        n_vocab=int(kv(f"{arch}.vocab_size")),
        rms_eps=float(kv(f"{arch}.attention.layer_norm_rms_epsilon")),
        rope_freq_base=float(kv(f"{arch}.rope.freq_base")),
    )

    tensors: dict[str, np.ndarray] = {}
    for t in reader.tensors:
        if t.tensor_type != gguf.GGMLQuantizationType.F32:
            raise ValueError(
                f"{t.name} is {t.tensor_type.name}, not F32. The float64 reference needs the F32 "
                "base; the quantized fixtures are compared against this same reference with their "
                "own documented band (see tests/convergence/README.md)."
            )
        tensors[t.name] = np.array(t.data, dtype=np.float64)

    return tensors, hp


def lora_scale(alpha: float, rank: int, user_scale: float = 1.0) -> float:
    """llama.cpp's effective LoRA scale, trapdoor included.

    ``scale = alpha ? user_scale * alpha / rank : user_scale`` (``llama-adapter.h:52-57``). An
    ``alpha`` of zero does **not** scale the adapter to nothing — it silently drops the
    ``alpha/rank`` factor. PEFT's convention is ``alpha / r`` with no such branch.
    """
    return user_scale * alpha / rank if alpha else user_scale


# ---------------------------------------------------------------------------
# The pieces. Written from the architecture, not from the graph.
# ---------------------------------------------------------------------------


def rms_norm(x: np.ndarray, weight: np.ndarray, eps: float) -> tuple[np.ndarray, np.ndarray]:
    """``y = x / sqrt(mean(x^2) + eps) * weight``, row-wise.

    The epsilon is **inside** the square root, added to the mean of squares
    (``ggml-cpu/ops.cpp``'s ``rms_norm``: ``scale = 1/sqrtf(mean + eps)``). Putting it outside —
    ``sqrt(mean) + eps`` — is a different function, and on a well-conditioned tensor it is a
    *nearly* identical one, which is what makes the mistake survive.

    Returns:
        A ``(y, inv_rms)`` pair; ``inv_rms`` is kept for the backward.
    """
    inv_rms = 1.0 / np.sqrt(np.mean(x**2, axis=-1, keepdims=True) + eps)
    return x * inv_rms * weight, inv_rms


def rms_norm_back(
    dy: np.ndarray, x: np.ndarray, weight: np.ndarray, inv_rms: np.ndarray
) -> np.ndarray:
    """VJP of :func:`rms_norm` with respect to ``x`` (the weight is frozen, so it takes none).

    With ``s = inv_rms`` and ``g = dy * weight``::

        dx = s * (g - (s^2 / n) * x * <g, x>)

    The second term is the one that is easy to drop: it is the gradient flowing through the *norm*
    itself, and without it the gradient is still plausible, still descends, and is wrong.
    """
    n = x.shape[-1]
    g = dy * weight
    dot = np.sum(g * x, axis=-1, keepdims=True)
    return inv_rms * (g - (inv_rms**2 / n) * x * dot)


def rope(x: np.ndarray, positions: np.ndarray, freq_base: float) -> np.ndarray:
    """RoPE in ggml's mode-0 (``LLAMA_ROPE_TYPE_NORM``) convention: interleaved adjacent pairs.

    For pair ``k`` of a head of width ``d``::

        theta_k   = pos * freq_base ** (-2k/d)
        out[2k]   = x[2k]*cos - x[2k+1]*sin
        out[2k+1] = x[2k]*sin + x[2k+1]*cos

    HF rotates ``(x[k], x[k + d/2])`` instead. Both are rotations, both preserve norms, and
    swapping them produces a model that trains perfectly well to a *different* place — so this is
    the single most important line in the file to get right.

    Args:
        x: ``(T, n_head, head_dim)``.
        positions: ``(T,)`` position ids.
        freq_base: ``rope.freq_base`` from the GGUF.
    """
    d = x.shape[-1]
    k = np.arange(d // 2, dtype=np.float64)
    theta = positions[:, None].astype(np.float64) * (freq_base ** (-2.0 * k / d))  # (T, d/2)
    cos, sin = np.cos(theta)[:, None, :], np.sin(theta)[:, None, :]

    x0, x1 = x[..., 0::2], x[..., 1::2]
    out = np.empty_like(x)
    out[..., 0::2] = x0 * cos - x1 * sin
    out[..., 1::2] = x0 * sin + x1 * cos
    return out


def rope_back(dy: np.ndarray, positions: np.ndarray, freq_base: float) -> np.ndarray:
    """VJP of :func:`rope`. A rotation's transpose is the rotation by ``-theta``."""
    d = dy.shape[-1]
    k = np.arange(d // 2, dtype=np.float64)
    theta = positions[:, None].astype(np.float64) * (freq_base ** (-2.0 * k / d))
    cos, sin = np.cos(theta)[:, None, :], np.sin(theta)[:, None, :]

    g0, g1 = dy[..., 0::2], dy[..., 1::2]
    dx = np.empty_like(dy)
    dx[..., 0::2] = g0 * cos + g1 * sin
    dx[..., 1::2] = -g0 * sin + g1 * cos
    return dx


def silu(x: np.ndarray) -> np.ndarray:
    """``x * sigmoid(x)``."""
    return x / (1.0 + np.exp(-x))


def silu_back(x: np.ndarray) -> np.ndarray:
    """``silu'(x) = sigmoid(x) * (1 + x*(1 - sigmoid(x)))``.

    It has a zero at ``x ~= -1.2784``, which is why ``test-backend-ops grad -o SILU`` is
    ill-conditioned — see docs/dev/backward-coverage.md. The derivative itself is unremarkable.
    """
    s = 1.0 / (1.0 + np.exp(-x))
    return s * (1.0 + x * (1.0 - s))


def softmax(x: np.ndarray) -> np.ndarray:
    """Row-wise softmax over the last axis, shifted by the max for stability."""
    z = x - np.max(x, axis=-1, keepdims=True)
    e = np.exp(z)
    return e / np.sum(e, axis=-1, keepdims=True)


# ---------------------------------------------------------------------------
# LoRA
# ---------------------------------------------------------------------------


@dataclasses.dataclass
class Lora:
    """One target's LoRA pair, in the shapes llama.cpp's loader wants.

    Attributes:
        a: ``(r, n_in)``. The GGUF writes it as ``ne = [n_in, r]``.
        b: ``(n_out, r)``. Zero at init, which is what makes a fresh adapter an exact no-op.
    """

    a: np.ndarray
    b: np.ndarray


def linear(
    x: np.ndarray, w: np.ndarray, lora: Lora | None, scale: float
) -> tuple[np.ndarray, np.ndarray | None]:
    """``y = x @ W.T + scale * (x @ A.T) @ B.T``.

    Returns:
        A ``(y, z)`` pair, where ``z = x @ A.T`` is the rank-``r`` bottleneck kept for the
        backward (``None`` when the target has no adapter).
    """
    y = x @ w.T
    if lora is None:
        return y, None
    z = x @ lora.a.T
    return y + scale * (z @ lora.b.T), z


def linear_back(
    dy: np.ndarray,
    x: np.ndarray,
    w: np.ndarray,
    lora: Lora | None,
    z: np.ndarray | None,
    scale: float,
    grads: dict[str, np.ndarray],
    name: str,
) -> np.ndarray:
    """VJP of :func:`linear`. Accumulates ``dA``/``dB`` into ``grads``; returns ``dx``.

    The base weight is frozen and takes no gradient — but it still *carries* one, because ``dx``
    must flow back through ``W`` to reach whatever is upstream. Dropping that term would isolate
    each LoRA from every other and the model would still train.
    """
    dx = dy @ w
    if lora is None:
        return dx

    assert z is not None
    grads[f"{name}.lora_b"] = grads.get(f"{name}.lora_b", 0.0) + scale * (dy.T @ z)
    dz = scale * (dy @ lora.b)
    grads[f"{name}.lora_a"] = grads.get(f"{name}.lora_a", 0.0) + (dz.T @ x)
    return dx + dz @ lora.a


# ---------------------------------------------------------------------------
# The model
# ---------------------------------------------------------------------------


def forward(
    tensors: dict[str, np.ndarray],
    hp: RefHParams,
    loras: dict[str, Lora],
    scale: float,
    tokens: np.ndarray,
    positions: np.ndarray | None = None,
    seq_ids: np.ndarray | None = None,
) -> tuple[np.ndarray, dict[str, object]]:
    """Run the llama forward in float64.

    Args:
        tensors: The base weights, from :func:`load_model`.
        hp: The hyperparameters.
        loras: LoRA pairs keyed by base tensor name (e.g. ``blk.0.attn_q.weight``). A target with
            no entry is simply not adapted.
        scale: The effective LoRA scale, from :func:`lora_scale`.
        tokens: ``(T,)`` input token ids.
        positions: ``(T,)`` position ids. Defaults to ``0..T-1``.
        seq_ids: ``(T,)`` sequence ids for a packed batch (S1-07). When given, attention is causal
            *within* a sequence and forbidden across sequences — the block mask the packer's
            ``seq_ids`` induce in llama.cpp. ``None`` is one sequence, plain causal.

    Returns:
        A ``(logits, cache)`` pair. ``logits`` is ``(T, n_vocab)``; ``cache`` holds every
        intermediate the backward needs.
    """
    t_len = len(tokens)
    if positions is None:
        positions = np.arange(t_len)

    d, n_h, n_kv, g = hp.head_dim, hp.n_head, hp.n_head_kv, hp.n_rep

    x = tensors["token_embd.weight"][tokens]  # (T, E)

    # A causal mask, additive: 0 where a query may attend, -inf where it may not. For a packed
    # batch, "may" additionally requires being the same sequence.
    mask = np.triu(np.full((t_len, t_len), -np.inf), k=1)
    if seq_ids is not None:
        ids = np.asarray(seq_ids)
        mask = np.where(ids[:, None] == ids[None, :], mask, -np.inf)

    cache: dict[str, object] = {"tokens": tokens, "positions": positions, "mask": mask}
    layers: list[dict[str, object]] = []

    for il in range(hp.n_layer):
        p = f"blk.{il}."
        lc: dict[str, object] = {}

        lc["resid_attn"] = x
        h, inv1 = rms_norm(x, tensors[p + "attn_norm.weight"], hp.rms_eps)
        lc["h_attn"], lc["inv1"] = h, inv1

        q, zq = linear(h, tensors[p + "attn_q.weight"], loras.get(p + "attn_q.weight"), scale)
        k, zk = linear(h, tensors[p + "attn_k.weight"], loras.get(p + "attn_k.weight"), scale)
        v, zv = linear(h, tensors[p + "attn_v.weight"], loras.get(p + "attn_v.weight"), scale)
        lc["zq"], lc["zk"], lc["zv"] = zq, zk, zv

        q = q.reshape(t_len, n_h, d)
        k = k.reshape(t_len, n_kv, d)
        v = v.reshape(t_len, n_kv, d)

        q_r = rope(q, positions, hp.rope_freq_base)
        k_r = rope(k, positions, hp.rope_freq_base)
        lc["q_rope"], lc["k_rope"], lc["v"] = q_r, k_r, v

        # Grouped-query attention: kv head `h // g` serves query heads [h*g, (h+1)*g).
        k_rep = np.repeat(k_r, g, axis=1)  # (T, n_h, d)
        v_rep = np.repeat(v, g, axis=1)
        lc["k_rep"], lc["v_rep"] = k_rep, v_rep

        scores = np.einsum("thd,shd->hts", q_r, k_rep) / np.sqrt(d)
        scores = scores + mask[None, :, :]
        probs = softmax(scores)
        lc["probs"] = probs

        attn = np.einsum("hts,shd->thd", probs, v_rep).reshape(t_len, n_h * d)
        lc["attn"] = attn

        ao, zo = linear(
            attn, tensors[p + "attn_output.weight"], loras.get(p + "attn_output.weight"), scale
        )
        lc["zo"] = zo

        x = x + ao
        lc["resid_ffn"] = x

        h2, inv2 = rms_norm(x, tensors[p + "ffn_norm.weight"], hp.rms_eps)
        lc["h_ffn"], lc["inv2"] = h2, inv2

        gate, zg = linear(
            h2, tensors[p + "ffn_gate.weight"], loras.get(p + "ffn_gate.weight"), scale
        )
        up, zu = linear(h2, tensors[p + "ffn_up.weight"], loras.get(p + "ffn_up.weight"), scale)
        lc["gate"], lc["up"], lc["zg"], lc["zu"] = gate, up, zg, zu

        act = silu(gate) * up
        lc["act"] = act

        fo, zd = linear(
            act, tensors[p + "ffn_down.weight"], loras.get(p + "ffn_down.weight"), scale
        )
        lc["zd"] = zd

        x = x + fo
        layers.append(lc)

    cache["layers"] = layers
    cache["x_final"] = x

    xf, inv_f = rms_norm(x, tensors["output_norm.weight"], hp.rms_eps)
    cache["xf"], cache["inv_f"] = xf, inv_f

    logits = xf @ tensors["output.weight"].T
    return logits, cache


def loss_from_logits(
    logits: np.ndarray, targets: np.ndarray, weights: np.ndarray
) -> tuple[float, np.ndarray]:
    """The masked-mean cross-entropy, and its gradient with respect to the logits.

    ``L = sum_i w_i' * (logsumexp(x_i) - x_i[target_i])`` with ``w' = w / sum(w)``.

    The normalization is by the weight actually *carried*, not by the token count: a batch that is
    90% prompt must not have its gradient scaled down tenfold relative to one that is 10% prompt.
    A batch with no unmasked token has zero loss and zero gradient, which is the honest answer.

    Returns:
        A ``(loss, dlogits)`` pair.
    """
    total = float(np.sum(weights))
    w = weights / total if total > 0 else np.zeros_like(weights)

    probs = softmax(logits)
    logp = np.log(probs[np.arange(len(targets)), targets])
    loss = float(-np.sum(w * logp))

    dlogits = probs.copy()
    dlogits[np.arange(len(targets)), targets] -= 1.0
    return loss, dlogits * w[:, None]


def backward(
    tensors: dict[str, np.ndarray],
    hp: RefHParams,
    loras: dict[str, Lora],
    scale: float,
    cache: dict[str, object],
    dlogits: np.ndarray,
) -> dict[str, np.ndarray]:
    """Backpropagate to the LoRA tensors, and to nothing else.

    The base weights are frozen — but every one of them still *transports* a gradient, and the
    embedding table is where the chain finally stops.

    Returns:
        ``dA``/``dB`` keyed by ``<base name>.lora_a`` / ``<base name>.lora_b``.
    """
    d, n_h, g = hp.head_dim, hp.n_head, hp.n_rep
    grads: dict[str, np.ndarray] = {}

    dxf = dlogits @ tensors["output.weight"]
    dx = rms_norm_back(
        dxf,
        cache["x_final"],
        tensors["output_norm.weight"],
        cache["inv_f"],  # type: ignore[arg-type]
    )

    for il in reversed(range(hp.n_layer)):
        p = f"blk.{il}."
        lc = cache["layers"][il]  # type: ignore[index]
        t_len = len(cache["tokens"])  # type: ignore[arg-type]

        # --- FFN --------------------------------------------------------------------------
        # The residual: the gradient reaching x arrives BOTH through the FFN and around it.
        d_act = linear_back(
            dx,
            lc["act"],
            tensors[p + "ffn_down.weight"],
            loras.get(p + "ffn_down.weight"),
            lc["zd"],
            scale,
            grads,
            p + "ffn_down.weight",
        )

        d_gate = d_act * lc["up"] * silu_back(lc["gate"])
        d_up = d_act * silu(lc["gate"])

        d_h2 = linear_back(
            d_gate,
            lc["h_ffn"],
            tensors[p + "ffn_gate.weight"],
            loras.get(p + "ffn_gate.weight"),
            lc["zg"],
            scale,
            grads,
            p + "ffn_gate.weight",
        )
        d_h2 += linear_back(
            d_up,
            lc["h_ffn"],
            tensors[p + "ffn_up.weight"],
            loras.get(p + "ffn_up.weight"),
            lc["zu"],
            scale,
            grads,
            p + "ffn_up.weight",
        )

        dx = dx + rms_norm_back(d_h2, lc["resid_ffn"], tensors[p + "ffn_norm.weight"], lc["inv2"])

        # --- attention --------------------------------------------------------------------
        d_attn = linear_back(
            dx,
            lc["attn"],
            tensors[p + "attn_output.weight"],
            loras.get(p + "attn_output.weight"),
            lc["zo"],
            scale,
            grads,
            p + "attn_output.weight",
        )
        d_attn = d_attn.reshape(t_len, n_h, d)

        probs = lc["probs"]
        d_probs = np.einsum("thd,shd->hts", d_attn, lc["v_rep"])
        d_v_rep = np.einsum("hts,thd->shd", probs, d_attn)

        # Softmax VJP: dz = p * (dp - <dp, p>). Masked entries have p = 0, so they contribute
        # nothing and need no special-casing — which is the whole reason the mask is additive.
        d_scores = probs * (d_probs - np.sum(d_probs * probs, axis=-1, keepdims=True))
        d_scores = d_scores / np.sqrt(d)

        d_q = np.einsum("hts,shd->thd", d_scores, lc["k_rep"])
        d_k_rep = np.einsum("hts,thd->shd", d_scores, lc["q_rope"])

        # Fold the grouped-query repeat back up: each kv head collects from the g query heads
        # it served. np.repeat's inverse is a sum over the group.
        d_k = d_k_rep.reshape(t_len, hp.n_head_kv, g, d).sum(axis=2)
        d_v = d_v_rep.reshape(t_len, hp.n_head_kv, g, d).sum(axis=2)

        positions = cache["positions"]
        d_q = rope_back(d_q, positions, hp.rope_freq_base)  # type: ignore[arg-type]
        d_k = rope_back(d_k, positions, hp.rope_freq_base)  # type: ignore[arg-type]

        d_h = linear_back(
            d_q.reshape(t_len, -1),
            lc["h_attn"],
            tensors[p + "attn_q.weight"],
            loras.get(p + "attn_q.weight"),
            lc["zq"],
            scale,
            grads,
            p + "attn_q.weight",
        )
        d_h += linear_back(
            d_k.reshape(t_len, -1),
            lc["h_attn"],
            tensors[p + "attn_k.weight"],
            loras.get(p + "attn_k.weight"),
            lc["zk"],
            scale,
            grads,
            p + "attn_k.weight",
        )
        d_h += linear_back(
            d_v.reshape(t_len, -1),
            lc["h_attn"],
            tensors[p + "attn_v.weight"],
            loras.get(p + "attn_v.weight"),
            lc["zv"],
            scale,
            grads,
            p + "attn_v.weight",
        )

        dx = dx + rms_norm_back(d_h, lc["resid_attn"], tensors[p + "attn_norm.weight"], lc["inv1"])

    return grads


# ---------------------------------------------------------------------------
# AdamW — ggml's, not PyTorch's default
# ---------------------------------------------------------------------------


@dataclasses.dataclass
class AdamW:
    """ggml-opt's AdamW, to the letter (``ggml-cpu/ops.cpp``'s ``opt_step_adamw``).

    ::

        m = m*b1 + g*(1 - b1)
        v = v*b2 + g^2*(1 - b2)
        w = w*(1 - lr*wd) - lr * (m/(1 - b1^t)) / (sqrt(v/(1 - b2^t)) + eps)

    Two details that a from-memory implementation gets wrong. The bias-correction counter ``t``
    starts at **1**, not 0 (``ggml-opt.cpp:987-988``, ``powf(beta, opt_ctx->iter)`` with ``iter``
    initialized to 1) — starting at 0 makes the first step divide by zero. And ``eps`` is added to
    the **bias-corrected** ``sqrt(v_hat)``, outside the square root, not to ``v`` inside it.
    """

    lr: float
    betas: tuple[float, float] = (0.9, 0.999)
    eps: float = 1e-8
    weight_decay: float = 0.0
    _t: int = 0
    _m: dict[str, np.ndarray] = dataclasses.field(default_factory=dict)
    _v: dict[str, np.ndarray] = dataclasses.field(default_factory=dict)

    def step(
        self, params: dict[str, np.ndarray], grads: dict[str, np.ndarray], lr: float | None = None
    ) -> None:
        """Update ``params`` in place from ``grads``."""
        self._t += 1
        b1, b2 = self.betas
        alpha = self.lr if lr is None else lr

        b1h = 1.0 / (1.0 - b1**self._t)
        b2h = 1.0 / (1.0 - b2**self._t)
        keep = 1.0 - alpha * self.weight_decay

        for name, w in params.items():
            g = grads[name]
            m = self._m.setdefault(name, np.zeros_like(w))
            v = self._v.setdefault(name, np.zeros_like(w))

            m *= b1
            m += g * (1.0 - b1)
            v *= b2
            v += g * g * (1.0 - b2)

            w *= keep
            w -= alpha * (m * b1h) / (np.sqrt(v * b2h) + self.eps)
