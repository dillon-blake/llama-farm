"""A float64 numpy Mamba-1: forward, LoRA backward -- the oracle the SSM path never had.

The SSM backward kernels (``SSM_CONV_BACK``, ``SSM_SCAN_BACK``) are each finite-difference-checked
op-by-op, and ``test_ssm_training.py``'s e2e test proves a Mamba model *trains* -- loss falls. The
2026-07-15 audit's point, the same one it made about MoE: none of that says the **composed**
gradient is right. A conv backward off by its window alignment, a scan backward that drops the
``dA`` path of ``ddt``, a gate gradient on the wrong branch -- each still trains, still converges,
somewhere slightly and silently else.

So this is the tight oracle, in the shape of ``reference_moe.py``: the same Mamba-1 architecture,
the same weights, the same A/B init, in **float64**, with an analytic backward derived from the
maths and audited against a finite difference of *this file's own forward*
(``test_ssm_training.py::test_the_mamba_reference_backward_is_itself_correct``).

Nothing here is transcribed from ``ggml_compute_forward_ssm_scan`` or ``mamba-base.cpp``. The
selective-scan recurrence is written from the Mamba paper (S6, Annex D) and the block structure
from the architecture; the backward is the reverse-mode AD of that recurrence, derived by hand.
That it lands on the same reverse recurrence the kernel comment describes is not copying -- there is
one correct VJP of a linear recurrence -- but the two were arrived at independently, which is the
whole point of an oracle.

The five things that must be twinned with llama.cpp's Mamba-1, and are
--------------------------------------------------------------------
1. **The expansion factor.** ``d_inner == 2 * n_embd`` -- llama.cpp supports no other
   (``models/mamba.cpp:43``), and the ``ssm_in`` projection produces ``2*d_inner`` = ``[x | z]``.
2. **The causal conv.** A depthwise conv1d over time with ``d_conv-1`` zeros of left padding, per
   channel, then a bias, then SiLU. Not a flipped kernel: ``ggml_ssm_conv`` is a correlation
   (``out[t] = sum_k sx[t+k]*c[k]``), and getting the window direction wrong is a shift no loss
   curve would flag.
3. **The selective scan.** ``dt`` passes through **softplus** before use; the decay is
   ``exp(dt_softplus * A)`` with A **per state** (Mamba-1); the state recurrence is
   ``s_t = s_{t-1}*dA + B*(x*dt_softplus)`` and ``y = <s_t, C>``. A is frozen (negative, the decay),
   so no dA gradient is produced -- but ``ddt`` still chains through the dA path, which is the term
   an "``ddt`` is just the softplus of the Bx path" shortcut would silently drop.
4. **The D skip and the gate.** ``y = y_scan + D*x_conv`` (D over the post-conv SiLU activation, not
   the raw input), then ``y_gated = silu(z) * y`` -- the gate SiLU is on ``z``, the *other* split of
   ssm_in, and ``ggml_swiglu_split(z, y)`` computes ``silu(z)*y`` (``vec.cpp``: ``silu(x)*g``).
5. **eps** is read from the GGUF, not typed here, so the reference cannot drift from its fixture.

Everything is float64. The fixture is F32 and ggml computes in F32, so a residual disagreement of
order 1e-6 relative is expected and is exactly the gap the tolerances are sized against.
"""

from __future__ import annotations

import dataclasses
import pathlib

import gguf
import numpy as np

from .reference_llama import (
    Lora,
    linear,
    linear_back,
    rms_norm,
    rms_norm_back,
    silu,
    silu_back,
)


@dataclasses.dataclass(frozen=True)
class MambaHParams:
    """Read from the GGUF, plus the SSM dimensions that a dense arch does not carry."""

    n_layer: int
    n_embd: int
    d_conv: int
    d_inner: int
    d_state: int
    dt_rank: int
    n_vocab: int
    rms_eps: float


def load_model(path: str | pathlib.Path) -> tuple[dict[str, np.ndarray], MambaHParams]:
    """Read the F32 Mamba GGUF into float64.

    The frozen SSM tensors arrive in GGUF ``ne`` order reversed by ``GGUFReader``:
    ``ssm_a`` as ``(d_inner, d_state)`` = ``A[c, s]``; ``ssm_conv1d.weight`` as
    ``(d_inner, d_conv)`` = ``conv_w[c, k]``; the biases and ``ssm_d`` as ``(d_inner,)``.
    """
    reader = gguf.GGUFReader(str(path), "r")

    def kv(key: str) -> object:
        field = reader.get_field(key)
        if field is None:
            raise ValueError(f"{path} has no {key}")
        return field.contents()

    arch = str(kv("general.architecture"))
    if arch != "mamba":
        raise ValueError(f"{path} is {arch!r}, not a mamba GGUF")

    hp = MambaHParams(
        n_layer=int(kv(f"{arch}.block_count")),
        n_embd=int(kv(f"{arch}.embedding_length")),
        d_conv=int(kv(f"{arch}.ssm.conv_kernel")),
        d_inner=int(kv(f"{arch}.ssm.inner_size")),
        d_state=int(kv(f"{arch}.ssm.state_size")),
        dt_rank=int(kv(f"{arch}.ssm.time_step_rank")),
        n_vocab=int(kv(f"{arch}.vocab_size")),
        rms_eps=float(kv(f"{arch}.attention.layer_norm_rms_epsilon")),
    )

    tensors: dict[str, np.ndarray] = {}
    for t in reader.tensors:
        if t.tensor_type != gguf.GGMLQuantizationType.F32:
            raise ValueError(f"{t.name} is {t.tensor_type.name}, not F32")
        tensors[t.name] = np.array(t.data, dtype=np.float64)

    return tensors, hp


def softplus(x: np.ndarray) -> np.ndarray:
    """``log(1 + exp(x))``, computed stably as ``logaddexp(0, x)`` (ggml's ``softplus_f32``)."""
    return np.logaddexp(0.0, x)


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


# ---------------------------------------------------------------------------
# The SSM-specific pieces. Written from the architecture; the VJPs derived by hand.
# ---------------------------------------------------------------------------


def conv1d_causal(
    x_in: np.ndarray, conv_w: np.ndarray, conv_bias: np.ndarray
) -> np.ndarray:
    """Depthwise causal conv1d + bias.

    ``out[t, c] = bias[c] + sum_{k<K} x_pad[t+k, c] * conv_w[c, k]`` with ``x_pad`` = ``x_in``
    left-padded by ``K-1`` zeros in time. A correlation, matching ``ggml_ssm_conv`` -- the
    convolution's own state (the padding) is a recurrent-cache constant and is zero for a fresh
    single-sequence batch, which is the truncation-at-ubatch boundary this fixture trains at.

    Args:
        x_in: ``(T, d_inner)``.
        conv_w: ``(d_inner, d_conv)``.
        conv_bias: ``(d_inner,)``.
    """
    t_len, _ = x_in.shape
    k_conv = conv_w.shape[1]
    x_pad = np.concatenate([np.zeros((k_conv - 1, x_in.shape[1]), dtype=x_in.dtype), x_in], axis=0)
    out = np.zeros_like(x_in)
    for k in range(k_conv):
        out += x_pad[k : k + t_len, :] * conv_w[:, k][None, :]
    return out + conv_bias[None, :]


def conv1d_causal_back(d_out: np.ndarray, conv_w: np.ndarray) -> np.ndarray:
    """VJP of :func:`conv1d_causal` w.r.t. ``x_in`` (the weight and bias are frozen).

    Each input column receives every window that touched it: ``d_x_pad[j] = sum_k d_out[j-k]*w[k]``,
    then drop the ``K-1`` padding rows (they are the frozen conv-state slot).
    """
    t_len, d_inner = d_out.shape
    k_conv = conv_w.shape[1]
    d_x_pad = np.zeros((t_len + k_conv - 1, d_inner), dtype=d_out.dtype)
    for k in range(k_conv):
        d_x_pad[k : k + t_len, :] += d_out * conv_w[:, k][None, :]
    return d_x_pad[k_conv - 1 :, :]


def selective_scan(
    x_conv: np.ndarray, dt: np.ndarray, b: np.ndarray, c: np.ndarray, a: np.ndarray
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    """Mamba-1's selective scan (S6). Returns ``(y, cache)`` for the backward.

    Per channel ``ch`` and state ``st``, over tokens ``t`` (initial state zero)::

        dt_sp[t,ch] = softplus(dt[t,ch])
        dA[t,st,ch] = exp(dt_sp[t,ch] * A[ch,st])
        s_t[st,ch]  = s_{t-1}[st,ch]*dA[t,st,ch] + B[t,st]*(x_conv[t,ch]*dt_sp[t,ch])
        y[t,ch]     = sum_st s_t[st,ch] * C[t,st]

    Args:
        x_conv: ``(T, d_inner)`` -- the post-conv, post-SiLU activation.
        dt: ``(T, d_inner)`` -- pre-softplus time step (ssm_dt projection + bias).
        b: ``(T, d_state)``.
        c: ``(T, d_state)``.
        a: ``(d_inner, d_state)`` -- the per-state decay, frozen and negative.
    """
    t_len, d_inner = x_conv.shape
    d_state = a.shape[1]
    a_t = a.T  # (d_state, d_inner): A[ch, st] -> a_t[st, ch]

    dt_sp = softplus(dt)  # (T, d_inner)
    y = np.zeros((t_len, d_inner), dtype=x_conv.dtype)
    states = np.zeros((t_len, d_state, d_inner), dtype=x_conv.dtype)
    dA = np.zeros((t_len, d_state, d_inner), dtype=x_conv.dtype)

    s = np.zeros((d_state, d_inner), dtype=x_conv.dtype)
    for t in range(t_len):
        dA[t] = np.exp(dt_sp[t][None, :] * a_t)  # (d_state, d_inner)
        bx = b[t][:, None] * (x_conv[t] * dt_sp[t])[None, :]  # (d_state, d_inner)
        s = s * dA[t] + bx
        states[t] = s
        y[t] = np.sum(s * c[t][:, None], axis=0)  # (d_inner,)

    return y, {"dt_raw": dt, "dt_sp": dt_sp, "states": states, "dA": dA}


def selective_scan_back(
    dy: np.ndarray,
    cache: dict[str, np.ndarray],
    x_conv: np.ndarray,
    b: np.ndarray,
    c: np.ndarray,
    a: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """VJP of :func:`selective_scan`: returns ``(d_x_conv, d_dt, d_B, d_C)``.

    A reverse recurrence carrying ``ds`` = the state gradient handed back from later tokens. ``A``
    is frozen (no ``dA`` output), but ``ddt`` still chains through the ``dA`` path -- that is the
    ``dot_A`` term below, and it needs the *previous* recomputed state ``s_{t-1}``.
    """
    t_len, d_inner = x_conv.shape
    d_state = a.shape[1]
    a_t = a.T  # (d_state, d_inner)

    dt_sp = cache["dt_sp"]
    states = cache["states"]
    dA = cache["dA"]

    d_x_conv = np.zeros((t_len, d_inner), dtype=x_conv.dtype)
    d_dt_sp = np.zeros((t_len, d_inner), dtype=x_conv.dtype)
    d_b = np.zeros((t_len, d_state), dtype=x_conv.dtype)
    d_c = np.zeros((t_len, d_state), dtype=x_conv.dtype)

    ds = np.zeros((d_state, d_inner), dtype=x_conv.dtype)
    zero_state = np.zeros((d_state, d_inner), dtype=x_conv.dtype)

    for t in reversed(range(t_len)):
        s_t = states[t]
        s_prev = states[t - 1] if t > 0 else zero_state

        # The state's total gradient: what later tokens sent back, plus what this token's own
        # output y[t] = <s_t, C[t]> wants of it.
        d_state_total = ds + c[t][:, None] * dy[t][None, :]  # (d_state, d_inner)

        d_c[t] = np.sum(s_t * dy[t][None, :], axis=1)  # (d_state,)

        xdt = x_conv[t] * dt_sp[t]  # (d_inner,)
        d_b[t] = np.sum(d_state_total * xdt[None, :], axis=1)  # (d_state,)

        dot_b = np.sum(d_state_total * b[t][:, None], axis=0)  # (d_inner,)
        d_x_conv[t] = dt_sp[t] * dot_b

        dot_a = np.sum(d_state_total * s_prev * a_t * dA[t], axis=0)  # (d_inner,)
        d_dt_sp[t] = x_conv[t] * dot_b + dot_a

        ds = d_state_total * dA[t]

    # dt_sp = softplus(dt), and softplus'(dt) is the sigmoid -- so the raw dt gradient chains here.
    d_dt = d_dt_sp * _sigmoid(cache["dt_raw"])
    return d_x_conv, d_dt, d_b, d_c


# ---------------------------------------------------------------------------
# One Mamba block
# ---------------------------------------------------------------------------


def _mamba_block(
    resid: np.ndarray,
    tensors: dict[str, np.ndarray],
    hp: MambaHParams,
    loras: dict[str, Lora],
    scale: float,
    p: str,
) -> tuple[np.ndarray, dict[str, object]]:
    """A single Mamba-1 mixer + residual. Returns ``(block_out, cache)``."""
    d_inner, d_state, dt_rank = hp.d_inner, hp.d_state, hp.dt_rank
    lc: dict[str, object] = {"resid": resid}

    h, inv = rms_norm(resid, tensors[p + "attn_norm.weight"], hp.rms_eps)
    lc["h"], lc["inv"] = h, inv

    xz, z_in = linear(h, tensors[p + "ssm_in.weight"], loras.get(p + "ssm_in.weight"), scale)
    lc["z_in"] = z_in
    x_in = xz[:, :d_inner]
    z = xz[:, d_inner:]
    lc["z"] = z

    x_conv_pre = conv1d_causal(
        x_in, tensors[p + "ssm_conv1d.weight"], tensors[p + "ssm_conv1d.bias"]
    )
    lc["x_conv_pre"] = x_conv_pre
    x_conv = silu(x_conv_pre)
    lc["x_conv"] = x_conv

    x_db, z_x = linear(x_conv, tensors[p + "ssm_x.weight"], loras.get(p + "ssm_x.weight"), scale)
    lc["z_x"] = z_x
    dt_slice = x_db[:, :dt_rank]
    b = x_db[:, dt_rank : dt_rank + d_state]
    c = x_db[:, dt_rank + d_state : dt_rank + 2 * d_state]
    lc["dt_slice"], lc["b"], lc["c"] = dt_slice, b, c

    dt_proj, z_dt = linear(
        dt_slice, tensors[p + "ssm_dt.weight"], loras.get(p + "ssm_dt.weight"), scale
    )
    lc["z_dt"] = z_dt
    dt = dt_proj + tensors[p + "ssm_dt.bias"][None, :]
    lc["dt"] = dt

    a = tensors[p + "ssm_a"]
    y_scan, scan_cache = selective_scan(x_conv, dt, b, c, a)
    lc["scan"] = scan_cache

    y = y_scan + tensors[p + "ssm_d"][None, :] * x_conv  # D skip over the conv activation
    y_gated = silu(z) * y
    lc["y"] = y

    cur, z_out = linear(
        y_gated, tensors[p + "ssm_out.weight"], loras.get(p + "ssm_out.weight"), scale
    )
    lc["y_gated"], lc["z_out"] = y_gated, z_out

    return resid + cur, lc


def _mamba_block_back(
    dx: np.ndarray,
    tensors: dict[str, np.ndarray],
    hp: MambaHParams,
    loras: dict[str, Lora],
    scale: float,
    p: str,
    lc: dict[str, object],
    grads: dict[str, np.ndarray],
) -> np.ndarray:
    """VJP of :func:`_mamba_block`. Accumulates LoRA grads; returns the grad flowing to resid."""
    d_state, dt_rank = hp.d_state, hp.dt_rank

    # block_out = resid + cur -> the skip carries dx straight through, the mixer gets it too.
    d_resid = dx

    d_y_gated = linear_back(
        dx, lc["y_gated"], tensors[p + "ssm_out.weight"], loras.get(p + "ssm_out.weight"),
        lc["z_out"], scale, grads, p + "ssm_out.weight",
    )

    # y_gated = silu(z) * y
    z, y = lc["z"], lc["y"]
    d_z = d_y_gated * y * silu_back(z)
    d_y = d_y_gated * silu(z)

    # y = y_scan + D * x_conv
    d_y_scan = d_y
    x_conv = lc["x_conv"]
    d_x_conv = d_y * tensors[p + "ssm_d"][None, :]

    # scan
    a = tensors[p + "ssm_a"]
    d_x_conv_scan, d_dt, d_b, d_c = selective_scan_back(
        d_y_scan, lc["scan"], x_conv, lc["b"], lc["c"], a
    )
    d_x_conv = d_x_conv + d_x_conv_scan

    # dt = ssm_dt(dt_slice) + bias  (bias frozen)
    d_dt_slice = linear_back(
        d_dt, lc["dt_slice"], tensors[p + "ssm_dt.weight"], loras.get(p + "ssm_dt.weight"),
        lc["z_dt"], scale, grads, p + "ssm_dt.weight",
    )

    # x_db = [dt_slice | B | C] = ssm_x(x_conv)
    d_x_db = np.zeros((dx.shape[0], dt_rank + 2 * d_state), dtype=np.float64)
    d_x_db[:, :dt_rank] = d_dt_slice
    d_x_db[:, dt_rank : dt_rank + d_state] = d_b
    d_x_db[:, dt_rank + d_state : dt_rank + 2 * d_state] = d_c

    d_x_conv_x = linear_back(
        d_x_db, x_conv, tensors[p + "ssm_x.weight"], loras.get(p + "ssm_x.weight"),
        lc["z_x"], scale, grads, p + "ssm_x.weight",
    )
    d_x_conv = d_x_conv + d_x_conv_x

    # x_conv = silu(x_conv_pre); conv weight & bias frozen
    d_x_conv_pre = d_x_conv * silu_back(lc["x_conv_pre"])
    d_x_in = conv1d_causal_back(d_x_conv_pre, tensors[p + "ssm_conv1d.weight"])

    # xz = ssm_in(h); xz = [x_in | z]
    d_xz = np.concatenate([d_x_in, d_z], axis=1)
    d_h = linear_back(
        d_xz, lc["h"], tensors[p + "ssm_in.weight"], loras.get(p + "ssm_in.weight"),
        lc["z_in"], scale, grads, p + "ssm_in.weight",
    )

    d_resid = d_resid + rms_norm_back(d_h, lc["resid"], tensors[p + "attn_norm.weight"], lc["inv"])
    return d_resid


# ---------------------------------------------------------------------------
# The model
# ---------------------------------------------------------------------------


def forward(
    tensors: dict[str, np.ndarray],
    hp: MambaHParams,
    loras: dict[str, Lora],
    scale: float,
    tokens: np.ndarray,
) -> tuple[np.ndarray, dict[str, object]]:
    """The full Mamba-1 forward: token embed, N mixer blocks, final norm, lm head."""
    x = tensors["token_embd.weight"][tokens]  # (T, E)
    cache: dict[str, object] = {"tokens": tokens}
    layers: list[dict[str, object]] = []

    for il in range(hp.n_layer):
        x, lc = _mamba_block(x, tensors, hp, loras, scale, f"blk.{il}.")
        layers.append(lc)

    cache["layers"] = layers
    cache["x_final"] = x

    xf, inv_f = rms_norm(x, tensors["output_norm.weight"], hp.rms_eps)
    cache["xf"], cache["inv_f"] = xf, inv_f
    return xf @ tensors["output.weight"].T, cache


def backward(
    tensors: dict[str, np.ndarray],
    hp: MambaHParams,
    loras: dict[str, Lora],
    scale: float,
    cache: dict[str, object],
    dlogits: np.ndarray,
) -> dict[str, np.ndarray]:
    """Backpropagate to every LoRA tensor -- the four Mamba projections, per layer."""
    grads: dict[str, np.ndarray] = {}

    dxf = dlogits @ tensors["output.weight"]
    dx = rms_norm_back(dxf, cache["x_final"], tensors["output_norm.weight"], cache["inv_f"])

    for il in reversed(range(hp.n_layer)):
        dx = _mamba_block_back(
            dx, tensors, hp, loras, scale, f"blk.{il}.", cache["layers"][il], grads
        )

    return grads
