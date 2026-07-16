"""A float64 numpy Mamba-2: forward + LoRA backward -- the oracle for ``n_group > 1`` (B-10).

``reference_mamba.py`` is the Mamba-1 oracle: per-state ``A``, ``head_dim == 1``, and
``n_group == 1``, so its selective scan never routes a head to a group. The ``SSM_SCAN`` backward's
group-index fold (``g = h/(nh/ng)``, summing every head of a group into one ``dB``/``dC`` slab) is
therefore invisible to it, and S1-47 refused ``n_group > 1`` for want of an oracle. This is that
oracle: the Mamba-2 block llama.cpp builds for arch ``mamba2`` with ``ssm.group_count > 1``, in
float64, with an analytic backward derived by hand from the maths -- NOT transcribed from
``ggml_compute_forward_ssm_scan`` or ``mamba2.cpp``.

What Mamba-2 changes from Mamba-1, and this file has to twin exactly
------------------------------------------------------------------
1. **One in-projection.** ``ssm_in`` emits ``[z | xBC | dt]`` in a single matmul; there is no
   ``ssm_x``/``ssm_dt``. The LoRA targets are ``ssm_in`` and ``ssm_out`` only.
2. **The conv covers x, B and C.** The causal conv1d + SiLU runs over the full
   ``d_inner + 2*n_group*d_state`` channels; x, B and C are split off *after* the activation.
3. **Heads and scalar A.** ``x`` is ``head_dim`` x ``n_head``; ``A``, ``D`` and ``dt`` are one value
   per head. The decay is ``exp(dt_softplus * A[h])`` -- the scalar-``A`` branch of the scan.
4. **Grouped B/C.** ``B``/``C`` are ``d_state`` x ``n_group``. Head ``h`` reads group
   ``g = h // (n_head/n_group)`` (a ``np.repeat`` in the forward), so its ``dB``/``dC`` fold back a
   ``np.repeat``'s worth of heads (the sum this oracle exists to check). This is the same shape of
   reasoning as ``reference_llama``'s GQA ``k_rep`` fold.
5. **A grouped gate norm.** After ``silu(z) * y`` an RMSNorm runs per group over ``d_inner/n_group``
   channels (``ssm_norm``), then ``ssm_out``.

Everything is float64; the fixture is F32 and ggml computes in F32, so a residual disagreement of
order 1e-6 relative is expected and is what the tolerances are sized against. ``eps`` is read from
the GGUF so the reference cannot drift from its fixture.
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
from .reference_mamba import conv1d_causal, conv1d_causal_back, softplus


@dataclasses.dataclass(frozen=True)
class Mamba2HParams:
    """Read from the GGUF, plus the SSM dimensions a dense arch does not carry."""

    n_layer: int
    n_embd: int
    d_inner: int
    n_head: int
    n_group: int
    d_state: int
    d_conv: int
    n_vocab: int
    rms_eps: float

    @property
    def head_dim(self) -> int:
        return self.d_inner // self.n_head

    @property
    def conv_dim(self) -> int:
        return self.d_inner + 2 * self.n_group * self.d_state

    @property
    def d_in_proj(self) -> int:
        """``ssm_in`` emits ``[z | xBC | dt]``: ``2*d_inner + 2*n_group*d_state + n_head``."""
        return self.d_inner + self.conv_dim + self.n_head

    @property
    def heads_per_group(self) -> int:
        return self.n_head // self.n_group


def load_model(path: str | pathlib.Path) -> tuple[dict[str, np.ndarray], Mamba2HParams]:
    """Read the F32 Mamba-2 GGUF into float64.

    The frozen per-head ``ssm_a``/``ssm_d`` arrive with a trailing singleton axis (GGUF ``ne``
    ``{1, n_head}``); this squeezes them to ``(n_head,)``.
    """
    reader = gguf.GGUFReader(str(path), "r")

    def kv(key: str) -> object:
        field = reader.get_field(key)
        if field is None:
            raise ValueError(f"{path} has no {key}")
        return field.contents()

    arch = str(kv("general.architecture"))
    if arch != "mamba2":
        raise ValueError(f"{path} is {arch!r}, not a mamba2 GGUF")

    hp = Mamba2HParams(
        n_layer=int(kv(f"{arch}.block_count")),
        n_embd=int(kv(f"{arch}.embedding_length")),
        d_inner=int(kv(f"{arch}.ssm.inner_size")),
        n_head=int(kv(f"{arch}.ssm.time_step_rank")),
        n_group=int(kv(f"{arch}.ssm.group_count")),
        d_state=int(kv(f"{arch}.ssm.state_size")),
        d_conv=int(kv(f"{arch}.ssm.conv_kernel")),
        n_vocab=int(kv(f"{arch}.vocab_size")),
        rms_eps=float(kv(f"{arch}.attention.layer_norm_rms_epsilon")),
    )

    tensors: dict[str, np.ndarray] = {}
    for t in reader.tensors:
        if t.tensor_type != gguf.GGMLQuantizationType.F32:
            raise ValueError(f"{t.name} is {t.tensor_type.name}, not F32")
        arr = np.array(t.data, dtype=np.float64)
        if t.name.endswith(("ssm_a", "ssm_d")):
            arr = arr.reshape(-1)  # {1, n_head} -> (n_head,)
        tensors[t.name] = arr

    return tensors, hp


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


# ---------------------------------------------------------------------------
# The Mamba-2 selective scan: heads, scalar A, grouped B/C. VJP derived by hand.
# ---------------------------------------------------------------------------


def selective_scan2(
    x: np.ndarray,
    dt: np.ndarray,
    b: np.ndarray,
    c: np.ndarray,
    a: np.ndarray,
    n_group: int,
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    """Mamba-2's grouped selective scan. Returns ``(y, cache)``.

    Per head ``h`` (group ``g = h // (n_head/n_group)``), head-dim ``j``, state ``i0``, token ``t``
    (initial state zero)::

        dt_sp[t,h] = softplus(dt[t,h])
        dA[t,h]    = exp(dt_sp[t,h] * A[h])                       scalar per head
        s_t[i0,h,j] = s_{t-1}[i0,h,j]*dA[t,h] + B[t,g,i0]*(x[t,h,j]*dt_sp[t,h])
        y[t,h,j]    = sum_i0 s_t[i0,h,j] * C[t,g,i0]

    Args:
        x: ``(T, n_head, head_dim)`` -- the post-conv, post-SiLU head activations.
        dt: ``(T, n_head)`` -- pre-softplus time step (already includes ``ssm_dt.bias``).
        b: ``(T, n_group, d_state)``.
        c: ``(T, n_group, d_state)``.
        a: ``(n_head,)`` -- the scalar per-head decay, frozen and negative.
        n_group: the number of B/C groups.
    """
    t_len, n_head, head_dim = x.shape
    d_state = b.shape[2]
    hpg = n_head // n_group  # heads per group; g = h // hpg == np.repeat over groups

    dt_sp = softplus(dt)  # (T, n_head)
    y = np.zeros((t_len, n_head, head_dim), dtype=np.float64)
    states = np.zeros((t_len, d_state, n_head, head_dim), dtype=np.float64)
    dA = np.zeros((t_len, n_head), dtype=np.float64)

    s = np.zeros((d_state, n_head, head_dim), dtype=np.float64)
    for t in range(t_len):
        dA[t] = np.exp(dt_sp[t] * a)  # (n_head,)
        b_head = np.repeat(b[t], hpg, axis=0)  # (n_head, d_state): head h -> group h//hpg
        c_head = np.repeat(c[t], hpg, axis=0)  # (n_head, d_state)
        x_dt = x[t] * dt_sp[t][:, None]  # (n_head, head_dim)

        # s[i0,h,j] = s*dA[h] + B_head[h,i0]*x_dt[h,j]
        s = s * dA[t][None, :, None] + b_head.T[:, :, None] * x_dt[None, :, :]
        states[t] = s
        y[t] = np.sum(s * c_head.T[:, :, None], axis=0)  # (n_head, head_dim)

    return y, {"dt_raw": dt, "dt_sp": dt_sp, "states": states, "dA": dA}


def selective_scan2_back(
    dy: np.ndarray,
    cache: dict[str, np.ndarray],
    x: np.ndarray,
    b: np.ndarray,
    c: np.ndarray,
    a: np.ndarray,
    n_group: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """VJP of :func:`selective_scan2`: returns ``(d_x, d_dt, d_B, d_C)``.

    A reverse recurrence carrying ``ds`` = the state gradient handed back from later tokens. ``A``
    is frozen (no ``dA`` output), but ``ddt`` still chains through the ``dA`` path. ``dB``/``dC``
    are the group fold: each head's contribution is summed back into its group's slab -- the reverse
    of the ``np.repeat`` the forward does.
    """
    t_len, n_head, head_dim = x.shape
    d_state = b.shape[2]
    hpg = n_head // n_group

    dt_sp = cache["dt_sp"]
    states = cache["states"]
    dA = cache["dA"]

    d_x = np.zeros((t_len, n_head, head_dim), dtype=np.float64)
    d_dt_sp = np.zeros((t_len, n_head), dtype=np.float64)
    d_b = np.zeros((t_len, n_group, d_state), dtype=np.float64)
    d_c = np.zeros((t_len, n_group, d_state), dtype=np.float64)

    ds = np.zeros((d_state, n_head, head_dim), dtype=np.float64)
    zero_state = np.zeros((d_state, n_head, head_dim), dtype=np.float64)

    def fold(per_head: np.ndarray) -> np.ndarray:
        """Sum a ``(d_state, n_head)`` array back into its ``(n_group, d_state)`` groups."""
        return per_head.reshape(d_state, n_group, hpg).sum(axis=2).T

    for t in reversed(range(t_len)):
        s_t = states[t]
        s_prev = states[t - 1] if t > 0 else zero_state
        b_head = np.repeat(b[t], hpg, axis=0)  # (n_head, d_state)
        c_head = np.repeat(c[t], hpg, axis=0)  # (n_head, d_state)
        x_dt = x[t] * dt_sp[t][:, None]  # (n_head, head_dim)

        # The state's total gradient: later tokens plus this token's own output y[t]=<s_t,C>.
        d_state_total = ds + c_head.T[:, :, None] * dy[t][None, :, :]  # (d_state, n_head, head_dim)

        d_c[t] = fold(np.sum(s_t * dy[t][None, :, :], axis=2))  # sum over head_dim, then group fold
        d_b[t] = fold(np.sum(d_state_total * x_dt[None, :, :], axis=2))

        dot_b = np.sum(d_state_total * b_head.T[:, :, None], axis=0)  # (n_head, head_dim)
        d_x[t] = dt_sp[t][:, None] * dot_b

        # d(dt_sp): the x_dt path (x*dot_b, summed over head_dim) plus the dA path.
        dot_a = a * dA[t] * np.sum(d_state_total * s_prev, axis=(0, 2))  # (n_head,)
        d_dt_sp[t] = np.sum(x[t] * dot_b, axis=1) + dot_a

        ds = d_state_total * dA[t][None, :, None]

    d_dt = d_dt_sp * _sigmoid(cache["dt_raw"])  # softplus'(dt) is the sigmoid
    return d_x, d_dt, d_b, d_c


# ---------------------------------------------------------------------------
# The grouped gate RMSNorm (ssm_norm): per group over d_inner/n_group channels.
# ---------------------------------------------------------------------------


def grouped_rms_norm(
    y: np.ndarray, weight: np.ndarray, n_group: int, eps: float
) -> tuple[np.ndarray, np.ndarray]:
    """RMSNorm over ``d_inner/n_group`` channels, per group. ``weight`` is ``(n_group, gsz)``.

    Returns ``(out, inv_rms)`` with ``out`` flattened back to ``(T, d_inner)`` and ``inv_rms``
    ``(T, n_group, 1)`` kept for the backward.
    """
    t_len, d_inner = y.shape
    gsz = d_inner // n_group
    yg = y.reshape(t_len, n_group, gsz)
    out, inv = rms_norm(yg, weight[None, :, :], eps)  # broadcast weight over tokens
    return out.reshape(t_len, d_inner), inv


def grouped_rms_norm_back(
    d_out: np.ndarray, y: np.ndarray, weight: np.ndarray, inv: np.ndarray, n_group: int
) -> np.ndarray:
    """VJP of :func:`grouped_rms_norm` w.r.t. ``y`` (the weight is frozen)."""
    t_len, d_inner = y.shape
    gsz = d_inner // n_group
    yg = y.reshape(t_len, n_group, gsz)
    dg = d_out.reshape(t_len, n_group, gsz)
    d_yg = rms_norm_back(dg, yg, weight[None, :, :], inv)
    return d_yg.reshape(t_len, d_inner)


# ---------------------------------------------------------------------------
# One Mamba-2 block
# ---------------------------------------------------------------------------


def _mamba2_block(
    resid: np.ndarray,
    tensors: dict[str, np.ndarray],
    hp: Mamba2HParams,
    loras: dict[str, Lora],
    scale: float,
    p: str,
) -> tuple[np.ndarray, dict[str, object]]:
    """A single Mamba-2 mixer + residual. Returns ``(block_out, cache)``."""
    d_inner, d_state, n_head, n_group = hp.d_inner, hp.d_state, hp.n_head, hp.n_group
    head_dim = hp.head_dim
    lc: dict[str, object] = {"resid": resid}

    h, inv = rms_norm(resid, tensors[p + "attn_norm.weight"], hp.rms_eps)
    lc["h"], lc["inv"] = h, inv

    zxbcdt, z_in = linear(h, tensors[p + "ssm_in.weight"], loras.get(p + "ssm_in.weight"), scale)
    lc["z_in"] = z_in
    z = zxbcdt[:, :d_inner]
    xbc = zxbcdt[:, d_inner : d_inner + hp.conv_dim]
    dt_raw = zxbcdt[:, d_inner + hp.conv_dim :]
    lc["z"] = z

    xbc_pre = conv1d_causal(xbc, tensors[p + "ssm_conv1d.weight"], tensors[p + "ssm_conv1d.bias"])
    lc["xbc_pre"] = xbc_pre
    xbc_act = silu(xbc_pre)
    lc["xbc_act"] = xbc_act

    x_flat = xbc_act[:, :d_inner]
    b = xbc_act[:, d_inner : d_inner + n_group * d_state].reshape(-1, n_group, d_state)
    c = xbc_act[:, d_inner + n_group * d_state :].reshape(-1, n_group, d_state)
    x = x_flat.reshape(-1, n_head, head_dim)
    lc["x"], lc["b"], lc["c"] = x, b, c

    dt = dt_raw + tensors[p + "ssm_dt.bias"][None, :]
    lc["dt"] = dt

    a = tensors[p + "ssm_a"]
    y_scan, scan_cache = selective_scan2(x, dt, b, c, a, n_group)
    lc["scan"] = scan_cache

    d_skip = tensors[p + "ssm_d"]  # (n_head,)
    y_heads = y_scan + x * d_skip[None, :, None]  # D skip per head over the conv activation
    y = y_heads.reshape(-1, d_inner)
    lc["y"] = y

    y_gated = silu(z) * y
    lc["y_gated"] = y_gated

    y_norm, norm_inv = grouped_rms_norm(
        y_gated, tensors[p + "ssm_norm.weight"], n_group, hp.rms_eps
    )
    lc["norm_inv"] = norm_inv

    cur, z_out = linear(
        y_norm, tensors[p + "ssm_out.weight"], loras.get(p + "ssm_out.weight"), scale
    )
    lc["y_norm"], lc["z_out"] = y_norm, z_out

    return resid + cur, lc


def _mamba2_block_back(
    dx: np.ndarray,
    tensors: dict[str, np.ndarray],
    hp: Mamba2HParams,
    loras: dict[str, Lora],
    scale: float,
    p: str,
    lc: dict[str, object],
    grads: dict[str, np.ndarray],
) -> np.ndarray:
    """VJP of :func:`_mamba2_block`. Accumulates LoRA grads; returns the grad flowing to resid."""
    d_inner, d_state, n_head, n_group = hp.d_inner, hp.d_state, hp.n_head, hp.n_group
    head_dim = hp.head_dim

    d_resid = dx  # block_out = resid + cur

    d_y_norm = linear_back(
        dx,
        lc["y_norm"],
        tensors[p + "ssm_out.weight"],
        loras.get(p + "ssm_out.weight"),
        lc["z_out"],
        scale,
        grads,
        p + "ssm_out.weight",
    )

    d_y_gated = grouped_rms_norm_back(
        d_y_norm, lc["y_gated"], tensors[p + "ssm_norm.weight"], lc["norm_inv"], n_group
    )

    # y_gated = silu(z) * y
    z, y = lc["z"], lc["y"]
    d_z = d_y_gated * y * silu_back(z)
    d_y = d_y_gated * silu(z)

    # y = y_scan + D * x  (reshaped to heads)
    d_y_heads = d_y.reshape(-1, n_head, head_dim)
    d_skip = tensors[p + "ssm_d"]
    d_y_scan = d_y_heads
    x = lc["x"]
    d_x = d_y_heads * d_skip[None, :, None]  # the D-skip path into x

    d_x_scan, d_dt, d_b, d_c = selective_scan2_back(
        d_y_scan, lc["scan"], x, lc["b"], lc["c"], tensors[p + "ssm_a"], n_group
    )
    d_x = d_x + d_x_scan

    # Reassemble the post-activation conv gradient: [x | B | C].
    d_xbc_act = np.zeros((dx.shape[0], hp.conv_dim), dtype=np.float64)
    d_xbc_act[:, :d_inner] = d_x.reshape(-1, d_inner)
    d_xbc_act[:, d_inner : d_inner + n_group * d_state] = d_b.reshape(-1, n_group * d_state)
    d_xbc_act[:, d_inner + n_group * d_state :] = d_c.reshape(-1, n_group * d_state)

    d_xbc_pre = d_xbc_act * silu_back(lc["xbc_pre"])
    d_xbc = conv1d_causal_back(d_xbc_pre, tensors[p + "ssm_conv1d.weight"])

    # zxbcdt = ssm_in(h); split [z | xBC | dt]. dt's bias is frozen, so d_dt flows straight up.
    d_zxbcdt = np.zeros((dx.shape[0], hp.d_in_proj), dtype=np.float64)
    d_zxbcdt[:, :d_inner] = d_z
    d_zxbcdt[:, d_inner : d_inner + hp.conv_dim] = d_xbc
    d_zxbcdt[:, d_inner + hp.conv_dim :] = d_dt

    d_h = linear_back(
        d_zxbcdt,
        lc["h"],
        tensors[p + "ssm_in.weight"],
        loras.get(p + "ssm_in.weight"),
        lc["z_in"],
        scale,
        grads,
        p + "ssm_in.weight",
    )

    d_resid = d_resid + rms_norm_back(d_h, lc["resid"], tensors[p + "attn_norm.weight"], lc["inv"])
    return d_resid


# ---------------------------------------------------------------------------
# The model
# ---------------------------------------------------------------------------


def forward(
    tensors: dict[str, np.ndarray],
    hp: Mamba2HParams,
    loras: dict[str, Lora],
    scale: float,
    tokens: np.ndarray,
) -> tuple[np.ndarray, dict[str, object]]:
    """The full Mamba-2 forward: token embed, N mixer blocks, final norm, lm head."""
    x = tensors["token_embd.weight"][tokens]  # (T, E)
    cache: dict[str, object] = {"tokens": tokens}
    layers: list[dict[str, object]] = []

    for il in range(hp.n_layer):
        x, lc = _mamba2_block(x, tensors, hp, loras, scale, f"blk.{il}.")
        layers.append(lc)

    cache["layers"] = layers
    cache["x_final"] = x

    xf, inv_f = rms_norm(x, tensors["output_norm.weight"], hp.rms_eps)
    cache["xf"], cache["inv_f"] = xf, inv_f
    return xf @ tensors["output.weight"].T, cache


def backward(
    tensors: dict[str, np.ndarray],
    hp: Mamba2HParams,
    loras: dict[str, Lora],
    scale: float,
    cache: dict[str, object],
    dlogits: np.ndarray,
) -> dict[str, np.ndarray]:
    """Backpropagate to every LoRA tensor -- ``ssm_in`` and ``ssm_out``, per layer."""
    grads: dict[str, np.ndarray] = {}

    dxf = dlogits @ tensors["output.weight"]
    dx = rms_norm_back(dxf, cache["x_final"], tensors["output_norm.weight"], cache["inv_f"])

    for il in reversed(range(hp.n_layer)):
        dx = _mamba2_block_back(
            dx, tensors, hp, loras, scale, f"blk.{il}.", cache["layers"][il], grads
        )

    return grads
