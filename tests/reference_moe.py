"""A float64 numpy Mixtral-shaped MoE: forward, LoRA backward — the oracle S1-25..28 never had.

The MoE kernels are checked op-by-op, and ``test_moe.py`` proves a MoE model *trains* — loss
falls, expert adapters move. The audit's point: none of that says the **composed** gradient is
right. A router gradient wrong by a factor, a mis-normalized top-k weight, or an expert LoRA
gradient scattered to the wrong expert still trains, still converges, somewhere slightly else.

Semantics twinned from ``build_moe_ffn`` for LLM_ARCH_LLAMA (``llama-graph.cpp``; the flags at
``models/llama.cpp:203``), because those are what a Mixtral-shaped GGUF runs:

1. router logits = h @ Wg^T; probs = **softmax over all experts** (gating_op SOFTMAX);
2. top-k by prob (``argsort_top_k``) — the *indices* are I32 and carry no gradient, the *weights*
   stay on the gradient path;
3. selected weights renormalized by their sum, clamped at F16's smallest normal
   (``6.103515625e-5``) — the clamp never binds for a softmax's top-k sum, so its VJP is identity
   here;
4. per-expert SiLU FFN with the LoRA delta injected **per expert slice** of the 3D A/B stacks
   (``build_lora_mm_id``);
5. output = sum over the k experts of weight_k * down_k (weighting after the FFN — llama4's
   weight-before-FFN path is a different architecture).

The attention half is identical to the dense fixture and reuses :mod:`tests.reference_llama`'s
primitives. Like that file, nothing here is transcribed from the graph: the backward is derived by
hand and audited against a finite difference of this file's own forward
(``test_moe_gradients.py::test_the_moe_reference_backward_is_itself_correct``).
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
    rope,
    rope_back,
    silu,
    silu_back,
    softmax,
)

WEIGHT_SUM_CLAMP = 6.103515625e-5  # build_moe_ffn's F16-min clamp on the top-k weight sum


@dataclasses.dataclass(frozen=True)
class MoEHParams:
    """Read from the GGUF, plus the two keys that make a llama a Mixtral."""

    n_layer: int
    n_embd: int
    n_head: int
    n_head_kv: int
    n_vocab: int
    n_expert: int
    n_expert_used: int
    rms_eps: float
    rope_freq_base: float

    @property
    def head_dim(self) -> int:
        return self.n_embd // self.n_head

    @property
    def n_rep(self) -> int:
        return self.n_head // self.n_head_kv


def load_model(path: str | pathlib.Path) -> tuple[dict[str, np.ndarray], MoEHParams]:
    """Read the F32 MoE GGUF into float64. Expert stacks arrive as ``(n_expert, n_out, n_in)``."""
    reader = gguf.GGUFReader(str(path), "r")

    def kv(key: str) -> object:
        field = reader.get_field(key)
        if field is None:
            raise ValueError(f"{path} has no {key}")
        return field.contents()

    arch = str(kv("general.architecture"))
    hp = MoEHParams(
        n_layer=int(kv(f"{arch}.block_count")),
        n_embd=int(kv(f"{arch}.embedding_length")),
        n_head=int(kv(f"{arch}.attention.head_count")),
        n_head_kv=int(kv(f"{arch}.attention.head_count_kv")),
        n_vocab=int(kv(f"{arch}.vocab_size")),
        n_expert=int(kv(f"{arch}.expert_count")),
        n_expert_used=int(kv(f"{arch}.expert_used_count")),
        rms_eps=float(kv(f"{arch}.attention.layer_norm_rms_epsilon")),
        rope_freq_base=float(kv(f"{arch}.rope.freq_base")),
    )

    tensors: dict[str, np.ndarray] = {}
    for t in reader.tensors:
        if t.tensor_type != gguf.GGMLQuantizationType.F32:
            raise ValueError(f"{t.name} is {t.tensor_type.name}, not F32")
        tensors[t.name] = np.array(t.data, dtype=np.float64)

    return tensors, hp


def _expert_linear(
    x: np.ndarray, w_e: np.ndarray, lora: Lora | None, e: int, scale: float
) -> tuple[np.ndarray, np.ndarray | None]:
    """One expert's slice of a 3D stack: ``y = x @ W_e.T + scale * (x @ A_e.T) @ B_e.T``."""
    if lora is None:
        return x @ w_e.T, None
    z = x @ lora.a[e].T
    return x @ w_e.T + scale * (z @ lora.b[e].T), z


def _expert_linear_back(
    dy: np.ndarray,
    x: np.ndarray,
    w_e: np.ndarray,
    lora: Lora | None,
    z: np.ndarray | None,
    e: int,
    scale: float,
    grads: dict[str, np.ndarray],
    name: str,
) -> np.ndarray:
    """VJP of :func:`_expert_linear`; dA/dB accumulate into the 3D grad's expert slice."""
    dx = dy @ w_e
    if lora is None:
        return dx

    assert z is not None
    if f"{name}.lora_b" not in grads:
        grads[f"{name}.lora_b"] = np.zeros_like(lora.b)
        grads[f"{name}.lora_a"] = np.zeros_like(lora.a)
    grads[f"{name}.lora_b"][e] += scale * (dy.T @ z)
    dz = scale * (dy @ lora.b[e])
    grads[f"{name}.lora_a"][e] += dz.T @ x
    return dx + dz @ lora.a[e]


def moe_ffn(
    h: np.ndarray,
    tensors: dict[str, np.ndarray],
    hp: MoEHParams,
    loras: dict[str, Lora],
    scale: float,
    prefix: str,
) -> tuple[np.ndarray, dict[str, object]]:
    """The MoE block: router -> top-k -> renormalize -> per-expert SiLU FFN -> weighted sum."""
    k = hp.n_expert_used

    r_logits = h @ tensors[prefix + "ffn_gate_inp.weight"].T  # (T, E); router is never a target
    probs = softmax(r_logits)

    sel = np.argsort(-probs, axis=-1)[:, :k]  # (T, k), descending by prob
    rows = np.arange(len(h))[:, None]
    q = probs[rows, sel]  # (T, k)
    s = np.maximum(q.sum(axis=-1, keepdims=True), WEIGHT_SUM_CLAMP)
    w = q / s  # (T, k)

    cache: dict[str, object] = {"probs": probs, "sel": sel, "q": q, "s": s, "w": w, "h": h}
    y = np.zeros_like(h)
    expert_cache: list[dict[str, object]] = []

    for slot in range(k):
        per_slot: dict[str, object] = {}
        for e in range(hp.n_expert):
            idx = np.nonzero(sel[:, slot] == e)[0]
            if len(idx) == 0:
                continue
            he = h[idx]
            gate, zg = _expert_linear(
                he,
                tensors[prefix + "ffn_gate_exps.weight"][e],
                loras.get(prefix + "ffn_gate_exps.weight"),
                e,
                scale,
            )
            up, zu = _expert_linear(
                he,
                tensors[prefix + "ffn_up_exps.weight"][e],
                loras.get(prefix + "ffn_up_exps.weight"),
                e,
                scale,
            )
            act = silu(gate) * up
            down, zd = _expert_linear(
                act,
                tensors[prefix + "ffn_down_exps.weight"][e],
                loras.get(prefix + "ffn_down_exps.weight"),
                e,
                scale,
            )
            y[idx] += w[idx, slot, None] * down
            per_slot[e] = {
                "idx": idx,
                "gate": gate,
                "up": up,
                "act": act,
                "down": down,
                "zg": zg,
                "zu": zu,
                "zd": zd,
            }
        expert_cache.append(per_slot)

    cache["experts"] = expert_cache
    return y, cache


def moe_ffn_back(
    dy: np.ndarray,
    tensors: dict[str, np.ndarray],
    hp: MoEHParams,
    loras: dict[str, Lora],
    scale: float,
    prefix: str,
    cache: dict[str, object],
    grads: dict[str, np.ndarray],
    detach_router: bool = False,
) -> np.ndarray:
    """VJP of :func:`moe_ffn`. Returns dh; expert-LoRA grads land in their expert's slice.

    ``detach_router=True`` drops the gradient through the routing weights (treats ``w`` as a
    constant) — a DIAGNOSTIC mode, kept because it is how the 2026-07-15 audit proved what the
    fork's backward was actually computing.
    """
    probs, sel = cache["probs"], cache["sel"]
    q, s, w, h = cache["q"], cache["s"], cache["w"], cache["h"]
    k = hp.n_expert_used
    rows = np.arange(len(dy))[:, None]

    dh = np.zeros_like(dy)
    dw = np.zeros_like(w)  # (T, k)

    for slot in range(k):
        for e, ec in cache["experts"][slot].items():
            idx = ec["idx"]
            # y[idx] += w * down: both factors carry gradient.
            dw[idx, slot] = np.sum(dy[idx] * ec["down"], axis=-1)
            d_down = w[idx, slot, None] * dy[idx]

            d_act = _expert_linear_back(
                d_down,
                ec["act"],
                tensors[prefix + "ffn_down_exps.weight"][e],
                loras.get(prefix + "ffn_down_exps.weight"),
                ec["zd"],
                e,
                scale,
                grads,
                prefix + "ffn_down_exps.weight",
            )
            d_gate = d_act * ec["up"] * silu_back(ec["gate"])
            d_up = d_act * silu(ec["gate"])

            dh[idx] += _expert_linear_back(
                d_gate,
                h[idx],
                tensors[prefix + "ffn_gate_exps.weight"][e],
                loras.get(prefix + "ffn_gate_exps.weight"),
                ec["zg"],
                e,
                scale,
                grads,
                prefix + "ffn_gate_exps.weight",
            )
            dh[idx] += _expert_linear_back(
                d_up,
                h[idx],
                tensors[prefix + "ffn_up_exps.weight"][e],
                loras.get(prefix + "ffn_up_exps.weight"),
                ec["zu"],
                e,
                scale,
                grads,
                prefix + "ffn_up_exps.weight",
            )

    if detach_router:
        return dh

    # w = q / s with s = clamp(sum(q)) — the clamp never binds (a softmax top-k sum is far above
    # F16-min), so ds/dq_j = 1 and dq = dw/s - <dw, q> / s^2.
    dq = dw / s - np.sum(dw * q, axis=-1, keepdims=True) / (s**2)

    # get_rows scatter: only the selected experts' probs received gradient directly...
    dprobs = np.zeros_like(probs)
    np.add.at(dprobs, (rows, sel), dq)

    # ...but the softmax couples all of them, which is how unselected experts' router weights
    # still learn to compete. This is the term a "detach the router" bug would drop.
    d_r_logits = probs * (dprobs - np.sum(dprobs * probs, axis=-1, keepdims=True))

    dh += d_r_logits @ tensors[prefix + "ffn_gate_inp.weight"]
    return dh


def forward(
    tensors: dict[str, np.ndarray],
    hp: MoEHParams,
    loras: dict[str, Lora],
    scale: float,
    tokens: np.ndarray,
) -> tuple[np.ndarray, dict[str, object]]:
    """The full MoE llama forward: dense attention (twinned with reference_llama), MoE FFN."""
    t_len = len(tokens)
    positions = np.arange(t_len)
    d, n_h, n_kv, g = hp.head_dim, hp.n_head, hp.n_head_kv, hp.n_rep

    x = tensors["token_embd.weight"][tokens]
    mask = np.triu(np.full((t_len, t_len), -np.inf), k=1)

    cache: dict[str, object] = {"tokens": tokens, "positions": positions}
    layers: list[dict[str, object]] = []

    for il in range(hp.n_layer):
        p = f"blk.{il}."
        lc: dict[str, object] = {}

        lc["resid_attn"] = x
        h, inv1 = rms_norm(x, tensors[p + "attn_norm.weight"], hp.rms_eps)
        lc["h_attn"], lc["inv1"] = h, inv1

        q, zq = linear(h, tensors[p + "attn_q.weight"], loras.get(p + "attn_q.weight"), scale)
        kk, zk = linear(h, tensors[p + "attn_k.weight"], loras.get(p + "attn_k.weight"), scale)
        v, zv = linear(h, tensors[p + "attn_v.weight"], loras.get(p + "attn_v.weight"), scale)
        lc["zq"], lc["zk"], lc["zv"] = zq, zk, zv

        q = rope(q.reshape(t_len, n_h, d), positions, hp.rope_freq_base)
        kk = rope(kk.reshape(t_len, n_kv, d), positions, hp.rope_freq_base)
        v = v.reshape(t_len, n_kv, d)
        lc["q_rope"], lc["k_rope"], lc["v"] = q, kk, v

        k_rep, v_rep = np.repeat(kk, g, axis=1), np.repeat(v, g, axis=1)
        lc["k_rep"], lc["v_rep"] = k_rep, v_rep

        scores = np.einsum("thd,shd->hts", q, k_rep) / np.sqrt(d) + mask[None, :, :]
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

        fo, moe_cache = moe_ffn(h2, tensors, hp, loras, scale, p)
        lc["moe"] = moe_cache

        x = x + fo
        layers.append(lc)

    cache["layers"] = layers
    cache["x_final"] = x

    xf, inv_f = rms_norm(x, tensors["output_norm.weight"], hp.rms_eps)
    cache["xf"], cache["inv_f"] = xf, inv_f
    return xf @ tensors["output.weight"].T, cache


def backward(
    tensors: dict[str, np.ndarray],
    hp: MoEHParams,
    loras: dict[str, Lora],
    scale: float,
    cache: dict[str, object],
    dlogits: np.ndarray,
    detach_router: bool = False,
) -> dict[str, np.ndarray]:
    """Backpropagate to every LoRA tensor — 2D attention pairs and 3D expert stacks alike."""
    d, n_h, g = hp.head_dim, hp.n_head, hp.n_rep
    grads: dict[str, np.ndarray] = {}

    dxf = dlogits @ tensors["output.weight"]
    dx = rms_norm_back(dxf, cache["x_final"], tensors["output_norm.weight"], cache["inv_f"])

    for il in reversed(range(hp.n_layer)):
        p = f"blk.{il}."
        lc = cache["layers"][il]
        t_len = len(cache["tokens"])

        d_h2 = moe_ffn_back(dx, tensors, hp, loras, scale, p, lc["moe"], grads, detach_router)
        dx = dx + rms_norm_back(d_h2, lc["resid_ffn"], tensors[p + "ffn_norm.weight"], lc["inv2"])

        d_attn = linear_back(
            dx,
            lc["attn"],
            tensors[p + "attn_output.weight"],
            loras.get(p + "attn_output.weight"),
            lc["zo"],
            scale,
            grads,
            p + "attn_output.weight",
        ).reshape(t_len, n_h, d)

        probs = lc["probs"]
        d_probs = np.einsum("thd,shd->hts", d_attn, lc["v_rep"])
        d_v_rep = np.einsum("hts,thd->shd", probs, d_attn)

        d_scores = probs * (d_probs - np.sum(d_probs * probs, axis=-1, keepdims=True)) / np.sqrt(d)
        d_q = np.einsum("hts,shd->thd", d_scores, lc["k_rep"])
        d_k_rep = np.einsum("hts,thd->shd", d_scores, lc["q_rope"])

        d_k = d_k_rep.reshape(t_len, hp.n_head_kv, g, d).sum(axis=2)
        d_v = d_v_rep.reshape(t_len, hp.n_head_kv, g, d).sum(axis=2)

        positions = cache["positions"]
        d_q = rope_back(d_q, positions, hp.rope_freq_base)
        d_k = rope_back(d_k, positions, hp.rope_freq_base)

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
