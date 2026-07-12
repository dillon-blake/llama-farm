"""Create, read, and enumerate LoRA adapter GGUFs.

learning-llamas adopts llama.cpp's LoRA adapter GGUF as **both** its adapter format and its
checkpoint format (BLUEPRINT D3): whatever we write loads in stock llama.cpp, llama-server and
ollama with zero conversion. Nothing here needs the native library — gguf-py alone can build a
complete adapter from the base model's metadata.

The zero-B no-op property
-------------------------
A freshly initialized adapter has ``A ~ N(0, sigma)`` and ``B = 0``, which makes the LoRA delta
``scale * B(A @ x)`` **exactly** zero — not approximately, exactly, for any A. So attaching a
step-0 adapter at scale 1.0 must leave the model's logits bit-identical. That is this project's
first end-to-end correctness gate; the executable test lives in S0-06
(``tests/test_adapter_noop.py``), and trained-adapter fidelity against stock ``llama-cli --lora``
is exercised from S1-03 onward.

Three loader rules that are easy to get wrong
---------------------------------------------
All three were read out of the loader at the pinned commit, not assumed.

1. **alpha must be non-zero.** The effective scale is ``user_scale * alpha / rank``, but the
   loader computes it as ``alpha ? user_scale * alpha / rank : user_scale``
   (``src/llama-adapter.h:55``) — so ``alpha == 0`` does not mean "scale 0", it means the
   ``alpha/rank`` factor is **silently dropped**. :func:`create_zero_adapter` rejects it.

2. **Tensor names keep ``.weight``.** The loader strips only the ``.lora_a`` / ``.lora_b``
   suffix and then looks the remainder up with ``model.get_tensor(name)``
   (``src/llama-adapter.cpp:273-285, 330``). So the name is ``blk.0.attn_q.weight.lora_a`` —
   with ``.weight`` — which is also what the stock converter emits
   (``convert_lora_to_gguf.py:525``). Dropping ``.weight`` produces a file that fails to load.

3. **``token_embd.weight`` uses a flipped, A-transposed convention.** Normal targets are checked
   as ``model.ne[0] == a.ne[0] && model.ne[1] == b.ne[1] && a.ne[1] == b.ne[0]``; token_embd is
   checked as ``model.ne[0] == b.ne[1] && model.ne[1] == a.ne[1]``
   (``src/llama-adapter.cpp:356-368``). :func:`create_zero_adapter` writes each convention.

The loader also rejects an adapter whose ``general.architecture`` differs from the base model's
(``src/llama-adapter.cpp:206-211``), so the architecture is copied from the base GGUF.

Provenance: the KV set and tensor-naming scheme mirror ``convert_lora_to_gguf.py`` (llama.cpp,
MIT, commit ``4f37f519722aa3242eecb7649466b4a4a2d6d6da``) as a *format reference*. No code is
copied; see docs/PROVENANCE.md.
"""

from __future__ import annotations

import pathlib
from dataclasses import dataclass

import numpy as np

try:
    import gguf
except ImportError as exc:  # pragma: no cover - environment problem, not a code path
    raise ImportError(
        "learning-llamas uses the gguf-py that ships with the vendored llama.cpp, so that the "
        "file-format code and the pinned llama.cpp commit stay one atomic version (ADR-0001).\n"
        "  install it: pip install -e vendor/llama.cpp/gguf-py"
    ) from exc

# The default LoRA target set (BLUEPRINT D5): attention projections and the FFN.
DEFAULT_PRESET: tuple[str, ...] = (
    "attn_q",
    "attn_k",
    "attn_v",
    "attn_qkv",
    "attn_output",
    "ffn_up",
    "ffn_gate",
    "ffn_down",
)

TOKEN_EMBD = "token_embd"
OUTPUT = "output"


@dataclass(frozen=True)
class LoraTarget:
    """One base-model tensor a LoRA adapter can target.

    Attributes:
        name: The full base tensor name, including ``.weight`` (e.g. ``blk.0.attn_q.weight``).
        n_in: The tensor's ``ne[0]`` — the input dimension.
        n_out: The tensor's ``ne[1]`` — the output dimension.
        dtype: The base tensor's quantization type. Recorded for reporting only: the adapter's
            own A/B tensors are always F32, whatever the base is quantized to.
        is_token_embd: Whether this is ``token_embd.weight``, which the loader validates with
            the flipped shape convention.
    """

    name: str
    n_in: int
    n_out: int
    dtype: gguf.GGMLQuantizationType
    is_token_embd: bool


@dataclass(frozen=True)
class AdapterInfo:
    """What :func:`read_adapter` recovers from an adapter GGUF.

    Attributes:
        architecture: ``general.architecture``. Must equal the base model's or the loader throws.
        alpha: ``adapter.lora.alpha``. Zero here means the alpha/rank scale factor is dropped.
        ranks: Per-target LoRA rank, keyed by base tensor name.
        shapes: Per-target ``(a_ne, b_ne)`` in GGUF ``ne`` order.
    """

    architecture: str
    alpha: float
    ranks: dict[str, int]
    shapes: dict[str, tuple[tuple[int, ...], tuple[int, ...]]]


def _module_of(tensor_name: str) -> str:
    """Return the module component of a GGUF tensor name.

    ``blk.0.attn_q.weight`` → ``attn_q``; ``token_embd.weight`` → ``token_embd``.

    Matching the component exactly, rather than testing ``endswith("ffn_down.weight")``, is what
    keeps MoE expert tensors like ``blk.0.ffn_down_exps.weight`` out of the default preset —
    those are ``build_lora_mm_id`` operands and are out of scope until the MoE tickets.
    """
    parts = tensor_name.split(".")
    return parts[-2] if len(parts) >= 2 else ""


def enumerate_targets(
    base_gguf_path: str | pathlib.Path,
    preset: tuple[str, ...] = DEFAULT_PRESET,
    include_output: bool = False,
    include_token_embd: bool = False,
) -> list[LoraTarget]:
    """List the LoRA-targetable tensors of a base model.

    Pure name-and-shape logic over the base GGUF's metadata — no graph walk. (The graph walk
    that decides what is actually *trainable* is the S1-11 preflight.)

    Args:
        base_gguf_path: Path to the base model GGUF.
        preset: Module names to target, matched against the tensor name's module component.
        include_output: Also target ``output.weight`` (the LM head).
        include_token_embd: Also target ``token_embd.weight`` (the embedding table).

    Returns:
        The matching targets, in the order they appear in the file.
    """
    reader = gguf.GGUFReader(str(base_gguf_path), "r")

    wanted = set(preset)
    if include_output:
        wanted.add(OUTPUT)
    if include_token_embd:
        wanted.add(TOKEN_EMBD)

    targets: list[LoraTarget] = []
    for tensor in reader.tensors:
        if not tensor.name.endswith(".weight"):
            continue
        module = _module_of(tensor.name)
        if module not in wanted:
            continue

        # GGUFReader exposes `shape` in GGUF ne order: ne[0] is the fastest-moving dimension.
        n_in, n_out = int(tensor.shape[0]), int(tensor.shape[1])
        targets.append(
            LoraTarget(
                name=tensor.name,
                n_in=n_in,
                n_out=n_out,
                dtype=tensor.tensor_type,
                is_token_embd=(module == TOKEN_EMBD),
            )
        )

    return targets


def _base_architecture(base_gguf_path: str | pathlib.Path) -> str:
    reader = gguf.GGUFReader(str(base_gguf_path), "r")
    field = reader.get_field(gguf.Keys.General.ARCHITECTURE)
    if field is None:
        raise ValueError(f"{base_gguf_path} has no general.architecture key")
    return str(field.contents())


def create_zero_adapter(
    base_gguf_path: str | pathlib.Path,
    out_path: str | pathlib.Path,
    r: int = 16,
    alpha: float | None = None,
    sigma: float | None = None,
    seed: int = 0,
    preset: tuple[str, ...] = DEFAULT_PRESET,
    include_output: bool = False,
    include_token_embd: bool = False,
) -> list[LoraTarget]:
    """Write a zero-initialized LoRA adapter GGUF for a base model.

    ``A ~ N(0, sigma)`` and ``B = 0``, so the adapter is a provable no-op at step 0 (see the
    module docstring). Both tensors are F32 regardless of the base model's quantization — only
    the adapter trains, and it trains in F32.

    Args:
        base_gguf_path: The base model to build the adapter for. Its architecture is copied into
            the adapter; the loader rejects a mismatch.
        out_path: Where to write the adapter GGUF.
        r: LoRA rank.
        alpha: LoRA alpha. Defaults to ``r``, which makes the effective scale ``alpha/rank ==
            1.0`` at user scale 1.0. **Must be non-zero** — see the module docstring.
        sigma: Standard deviation for A. Defaults to ``1/sqrt(r)``. Any small value is correct
            for the no-op property, since B is zero regardless.
        seed: Seed for A's RNG. The same seed produces a byte-identical file.
        preset: Module names to target.
        include_output: Also adapt ``output.weight``.
        include_token_embd: Also adapt ``token_embd.weight``.

    Returns:
        The targets that were written.

    Raises:
        ValueError: If ``alpha`` is zero, ``r`` is not positive, or no targets matched.
    """
    if r <= 0:
        raise ValueError(f"LoRA rank must be positive, got r={r}")

    if alpha is None:
        alpha = float(r)
    if alpha == 0:
        raise ValueError(
            "adapter.lora.alpha must be non-zero. llama.cpp computes the effective scale as "
            "`alpha ? user_scale * alpha / rank : user_scale` (src/llama-adapter.h:55), so "
            "alpha == 0 does not scale the adapter to zero — it silently DROPS the alpha/rank "
            "factor and applies the user scale alone. Pass alpha=r for a scale of 1.0."
        )

    if sigma is None:
        sigma = 1.0 / np.sqrt(r)

    targets = enumerate_targets(
        base_gguf_path,
        preset=preset,
        include_output=include_output,
        include_token_embd=include_token_embd,
    )
    if not targets:
        raise ValueError(f"no LoRA targets matched in {base_gguf_path} (preset={preset})")

    architecture = _base_architecture(base_gguf_path)
    rng = np.random.default_rng(seed)

    writer = gguf.GGUFWriter(str(out_path), arch=architecture)
    writer.add_type(gguf.GGUFType.ADAPTER)
    writer.add_string(gguf.Keys.Adapter.TYPE, "lora")
    writer.add_float32(gguf.Keys.Adapter.LORA_ALPHA, float(alpha))

    for target in targets:
        a, b = _zero_init_pair(target, r, sigma, rng)
        # The loader strips only ".lora_a"/".lora_b" and looks the rest up by name, so the
        # ".weight" stays (src/llama-adapter.cpp:273-285, 330).
        writer.add_tensor(f"{target.name}.lora_a", a)
        writer.add_tensor(f"{target.name}.lora_b", b)

    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()

    return targets


def _zero_init_pair(
    target: LoraTarget, r: int, sigma: float, rng: np.random.Generator
) -> tuple[np.ndarray, np.ndarray]:
    """Build the ``(A, B)`` numpy pair for one target, in the convention its shape check wants.

    numpy shapes are the reverse of GGUF ``ne``, so a numpy array of shape ``(r, n_in)`` is
    written with ``ne = [n_in, r]``.
    """
    if target.is_token_embd:
        # Flipped convention: the loader wants b.ne[1] == n_embd and a.ne[1] == n_vocab
        # (src/llama-adapter.cpp:356-360). For token_embd, ne[0] is n_embd and ne[1] is n_vocab.
        n_embd, n_vocab = target.n_in, target.n_out
        a = rng.normal(0.0, sigma, size=(n_vocab, r)).astype(np.float32)  # ne = [r, n_vocab]
        b = np.zeros((n_embd, r), dtype=np.float32)  # ne = [r, n_embd]
        return a, b

    # Normal convention: a.ne = [n_in, r], b.ne = [r, n_out]
    # (src/llama-adapter.cpp:362-367).
    a = rng.normal(0.0, sigma, size=(r, target.n_in)).astype(np.float32)
    b = np.zeros((target.n_out, r), dtype=np.float32)
    return a, b


def read_adapter(path: str | pathlib.Path) -> AdapterInfo:
    """Read an adapter GGUF's metadata and validate its structure.

    Args:
        path: Path to the adapter GGUF.

    Returns:
        The adapter's architecture, alpha, per-target ranks, and per-target A/B shapes.

    Raises:
        ValueError: If the file is not a LoRA adapter, a ``lora_a``/``lora_b`` pair is
            incomplete, or an A/B tensor is not F32.
    """
    reader = gguf.GGUFReader(str(path), "r")

    def _kv(key: str) -> object | None:
        field = reader.get_field(key)
        return None if field is None else field.contents()

    if str(_kv(gguf.Keys.General.TYPE)) != gguf.GGUFType.ADAPTER:
        raise ValueError(f"{path} is not an adapter GGUF (general.type is not 'adapter')")
    if str(_kv(gguf.Keys.Adapter.TYPE)) != "lora":
        raise ValueError(f"{path} is not a LoRA adapter (adapter.type is not 'lora')")

    architecture = str(_kv(gguf.Keys.General.ARCHITECTURE))
    alpha = float(_kv(gguf.Keys.Adapter.LORA_ALPHA) or 0.0)

    pairs: dict[str, dict[str, gguf.ReaderTensor]] = {}
    for tensor in reader.tensors:
        for suffix in ("lora_a", "lora_b"):
            if tensor.name.endswith(f".{suffix}"):
                base = tensor.name[: -len(f".{suffix}")]
                pairs.setdefault(base, {})[suffix] = tensor

    ranks: dict[str, int] = {}
    shapes: dict[str, tuple[tuple[int, ...], tuple[int, ...]]] = {}

    for base, pair in sorted(pairs.items()):
        if "lora_a" not in pair or "lora_b" not in pair:
            missing = "lora_b" if "lora_a" in pair else "lora_a"
            raise ValueError(f"{path}: '{base}' is missing its .{missing} tensor")

        a, b = pair["lora_a"], pair["lora_b"]
        for tensor in (a, b):
            if tensor.tensor_type != gguf.GGMLQuantizationType.F32:
                raise ValueError(
                    f"{path}: '{tensor.name}' is {tensor.tensor_type.name}, but adapter "
                    "tensors must be F32 — only the adapter trains, and it trains in F32"
                )

        a_ne = tuple(int(d) for d in a.shape)
        b_ne = tuple(int(d) for d in b.shape)
        # The loader takes the rank from b->ne[0] when computing the scale
        # (src/llama-adapter.h:53).
        ranks[base] = b_ne[0]
        shapes[base] = (a_ne, b_ne)

    if alpha == 0.0:
        # Not fatal: a foreign adapter may legitimately intend user-scale-only semantics. But
        # it is almost never what someone means, so say so.
        import warnings

        warnings.warn(
            f"{path}: adapter.lora.alpha is 0, so llama.cpp will drop the alpha/rank scale "
            "factor and apply the user scale alone (src/llama-adapter.h:55)",
            stacklevel=2,
        )

    return AdapterInfo(architecture=architecture, alpha=alpha, ranks=ranks, shapes=shapes)
