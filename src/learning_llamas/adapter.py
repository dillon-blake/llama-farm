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

import ctypes
import pathlib
from dataclasses import dataclass

import numpy as np

from learning_llamas import _ffi

try:
    import gguf
except ImportError as exc:  # pragma: no cover - environment problem, not a code path
    raise ImportError(
        "learning-llamas uses the gguf-py that ships with the vendored llama.cpp, so that the "
        "file-format code and the pinned llama.cpp commit stay one atomic version (ADR-0001).\n"
        "  install it: pip install -e vendor/llama.cpp/gguf-py"
    ) from exc

# The default LoRA target set (BLUEPRINT D5): attention projections and the FFN.
#
# The `_exps` entries are the MoE expert stacks (S1-28). They are 3D -- [n_in, n_out, n_expert] --
# and adapting them is not a variation on the dense case, it is the whole reason MUL_MAT_ID needed a
# backward at all: build_lora_mm_id computes mul_mat_id(B, mul_mat_id(A, cur, ids), ids), so the
# trainable A/B tensors ARE the expert operand.
#
# The `ssm_*` entries are Mamba's four LoRA-able projections (S1-47). They are plain MUL_MAT through
# build_lora_mm (models/mamba-base.cpp), so they need no new kernel -- but a mamba GGUF names them
# ssm_in/ssm_x/ssm_dt/ssm_out, none of which the attention/FFN preset matches, so without these a
# mamba adapter would come out empty and create_zero_adapter would (correctly) refuse it. The frozen
# SSM tensors -- ssm_a, ssm_d (no `.weight` suffix), ssm_conv1d, and the biases -- are deliberately
# absent: A and the conv weight take no gradient (the backward switch asserts as much), and
# enumerate_targets only matches `*.weight` regardless.
#
# A dense model simply has no tensor with these names, so listing them costs nothing there.
DEFAULT_PRESET: tuple[str, ...] = (
    "attn_q",
    "attn_k",
    "attn_v",
    "attn_qkv",
    "attn_output",
    "ffn_up",
    "ffn_gate",
    "ffn_down",
    "ffn_up_exps",
    "ffn_gate_exps",
    "ffn_down_exps",
    "ssm_in",
    "ssm_x",
    "ssm_dt",
    "ssm_out",
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
        n_expert: The tensor's ``ne[2]``, for a 3D MoE expert stack; 0 for an ordinary 2D
            tensor. A nonzero value makes the adapter's A and B 3D as well — one slice per
            expert — because that is what ``ggml_mul_mat_id`` consumes.
    """

    name: str
    n_in: int
    n_out: int
    dtype: gguf.GGMLQuantizationType
    is_token_embd: bool
    n_expert: int = 0


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

        # ne[2] is the expert count on a MoE stack, and reading only ne[0] and ne[1] would silently
        # flatten it: the adapter would come out 2D, the loader's shape check (which only validates
        # ne[0] and ne[1]) would ACCEPT it, and ggml_mul_mat_id would then read an expert axis of 1
        # for a 4-expert model. A wrong answer with no error anywhere.
        n_expert = int(tensor.shape[2]) if len(tensor.shape) > 2 and tensor.shape[2] > 1 else 0

        targets.append(
            LoraTarget(
                name=tensor.name,
                n_in=n_in,
                n_out=n_out,
                dtype=tensor.tensor_type,
                is_token_embd=(module == TOKEN_EMBD),
                n_expert=n_expert,
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

    rng = np.random.default_rng(seed)
    pairs = [(t.name, *_zero_init_pair(t, r, sigma, rng)) for t in targets]

    _write_adapter_gguf(out_path, _base_architecture(base_gguf_path), float(alpha), pairs)

    return targets


def _write_adapter_gguf(
    out_path: str | pathlib.Path,
    architecture: str,
    alpha: float,
    pairs: list[tuple[str, np.ndarray, np.ndarray]],
) -> None:
    """Write a LoRA adapter GGUF.

    ``add_tensor`` writes ``ne`` as the numpy shape reversed (``gguf_writer.py:265-268``), so a 3D
    MoE pair handed over as ``(n_expert, r, n_in)`` lands in the file as ``ne = [n_in, r,
    n_expert]`` with its expert axis intact and nothing here to do about it. The axis can only be
    lost upstream, by whoever built the array — which is exactly where it used to be lost
    (:func:`save_adapter`).

    Args:
        out_path: Where to write it.
        architecture: The base model's architecture string. The loader requires a match.
        alpha: ``adapter.lora.alpha``.
        pairs: One ``(target_name, A, B)`` per target, in GGUF-ready numpy layout.
    """
    writer = gguf.GGUFWriter(str(out_path), arch=architecture)
    writer.add_type(gguf.GGUFType.ADAPTER)
    writer.add_string(gguf.Keys.Adapter.TYPE, "lora")
    writer.add_float32(gguf.Keys.Adapter.LORA_ALPHA, alpha)

    for name, a, b in pairs:
        # The loader strips only ".lora_a"/".lora_b" and looks the rest up by name, so the
        # ".weight" stays (src/llama-adapter.cpp:273-285, 330).
        writer.add_tensor(f"{name}.lora_a", a)
        writer.add_tensor(f"{name}.lora_b", b)

    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()


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

    if target.n_expert:
        # A MoE expert stack. A and B get one slice per expert, so that build_lora_mm_id's
        #
        #     mul_mat_id(B, mul_mat_id(A, cur, ids), ids)
        #
        # gathers the same expert from the adapter that the base gathered from itself.
        #
        #     a.ne = [n_in,  r,     n_expert]
        #     b.ne = [r,     n_out, n_expert]
        #
        # The loader validates only ne[0] and ne[1] (llama-adapter.cpp:362-367), so a 2D adapter
        # would load happily against a 3D base and then be read as a 1-expert stack. The shape has
        # to be right here, because nothing downstream will complain.
        a = rng.normal(0.0, sigma, size=(target.n_expert, r, target.n_in)).astype(np.float32)
        b = np.zeros((target.n_expert, target.n_out, r), dtype=np.float32)
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


def save_adapter(
    libs: _ffi.Libraries,
    adapter: int,
    out_path: str | pathlib.Path,
    architecture: str,
    alpha: float,
) -> int:
    """Write a live adapter's tensors — trained or not — out as a GGUF.

    The tensors are read from the adapter itself, not from a training context, so this works after
    ``ll_opt_free`` and on an adapter that was never trained at all. It is the counterpart of
    :func:`create_zero_adapter`: what that writes, this reads back.

    Args:
        libs: The loaded native libraries.
        adapter: A ``llama_adapter_lora *``.
        out_path: Where to write the GGUF.
        architecture: The base model's architecture string. The loader refuses a mismatch, so
            passing the wrong one produces a file that cannot be loaded rather than one that
            silently misbehaves.
        alpha: ``adapter.lora.alpha``. Must be non-zero: llama.cpp computes the effective scale as
            ``alpha ? user_scale * alpha / rank : user_scale`` (src/llama-adapter.h:55), so a zero
            alpha does not scale the adapter to nothing — it silently DROPS the ``alpha/rank``
            factor and applies the user scale alone.

    Returns:
        How many base tensors were written (each contributing one A and one B).

    Raises:
        ValueError: If ``alpha`` is zero, or the adapter has no tensors.
        RuntimeError: If the shim rejects the adapter.
    """
    if alpha == 0:
        raise ValueError(
            "adapter.lora.alpha must be non-zero. llama.cpp computes the effective scale as "
            "`alpha ? user_scale * alpha / rank : user_scale`, so alpha == 0 does not scale the "
            "adapter to zero — it drops the alpha/rank factor entirely. Pass alpha=rank for 1.0."
        )

    n = _ffi.check(libs.farm.ll_adapter_n_tensors(adapter), "ll_adapter_n_tensors")
    if n == 0:
        raise ValueError("the adapter has no tensors")

    pairs = []

    for i in range(n):
        name_buf = ctypes.create_string_buffer(256)
        ne_a = (ctypes.c_int64 * 4)()
        ne_b = (ctypes.c_int64 * 4)()

        _ffi.check(
            libs.farm.ll_adapter_tensor_info(adapter, i, name_buf, 256, ne_a, ne_b),
            "ll_adapter_tensor_info",
        )

        name = name_buf.value.decode()

        # numpy shapes are the reverse of GGUF ne, and the writer wants numpy. An A of
        # ne = [n_in, r] is a numpy array of shape (r, n_in) -- and an A of a MoE expert stack,
        # ne = [n_in, r, n_expert], is (n_expert, r, n_in). See _np_shape.
        a = _read(libs, adapter, i, is_b=False).reshape(_np_shape(ne_a))
        b = _read(libs, adapter, i, is_b=True).reshape(_np_shape(ne_b))

        pairs.append((name, a, b))

    _write_adapter_gguf(out_path, architecture, float(alpha), pairs)

    return n


def _np_shape(ne: ctypes.Array[ctypes.c_int64]) -> tuple[int, ...]:
    """The numpy shape of an adapter tensor from its ggml ``ne``.

    ``ll_adapter_tensor_info`` always fills GGML_MAX_DIMS (4) entries, padding the unused ones with
    1 (farm_api.h), and a numpy shape is ``ne`` reversed. So the dimensionality has to be recovered
    by dropping the trailing 1s — but only down to **two** dimensions, never further: a rank-1 LoRA
    has ``a.ne = [n_in, 1, 1, 1]``, and collapsing that to a 1-D ``(n_in,)`` would write an A whose
    ``ne[1]`` the loader can no longer compare against ``b.ne[0]``
    (``src/llama-adapter.cpp:362-367``).

    The dimension this exists for is ``ne[2]``, the MoE expert axis. Reshaping unconditionally to
    ``(ne[1], ne[0])`` — which is what this did before S1-50 — cannot express a
    ``[n_in, r, n_expert]`` expert stack at all: it raises ValueError, so a MoE run could not save
    its adapter after spending the entire training budget on it. There is no version of that bug
    that shows up before the end of the run.
    """
    dims = [int(x) for x in ne]
    while len(dims) > 2 and dims[-1] == 1:
        dims.pop()
    return tuple(reversed(dims))


def _index_by_name(libs: _ffi.Libraries, adapter: int) -> dict[str, int]:
    """Map each base tensor name to its index in the shim's adapter accessors.

    **The shim's index is not** :func:`enumerate_targets`' **index, and confusing the two is
    silent.** ``ll_adapter_*`` addresses tensors by their position in the base-tensor names sorted
    **lexicographically** (``farm_api.h``); :func:`enumerate_targets` returns them in **GGUF file
    order**. For a dense llama those orders differ from the very first FFN tensor — file order is
    ``ffn_gate, ffn_up, ffn_down``, lexicographic is ``ffn_down, ffn_gate, ffn_up`` — so index 5
    means two different tensors depending on who you ask.

    When the two tensors happen to have the same shape the mistake does not even raise: you read
    the wrong tensor's weights and get plausible numbers. Address them by name.
    """
    n = _ffi.check(libs.farm.ll_adapter_n_tensors(adapter), "ll_adapter_n_tensors")

    out: dict[str, int] = {}
    for i in range(n):
        name_buf = ctypes.create_string_buffer(256)
        ne_a = (ctypes.c_int64 * 4)()
        ne_b = (ctypes.c_int64 * 4)()
        _ffi.check(
            libs.farm.ll_adapter_tensor_info(adapter, i, name_buf, 256, ne_a, ne_b),
            "ll_adapter_tensor_info",
        )
        out[name_buf.value.decode()] = i

    return out


def _read(libs: _ffi.Libraries, adapter: int, index: int, is_b: bool) -> np.ndarray:
    """One adapter tensor, flat. ``index`` is the shim's — see :func:`_index_by_name`."""
    n = _ffi.check(libs.farm.ll_adapter_get(adapter, index, is_b, None, 0), "ll_adapter_get")

    buf = (ctypes.c_float * n)()
    got = _ffi.check(libs.farm.ll_adapter_get(adapter, index, is_b, buf, n), "ll_adapter_get")

    return np.frombuffer(buf, dtype=np.float32, count=got).copy()
