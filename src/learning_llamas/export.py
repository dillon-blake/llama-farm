"""Merge a LoRA adapter into its base model and write the result as a GGUF.

An adapter is the right thing to *train* and often the wrong thing to *ship*. Serving a base plus
an adapter means every adapted matmul runs three times instead of once, and every downstream tool —
ollama, llama-server, a GGUF-only runtime — has to know about adapters at all. Merging folds the
delta back into the weights: one file, no adapter, no extra matmuls, and nothing to configure.

The arithmetic
--------------
This is the numpy transcription of the graph llama.cpp's ``export-lora`` builds
(``tools/export-lora/export-lora.cpp:348-368``), and the only interesting thing about it is the
index order, which is easy to get exactly backwards.

GGUF's ``ne`` is the reverse of a numpy shape. A base tensor of ``ne = [n_in, n_out]`` is a numpy
array of shape ``(n_out, n_in)``; an ``A`` of ``ne = [n_in, r]`` is ``(r, n_in)``; a ``B`` of
``ne = [r, n_out]`` is ``(n_out, r)``. Working the ggml ``mul_mat`` through into numpy indices::

    delta = B @ A                        (n_out, r) @ (r, n_in) -> (n_out, n_in)
    merged = base + scale * delta

and for ``token_embd``, whose adapter uses the flipped convention the loader validates
(``llama-adapter.cpp:356-368``)::

    delta = A @ B.T                      (n_vocab, r) @ (r, n_embd) -> (n_vocab, n_embd)

The scale is llama.cpp's, exactly: ``alpha ? user_scale * alpha / rank : user_scale``
(``llama-adapter.h:55``) — note that ``alpha == 0`` does not mean "no adapter", it means the
``alpha/rank`` factor is dropped.

Re-quantization
---------------
The merged tensor goes back to the base tensor's **original type**, which is the whole point: merge
a Q4_K model and you get a Q4_K model, the same size, that runs on the same hardware. Doing that
requires libggml's quantizer, because gguf-py cannot write K-quants — see
:mod:`learning_llamas.quant`.

The merge itself happens in float32, so the delta is added at full precision and only the final
tensor is re-quantized. That is one quantization round-trip, not two.

Provenance: the merge graph and scale rule are read from ``export-lora.cpp`` (llama.cpp, MIT,
commit ``4f37f519722aa3242eecb7649466b4a4a2d6d6da``) as an *algorithm reference*, and transcribed
into numpy. No code is copied; see docs/PROVENANCE.md.
"""

from __future__ import annotations

import logging
import pathlib

import numpy as np

from learning_llamas import _ffi
from learning_llamas.quant import ImatrixRequired, dequantize, quantize

try:
    import gguf
except ImportError as exc:  # pragma: no cover - environment problem, not a code path
    raise ImportError("learning-llamas needs the vendored gguf-py; see adapter.py") from exc

log = logging.getLogger(__name__)

# Types a merged tensor is written as when its original type cannot be reproduced. F16 rather than
# F32: it is lossless relative to the merge (which is F32 but whose inputs came from a lossier
# quant), and half the size.
FALLBACK_TYPE = gguf.GGMLQuantizationType.F16


def _adapter_pairs(
    adapter_path: str | pathlib.Path,
) -> tuple[dict[str, tuple[np.ndarray, np.ndarray]], float]:
    """Read an adapter GGUF into ``{base_name: (A, B)}`` numpy arrays, plus its alpha."""
    reader = gguf.GGUFReader(str(adapter_path), "r")

    alpha_field = reader.get_field("adapter.lora.alpha")
    alpha = float(alpha_field.contents()) if alpha_field else 0.0

    parts: dict[str, dict[str, np.ndarray]] = {}

    for tensor in reader.tensors:
        for suffix, role in ((".lora_a", "a"), (".lora_b", "b")):
            if not tensor.name.endswith(suffix):
                continue

            if tensor.tensor_type != gguf.GGMLQuantizationType.F32:
                # Upstream refuses these too (export-lora.cpp:304-306). A quantized adapter cannot
                # have been trained by us, and merging one would compound two quantizations.
                raise ValueError(
                    f"{tensor.name} is {tensor.tensor_type.name}; LoRA A/B must be F32"
                )

            base_name = tensor.name[: -len(suffix)]
            values = np.array(tensor.data, dtype=np.float32).reshape(tuple(tensor.shape)[::-1])
            parts.setdefault(base_name, {})[role] = values

    pairs: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for base_name, ab in parts.items():
        if "a" not in ab or "b" not in ab:
            raise ValueError(f"{base_name} has an A without a B, or a B without an A")
        pairs[base_name] = (ab["a"], ab["b"])

    return pairs, alpha


def _delta(a: np.ndarray, b: np.ndarray, is_token_embd: bool) -> np.ndarray:
    """The LoRA delta, in the base tensor's numpy shape. See the module docstring."""
    if is_token_embd:
        # A is (n_vocab, r), B is (n_embd, r) -> (n_vocab, n_embd).
        return a @ b.T

    # A is (r, n_in), B is (n_out, r) -> (n_out, n_in).
    return b @ a


def merge(
    base_path: str | pathlib.Path,
    adapter_path: str | pathlib.Path,
    out_path: str | pathlib.Path,
    libs: _ffi.Libraries,
    scale: float = 1.0,
) -> dict[str, str]:
    """Fold an adapter into its base model and write a standalone GGUF.

    Args:
        base_path: The base model GGUF.
        adapter_path: The adapter GGUF.
        out_path: Where to write the merged model.
        libs: The loaded native libraries, for re-quantizing K-quants.
        scale: The user scale, as ``llama-cli --lora-scaled`` would take it. The effective scale
            also folds in ``alpha / rank``.

    Returns:
        ``{base_tensor_name: output_type_name}`` for every tensor the adapter touched — so a caller
        can see at a glance whether anything fell back to F16.

    Raises:
        ValueError: If the adapter's architecture does not match the base's, or an adapter tensor
            is quantized, or a shape does not line up.
    """
    reader = gguf.GGUFReader(str(base_path), "r")
    pairs, alpha = _adapter_pairs(adapter_path)

    arch_field = reader.get_field(gguf.Keys.General.ARCHITECTURE)
    if arch_field is None:
        raise ValueError(f"{base_path} has no general.architecture")
    architecture = str(arch_field.contents())

    adapter_arch = gguf.GGUFReader(str(adapter_path), "r").get_field(gguf.Keys.General.ARCHITECTURE)
    if adapter_arch is None or str(adapter_arch.contents()) != architecture:
        raise ValueError(
            f"the adapter's architecture "
            f"({adapter_arch.contents() if adapter_arch else None!r}) is not the base's "
            f"({architecture!r}). llama.cpp refuses that pairing and so does this."
        )

    writer = gguf.GGUFWriter(str(out_path), arch=architecture)

    _copy_metadata(reader, writer)

    merged_types: dict[str, str] = {}

    for tensor in reader.tensors:
        shape = tuple(int(x) for x in tensor.shape)[::-1]  # numpy order
        pair = pairs.get(tensor.name)

        if pair is None:
            # Untouched: hand the original bytes straight through. No dequantize/requantize round
            # trip, so a tensor the adapter never saw is bit-identical in the output.
            _write(writer, tensor.name, np.asarray(tensor.data), tensor.tensor_type)
            continue

        a, b = pair
        rank = a.shape[0] if tensor.name.split(".")[0] != "token_embd" else a.shape[1]

        # llama.cpp's rule, exactly: a zero alpha DROPS the alpha/rank factor rather than zeroing
        # the adapter (llama-adapter.h:55).
        effective = scale * alpha / rank if alpha else scale

        base = dequantize(tensor.data, tensor.tensor_type, shape)
        delta = _delta(a, b, is_token_embd=tensor.name.startswith("token_embd"))

        if delta.shape != base.shape:
            raise ValueError(
                f"{tensor.name}: the LoRA delta is {delta.shape} but the base tensor is "
                f"{base.shape}. The adapter does not fit this model."
            )

        merged = base + effective * delta

        out_type = tensor.tensor_type
        try:
            raw = quantize(merged, out_type, libs)
        except ImatrixRequired:
            log.warning(
                "%s is %s, which cannot be re-quantized without an importance matrix; writing it "
                "as %s instead. The merged model will be larger than the base.",
                tensor.name,
                out_type.name,
                FALLBACK_TYPE.name,
            )
            out_type = FALLBACK_TYPE
            raw = quantize(merged, out_type, libs)

        _write(writer, tensor.name, np.frombuffer(raw, dtype=np.uint8), out_type, shape)

        merged_types[tensor.name] = out_type.name

    if not merged_types:
        raise ValueError(
            "the adapter matched no tensor in the base model. Its tensor names are probably from a "
            "different model."
        )

    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()

    return merged_types


def _write(
    writer: gguf.GGUFWriter,
    name: str,
    data: np.ndarray,
    dtype: gguf.GGMLQuantizationType,
    shape: tuple[int, ...] | None = None,
) -> None:
    """Add a tensor to the writer in its ORIGINAL type, quantized or not.

    GGUFWriter takes a **byte-shaped** array plus ``raw_dtype`` and derives the logical shape itself
    (``quant_shape_from_byte_shape``). Handing it the logical shape as well is not merely redundant,
    it contradicts what it computes -- a Q4_K row of 256 elements is 144 bytes, not 256, and the
    error it raises says so in terms of neither the tensor nor the caller.

    So everything goes through one path: raw bytes, and let the writer do the arithmetic.

    Args:
        writer: The GGUF writer.
        name: The tensor name.
        data: Its data. Quantized bytes, or a float array (which is viewed as bytes).
        dtype: Its GGML type.
        shape: The numpy logical shape. Needed only when ``data`` arrives flat, i.e. straight from
            :func:`~learning_llamas.quant.quantize`.
    """
    raw = np.ascontiguousarray(data)

    if raw.dtype != np.uint8:
        raw = raw.view(np.uint8)
    elif shape is not None:
        raw = raw.reshape(gguf.quants.quant_shape_to_byte_shape(shape, dtype))

    writer.add_tensor(name, raw, raw_dtype=dtype)


# Keys the writer sets itself, or that describe the file rather than the model. Copying them would
# either duplicate a key or assert something about the output that is no longer true.
_SKIP_KEYS = frozenset(
    {
        gguf.Keys.General.ARCHITECTURE,
        "GGUF.version",
        "GGUF.tensor_count",
        "GGUF.kv_count",
    }
)


def _copy_metadata(reader: gguf.GGUFReader, writer: gguf.GGUFWriter) -> None:
    """Carry every KV across unchanged.

    Unchanged, and that includes the tokenizer, the chat template, the rope settings and everything
    else the model needs to be usable. A merged model that lost its tokenizer is not a model.
    """
    for key, field in reader.fields.items():
        if key in _SKIP_KEYS:
            continue
        if not field.types:
            continue

        writer.add_key_value(key, field.contents(), field.types[0], sub_type=_sub_type(field))


def _sub_type(field: gguf.ReaderField) -> gguf.GGUFValueType | None:
    """An array field's element type, which ``add_key_value`` needs and cannot infer."""
    if field.types and field.types[0] == gguf.GGUFValueType.ARRAY and len(field.types) > 1:
        return field.types[1]
    return None


def _main(argv: list[str] | None = None) -> int:
    """``python -m learning_llamas.export`` — merge an adapter into its base model.

    There is deliberately no ``save-adapter`` subcommand. Saving reads a **live** adapter out of a
    training context, so it is a library call by nature
    (:func:`~learning_llamas.adapter.save_adapter`); a command-line version could only take an
    adapter file and write it back out again, which is ``cp``.
    """
    import argparse

    parser = argparse.ArgumentParser(
        prog="python -m learning_llamas.export",
        description="Fold a LoRA adapter into its base model and write a standalone GGUF.",
    )
    parser.add_argument("base", type=pathlib.Path, help="the base model GGUF")
    parser.add_argument("adapter", type=pathlib.Path, help="the adapter GGUF")
    parser.add_argument("out", type=pathlib.Path, help="where to write the merged model")
    parser.add_argument(
        "--scale",
        type=float,
        default=1.0,
        help="the user scale, as llama-cli --lora-scaled takes it (default: 1.0). The effective "
        "scale also folds in alpha/rank.",
    )

    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(message)s")

    libs = _ffi.load()
    merged = merge(args.base, args.adapter, args.out, libs, scale=args.scale)

    types = sorted(set(merged.values()))
    log.info(
        "merged %d tensors into %s (output types: %s)", len(merged), args.out, ", ".join(types)
    )

    return 0


if __name__ == "__main__":  # pragma: no cover - exercised by hand and by the docs
    raise SystemExit(_main())
