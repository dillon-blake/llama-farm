"""Record the PEFT reference curve. Run ONCE, in a torch env; commit the JSON it writes.

``torch`` and ``peft`` are **not** dependencies of learning-llamas — not runtime, not test. This
script is the only thing that imports them, it is never run by the test suite, and the gate
(``tests/test_convergence.py``) reads only the committed ``reference_curve.json``.

    python -m tests.convergence.record_reference

See ``tests/convergence/README.md`` for the recording environment and when to regenerate.

What makes this a reference rather than a coincidence
-----------------------------------------------------
Three things are twinned explicitly, and the script **refuses to record** if any of them fails:

1. **The model.** The HF twin is built from the same weight tensors as the GGUF fixture, with the
   Q/K RoPE row permutation applied — and :func:`hf_twin.verify` checks its logits against the
   float64 reference before a single step is taken. Without that check, a wrong permutation gives
   a perfectly plausible curve for the wrong model.
2. **The adapter init.** PEFT initializes LoRA A with ``kaiming_uniform(a=sqrt(5))``; learning-
   llamas uses ``normal(0, 1/sqrt(r))``. Neither is wrong, and they are not the same, so the
   recorder **overwrites** PEFT's A with the fixture adapter's own A. (B is zero on both sides by
   construction, which is what makes a fresh adapter an exact no-op.)
3. **The optimizer.** ``torch.optim.AdamW`` with the same betas/eps/weight-decay, and — crucially
   — the same loss: a weighted mean normalized by ``sum(w)``, not by the token count. HF's own
   ``labels=`` path shifts internally and averages over unmasked tokens; the loss is computed by
   hand here instead, from the same ``(tokens, targets, weights)`` arrays the gate feeds
   learning-llamas, so there is nothing to get subtly wrong.
"""

from __future__ import annotations

import json
import pathlib
import sys
from importlib.metadata import version

import numpy as np

from ..fixtures import gen_tiny_llama
from . import config, hf_twin


def main() -> int:
    import torch
    from peft import LoraConfig, get_peft_model

    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))
    from tests import reference_llama as ref  # noqa: PLC0415 — needs the path above

    torch.manual_seed(0)
    torch.use_deterministic_algorithms(True)

    hp = gen_tiny_llama.HPARAMS
    cache = pathlib.Path(__file__).resolve().parents[1] / ".fixtures"
    gguf_path, _ = gen_tiny_llama.build("f32", cache)

    tokens, targets, weights = config.dataset(hp.n_vocab)

    # ---------------------------------------------------------------------------------------
    # 1. The twin is the same model. Checked, not assumed.
    # ---------------------------------------------------------------------------------------
    model = hf_twin.build(hp)

    base, ref_hp = ref.load_model(gguf_path)
    ref_logits, _ = ref.forward(base, ref_hp, {}, 1.0, tokens[0])
    deviation = hf_twin.verify(model, tokens[0], ref_logits)
    print(f"HF twin vs the float64 reference: max logit deviation {deviation:.3e}  OK")

    # ---------------------------------------------------------------------------------------
    # 2. PEFT, with OUR A.
    # ---------------------------------------------------------------------------------------
    peft_model = get_peft_model(
        model,
        LoraConfig(
            r=config.RANK,
            lora_alpha=config.ALPHA,
            lora_dropout=config.LORA_DROPOUT,
            target_modules=config.HF_TARGET_MODULES,
            bias="none",
            task_type="CAUSAL_LM",
        ),
    )

    a_init = _adapter_a(gguf_path)
    n_set = 0
    for name, param in peft_model.named_parameters():
        if ".lora_A." in name:
            param.data = torch.tensor(a_init[_gguf_name(name)], dtype=torch.float32)
            n_set += 1
        elif ".lora_B." in name:
            # Zero by construction on both sides. Assert it rather than trust it: a nonzero B makes
            # the step-0 adapter a no-op no longer, and the two curves would start apart.
            assert param.data.abs().max() == 0.0, f"{name} is not zero at init"
    print(f"seeded {n_set} lora_A tensors from the fixture adapter")

    # ---------------------------------------------------------------------------------------
    # 3. Train, recording the loss BEFORE each update (which is what the shim reports too).
    # ---------------------------------------------------------------------------------------
    opt = torch.optim.AdamW(
        [p for p in peft_model.parameters() if p.requires_grad],
        lr=config.LR,
        betas=config.BETAS,
        eps=config.EPS,
        weight_decay=config.WEIGHT_DECAY,
    )

    tok_t = torch.tensor(tokens, dtype=torch.long)
    tgt_t = torch.tensor(targets, dtype=torch.long)
    w_t = torch.tensor(weights, dtype=torch.float32)

    curve: list[float] = []
    for _epoch in range(config.EPOCHS):
        for i in range(config.N_SAMPLES):
            logits = peft_model(tok_t[i : i + 1]).logits[0]

            # The shim's loss exactly: sum_i w_i' * (logsumexp(x_i) - x_i[target_i]), with the
            # 1/sum(w) folded into the weights. NOT a mean over all tokens.
            logp = torch.log_softmax(logits.float(), dim=-1)
            picked = logp[torch.arange(config.SEQ_LEN), tgt_t[i]]
            loss = -(w_t[i] * picked).sum() / w_t[i].sum()

            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()

            curve.append(float(loss.item()))

    print(f"recorded {len(curve)} steps: {curve[0]:.6f} -> {curve[-1]:.6f}")

    payload = {
        "_comment": (
            "Recorded once by tests/convergence/record_reference.py. Do not hand-edit. "
            "See tests/convergence/README.md to regenerate."
        ),
        "identity": config.identity(hp.n_vocab),
        "twin_max_logit_deviation": deviation,
        "versions": {
            "torch": version("torch"),
            "peft": version("peft"),
            "transformers": version("transformers"),
            "numpy": version("numpy"),
        },
        "config": {
            "rank": config.RANK,
            "alpha": config.ALPHA,
            "lr": config.LR,
            "betas": list(config.BETAS),
            "eps": config.EPS,
            "weight_decay": config.WEIGHT_DECAY,
            "lora_dropout": config.LORA_DROPOUT,
            "seq_len": config.SEQ_LEN,
            "n_samples": config.N_SAMPLES,
            "epochs": config.EPOCHS,
            "shuffle": config.SHUFFLE,
            "targets": config.HF_TARGET_MODULES,
        },
        "curve": curve,
    }
    config.CURVE_PATH.write_text(json.dumps(payload, indent=2) + "\n")
    print(f"wrote {config.CURVE_PATH}")
    return 0


def _gguf_name(peft_param: str) -> str:
    """PEFT's parameter name to the GGUF base tensor name.

    ``...model.layers.0.self_attn.q_proj.lora_A.default.weight`` -> ``blk.0.attn_q.weight``.
    """
    parts = peft_param.split(".")
    layer = parts[parts.index("layers") + 1]
    hf_module = parts[parts.index("lora_A") - 1]
    gguf_module = next(g for g, h in config.GGUF_TO_HF.items() if h == hf_module)
    return f"blk.{layer}.{gguf_module}.weight"


def _adapter_a(gguf_path: pathlib.Path) -> dict[str, np.ndarray]:
    """The fixture adapter's A tensors, keyed by base tensor name.

    Built with the same seed the gate uses, so both sides start from the identical A.
    """
    import tempfile

    import gguf as gguf_py

    from learning_llamas import create_zero_adapter

    with tempfile.TemporaryDirectory() as tmp:
        out = pathlib.Path(tmp) / "a.gguf"
        create_zero_adapter(
            gguf_path, out, r=config.RANK, alpha=config.ALPHA, seed=config.ADAPTER_SEED
        )
        reader = gguf_py.GGUFReader(str(out), "r")
        return {
            t.name[: -len(".lora_a")]: np.array(t.data)
            for t in reader.tensors
            if t.name.endswith(".lora_a")
        }


if __name__ == "__main__":
    raise SystemExit(main())
