"""Will this model train, and if not, what exactly stops it?

Without this, the answer to *"can I LoRA-tune Mixtral?"* is: try it, and watch ggml abort inside
``ggml_compute_backward`` with a message naming an op enum and nothing else — not the tensor, not
the layer, not what to do about it. And that abort comes on the first **backward** pass, so it
arrives after the model has loaded, the data has tokenized, and the user has waited.

So walk the forward graph first. Work out which nodes the backward would actually **reach**, and
check those against the ops ggml can differentiate.

Two things make this a real check rather than a lint:

**Only the gradient path counts.** A model is full of ops with no backward rule — `ARGSORT`,
`ARGMAX`, whatever the sampler does — that the backward pass never touches. Reporting them would be
noise, and noise trains people to ignore the report. So the walk propagates "needs a gradient" from
the adapter tensors outward, exactly as ``ggml_build_backward_expand`` does, and only nodes it
actually reaches can block.

**And the quiet one: an adapter tensor that is in no graph node at all.** That means its target
projection does not go through ``build_lora_mm`` on this architecture. The tensor is flagged
trainable, ggml dutifully allocates it a gradient, and the gradient is **always zero**. It sits at
its initial value forever — while the loss falls perfectly well, because the *other* adapter tensors
are learning. Nothing fails. You have simply trained a smaller adapter than you asked for, and
nothing anywhere says so. It is reported as a warning, and it is the reason this module exists as
much as the blockers are.
"""

from __future__ import annotations

import ctypes
from dataclasses import dataclass
from enum import IntEnum

from learning_llamas import _ffi

MAX_ENTRIES = 64


class Status(IntEnum):
    """What the preflight thinks of a node."""

    OK = 0
    BLOCKED = 1
    """The backward pass would abort here."""
    WARN = 2
    """It will train — but not the way you think."""


class PreflightError(RuntimeError):
    """Training was refused because an op on the gradient path has no backward rule.

    Raised by :class:`~learning_llamas.train.loop.Trainer` before the first step. The whole point is
    that this arrives *early and legible* — before the model has loaded a batch, before anything has
    been differentiated — instead of the raw ``GGML_ABORT`` deep inside ``ggml_compute_backward``
    that names an op enum and nothing else, on the first backward pass, after the user has waited.

    Attributes:
        report: The full :class:`Report`, so a caller can inspect every blocker programmatically
            rather than scraping the message.
    """

    def __init__(self, report: Report) -> None:
        self.report = report
        super().__init__(
            f"{report.summary()}\n\n"
            "This model will not train: the backward pass would abort at the op(s) above, which is "
            "what this preflight exists to catch before a long run rather than during one. If you "
            "mean to try anyway — to reach the raw GGML_ABORT for debugging, or because you think "
            "the table is wrong — build the trainer with a config whose preflight=False."
        )


@dataclass(frozen=True)
class Finding:
    """One thing wrong with the graph.

    Attributes:
        node: The offending tensor's name.
        op: Its op — the *unary* op where that is the one that matters, since ``UNARY`` alone would
            not tell you whether it is a RELU (fine) or a HARDSWISH (not).
        status: :class:`Status`.
        detail: What is wrong, and which ticket unblocks it.
    """

    node: str
    op: str
    status: Status
    detail: str

    def __str__(self) -> str:
        """One line, readable in a terminal."""
        return f"[{self.status.name}] {self.op} ({self.node}): {self.detail}"


@dataclass(frozen=True)
class Report:
    """What the preflight found.

    Attributes:
        findings: Everything wrong, in graph order. Truncated at :data:`MAX_ENTRIES`.
        n_blocked: The **total** number of blocked nodes — which may exceed ``len(findings)``, since
            one unsupported op in a 32-layer model appears 32 times and a report that listed all of
            them would be worse than one that said "32".
    """

    findings: tuple[Finding, ...]
    n_blocked: int

    @property
    def trainable(self) -> bool:
        """Whether a backward pass would complete at all."""
        return self.n_blocked == 0

    @property
    def blockers(self) -> tuple[Finding, ...]:
        """The findings that stop a backward pass from completing at all."""
        return tuple(f for f in self.findings if f.status is Status.BLOCKED)

    @property
    def warnings(self) -> tuple[Finding, ...]:
        """The findings that let it train — but not the way you think."""
        return tuple(f for f in self.findings if f.status is Status.WARN)

    def summary(self) -> str:
        """The report a human should be shown before a long run starts."""
        if self.trainable and not self.warnings:
            return "trainable: every op on the gradient path has a backward rule."

        lines: list[str] = []

        if self.n_blocked:
            ops = sorted({f.op for f in self.blockers})
            lines.append(
                f"NOT trainable: {self.n_blocked} node(s) on the gradient path have no backward "
                f"rule ({', '.join(ops)})."
            )
        else:
            lines.append("trainable, with warnings.")

        lines.extend(f"  {f}" for f in self.findings)

        return "\n".join(lines)


class ll_preflight_entry(ctypes.Structure):  # noqa: N801 — mirrors the C name
    """``struct ll_preflight_entry`` (farm_api.h)."""

    _fields_ = [
        ("node", ctypes.c_char * 128),
        ("op", ctypes.c_char * 32),
        ("status", ctypes.c_int32),
        ("detail", ctypes.c_char * 256),
    ]


def preflight(libs: _ffi.Libraries, ctx: int, tokens: list[int]) -> Report:
    """Check whether a prepared training context can actually run a backward pass.

    Args:
        libs: The loaded native libraries.
        ctx: A ``llama_context *`` that :func:`_ffi.opt_init_lora` has already prepared — the walk
            is seeded from the trainable tensors, so there has to be a set of them.
        tokens: A representative batch. Its *length* is what shapes the graph; its contents are not
            read for anything but building it.

    Returns:
        What is wrong, if anything.

    Raises:
        RuntimeError: If the context has no training state, or the graph could not be built.
    """
    entries = (ll_preflight_entry * MAX_ENTRIES)()
    n_blocked = ctypes.c_int32()

    n = _ffi.check(
        libs.farm.ll_preflight(
            ctx,
            (ctypes.c_int32 * len(tokens))(*tokens),
            len(tokens),
            entries,
            MAX_ENTRIES,
            ctypes.byref(n_blocked),
        ),
        "ll_preflight",
    )

    findings = tuple(
        Finding(
            node=entries[i].node.decode(),
            op=entries[i].op.decode(),
            status=Status(entries[i].status),
            detail=entries[i].detail.decode(),
        )
        for i in range(n)
    )

    return Report(findings=findings, n_blocked=n_blocked.value)


def representative_tokens(libs: _ffi.Libraries, ctx: int) -> list[int]:
    """A throwaway batch the shape of a training batch.

    Only the *length* shapes the graph the walker reads — which ops appear is fixed by the
    architecture, not the token values — so the contents are irrelevant and token 0 (always a valid
    id) is used throughout. The length is the context's ``n_ubatch``: a training batch goes through
    the graph in one piece, so one ubatch of it is exactly the graph a real step would build.

    Args:
        libs: The loaded native libraries.
        ctx: A ``llama_context *``.

    Returns:
        ``n_ubatch`` copies of token 0.
    """
    n_ubatch = libs.llama.llama_n_ubatch(ctx)
    return [0] * max(1, int(n_ubatch))


def preflight_context(libs: _ffi.Libraries, ctx: int) -> Report:
    """Run the preflight over a context that already carries optimizer state.

    A convenience over :func:`preflight` that builds the representative batch itself. The context
    must already have optimizer state (``opt_init_lora`` has run): the walk is seeded from the
    trainable tensors. Most callers want :func:`preflight_adapter`, which manages that state.

    Args:
        libs: The loaded native libraries.
        ctx: A ``llama_context *`` with optimizer state.

    Returns:
        What is wrong, if anything.

    Raises:
        RuntimeError: If the context has no training state, or the graph could not be built.
    """
    return preflight(libs, ctx, representative_tokens(libs, ctx))


def preflight_adapter(libs: _ffi.Libraries, ctx: int, model: int, adapters: list[int]) -> Report:
    """Preflight a model+adapter on **throwaway** optimizer state, leaving the context untouched.

    The walk has to be seeded from a set of flagged trainable tensors, so optimizer state must
    exist — but it must not be the *training* optimizer state, and this is the subtle part.
    ``opt_step_custom`` sizes an optimizer context from the first graph it is handed, and the
    preflight's forward-only graph is not the training graph. Run the walk on the very context a
    trainer will then step through and that context is mis-sized: the first real backward aborts
    inside ``ggml_build_backward_expand`` (``GGML_ASSERT(ggml_is_scalar(b))``), which is precisely
    the kind of late, cryptic abort the preflight exists to replace.

    So this stands up its own optimizer state, walks, and tears it down in a ``finally``. The
    context is left exactly as it was found — and a trainer that builds its real optimizer state
    afterwards gets to hand it the training graph first, as ggml-opt requires.

    Args:
        libs: The loaded native libraries.
        ctx: A ``llama_context *`` in training mode.
        model: The ``llama_model *`` the adapters were loaded against.
        adapters: The ``llama_adapter_lora *`` whose A/B tensors seed the walk.

    Returns:
        What is wrong, if anything.

    Raises:
        RuntimeError: If the optimizer state cannot be set up (e.g. no trainable tensors), or the
            graph could not be built.
    """
    # Hyperparameters are irrelevant: no step is ever taken, only a forward pass at train=false.
    params = _ffi.ll_opt_params(alpha=1e-4, beta1=0.9, beta2=0.999, eps=1e-8, wd=0.0)
    _ffi.opt_init_lora(libs, ctx, model, adapters, params, opt_period=1, grad_clip=0.0)
    try:
        return preflight_context(libs, ctx)
    finally:
        _ffi.opt_free(libs, ctx)
