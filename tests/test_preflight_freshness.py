"""S1-42: the preflight's supported-backward op table must mirror the fork's actual ggml switch.

The preflight (``csrc/farm_preflight.cpp``) decides whether a model trains by checking each op on
the gradient path against a **hand-written** table — ``op_has_backward`` / ``unary_has_backward`` /
``glu_has_backward`` — that claims to list exactly the ops ``ggml_compute_backward`` can
differentiate. A hand-written mirror of another file's ``switch`` is only safe if something fails
when the two drift, and until now nothing did. So the table could lie in either direction, silently:

* **The dangerous direction** — the table claims a backward the switch does not have. The preflight
  says *trainable*, the user waits through load and tokenization, and then ``ggml_compute_backward``
  hits its ``default:`` and aborts, naming an op enum and nothing else. This is the exact outcome
  the preflight exists to prevent, now produced *by* the preflight.

* **The conservative direction** — the switch gained a backward the table still denies. The
  preflight reports a working model as blocked and refuses to train it. Less catastrophic, equally
  wrong; it is how SSM_CONV / SSM_SCAN sat on the blocked list for three tickets after S1-30/S1-31
  gave them backwards (fixed in this ticket, and this test is what would have caught it).

This is the same single-registry discipline as :mod:`tests.project_ops` and
``docs/dev/backward-coverage.md``: the fact lives in exactly one place — ``ggml.c`` — and everything
else is checked against it rather than trusted to have been kept in sync by hand.

The check is a source cross-parse, not a run of ``test-backend-ops``: the question here is purely
*which case labels exist* in two switch statements, which the text answers exactly and for free,
with no build. (Whether each of those backwards is numerically *correct* is ``test-backend-ops``
MODE_GRAD's job, guarded by :mod:`tests.test_backend_ops_grad`.)
"""

from __future__ import annotations

import pathlib
import re

ROOT = pathlib.Path(__file__).resolve().parents[1]
GGML_C = ROOT / "vendor" / "llama.cpp" / "ggml" / "src" / "ggml.c"
GGML_H = ROOT / "vendor" / "llama.cpp" / "ggml" / "include" / "ggml.h"
PREFLIGHT_CPP = ROOT / "csrc" / "farm_preflight.cpp"

# Ops the preflight table lists that are deliberately NOT cases in ggml_compute_backward.
#
# They are backward-only ops: the ops the backward pass itself EMITS (MUL_MAT_ID's backward emits
# OUT_PROD_ID*; the GLU family's backward emits GLU_BACK) and therefore never has to differentiate a
# second time. They can never appear on a forward graph, so the walker never truly reaches them --
# but ``op_has_backward`` lists them anyway so that a future graph-shape surprise reads as "fine",
# not "no backward rule". This is the freshness guard's own registry of known, explained exceptions;
# an extra op in the table that is NOT here is drift and fails the check.
EMITTED_ONLY_OPS = frozenset({"OUT_PROD_ID", "OUT_PROD_ID_GRP", "GLU_BACK"})


def _balanced_body(text: str, start_pattern: str) -> str:
    """The ``{...}`` block of the first construct matching ``start_pattern``, braces balanced."""
    m = re.search(start_pattern, text)
    assert m is not None, f"could not find {start_pattern!r} — did the vendored source move?"

    start = text.index("{", m.end() - 1)
    depth = 0
    for i in range(start, len(text)):
        if text[i] == "{":
            depth += 1
        elif text[i] == "}":
            depth -= 1
            if depth == 0:
                return text[start : i + 1]
    raise AssertionError(f"unbalanced braces after {start_pattern!r}")


def _case_labels(text: str, prefix: str) -> set[str]:
    """Every ``case <prefix>_NAME:`` label in ``text`` (anchored, so uses in expressions do not)."""
    return set(re.findall(rf"^\s*case\s+{prefix}_(\w+)\s*:", text, re.MULTILINE))


# --- the truth: what ggml_compute_backward actually differentiates -----------------------------


def ggml_backward_ops() -> set[str]:
    """The ``GGML_OP_*`` the fork's ``ggml_compute_backward`` has a real case for.

    ``GGML_OP_COUNT`` is excluded: it is the enum sentinel, grouped with ``default:`` onto the
    ``GGML_ABORT`` — i.e. it is explicitly *not* differentiable, exactly like anything unlisted.
    """
    body = _balanced_body(GGML_C.read_text(), r"static void ggml_compute_backward\s*\(")
    return _case_labels(body, "GGML_OP") - {"COUNT"}


def ggml_backward_unary_ops() -> set[str]:
    """The ``GGML_UNARY_OP_*`` the unary sub-switch differentiates (the rest hit its abort)."""
    body = _balanced_body(GGML_C.read_text(), r"static void ggml_compute_backward\s*\(")
    unary = body[body.index("switch (ggml_get_unary_op(tensor))") :]
    unary = unary[: unary.index("default:")]
    return _case_labels(unary, "GGML_UNARY_OP")


def glu_enum_variants() -> set[str]:
    """Every ``GGML_GLU_OP_*`` the enum defines (minus the ``COUNT`` sentinel)."""
    return set(re.findall(r"GGML_GLU_OP_(\w+),", GGML_H.read_text())) - {"COUNT"}


# --- the claim: what the preflight table says --------------------------------------------------


def table_ops() -> set[str]:
    body = _balanced_body(PREFLIGHT_CPP.read_text(), r"bool op_has_backward\s*\(")
    return _case_labels(body, "GGML_OP")


def table_unary_ops() -> set[str]:
    body = _balanced_body(PREFLIGHT_CPP.read_text(), r"bool unary_has_backward\s*\(")
    return _case_labels(body, "GGML_UNARY_OP")


def table_glu_ops() -> set[str]:
    body = _balanced_body(PREFLIGHT_CPP.read_text(), r"bool glu_has_backward\s*\(")
    return _case_labels(body, "GGML_GLU_OP")


def reconcile(
    truth: set[str], claimed: set[str], allowed_extra: frozenset[str]
) -> tuple[set[str], set[str]]:
    """Compare a claimed backward set against the ground truth.

    Returns ``(missing, extra)``:

    * ``missing`` — ops the truth differentiates that the table denies (blocks a trainable model);
    * ``extra`` — ops the table claims that the truth lacks, minus the documented backward-only ops
      (promises a run that then aborts).

    Both empty means the mirror is faithful. This is the single comparison the real checks and the
    mutation test below share, so the mutation test proves the checks themselves can fail.
    """
    missing = truth - claimed
    extra = claimed - truth - allowed_extra
    return missing, extra


# ---------------------------------------------------------------------------
# The guard.
# ---------------------------------------------------------------------------


def test_op_table_mirrors_ggml_backward_switch() -> None:
    """``op_has_backward`` must claim a backward for exactly the ops ggml differentiates."""
    missing, extra = reconcile(ggml_backward_ops(), table_ops(), EMITTED_ONLY_OPS)

    assert not missing, (
        f"ggml_compute_backward now differentiates {sorted(missing)}, but op_has_backward "
        f"(csrc/farm_preflight.cpp) still denies it — the preflight will report a trainable model "
        f"as blocked. Add these ops to op_has_backward (and drop any stale blocker_detail entry)."
    )
    assert not extra, (
        f"op_has_backward claims a backward for {sorted(extra)}, but ggml_compute_backward has no "
        f"case for it — the preflight will pass a model that then aborts in the backward pass. "
        f"Either ggml lost the case (remove these) or they are new backward-only emitted ops (add "
        f"them to EMITTED_ONLY_OPS with a note on why they never reach a forward graph)."
    )


def test_unary_table_mirrors_ggml_backward_switch() -> None:
    """``unary_has_backward`` must mirror the unary sub-switch, whose ``default:`` also aborts."""
    missing, extra = reconcile(ggml_backward_unary_ops(), table_unary_ops(), frozenset())

    assert not missing, (
        f"the ggml unary backward sub-switch now handles {sorted(missing)}, which "
        f"unary_has_backward still denies — a model using that activation reports untrainable."
    )
    assert not extra, (
        f"unary_has_backward claims {sorted(extra)}, which the ggml unary sub-switch drops to its "
        f"aborting default — the preflight would pass a model that then aborts on that activation."
    )


def test_glu_table_covers_every_glu_variant() -> None:
    """``glu_has_backward`` must list every GLU variant the enum defines.

    Unlike the other two switches, ggml's ``GGML_OP_GLU`` backward has no per-variant abort: since
    S1-28 it routes the whole family through ``GLU_BACK``. So the meaningful invariant is that the
    table keeps pace with the *enum* — a variant added upstream that the table does not list would
    be reported blocked (and, worse, routed into a ``ggml_glu_back`` kernel that may not handle it).
    """
    assert table_glu_ops() == glu_enum_variants(), (
        "glu_has_backward (csrc/farm_preflight.cpp) has drifted from the GGML_GLU_OP enum in "
        "ggml.h. If a variant was added upstream, confirm ggml_glu_back handles it, then list it; "
        "if one was removed, drop it."
    )


def test_the_freshness_guard_can_actually_detect_drift() -> None:
    """The guard is only worth having if it fails when the table lies — so make it lie, both ways.

    The analogue of ``test_backend_ops_grad.test_the_vacuity_guard_can_actually_detect_vacuity``:
    that test points its guard at ops it knows check nothing; this one points :func:`reconcile` at a
    table it knows has drifted and asserts each direction is caught. If this ever passes while the
    two assertions below cannot be provoked, the guard has gone blind and the table is on trust.
    """
    truth = ggml_backward_ops()
    table = table_ops()

    # Precondition: the real, unmutated state is clean. (Also the load-bearing assertion of the
    # first test — restated here so a regression that made reconcile always-empty is caught twice.)
    assert reconcile(truth, table, EMITTED_ONLY_OPS) == (set(), set())

    # Conservative-direction drift: the table drops an op ggml still differentiates.
    dropped = sorted(truth)[0]
    missing, extra = reconcile(truth, table - {dropped}, EMITTED_ONLY_OPS)
    assert dropped in missing and not extra, "reconcile failed to flag an op the table dropped"

    # Dangerous-direction drift: the table claims a backward ggml does not have.
    missing, extra = reconcile(truth, table | {"TOTALLY_MADE_UP_OP"}, EMITTED_ONLY_OPS)
    assert "TOTALLY_MADE_UP_OP" in extra and not missing, "reconcile failed to flag a bogus claim"

    # The emitted-only allowance is a named list, not a blanket pardon: an undocumented extra op
    # still fails even though real emitted-only ops are forgiven.
    _, extra = reconcile(truth, table | {"OUT_PROD_ID_MADE_UP"}, EMITTED_ONLY_OPS)
    assert "OUT_PROD_ID_MADE_UP" in extra, "the emitted-only allowlist swallowed an undocumented op"


def test_the_parsers_find_something() -> None:
    """A cross-parse that silently matched nothing would pass every check above vacuously.

    If a vendor bump renamed the function or changed the case style, the regexes could return empty
    sets and ``reconcile(set(), set())`` would be clean — a green that checked nothing, the exact
    failure mode ``tests.project_ops`` and the ``grad -o`` guards were built to refuse. So assert
    the parse is non-trivial and anchored on ops that are not going anywhere.
    """
    assert {"MUL_MAT", "CROSS_ENTROPY_LOSS", "RMS_NORM"} <= ggml_backward_ops()
    assert {"MUL_MAT", "CROSS_ENTROPY_LOSS", "RMS_NORM"} <= table_ops()
    assert {"SILU", "TANH"} <= ggml_backward_unary_ops()
    assert {"SILU", "TANH"} <= table_unary_ops()
    assert "SWIGLU" in glu_enum_variants()
    assert EMITTED_ONLY_OPS <= table_ops(), "the emitted-only ops should really be in the table"
