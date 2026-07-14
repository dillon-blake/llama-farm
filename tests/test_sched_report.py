"""The GPU-vs-CPU fallback report (S2-01).

ROADMAP §11: a backend lane may run its kernel gaps on the CPU via ``ggml_backend_sched`` — that is
how training works on a backend before its kernels land — but it must **say so**. Acceptable but
reported, never hidden. So a fallback does not fail the build; a run with **no report** does.

The parser is tested against the real printer format (``ggml-backend.cpp:945``) rather than a
convenient one, because the format has a trap in it: the backend field is ``%5.5s``.
"""

from __future__ import annotations

import importlib.util
import pathlib

import pytest

_SCRIPT = pathlib.Path(__file__).resolve().parents[1] / ".github/scripts/sched_fallback_report.py"
_spec = importlib.util.spec_from_file_location("sched_fallback_report", _SCRIPT)
assert _spec and _spec.loader
report = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(report)


# Lifted from the real printer's format string:
#   node #%3d (%10.10s): %20.20s (%5.5s) [%5.5s %8.8s] use=%d,c=%d:
SAMPLE = """
## SPLIT #0: MTL0 # 2 inputs: [inp_tokens (   1K)] [kq_mask (  16M)]
node #  0 (   GET_ROWS):             inp_embd (   2M) [ MTL0 ALLOC   ] use=1,c=0:
node #  1 (   RMS_NORM):                 norm (   2M) [ MTL0 ALLOC   ] use=1,c=0:
node #  2 (    MUL_MAT):                   kq (   4M) [ MTL0 ALLOC   ] use=1,c=0:
node #  3 (   SOFT_MAX):          kq_soft_max (   4M) [ MTL0 ALLOC   ] use=1,c=0:
node #  4 (   OUT_PROD):          lora_a_grad (  32K) [  CPU SUPPORT ] use=1,c=0:
node #  5 (   OUT_PROD):          lora_b_grad (  32K) [  CPU SUPPORT ] use=1,c=0:
node #  6 (SOFT_MAX_BACK):          d_soft_max (   4M) [  CPU SUPPORT ] use=1,c=0:
"""


def test_it_counts_gpu_and_cpu_nodes() -> None:
    on_gpu, on_cpu = report.parse(SAMPLE.splitlines(), "MTL0")

    assert on_gpu == {"GET_ROWS": 1, "RMS_NORM": 1, "MUL_MAT": 1, "SOFT_MAX": 1}
    assert on_cpu == {"OUT_PROD": 2, "SOFT_MAX_BACK": 1}


def test_the_backend_name_is_truncated_to_five_characters() -> None:
    """``%5.5s``. ``Vulkan0`` prints as ``Vulka``, and matching the full name finds NOTHING.

    A parser that compared against ``Vulkan0`` would report 0% GPU on a perfectly healthy run, and
    the honest-looking conclusion would be "the Vulkan lane is doing nothing on the GPU".
    """
    lines = ["node #  0 (    MUL_MAT):                   kq (   4M) [Vulka ALLOC   ] use=1,c=0:"]

    on_gpu, on_cpu = report.parse(lines, "Vulkan0")
    assert on_gpu == {"MUL_MAT": 1}
    assert on_cpu == {}


def test_split_headers_and_prose_are_ignored() -> None:
    """Only ``node #`` lines count. The SPLIT banners name a backend too, and are not nodes."""
    on_gpu, on_cpu = report.parse(
        ["## SPLIT #0: MTL0 # 2 inputs: [x (1K)]", "some other log line", ""], "MTL0"
    )
    assert on_gpu == {} and on_cpu == {}


def test_a_run_with_no_nodes_is_an_error_not_a_clean_bill_of_health() -> None:
    """The one failure mode that matters: no report at all.

    An empty parse must NOT render as "nothing fell back to the CPU" — that reads as success and is
    the exact opposite of what happened.
    """
    on_gpu, on_cpu = report.parse(["nothing here"], "MTL0")
    rendered = report.render(on_gpu, on_cpu, "MTL0")

    assert "no nodes seen" in rendered
    assert "should be red" in rendered


@pytest.mark.parametrize("gpu", ["MTL0", "CUDA0", "Vulkan0"])
def test_the_report_names_every_op_that_fell_back(gpu: str) -> None:
    """Each fallback is a kernel that backend still owes, so the report has to name it."""
    lines = [
        f"node #  0 (    MUL_MAT):   kq (   4M) [{gpu[:5]:>5.5} ALLOC   ] use=1,c=0:",
        "node #  1 (   OUT_PROD):    g (  32K) [  CPU SUPPORT ] use=1,c=0:",
    ]
    on_gpu, on_cpu = report.parse(lines, gpu)
    rendered = report.render(on_gpu, on_cpu, gpu)

    assert "OUT_PROD" in rendered
    assert "Fell back to the CPU" in rendered
    assert "50.0%" in rendered
