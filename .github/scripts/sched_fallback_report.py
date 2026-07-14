#!/usr/bin/env python3
"""Which ops actually ran on the GPU, and which quietly fell back to the CPU.

ROADMAP §11's scheduler note is the reason this exists. Until a backend's kernels are complete,
``ggml_backend_sched`` transparently runs the gaps on the CPU — so training *works* on every
backend from day one, at reduced speed. That is the right behaviour, and it is also a way to ship a
green GPU lane that is doing almost nothing on the GPU.

So the rule is **acceptable but reported, never hidden**: a fallback does not fail the build, but a
run with *no report* does.

Reads ``GGML_SCHED_DEBUG=2`` output (``ggml-backend.cpp:1740``; printer at ``:945``), whose
per-node line is::

    node #  7 (   MUL_MAT):            kq (  4M) [ MTL0 ALLOC   ] use=1,c=0:  ...
             ^op                   ^name  ^size    ^backend ^cause

Two details that bite:

* The backend field is ``%5.5s`` — **truncated to five characters**. ``Vulkan0`` prints as
  ``Vulka``. Matching it against the full device name finds nothing.
* View ops are skipped by the printer entirely, so they never appear and must not be counted as
  missing.

Usage::

    GGML_SCHED_DEBUG=2 <command> 2>&1 | sched_fallback_report.py --gpu MTL0 --out report.md
"""

from __future__ import annotations

import argparse
import collections
import json
import pathlib
import re
import sys

# node #%3d (%10.10s): %20.20s (%5.5s) [%5.5s %8.8s] use=%d,c=%d:
_NODE = re.compile(
    r"^node #\s*(?P<idx>\d+)\s+\(\s*(?P<op>[^)]+?)\s*\):\s+"
    r"(?P<name>\S+)\s+\(\s*\S+\s*\)\s+\[\s*(?P<backend>\S+)\s+(?P<cause>\S*)\s*\]"
)


def parse(lines: list[str], gpu: str) -> tuple[dict[str, int], dict[str, int]]:
    """Count nodes per op that ran on the GPU, and per op that fell back to the CPU.

    Args:
        lines: ``GGML_SCHED_DEBUG=2`` output. Anything that is not a ``node #`` line is ignored.
        gpu: The GPU device's name as ggml registered it (``MTL0``, ``CUDA0``, ``Vulkan0``).
            Compared on its first five characters, because the printer truncates.

    Returns:
        A ``(on_gpu, on_cpu)`` pair of op-name -> node-count maps.
    """
    # The printer's field is %5.5s, so compare on what actually gets printed.
    gpu_tag = gpu[:5]

    on_gpu: dict[str, int] = collections.Counter()
    on_cpu: dict[str, int] = collections.Counter()

    for line in lines:
        m = _NODE.match(line.strip())
        if not m:
            continue
        op = m.group("op")
        if m.group("backend") == gpu_tag:
            on_gpu[op] += 1
        else:
            on_cpu[op] += 1

    return dict(on_gpu), dict(on_cpu)


def render(on_gpu: dict[str, int], on_cpu: dict[str, int], gpu: str) -> str:
    """Render the report as markdown for the job summary."""
    total = sum(on_gpu.values()) + sum(on_cpu.values())
    if total == 0:
        return (
            "## ⚠️ Scheduler report: **no nodes seen**\n\n"
            "`GGML_SCHED_DEBUG=2` produced no `node #` lines. Either the step never ran, or the "
            "debug output was not captured. This report is worthless and the job should be red.\n"
        )

    pct = 100.0 * sum(on_gpu.values()) / total
    fell_back = sorted(set(on_cpu) - set(on_gpu)) + sorted(set(on_cpu) & set(on_gpu))

    out = [
        f"## Scheduler report — `{gpu}`",
        "",
        f"**{sum(on_gpu.values())}/{total} nodes ({pct:.1f}%) ran on {gpu}.**",
        "",
    ]

    if not on_cpu:
        out += ["Nothing fell back to the CPU.", ""]
    else:
        out += [
            f"### Fell back to the CPU ({sum(on_cpu.values())} nodes)",
            "",
            "Acceptable — `ggml_backend_sched` runs the gaps on the CPU so training works before "
            "the kernels land — but it is **reported, never hidden**. Each of these is a kernel "
            "this backend still owes.",
            "",
            "| op | nodes on CPU | nodes on GPU |",
            "|---|---:|---:|",
        ]
        for op in fell_back:
            out.append(f"| `{op}` | {on_cpu.get(op, 0)} | {on_gpu.get(op, 0)} |")
        out.append("")

    return "\n".join(out)


def main() -> int:
    """Parse stdin, write the report, and fail only if there is no report at all."""
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--gpu", required=True, help="ggml's name for the GPU device (e.g. MTL0)")
    ap.add_argument("--out", type=pathlib.Path, help="write the markdown report here")
    ap.add_argument("--json", type=pathlib.Path, help="write the raw counts here")
    args = ap.parse_args()

    on_gpu, on_cpu = parse(sys.stdin.read().splitlines(), args.gpu)
    report = render(on_gpu, on_cpu, args.gpu)

    if args.out:
        args.out.write_text(report)
    if args.json:
        args.json.write_text(
            json.dumps({"gpu": args.gpu, "on_gpu": on_gpu, "on_cpu": on_cpu}, indent=2)
        )

    print(report)

    # A run with no report is a failure; a run WITH fallbacks is not.
    return 1 if not on_gpu and not on_cpu else 0


if __name__ == "__main__":
    raise SystemExit(main())
