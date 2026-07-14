"""Same host, same inputs, different thread count — the curve must be bit-identical (S1-38).

ADR-0002's determinism story has two halves. Across *hosts*, a different SIMD width changes the
float32 reduction order and the curve genuinely moves; that cannot be pinned from one box, and it
is what ``CURVE_TOL``'s ~190x margin exists for. Across *thread counts on one host*, the curve was
measured bit-identical at 1, 2 and 4 threads — a claim ``tests/convergence/README.md`` has stated
since S1-12 and, until now, nothing asserted.

If this goes red, either a reduction's result became dependent on the thread split (a real
portability regression — the Metal/CUDA/Vulkan lanes inherit this exact claim on *their* hosts) or
nondeterminism crept into collation. Neither is a tolerance problem, so there is no tolerance.
"""

from __future__ import annotations

from .convergence import config, harness


def test_the_curve_is_bit_identical_across_thread_counts(
    tiny_f32, tmp_path, libs, conv_data
) -> None:
    """Sixteen optimizer steps at 1, 2 and 4 threads: exactly equal, not approximately."""
    spec = config.RunSpec(epochs=2)

    curves = {}
    for n_threads in (1, 2, 4):
        adapter = tmp_path / f"adapter-{n_threads}.gguf"
        curves[n_threads] = harness.train(
            libs, tiny_f32, adapter, conv_data, spec=spec, n_threads=n_threads
        )

    assert curves[1] == curves[2] == curves[4], (
        "the loss curve depends on the thread count. A reduction's result now depends on how the "
        "work was split, which breaks ADR-0002's same-host determinism claim and will make every "
        "backend lane's tolerances meaningless."
    )
