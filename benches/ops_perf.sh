#!/usr/bin/env bash
# Per-op timings for the ops a training step is actually made of (S1-32).
#
# The training-critical set is short and the reason each one is in it is worth stating:
#
#   OUT_PROD    every MUL_MAT gradient goes through it, and it DEQUANTIZES the base weight on
#               every call -- which is why the backward's share of a step RISES as the base gets
#               more compressed. It is the single most important number here.
#   MUL_MAT     the forward, for comparison.
#   SOFT_MAX    attention's normalization, and its backward.
#   RMS_NORM    the other normalization in the graph.
#   CROSS_ENTROPY_LOSS_SPARSE   the loss (S1-04).
set -euo pipefail

BIN="${1:-build/vendor-tests/bin/test-backend-ops}"

for op in OUT_PROD MUL_MAT SOFT_MAX RMS_NORM CROSS_ENTROPY_LOSS_SPARSE; do
    echo "=== $op"
    "$BIN" perf -o "$op" -b CPU 2>/dev/null | grep -E "GFLOPS|GB/s|runs" | head -6
done
