# Developer documentation

| Document | What it covers |
|---|---|
| [`building.md`](building.md) | Prerequisites, the editable install, per-backend CMake flags, building the vendored test binaries |
| [`testing.md`](testing.md) | The pytest suite and its markers; `test-backend-ops` including MODE_GRAD; where the tolerances come from |
| [`vm-playbooks.md`](vm-playbooks.md) | Copy-pasteable, idempotent provisioning for each backend's VM, plus self-hosted GitHub Actions runners |
| [`backward-coverage.md`](backward-coverage.md) | **Measured** MODE_GRAD coverage per op — which gradients are actually checked today, which abort, and which are wrong |

Two architecture decisions bind every kernel PR. Read them before writing one:

- [`../adr/ADR-0001-vendor-lineage.md`](../adr/ADR-0001-vendor-lineage.md) — the pinned llama.cpp
  commit, the rebase cadence, and the two-repo PR flow.
- [`../adr/ADR-0002-numerics-determinism-parity.md`](../adr/ADR-0002-numerics-determinism-parity.md)
  — F32 gradient accumulation, determinism by default, and the parity criterion your kernel is
  measured against.

The work itself is planned in [`../../tickets/`](../../tickets/); start with
[`tickets/README.md`](../../tickets/README.md).
