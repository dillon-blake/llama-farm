# Contributing to learning-llamas

## Workflow: one ticket, one pull request

All work is planned as tickets under [`tickets/`](tickets/). Read
[`tickets/README.md`](tickets/README.md) first — it is the authoritative guide to stage
ordering, the claim protocol, the definition of done, and the CI model. In short:

1. Pick a ticket whose `deps` are all `status: done`.
2. Claim it by setting `status: in-progress` in the ticket's frontmatter.
3. Branch `ticket/<id>-<slug>`, implement exactly the ticket's "What to do".
4. Run the ticket's "Testing & verification" section locally before opening the PR.
5. Open one PR titled `[<id>] <title>`; record the URL in the ticket's `pr:` field.

Tickets that change vendored llama.cpp code (track `kernels`) follow the **two-repo flow**:
the real PR goes against the project fork's `learning-llamas-base` branch, and a trivial PR
here bumps the `vendor/llama.cpp` submodule. See
[`docs/adr/ADR-0001-vendor-lineage.md`](docs/adr/ADR-0001-vendor-lineage.md).

## Commit format

```
<type>(<ticket-id>): <imperative summary>
```

`type` is one of `feat`, `fix`, `docs`, `test`, `build`, `ci`, `refactor`, `chore`.
Example: `feat(S0-03): build vendored llama.cpp and the liblearningllamas shim`.

## Licensing and provenance

learning-llamas is MIT. Before copying or adapting any code, read
[`docs/PROVENANCE.md`](docs/PROVENANCE.md). The two rules that catch people out:

- llama.cpp code may be copied, but **only with a provenance header** (source path, commit,
  license, summary of changes).
- unsloth is a **design and mathematics reference only** — no code from any unsloth file,
  under any license.

## Binding decisions

Architecture decisions that constrain later tickets live in [`docs/adr/`](docs/adr/). Read
them before writing kernels; in particular ADR-0002 fixes the numerics, determinism, and
cross-backend parity policy that every kernel PR is measured against.

## Local setup

See [`docs/dev/building.md`](docs/dev/building.md) for the build guide and
[`docs/dev/testing.md`](docs/dev/testing.md) for how to run the suites. Style is enforced by
pre-commit:

```bash
pip install pre-commit
pre-commit install
pre-commit run --all-files
```

Python is formatted and linted with `ruff`; C and C++ with `clang-format` (LLVM base,
4-space indent, matching llama.cpp's conventions).
