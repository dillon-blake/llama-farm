# Provenance policy

learning-llamas is MIT-licensed and is built on top of two external codebases with very
different licensing regimes. This document is the rule every pull request follows, and
reviewers block any diff that violates it.

The short version:

| Source | License | What we may take |
|---|---|---|
| llama.cpp / ggml | MIT | **Code**, copied or adapted, with a provenance header |
| unsloth | Apache-2.0 with AGPL/LGPL carve-outs | **Math and design only — never code, from any file** |

## 1. llama.cpp — copied code is allowed, headers are mandatory

learning-llamas vendors llama.cpp (`vendor/llama.cpp`, pinned commit
`4f37f519722aa3242eecb7649466b4a4a2d6d6da`) and copies or adapts code from it in two
places: the C shim in `csrc/`, and project-authored ggml kernels that land in the
project's llama.cpp fork. Both are MIT-to-MIT, which is exactly why learning-llamas chose MIT:
kernels we write against ggml can be upstreamed with no license friction.

Every file containing copied or adapted llama.cpp code carries this header, immediately
after any existing copyright notice, which must be retained:

```c
// -----------------------------------------------------------------------------
// Adapted from llama.cpp
//   source:  src/llama-context.cpp  (llama_context::opt_epoch_iter)
//   commit:  4f37f519722aa3242eecb7649466b4a4a2d6d6da
//   license: MIT (see vendor/llama.cpp/LICENSE)
//   changes: forked the per-ubatch loop to accept a pluggable loss callback and
//            extra graph inputs; removed the hardcoded cross-entropy loss.
// -----------------------------------------------------------------------------
```

The Python equivalent is the same four fields in the module docstring:

```python
"""...

Adapted from llama.cpp:
    source:  gguf-py/tests/test_quants.py
    commit:  4f37f519722aa3242eecb7649466b4a4a2d6d6da
    license: MIT (see vendor/llama.cpp/LICENSE)
    changes: kept only the ctypes declaration of ggml_quantize_chunk.
"""
```

`changes:` is not optional and is not a formality — it is what a reviewer reads to decide
whether a rebase onto a newer llama.cpp needs to revisit the file.

Mirroring a *declaration* (a struct layout or a function signature, as `src/learning_llamas/_ffi/`
does) is interface use, not code copying, and does not require the full header. It still
records the source header path and pinned commit in the module docstring, because a vendor
bump that changes a struct layout silently corrupts memory and the trail is how we find it.

## 2. unsloth — ideas only, and the exclusion list is not obvious

unsloth is a design and mathematics reference. **No unsloth code is copied into this
repository, from any file, under any license.** Its published approaches (the chunked
attention backward formulation, the sparse cross-entropy gradient derivation, the
max-abs ≤ 0.05 fp16 gradient tolerance) are prior art we reimplement from first principles.

This rule is absolute rather than per-file because the license boundary inside the unsloth
tree is genuinely hard to see. The repository's top-level license is Apache-2.0, but the
LICENSE file carries a carve-out (`LICENSE:190-191`) and the root `COPYING` is the AGPLv3
text. The paths below are **not** Apache-2.0 and code must never be copied from them:

**AGPLv3:**

- `unsloth/kernels/moe/**`
- `studio/**`
- `unsloth_cli/**`
- `cli.py`
- `unsloth/models/rl_replacements.py:1191` — a *function-level* AGPL marker inside an
  otherwise-Apache file. This is the one a naive grep for a file-level license header
  misses entirely.

**LGPL-3.0-or-later:**

- `unsloth/utils/packing.py`
- `unsloth/utils/attention_dispatch.py`
- `unsloth/utils/__init__.py`
- `unsloth/kernels/rope_embedding.py`

Even for the Apache-2.0 remainder, copying is not permitted under this policy — Apache-2.0
code in an MIT project is a compliance obligation (attribution, NOTICE, patent grant) that
buys nothing, since everything we want from unsloth is mathematics that we can and do
rederive. See `docs/KERNEL-ROADMAP.md` §13 for the full license map.

## 3. What reviewers enforce

A pull request is blocked if:

- it adds a file with copied or adapted llama.cpp code and no provenance header;
- a provenance header's `commit:` does not match the pinned vendor commit, without an
  explanation of why the file tracks a different commit;
- any diff hunk originates from an unsloth file (the correct move is to describe the
  mathematics in the PR and reimplement it);
- it removes an existing upstream copyright notice.

New copied code from any *other* third-party project requires the same header plus a NOTICE
entry, and a license compatible with MIT redistribution. When in doubt, reimplement.
