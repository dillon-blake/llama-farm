# CPU throughput — what one LoRA train step costs (S1-32, partial)

This records the numbers `benches/cpu_train_step.py` produces today on a target-class box. It is
**not** the full S1-32 audit: the sizing worksheet, JSON report, per-op microbenchmarks, the
`examples/training/finetune` baseline, the multi-CPU-class tables and the nightly CI wiring are all
deferred to backlog ticket **B-11**. What is here is the cheap half — one bench run on the hardware
class the CI lane and a GPU-less user actually have — captured so the number exists with its
assumptions attached.

## Measured

- **Machine:** Intel N100, 4 physical cores, AVX2 (no AVX-512), Linux x86_64.
- **Base:** `tests/.fixtures/*/tiny-llama-q4_k.gguf` — the tiny CI fixture, *not* a realistic ~1B
  model. Treat absolute tok/s as a plumbing data point, not a sizing figure.
- **Config:** `n_ubatch=128`, LoRA `rank=8`, 20 timed steps, 4 warmup.
- **Command:** `.venv/bin/python benches/cpu_train_step.py --threads 1,2,4 --steps 20 --warmup 4`

| threads | tok/s | step (ms) | fwd (ms) | bwd share | scaling |
|--------:|------:|----------:|---------:|----------:|--------:|
| 1 | 832 | 153.88 | 39.96 | 74% | 1.00x |
| 2 | 819 | 156.19 | 39.80 | 75% | 0.99x |
| 4 | 845 | 151.47 | 39.83 | 74% | 1.02x |

## Two honest reads

1. **Thread scaling is flat (~1.0x) on this fixture — and that is a property of the fixture, not the
   threadpool.** The tiny model's matmuls are too small for the ggml threadpool to amortize its
   per-op fork/join over, so 1, 2 and 4 threads all land within noise. The *shape* of the
   scaling-cliff question S1-32 wants answered (where does adding threads stop helping?) needs a
   realistic ~1B Q4_K base to be meaningful; on the tiny fixture the cliff is at one thread. B-11
   adds that model.

2. **The backward is ~74% of the step**, and stays there across thread counts. This matches the
   ticket's stated expectation: the backward is dominated by `OUT_PROD`, which dequantizes the Q4_K
   base weight on every call, so the more compressed the base the larger the backward's share. The
   forward/backward split is measured (a forward-only `evaluate` timed against a full `step`), not
   assumed.

## What is deferred (B-11)

Everything in S1-32's acceptance criteria beyond "the bench runs and produces a number": the JSON
machine-readable report, `src/learning_llamas/sizing.py` (the BLUEPRINT §10 memory worksheet) and its
unit tests, the per-op microbenchmark runner, the thread-scaling analysis on a realistic model, the
`examples/training/finetune` baseline comparison, tok/s tables for a second (arm64) CPU class, and the
report-only nightly `ci-cpu` job. See `tickets/backlog/B-11-cpu-throughput-audit.md`.
