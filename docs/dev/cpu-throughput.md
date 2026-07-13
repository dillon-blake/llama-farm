# CPU training throughput (S1-32)

"Can I fine-tune on this machine?" has an answer. The honest form of it is a number with the
assumptions attached.

Run it yourself:

```
python benches/cpu_train_step.py --threads 1,2,4 --n-ubatch 512
```

## Measured

Intel N100 (4 cores, 6W, the class of machine this project exists for), tiny-llama fixture
(2 layers, n_embd 256, Q4_K base), LoRA rank 8, mmap on:

| threads | tok/s | step (ms) | forward (ms) | **backward's share** | scaling |
|--------:|------:|----------:|-------------:|---------------------:|--------:|
| 1 | 3205 | 159.8 | 44.5 | **72%** | 1.00× |
| 2 | 3014 | 169.9 | 47.1 | **72%** | 0.94× |
| 4 | 3346 | 153.0 | 48.5 | **68%** | 1.04× |
| 8 | 3167 | 161.6 | 55.8 | **65%** | 0.99× |

*(n_ubatch 512. At n_ubatch 128 the same machine does ~4670 tok/s single-threaded, with the same
~73% backward share — the shape of the result does not depend on the batch.)*

## Two findings, and neither is what a single tok/s number would tell you

### The backward is roughly 70% of the step, not 50%

The textbook figure for backprop is "about twice the forward". Here it is closer to **2.5×**, and the
reason is specific to training a **quantized** base.

Every `MUL_MAT` gradient goes through `OUT_PROD`, and `OUT_PROD` **dequantizes the base weight on
every call**. The forward gets to use the quantized kernel — Q4_K weights, Q8_K activations,
integer dot products. The backward does not: it materializes the weight in F32 first.

So the backward's share **rises as the base gets more compressed**. Quantizing harder makes the model
smaller and the *forward* faster, and it makes the backward's relative cost worse. That is not
obvious, and it is the number to watch when sizing a run.

### Threads buy nothing on this machine

0.94× to 1.04× across 1→8 threads. Not a defect in the threadpool — the workload is
**memory-bandwidth-bound**, and an N100's four cores share one modest memory controller. Dequantizing
Q4_K weights on every `OUT_PROD` is exactly the kind of work that saturates bandwidth long before it
saturates ALUs, and more threads do not add bandwidth.

**Practical consequence:** on a small CPU device, do not spend threads on this. Spend them on
something else, or spend the power budget on nothing at all. On a machine with real memory bandwidth
(a desktop, a server) the sweep will look different, which is exactly why it is a sweep and not a
constant.

## What this does *not* measure

The fixture is a 2-layer toy. It measures the **shape** of the cost — the forward/backward split, the
threading response — faithfully, because those are properties of the ops and the memory system rather
than of the model's size. It does not give you a tok/s figure for a 1B model, and no amount of
extrapolating from a 256-dim model would.

For that, point `--model` at the real thing. The bench takes any GGUF.

## Per-op numbers

`benches/ops_perf.sh` drives the vendored `test-backend-ops perf` over the training-critical ops
(`OUT_PROD`, `MUL_MAT`, `SOFT_MAX`, `RMS_NORM`, `ce_sparse`).

It is **not** wired into CI, and that is deliberate: the full `perf` sweep for a single op takes
longer than ten minutes on an N100. It is a thing you run when you are optimizing a kernel, on a
machine that can afford it.
