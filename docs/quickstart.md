# Quickstart

Fine-tune a quantized GGUF on your own data, on a CPU, and get back an adapter that stock
`llama-cli` can load. The base model stays quantized and memory-mapped throughout — nothing here
ever dequantizes it, writes to it, or needs it to fit in memory twice.

## Install

```bash
git clone --recurse-submodules https://github.com/dillon-blake/llama-farm.git
cd llama-farm

pip install scikit-build-core cmake ninja pytest numpy
pip install -e . --no-build-isolation        # builds vendored llama.cpp + the C shim
pip install -e vendor/llama.cpp/gguf-py
```

You will also need a base model in GGUF form. Any quantization works — Q4_K included, which is the
whole point.

## Supervised fine-tuning, end to end

Four steps, and the shape never changes: **write a zero adapter, train through it, save it, load
it with llama.cpp.**

```python
import json

from learning_llamas import (
    Message, Model, build_masked_sample,
    create_zero_adapter, libraries, read_adapter, save_adapter,
)
from learning_llamas.train import SFTConfig, train_sft

BASE = "models/qwen2.5-0.5b-instruct-q4_k_m.gguf"
SEQ_LEN = 512

libs = libraries()

# 1. A zero-initialized adapter: A ~ N(0, 1/sqrt(r)), B = 0. It is a provable no-op at step 0,
#    so attaching it cannot change the model's output until the first gradient lands.
create_zero_adapter(BASE, "adapter.gguf", r=16)

# 2. Open the model in TRAINING mode and attach the adapter. Both flags matter -- see below.
with Model(BASE, n_ctx=SEQ_LEN, n_ubatch=SEQ_LEN, training=True) as model:
    model.attach_adapter("adapter.gguf", scale=1.0)

    # 3. Your data. One JSON object per line: {"prompt": ..., "completion": ...}
    #    The template comes out of the GGUF, not out of a config file we guessed at. A *base*
    #    model embeds none -- that raises, and you pass ChatTemplate("...") explicitly instead.
    template = model.chat_template()
    samples = []
    for line in open("data.jsonl"):
        row = json.loads(line)
        samples.append(
            build_masked_sample(
                [Message("user", row["prompt"]), Message("assistant", row["completion"])],
                template,
                model.tokenizer,
            )
        )

    result = train_sft(
        libs,
        model,
        samples,
        SFTConfig(
            lr=1e-4,
            seq_len=SEQ_LEN,
            pad_id=model.tokenizer.eos,
            epochs=3,
            grad_accum=4,       # 4 batches per optimizer step
            grad_clip=1.0,      # global norm over every trainable tensor
            schedule="cosine",
            warmup_steps=10,
        ),
    )

    print("loss:", result.steps[0].loss, "->", result.steps[-1].loss)

    # 4. Write the trained A/B tensors back out as an adapter GGUF.
    info = read_adapter("adapter.gguf")
    save_adapter(libs, model.adapter, "trained.gguf",
                 architecture=info.architecture, alpha=info.alpha)
```

That is the whole loop. `trained.gguf` is a standard LoRA adapter GGUF:

```bash
llama-cli -m models/qwen2.5-0.5b-instruct-q4_k_m.gguf --lora trained.gguf -p "..."
```

No merge step is required — llama.cpp applies the adapter at load time. If you *want* a single
standalone file (to ship one artifact, or to serve it somewhere that cannot take a `--lora` flag),
fold it in:

```python
from learning_llamas import merge

merge(BASE, "trained.gguf", "merged.gguf", libs)     # re-quantizes the touched tensors
```

Only the tensors the adapter touched are rewritten, and they are re-quantized back to the base's
own type — so a Q4_K base merges to a Q4_K model, not to an F16 one four times the size.

## The three things that will bite you

**`training=True` is not a hint.** It disables ggml-cpu's *extra buffer types*, which repack
quantized weights to speed up `MUL_MAT`. A repacked weight makes its own gradient node
unschedulable, and `ggml_backend_sched` aborts naming neither the op nor the tensor. It bites Q4_K
and not Q8_0, purely because a q4_K repack variant happens to exist. Leave the flag off and you get
an abort with no useful message.

**Every batch must be exactly `seq_len` long.** ggml-opt sizes its optimizer state from the first
graph it sees and indexes it by node index forever, so a batch of a different shape is rejected
with `LL_ERR_SHAPE_MISMATCH` rather than silently retraining against the wrong moments. `train_sft`
pads for you — a pad token carries weight 0 and contributes exactly nothing — but a *sample* longer
than `seq_len` is an error, not a silent truncation.

**If you run two contexts, share one adapter object.** Attaching the same adapter *file* to two
contexts loads it twice and gives you two independent adapters that start out equal and diverge on
the first step. There is no error and no NaN — just a model that never changes. Pass the handle:

```python
handle = policy.attach_adapter("adapter.gguf")
sampler.attach_adapter(adapter=handle)          # the same A/B tensors, not a copy
```

## Reinforcement learning (GRPO)

Same adapter, same base, no value network — the group of rollouts is its own baseline.

```python
from learning_llamas import Model, create_zero_adapter, libraries
from learning_llamas.train import (
    GRPOConfig, RolloutEngine, SamplerConfig, substring_reward, train_grpo,
)

libs = libraries()
create_zero_adapter(BASE, "adapter.gguf", r=16)

PROMPTS = ["What is 2+2?", "Name a colour."]
G = 8            # rollouts per prompt
SEQ_LEN = 128    # tokens per rollout

# A training step packs EVERY rollout as its own sequence and pushes them through in ONE forward
# pass -- the shim will not split it, because ggml-opt keys its optimizer state to the graph. So
# the training context has to hold all of them at once, and this product is what sizes GRPO's
# memory. Raising the number of prompts per round is not free; batch them in small chunks.
N_SEQ = len(PROMPTS) * G      # 16 sequences
N_TOK = N_SEQ * SEQ_LEN       # 2048 tokens in a single ubatch

# Generation and training need SEPARATE contexts -- a llama_decode on a training context frees the
# scheduler the optimizer is holding -- but ONE adapter, so a step moves the weights the sampler
# reads. That is what makes every round on-policy without a copy.
with (
    Model(BASE, n_ctx=N_TOK, n_ubatch=N_TOK, n_seq_max=N_SEQ, training=True) as policy,
    Model(BASE, n_ctx=512, n_seq_max=G) as sampler,     # one sequence per rollout in the group
):
    handle = policy.attach_adapter("adapter.gguf")
    sampler.attach_adapter(adapter=handle)

    engine = RolloutEngine(
        libs, sampler.ctx, sampler.model,
        n_rollouts=G,
        sampler=SamplerConfig(temperature=0.8, top_p=0.95, max_new_tokens=64),
        adapter=handle,
    )

    result = train_grpo(
        libs, policy, engine,
        prompts=PROMPTS,
        reward_fn=substring_reward("4"),
        config=GRPOConfig(lr=1e-4, clip_eps=0.2, seq_len=SEQ_LEN, iterations=20),
    )

    print(result.rewards())     # the curve that should be going up
```

`reward_fn` is any `Callable[[str, Rollout], float]` — the decoded completion and the rollout that
produced it. To add a KL penalty against the *untuned* model, set `kl_coef` and pass
`lm_head=load_lm_head(BASE)`; the reference is this same model with the adapter switched off, never
a second set of weights. Without the `lm_head` the call is refused rather than silently running
with the KL term contributing nothing.

**A KL penalty also grows the sampler context.** The reference pass scores every rollout in one
decode on the *rollout* context, so with `kl_coef > 0` the sampler above is too small twice over,
and `train_grpo` says so before the first round: it needs `n_seq_max >= N_SEQ` (16, not `G`) and
`n_batch >= N_TOK` — and `Model` takes `n_batch` from `n_ctx`, so that means
`Model(BASE, n_ctx=N_TOK, n_seq_max=N_SEQ)`. Generation alone needs neither; only the KL does.

The details, and the several ways a GRPO implementation can look like it is training when it is
not, are in [`dev/grpo.md`](dev/grpo.md).

## Where things are

| | |
|---|---|
| `learning_llamas.Model` | Load a GGUF, open a context, attach an adapter |
| `learning_llamas.train` | `train_sft`, `train_dpo`, `train_grpo`, and the shared step loop |
| `learning_llamas.data` | Chat templating, tokenization, and the loss mask |
| `learning_llamas.preflight` | *Will this model train?* — run it before a long job, not after |
| `learning_llamas.checkpoint` | Save and resume, optimizer moments included |
| `learning_llamas.export` | Merge an adapter into its base |

`preflight(libs, model.ctx, tokens)` is worth the thirty seconds: it walks the actual training
graph and reports every op that has no backward on this backend, before you spend an afternoon
finding out one at a time.
