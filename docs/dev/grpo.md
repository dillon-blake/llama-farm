# GRPO (S1-15, S1-16)

Sample G answers to the same prompt, score them, and let the group be its own baseline. A completion
that beat its siblings gets a positive advantage; one that lost gets a negative one. No value network
— which is the whole point, because a value network is a second model to train and this project's
premise is that you only have room for one.

```python
from learning_llamas.train import (
    GRPOConfig, RolloutEngine, SamplerConfig, token_reward, train_grpo,
)

result = train_grpo(libs, train_model, engine, prompts, reward_fn, GRPOConfig(lr=1.0, seq_len=32))
result.rewards()   # the curve that should be going up
```

## Two contexts, one adapter

This is the part to get right, because getting it wrong has **no symptom**.

GRPO is online: generate, update, generate again. But a `llama_decode` on a training-mode context
frees the scheduler the optimizer is holding, and the next one segfaults. So generation and training
run on **two separate contexts** — and they must share **one `llama_adapter_lora` object**, because
adapters are model-level and their A/B tensors live in the adapter rather than the context. A
training step then mutates the very weights the rollout engine samples through, and every round is
on-policy for free, with no copying.

Attach the adapter *file* to each context and you get two adapters that start out equal and diverge
the moment a step is taken. The trainer moves one; the engine keeps sampling from the other. GRPO
runs forever on a policy that never changes: no error, no NaN, and a reward curve that is flat for no
reason anyone can see.

That is not a hypothetical — it is what the toy-task test did on its first run, and the reward was
bit-identical across all six iterations. `train_grpo` now refuses it.

## The rollouts

Ordinary llama.cpp inference. **No new native code**: KV cache, sampler chains, parallel sequences,
public C API throughout.

**The prompt is decoded once, not G times.** Decode it on sequence 0, `llama_memory_seq_cp` its KV to
the other G−1 members, then let them diverge. A group of 8 costs one prompt forward pass, not eight.
(This is also the sanctioned replacement for unsloth's prefix-grouper, which is AGPL — ROADMAP §13.)

**`logp_old` is captured at sample time, from the raw logits.** The ratio GRPO trains on is
`exp(logp_new − logp_old)`, and `logp_old` must be the *policy's* logprob of the drawn token — not the
sampler's. Temperature and top-p change what gets drawn; they do not change the distribution the ratio
is defined against. The logits row is snapshotted **before** the sampler touches it. It costs no extra
forward pass: the row is already there from the decode that produced it.

## The loss, and why the clip is built out of RELU

```
logp_new = -ce_sparse(logits, labels, mask)            # S1-04, already carries the mask
r        = exp(logp_new - logp_old)                    # logp_old is a constant input
L        = min( r*A , clip(r, 1-eps, 1+eps)*A )        # PPO's pessimistic surrogate
loss     = -sum(L) + kl_coef * sum(k3)                 # k3 = expm1(d) - d,  d = logp_ref - logp_new
```

`ggml_clamp` does now have a backward rule — but the RELU composite is the reference form, it is
topology-stable, and it needs nothing that was not already there:

```
clip(r, lo, hi) = lo + relu(r - lo) - relu(r - hi)
min(a, b)       = b - relu(b - a)
```

Check the clip on its three regions: `r < lo` gives `lo + 0 - 0`; `lo ≤ r ≤ hi` gives
`lo + (r - lo) - 0 = r`; `r > hi` gives `lo + (r - lo) - (r - hi) = hi`.

**The `min` is taken after multiplying by the advantage, and it has to be.** A negative advantage
swaps the two arguments, so the clip binds from the *other* side — and PPO's entire asymmetry, between
having made a good token likelier and having made a bad token likelier, lives in that swap. An
implementation that clipped the ratio and *then* applied the sign would be wrong on half the tokens,
while still training, still converging, and still looking fine.

### The k3 KL is `expm1(d) − d` — and ggml's `expm1` was a lie

k3 is `exp(d) − d − 1`, and it is **non-negative by construction**. That is the entire reason to
prefer it over the naive estimator, which is unbiased but can go negative on a single sample and then
*rewards* divergence from the reference.

Written as `expm1(d) − d` it is the same number, one node fewer, and — the reason — **exact near
`d = 0`**, which is exactly where a GRPO run lives: `d = logp_ref − logp_new` is approximately zero on
every on-policy step *by design*.

Except that ggml's `GGML_UNARY_OP_EXPM1` was implemented as `expf(x) - 1.0f` — precisely the
cancellation the op exists to avoid, and precisely what the paragraph above claims it avoids. Measured
against the true value (`d²/2`):

| d | ggml's k3 | true | |
|---|---|---|---|
| 1e-5 | 1.36e-8 | 5.0e-11 | 271× too large |
| 1e-6 | 7.29e-8 | 5.0e-13 | 145,000× too large |
| 5e-5 | **−5.13e-8** | +1.25e-9 | **negative** |

A KL penalty that goes negative does not penalize divergence; it **pays for it**. Fixed in the fork
(`op_expm1` → `expm1f`, which was already used twice in the same file). `test_the_k3_kl_is_accurate_
and_non_negative_near_zero` pins it: put the old form back and it fails immediately.

This was found by an adversarial review, not by the tests — every test passed, because no test looked
at the KL in the regime the KL actually operates in.

### Masked tokens are zeroed by the shim, not by the caller

On a masked token `ce_sparse` makes `logp_new` exactly 0. So a large positive `logp_ref` left in a
padding slot makes `d = logp_ref − 0` big, `expm1(d)` overflows float32 to `+inf`, and the loss —
along with every gradient in the batch — is `inf` or `NaN`. From one stray number in a slot that was
supposed to be ignored.

So `ll_train_step_grpo` zeroes every GRPO input where the weight is zero, rather than trusting the
caller. And a `NULL` `logp_ref` is filled with `logp_old`, **not with zeros**: zero is not a neutral
logprob, it is *certainty*, and a zero-filled reference claims the policy has diverged enormously from
a model that assigns probability 1 to every token. (Measured: it turns a loss of 0 into 532534.)

### The weights must be exactly 0 or 1

They are the completion mask. `logp_new` is `-ce_sparse(...)`, and ce_sparse multiplies by the weight
— so a weight of 0.5 does not half-grade the token, it **halves the log-probability** that goes into
the importance ratio, and `exp(0.5·logp − logp_old)` is not a ratio of anything. Nothing would fail;
the run would just optimize something else. The shim rejects it.

Every normalization GRPO needs belongs in the advantages and the KL weights, where it scales the loss
without touching the logprob.

### The reference pass

`kl_coef > 0` needs something to be a KL *to*, and the reference is **this model with the adapter
off** — never a second model, never a second set of weights (BLUEPRINT D6). It is scored on the
**rollout** context, because that is the one without an optimizer holding its scheduler.

`train_grpo(..., lm_head=load_lm_head(base_gguf))` is required when `kl_coef > 0`, and refused
otherwise. The first version of this code accepted `kl_coef` and then never ran a reference pass at
all — the KL term was multiplied by a zero weight and contributed exactly nothing. The run *looked*
regularized. A knob that reads as if it regularizes and does not is worse than no knob.

## Self-verification

`verify.py`. A fast path is compared against the naive one **on the first real call, on the real
inputs**; if they agree it is used from then on, and if they do not it is logged loudly and the naive
path is used **for the rest of the process**.

"Forever" is deliberate. A fast path that diverged once will diverge again, on inputs nobody can
predict, and it will diverge *silently* — the loss stays finite and the model still trains.
Re-testing it periodically would only mean being wrong between tests.

The discipline is unsloth's; the code is not (their GRPO orchestration is AGPL-marked and was not
read — ROADMAP §13).

## The toy task

Reward = *how much of the completion came from this set of tokens*, and that is the point rather than
a shortcut. A GRPO update raises the log-probability of the tokens in high-advantage completions, so a
reward that says **use more of these tokens** is precisely what the gradient can act on, with nothing
in between. A policy that cannot learn it has not learned anything, and the fault is in the update.

An output-length target — the obvious toy — asks a two-layer random model to control the decoded
*character count* of its own samples. Measured, the signal is real but buried under sampling noise
long before it reaches the weights: it trends up, and not far enough above the noise to assert on.

| iteration | 1 | 2 | 3 | … | 10 | 11 | 12 |
|---|---|---|---|---|---|---|---|
| mean group reward | 0.254 | 0.258 | 0.262 | … | 0.277 | 0.273 | **0.281** |

Robust across seeds (11, 23, 47 all rise), 3 seconds, `lr=1.0`, G=8. At `lr=3.0` the policy
**collapses** — which is what the gradient clip (S1-10) is for, and a fair reminder that RL is not SFT.
