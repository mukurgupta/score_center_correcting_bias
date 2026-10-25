# Correcting Tail Bias in Top-k Score Centering

This project studies a limitation of the top-k approximation in [Score Centering Stabilizes Off-policy Reinforcement Learning](https://arxiv.org/abs/2609.20807). Our proposed fix removes this error in expectation.

The simulations below illustrate the mechanism at a fixed generation prefix. They do not demonstrate improved LLM training performance. The estimator applies the existing exact-head/sampled-tail construction of [Liu et al.](https://proceedings.mlr.press/v97/liu19c.html).

- **[Mathematical note](notes/notes.tex):** Problem setup, proposed fix and main derivations.

- **[Setup](#setup):** Gives the commands to run both simulations using Python and NumPy on a CPU.

- **[Toy simulation](#toy-simulation) — [simulate_toy.py](simulate_toy.py):** Uses the smallest three-token counterexample. Even with identical rewards and 99% mass coverage by head tokens, the top-k approximation produces a nonzero expected update; Score Centering and our proposed fix (MC Centering) give zero mean updates.

- **[Large vocabulary simulation](#large-vocabulary-simulation) — [simulate_large_vocab.py](simulate_large_vocab.py):** Tests whether the same problem can affect a shared parameter when many related tail tokens shift together. In a constructed example with 152,064 tokens and nonconstant rewards, the Top-128 Approximation reverses the expected update. MC Centering restores the Score Centering expectation, with additional sampling noise. The section explains the model, rewards, and results.

- **[Implementation](#implementation):** Describes how vLLM and SGLang expose the required probs, and pseudocode to expand the sampler to obtain independent tail-token IDs. This requires no additional continuations or rewards.

## Setup

Both scripts require Python 3 and NumPy:

```sh
python3 -m pip install numpy
python3 simulate_toy.py
python3 simulate_large_vocab.py
```

Each script runs 500,000 trials with fixed random seeds and prints exact and empirical mean updates. Neither creates files or requires a GPU.

## Toy simulation

Let $p$ be the trainer distribution and $q$ the sampler distribution over tokens $(A,B,C)$:

$$
p=(0.990,0.009,0.001),\qquad q=(0.990,0.001,0.009).
$$

Store only the sampler's most likely token, $H=\{A\}$, so $k=1$. Both tails have mass 0.01. The paper's reconstruction keeps $q_A$ and fills the tail with rescaled trainer probabilities; here it gives $\widehat q=p$.

The trainer parameters are three softmax logits, giving token scores $s_v=\nabla_\ell\log p_v=e_v-p$. Every token has raw reward $R=1$, so the expected-reward objective is constant and its gradient is zero.

| Approach | Exact mean update |
| --- | --- |
| Score Centering | $(0,0,0)$ |
| Top-k Approximation, $k=1$ | $(0,-0.008,0.008)$ |
| MC Centering, one tail draw | $(0,0,0)$ |

Thus 99% head coverage does not ensure a zero mean update. The script enumerates the six possible rollout/tail-token pairs to obtain the exact MC expectation, then checks it with independent random draws. Individual updates remain random.

## Large vocabulary simulation

The three-token example shows that the top-$k$ approximation can produce an unwanted mean update. This simulation asks a further question: Can errors across many tail tokens combine to push a shared parameter in the wrong direction, even when rewards differ between tokens?

We construct a synthetic vocabulary of 152,064 tokens and compare full Score Centering, the Top-128 Approximation, and MC Centering. All distributions and rewards stay fixed. We measure the updates each method would produce; we do not run a training trajectory.

### One parameter affects many tokens

Let $p_v$ be the trainer's initial probability of token $v$. We use a synthetic distribution with a few likely tokens and many less likely ones. Its 128 most likely tokens contain 95% of the probability mass.

The model has one parameter, $\beta$. Each token has a fixed label $c_v$ describing how its logit changes when $\beta$ increases:

| Label | Effect of increasing $\beta$ |
| --- | --- |
| $c_v=+1$ | Raises the token's logit |
| $c_v=-1$ | Lowers the token's logit |
| $c_v=0$ | Leaves the token's logit unchanged |

The trainer's original top 128 tokens have label zero. Each remaining token is randomly assigned $+1$ or $-1$, with equal probability. These assignments remain fixed throughout the experiment.

The resulting trainer probabilities are

$$
p_v(\beta)=
\frac{p_v\exp(\beta c_v)}
{\sum_u p_u\exp(\beta c_u)}.
$$

We evaluate everything at $\beta=0$, where this equals the initial distribution $p$.

This models a shared parameter that favors one family of tokens over another. Related identifiers or API choices provide a possible interpretation, but the labels here are synthetic and have no actual linguistic meaning.

### Reward and sampler

Assign each token the reward

$$
R(v)=1-0.2c_v.
$$

The $+1$ group receives reward 0.8, neutral tokens receive 1.0, and the $-1$ group receives 1.2. Therefore, **decreasing $\beta$ improves expected reward**: it favors the higher-reward group and suppresses the lower-reward group.

Now construct a sampler that favors the lower-reward $+1$ group:

$$
q_v\propto p_v\exp(0.7c_v+\epsilon_v),
\qquad
\epsilon_v\sim\mathcal N(0,0.1^2).
$$

The $0.7c_v$ term shifts many tokens together. The noise adds small token-specific differences. Both the labels and noise are drawn once and then held fixed. This alignment is deliberate: we want a controlled example where sampler mismatch acts against the reward preference.

We select the logged head $H$ as the **sampler's** top 128 tokens. This can differ from the trainer's original head. It contains approximately 94.02% of the sampler's probability, leaving approximately 5.98% in the tail $T$.

### What the three methods subtract

The score measures how increasing $\beta$ changes a token's log-probability:

$$
s_v=
\left.\frac{\partial\log p_v(\beta)}{\partial\beta}\right|_{\beta=0}
=c_v-\sum_u p_uc_u.
$$

Because there is only one parameter, each score and update is a scalar. For an actual token $y_t\sim q$, all three methods produce an ascent direction of the form

$$
g=R(y_t)(s_{y_t}-C),
$$

where $C$ is the centering term:

- **Score Centering** uses the exact sampler mean, $C=\bar s=\sum_vq_vs_v$.
- **Top-128 Approximation** uses $C=\widehat{\bar s}=\sum_v\widehat q_vs_v$, retaining the logged head probabilities and reconstructing the tail from the trainer.
- **MC Centering** sums the head exactly and estimates the tail using one independent auxiliary token:

$$
C=\bar s_{\mathrm{MC}}
=\sum_{v\in H}q_vs_v+Q_Ts_z,
\qquad
z\sim q(\cdot\mid T),
\qquad
Q_T=\sum_{v\in T}q_v.
$$

The auxiliary token is drawn independently of the actual token. It requires no continuation or reward evaluation.

### Results: the approximation reverses the mean update

A negative update decreases $\beta$, which is the reward-improving direction in this example. A positive update increases it.

The script computes exact expectations by summing over the vocabulary, then checks them with 500,000 independent trials:

| Approach | Exact mean update | Mean across trials |
| --- | ---: | ---: |
| Score Centering | −0.0121621 | −0.0124142 |
| Top-128 Approximation | **+0.0224561** | +0.0222065 |
| MC Centering | −0.0121621 | −0.0122861 |

Full Score Centering points in the reward-improving direction. **The Top-128 Approximation points in the opposite direction.** MC Centering restores the full-centering expectation; its trial mean differs slightly because of sampling noise.

The reason is that the approximation substantially underestimates the correction:

$$
\bar s\approx0.0378720,
\qquad
\widehat{\bar s}\approx0.0029959.
$$

### What this demonstrates

Many tail tokens can contribute errors in the same direction when they share sensitivity to a parameter. Their contributions can collectively outweigh the useful reward signal, even though the logged head contains most of the probability mass.

## Implementation

Existing serving APIs return the top-k logprobs of the head:

| Engine | Top-128 logging options |
| --- | --- |
| [vLLM](https://docs.vllm.ai/en/latest/api/vllm/sampling_params/) | `logprobs=128`; configure sufficient `max_logprobs` |
| [SGLang](https://docs.sglang.io/docs/basic_usage/sampling_params) | `return_logprob=true`, `top_logprobs_num=128` |

MC Centering additionally needs independent tail draws while the rollout distribution is available. A sampler extension can mask the logged head and return the sampled IDs as extra metadata:

```python
tail_weights = q.clone()
tail_weights[H] = 0
Q_T = tail_weights.sum()
if Q_T > 0:
    z = torch.multinomial(tail_weights, m,
                          replacement=True, generator=aux_rng)
```

Keep these draws independent of the rollout and reward, and preserve repeated IDs. No extra continuation or reward is needed. Use the same processed distribution for head probabilities and tail draws, including temperature, masks, and truncation; do not reapply top-p after removing the head. Runtime overhead depends on the sampling backend and has not been measured here.

## References

- Marek and Ryabinin (2026). [Score Centering Stabilizes Off-policy Reinforcement Learning](https://arxiv.org/abs/2609.20807).
- Liu et al. (2019). [Rao-Blackwellized Stochastic Gradients for Discrete Distributions](https://proceedings.mlr.press/v97/liu19c.html).

## Citation

Please cite the following when using any part of these findings:

```bibtex
@misc{correctingtailbias,
  title     = {Correcting Tail Bias in Top-k Score Centering},
  author    = {Mukur Gupta},
  year      = {2026},
  publisher = {GitHub},
  journal   = {GitHub repository},
}
```
