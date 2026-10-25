import numpy as np


def softmax(logits):
    weights = np.exp(logits - logits.max())
    return weights / weights.sum()


vocab_size = 152_064
k = 128
n_trials = 500_000
head_mass = 0.95

# Choose the Zipf exponent so the trainer's top 128 have mass 0.95.
log_ranks = np.log(np.arange(1, vocab_size + 1, dtype=float))
lower, upper = 0.05, 5.0
for _ in range(65):
    alpha = (lower + upper) / 2
    p = softmax(-alpha * log_ranks)
    if p[k:].sum() > 1 - head_mass:
        lower = alpha
    else:
        upper = alpha

# One shared parameter beta has logits a_v + beta*c_v, evaluated at beta=0.
rng = np.random.default_rng(20261025)
c = rng.choice([-1.0, 1.0], size=vocab_size)
c[:k] = 0
noise = rng.normal(0, 0.1, size=vocab_size)
q = softmax(np.log(p) + 0.7 * c + noise)
scores = c - np.sum(p * c)
rewards = 1 - 0.2 * c

head = np.argsort(-q)[:k]
tail = np.ones(vocab_size, dtype=bool)
tail[head] = False
Q_T = q[tail].sum()
P_T = p[tail].sum()
q_hat = q.copy()
q_hat[tail] = (Q_T / P_T) * p[tail]
s_bar = np.sum(q * scores)
s_bar_hat = np.sum(q_hat * scores)
head_center = np.sum(q[head] * scores[head])

full_mean = np.sum(q * rewards * (scores - s_bar))
exact = {
    "Score Centering": full_mean,
    "Top-128 Approximation": np.sum(q * rewards * (scores - s_bar_hat)),
    "MC Centering": full_mean,
}

# Each trial uses independent rollout and conditional-tail draws, with replacement.
rng = np.random.default_rng(97531)
actual = rng.choice(vocab_size, size=n_trials, p=q)
auxiliary = rng.choice(np.flatnonzero(tail), size=n_trials,
                       p=q[tail] / Q_T, replace=True)
s_bar_mc = head_center + Q_T * scores[auxiliary]
centers = {
    "Score Centering": s_bar,
    "Top-128 Approximation": s_bar_hat,
    "MC Centering": s_bar_mc,
}

print(f"Vocabulary: {vocab_size:,}; k = {k}; trials = {n_trials:,}")
print(f"alpha = {alpha:.9f}; trainer top-{k} mass = {p[:k].sum():.6f}")
print(f"Sampler top-{k} mass = {q[head].sum():.6f}; Q_T = {Q_T:.6f}")
print(f"s_bar = {s_bar:.7f}; s_bar_hat = {s_bar_hat:.7f}; "
      f"delta = {s_bar - s_bar_hat:.7f}")
print(f"\n{'Method':<24} {'Exact mean':>12} {'Trial mean':>12} {'Std. error':>12}")
for name, center in centers.items():
    updates = rewards[actual] * (scores[actual] - center)
    se = updates.std(ddof=1) / np.sqrt(n_trials)
    print(f"{name:<24} {exact[name]:12.7f} {updates.mean():12.7f} {se:12.7f}")
