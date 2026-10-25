import numpy as np


# Three-token softmax; gradients are with respect to its logits. Reward is 1.
p = np.array([0.990, 0.009, 0.001])
q = np.array([0.990, 0.001, 0.009])
head = np.array([0])
tail = np.array([1, 2])
scores = np.eye(3) - p

# Reconstruct the sampler tail from the trainer, as in the paper.
tail_mass = 1 - q[head].sum()
rho = tail_mass / (1 - p[head].sum())
q_hat = rho * p.copy()
q_hat[head] = q[head]
full_center = q @ scores
topk_center = q_hat @ scores
head_center = q[head] @ scores[head]

# Exact means. Enumerate independent pairs (actual token, auxiliary tail token).
tail_probabilities = q[tail] / tail_mass
repaired_updates = []
pair_probabilities = []
for actual in range(3):
    for auxiliary, probability in zip(tail, tail_probabilities):
        center = head_center + tail_mass * scores[auxiliary]
        repaired_updates.append(scores[actual] - center)
        pair_probabilities.append(q[actual] * probability)

exact = {
    "Top-k Approximation": q @ (scores - topk_center),
    "Score Centering": q @ (scores - full_center),
    "MC Centering": np.array(pair_probabilities) @ np.array(repaired_updates),
}
assert np.allclose(exact["Top-k Approximation"], [0, -0.008, 0.008])
assert np.allclose(exact["Score Centering"], 0)
assert np.allclose(exact["MC Centering"], 0)

# Monte Carlo uses separate draws for the actual and auxiliary tokens.
rng = np.random.default_rng(42)
n = 500_000
actual = rng.choice(3, size=n, p=q)
auxiliary = rng.choice(tail, size=n, p=tail_probabilities)
simulated = {
    "Top-k Approximation": scores[actual] - topk_center,
    "Score Centering": scores[actual] - full_center,
    "MC Centering": (
        scores[actual] - head_center - tail_mass * scores[auxiliary]
    ),
}

np.set_printoptions(precision=6, suppress=True)
print("p =", p, " q =", q, " reconstructed q =", q_hat)
print(f"\nMean updates ({n:,} Monte Carlo samples):")
print(f"{'Method':<20} {'Exact mean':<30} Monte Carlo mean")
for name in ["Score Centering", "Top-k Approximation", "MC Centering"]:
    mean = exact[name]
    sampled_mean = simulated[name].mean(axis=0)
    print(f"{name:<20} {str(mean):<30} {sampled_mean}")
