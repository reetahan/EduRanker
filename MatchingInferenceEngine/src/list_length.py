import numpy as np

def sample_truncated_normal_lengths(
    n_students,
    mean=10,
    std=2,
    min_len=1,
    max_len=12,
    rng=None
):
    rng = np.random.default_rng() if rng is None else rng
    x = rng.normal(loc=mean, scale=std, size=n_students)
    lengths = np.rint(x).astype(int)
    lengths = np.clip(lengths, min_len, max_len)
    return lengths


def sample_empirical_lengths(n_students, empirical_probs, rng):
    lengths = np.array(list(empirical_probs.keys()), dtype=int)
    probs   = np.array(list(empirical_probs.values()), dtype=float)
    probs  /= probs.sum()  
    return rng.choice(lengths, size=n_students, p=probs).astype(int)