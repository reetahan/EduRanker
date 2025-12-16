import time
import argparse
import multiprocessing as mp
import os
import json
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import binom
from tqdm import tqdm
from multiprocessing import Pool

# Global mixture configuration (can be set from CLI in __main__)
GLOBAL_MIXTURE_PHIS = None
GLOBAL_MIXTURE_WEIGHTS = None
# If GLOBAL_MIXTURE_CENTERS is the string 'random' we'll generate random centers on-demand
GLOBAL_MIXTURE_CENTERS = None
# ==========================================
# Variable List Length Distribution
# ==========================================

def generate_list_length_distribution(k_max, alpha=2.0, min_k=1, k_dist='power_tail', k_std=None):
    """
    Generate a power law distribution for list lengths.
    
    Parameters:
    - k_max: Maximum list length (most common)
    - alpha: Power law exponent (higher = more concentration at k_max)
    - min_k: Minimum list length
    
    Returns: Probability distribution over list lengths [min_k, min_k+1, ..., k_max]
    """
    k_values = np.arange(min_k, k_max + 1)

    # Choose weighting scheme
    if k_dist == 'power_tail':
        # Reverse power law: higher k gets higher probability (original behavior)
        weights = (k_values / k_max) ** alpha
    elif k_dist == 'centered_power':
        # Power-like decay away from center (center at k_max/2)
        center = k_max / 2.0
        # weight highest at center and decays as 1/(|d|+1)^alpha
        weights = 1.0 / (np.abs(k_values - center) + 1.0) ** alpha
    elif k_dist == 'normal':
        # Gaussian centered at k_max/2 with std k_std or default k_max/4
        mu = k_max / 2.0
        sigma = (k_std if k_std is not None else max(1.0, k_max / 4.0))
        # Use Gaussian pdf (not normalized over integer grid yet)
        weights = np.exp(-0.5 * ((k_values - mu) / sigma) ** 2)
    else:
        raise ValueError(f"Unknown k_dist '{k_dist}'")

    # Normalize to get probabilities
    probs = weights / weights.sum()
    
    return k_values, probs

def sample_list_length(k_max, alpha=2.0, min_k=1, k_dist='power_tail', k_std=None):
    """Sample a single list length from the chosen distribution over [min_k, ..., k_max]"""
    k_values, probs = generate_list_length_distribution(k_max, alpha, min_k, k_dist=k_dist, k_std=k_std)
    return int(np.random.choice(k_values, p=probs))

# Utility: safe normalization helper to avoid accidental double-normalization
def normalize_pi(pi):
    """Return a safely normalized copy of pi (sum -> 1). If sum==0 returns original array."""
    pi = np.asarray(pi, dtype=float)
    s = pi.sum()
    return pi / s if s > 0 else pi


def _make_filename_suffix(seed=None, k_dist=None, k_std=None):
    """Return a filename-safe suffix encoding optional parameters."""
    parts = []
    if k_dist:
        parts.append(f"kdist-{k_dist}")
    if k_std is not None:
        # format k_std safely
        try:
            kstd_str = f"{float(k_std):.2f}".replace('.', 'p')
        except Exception:
            kstd_str = str(k_std).replace('.', 'p')
        parts.append(f"kstd-{kstd_str}")
    if seed is not None:
        parts.append(f"seed-{int(seed)}")
    return ("_" + "_".join(parts)) if parts else ""


def save_figure(fname):
    """Save a matplotlib figure, creating parent directory if needed."""
    d = os.path.dirname(fname)
    if d and not os.path.exists(d):
        os.makedirs(d, exist_ok=True)
    plt.savefig(fname, dpi=300, bbox_inches='tight')


def _internal_test_normal_kstd():
    """Internal quick test: for 'normal' k_dist, increasing k_std should increase spread.

    This function raises AssertionError on failure so it can be used by CI or run
    locally via the --self_test flag.
    """
    print("Running internal test: normal k-dist std behavior...")
    k_max = 21
    min_k = 1

    small_std = 0.5
    large_std = 5.0

    k_vals_s, probs_s = generate_list_length_distribution(k_max, alpha=2.0, min_k=min_k, k_dist='normal', k_std=small_std)
    k_vals_l, probs_l = generate_list_length_distribution(k_max, alpha=2.0, min_k=min_k, k_dist='normal', k_std=large_std)

    # Probabilities should sum to ~1
    assert abs(probs_s.sum() - 1.0) < 1e-12, f"small std probs sum = {probs_s.sum()}"
    assert abs(probs_l.sum() - 1.0) < 1e-12, f"large std probs sum = {probs_l.sum()}"

    mu = k_max / 2.0
    var_s = np.sum(((k_vals_s - mu) ** 2) * probs_s)
    var_l = np.sum(((k_vals_l - mu) ** 2) * probs_l)

    assert var_s < var_l, f"Expected var({small_std}) < var({large_std}) but got {var_s} >= {var_l}"
    assert probs_s.max() > probs_l.max(), "Expected peak probability to decrease with larger std"

    print("Internal test passed: normal k-dist std behavior")

# ==========================================
# NYC School Data: Load Rankings as Centers
# ==========================================

def load_nyc_rankings_as_centers(aggregate_type='residential', data_dir=None):
    """
    Load NYC school rankings and convert to integer-indexed centers for Mallows mixture.
    
    Returns: (centers, categories, m) where centers is list of integer rankings
    """
    if data_dir is None:
        script_dir = Path(__file__).parent
        data_dir = script_dir / ".." / "Data-Analysis" / "output" / "mallows_rankings"
    else:
        data_dir = Path(data_dir)
    
    json_file = data_dir / f"{aggregate_type}_rankings.json"
    with open(json_file, 'r') as f:
        rankings_data = json.load(f)
    
    centers = []
    categories = []
    all_schools = set()
    
    for entry in rankings_data:
        categories.append(entry['category'])
        centers.append(entry['ranking'])
        all_schools.update(entry['ranking'])
    
    # Map DBNs to 1-indexed integers
    school_to_idx = {dbn: idx + 1 for idx, dbn in enumerate(sorted(all_schools))}
    m = len(all_schools)
    
    # Convert centers to integer indices
    centers_indexed = [[school_to_idx[dbn] for dbn in center] for center in centers]
    
    print(f"Loaded {len(centers)} {aggregate_type} rankings, {m} schools total")
    return centers_indexed, categories, m


def simulate_nyc_fixed_k(aggregate_type='residential', phi=0.8, k_values=None, 
                          n=72000, c=156, n_samples=500, n_workers=4):
    """
    Simulate NYC school choice with fixed k (no variable list lengths).
    Tests effect of varying k while keeping phi fixed.
    
    Parameters:
    - aggregate_type: 'residential', 'language', or 'zip'
    - phi: fixed Mallows phi parameter
    - k_values: list of k values to test (default: [3, 6, 9, 12])
    - n: number of students
    - c: school capacity
    - n_samples: samples for computing pi_r (default: 500, use 2000+ for higher accuracy)
    - n_workers: parallel workers
    
    Returns: dict with results for each k value
    """
    if k_values is None:
        k_values = [3, 6, 9, 12]
    
    # Load NYC rankings as mixture centers
    centers, categories, m = load_nyc_rankings_as_centers(aggregate_type)
    
    # Create equal-weight mixture (one component per district/language/zip)
    n_components = len(centers)
    mixture_phis = [phi] * n_components
    mixture_weights = [1.0 / n_components] * n_components
    
    print(f"\nSimulating with {n_components} components, phi={phi}")
    print(f"Testing k values: {k_values}")
    
    results = {}
    
    for k in k_values:
        print(f"\n{'='*60}")
        print(f"Running simulation for k={k}")
        print(f"{'='*60}")
        
        # Compute pi_r using mixture
        print("  Computing pi_r distribution...")
        pi_vals = compute_pi(
            phi=phi,
            k_or_kmax=k,
            m=m,
            n_samples=n_samples,
            n_workers=n_workers,
            variable=False,
            mixture_phis=mixture_phis,
            mixture_weights=mixture_weights,
            mixture_centers=centers
        )
        
        pi_vals = normalize_pi(pi_vals)
        
        # Sample 10% of lottery numbers for speed
        n_sample = max(1, n // 10)
        ell_sample = np.linspace(1, n, n_sample, dtype=int)
        probs_sample = prob_unmatched_vectorized(ell_sample, pi_vals, c, k)
        
        results[k] = {
            'pi_vals': pi_vals,
            'unmatched_probs_sample': probs_sample,
            'ell_sample': ell_sample,
            'avg_unmatched': np.mean(probs_sample),
            'median_unmatched': np.median(probs_sample)
        }
        
        print(f"  Average P(unmatched): {results[k]['avg_unmatched']:.4f}")
        print(f"  Median P(unmatched): {results[k]['median_unmatched']:.4f}")
    
    return results, m, categories

# ==========================================
# Step 1: Compute pi_r(phi, k) via sampling
# ==========================================

def sample_mallows_top_k_rsm(m, phi, k, center=None):
    """
    Sample top-k from Mallows using RSM directly.
    Only generates k items instead of full ranking.
    """
    # If a center permutation is provided, sample relative to that ordering.
    if center is None:
        remaining = np.arange(1, m+1)
    else:
        # center expected as a permutation of 1..m
        remaining = np.array(center, dtype=int).copy()
    top_k = np.zeros(k, dtype=int)
    
    # Only iterate k times (not m times!)
    for i in range(k):
        # Number of items remaining
        n_remaining = len(remaining)
        
        # Selection probabilities (RSM with Mallows parameters)
        adjusted_ranks = np.arange(n_remaining)
        weights = phi ** adjusted_ranks
        probs = weights / weights.sum()
        
        # Select position
        chosen_idx = np.random.choice(n_remaining, p=probs)
        top_k[i] = remaining[chosen_idx]
        
        # Remove selected item
        remaining = np.delete(remaining, chosen_idx)
    
    return top_k


def compute_pi_batch(args):
    """Worker function for parallel processing"""
    phi, k, m, n_samples = args
    counts = np.zeros(m)
    total_apps = 0

    for _ in range(n_samples):
        top_k = sample_mallows_top_k_rsm(m, phi, k)  # ← Use RSM directly!
        counts[top_k - 1] += 1
        total_apps += k

    return counts, total_apps

def compute_pi_batch_variable(args):
    """Worker function for parallel processing with variable list lengths"""
    # args: phi, k_max, m, n_samples, alpha, min_k, k_dist, k_std
    phi, k_max, m, n_samples, alpha, min_k, k_dist, k_std = args
    counts = np.zeros(m)
    total_apps = 0

    for _ in range(n_samples):
        # Sample list length for this person
        k = sample_list_length(k_max, alpha, min_k, k_dist=k_dist, k_std=k_std)
        # Sample their top-k preferences
        top_k = sample_mallows_top_k_rsm(m, phi, k)
        counts[top_k - 1] += 1
        total_apps += k

    return counts, total_apps


def compute_pi_batch_mixture(args):
    """Worker for mixture-of-Mallows (fixed k per person)."""
    phis, weights, centers, k, m, n_samples = args
    counts = np.zeros(m)
    total_apps = 0

    # Normalize weights
    weights = np.asarray(weights, dtype=float)
    if weights.sum() > 0:
        weights = weights / weights.sum()
    else:
        weights = np.ones(len(phis)) / len(phis)

    for _ in range(n_samples):
        comp = np.random.choice(len(phis), p=weights)
        phi_comp = phis[comp]
        center = None if centers is None else centers[comp]
        # sample top-k from selected component, respecting its center if provided
        top_k = sample_mallows_top_k_rsm(m, phi_comp, k, center=center)
        counts[top_k - 1] += 1
        total_apps += k

    return counts, total_apps


def compute_pi_batch_variable_mixture(args):
    """Worker for mixture-of-Mallows with variable k per person."""
    # args: phis, weights, centers, k_max, m, n_samples, alpha, min_k, k_dist, k_std
    phis, weights, centers, k_max, m, n_samples, alpha, min_k, k_dist, k_std = args
    counts = np.zeros(m)
    total_apps = 0

    weights = np.asarray(weights, dtype=float)
    if weights.sum() > 0:
        weights = weights / weights.sum()
    else:
        weights = np.ones(len(phis)) / len(phis)

    for _ in range(n_samples):
        # choose component
        comp = np.random.choice(len(phis), p=weights)
        phi_comp = phis[comp]
        center = None if centers is None else centers[comp]

        # sample list length
        k = sample_list_length(k_max, alpha, min_k, k_dist=k_dist, k_std=k_std)
        top_k = sample_mallows_top_k_rsm(m, phi_comp, k, center=center)
        counts[top_k - 1] += 1
        total_apps += k

    return counts, total_apps

def compute_pi_r(phi, k, m, n_samples=2000, n_workers=4):
    """
    Return pi_r as the probability a random application goes to school r (sums to 1).
    For fixed k each sample contributes exactly k applications.
    """
    # Delegate to unified compute_pi for consistency and to avoid duplication
    return compute_pi(phi, k, m, n_samples=n_samples, n_workers=n_workers, variable=False)


def compute_pi_r_variable(phi, k_max, m, n_samples=2000, n_workers=4, alpha=2.0, min_k=1):
    """
    Return pi_r as probability a random application goes to school r, when students have variable k.
    We account for the actual number of applications generated.
    """
    # Delegate to unified compute_pi for consistency and to avoid duplication
    return compute_pi(phi, k_max, m, n_samples=n_samples, n_workers=n_workers, variable=True, alpha=alpha, min_k=min_k)


def compute_pi(phi, k_or_kmax, m, n_samples=2000, n_workers=4, variable=False,
               alpha=2.0, min_k=1, mixture_phis=None, mixture_weights=None, mixture_centers=None,
               k_dist='power_tail', k_std=None):
    """
    Unified computation of pi_r.

    Parameters:
    - phi: single Mallows phi (ignored if mixture_phis provided)
    - k_or_kmax: fixed k (if variable=False) or k_max (if variable=True)
    - m: number of schools
    - n_samples, n_workers: sampling configuration
    - variable: whether to sample per-person k from power-law
    - alpha, min_k: parameters for power-law list-length distribution
    - mixture_phis: optional list/array of phi values for mixture components
    - mixture_weights: optional list of mixing weights (same length as mixture_phis)
    - mixture_centers: optional list of center permutations (not yet used)

    Returns: array length m summing to 1 (application probability per school)
    """
    is_mixture = mixture_phis is not None

    # If user set global mixture configuration via CLI, use it when explicit
    # mixture args are not provided. This lets the rest of the code call
    # compute_pi(...) unchanged while enabling a mixture model globally.
    global GLOBAL_MIXTURE_PHIS, GLOBAL_MIXTURE_WEIGHTS, GLOBAL_MIXTURE_CENTERS
    if not is_mixture and GLOBAL_MIXTURE_PHIS is not None:
        mixture_phis = GLOBAL_MIXTURE_PHIS
        mixture_weights = GLOBAL_MIXTURE_WEIGHTS
        mixture_centers = GLOBAL_MIXTURE_CENTERS
        is_mixture = True

    # If mixture centers are requested as 'random' (global flag), lazily
    # generate random center permutations consistent with m.
    if is_mixture and mixture_centers == 'random':
        rng = np.random
        mixture_centers = [rng.permutation(np.arange(1, m+1)).tolist() for _ in range(len(mixture_phis))]

    if not variable:
        k_fixed = int(k_or_kmax)
    else:
        k_max = int(k_or_kmax)

    # Prepare mixture weights if needed
    if is_mixture:
        phis = list(mixture_phis)
        weights = (list(mixture_weights) if mixture_weights is not None
                   else [1.0 / len(phis)] * len(phis))
        centers = mixture_centers

    # Single-worker path
    if n_workers == 1:
        counts = np.zeros(m)
        total_apps = 0
        for _ in tqdm(range(n_samples), desc="Sampling"):
            # choose k
            if variable:
                k = sample_list_length(k_max, alpha, min_k, k_dist=k_dist, k_std=k_std)
            else:
                k = k_fixed

            if is_mixture:
                comp = np.random.choice(len(phis), p=np.asarray(weights) / np.sum(weights))
                phi_comp = phis[comp]
                center = None if mixture_centers is None else mixture_centers[comp]
                top_k = sample_mallows_top_k_rsm(m, phi_comp, k, center=center)
            else:
                top_k = sample_mallows_top_k_rsm(m, phi, k)

            counts[top_k - 1] += 1
            total_apps += k

        return counts / total_apps if total_apps > 0 else counts

    # Multi-worker: distribute samples (include remainder)
    base = n_samples // n_workers
    rem = n_samples % n_workers
    samples_per_worker = [base + (1 if i < rem else 0) for i in range(n_workers)]

    if is_mixture:
        if variable:
            args = [(phis, weights, mixture_centers, k_max, m, s, alpha, min_k, k_dist, k_std) for s in samples_per_worker]
            with Pool(n_workers) as pool:
                results = pool.map(compute_pi_batch_variable_mixture, args)
        else:
            args = [(phis, weights, mixture_centers, k_fixed, m, s) for s in samples_per_worker]
            with Pool(n_workers) as pool:
                results = pool.map(compute_pi_batch_mixture, args)
    else:
        if variable:
            args = [(phi, k_max, m, s, alpha, min_k, k_dist, k_std) for s in samples_per_worker]
            with Pool(n_workers) as pool:
                results = pool.map(compute_pi_batch_variable, args)
        else:
            args = [(phi, k_fixed, m, s) for s in samples_per_worker]
            with Pool(n_workers) as pool:
                results = pool.map(compute_pi_batch, args)

    total_counts = np.sum([res[0] for res in results], axis=0)
    total_apps = sum(res[1] for res in results)
    return total_counts / total_apps if total_apps > 0 else total_counts

# ==========================================
# Step 2: Compute rejection probabilities
# ==========================================

def rejection_probability(ell, pi_r, c):
    """Compute P(Binomial(ell-1, pi_r) >= c)"""
    if ell <= 1:
        return 0.0
    return 1 - binom.cdf(c-1, ell-1, pi_r)

def prob_unmatched_weighted(ell, pi_values, c, k):
    """Compute P(unmatched | ell) with weighted average"""
    rejection_probs = np.array([rejection_probability(ell, pi_r, c) 
                                 for pi_r in pi_values])
    total_weight = np.sum(pi_values)
    if total_weight == 0:
        # No applications observed -> treat as everyone unmatched
        return 1.0
    weighted_avg_rejection = np.sum(pi_values * rejection_probs) / total_weight
    return weighted_avg_rejection ** k


def prob_unmatched_vectorized(ell_array, pi_values, c, k):
    """
    Vectorized computation for multiple lottery numbers at once
    """
    results = []
    for ell in tqdm(ell_array, desc="Computing unmatched probabilities", unit="lottery#"):
        results.append(prob_unmatched_weighted(ell, pi_values, c, k))
    return np.array(results)

def prob_unmatched_variable_k(ell, pi_values, c, k_max, alpha=2.0, min_k=1, k_dist='power_tail', k_std=None):
    """
    Compute P(unmatched | ell) with variable list lengths
    
    This accounts for the fact that different people have different list lengths,
    following a power law distribution.
    """
    k_values, k_probs = generate_list_length_distribution(k_max, alpha, min_k, k_dist=k_dist, k_std=k_std)
    
    total_prob = 0.0
    for k, k_prob in zip(k_values, k_probs):
        # P(unmatched | ell, k) * P(k)
        prob_for_this_k = prob_unmatched_weighted(ell, pi_values, c, k)
        total_prob += prob_for_this_k * k_prob
    
    return total_prob

def prob_unmatched_vectorized_variable(ell_array, pi_values, c, k_max, alpha=2.0, min_k=1, k_dist='power_tail', k_std=None):
    """
    Vectorized computation for multiple lottery numbers with variable k
    """
    results = []
    for ell in ell_array:
        results.append(prob_unmatched_variable_k(ell, pi_values, c, k_max, alpha, min_k, k_dist=k_dist, k_std=k_std))
    return np.array(results)


# ==========================================
# Monte-Carlo estimator (Option A)
# ==========================================
def prob_unmatched_mc_single(ell, pi_values, c, k, m=None, phi=None, center=None, n_mc=5000, reuse_samples=None):
    """
    Monte-Carlo estimate of P(unmatched | ell, k) by sampling top-k lists from
    the Mallows RSM sampler `sample_mallows_top_k_rsm` and averaging the product
    of per-school rejection probabilities.

    Parameters:
    - ell: lottery number (int)
    - pi_values: length-m array of application probabilities (normalized)
    - c: capacity
    - k: list length
    - m: number of schools (required if reuse_samples is None)
    - phi, center: passed to `sample_mallows_top_k_rsm` for sampling
    - n_mc: number of MC draws
    - reuse_samples: optional array shape (n_mc, k) of sampled top-k indices (1-based)
    """
    pi_values = np.asarray(pi_values)
    m_local = m if m is not None else len(pi_values)

    # Prepare sampled sets
    if reuse_samples is None:
        samples = np.zeros((n_mc, int(k)), dtype=int)
        for t in range(n_mc):
            if phi is None:
                # Fallback: sample uniformly without replacement using pi as weights
                # convert pi to probabilities for selection without replacement
                probs = pi_values / pi_values.sum() if pi_values.sum() > 0 else None
                # use numpy choice without replacement
                if probs is None:
                    samples[t] = np.random.choice(np.arange(1, m_local+1), size=int(k), replace=False)
                else:
                    samples[t] = np.random.choice(np.arange(1, m_local+1), size=int(k), replace=False, p=probs)
            else:
                samples[t] = sample_mallows_top_k_rsm(m_local, phi, int(k), center=center)
    else:
        samples = np.asarray(reuse_samples, dtype=int)

    # For each sampled top-k set, simulate earlier applicants jointly via multinomial
    # and check whether all schools in the set have at least c earlier applicants.
    indicators = np.zeros(samples.shape[0], dtype=float)
    n_prev = max(0, int(ell) - 1)
    if n_prev == 0:
        # If no previous applicants, nobody is rejected
        return 0.0, 0.0, samples

    # Precompute probability vector for multinomial sampling
    probs = pi_values / pi_values.sum() if pi_values.sum() > 0 else np.ones_like(pi_values) / len(pi_values)

    # Draw multinomial counts for all MC samples in one vectorized call
    try:
        counts = np.random.multinomial(n_prev, probs, size=samples.shape[0])
    except Exception:
        # Fallback to per-sample draws if vectorized call not supported
        counts = np.array([np.random.multinomial(n_prev, probs) for _ in range(samples.shape[0])])

    # For each sampled top-k set, check if all chosen schools have counts >= c
    # samples is shape (n_mc, k); convert to 0-based and index counts
    idx = samples - 1  # shape (n_mc, k)
    # Use advanced indexing: counts[np.arange(n_mc)[:,None], idx] -> shape (n_mc, k)
    n_mc = samples.shape[0]
    rows = np.arange(n_mc)[:, None]
    chosen_counts = counts[rows, idx]
    indicators = np.all(chosen_counts >= c, axis=1).astype(float)

    mean_ind = float(np.mean(indicators))
    stderr = float(np.std(indicators) / np.sqrt(len(indicators))) if len(indicators) > 0 else 0.0
    return mean_ind, stderr, samples


def prob_unmatched_vectorized_mc(ell_array, pi_values, c, k, m=None, phi=None, center=None, n_mc=5000, n_workers=1):
    """
    Vectorized Monte-Carlo estimator for multiple lottery numbers. Samples
    top-k lists once (per call) and reuses them across ell values to amortize cost.
    Returns array of estimates matching ell_array.
    """
    # Pre-sample top-k sets
    mean_estimates = []
    _, _, samples = prob_unmatched_mc_single(ell=1, pi_values=pi_values, c=c, k=k, m=m, phi=phi, center=center, n_mc=n_mc, reuse_samples=None)
    # Now compute products for each ell using the same samples
    samples = np.asarray(samples, dtype=int)
    for ell in ell_array:
        # Use the fast per-ell MC routine reusing the same sampled top-k sets
        mean_ind, stderr, _ = prob_unmatched_mc_single(ell, pi_values, c, k, m=m, phi=phi, center=center, n_mc=n_mc, reuse_samples=samples)
        mean_estimates.append(float(mean_ind))
    return np.array(mean_estimates)


def prob_unmatched_vectorized_variable_mc(ell_array, pi_values, c, k_max, alpha=2.0, min_k=1, k_dist='power_tail', k_std=None, n_mc=5000, m=None, phi=None, center=None):
    """
    Variable-k Monte-Carlo estimator: compute P(unmatched | ell) by mixing
    MC estimates computed per k with P(k).
    """
    k_values, k_probs = generate_list_length_distribution(k_max, alpha, min_k, k_dist=k_dist, k_std=k_std)
    # For each k with non-negligible probability compute MC estimate and mix
    estimates_per_k = {}
    for k, k_prob in zip(k_values, k_probs):
        if k_prob <= 0:
            continue
        ests = prob_unmatched_vectorized_mc(ell_array, pi_values, c, int(k), m=m, phi=phi, center=center, n_mc=n_mc)
        estimates_per_k[int(k)] = ests * k_prob

    # Sum weighted estimates across k
    total = np.zeros_like(ell_array, dtype=float)
    for est in estimates_per_k.values():
        total += est
    return total


# ==========================================
# DA-based Monte-Carlo estimator
# ==========================================
def prob_unmatched_mc_da_single(ell, c, k, m, phi, center=None, n_mc=1000):
    """
    Monte-Carlo estimator that simulates the DA process among the ell-1 earlier
    students to determine whether a focal student with lottery ell and a sampled
    top-k list would be rejected by all k schools.

    This explicitly samples earlier students' top-k lists and runs a greedy
    acceptance process (students in random order propose to their ranked schools
    until accepted or exhausted). Returns (mean, stderr).
    """
    n_prev = max(0, int(ell) - 1)
    if n_prev == 0:
        return 0.0, 0.0

    rng = np.random.default_rng()

    indicators = []
    for _ in range(n_mc):
        # sample focal student's list
        focal = sample_mallows_top_k_rsm(m, phi, k, center=center)

        # sample n_prev earlier students' lists
        earlier_lists = [sample_mallows_top_k_rsm(m, phi, k, center=center) for __ in range(n_prev)]

        # capacities copy
        caps = np.full(m, c, dtype=int)

        # process earlier students in a random order (their relative lotteries)
        order = rng.permutation(n_prev)
        for idx in order:
            lst = earlier_lists[idx]
            accepted = False
            for s in lst:
                s_idx = int(s) - 1
                if caps[s_idx] > 0:
                    caps[s_idx] -= 1
                    accepted = True
                    break
            # if not accepted, they exhaust list

        # after processing earlier students, check whether all focal schools are full
        focal_idx = focal - 1
        all_full = all(caps[i] <= 0 for i in focal_idx)
        indicators.append(1.0 if all_full else 0.0)

    arr = np.array(indicators)
    return float(arr.mean()), float(arr.std() / np.sqrt(len(arr)))


def prob_unmatched_vectorized_mc_da(ell_array, c, k, m, phi, center=None, n_mc=500):
    """
    Vectorized wrapper for DA-based MC estimator. Runs `prob_unmatched_mc_da_single`
    for each ell in ell_array. This is expensive; choose small n_mc for large ell.
    """
    results = []
    for ell in ell_array:
        mean, stderr = prob_unmatched_mc_da_single(ell, c, k, m, phi, center=center, n_mc=n_mc)
        results.append(mean)
    return np.array(results)

# ==========================================
# Step 3: Plotting functions (OPTIMIZED)
# ==========================================

def plot_effect_of_phi(n=72000, m=533, c=156, k=12, seed=None, k_dist=None, k_std=None, show=False):
    """Plot P(unmatched | ell) for different phi values - OPTIMIZED"""
    
    phi_values = [0.0, 0.3, 0.5, 0.7, 1.0]
    
    ell_range = np.linspace(1, n, 250, dtype=int)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    print("Starting computation...")
    total_start = time.time()
    
    for phi in tqdm(phi_values, desc="Computing φ curves"):
        start = time.time()
        
        # Compute pi_r values
        if phi == 0.0:
            pi_vals = np.array([1.0 if r <= k else 0.0 for r in range(1, m+1)])
        elif phi == 1.0:
            pi_vals = np.full(m, k/m)
        else:
            pi_vals = compute_pi(phi, k, m, n_samples=1000, n_workers=8, variable=False)  
        
        # Compute P(unmatched) - no inner progress bar needed now
        pi_values = normalize_pi(pi_vals)
        probs = prob_unmatched_vectorized(ell_range, pi_values, c, k)

        ax1.plot(ell_range, probs, label=f'φ = {phi}', linewidth=2)
        
        elapsed = time.time() - start
        tqdm.write(f"   φ = {phi} done in {elapsed:.1f}s")
    
    ax1.set_xlabel('Lottery Number ℓ', fontsize=12)
    ax1.set_ylabel('P(unmatched | ℓ)', fontsize=12)
    ax1.set_title(f'Effect of φ (n={n}, m={m}, c={c}, k={k})', fontsize=13)
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Average unmatched probability vs phi
    phi_range = np.linspace(0, 1, 15) 
    avg_unmatched = []
    
    for phi in tqdm(phi_range, desc="Computing avg vs φ"):
        if phi < 0.05:
            pi_vals = np.array([1.0 if r <= k else 0.0 for r in range(1, m+1)])
        elif phi > 0.95:
            pi_vals = np.full(m, k/m)
        else:
            pi_vals = compute_pi(phi, k, m, n_samples=500, n_workers=8, variable=False) 
        
        pi_values = normalize_pi(pi_vals)
        probs = prob_unmatched_vectorized(ell_range, pi_values, c, k)
        avg_unmatched.append(np.mean(probs))
    
    ax2.plot(phi_range, avg_unmatched, 'o-', linewidth=2, markersize=6)
    ax2.set_xlabel('φ (preference correlation)', fontsize=12)
    ax2.set_ylabel('Average P(unmatched)', fontsize=12)
    ax2.set_title('Average Unmatched Probability vs φ', fontsize=13)
    ax2.grid(True, alpha=0.3)
    
    total_elapsed = time.time() - total_start
    print(f"\n Total time: {total_elapsed:.1f}s ({total_elapsed/60:.1f} min)")
    
    suffix = _make_filename_suffix(seed=seed, k_dist=k_dist, k_std=k_std)
    fname = f'output_plots/effect_of_phi_varied_k_rsm{suffix}.png'
    plt.tight_layout()
    save_figure(fname)
    print(f" Saved: {fname}")
    if show:
        plt.show()
    else:
        plt.close()

def plot_effect_of_k(n=72000, m=533, c=156, phi=0.5, seed=None, k_dist=None, k_std=None, show=False):
    """Plot P(unmatched | ell) for different k values - OPTIMIZED"""
    
    k_values = [3, 5, 10, 15, 20]
    
    # KEY OPTIMIZATION: Sample only 200 lottery numbers
    ell_range = np.linspace(1, n, 250, dtype=int)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    for k in tqdm(k_values, desc="Computing k curves"):
        if phi == 0.0:
            pi_vals = np.array([1.0 if r <= k else 0.0 for r in range(1, m+1)])
        elif phi == 1.0:
            pi_vals = np.full(m, k/m)
        else:
            pi_vals = compute_pi(phi, k, m, n_samples=1000, n_workers=8, variable=False)

        pi_values = normalize_pi(pi_vals)
        probs = prob_unmatched_vectorized(ell_range, pi_values, c, k)
        ax1.plot(ell_range, probs, label=f'k = {k}', linewidth=2)
        tqdm.write(f"   k = {k} completed")
    
    ax1.set_xlabel('Lottery Number ℓ', fontsize=12)
    ax1.set_ylabel('P(unmatched | ℓ)', fontsize=12)
    ax1.set_title(f'Effect of k (n={n}, m={m}, c={c}, φ={phi})', fontsize=13)
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Average unmatched probability vs k
    k_range = range(1, 25)
    avg_unmatched = []
    
    for k in tqdm(k_range, desc="Computing avg vs k"):
        if phi == 0.0:
            pi_vals = np.array([1.0 if r <= k else 0.0 for r in range(1, m+1)])
        elif phi == 1.0:
            pi_vals = np.full(m, k/m)
        else:
            pi_vals = compute_pi(phi, k, m, n_samples=1000, n_workers=8, variable=False)

        pi_values = normalize_pi(pi_vals)
        probs = prob_unmatched_vectorized(ell_range, pi_values, c, k)
        avg_unmatched.append(np.mean(probs))
    
    ax2.plot(list(k_range), avg_unmatched, 'o-', linewidth=2, markersize=6)
    ax2.set_xlabel('k (list length)', fontsize=12)
    ax2.set_ylabel('Average P(unmatched)', fontsize=12)
    ax2.set_title('Average Unmatched Probability vs k', fontsize=13)
    ax2.grid(True, alpha=0.3)
    
    suffix = _make_filename_suffix(seed=seed, k_dist=k_dist, k_std=k_std)
    fname = f'output_plots/effect_of_k_varied_k_rsm{suffix}.png'
    plt.tight_layout()
    save_figure(fname)
    print(f" Saved: {fname}")
    if show:
        plt.show()
    else:
        plt.close()

def plot_pi_r_distribution(phi_values=[0.0, 0.3, 0.5, 0.7, 1.0], k=12, m=533, seed=None, k_dist=None, k_std=None, show=False):
    """Plot how pi_r varies with rank for different phi - OPTIMIZED"""
    
    fig, ax = plt.subplots(figsize=(10, 6))
    ranks = range(1, min(m+1, 50))  # Plot first 50 ranks
    
    for phi in tqdm(phi_values, desc="Computing πᵣ distributions"):
        if phi == 0.0:
            pi_vals = [1.0 if r <= k else 0.0 for r in ranks]
        elif phi == 1.0:
            pi_vals = [k/m for r in ranks]
        else:
            full_pi = compute_pi(phi, k, m, n_samples=2000, n_workers=8, variable=False)  # Reduced
            pi_vals = [full_pi[r-1] for r in ranks]
        
        ax.plot(ranks, pi_vals, 'o-', label=f'φ = {phi}', linewidth=2, markersize=5)
        tqdm.write(f"   φ = {phi} completed")
    
    ax.set_xlabel('School Rank r in σ*', fontsize=12)
    ax.set_ylabel('πᵣ(φ, k) = P(rank r in top-k)', fontsize=12)
    ax.set_title(f'Application Probability vs School Rank (k={k}, m={m})', fontsize=13)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    suffix = _make_filename_suffix(seed=seed, k_dist=k_dist, k_std=k_std)
    fname = f'output_plots/pi_r_distribution_varied_k_rsm{suffix}.png'
    plt.tight_layout()
    save_figure(fname)
    print(f" Saved: {fname}")
    if show:
        plt.show()
    else:
        plt.close()


def plot_effect_of_phi_variable(n=72000, m=533, c=156, k_max=12, alpha=2.0, min_k=1, k_dist='power_tail', k_std=None, seed=None, show=False):
    """Plot P(unmatched | ell) for different phi values with variable list lengths"""
    
    phi_values = [0.0, 0.3, 0.5, 0.7, 1.0]
    
    # Sample lottery numbers for plotting
    ell_range = np.linspace(1, n, 250, dtype=int)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    print("Starting computation with VARIABLE list lengths...")
    print(f"List length distribution: k ∈ [{min_k}, {k_max}], α = {alpha}")
    
    # Show the list length distribution
    k_vals, k_probs = generate_list_length_distribution(k_max, alpha, min_k, k_dist=k_dist, k_std=k_std)
    print(f"List length probabilities: {dict(zip(k_vals, k_probs.round(3)))}")
    
    total_start = time.time()
    
    for phi in tqdm(phi_values, desc="Computing φ curves (variable k)"):
        start = time.time()
        
        # Compute pi_r values using variable list lengths
        if phi == 0.0:
            # For perfect consensus, we need to weight by list length distribution
            pi_vals = np.zeros(m)
            for k, k_prob in zip(k_vals, k_probs):
                for r in range(1, min(k+1, m+1)):
                    pi_vals[r-1] += k_prob
        elif phi == 1.0:
            # For uniform preferences, compute expected applications per school
            expected_k = np.sum(k_vals * k_probs)
            pi_vals = np.full(m, expected_k/m)
        else:
            pi_vals = compute_pi(phi, k_max, m, n_samples=1000, n_workers=8, variable=True, alpha=alpha, min_k=min_k, k_dist=k_dist, k_std=k_std)
        
        # Compute P(unmatched) with variable k
        pi_values = normalize_pi(pi_vals)
        probs = prob_unmatched_vectorized_variable(ell_range, pi_values, c, k_max, alpha, min_k, k_dist=k_dist, k_std=k_std)

        ax1.plot(ell_range, probs, label=f'φ = {phi}', linewidth=2)
        
        elapsed = time.time() - start
        tqdm.write(f"   φ = {phi} done in {elapsed:.1f}s")
    
    ax1.set_xlabel('Lottery Number ℓ', fontsize=12)
    ax1.set_ylabel('P(unmatched | ℓ)', fontsize=12)
    ax1.set_title(f'Effect of φ - Variable k (n={n}, m={m}, c={c}, k_max={k_max}, α={alpha}, k_dist={k_dist})', fontsize=11)
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Average unmatched probability vs phi
    phi_range = np.linspace(0, 1, 15)
    avg_unmatched = []
    
    for phi in tqdm(phi_range, desc="Computing avg vs φ (variable k)"):
        if phi < 0.05:
            pi_vals = np.zeros(m)
            for k, k_prob in zip(k_vals, k_probs):
                for r in range(1, min(k+1, m+1)):
                    pi_vals[r-1] += k_prob
        elif phi > 0.95:
            expected_k = np.sum(k_vals * k_probs)
            pi_vals = np.full(m, expected_k/m)
        else:
            pi_vals = compute_pi(phi, k_max, m, n_samples=500, n_workers=8, variable=True, alpha=alpha, min_k=min_k, k_dist=k_dist, k_std=k_std)

        pi_values = normalize_pi(pi_vals)
        probs = prob_unmatched_vectorized_variable(ell_range, pi_values, c, k_max, alpha, min_k, k_dist=k_dist, k_std=k_std)
        avg_unmatched.append(np.mean(probs))
    
    ax2.plot(phi_range, avg_unmatched, 'o-', linewidth=2, markersize=6)
    ax2.set_xlabel('φ (preference correlation)', fontsize=12)
    ax2.set_ylabel('Average P(unmatched)', fontsize=12)
    ax2.set_title(f'Average Unmatched Probability vs φ\n(Variable k, α={alpha}, k_dist={k_dist})', fontsize=12)
    ax2.grid(True, alpha=0.3)
    
    total_elapsed = time.time() - total_start
    print(f"\n Total time: {total_elapsed:.1f}s ({total_elapsed/60:.1f} min)")
    
    suffix = _make_filename_suffix(seed=seed, k_dist=k_dist, k_std=k_std)
    fname = f'output_plots/effect_of_phi_variable_k_alpha{alpha}{suffix}.png'
    plt.tight_layout()
    save_figure(fname)
    print(f" Saved: {fname}")
    if show:
        plt.show()
    else:
        plt.close()

def plot_effect_of_alpha(n=72000, m=533, c=156, k_max=12, phi=0.5, k_dist='power_tail', k_std=None, seed=None, show=False):
    """Plot how the power law parameter α affects unmatched probability"""
    
    alpha_values = [0.5, 1.0, 1.5, 2.0, 3.0]
    ell_range = np.linspace(1, n, 250, dtype=int)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    print(f"Analyzing effect of α parameter (φ = {phi})...")
    # For alpha experiments, normal k distributions don't make sense (alpha
    # controls power-law behavior). If user passed 'normal', override to a
    # centered power distribution which respects alpha.
    if k_dist == 'normal':
        print("Note: 'normal' k_dist is incompatible with varying alpha; switching to 'centered_power' for this experiment.")
        k_dist = 'centered_power'
    
    for alpha in tqdm(alpha_values, desc="Computing α curves"):
        start = time.time()

        # Show distribution for this alpha
        k_vals, k_probs = generate_list_length_distribution(k_max, alpha, min_k=1, k_dist=k_dist, k_std=k_std)
        tqdm.write(f"  α = {alpha}: P(k=k_max) = {k_probs[-1]:.3f}")

        pi_values = compute_pi(phi, k_max, m, n_samples=1000, n_workers=8, variable=True, alpha=alpha, min_k=1, k_dist=k_dist, k_std=k_std)
        pi_values = normalize_pi(pi_values)
        probs = prob_unmatched_vectorized_variable(ell_range, pi_values, c, k_max, alpha, min_k=1, k_dist=k_dist, k_std=k_std)

        ax1.plot(ell_range, probs, label=f'α = {alpha}', linewidth=2)

        elapsed = time.time() - start
        tqdm.write(f"   α = {alpha} done in {elapsed:.1f}s")
    
    ax1.set_xlabel('Lottery Number ℓ', fontsize=12)
    ax1.set_ylabel('P(unmatched | ℓ)', fontsize=12)
    ax1.set_title(f'Effect of Power Law Parameter α\n(n={n}, m={m}, c={c}, φ={phi}, k_max={k_max}, k_dist={k_dist})', fontsize=12)
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: List length distributions
    for alpha in alpha_values:
        k_vals, k_probs = generate_list_length_distribution(k_max, alpha, min_k=1, k_dist=k_dist, k_std=k_std)
        ax2.plot(k_vals, k_probs, 'o-', label=f'α = {alpha}', linewidth=2, markersize=5)
    
    ax2.set_xlabel('List Length k', fontsize=12)
    ax2.set_ylabel('Probability P(k)', fontsize=12)
    ax2.set_title(f'List Length Distributions (k_max={k_max}, k_dist={k_dist})', fontsize=12)
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    suffix = _make_filename_suffix(seed=seed, k_dist=k_dist, k_std=k_std)
    fname = f'output_plots/effect_of_alpha_variable_k{suffix}.png'
    plt.tight_layout()
    save_figure(fname)
    print(f" Saved: {fname}")
    if show:
        plt.show()
    else:
        plt.close()


def plot_effect_of_kstd(n=72000, m=533, c=156, k_max=12, phi=0.5,
                        k_dist='normal', k_std_values=None, seed=None, n_samples=1000, n_workers=8, show=False):
    """Sweep k_std for the 'normal' k distribution and plot unmatched curves and k-distributions.

    This mirrors the α experiment but varies the standard deviation of the
    truncated-normal discrete distribution over list lengths.
    """
    if k_std_values is None:
        k_std_values = [0.5, 1.0, 2.0, 4.0]

    # Ensure we are working with the normal k distribution
    if k_dist != 'normal':
        print("Note: effect_kstd experiment requires k_dist='normal'. Overriding k_dist to 'normal'.")
        k_dist = 'normal'

    ell_range = np.linspace(1, n, 250, dtype=int)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    print(f"Starting k_std sweep (k_max={k_max}, φ={phi})...")

    for k_std in tqdm(k_std_values, desc="Computing k_std curves"):
        start = time.time()

        # Compute list-length distribution for this k_std
        k_vals, k_probs = generate_list_length_distribution(k_max, alpha=2.0, min_k=1, k_dist='normal', k_std=k_std)

        # Compute pi_values under variable-k normal distribution
        if phi < 0.05:
            pi_vals = np.zeros(m)
            for kk, kp in zip(k_vals, k_probs):
                for r in range(1, min(int(kk) + 1, m+1)):
                    pi_vals[r-1] += kp
        elif phi > 0.95:
            expected_k = np.sum(k_vals * k_probs)
            pi_vals = np.full(m, expected_k / m)
        else:
            pi_vals = compute_pi(phi, k_max, m, n_samples=n_samples, n_workers=n_workers,
                                 variable=True, alpha=2.0, min_k=1, k_dist='normal', k_std=k_std)

        pi_values = normalize_pi(pi_vals)
        probs = prob_unmatched_vectorized_variable(ell_range, pi_values, c, k_max, alpha=2.0, min_k=1, k_dist='normal', k_std=k_std)

        ax1.plot(ell_range, probs, label=f'σ={k_std}', linewidth=2)

        # Also show distribution over k on ax2
        ax2.plot(k_vals, k_probs, 'o-', label=f'σ={k_std}', linewidth=2, markersize=5)

        elapsed = time.time() - start
        tqdm.write(f"   σ = {k_std} done in {elapsed:.1f}s")

    ax1.set_xlabel('Lottery Number ℓ', fontsize=12)
    ax1.set_ylabel('P(unmatched | ℓ)', fontsize=12)
    ax1.set_title(f'Effect of normal k std (n={n}, m={m}, c={c}, φ={phi}, k_max={k_max})', fontsize=12)
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    ax2.set_xlabel('List Length k', fontsize=12)
    ax2.set_ylabel('Probability P(k)', fontsize=12)
    ax2.set_title(f'List Length Distributions (normal, varying σ, k_max={k_max})', fontsize=12)
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # Build a filename-friendly token for the list of k_std values we swept
    try:
        kstd_tokens = [f"{float(s):.2f}".replace('.', 'p') for s in k_std_values]
    except Exception:
        kstd_tokens = [str(s).replace('.', 'p') for s in k_std_values]
    kstd_part = "_".join(kstd_tokens)
    suffix = _make_filename_suffix(seed=seed, k_dist=k_dist)
    fname = f'output_plots/effect_of_kstd_normal_k_stds-{kstd_part}{suffix}.png'
    plt.tight_layout()
    save_figure(fname)
    print(f" Saved: {fname}")
    if show:
        plt.show()
    else:
        plt.close()

def compare_fixed_vs_variable_k(n=72000, m=533, c=156, k_max=12, phi=0.5, alpha=2.0, k_dist='power_tail', k_std=None, seed=None, show=False):
    """Compare fixed k vs variable k models"""
    
    ell_range = np.linspace(1, n, 250, dtype=int)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    print("Comparing Fixed vs Variable List Length Models...")
    
    # Fixed k model
    pi_vals_fixed = compute_pi(phi, k_max, m, n_samples=1000, n_workers=8, variable=False)
    pi_vals_fixed = normalize_pi(pi_vals_fixed)
    probs_fixed = prob_unmatched_vectorized(ell_range, pi_vals_fixed, c, k_max)

    # Variable k model
    pi_vals_variable = compute_pi(phi, k_max, m, n_samples=1000, n_workers=8, variable=True, alpha=alpha, min_k=1, k_dist=k_dist, k_std=k_std)
    pi_vals_variable = normalize_pi(pi_vals_variable)
    probs_variable = prob_unmatched_vectorized_variable(ell_range, pi_vals_variable, c, k_max, alpha, min_k=1, k_dist=k_dist, k_std=k_std)
    
    ax.plot(ell_range, probs_fixed, label=f'Fixed k = {k_max}', linewidth=2)
    ax.plot(ell_range, probs_variable, label=f'Variable k (α = {alpha})', linewidth=2, linestyle='--')
    
    ax.set_xlabel('Lottery Number ℓ', fontsize=12)
    ax.set_ylabel('P(unmatched | ℓ)', fontsize=12)
    ax.set_title(f'Fixed vs Variable List Length Models\n(n={n}, m={m}, c={c}, φ={phi}, k_max={k_max}, k_dist={k_dist})', fontsize=13)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Show difference
    difference = probs_variable - probs_fixed
    print(f"Average difference (variable - fixed): {np.mean(difference):.4f}")
    print(f"Max absolute difference: {np.max(np.abs(difference)):.4f}")
    
    suffix = _make_filename_suffix(seed=seed, k_dist=k_dist, k_std=k_std)
    fname = f'output_plots/fixed_vs_variable_k_comparison{suffix}.png'
    plt.tight_layout()
    save_figure(fname)
    print(f" Saved: {fname}")
    if show:
        plt.show()
    else:
        plt.close()


def plot_fixed_vs_variable_over_phi(r=5, n=72000, m=533, c=156, k=12, k_max=12,
                                    alpha=2.0, n_samples=1000, n_workers=4, k_dist='power_tail', k_std=None, seed=None, show=False):
    """
    Plot 2*r lines: for r phi values show unmatched probability curves for
    (1) fixed k, and (2) variable k (power-law distribution) so the role of
    varying k can be compared across φ levels.
    """
    # Prepare phi grid avoiding exact 0 and 1 so we use sampling path
    phi_values = np.linspace(0.05, 0.95, r)
    ell_range = np.linspace(1, n, 200, dtype=int)

    # Precompute list-length distribution for variable-k model
    k_vals, k_probs = generate_list_length_distribution(k_max, alpha, min_k=1, k_dist=k_dist, k_std=k_std)

    fig, ax = plt.subplots(figsize=(10, 6))

    for phi_val in tqdm(phi_values, desc="Computing fixed vs variable for φ"):
        # Fixed k pi
        if phi_val < 0.05:
            pi_fixed = np.array([1.0 if rnk <= k else 0.0 for rnk in range(1, m+1)])
        elif phi_val > 0.95:
            pi_fixed = np.full(m, k/m)
        else:
            pi_fixed = compute_pi(phi_val, k, m, n_samples=n_samples, n_workers=n_workers, variable=False)
        pi_fixed = normalize_pi(pi_fixed)
        probs_fixed = prob_unmatched_vectorized(ell_range, pi_fixed, c, k)

        # Variable k pi
        if phi_val < 0.05:
            pi_var = np.zeros(m)
            for kk, kp in zip(k_vals, k_probs):
                for rnk in range(1, min(int(kk)+1, m+1)):
                    pi_var[rnk-1] += kp
        elif phi_val > 0.95:
            expected_k = np.sum(k_vals * k_probs)
            pi_var = np.full(m, expected_k / m)
        else:
            pi_var = compute_pi(phi_val, k_max, m, n_samples=n_samples, n_workers=n_workers, variable=True, alpha=alpha, min_k=1, k_dist=k_dist, k_std=k_std)

        pi_var = normalize_pi(pi_var)
        probs_var = prob_unmatched_vectorized_variable(ell_range, pi_var, c, k_max, alpha, min_k=1, k_dist=k_dist, k_std=k_std)

        # Plot both lines for this phi
        ax.plot(ell_range, probs_fixed, label=f'φ={phi_val:.2f} (fixed k={k})', linestyle='-')
        ax.plot(ell_range, probs_var, label=f'φ={phi_val:.2f} (variable k)', linestyle='--')

    ax.set_xlabel('Lottery Number ℓ', fontsize=12)
    ax.set_ylabel('P(unmatched | ℓ)', fontsize=12)
    ax.set_title(f'Fixed vs Variable k across φ (r={r})\n(n={n}, m={m}, c={c}, k={k}, k_max={k_max}, α={alpha}, k_dist={k_dist})', fontsize=13)
    ax.legend(ncol=2, fontsize=9)
    ax.grid(True, alpha=0.3)

    suffix = _make_filename_suffix(seed=seed, k_dist=k_dist, k_std=k_std)
    fname = f'output_plots/fixed_vs_variable_over_phi{suffix}.png'
    plt.tight_layout()
    save_figure(fname)
    print(f" Saved: {fname}")
    if show:
        plt.show()
    else:
        plt.close()

def plot_school_utilization(n=72000, m=533, c=156, k=12, phi=0.5,
                           variable_k=False, alpha=2.0, k_dist='power_tail', 
                           k_std=None, n_samples=1000, n_workers=4, 
                           seed=None, show=False):
    """
    Plot cumulative distribution of school utilization rates.
    
    Shows: What percentage of schools have at least X% utilization?
    Utilization = (expected students assigned) / capacity
    """
    print(f"Computing school utilization (φ={phi}, k={k}, variable_k={variable_k})...")
    
    # Compute pi_r distribution
    if variable_k:
        pi_vals = compute_pi(phi, k, m, n_samples=n_samples, n_workers=n_workers,
                            variable=True, alpha=alpha, min_k=1, 
                            k_dist=k_dist, k_std=k_std)
        # Adjust k to expected value for calculations
        k_vals, k_probs = generate_list_length_distribution(
            k, alpha, min_k=1, k_dist=k_dist, k_std=k_std
        )
        k_effective = np.sum(k_vals * k_probs)
    else:
        pi_vals = compute_pi(phi, k, m, n_samples=n_samples, 
                            n_workers=n_workers, variable=False)
        k_effective = k
    
    pi_vals = normalize_pi(pi_vals)
    
    # Estimate school utilization
    # Expected applications per school
    total_apps = n * k_effective
    expected_apps_per_school = pi_vals * total_apps
    
    # Fill counts: min of (expected applications, capacity)
    # This is optimistic - assumes perfect matching
    fill_counts = np.minimum(expected_apps_per_school, c)
    
    # Utilization rate per school
    utilization = fill_counts / c
    
    # Sort utilization rates (descending for cumulative plot)
    sorted_util = np.sort(utilization)[::-1]
    
    # Create cumulative distribution
    util_thresholds = np.linspace(0, 1, 101)  # 0%, 1%, ..., 100%
    cumulative_counts = np.array([
        np.sum(sorted_util >= threshold) for threshold in util_thresholds
    ])
    cumulative_fraction = cumulative_counts / m  # Fraction of schools
    
    # Create plots
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    # Plot 1: Cumulative distribution (main plot you wanted)
    ax = axes[0]
    ax.plot(util_thresholds * 100, cumulative_fraction * 100, 
            linewidth=3, color='#2196F3')
    ax.fill_between(util_thresholds * 100, 0, cumulative_fraction * 100, 
                    alpha=0.3, color='#2196F3')
    ax.set_xlabel('Minimum Utilization Rate (%)', fontsize=12)
    ax.set_ylabel('Schools with ≥ X% Utilization (%)', fontsize=12)
    ax.set_title(f'Cumulative School Utilization\n(φ={phi}, k={k}, var_k={variable_k})', 
                fontsize=13)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    
    # Add reference lines
    ax.axhline(y=50, color='gray', linestyle='--', alpha=0.5, label='50% of schools')
    ax.axvline(x=80, color='red', linestyle='--', alpha=0.5, label='80% utilization')
    ax.legend(fontsize=9)
    
    # Plot 2: Histogram of utilization rates
    ax = axes[1]
    ax.hist(utilization * 100, bins=30, edgecolor='black', alpha=0.7, color='#FF9800')
    ax.axvline(np.mean(utilization) * 100, color='red', linestyle='--', 
               linewidth=2, label=f'Mean: {np.mean(utilization)*100:.1f}%')
    ax.axvline(np.median(utilization) * 100, color='green', linestyle='--',
               linewidth=2, label=f'Median: {np.median(utilization)*100:.1f}%')
    ax.set_xlabel('Utilization Rate (%)', fontsize=12)
    ax.set_ylabel('Number of Schools', fontsize=12)
    ax.set_title('Distribution of School Utilization', fontsize=13)
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    
    
    plt.tight_layout()
    
    # Save figure
    suffix = _make_filename_suffix(seed=seed, k_dist=k_dist, k_std=k_std)
    var_str = f"_var_k" if variable_k else ""
    fname = f'output_plots/school_utilization_phi{phi}_k{k}{var_str}{suffix}.png'
    save_figure(fname)
    print(f"✓ Saved: {fname}")
    
    if show:
        plt.show()
    else:
        plt.close()
    
    return utilization, fill_counts

def plot_utilization_vs_phi(phi_values=[0.3, 0.5, 0.7], n=72000, m=533, c=156, k=12,
                           variable_k=False, alpha=2.0, k_dist='power_tail',
                           n_samples=1000, n_workers=4, seed=None, show=False):
    """
    Compare school utilization across different phi values.
    Shows how preference concentration affects capacity utilization.
    """
    print(f"Comparing utilization across φ values...")
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    utilization_data = {}
    
    for phi in tqdm(phi_values, desc="Computing utilization for φ"):
        # Compute pi_r
        if variable_k:
            pi_vals = compute_pi(phi, k, m, n_samples=n_samples, n_workers=n_workers,
                                variable=True, alpha=alpha, min_k=1, k_dist=k_dist)
            k_vals, k_probs = generate_list_length_distribution(k, alpha, min_k=1, k_dist=k_dist)
            k_effective = np.sum(k_vals * k_probs)
        else:
            pi_vals = compute_pi(phi, k, m, n_samples=n_samples, 
                                n_workers=n_workers, variable=False)
            k_effective = k
        
        pi_vals = normalize_pi(pi_vals)
        
        # Estimate utilization
        expected_apps = pi_vals * n * k_effective
        fill_counts = np.minimum(expected_apps, c)
        utilization = fill_counts / c
        
        utilization_data[phi] = utilization
        
        # Sort and create cumulative
        sorted_util = np.sort(utilization)[::-1]
        util_thresholds = np.linspace(0, 1, 101)
        cumulative_fraction = np.array([
            np.sum(sorted_util >= t) / m for t in util_thresholds
        ])
        
        # Plot 1: Cumulative curves
        axes[0].plot(util_thresholds * 100, cumulative_fraction * 100,
                    linewidth=2, label=f'φ = {phi}')
    
    axes[0].set_xlabel('Minimum Utilization Rate (%)', fontsize=12)
    axes[0].set_ylabel('Schools with ≥ X% Utilization (%)', fontsize=12)
    axes[0].set_title(f'School Utilization vs Preference Concentration\n(n={n}, m={m}, c={c}, k={k})', 
                     fontsize=13)
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    axes[0].set_xlim(0, 100)
    axes[0].set_ylim(0, 100)
    
    # Plot 2: Mean utilization vs phi
    phi_plot = list(utilization_data.keys())
    mean_utils = [np.mean(utilization_data[p]) * 100 for p in phi_plot]
    std_utils = [np.std(utilization_data[p]) * 100 for p in phi_plot]
    
    axes[1].errorbar(phi_plot, mean_utils, yerr=std_utils, 
                     fmt='o-', linewidth=2, markersize=8, capsize=5)
    axes[1].set_xlabel('φ (preference correlation)', fontsize=12)
    axes[1].set_ylabel('Mean Utilization Rate (%)', fontsize=12)
    axes[1].set_title('Average Utilization vs φ', fontsize=13)
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    
    # Save
    suffix = _make_filename_suffix(seed=seed, k_dist=k_dist)
    var_str = f"_var_k" if variable_k else ""
    fname = f'output_plots/utilization_vs_phi{var_str}{suffix}.png'
    save_figure(fname)
    print(f"✓ Saved: {fname}")
    
    if show:
        plt.show()
    else:
        plt.close()

def plot_effect_of_components(y_values=[1,2,3,5,10], n=72000, m=533, c=156, k=12,
                              n_samples=2000, n_workers=4, random_seed=None, show=False):
    """
    For each y in y_values, build a mixture of y Mallows components with
    distinct central rankings and (arbitrary) phi values, compute pi,
    evaluate average unmatched probability and plot avg unmatched vs y.
    """
    if random_seed is not None:
        np.random.seed(random_seed)

    ell_range = np.linspace(1, n, 250, dtype=int)
    avg_unmatched_list = []

    for y in tqdm(y_values, desc="Varying #components y"):
        # Create y phis (spread in (0.1, 0.9)) and random centers
        phis = list(np.linspace(0.1, 0.9, y))
        weights = [1.0 / y] * y
        centers = [ (np.random.permutation(np.arange(1, m+1)).tolist()) for _ in range(y) ]

        # Compute pi for this mixture (fixed k here)
        pi_vals = compute_pi(None, k, m, n_samples=n_samples, n_workers=n_workers,
                             variable=False, mixture_phis=phis, mixture_weights=weights,
                             mixture_centers=centers)
        pi_vals = normalize_pi(pi_vals)

        # Compute avg unmatched probability over ell_range
        probs = prob_unmatched_vectorized(ell_range, pi_vals, c, k)
        avg_unmatched = float(np.mean(probs))
        avg_unmatched_list.append(avg_unmatched)

        tqdm.write(f" y={y}: avg_unmatched={avg_unmatched:.4f}")

    # Plot results
    _ , ax = plt.subplots(figsize=(8,5))
    ax.plot(y_values, avg_unmatched_list, 'o-', linewidth=2)
    ax.set_xlabel('Number of mixture components y', fontsize=12)
    ax.set_ylabel('Average P(unmatched)', fontsize=12)
    ax.set_title(f'Effect of number of Mallows components (y) on avg unmatched\n(n={n}, m={m}, c={c}, k={k})', fontsize=12)
    ax.grid(True, alpha=0.3)
    # include random seed in filename if provided
    suffix = _make_filename_suffix(seed=random_seed)
    fname = f'output_plots/effect_of_components_y{suffix}.png'
    plt.tight_layout()
    save_figure(fname)
    print(f" Saved: {fname}")
    if show:
        plt.show()
    else:
        plt.close()

# ==========================================
# Run the plots
# ==========================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run school choice Mallows experiments and plots.')
    parser.add_argument('--plots', nargs='+', default=None,
                        help=('Which plots to run. Options: effect_phi, effect_k, pi_r_dist, '
                              'effect_phi_variable, effect_alpha, compare_fixed_variable, effect_components, all'))
    parser.add_argument('--quick', action='store_true', help='Run quick (reduced-sample) versions for testing')
    parser.add_argument('--n_workers', type=int, default=min(4, max(1, mp.cpu_count()-1)),
                        help='Number of worker processes for sampling (default: cpu_count-1, capped at 4)')
    parser.add_argument('--seed', type=int, default=42, help='Random seed for reproducibility')
    parser.add_argument('--k_dist', choices=['power_tail', 'centered_power', 'normal'], default='power_tail',
                        help='Distribution for variable list length (default: power_tail)')
    parser.add_argument('--k_std', type=float, default=None, help='Std dev for normal k distribution (optional)')
    parser.add_argument('--show', action='store_true', help='Also display plots interactively (default: save only)')
    parser.add_argument('--self_test', action='store_true', help='Run internal self tests and exit')
    parser.add_argument('--mixture_y', type=int, default=0, help='If >0, use a mixture of y Mallows components')
    parser.add_argument('--mixture_phis', type=str, default=None, help='Comma-separated list of phi values for mixture components (overrides auto-generation)')
    parser.add_argument('--mixture_weights', type=str, default=None, help='Comma-separated mixing weights for components (must match mixture_phis or mixture_y)')
    parser.add_argument('--mixture_centers_random', action='store_true', help='If set, generate random centers for each mixture component')
    
    # NYC simulation arguments
    parser.add_argument('--nyc_sim', action='store_true', help='Run NYC school choice simulation with real rankings')
    parser.add_argument('--aggregate_type', choices=['residential', 'language', 'zip'], default='residential',
                        help='Type of NYC rankings to use (default: residential)')
    parser.add_argument('--phi', type=float, default=0.8, help='Phi parameter for NYC simulation (default: 0.8)')
    parser.add_argument('--k_values', type=str, default='3,6,9,12,15,18', help='Comma-separated k values to test (default: 3,6,9,12,15,18)')
    parser.add_argument('--n', type=int, default=72000, help='Number of students (default: 72000)')
    parser.add_argument('--capacity', type=int, default=156, help='School capacity (default: 156)')
    parser.add_argument('--n_samples_nyc', type=int, default=500, help='Number of samples for NYC pi_r computation (default: 500, use 2000+ for high accuracy)')
    
    args = parser.parse_args()

    np.random.seed(args.seed)
    
    # NYC simulation mode
    if args.nyc_sim:
        print("="*80)
        print("NYC SCHOOL CHOICE SIMULATION")
        print("="*80)
        
        k_vals = [int(k) for k in args.k_values.split(',')]
        
        results, m, categories = simulate_nyc_fixed_k(
            aggregate_type=args.aggregate_type,
            phi=args.phi,
            k_values=k_vals,
            n=args.n,
            c=args.capacity,
            n_samples=args.n_samples_nyc,
            n_workers=args.n_workers
        )
        
        print("\n" + "="*80)
        print("SUMMARY OF RESULTS")
        print("="*80)
        print(f"Aggregate type: {args.aggregate_type}")
        print(f"Phi: {args.phi}")
        print(f"Number of schools (m): {m}")
        print(f"Number of components: {len(categories)}")
        print(f"\nResults by k:")
        for k in k_vals:
            print(f"  k={k:2d}: avg unmatched = {results[k]['avg_unmatched']:.4f}, "
                  f"median unmatched = {results[k]['median_unmatched']:.4f}")
        
        exit(0)

    # Configure global mixture settings if requested
    if args.mixture_y and args.mixture_y > 0:
        # Determine phis
        if args.mixture_phis:
            try:
                phis = [float(x) for x in args.mixture_phis.split(',')]
            except Exception:
                print("Could not parse --mixture_phis. Expect comma-separated floats. Falling back to auto-generated phis.")
                phis = list(np.linspace(0.1, 0.9, args.mixture_y))
        else:
            phis = list(np.linspace(0.1, 0.9, args.mixture_y))

        # Determine weights
        if args.mixture_weights:
            try:
                weights = [float(x) for x in args.mixture_weights.split(',')]
            except Exception:
                print("Could not parse --mixture_weights. Falling back to uniform weights.")
                weights = [1.0 / len(phis)] * len(phis)
        else:
            weights = [1.0 / len(phis)] * len(phis)

        # Centers: either 'random' flag or None (lazy generation inside compute_pi)
        centers_flag = 'random' if args.mixture_centers_random else None

        # Set globals
        GLOBAL_MIXTURE_PHIS = phis
        GLOBAL_MIXTURE_WEIGHTS = weights
        GLOBAL_MIXTURE_CENTERS = centers_flag
        print(f"Using mixture Mallows model: y={len(phis)}, phis={phis}, weights={weights}, centers_random={args.mixture_centers_random}")

    # Run internal self tests if requested and exit
    if args.self_test:
        try:
            _internal_test_normal_kstd()
            print("All internal self tests passed.")
            exit(0)
        except AssertionError as e:
            print(f"Internal self test FAILED: {e}")
            exit(2)

    # Helper to choose n_samples based on quick flag
    def choose_samples(default):
        return max(100, default // 4) if args.quick else default

    available = {
    # Fixed-k experiments do not use variable-k distributions
    'effect_phi': lambda: plot_effect_of_phi(n=72000, m=533, c=156, k=12, seed=args.seed, show=args.show),
    'effect_k': lambda: plot_effect_of_k(n=72000, m=533, c=156, phi=0.5, seed=args.seed, show=args.show),
    'pi_r_dist': lambda: plot_pi_r_distribution(k=12, m=533, seed=args.seed, show=args.show),
    'effect_phi_variable': lambda: plot_effect_of_phi_variable(n=72000, m=533, c=156, k_max=12, alpha=2.0, 
                                                               min_k=1, k_dist=args.k_dist, k_std=args.k_std, seed=args.seed, show=args.show),
    'effect_alpha': lambda: plot_effect_of_alpha(n=72000, m=533, c=156, k_max=12, phi=0.5, 
                                                 k_dist=args.k_dist, k_std=args.k_std, seed=args.seed, show=args.show),
    'compare_fixed_variable': lambda: compare_fixed_vs_variable_k(n=72000, m=533, c=156, k_max=12, phi=0.5, 
                                                                  alpha=2.0, k_dist=args.k_dist, k_std=args.k_std, seed=args.seed, show=args.show),
    'fixed_vs_variable_over_phi': lambda: plot_fixed_vs_variable_over_phi(r=5, n=72000, m=533, c=156, k=12, k_max=12,
                                        alpha=2.0, n_samples=choose_samples(1000), n_workers=args.n_workers, k_dist=args.k_dist, k_std=args.k_std, seed=args.seed, show=args.show),
    'effect_components': lambda: plot_effect_of_components(y_values=[1,2,3,5,10], n=72000, m=533, c=156, k=12,
                               n_samples=choose_samples(2000), n_workers=args.n_workers, random_seed=args.seed, show=args.show),
    'effect_kstd': lambda: plot_effect_of_kstd(n=72000, m=533, c=156, k_max=12, phi=0.5, k_dist='normal', k_std_values=[0.5,1.0,2.0,4.0], 
                                               seed=args.seed, n_samples=choose_samples(1000), n_workers=args.n_workers, show=args.show),
    'school_utilization': lambda: plot_school_utilization(
        n=72000, m=533, c=156, k=12, phi=0.5, variable_k=False, n_samples=choose_samples(1000), 
        n_workers=args.n_workers, seed=args.seed, show=args.show),
    'utilization_vs_phi': lambda: plot_utilization_vs_phi(
        phi_values=[0.3, 0.5, 0.7], n=72000, m=533, c=156, k=12, variable_k=False, n_samples=choose_samples(1000), 
        n_workers=args.n_workers, seed=args.seed, show=args.show)
    }

    to_run = []
    if args.plots is None:
        print("No plots specified. Use --plots [names] or --plots all. Use --quick for a faster test run.")
        print("Available plots:")
        for k in sorted(available.keys()):
            print(f"  - {k}")
        print('\nExample: python school_mallows_sim.py --plots effect_phi effect_components --quick')
        exit(0)

    if 'all' in args.plots:
        to_run = list(available.keys())
    else:
        # Validate requested plot names
        for name in args.plots:
            if name not in available:
                print(f"Unknown plot name: {name}")
                exit(1)
        to_run = args.plots

    print("SCHOOL CHOICE SIMULATION - Variable List Lengths")
    print("Running plots:", to_run)

    # Run selected plots
    for name in to_run:
        print(f"\nRunning: {name}")
        try:
            available[name]()
        except Exception as e:
            print(f"Error while running {name}: {e}")

    print("\nRequested plots completed.")