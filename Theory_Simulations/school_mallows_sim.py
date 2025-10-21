import time
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import binom
from tqdm import tqdm
from multiprocessing import Pool

# ==========================================
# Step 1: Compute pi_r(phi, k) via sampling
# ==========================================

def sample_mallows_top_k_rsm(m, phi, k):
    """
    Sample top-k from Mallows using RSM directly.
    Only generates k items instead of full ranking.
    """
    remaining = np.arange(1, m+1)
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
    
    for _ in range(n_samples):
        top_k = sample_mallows_top_k_rsm(m, phi, k)  # ← Use RSM directly!
        counts[top_k - 1] += 1
    
    return counts

def compute_pi_r(phi, k, m, n_samples=2000, n_workers=4):
    """
    Vectorized + Parallel computation using RSM
    """
    if n_workers == 1:
        counts = np.zeros(m)
        for _ in range(n_samples):
            top_k = sample_mallows_top_k_rsm(m, phi, k)  # ← Use RSM directly!
            counts[top_k - 1] += 1
        return counts / n_samples
    else:
        samples_per_worker = n_samples // n_workers
        args = [(phi, k, m, samples_per_worker) for _ in range(n_workers)]
        
        with Pool(n_workers) as pool:
            results = pool.map(compute_pi_batch, args)
        
        total_counts = np.sum(results, axis=0)
        return total_counts / n_samples

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
    weighted_avg_rejection = np.sum(pi_values * rejection_probs) / np.sum(pi_values)
    return weighted_avg_rejection ** k


def prob_unmatched_vectorized(ell_array, pi_values, c, k):
    """
    Vectorized computation for multiple lottery numbers at once
    """
    results = []
    for ell in ell_array:
        results.append(prob_unmatched_weighted(ell, pi_values, c, k))
    return np.array(results)

# ==========================================
# Step 3: Plotting functions (OPTIMIZED)
# ==========================================

def plot_effect_of_phi(n=72000, m=533, c=156, k=12):
    """Plot P(unmatched | ell) for different phi values - OPTIMIZED"""
    
    phi_values = [0.0, 0.3, 0.5, 0.7, 1.0]
    
    # KEY OPTIMIZATION: Sample only 200 lottery numbers instead of all 72,000!
    ell_range = np.linspace(1, n, 200, dtype=int)
    
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
            pi_vals = compute_pi_r(phi, k, m, n_samples=1000, n_workers=8)  # Reduced samples
        
        # Compute P(unmatched) - no inner progress bar needed now
        probs = prob_unmatched_vectorized(ell_range, pi_vals, c, k)
        
        ax1.plot(ell_range, probs, label=f'φ = {phi}', linewidth=2)
        
        elapsed = time.time() - start
        tqdm.write(f"  ✓ φ = {phi} done in {elapsed:.1f}s")
    
    ax1.set_xlabel('Lottery Number ℓ', fontsize=12)
    ax1.set_ylabel('P(unmatched | ℓ)', fontsize=12)
    ax1.set_title(f'Effect of φ (n={n}, m={m}, c={c}, k={k})', fontsize=13)
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Average unmatched probability vs phi
    phi_range = np.linspace(0, 1, 15)  # Reduced from 20 to 15
    avg_unmatched = []
    
    for phi in tqdm(phi_range, desc="Computing avg vs φ"):
        if phi < 0.05:
            pi_vals = np.array([1.0 if r <= k else 0.0 for r in range(1, m+1)])
        elif phi > 0.95:
            pi_vals = np.full(m, k/m)
        else:
            pi_vals = compute_pi_r(phi, k, m, n_samples=500, n_workers=8)  # Reduced samples
        
        probs = prob_unmatched_vectorized(ell_range, pi_vals, c, k)
        avg_unmatched.append(np.mean(probs))
    
    ax2.plot(phi_range, avg_unmatched, 'o-', linewidth=2, markersize=6)
    ax2.set_xlabel('φ (preference correlation)', fontsize=12)
    ax2.set_ylabel('Average P(unmatched)', fontsize=12)
    ax2.set_title('Average Unmatched Probability vs φ', fontsize=13)
    ax2.grid(True, alpha=0.3)
    
    total_elapsed = time.time() - total_start
    print(f"\n✓ Total time: {total_elapsed:.1f}s ({total_elapsed/60:.1f} min)")
    
    plt.tight_layout()
    plt.savefig('effect_of_phi_rsm.png', dpi=300, bbox_inches='tight')
    print("✓ Saved: effect_of_phi_rsm.png")
    plt.show()

def plot_effect_of_k(n=72000, m=533, c=156, phi=0.5):
    """Plot P(unmatched | ell) for different k values - OPTIMIZED"""
    
    k_values = [3, 5, 10, 15, 20]
    
    # KEY OPTIMIZATION: Sample only 200 lottery numbers
    ell_range = np.linspace(1, n, 200, dtype=int)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    for k in tqdm(k_values, desc="Computing k curves"):
        if phi == 0.0:
            pi_vals = np.array([1.0 if r <= k else 0.0 for r in range(1, m+1)])
        elif phi == 1.0:
            pi_vals = np.full(m, k/m)
        else:
            pi_vals = compute_pi_r(phi, k, m, n_samples=1000, n_workers=8)
        
        probs = prob_unmatched_vectorized(ell_range, pi_vals, c, k)
        ax1.plot(ell_range, probs, label=f'k = {k}', linewidth=2)
        tqdm.write(f"  ✓ k = {k} completed")
    
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
            pi_vals = compute_pi_r(phi, k, m, n_samples=1000, n_workers=8)

        probs = prob_unmatched_vectorized(ell_range, pi_vals, c, k)
        avg_unmatched.append(np.mean(probs))
    
    ax2.plot(list(k_range), avg_unmatched, 'o-', linewidth=2, markersize=6)
    ax2.set_xlabel('k (list length)', fontsize=12)
    ax2.set_ylabel('Average P(unmatched)', fontsize=12)
    ax2.set_title('Average Unmatched Probability vs k', fontsize=13)
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('effect_of_k_rsm.png', dpi=300, bbox_inches='tight')
    print("✓ Saved: effect_of_k_rsm.png")
    plt.show()

def plot_pi_r_distribution(phi_values=[0.0, 0.3, 0.5, 0.7, 1.0], k=12, m=533):
    """Plot how pi_r varies with rank for different phi - OPTIMIZED"""
    
    fig, ax = plt.subplots(figsize=(10, 6))
    ranks = range(1, min(m+1, 50))  # Plot first 50 ranks
    
    for phi in tqdm(phi_values, desc="Computing πᵣ distributions"):
        if phi == 0.0:
            pi_vals = [1.0 if r <= k else 0.0 for r in ranks]
        elif phi == 1.0:
            pi_vals = [k/m for r in ranks]
        else:
            full_pi = compute_pi_r(phi, k, m, n_samples=2000, n_workers=8)  # Reduced
            pi_vals = [full_pi[r-1] for r in ranks]
        
        ax.plot(ranks, pi_vals, 'o-', label=f'φ = {phi}', linewidth=2, markersize=5)
        tqdm.write(f"  ✓ φ = {phi} completed")
    
    ax.set_xlabel('School Rank r in σ*', fontsize=12)
    ax.set_ylabel('πᵣ(φ, k) = P(rank r in top-k)', fontsize=12)
    ax.set_title(f'Application Probability vs School Rank (k={k}, m={m})', fontsize=13)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('pi_r_distribution_rsm.png', dpi=300, bbox_inches='tight')
    print("✓ Saved: pi_r_distribution_rsm.png")
    plt.show()

# ==========================================
# Run the plots
# ==========================================

if __name__ == "__main__":
    
    # Plot 1: Effect of phi
    print("\n📊 Plot 1: Effect of φ")
    print("-" * 60)
    plot_effect_of_phi(n=72000, m=533, c=156, k=12)
    
    # Plot 2: Effect of k
    print("\n📊 Plot 2: Effect of k")
    print("-" * 60)
    plot_effect_of_k(n=72000, m=533, c=156, phi=0.5)
    
    # Plot 3: pi_r distribution
    print("\n📊 Plot 3: πᵣ distribution")
    print("-" * 60)
    plot_pi_r_distribution(k=12, m=533)
    
    print("\n" + "=" * 60)
    print("✅ ALL PLOTS COMPLETED!")
    print("=" * 60)
    print("Generated files:")
    print("  • effect_of_phi.png")
    print("  • effect_of_k.png")
    print("  • pi_r_distribution.png")