import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import time

def simulate_matching(n_students, n_schools, capacity_per_school, k, n_simulations=100):
    """
    Simulate school matching with uniform random preferences.
    
    Parameters:
    -----------
    n_students : int
        Number of students
    n_schools : int
        Number of schools
    capacity_per_school : int or array
        Capacity per school (can be constant or array)
    k : int
        Number of schools each student ranks
    n_simulations : int
        Number of Monte Carlo simulations
    
    Returns:
    --------
    float : Fraction of students unmatched (averaged over simulations)
    """
    unmatched_counts = []
    
    for sim in range(n_simulations):
        # Generate all applications at once
        applications = np.zeros((n_students, n_schools), dtype=bool)
        for i in range(n_students):
            schools = np.random.choice(n_schools, size=k, replace=False)
            applications[i, schools] = True
        
        matched = np.zeros(n_students, dtype=bool)
        
        # Process each school
        for school_id in range(n_schools):
            applicants = np.where(applications[:, school_id])[0]
            
            if len(applicants) == 0:
                continue
            
            capacity = capacity_per_school if np.isscalar(capacity_per_school) else capacity_per_school[school_id]
            
            if len(applicants) <= capacity:
                selected = applicants
            else:
                selected = np.random.choice(applicants, size=capacity, replace=False)
            
            matched[selected] = True
        
        unmatched_counts.append(np.sum(~matched))
    
    return np.mean(unmatched_counts) / n_students


def theoretical_prediction(beta, k):
    """
    Compute theoretical prediction for P(unmatched).
    
    Parameters:
    -----------
    beta : float
        Capacity ratio (total_capacity / n_students)
    k : int
        Number of schools ranked
    
    Returns:
    --------
    float : Predicted P(unmatched)
    """
    if beta >= k:
        return 0.0  # Approximation: essentially zero
    else:
        return max(0, (1 - beta / k) ** k)


def plot_k_variation(n_students=72000, n_schools=533, total_capacity=83000, 
                     k_values=None, n_simulations=50):
    """
    Generate plot showing P(unmatched) vs k.
    
    Parameters:
    -----------
    n_students : int
        Number of students
    n_schools : int
        Number of schools
    total_capacity : int
        Total capacity across all schools (if None, set to 1.15 * n_students)
    k_values : array-like
        Values of k to test (if None, use [1, 2, 3, 5, 7, 10, 15, 20])
    n_simulations : int
        Number of Monte Carlo simulations per k value
    """
    if total_capacity is None:
        total_capacity = int(1.15 * n_students)
    
    if k_values is None:
        k_values = [1, 2, 3, 5, 7, 10, 15, 20]
    
    beta = total_capacity / n_students
    capacity_per_school = total_capacity // n_schools
    
    print(f"Parameters:")
    print(f"  n_students: {n_students:,}")
    print(f"  n_schools: {n_schools}")
    print(f"  total_capacity: {total_capacity:,}")
    print(f"  β = {beta:.3f}")
    print(f"  capacity_per_school: {capacity_per_school}")
    print(f"\nRunning simulations...")
    
    # Run simulations
    simulation_results = []
    theoretical_results = []
    
    for k in tqdm(k_values, desc="Simulating k values"):
        # Simulation
        p_unmatched_sim = simulate_matching(
            n_students, n_schools, capacity_per_school, k, n_simulations
        )
        simulation_results.append(p_unmatched_sim)
        
        # Theory
        p_unmatched_theory = theoretical_prediction(beta, k)
        theoretical_results.append(p_unmatched_theory)
        
        print(f"  k={k:2d}: Simulation={p_unmatched_sim:.4f}, Theory={p_unmatched_theory:.4f}")
    
    # Plot results
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Plot 1: P(unmatched) vs k
    ax1.plot(k_values, simulation_results, 'bo-', linewidth=2, markersize=8, 
             label='Simulation', alpha=0.7)
    ax1.plot(k_values, theoretical_results, 'r--', linewidth=2, 
             label='Theory: $(1-β/k)^k$')
    ax1.axhline(0, color='k', linestyle='-', linewidth=0.5, alpha=0.3)
    ax1.set_xlabel('k (number of schools ranked)', fontsize=12)
    ax1.set_ylabel('P(unmatched)', fontsize=12)
    ax1.set_title(f'P(unmatched) vs k\n(n={n_students:,}, m={n_schools}, β={beta:.3f})', 
                  fontsize=13)
    ax1.legend(fontsize=11)
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim(bottom=-0.02)
    
    # Plot 2: Number of unmatched students
    n_unmatched_sim = [p * n_students for p in simulation_results]
    n_unmatched_theory = [p * n_students for p in theoretical_results]
    
    ax2.plot(k_values, n_unmatched_sim, 'bo-', linewidth=2, markersize=8,
             label='Simulation', alpha=0.7)
    ax2.plot(k_values, n_unmatched_theory, 'r--', linewidth=2,
             label='Theory')
    ax2.set_xlabel('k (number of schools ranked)', fontsize=12)
    ax2.set_ylabel('Number of unmatched students', fontsize=12)
    ax2.set_title(f'Unmatched Students vs k\n(n={n_students:,}, m={n_schools}, β={beta:.3f})', 
                  fontsize=13)
    ax2.legend(fontsize=11)
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('matching_vs_k.png', dpi=300, bbox_inches='tight')
    print(f"\nPlot saved as 'matching_vs_k.png'")
    plt.show()
    
    return k_values, simulation_results, theoretical_results


def plot_beta_variation(n_students=10000, n_schools=500, k=12, 
                        beta_values=None, n_simulations=50):
    """
    Generate plot showing P(unmatched) vs β for fixed k.
    
    Parameters:
    -----------
    n_students : int
        Number of students
    n_schools : int
        Number of schools
    k : int
        Number of schools each student ranks
    beta_values : array-like
        Values of β to test
    n_simulations : int
        Number of Monte Carlo simulations per β value
    """
    if beta_values is None:
        beta_values = np.linspace(0.5, 2.0, 16)
    
    print(f"Parameters:")
    print(f"  n_students: {n_students:,}")
    print(f"  n_schools: {n_schools}")
    print(f"  k: {k}")
    print(f"  n_simulations: {n_simulations} (reduced for speed)")
    print(f"\nRunning optimized simulations...")
    
    simulation_results = []
    theoretical_results = []
    
    start_time = time.time()
    
    for beta in tqdm(beta_values, desc="Simulating β values"):
        total_capacity = int(beta * n_students)
        capacity_per_school = total_capacity // n_schools
        
        # Fast simulation
        p_unmatched_sim = simulate_matching(
            n_students, n_schools, capacity_per_school, k, n_simulations
        )
        simulation_results.append(p_unmatched_sim)
        
        # Theory
        p_unmatched_theory = theoretical_prediction(beta, k)
        theoretical_results.append(p_unmatched_theory)
    
    elapsed = time.time() - start_time
    print(f"\n✓ Completed in {elapsed:.1f}s ({elapsed/60:.1f} minutes)")
    
    # Plot
    plt.figure(figsize=(10, 6))
    plt.plot(beta_values, simulation_results, 'bo-', linewidth=2, markersize=8,
             label='Simulation', alpha=0.7)
    plt.plot(beta_values, theoretical_results, 'r--', linewidth=2,
             label='Theory: $\\max(0, (1-β/k)^k)$')
    plt.axvline(k, color='g', linestyle=':', linewidth=2, alpha=0.5,
                label=f'β = k = {k}')
    plt.axhline(0, color='k', linestyle='-', linewidth=0.5, alpha=0.3)
    plt.xlabel('β (capacity ratio)', fontsize=12)
    plt.ylabel('P(unmatched)', fontsize=12)
    plt.title(f'P(unmatched) vs β\n(n={n_students:,}, m={n_schools}, k={k})', 
              fontsize=13)
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('matching_vs_beta.png', dpi=300, bbox_inches='tight')
    print(f"✓ Plot saved as 'matching_vs_beta.png'")
    plt.show()
    
    return beta_values, simulation_results, theoretical_results



print("="*60)
print("SCHOOL MATCHING SIMULATION")
print("="*60)
'''
print("\n### Experiment 1: Varying k ###\n")
k_vals, sim_k, theory_k = plot_k_variation(
    n_students=72000,
    n_schools=533,
    total_capacity=83000, 
    k_values=[1, 2, 3, 5, 7, 10, 12, 15, 20],
    n_simulations=100
)
'''

print("\n### Experiment 2: Varying β ###\n")
beta_vals, sim_beta, theory_beta = plot_beta_variation(
    n_students=72000,
    n_schools=533,
    k=12,
    beta_values=np.linspace(0.5, 12, 25),
    n_simulations=10
)

print("\n" + "="*60)
print("SIMULATION COMPLETE")
print("="*60)