import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import sys
import os


try:
    from school_mallows_sim import (
        sample_mallows_top_k_rsm,
        compute_pi,
        normalize_pi,
        prob_unmatched_vectorized,
        prob_unmatched_vectorized_variable,
        generate_list_length_distribution,
        sample_list_length
    )
    from gale_shapley import run_matching  # Your DA implementation
except ImportError as e:
    print(f"Error importing modules: {e}")
    print("Make sure school_mallows_sim.py and gale_shapley.py are in the same directory")
    sys.exit(1)


def generate_mallows_preferences_for_da(
    n_students=10000,
    m_schools=533,
    phi=0.5,
    k_max=12,
    variable_k=False,
    alpha=2.0,
    min_k=1,
    k_dist='normal',
    k_std=None,
    seed=42
):
    """
    Generate student rankings using Mallows RSM for DA matching.
    
    Returns:
    --------
    student_rankings : dict
        {student_id: [school_dbn_1, school_dbn_2, ...]}
    student_info : dict
        {student_id: [id, lottery_number, ...]}
    metadata : dict
        Actual parameters used (phi, k) for each student
    """
    np.random.seed(seed)
    
    student_rankings = {}
    student_info = {}
    metadata = {
        'student_params': [],  # List of (student_id, phi, k, lottery)
        'phi': phi,
        'k_max': k_max,
        'variable_k': variable_k,
        'alpha': alpha,
        'seed': seed
    }
    
    # Generate lottery numbers (uniformly random priorities for DA)
    lottery_numbers = np.random.permutation(n_students) + 1  # 1 to n_students
    
    for i in range(n_students):
        student_id = f"Student #{i}"
        lottery = int(lottery_numbers[i])
        
        # Determine list length
        if variable_k:
            k_i = sample_list_length(k_max, alpha, min_k, k_dist=k_dist, k_std=k_std)
        else:
            k_i = k_max
        
        # Generate ranking using Mallows RSM
        ranking = sample_mallows_top_k_rsm(m_schools, phi, k_i, center=None)
        
        # Convert to school IDs (School #0 through School #532)
        school_ids = [f"School #{int(r-1)}" for r in ranking]
        
        student_rankings[student_id] = school_ids
        
        # Store student info (format expected by matching code)
        # [id, lottery, selection_policy, ranking_policy, num_schools, test_score, seat, screen, name, district, borough, location]
        student_info[student_id] = [
            student_id,      # id
            lottery,         # lottery number
            -1,              # selection_policy (unused)
            -1,              # ranking_policy (unused)
            k_i,             # num_schools
            -1,              # test_score (unused)
            -1,              # seat (unused)
            -1,              # screen (unused)
            "",              # name
            0,               # district
            None,            # borough
            None             # location
        ]
        
        metadata['student_params'].append({
            'student_id': student_id,
            'phi': phi,
            'k': k_i,
            'lottery': lottery
        })
    
    return student_rankings, student_info, metadata


def generate_school_rankings_for_da(
    student_info,
    m_schools=533,
    capacity=156,
    seed=42
):
    """
    Generate school priority rankings for DA matching.
    Uses lottery-based priorities (all schools rank students by lottery number).
    
    Returns:
    --------
    school_rankings : dict
        {school_dbn: [student_id_1, student_id_2, ...]}
    school_info : dict
        {school_dbn: [dbn, capacity, policy, ...]}
    """
    np.random.seed(seed)
    
    school_rankings = {}
    school_info = {}
    
    # Sort all students by lottery number
    students_by_lottery = sorted(
        student_info.items(),
        key=lambda x: x[1][1]  # x[1][1] is the lottery number
    )
    all_students_ranked = [sid for sid, _ in students_by_lottery]
    
    for s in range(m_schools):
        school_dbn = f"School #{s}"
        
        # All schools use same lottery-based ranking (open seats)
        school_rankings[school_dbn] = all_students_ranked.copy()
        
        # Store school info
        # [dbn, capacity, policy, popularity, likeability, name, district, borough, location]
        school_info[school_dbn] = [
            school_dbn,
            capacity,
            1,        # policy (1 = open/lottery)
            -1,       # popularity (unused)
            -1,       # likeability (unused)
            "",       # name
            0,        # district
            None,     # borough
            None      # location
        ]
    
    return school_rankings, school_info


def run_single_da_simulation(
    n_students=10000,
    m_schools=533,
    capacity=156,
    phi=0.5,
    k_max=12,
    variable_k=False,
    alpha=2.0,
    k_dist='normal',
    k_std=None,
    seed=42
):
    """
    Run a single DA simulation with Mallows-generated preferences.
    
    Returns:
    --------
    matches : dict
        {student_id: {'dbn': school_dbn, 'rank': rank_in_list}}
    metadata : dict
        Parameters used for generation
    """
    # Generate preferences
    student_rankings, student_info, metadata = generate_mallows_preferences_for_da(
        n_students=n_students,
        m_schools=m_schools,
        phi=phi,
        k_max=k_max,
        variable_k=variable_k,
        alpha=alpha,
        k_dist=k_dist,
        k_std=k_std,
        seed=seed
    )
    
    # Generate school priorities
    school_rankings, school_info = generate_school_rankings_for_da(
        student_info,
        m_schools=m_schools,
        capacity=capacity,
        seed=seed
    )
    
    # Run DA matching
    bins, matches, school_data = run_matching(
        student_rankings,
        student_info,
        school_rankings,
        school_info
    )
    
    return matches, metadata, student_info


def analyze_da_results(matches, student_info, n_students):
    """
    Analyze DA matching results.
    
    Returns:
    --------
    results : dict
        Statistics about matching outcomes
    """
    unmatched_students = []
    matched_students = []
    match_ranks = []
    
    for student_id, match_info in matches.items():
        lottery = student_info[student_id][1]
        
        if match_info['dbn'] is None:  # Unmatched
            unmatched_students.append(lottery)
        else:  # Matched
            matched_students.append(lottery)
            if match_info['rank'] is not None:
                match_ranks.append(match_info['rank'])
    
    return {
        'n_unmatched': len(unmatched_students),
        'unmatched_rate': len(unmatched_students) / n_students,
        'unmatched_lotteries': sorted(unmatched_students),
        'matched_lotteries': sorted(matched_students),
        'mean_match_rank': np.mean(match_ranks) if match_ranks else None,
        'median_match_rank': np.median(match_ranks) if match_ranks else None
    }


def compute_school_utilization(matches, school_capacities):
    """Compute utilization per school given DA matches.

    matches: dict student_id -> match_info (expects match_info['dbn'] or None)
    school_capacities: dict school_dbn -> capacity (int)

    Returns dict school_dbn -> utilization (assigned / capacity)
    """
    # Initialize counts
    counts = {dbn: 0 for dbn in school_capacities.keys()}

    # matches maps student_id -> {'dbn': ..., ...}
    for sid, info in matches.items():
        dbn = info.get('dbn') if isinstance(info, dict) else info
        if dbn is None:
            continue
        if dbn not in counts:
            counts[dbn] = counts.get(dbn, 0) + 1
        else:
            counts[dbn] += 1

    util = {}
    for dbn, cap in school_capacities.items():
        cap_val = float(cap) if cap is not None and cap > 0 else 1.0
        util[dbn] = counts.get(dbn, 0) / cap_val

    return util


def plot_school_utilization_cdf_across_runs(matches_list, school_capacities,
                                             bins=100, fname=None, show=False):
    """Plot mean cumulative utilization curve across multiple simulation runs.

    matches_list: list of matches dicts (one per simulation)
    school_capacities: dict dbn -> capacity
    Returns: xs, mean_ys, std_ys
    """
    # Compute per-run utilization arrays
    utils_per_run = []
    for matches in matches_list:
        util = compute_school_utilization(matches, school_capacities)
        vals = np.array(list(util.values()), dtype=float)
        utils_per_run.append(vals)

    if not utils_per_run:
        raise ValueError("No match runs provided to plot_school_utilization_cdf_across_runs")

    utils_per_run = np.vstack(utils_per_run)  # shape (n_runs, n_schools)

    # x grid
    xmax = max(1.0, utils_per_run.max())
    xs = np.linspace(0.0, xmax, bins)

    # For each run compute the CDF-like curve: fraction of schools with util >= x
    curves = []
    for run_vals in utils_per_run:
        curves.append([np.mean(run_vals >= x) for x in xs])
    curves = np.array(curves)

    mean_curve = curves.mean(axis=0)
    std_curve = curves.std(axis=0)

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(xs, mean_curve, lw=2, label='Mean across runs')
    ax.fill_between(xs, mean_curve - std_curve, mean_curve + std_curve, alpha=0.2,
                    label='±1 std')
    ax.set_xlim(0, xmax)
    ax.set_ylim(0, 1.0)
    ax.set_xlabel('Utilization (assigned / capacity)')
    ax.set_ylabel('Fraction of schools with utilization ≥ x')
    ax.set_title(f'School utilization CDF across runs (n_runs={len(matches_list)})')
    ax.grid(alpha=0.3)
    ax.legend()

    if fname:
        d = os.path.dirname(fname)
        if d and not os.path.exists(d):
            os.makedirs(d, exist_ok=True)
        plt.tight_layout()
        plt.savefig(fname, dpi=300, bbox_inches='tight')
        print(f"✓ Saved utilization plot: {fname}")

    if show:
        plt.show()
    else:
        plt.close()

    return xs, mean_curve, std_curve


def compare_theoretical_vs_simulation(
    n_students=10000,
    m_schools=533,
    capacity=156,
    phi=0.5,
    k_max=12,
    variable_k=False,
    alpha=2.0,
    k_dist='normal',
    k_std=None,
    seed=42,
    n_simulations=20,
    n_samples_pi=2000,
    n_workers=4
):
    """
    Main comparison: theoretical model vs DA simulations.
    
    Returns:
    --------
    comparison_results : dict
        Complete results including plots
    """
    print(f"\n{'='*80}")
    print(f"VALIDATION: Theoretical vs DA Simulation")
    print(f"{'='*80}")
    print(f"Parameters:")
    print(f"  n_students={n_students}, m_schools={m_schools}, capacity={capacity}")
    print(f"  phi={phi}, k_max={k_max}, variable_k={variable_k}")
    if variable_k:
        print(f"  alpha={alpha}, k_dist={k_dist}, k_std={k_std}")
    print(f"  seed={seed}, n_simulations={n_simulations}")
    print(f"{'='*80}\n")
    
    # ============================================
    # PART 1: Theoretical Predictions
    # ============================================
    print("[1/3] Computing theoretical predictions...")
    
    # Compute pi_r distribution
    if variable_k:
        pi_vals = compute_pi(
            phi, k_max, m_schools,
            n_samples=n_samples_pi,
            n_workers=n_workers,
            variable=True,
            alpha=alpha,
            min_k=1,
            k_dist=k_dist,
            k_std=k_std
        )
    else:
        pi_vals = compute_pi(
            phi, k_max, m_schools,
            n_samples=n_samples_pi,
            n_workers=n_workers,
            variable=False
        )
    
    pi_vals = normalize_pi(pi_vals)
    
    # Sample lottery numbers for evaluation
    lottery_samples = np.linspace(1, n_students, 100, dtype=int)
    
    # Compute theoretical P(unmatched | ℓ)
    if variable_k:
        theoretical_probs = prob_unmatched_vectorized_variable(
            lottery_samples, pi_vals, capacity, k_max, alpha, min_k=1,
            k_dist=k_dist, k_std=k_std
        )
    else:
        theoretical_probs = prob_unmatched_vectorized(
            lottery_samples, pi_vals, capacity, k_max
        )
    
    avg_theoretical = np.mean(theoretical_probs)
    print(f"  Theoretical avg P(unmatched): {avg_theoretical:.4f}")
    
    # ============================================
    # PART 2: Run DA Simulations
    # ============================================
    print(f"\n[2/3] Running {n_simulations} DA simulations...")
    
    simulation_results = []
    
    for sim in tqdm(range(n_simulations), desc="DA Simulations"):
        sim_seed = seed + sim * 1000  # Ensure different seeds
        
        # Run DA
        matches, metadata, student_info = run_single_da_simulation(
            n_students=n_students,
            m_schools=m_schools,
            capacity=capacity,
            phi=phi,
            k_max=k_max,
            variable_k=variable_k,
            alpha=alpha,
            k_dist=k_dist,
            k_std=k_std,
            seed=sim_seed
        )
        
        # Analyze results
        results = analyze_da_results(matches, student_info, n_students)
    # Keep matches with the analysis so we can compute utilizations later
    results['matches'] = matches
    simulation_results.append(results)
    
    # ============================================
    # PART 3: Compare Results
    # ============================================
    print(f"\n[3/3] Comparing theoretical vs simulation...")
    
    # Aggregate simulation results
    sim_unmatched_rates = [r['unmatched_rate'] for r in simulation_results]
    avg_sim_unmatched = np.mean(sim_unmatched_rates)
    std_sim_unmatched = np.std(sim_unmatched_rates)
    
    # Compute empirical P(unmatched | ℓ) from simulations
    empirical_unmatched_by_lottery = {ell: [] for ell in lottery_samples}
    
    for result in simulation_results:
        for lottery in result['unmatched_lotteries']:
            # Find closest sampled lottery number
            closest = lottery_samples[np.argmin(np.abs(lottery_samples - lottery))]
            empirical_unmatched_by_lottery[closest].append(1)
        
        for lottery in result['matched_lotteries']:
            closest = lottery_samples[np.argmin(np.abs(lottery_samples - lottery))]
            empirical_unmatched_by_lottery[closest].append(0)
    
    # Compute empirical probabilities
    empirical_probs = []
    for lottery in lottery_samples:
        outcomes = empirical_unmatched_by_lottery[lottery]
        if len(outcomes) > 0:
            empirical_prob = np.mean(outcomes)
        else:
            empirical_prob = 0.0
        empirical_probs.append(empirical_prob)
    
    empirical_probs = np.array(empirical_probs)
    
    # ============================================
    # PART 4: Print Summary
    # ============================================
    print("\n" + "="*80)
    print("RESULTS SUMMARY")
    print("="*80)
    print(f"Theoretical avg P(unmatched): {avg_theoretical:.4f}")
    print(f"Simulation avg P(unmatched):  {avg_sim_unmatched:.4f} ± {std_sim_unmatched:.4f}")
    print(f"Absolute error:               {abs(avg_theoretical - avg_sim_unmatched):.4f}")
    if avg_sim_unmatched > 0:
        rel_error = 100 * abs(avg_theoretical - avg_sim_unmatched) / avg_sim_unmatched
        print(f"Relative error:               {rel_error:.2f}%")
    
    # Error statistics across lottery numbers
    valid_mask = empirical_probs > 0  # Only compare where we have data
    if np.sum(valid_mask) > 0:
        errors = theoretical_probs[valid_mask] - empirical_probs[valid_mask]
        print(f"\nLottery-specific comparison:")
        print(f"  Mean error:                 {np.mean(errors):.4f}")
        print(f"  Std error:                  {np.std(errors):.4f}")
        print(f"  Max absolute error:         {np.max(np.abs(errors)):.4f}")
    
    # ============================================
    # PART 5: Generate Plots
    # ============================================
    print(f"\n[4/4] Generating comparison plots...")
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Plot 1: Theoretical vs Empirical curves
    ax = axes[0, 0]
    ax.plot(lottery_samples, theoretical_probs, 'b-', 
            label='Theoretical', linewidth=2)
    ax.scatter(lottery_samples, empirical_probs, c='red', 
               label='Simulation', s=30, alpha=0.6)
    ax.set_xlabel('Lottery Number ℓ')
    ax.set_ylabel('P(unmatched | ℓ)')
    ax.set_title(f'Theoretical vs Simulation (φ={phi}, k={k_max}, var_k={variable_k})')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot 2: Error distribution
    ax = axes[0, 1]
    if np.sum(valid_mask) > 0:
        ax.hist(errors, bins=20, edgecolor='black', alpha=0.7)
        ax.axvline(0, color='red', linestyle='--', linewidth=2, label='Zero error')
        ax.axvline(np.mean(errors), color='blue', linestyle='--', linewidth=2, 
                   label=f'Mean: {np.mean(errors):.4f}')
    ax.set_xlabel('Error (Theoretical - Simulation)')
    ax.set_ylabel('Frequency')
    ax.set_title('Error Distribution Across Lottery Numbers')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot 3: Distribution of unmatched rates across simulations
    ax = axes[1, 0]
    ax.hist(sim_unmatched_rates, bins=15, edgecolor='black', alpha=0.7)
    ax.axvline(avg_theoretical, color='blue', linestyle='--', linewidth=2,
               label=f'Theoretical: {avg_theoretical:.4f}')
    ax.axvline(avg_sim_unmatched, color='red', linestyle='--', linewidth=2,
               label=f'Simulation: {avg_sim_unmatched:.4f}')
    ax.set_xlabel('Unmatched Rate')
    ax.set_ylabel('Frequency (across simulations)')
    ax.set_title(f'Distribution of Unmatched Rates ({n_simulations} simulations)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot 4: Mean match rank distribution (if available)
    ax = axes[1, 1]
    mean_ranks = [r['mean_match_rank'] for r in simulation_results 
                  if r['mean_match_rank'] is not None]
    if mean_ranks:
        ax.hist(mean_ranks, bins=15, edgecolor='black', alpha=0.7)
        ax.set_xlabel('Mean Match Rank')
        ax.set_ylabel('Frequency')
        ax.set_title('Distribution of Mean Match Rank Across Simulations')
        ax.grid(True, alpha=0.3)
    else:
        ax.text(0.5, 0.5, 'No match rank data', 
                ha='center', va='center', transform=ax.transAxes)
    
    plt.tight_layout()
    
    var_str = f"_var_k_{k_dist}" if variable_k else ""
    fname = f'output_plots/theory_gs_comparison_validation_phi{phi}_k{k_max}{var_str}_seed{seed}.png'
    plt.savefig(fname, dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {fname}")
    plt.show()
    
    # ============================================
    # Extra: school utilization across runs
    # ============================================
    try:
        matches_list = [r['matches'] for r in simulation_results]
        # Build capacities map consistent with generate_school_rankings_for_da
        capacities = {f"School #{s}": capacity for s in range(m_schools)}
        util_fname = f'output_plots/school_utilization_phi{phi}_k{k_max}_seed{seed}.png'
        plot_school_utilization_cdf_across_runs(matches_list, capacities, bins=200, fname=util_fname, show=False)
    except Exception as e:
        print(f"Could not generate utilization plot: {e}")
    return {
        'theoretical_probs': theoretical_probs,
        'empirical_probs': empirical_probs,
        'lottery_samples': lottery_samples,
        'simulation_results': simulation_results,
        'avg_theoretical': avg_theoretical,
        'avg_simulation': avg_sim_unmatched,
        'std_simulation': std_sim_unmatched,
        'errors': errors if np.sum(valid_mask) > 0 else None
    }


def validation_suite():
    """
    Run comprehensive validation across multiple parameter settings.
    """
    print("\n" + "="*80)
    print("COMPREHENSIVE VALIDATION SUITE")
    print("="*80 + "\n")
    
    test_cases = [
        # (phi, k, variable_k, alpha, k_dist, description)
        (0.3, 12, False, 2.0, 'power_tail', 'Concentrated preferences, fixed k'),
        (0.5, 12, False, 2.0, 'power_tail', 'Moderate preferences, fixed k'),
        (0.7, 12, False, 2.0, 'power_tail', 'Dispersed preferences, fixed k'),
        (0.5, 6, False, 2.0, 'power_tail', 'Moderate preferences, short lists'),
        (0.5, 20, False, 2.0, 'power_tail', 'Moderate preferences, long lists'),
        (0.5, 12, True, 2.0, 'power_tail', 'Moderate preferences, variable k'),
        (0.3, 12, True, 1.0, 'power_tail', 'Concentrated prefs, variable k, flat alpha'),
    ]
    
    summary = []
    
    for i, (phi, k, var_k, alpha, k_dist, desc) in enumerate(test_cases):
        print(f"\n{'='*80}")
        print(f"TEST CASE {i+1}/{len(test_cases)}: {desc}")
        print(f"{'='*80}")
        
        result = compare_theoretical_vs_simulation(
            n_students=5000,  # Smaller for speed
            m_schools=533,
            capacity=156,
            phi=phi,
            k_max=k,
            variable_k=var_k,
            alpha=alpha,
            k_dist=k_dist,
            seed=42 + i,
            n_simulations=10,  # Fewer for speed
            n_samples_pi=1000,
            n_workers=4
        )
        
        summary.append({
            'description': desc,
            'phi': phi,
            'k': k,
            'variable_k': var_k,
            'alpha': alpha,
            'theoretical': result['avg_theoretical'],
            'simulation': result['avg_simulation'],
            'abs_error': abs(result['avg_theoretical'] - result['avg_simulation']),
            'rel_error': 100 * abs(result['avg_theoretical'] - result['avg_simulation']) / result['avg_simulation']
        })
    
    # Print summary table
    print("\n" + "="*80)
    print("VALIDATION SUMMARY")
    print("="*80 + "\n")
    
    print(f"{'Test Case':<50} {'Theory':>8} {'Sim':>8} {'Error':>8} {'Rel%':>6}")
    print("-" * 80)
    for s in summary:
        print(f"{s['description']:<50} {s['theoretical']:>8.4f} {s['simulation']:>8.4f} "
              f"{s['abs_error']:>8.4f} {s['rel_error']:>6.2f}")
    
    print("\n" + "-" * 80)
    avg_abs_error = np.mean([s['abs_error'] for s in summary])
    avg_rel_error = np.mean([s['rel_error'] for s in summary])
    max_rel_error = np.max([s['rel_error'] for s in summary])
    
    print(f"Average absolute error: {avg_abs_error:.4f}")
    print(f"Average relative error: {avg_rel_error:.2f}%")
    print(f"Maximum relative error: {max_rel_error:.2f}%")
    print("="*80 + "\n")
    
    return summary


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Validate theoretical model against DA simulations')
    parser.add_argument('--mode', choices=['single', 'suite'], default='single',
                        help='Run single comparison or full validation suite')
    parser.add_argument('--n_students', type=int, default=10000,
                        help='Number of students (default: 10000)')
    parser.add_argument('--phi', type=float, default=0.5,
                        help='Mallows phi parameter (default: 0.5)')
    parser.add_argument('--k', type=int, default=12,
                        help='List length (default: 12)')
    parser.add_argument('--variable_k', action='store_true',
                        help='Use variable list lengths')
    parser.add_argument('--alpha', type=float, default=2.0,
                        help='Power law alpha (default: 2.0)')
    parser.add_argument('--k_dist', choices=['power_tail', 'centered_power', 'normal'],
                        default='power_tail', help='List length distribution')
    parser.add_argument('--n_sims', type=int, default=20,
                        help='Number of DA simulations (default: 20)')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed (default: 42)')
    
    args = parser.parse_args()
    
    if args.mode == 'suite':
        validation_suite()
    else:
        compare_theoretical_vs_simulation(
            n_students=args.n_students,
            m_schools=533,
            capacity=156,
            phi=args.phi,
            k_max=args.k,
            variable_k=args.variable_k,
            alpha=args.alpha,
            k_dist=args.k_dist,
            seed=args.seed,
            n_simulations=args.n_sims,
            n_samples_pi=2000,
            n_workers=4
        )