import pandas as pd
import numpy as np
from scipy.optimize import minimize_scalar
import copy
from concurrent.futures import ProcessPoolExecutor
from analysis import log_and_print
from data_ingestion import extract_observed_aggregates
from gale_shapley import gale_shapley, compute_aggregates, gale_shapley_per_school
from mallows import  _sample_students_chunk

def run_single_simulation(params, df, match_stats_df, school_info_df, 
                         lottery_global=None, k_ranking_length=10, outfile=None,
                         sampling_n_jobs=32, sampling_chunk_size=2000, executor=None,
                         per_school_lottery=False):
    
    
    all_rankings = []
    all_district_assignments = []
    districts = list(params['districts'].keys())
    
    # Collect all chunks across all districts
    all_chunks = []  # (district, schools_list, sigma_indices, chunk_components, seed)
    rng = np.random.default_rng(seed=np.random.randint(0, 2**32))
    
    for district in districts:
        n_students = int(match_stats_df[
            match_stats_df['Residential District'] == district
        ]['Total Applicants'].iloc[0])
        
        sigma_d = params['districts'][district]['central_ranking']
        schools_list = params['districts'][district]['schools']
        school_to_idx = {s: i for i, s in enumerate(schools_list)}
        sigma_indices = np.array([school_to_idx[s] for s in sigma_d])
        
        component_indices = rng.choice(len(params['global_phis']),
                                       size=n_students, p=params['global_weights'])
        
        for start in range(0, n_students, sampling_chunk_size):
            chunk = component_indices[start:start + sampling_chunk_size]
            all_chunks.append((district, schools_list, sigma_indices, chunk, rng.integers(2**32)))
        
        all_district_assignments.extend([district] * n_students)
    
    # ONE pool for all districts
    results_by_district = {d: [] for d in districts}
    
    if sampling_n_jobs > 1 and executor is not None:
        futures = []
        for district, schools_list, sigma_indices, chunk, seed in all_chunks:
            future = executor.submit(
                _sample_students_chunk, sigma_indices, params['global_phis'], chunk, seed
            )
            futures.append((district, future))
        
        for district, future in futures:
            results_by_district[district].extend(future.result())
    elif sampling_n_jobs > 1:
        with ProcessPoolExecutor(max_workers=sampling_n_jobs) as pool:
            futures = []
            for district, schools_list, sigma_indices, chunk, seed in all_chunks:
                future = pool.submit(
                    _sample_students_chunk, sigma_indices, params['global_phis'], chunk, seed
                )
                futures.append((district, future))
            
            for district, future in futures:
                results_by_district[district].extend(future.result())
    else:
        for district, schools_list, sigma_indices, chunk, seed in all_chunks:
            results_by_district[district].extend(
                _sample_students_chunk(sigma_indices, params['global_phis'], chunk, seed)
            )
    
    # Convert to school DBNs and truncate
    for district in districts:
        schools_list = params['districts'][district]['schools']
        rankings = results_by_district[district]
        rankings = [r[:k_ranking_length] for r in rankings]
        rankings_as_schools = [[schools_list[idx] for idx in r] for r in rankings]
        all_rankings.extend(rankings_as_schools)
    
    log_and_print(f"    Generated {len(all_rankings)} student rankings across {len(districts)} districts ({len(all_chunks)} chunks)", log_file=outfile)
    all_schools = df['School DBN'].unique()
    school_to_idx = {s: i for i, s in enumerate(all_schools)}
    rankings_as_indices = [np.array([school_to_idx[s] for s in r]) for r in all_rankings]
    capacities_dict = school_info_df.set_index('School DBN')['Capacity'].to_dict()
    capacities = np.array([capacities_dict.get(s, 0) for s in all_schools])
    
    if per_school_lottery:
        n_students = len(rankings_as_indices)
        n_schools = len(all_schools)
        school_lotteries = rng.random((n_schools, n_students))
        matches_idx = gale_shapley_per_school(rankings_as_indices, school_lotteries, capacities)
    else:
        matches_idx = gale_shapley(rankings_as_indices, lottery_global, capacities)
    matches_schools = np.array([all_schools[m] if m >= 0 else '-1' for m in matches_idx])
    
    agg = compute_aggregates(all_rankings, matches_schools,
                            np.array(all_district_assignments), all_schools)
    return agg


def EM_algorithm(df, match_stats_df, school_info_df,
                 max_iter=10, tol=0.01, K=1, M_simulations=20, seed=40, outfile=None, 
                 sampling_n_jobs=32, max_iter_opt=5):
    """
    EM algorithm with GLOBAL MIXTURE
    """
    
    np.random.seed(seed)
    
    log_and_print("="*60, log_file=outfile)
    log_and_print("EM ALGORITHM - GLOBAL MIXTURE", log_file=outfile)
    log_and_print("="*60, log_file=outfile)
    
    districts = sorted(df['Residential District'].unique())
    n_total_students = int(match_stats_df['Total Applicants'].sum())
    lottery_global = np.random.permutation(n_total_students)
    
    log_and_print(f"\nInitialization:", log_file=outfile)
    log_and_print(f"  Districts: {len(districts)}", log_file=outfile)
    log_and_print(f"  Total students: {n_total_students}", log_file=outfile)
    log_and_print(f"  Global mixture components: K={K}", log_file=outfile)
    log_and_print(f"  Max iterations of EM Algorithm: {max_iter}", log_file=outfile)
    log_and_print(f"  Max iterations of nonconvex optimizer: {max_iter_opt}", log_file=outfile)
    log_and_print(f"  Simulations per evaluation: M={M_simulations}\n", log_file=outfile)
    
    # Initialize with GLOBAL mixture
    params = initialize_parameters_global_mixture(districts, df, K)

    observed_agg = extract_observed_aggregates(df, match_stats_df)
    
    log_likelihoods = []
    best_params = None
    best_log_like = -np.inf
    best_agg = None
    
    warm_executor = ProcessPoolExecutor(max_workers=sampling_n_jobs)
    # EM loop
    for iteration in range(max_iter):
        log_and_print(f"\n{'='*60}", log_file=outfile)
        log_and_print(f"EM ITERATION {iteration + 1}/{max_iter}", log_file=outfile)
        log_and_print(f"{'='*60}", log_file=outfile)
        
        old_params = copy.deepcopy(params)
        
        # M-STEP: Optimize global parameters
        params, final_agg, total_log_like = optimize_global_mixture(
            params, observed_agg, df, match_stats_df, 
            school_info_df, M=M_simulations, seed=seed,
            iteration=iteration, outfile=outfile, sampling_n_jobs=sampling_n_jobs,
            executor=warm_executor, max_iter_em=max_iter, max_iter_opt=max_iter_opt
        )

        # Sort them to remove indexing ambiguity
        sorted_indices = np.argsort(params['global_phis'])
        params['global_phis'] = params['global_phis'][sorted_indices]
        params['global_weights'] = params['global_weights'][sorted_indices]

        # M-STEP: Nudge sigmas using the result of the simulation above
        params = nudge_district_sigmas(
            params,
            final_agg,
            school_info_df,
            all_schools=df['School DBN'].unique(),
            outfile=outfile
        )
        
        log_likelihoods.append(total_log_like)
        log_and_print(f"\nTotal log-likelihood: {total_log_like:.2f}", log_file=outfile)
        if total_log_like > best_log_like:
            best_log_like = total_log_like
            best_params = copy.deepcopy(params)
            best_agg = copy.deepcopy(final_agg)
            log_and_print(f"  New best log-likelihood! - {best_log_like:.2f}", log_file=outfile)
        
        # Check convergence
        max_phi_change = max(
            abs(params['global_phis'][k] - old_params['global_phis'][k])
            for k in range(K)
        )
        
        log_and_print(f"Max phi change: {max_phi_change:.4f}", log_file=outfile)
        
        if iteration > 0:
            delta_log_lik = log_likelihoods[-1] - log_likelihoods[-2]
            log_and_print(f"Log-likelihood change: {delta_log_lik:.4f}", log_file=outfile)
        
        if max_phi_change < tol:
            log_and_print(f"\n{'='*60}", log_file=outfile)
            log_and_print("EM CONVERGED!", log_file=outfile)
            log_and_print(f"{'='*60}", log_file=outfile)
            break
    
    warm_executor.shutdown()
    log_and_print(f"\nFinal global parameters:", log_file=outfile)
    log_and_print(f"  Global phis: {best_params['global_phis']}", log_file=outfile)
    log_and_print(f"  Global weights: {best_params['global_weights']}", log_file=outfile)
    log_and_print(f"\nEstimated central rankings (sigma) per district:", log_file=outfile)
    for district in sorted(best_params['districts'].keys()):
        sigma = best_params['districts'][district]['central_ranking']
        log_and_print(f"\n  District {district}: {sigma}", log_file=outfile)
    
    return best_params, lottery_global, log_likelihoods, best_agg

def initialize_parameters_global_mixture(districts, df, K=1):
    """
    Initialize with global phis, district-specific sigmas
    """
    
    # Global mixture parameters (shared across districts)
    global_phis = np.random.beta(3, 2, K)
    global_phis = np.clip(global_phis, 0.5, 0.99)
    
    global_weights = np.ones(K) / K  # Uniform initially
    
    params = {
        'global_phis': global_phis,
        'global_weights': global_weights,
        'districts': {}
    }
    
    # District-specific central rankings
    for district in districts:
        df_district = df[df['Residential District'] == district]
        schools_list = df_district['School DBN'].values
        
        obs_total = df_district.set_index('School DBN')['Ratio'].to_dict()
        
        central_ranking = sorted(schools_list, key=lambda s: obs_total[s], reverse=True)
        
        params['districts'][district] = {
            'schools': schools_list,
            'central_ranking': central_ranking
        }
    
    return params

def compute_log_likelihood_gaussian_all_districts(params_global, observed_agg,
                                                   df, match_stats_df, school_info_df,
                                                   M=1, seed=42, iteration=1, outfile=None, 
                                                   executor=None, sampling_n_jobs=32):
    """
    Compute log-likelihood for ALL districts at once
    
    This is more efficient than calling compute_log_likelihood_gaussian() 
    separately for each district because we only run M simulations total
    instead of M x num_districts simulations.
    
    Returns:
        total_log_lik: Sum of log-likelihoods across all districts
    """
    districts = sorted(observed_agg.keys())
    n_students_total = int(match_stats_df['Total Applicants'].sum())
    
    # Run M simulations, collecting stats for all districts
    simulated_samples = {d: [] for d in districts}

    # Initialize based on actual unique schools in df, not school_info_df rows
    all_schools = df['School DBN'].unique()
    capacities_dict = school_info_df.set_index('School DBN')['Capacity'].to_dict()
    total_filled = np.zeros(len(all_schools))
    
    # Fixed lottery across all M simulations
    rng_lottery = np.random.default_rng(seed=seed)
    lottery_fixed = rng_lottery.permutation(n_students_total)
    
    for sim in range(M):
        log_and_print(f"      Simulation {sim+1}/{M}...", log_file=outfile)
        
        # Only vary the Mallows preference sampling, not the lottery
        np.random.seed(seed + sim)
        
        # Simulate ALL districts together (do this ONCE per M iteration)
        agg = run_single_simulation(
            params_global, df, match_stats_df, school_info_df, 
            lottery_fixed, outfile=outfile, executor=executor,
            sampling_n_jobs=sampling_n_jobs
        )

        total_filled += agg['filled']
        
        # Extract stats for EACH district from this single simulation
        for d_idx, district in enumerate(districts):
            agg_vec = agg['match_stats'][d_idx, :]
            simulated_samples[district].append(agg_vec)
    
    mean_filled = total_filled / M
    # Get capacities in same order as all_schools
    capacities = np.array([capacities_dict.get(s, 0) for s in all_schools])
    sim_util = np.full_like(mean_filled, np.nan, dtype=float)
    np.divide(mean_filled, capacities, out=sim_util, where=capacities > 0)
    sim_util = sim_util * 100

    # Get observed utilization only for schools we have
    obs_util_dict = school_info_df.set_index('School DBN')['Utilization'].to_dict()
    obs_util = np.array([obs_util_dict.get(s, np.nan) for s in all_schools], dtype=float)
    util_valid_mask = np.isfinite(obs_util) & np.isfinite(sim_util)
    if np.any(util_valid_mask):
        util_penalty = -0.1 * np.mean((obs_util[util_valid_mask] - sim_util[util_valid_mask])**2)
    else:
        util_penalty = 0.0
        log_and_print("Warning: No valid utilization pairs after NaN filtering.", log_file=outfile)
    
    log_and_print('')  # New line after progress indicator
    
    log_and_print("\n" + "="*60, log_file=outfile)
    log_and_print(f"FIT DIAGNOSTICS | Seed: {seed} | Iteration: {iteration}", log_file=outfile)
    log_and_print("="*60, log_file=outfile)
    
    metric_names = ["top3", "top5", "top10", "unmatched"]
    for d_idx, district in enumerate(districts):
        obs = np.array(observed_agg[district]['match_stats'], dtype=float)
        sim = np.array(agg['match_stats'][d_idx, :], dtype=float)
        valid_mask = np.isfinite(obs) & np.isfinite(sim)

        log_and_print(f"\nDistrict {district}:", log_file=outfile)
        if not np.any(valid_mask):
            log_and_print("  No valid observed/simulated pairs after NaN filtering.", log_file=outfile)
            continue

        obs_parts = [
            f"{metric_names[i]}={obs[i]:5.1f}%" for i in range(len(metric_names)) if valid_mask[i]
        ]
        sim_parts = [
            f"{metric_names[i]}={sim[i]:5.1f}%" for i in range(len(metric_names)) if valid_mask[i]
        ]
        diff_parts = [
            f"{metric_names[i]}={obs[i]-sim[i]:+5.1f}" for i in range(len(metric_names)) if valid_mask[i]
        ]

        log_and_print(f"  Observed:  {', '.join(obs_parts)}", log_file=outfile)
        log_and_print(f"  Simulated: {', '.join(sim_parts)}", log_file=outfile)
        log_and_print(f"  Difference: {', '.join(diff_parts)}", log_file=outfile)
    
    log_and_print("Global School Utilization (Top 5 Mismatches):", log_file=outfile)
    util_diff = obs_util - sim_util
    valid_indices = np.where(np.isfinite(util_diff) & util_valid_mask)[0]
    if len(valid_indices) > 0:
        sorted_valid = valid_indices[np.argsort(np.abs(util_diff[valid_indices]))[::-1]]
        mismatch_indices = sorted_valid[:5]
        for idx in mismatch_indices:
            s_name = all_schools[idx]
            log_and_print(f"  {s_name}: Obs={obs_util[idx]:5.1f}%, Sim={sim_util[idx]:5.1f}%, Diff={util_diff[idx]:+5.1f}%", log_file=outfile)
        log_and_print(
            f"  Mean Absolute Utilization Error: {np.mean(np.abs(util_diff[valid_indices])):.2f}%",
            log_file=outfile,
        )
    else:
        log_and_print("  No valid utilization differences after NaN filtering.", log_file=outfile)
    
    log_and_print("="*60 + "\n", log_file=outfile)
    # Now compute likelihood for each district separately
    total_log_lik = 0
    
    for district in districts:
        X = np.array(simulated_samples[district])  # M × 4 array
        
        # Check for valid data
        if len(X) == 0 or np.any(np.isnan(X)) or np.any(np.isinf(X)):
            log_and_print(f"      Warning: Invalid data for district {district}", log_file=outfile)
            continue
        
        # Estimate mean and covariance
        mu = np.mean(X, axis=0)
        
        if M > 1:
            Sigma = np.cov(X, rowvar=False)
            
            # Handle different dimensionalities
            if Sigma.ndim == 0:  # Scalar
                Sigma = np.array([[Sigma]])
            elif Sigma.ndim == 1:  # 1D
                Sigma = np.diag(Sigma)
            
            # Add regularization for numerical stability
            regularization = 1e-3 * np.eye(len(Sigma))
            Sigma = Sigma + regularization
            
            # Check for singularity
            try:
                np.linalg.cholesky(Sigma)
            except np.linalg.LinAlgError:
                Sigma = Sigma + 1e-2 * np.eye(len(Sigma))
        else:
            # Not enough samples for covariance
            Sigma = 1e-2 * np.eye(4)
        
        # Get observed vector
        obs_vec = observed_agg[district]['match_stats']
        
        # Compute Mahalanobis distance
        try:
            diff = obs_vec - mu
            inv_Sigma = np.linalg.inv(Sigma)
            mahalanobis_sq = diff @ inv_Sigma @ diff
            
            # Log-likelihood (unnormalized)
            log_lik = -0.5 * mahalanobis_sq
            
            # Sanity check
            if np.isnan(log_lik) or np.isinf(log_lik):
                log_and_print(f"      Warning: Invalid log-likelihood for district {district}", log_file=outfile)
                log_lik = -1e10
                
        except Exception as e:
            log_and_print(f"      Warning: Likelihood computation failed for district {district}: {e}", log_file=outfile)
            
            # Fall back to simple MSE
            mse = np.mean((obs_vec - mu)**2)
            log_lik = -mse * 100
        
        total_log_lik += log_lik
    
    log_and_print(f"  Match stats log-likelihood: {total_log_lik:.2f}, Util penalty: {util_penalty:.2f}, Combined: {total_log_lik + util_penalty:.2f}", log_file=outfile)
    return total_log_lik + util_penalty

def optimize_global_mixture(params, observed_agg, df, match_stats_df, 
                            school_info_df, M=20, seed=42, iteration=1,
                            sampling_n_jobs=32, outfile=None, executor=None, 
                            max_iter_em=5, max_iter_opt=5):
    K = len(params['global_phis'])
    best_agg_stats = None  # To capture utilization for the nudge
    eval_count = [0]
    last_log_like = [None]

    for k in range(K):
        phi_k_initial = params['global_phis'][k]
        log_and_print(f"\n  [EM iter {iteration+1}/{max_iter_em}] Optimizing phi[{k+1}/{K}], starting at {phi_k_initial:.4f}", log_file=outfile)

        
        def objective_global_phi_k(phi):
            nonlocal best_agg_stats
            eval_count[0] += 1
            log_and_print(f"    [EM iter {iteration+1}/{max_iter_em}] phi[{k+1}/{K}] eval #{eval_count[0]}, trying phi={phi:.4f}", log_file=outfile)
            original_phi = params['global_phis'][k]
            params['global_phis'][k] = phi
            
           
            total_log_lik = compute_log_likelihood_gaussian_all_districts(
                params, observed_agg, df, match_stats_df, 
                school_info_df, M=M, seed=seed, iteration=iteration, outfile=outfile, 
                executor=executor, sampling_n_jobs=sampling_n_jobs
            )
            last_log_like[0] = total_log_lik
            
            params['global_phis'][k] = original_phi
            return -total_log_lik
        
        result = minimize_scalar(
            objective_global_phi_k,
            bounds=(0.01, 0.99),
            method='bounded',
            options={'xatol': 0.01, 'maxiter': max_iter_opt}
        )
        params['global_phis'][k] = result.x
        log_and_print(f"  [EM iter {iteration+1}/{max_iter_em}] phi[{k+1}/{K}] -> {result.x:.4f} (took {eval_count[0]} evals)", log_file=outfile)

    # Average M simulations to get robust aggregate for the nudge
    n_students_total = int(match_stats_df['Total Applicants'].sum())
    # Fixed lottery across all M simulations
    lottery_fixed = np.random.permutation(n_students_total)
    agg_accum = None
    for sim in range(M):
        # Only vary the Mallows preference sampling, not the lottery
        np.random.seed(seed + sim)
        log_and_print(f"  [EM iter {iteration+1}/{max_iter_em}] Final averaging sim {sim+1}/{M}...", log_file=outfile)
        agg_sim = run_single_simulation(params, df, match_stats_df, school_info_df, lottery_fixed, 
                                        sampling_n_jobs=sampling_n_jobs, outfile=outfile, executor=executor)
        
        if agg_accum is None:
            agg_accum = {k: v.copy() for k, v in agg_sim.items()}
        else:
            for key in agg_accum:
                agg_accum[key] = agg_accum[key] + agg_sim[key]
    
    # Average the accumulated results
    final_agg = {k: v / M for k, v in agg_accum.items()}
    
    return params, final_agg, last_log_like[0]

def nudge_district_sigmas(params, final_agg, school_info_df, eta=0.1, all_schools=None, outfile=None):
    if all_schools is None:
        all_schools = school_info_df['School DBN'].values

    sim_filled = pd.Series(final_agg['filled'], index=all_schools)
    real_util_counts = (school_info_df.set_index('School DBN')['Utilization'] / 100) * school_info_df.set_index('School DBN')['Capacity']
    
    util_error = real_util_counts - sim_filled
    
    for d_id, d_data in params['districts'].items():
        if 'pop_scores' not in d_data:
            d_data['pop_scores'] = {s: (len(d_data['schools']) - i) 
                                   for i, s in enumerate(d_data['central_ranking'])}
        
        for s_dbn, error in util_error.items():
            if s_dbn in d_data['pop_scores'] and np.isfinite(error):
                d_data['pop_scores'][s_dbn] += eta * error 
        
        old_top3 = d_data['central_ranking'][:3] if 'central_ranking' in d_data else []
        new_sigma = sorted(d_data['pop_scores'].items(), key=lambda x: x[1], reverse=True)
        d_data['central_ranking'] = [s[0] for s in new_sigma]
        new_top3 = d_data['central_ranking'][:3]
        if old_top3 != new_top3:
            log_and_print(f"    District {d_id} sigma changed: {old_top3} -> {new_top3}", log_file=outfile)
        
    return params