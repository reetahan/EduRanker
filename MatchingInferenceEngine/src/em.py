import pandas as pd
import numpy as np
from scipy.optimize import minimize_scalar
import copy
from analysis import log_and_print
from data_ingestion import extract_observed_aggregates
from gale_shapley import gale_shapley, compute_aggregates
from mallows import mallows_insertion_sampling

def run_single_simulation(params, df, match_stats_df, school_info_df, 
                         lottery_global, k_ranking_length=10, M_val=1, outfile=None):
    """
    Run one simulation with GLOBAL MIXTURE parameters
    
    Args:
        params: Global mixture structure with 'global_phis', 'global_weights', 'districts'
    """

    all_rankings = []
    all_district_assignments = []
    
    districts = list(params['districts'].keys())
    global_phis = params['global_phis']
    global_weights = params['global_weights']
    K = len(global_phis)
    
    for district in  districts:
        n_students = int(match_stats_df[
            match_stats_df['Residential District'] == district
        ]['Total Applicants'].iloc[0])
        log_and_print(f"Handling district {district} with {n_students} students", outfile)
        # Get district-specific info
        sigma_d = params['districts'][district]['central_ranking']
        schools_list = params['districts'][district]['schools']
        school_to_idx = {s: i for i, s in enumerate(schools_list)}
        
        #log_and_print(f" Generating rankings for {n_students} students of length {k_ranking_length} amongst {len(schools_list)} schools")
        
        # Sample from global mixture
        rankings = []
        for _ in range(n_students):
            # Choose component from global mixture
            k = np.random.choice(K, p=global_weights)
            
            # Sample from Mallows(σ_d, φ_k)
            sigma_indices = np.array([school_to_idx[s] for s in sigma_d])
            ranking = mallows_insertion_sampling(sigma_indices, global_phis[k])
            
            # Truncate to k schools
            ranking = ranking[:k_ranking_length]
            
            rankings.append(ranking)

        rankings_as_schools = [[schools_list[idx] for idx in r] for r in rankings]
        
        all_rankings.extend(rankings_as_schools)
        all_district_assignments.extend([district] * n_students)
    
    all_schools = df['School DBN'].unique()
    school_to_idx = {s: i for i, s in enumerate(all_schools)}
    
    rankings_as_indices = []
    for ranking in all_rankings:
        rankings_as_indices.append(np.array([school_to_idx[s] for s in ranking]))
    
    capacities_dict = school_info_df.set_index('School DBN')['Capacity'].to_dict()
    capacities = np.array([capacities_dict.get(s, 0) for s in all_schools])
    log_and_print(f"  Total schools: {len(all_schools)}, Total capacity: {capacities.sum()}, Total students: {len(all_rankings)}", log_file=outfile)

    matches_idx = gale_shapley(rankings_as_indices, lottery_global, capacities)
    matches_schools = np.array([all_schools[m] if m >= 0 else '-1' for m in matches_idx])

    num_matched = np.sum(matches_idx >= 0)
    num_unmatched = np.sum(matches_idx == -1)
    log_and_print(f"    Matched: {num_matched}/{len(matches_idx)}, Unmatched: {num_unmatched}", log_file=outfile)

    if num_matched > 0:
        match_positions = []
        for i, ranking in enumerate(rankings_as_indices):
            if matches_idx[i] >= 0:
                match_pos = np.where(ranking == matches_idx[i])[0]
                if len(match_pos) > 0:
                    match_positions.append(match_pos[0])

    agg = compute_aggregates(all_rankings, matches_schools, 
                            np.array(all_district_assignments), all_schools)
    #log_and_print(f" Aggregate results: {agg}", log_file=log_file)
    return agg



def EM_algorithm(df, match_stats_df, school_info_df,
                 max_iter=10, tol=0.01, K=1, M_simulations=20, seed=42, outfile=None):
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
    log_and_print(f"  Max iterations: {max_iter}", log_file=outfile)
    log_and_print(f"  Simulations per evaluation: M={M_simulations}\n", log_file=outfile)
    
    # Initialize with GLOBAL mixture
    params = initialize_parameters_global_mixture(districts, df, K)

    observed_agg = extract_observed_aggregates(df, match_stats_df)
    
    log_likelihoods = []
    
    # EM loop
    for iteration in range(max_iter):
        log_and_print(f"\n{'='*60}", log_file=outfile)
        log_and_print(f"EM ITERATION {iteration + 1}/{max_iter}", log_file=outfile)
        log_and_print(f"{'='*60}", log_file=outfile)
        
        old_params = copy.deepcopy(params)
        
        # M-STEP: Optimize global parameters
        params, final_agg = optimize_global_mixture(
            params, observed_agg, df, match_stats_df, 
            school_info_df, M=M_simulations, seed=seed,
            iteration=iteration, outfile=outfile
        )

        # Sort them to remove indexing ambiguity
        sorted_indices = np.argsort(params['global_phis'])
        params['global_phis'] = params['global_phis'][sorted_indices]
        params['global_weights'] = params['global_weights'][sorted_indices]

        # M-STEP: Nudge sigmas using the result of the simulation above
        params = nudge_district_sigmas(params, final_agg, school_info_df)
        
        # Compute total log-likelihood
        log_and_print("\n  Computing final log-likelihood at optimized parameters...", log_file=outfile)
        total_log_lik = compute_log_likelihood_gaussian_all_districts(
            params, observed_agg, df, match_stats_df, 
            school_info_df, M=M_simulations, seed=seed,
            iteration=iteration, outfile=outfile
        )
        
        log_likelihoods.append(total_log_lik)
        log_and_print(f"\nTotal log-likelihood: {total_log_lik:.2f}", log_file=outfile)
        
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
    
    log_and_print(f"\nFinal global parameters:", log_file=outfile)
    log_and_print(f"  Global phis: {params['global_phis']}", log_file=outfile)
    log_and_print(f"  Global weights: {params['global_weights']}", log_file=outfile)
    
    return params, lottery_global, log_likelihoods, final_agg

def initialize_parameters_global_mixture(districts, df, K=1):
    """
    Initialize with global phis, district-specific sigmas
    """
    
    # Global mixture parameters (shared across districts)
    global_phis = np.random.beta(5, 1, K)
    global_phis = np.clip(global_phis, 0.75, 0.99)
    
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
                                                   M=1, seed=42, iteration=1, outfile=None):
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
            lottery_fixed, M_val=sim, outfile=outfile
        )

        total_filled += agg['filled']
        
        # Extract stats for EACH district from this single simulation
        for d_idx, district in enumerate(districts):
            agg_vec = agg['match_stats'][d_idx, :]
            simulated_samples[district].append(agg_vec)
    
    mean_filled = total_filled / M
    # Get capacities in same order as all_schools
    capacities = np.array([capacities_dict.get(s, 0) for s in all_schools])
    sim_util = mean_filled / capacities * 100

    # Get observed utilization only for schools we have
    obs_util_dict = school_info_df.set_index('School DBN')['Utilization'].to_dict()
    obs_util = np.array([obs_util_dict.get(s, 0) for s in all_schools])
    util_penalty = -0.1 * np.mean((obs_util - sim_util)**2)
    
    log_and_print('')  # New line after progress indicator
    
    log_and_print("\n" + "="*60, log_file=outfile)
    log_and_print(f"FIT DIAGNOSTICS | Seed: {seed} | Iteration: {iteration}", log_file=outfile)
    log_and_print("="*60, log_file=outfile)
    
    for d_idx, district in enumerate(districts):  
        obs = observed_agg[district]['match_stats']
        sim = agg['match_stats'][d_idx, :]
        
        log_and_print(f"\nDistrict {district}:", log_file=outfile)
        log_and_print(f"  Observed:  top3={obs[0]:5.1f}%, top5={obs[1]:5.1f}%, top10={obs[2]:5.1f}%, unmatched={obs[3]:5.1f}%", log_file=outfile)
        log_and_print(f"  Simulated: top3={sim[0]:5.1f}%, top5={sim[1]:5.1f}%, top10={sim[2]:5.1f}%, unmatched={sim[3]:5.1f}%", log_file=outfile)
        log_and_print(f"  Difference: top3={obs[0]-sim[0]:+5.1f}, top5={obs[1]-sim[1]:+5.1f}, top10={obs[2]-sim[2]:+5.1f}, unmatched={obs[3]-sim[3]:+5.1f}", log_file=outfile)
    
    log_and_print("Global School Utilization (Top 5 Mismatches):", log_file=outfile)
    util_diff = obs_util - sim_util
    mismatch_indices = np.argsort(np.abs(util_diff))[::-1][:5]
    for idx in mismatch_indices:
        s_name = school_info_df.iloc[idx]["School DBN"]
        log_and_print(f"  {s_name}: Obs={obs_util[idx]:5.1f}%, Sim={sim_util[idx]:5.1f}%, Diff={util_diff[idx]:+5.1f}%", log_file=outfile)
    
    log_and_print(f"  Mean Absolute Utilization Error: {np.mean(np.abs(util_diff)):.2f}%", log_file=outfile)
    
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
    
    return total_log_lik + util_penalty

def optimize_global_mixture(params, observed_agg, df, match_stats_df, 
                            school_info_df, M=20, seed=42, iteration=1, outfile=None):
    K = len(params['global_phis'])
    best_agg_stats = None  # To capture utilization for the nudge
    
    for k in range(K):
        phi_k_initial = params['global_phis'][k]
        
        def objective_global_phi_k(phi):
            nonlocal best_agg_stats
            original_phi = params['global_phis'][k]
            params['global_phis'][k] = phi
            
           
            total_log_lik = compute_log_likelihood_gaussian_all_districts(
                params, observed_agg, df, match_stats_df, 
                school_info_df, M=M, seed=seed, iteration=iteration, outfile=outfile
            )
            
            params['global_phis'][k] = original_phi
            return -total_log_lik
        
        result = minimize_scalar(
            objective_global_phi_k,
            bounds=(0.01, 0.99),
            method='bounded',
            options={'xatol': 0.01, 'maxiter': 10}
        )
        params['global_phis'][k] = result.x

    # Average M simulations to get robust aggregate for the nudge
    n_students_total = int(match_stats_df['Total Applicants'].sum())
    # Fixed lottery across all M simulations
    lottery_fixed = np.random.permutation(n_students_total)
    agg_accum = None
    for sim in range(M):
        # Only vary the Mallows preference sampling, not the lottery
        np.random.seed(seed + sim)
        agg_sim = run_single_simulation(params, df, match_stats_df, school_info_df, lottery_fixed)
        
        if agg_accum is None:
            agg_accum = {k: v.copy() for k, v in agg_sim.items()}
        else:
            for key in agg_accum:
                agg_accum[key] = agg_accum[key] + agg_sim[key]
    
    # Average the accumulated results
    final_agg = {k: v / M for k, v in agg_accum.items()}
    
    return params, final_agg

def nudge_district_sigmas(params, final_agg, school_info_df, eta=0.1):
    sim_filled = pd.Series(final_agg['filled'], index=params['districts'][next(iter(params['districts']))]['schools']) # or your master schools list
    real_util_counts = (school_info_df.set_index('School DBN')['Utilization'] / 100) * school_info_df.set_index('School DBN')['Capacity']
    
    util_error = real_util_counts - sim_filled
    
    for d_id, d_data in params['districts'].items():
        if 'pop_scores' not in d_data:
            d_data['pop_scores'] = {s: (len(d_data['schools']) - i) 
                                   for i, s in enumerate(d_data['central_ranking'])}
        
        for s_dbn, error in util_error.items():
            if s_dbn in d_data['pop_scores']:
                d_data['pop_scores'][s_dbn] += eta * error 
        
        new_sigma = sorted(d_data['pop_scores'].items(), key=lambda x: x[1], reverse=True)
        d_data['central_ranking'] = [s[0] for s in new_sigma]
        
    return params