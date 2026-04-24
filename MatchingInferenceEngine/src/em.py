import pandas as pd
import numpy as np
from scipy.optimize import minimize_scalar
import copy
import time
from concurrent.futures import ProcessPoolExecutor
from util import log_and_print
from data_ingestion import extract_observed_aggregates
from gale_shapley import compute_aggregates, gale_shapley_per_school_numba_wrapper
from mallows import  _sample_students_chunk
from list_length import sample_truncated_normal_lengths, sample_empirical_lengths
from priority_attributes import sample_student_attributes, build_composite_rank_matrix


def sample_rankings(
    params,
    match_stats_df,
    sampling_n_jobs=32,
    sampling_chunk_size=2000,
    list_length_max=10,
    executor=None,
):
    """
    Sample Mallows preference rankings for all students across all districts.
    Returns raw index-based rankings (before list length truncation),
    district assignments, and the rng used.
    """
    districts = list(params['districts'].keys())
    all_rankings = []
    all_district_assignments = []
    all_chunks = []
    rng = np.random.default_rng(seed=np.random.randint(0, 2**32))

    for district in districts:
        n_students = int(
            match_stats_df[
                match_stats_df['Residential District'] == district
            ]['Total Applicants'].iloc[0]
        )
        sigma_d = params['districts'][district]['central_ranking']
        schools_list = params['districts'][district]['schools']
        school_to_idx = {s: i for i, s in enumerate(schools_list)}
        sigma_indices = np.array([school_to_idx[s] for s in sigma_d])

        component_indices = rng.choice(
            len(params['global_phis']),
            size=n_students,
            p=params['global_weights']
        )
        for start in range(0, n_students, sampling_chunk_size):
            chunk = component_indices[start:start + sampling_chunk_size]
            all_chunks.append(
                (district, schools_list, sigma_indices, chunk, rng.integers(2**32), list_length_max)
            )
        all_district_assignments.extend([district] * n_students)

    results_by_district = {d: [] for d in districts}

    if sampling_n_jobs > 1 and executor is not None:
        futures = []
        for district, schools_list, sigma_indices, chunk, seed, list_length_max in all_chunks:
            future = executor.submit(_sample_students_chunk, sigma_indices, params['global_phis'], chunk, seed, list_length_max)
            futures.append((district, future))
        for district, future in futures:
            results_by_district[district].extend(future.result())
    elif sampling_n_jobs > 1:
        with ProcessPoolExecutor(max_workers=sampling_n_jobs) as pool:
            futures = []
            for district, schools_list, sigma_indices, chunk, seed, list_length_max in all_chunks:
                future = pool.submit(_sample_students_chunk, sigma_indices, params['global_phis'], chunk, seed, list_length_max)
                futures.append((district, future))
            for district, future in futures:
                results_by_district[district].extend(future.result())
    else:
        for district, schools_list, sigma_indices, chunk, seed, list_length_max in all_chunks:
            results_by_district[district].extend(
                _sample_students_chunk(sigma_indices, params['global_phis'], chunk, seed, list_length_max)
            )

    # Convert to school DBNs — full lists, no truncation
    for district in districts:
        schools_list = params['districts'][district]['schools']
        rankings = results_by_district[district]
        rankings_as_schools = [[schools_list[idx] for idx in r] for r in rankings]
        all_rankings.extend(rankings_as_schools)

    return all_rankings, all_district_assignments, rng

def run_matching(
    all_rankings,
    all_district_assignments,
    df,
    school_info_df,
    lottery_global,
    list_length_min,
    list_length_mean,
    list_length_std,
    list_length_max,
    rng,
    priority_config=None,
    district_to_region=None,
    district_to_borough=None,
    per_school_lottery=False,
    student_attrs=None
):
    """
    Given pre-sampled full rankings, applies list length truncation and runs DA.
    Isolates the matching step so the same preference profile can be reused
    across multiple list_length_min values.
    """
    from list_length import sample_truncated_normal_lengths

    all_schools = df['School DBN'].unique()
    school_to_idx = {s: i for i, s in enumerate(all_schools)}
    capacities_dict = school_info_df.set_index('School DBN')['Capacity'].to_dict()
    capacities = np.array([capacities_dict.get(s, 0) for s in all_schools])
    dbn_to_progs = {s: [s] for s in all_schools}
    n_students = len(all_rankings)
    n_schools = len(all_schools)

    # Group rankings by district to apply per-district list length sampling
    district_list = list(dict.fromkeys(all_district_assignments))  # preserve order
    district_indices = {}
    for i, d in enumerate(all_district_assignments):
        district_indices.setdefault(d, []).append(i)

    truncated_rankings = [None] * n_students
    all_list_lengths = [None] * n_students

    for district, indices in district_indices.items():
        n_d = len(indices)
        schools_list = [s for r in [all_rankings[i] for i in indices] for s in r]
        max_schools = max(len(all_rankings[i]) for i in indices)
        max_len_here = min(list_length_max, max_schools)
        effective_min = min(list_length_min, max_len_here)

        list_lengths = sample_truncated_normal_lengths(
            n_students=n_d,
            mean=list_length_mean,
            std=list_length_std,
            min_len=effective_min,
            max_len=max_len_here,
            rng=rng,
        )
        for j, (idx, L) in enumerate(zip(indices, list_lengths)):
            truncated_rankings[idx] = all_rankings[idx][:L]
            all_list_lengths[idx] = L

    def expand_ranking(ranking):
        seen = set()
        expanded = []
        for dbn in ranking:
            if dbn in seen or dbn not in school_to_idx:
                continue
            seen.add(dbn)
            expanded.append(school_to_idx[dbn])
        return np.array(expanded, dtype=np.int32)

    rankings_as_indices = [expand_ranking(r) for r in truncated_rankings]

    if per_school_lottery:
        school_lotteries = rng.random((n_schools, n_students))
    else:
        lottery_1d = np.argsort(lottery_global[:n_students]).astype(np.float64) / n_students
        school_lotteries = np.tile(lottery_1d, (n_schools, 1))

    if priority_config is not None and student_attrs is not None:
        _d2r = district_to_region or {str(d): str(d) for d in set(all_district_assignments)}
        school_lotteries = build_composite_rank_matrix(
            all_schools, student_attrs, priority_config,
            school_lotteries, _d2r, all_district_assignments,
        )

    matches_idx = gale_shapley_per_school_numba_wrapper(rankings_as_indices, school_lotteries, capacities)
    matches_schools = np.array([all_schools[m] if m >= 0 else '-1' for m in matches_idx])

    agg = compute_aggregates(
        truncated_rankings,
        matches_schools,
        np.array(all_district_assignments),
        all_schools,
    )

    return agg, truncated_rankings, rankings_as_indices, matches_idx, student_attrs

def run_single_simulation(
    params,
    df,
    match_stats_df,
    school_info_df,
    lottery_global=None,
    k_ranking_length=10,
    list_length_mode="fixed",  
    list_length_mean=10,
    list_length_std=2,
    list_length_min=1,
    list_length_max=12,
    list_length_empirical_probs=None,
    return_student_data=False,
    outfile=None,
    sampling_n_jobs=32,
    sampling_chunk_size=2000,
    executor=None,
    per_school_lottery=False,
    return_rankings=False,
    profile_timing=False,
    priority_config=None,
    district_to_region=None,
    district_to_borough=None
):
    t_total_start = time.perf_counter()
    timings = {}

    def _mark_timing(label, t_start):
        timings[label] = timings.get(label, 0.0) + (time.perf_counter() - t_start)

    all_rankings = []
    all_district_assignments = []
    all_list_lengths = []

    districts = list(params['districts'].keys())

    # Collect all chunks across all districts
    t_chunks_start = time.perf_counter()
    all_chunks = []  # (district, schools_list, sigma_indices, chunk_components, seed)
    rng = np.random.default_rng(seed=np.random.randint(0, 2**32))
    
    for district in districts:
        n_students = int(
            match_stats_df[
                match_stats_df['Residential District'] == district
            ]['Total Applicants'].iloc[0]
        )

        sigma_d = params['districts'][district]['central_ranking']
        schools_list = params['districts'][district]['schools']
        school_to_idx = {s: i for i, s in enumerate(schools_list)}
        sigma_indices = np.array([school_to_idx[s] for s in sigma_d])

        component_indices = rng.choice(
            len(params['global_phis']),
            size=n_students,
            p=params['global_weights']
        )

        for start in range(0, n_students, sampling_chunk_size):
            chunk = component_indices[start:start + sampling_chunk_size]
            all_chunks.append(
                (district, schools_list, sigma_indices, chunk, rng.integers(2**32))
            )

        all_district_assignments.extend([district] * n_students)
    _mark_timing('build_chunks', t_chunks_start)

    # ONE pool for all districts
    results_by_district = {d: [] for d in districts}

    t_sampling_start = time.perf_counter()
    if sampling_n_jobs > 1 and executor is not None:
        futures = []
        for district, schools_list, sigma_indices, chunk, seed in all_chunks:
            future = executor.submit(
                _sample_students_chunk,
                sigma_indices,
                params['global_phis'],
                chunk,
                seed
            )
            futures.append((district, future))

        for district, future in futures:
            results_by_district[district].extend(future.result())

    elif sampling_n_jobs > 1:
        with ProcessPoolExecutor(max_workers=sampling_n_jobs) as pool:
            futures = []
            for district, schools_list, sigma_indices, chunk, seed in all_chunks:
                future = pool.submit(
                    _sample_students_chunk,
                    sigma_indices,
                    params['global_phis'],
                    chunk,
                    seed
                )
                futures.append((district, future))

            for district, future in futures:
                results_by_district[district].extend(future.result())

    else:
        for district, schools_list, sigma_indices, chunk, seed in all_chunks:
            results_by_district[district].extend(
                _sample_students_chunk(
                    sigma_indices,
                    params['global_phis'],
                    chunk,
                    seed
                )
            )
    _mark_timing('sample_preferences', t_sampling_start)

    # Convert to school DBNs and truncate
    t_convert_start = time.perf_counter()
    for district in districts:
        schools_list = params['districts'][district]['schools']
        rankings = results_by_district[district]
        n_students_d = len(rankings)

        if list_length_mode == "fixed":
            max_len_here = min(k_ranking_length, len(schools_list))
            list_lengths = np.full(n_students_d, max_len_here, dtype=int)

        elif list_length_mode == "gaussian":
            max_len_here = min(list_length_max, len(schools_list))
            list_lengths = sample_truncated_normal_lengths(
                n_students=n_students_d,
                mean=list_length_mean,
                std=list_length_std,
                min_len=list_length_min,
                max_len=max_len_here,
                rng=rng
            )
        elif list_length_mode == "empirical":
            if list_length_empirical_probs is None:
                raise ValueError("list_length_mode='empirical' requires list_length_empirical_probs")
            list_lengths = sample_empirical_lengths(n_students_d, list_length_empirical_probs, rng)
        else:
            raise ValueError(f"Unknown list_length_mode: {list_length_mode}")

        truncated_rankings = [r[:L] for r, L in zip(rankings, list_lengths)]
        rankings_as_schools = [[schools_list[idx] for idx in r] for r in truncated_rankings]

        all_rankings.extend(rankings_as_schools)
        all_list_lengths.extend(list_lengths.tolist())
    _mark_timing('convert_and_truncate', t_convert_start)

    log_and_print(
        f" Generated {len(all_rankings)} student rankings across {len(districts)} districts ({len(all_chunks)} chunks)",
        log_file=outfile
    )

    t_match_prep_start = time.perf_counter()
    all_schools = df['School DBN'].unique()
    school_to_idx = {s: i for i, s in enumerate(all_schools)}


    capacities_dict = school_info_df.set_index('School DBN')['Capacity'].to_dict()
    capacities = np.array([capacities_dict.get(s, 0) for s in all_schools])
    _mark_timing('prepare_matching_inputs', t_match_prep_start)

    dbn_to_progs = {s: [s] for s in all_schools} 

    def expand_ranking(ranking):
        seen = set()
        expanded = []
        for dbn in ranking:
            if dbn in seen or dbn not in school_to_idx:
                continue
            seen.add(dbn)
            expanded.append(school_to_idx[dbn])
        return np.array(expanded, dtype=np.int32)

    rankings_as_indices = [expand_ranking(r) for r in all_rankings]

    student_attrs = None
    if priority_config is not None:
        student_attrs = sample_student_attributes(
            district_assignments=all_district_assignments,
            all_schools=all_schools,
            dbn_to_progs=dbn_to_progs,
            priority_config=priority_config,
            district_to_region=district_to_region or {str(d): str(d) for d in set(all_district_assignments)},
            rng=rng,
            district_to_borough=district_to_borough,
        )

    t_matching_start = time.perf_counter()
    n_students = len(rankings_as_indices)
    n_schools = len(all_schools)

    if per_school_lottery:
        school_lotteries = rng.random((n_schools, n_students))
    else:
        lottery_1d = np.argsort(lottery_global[:n_students]).astype(np.float64) / n_students
        school_lotteries = np.tile(lottery_1d, (n_schools, 1))

    if priority_config is not None and student_attrs is not None:
        _d2r = district_to_region or {str(d): str(d) for d in set(all_district_assignments)}
        school_lotteries = build_composite_rank_matrix(
            all_schools, student_attrs, priority_config,
            school_lotteries, _d2r, all_district_assignments,
        )

    matches_idx = gale_shapley_per_school_numba_wrapper(rankings_as_indices, school_lotteries, capacities)
    matches_schools = np.array([all_schools[m] if m >= 0 else '-1' for m in matches_idx])
    _mark_timing('matching', t_matching_start)

    t_agg_start = time.perf_counter()
    agg = compute_aggregates(
        all_rankings,
        matches_schools,
        np.array(all_district_assignments),
        all_schools
    )
    _mark_timing('compute_aggregates', t_agg_start)

    if profile_timing:
        timings['total'] = time.perf_counter() - t_total_start
        log_and_print(
            (
                " [TIMING] run_single_simulation "
                f"total={timings['total']:.3f}s | "
                f"build_chunks={timings.get('build_chunks', 0.0):.3f}s | "
                f"sample_preferences={timings.get('sample_preferences', 0.0):.3f}s | "
                f"convert_and_truncate={timings.get('convert_and_truncate', 0.0):.3f}s | "
                f"prepare_matching_inputs={timings.get('prepare_matching_inputs', 0.0):.3f}s | "
                f"matching={timings.get('matching', 0.0):.3f}s | "
                f"compute_aggregates={timings.get('compute_aggregates', 0.0):.3f}s"
            ),
            log_file=outfile,
        )
        log_and_print(
            (
                " [TIMING] workload "
                f"districts={len(districts)} | students={len(all_rankings)} | "
                f"chunks={len(all_chunks)} | sampling_n_jobs={sampling_n_jobs} | "
                f"chunk_size={sampling_chunk_size}"
            ),
            log_file=outfile,
        )

    if return_student_data:
        student_df = pd.DataFrame({
            "student_id": np.arange(len(all_rankings)),
            "district": np.array(all_district_assignments),
            "list_length": np.array(all_list_lengths),
            "match": matches_schools,
            "unmatched": (matches_schools == "-1").astype(int),
        })
        return agg, student_df


    if return_rankings:
        return agg, all_rankings, rankings_as_indices, matches_idx, np.array(all_district_assignments), student_attrs
    return agg


def EM_algorithm(df, match_stats_df, school_info_df,
                 max_iter=10, tol=0.01, K=1, M_simulations=20, seed=40, outfile=None, 
                 sampling_n_jobs=32, max_iter_opt=5, per_school_lottery=False, simulation_kwargs=None):
    """
    EM algorithm with GLOBAL MIXTURE
    """
    
    np.random.seed(seed)
    simulation_kwargs = {} if simulation_kwargs is None else simulation_kwargs
    profile_timing = bool(simulation_kwargs.get('profile_timing', False))
                   
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
    if profile_timing:
        log_and_print("  Timing instrumentation: ENABLED", log_file=outfile)
    
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
            executor=warm_executor, max_iter_em=max_iter, max_iter_opt=max_iter_opt,
            per_school_lottery=per_school_lottery, simulation_kwargs=simulation_kwargs
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
                                                   executor=None, sampling_n_jobs=32, per_school_lottery=False, simulation_kwargs=None):
    """
    Compute log-likelihood for ALL districts at once
    
    This is more efficient than calling compute_log_likelihood_gaussian() 
    separately for each district because we only run M simulations total
    instead of M x num_districts simulations.
    
    Returns:
        total_log_lik: Sum of log-likelihoods across all districts
    """
    simulation_kwargs = {} if simulation_kwargs is None else simulation_kwargs
    profile_timing = bool(simulation_kwargs.get('profile_timing', False))
    t_total_start = time.perf_counter()
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
    lottery_fixed = None if per_school_lottery else rng_lottery.permutation(n_students_total)
    
    for sim in range(M):
        log_and_print(f"      Simulation {sim+1}/{M}...", log_file=outfile)
        t_sim_start = time.perf_counter()
        
        # Only vary the Mallows preference sampling, not the lottery
        np.random.seed(seed + sim)
        
        # Simulate ALL districts together (do this ONCE per M iteration)
        agg = run_single_simulation(
            params_global, df, match_stats_df, school_info_df, 
            lottery_fixed, outfile=outfile, executor=executor,
            sampling_n_jobs=sampling_n_jobs, per_school_lottery=per_school_lottery, **simulation_kwargs
        )

        total_filled += agg['filled']
        
        # Extract stats for EACH district from this single simulation
        for d_idx, district in enumerate(districts):
            agg_vec = agg['match_stats'][d_idx, :]
            simulated_samples[district].append(agg_vec)

        if profile_timing:
            log_and_print(
                f"      [TIMING] simulation {sim+1}/{M}: {time.perf_counter() - t_sim_start:.3f}s",
                log_file=outfile,
            )
    
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

    if profile_timing:
        log_and_print(
            f"  [TIMING] compute_log_likelihood_gaussian_all_districts total: {time.perf_counter() - t_total_start:.3f}s",
            log_file=outfile,
        )
    
    log_and_print(f"  Match stats log-likelihood: {total_log_lik:.2f}, Util penalty: {util_penalty:.2f}, Combined: {total_log_lik + util_penalty:.2f}", log_file=outfile)
    return total_log_lik + util_penalty

def optimize_global_mixture(params, observed_agg, df, match_stats_df, 
                            school_info_df, M=20, seed=42, iteration=1,
                            sampling_n_jobs=32, outfile=None, executor=None, 
                            max_iter_em=5, max_iter_opt=5, per_school_lottery=False, simulation_kwargs=None):

    simulation_kwargs = {} if simulation_kwargs is None else simulation_kwargs
    profile_timing = bool(simulation_kwargs.get('profile_timing', False))
    t_opt_start = time.perf_counter()
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
            t_eval_start = time.perf_counter()
            log_and_print(f"    [EM iter {iteration+1}/{max_iter_em}] phi[{k+1}/{K}] eval #{eval_count[0]}, trying phi={phi:.4f}", log_file=outfile)
            original_phi = params['global_phis'][k]
            params['global_phis'][k] = phi
            
           
            total_log_lik = compute_log_likelihood_gaussian_all_districts(
                params, observed_agg, df, match_stats_df, 
                school_info_df, M=M, seed=seed, iteration=iteration, outfile=outfile, 
                executor=executor, sampling_n_jobs=sampling_n_jobs, per_school_lottery=per_school_lottery, simulation_kwargs=simulation_kwargs
            )
            last_log_like[0] = total_log_lik
            
            params['global_phis'][k] = original_phi
            if profile_timing:
                log_and_print(
                    (
                        f"    [TIMING] phi[{k+1}/{K}] eval #{eval_count[0]} "
                        f"duration: {time.perf_counter() - t_eval_start:.3f}s"
                    ),
                    log_file=outfile,
                )
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
    lottery_fixed = None if per_school_lottery else np.random.permutation(n_students_total)
    agg_accum = None
    for sim in range(M):
        # Only vary the Mallows preference sampling, not the lottery
        np.random.seed(seed + sim)
        log_and_print(f"  [EM iter {iteration+1}/{max_iter_em}] Final averaging sim {sim+1}/{M}...", log_file=outfile)
        agg_sim = run_single_simulation(params, df, match_stats_df, school_info_df, lottery_fixed, per_school_lottery=per_school_lottery,
                                        sampling_n_jobs=sampling_n_jobs, outfile=outfile, executor=executor, **simulation_kwargs)
        
        if agg_accum is None:
            agg_accum = {k: v.copy() for k, v in agg_sim.items()}
        else:
            for key in agg_accum:
                agg_accum[key] = agg_accum[key] + agg_sim[key]
    
    # Average the accumulated results
    final_agg = {k: v / M for k, v in agg_accum.items()}
    if profile_timing:
        log_and_print(
            f"  [TIMING] optimize_global_mixture total: {time.perf_counter() - t_opt_start:.3f}s",
            log_file=outfile,
        )
    
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
