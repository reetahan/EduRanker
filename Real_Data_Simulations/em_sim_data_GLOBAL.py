import pandas as pd
import numpy as np
from scipy.stats import  kendalltau
from scipy.optimize import minimize_scalar
import copy
import matplotlib.pyplot as plt
import argparse


EXP_OUT_FOLDER = "experiment-results/"
DATA_GENERATION_SEED = 44

def read_data(file_path, sheet=0):
    """
    Reads data from the given file path and returns a pandas DataFrame.
    """
    if file_path.endswith('.csv'):
        data = pd.read_csv(file_path)
    else:
        data = pd.read_excel(file_path, sheet_name=sheet)
    return data

def log_and_print(message, log_file=None):
    """Print to console and optionally write to file with immediate flush"""
    print(message)
    if log_file is not None:
        f = open(log_file, "a+")
        f.write(message + '\n')
        f.flush()
        f.close()


def preprocess_data(df, match_stats_df, school_info_df, addtl_school_info_df):

    df = df[['School DBN', 'School Name', 'School District', 'Residential District', 
         'Total Applicants by Residential District', 'True Applicants by Residential District',
         'Total Applicants School', 'Total True Applicants School', 'Ratio', 'Rank']]
    df = df[df['Residential District'] != 'Unknown']
    dtype_mapping = {}
    for i in range(len(df.columns.array)):
        if(i > 2):
            dtype_mapping[df.columns.array[i]] = 'int64'
    df = df.astype(dtype_mapping)

    school_cols_sum = [f"seats9ge{i}" for i in range(1,12)] + [f"seats9swd{i}" for i in range(1,12)] 
    school_info_df['Capacity'] = school_info_df.apply(lambda x: sum(x[col] if pd.notnull(x[col]) else 0 for col in school_cols_sum), axis=1)
    
    # Calculate Utilization from enrollment data
    addtl_school_info_df = addtl_school_info_df[(addtl_school_info_df['Category'] == 'All Students') & (pd.to_numeric(addtl_school_info_df['Grade 9 Students'], errors='coerce').notna())]
    addtl_school_info_df  = addtl_school_info_df[['School DBN', 'Grade 9 Students']]
    addtl_school_info_df['Grade 9 Students'] = addtl_school_info_df['Grade 9 Students'].astype(int)
    school_info_df = school_info_df[['dbn','Capacity']]
    school_info_df = school_info_df.rename(columns={'dbn': 'School DBN'})
    school_info_df = school_info_df[school_info_df['School DBN'].isin(df['School DBN'].unique())]
    school_info_df = addtl_school_info_df.join(school_info_df.set_index('School DBN'), on='School DBN', how='inner')
    school_info_df['Utilization'] = (school_info_df['Grade 9 Students'] / school_info_df['Capacity'] * 100).clip(upper=100)
    school_info_df = school_info_df[['School DBN', 'Capacity', 'Utilization']]

    match_stats_df.columns = match_stats_df.iloc[0]
    match_stats_df = match_stats_df.drop(match_stats_df.index[0])
    match_stats_df = match_stats_df[['Residential District', 'Total Applicants', '% Matches to Choice 1-3', 
                                    '% Matches to Choice 1-5', '% Matches to Choice 1-10', '% Matches to Choice 1-12']]
    dtype_mapping = {}
    for i in range(len(match_stats_df.columns.array)):
        if(i > 0):
            match_stats_df[match_stats_df.columns.array[i]] = match_stats_df[match_stats_df.columns.array[i]].str.replace('%','').str.replace(',','')
            dtype_mapping[match_stats_df.columns.array[i]] = 'float64'
    match_stats_df = match_stats_df.astype(dtype_mapping)
    match_stats_df['Unmatched'] = 100.0 - match_stats_df['% Matches to Choice 1-12'].astype(float)
    match_stats_df = match_stats_df.drop(columns=['% Matches to Choice 1-12'])
    match_stats_df = match_stats_df[~match_stats_df['Residential District'].isin(['Total', 'Unknown '])]
    match_stats_df['Residential District'] = pd.to_numeric(match_stats_df['Residential District'])
    
    avg_list_length = df['Total Applicants by Residential District'].sum() / match_stats_df['Total Applicants'].sum()
    log_and_print(f"Average list length from data: {avg_list_length:.2f}")
     
    return df, match_stats_df, school_info_df


def mallows_insertion_sampling(central_ranking, phi):
    n = len(central_ranking)
    ranking = []
    
    for i in range(n):
        item = central_ranking[i]
        
        if len(ranking) == 0:
            ranking.append(item)
        else:
            positions = len(ranking) + 1
            probs = np.array([phi ** j for j in range(positions)])
            probs = probs / probs.sum()
            pos = np.random.choice(positions, p=probs)
            ranking.insert(pos, item)
    
    return np.array(ranking)


def compute_aggregates(student_rankings, matches, district_assignments, schools_list):
    n_students = len(student_rankings)
    n_schools = len(schools_list)
    districts = np.unique(district_assignments)
    n_districts = len(districts)
    
    district_to_idx = {d: i for i, d in enumerate(districts)}
    school_to_idx = {s: i for i, s in enumerate(schools_list)}
    
    total_app = np.zeros((n_districts, n_schools))
    true_app = np.zeros((n_districts, n_schools))
    match_stats = np.zeros((n_districts, 4))
    filled = np.zeros(n_schools)
    
    for student_id in range(n_students):
        district_idx = district_to_idx[district_assignments[student_id]]
        ranking = student_rankings[student_id]
        if isinstance(ranking, np.ndarray):
            ranking = ranking.tolist()
        match = matches[student_id]
        
        for school in ranking:
            school_idx = school_to_idx[school]
            total_app[district_idx, school_idx] += 1
        
        if match != '-1':
            match = str(match)  
            match_school_idx = school_to_idx[match]
            try:
                match_position = ranking.index(match)  
            except ValueError:
                log_and_print(f"Warning: Student matched to {match} not in ranking: {ranking}")
                continue
            
            for school in ranking[match_position:]:
                school_idx = school_to_idx[school]
                true_app[district_idx, school_idx] += 1
            
            filled[match_school_idx] += 1
            
            if match_position < 3:
                match_stats[district_idx, 0] += 1
            if match_position < 5:
                match_stats[district_idx, 1] += 1
            if match_position < 10:
                match_stats[district_idx, 2] += 1
        else:
            for school in ranking:
                school_idx = school_to_idx[school]
                true_app[district_idx, school_idx] += 1
            
            match_stats[district_idx, 3] += 1
    

    for d in range(n_districts):
        # Count total students in this district
        district_total = np.sum(district_assignments == districts[d])
        
        if district_total > 0:
            match_stats[d, :] = (match_stats[d, :] / district_total) * 100
    
    return {
        'total_app': total_app,
        'true_app': true_app,
        'match_stats': match_stats,
        'filled': filled
    }

def gale_shapley(student_rankings, student_lottery_numbers, school_capacities):
    n_students = len(student_rankings)
    n_schools = len(school_capacities)
    
    student_order = np.argsort(student_lottery_numbers)
    
    matches = np.full(n_students, -1)
    school_tentative = [[] for _ in range(n_schools)]
    
    for student in student_order:
        for school in student_rankings[student]:
            if len(school_tentative[school]) < school_capacities[school]:
                school_tentative[school].append(student)
                matches[student] = school
                break
            elif len(school_tentative[school]) > 0: 
                worst = max(school_tentative[school], 
                           key=lambda s: student_lottery_numbers[s])
                if student_lottery_numbers[student] < student_lottery_numbers[worst]:
                    school_tentative[school].remove(worst)
                    matches[worst] = -1
                    school_tentative[school].append(student)
                    matches[student] = school
                    break
    
    return matches


def run_single_simulation(params, df, match_stats_df, school_info_df, 
                         lottery_global, k_ranking_length=10, M_val=1, outfile=None):
    """
    Run one simulation with GLOBAL MIXTURE parameters
    
    Args:
        params: Global mixture structure with 'global_phis', 'global_weights', 'districts'
    """

    log_file = None
    if outfile:
        log_file = open(outfile, 'w', buffering=1)
    
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
        
        # Convert to school DBNs
        rankings_as_schools = [[schools_list[idx] for idx in r] for r in rankings]
        
        all_rankings.extend(rankings_as_schools)
        all_district_assignments.extend([district] * n_students)
    
    # Rest is same as before
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
    log_and_print(f" Aggregate results: {agg}", log_file=log_file)
    return agg


def create_synthetic_experiment(n_students=500, n_schools=20, 
                                             capacity_per_school=30, k_ranking_length=10, 
                                             true_K=1, district_ct =3, seed=42):
    np.random.seed(seed)
    
    if true_K == 1:
        true_phis = np.array([0.3])
        true_weights = np.array([1.0])
    elif true_K == 2:
        true_phis = np.array([0.2, 0.6])
        true_weights = np.array([0.6, 0.4])
    else: 
        true_phis = np.array([0.15, 0.4, 0.7])
        true_weights = np.array([0.5, 0.3, 0.2])
    
    schools_list = [f"SCHOOL_{i:02d}" for i in range(n_schools)]
    school_to_idx = {s: i for i, s in enumerate(schools_list)}
    
    
    districts = list(range(1, district_ct + 1))

    # We'll create 3 districts. To test "contention," let's give them 
    # slightly different preferences for which schools are "Top".
    # District 1 likes School 0-19 in order. 
    # District 2 likes School 10-19 then 0-9. 
    # District 3 likes odd then even schools.
    true_sigmas = {
        1: schools_list.copy(),
        2: schools_list[10:] + schools_list[:10],
        3: [s for i, s in enumerate(schools_list) if i%2==0] + [s for i, s in enumerate(schools_list) if i%2!=0]
    }
    
    student_districts = np.random.choice(districts, size=n_students)
    all_rankings = []
    
    for d_id in student_districts:
        k = np.random.choice(true_K, p=true_weights)
        
        local_sigma_indices = np.array([school_to_idx[s] for s in true_sigmas[d_id]])
        ranking = mallows_insertion_sampling(local_sigma_indices, true_phis[k])
        all_rankings.append(ranking[:k_ranking_length])
        
    lottery = np.random.permutation(n_students)
    capacities = np.array([capacity_per_school] * n_schools)
    
    matches_idx = gale_shapley(all_rankings, lottery, capacities)
    matches_schools = np.array([schools_list[m] if m >= 0 else '-1' for m in matches_idx])

    utilization_counts = pd.Series(matches_schools).value_counts()
    school_info_df = pd.DataFrame([
        {'School DBN': s, 'Capacity': capacity_per_school, 
         'Utilization': (utilization_counts.get(s, 0) / capacity_per_school) * 100} 
        for s in schools_list
    ])

    match_stats_list = []
    rankings_as_schools = [[schools_list[idx] for idx in r] for r in all_rankings]
    
    for d_id in districts:
        mask = (student_districts == d_id)
        d_rankings = [rankings_as_schools[i] for i, val in enumerate(mask) if val]
        d_matches = matches_schools[mask]
        
        d_agg = compute_aggregates(d_rankings, d_matches, [d_id]*len(d_rankings), schools_list)
        stats = d_agg['match_stats'][0, :]
        
        match_stats_list.append({
            'Residential District': d_id,
            'Total Applicants': mask.sum(),
            '% Matches to Choice 1-3': stats[0],
            '% Matches to Choice 1-5': stats[1],
            '% Matches to Choice 1-10': stats[2],
            'Unmatched': stats[3]
        })
    
    match_stats_df = pd.DataFrame(match_stats_list)
    
    app_data = []
    for d_id in districts:
        mask = (student_districts == d_id)
        d_rankings = [rankings_as_schools[i] for i, val in enumerate(mask) if val]
        d_matches = matches_schools[mask]
        for s_idx, s_name in enumerate(schools_list):
            total_apps = sum(s_name in r for r in d_rankings)

            true_apps_count = 0
            for i, ranking in enumerate(d_rankings):
                if s_name in ranking:
                    final_match = d_matches[i]
                    idx_of_s = ranking.index(s_name)
                    if final_match is None or final_match == '-1':
                        true_apps_count += 1
                    else:
                        idx_of_match = ranking.index(final_match)
                        if idx_of_match >= idx_of_s:
                            true_apps_count += 1

            app_data.append({
                'School DBN': s_name, 
                'Residential District': d_id, 
                'Total Applicants by Residential District': total_apps,
                'True Applicants by Residential District': true_apps_count,
                'Ratio': (total_apps / capacity_per_school) 
            })
    
    df = pd.DataFrame(app_data)
    
    true_params = {
        'true_K': true_K, 'true_phis': true_phis, 
        'true_weights': true_weights, 'true_sigmas': true_sigmas
    }
    
    return df, match_stats_df, school_info_df, true_params


def extract_observed_aggregates(df, match_stats_df):
    """
    Extract observed aggregates for each district
    
    Returns:
        dict mapping district -> observed statistics
    """
    observed = {}
    
    districts = sorted(df['Residential District'].unique())
    
    for district in districts:
        df_d = df[df['Residential District'] == district]
        match_d = match_stats_df[
            match_stats_df['Residential District'] == district
        ].iloc[0]
        
        observed[district] = {
            'match_stats': np.array([
                match_d['% Matches to Choice 1-3'],
                match_d['% Matches to Choice 1-5'],
                match_d['% Matches to Choice 1-10'],
                match_d['Unmatched']
            ]),
            'total_app': df_d['Total Applicants by Residential District'].values,
            'true_app': df_d['True Applicants by Residential District'].values,
            'schools': df_d['School DBN'].values
        }
    
    return observed


def EM_algorithm(df, match_stats_df, school_info_df,
                 max_iter=10, tol=0.01, K=1, M_simulations=20, seed=42, outfile=None):
    """
    EM algorithm with GLOBAL MIXTURE
    """
    
    np.random.seed(seed)
    log_file = None
    if outfile:
        log_file = open(outfile, 'w', buffering=1)
    
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
            iteration=iteration
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
        
        central_ranking = sorted(schools_list, key=lambda s: obs_total[s])
        
        params['districts'][district] = {
            'schools': schools_list,
            'central_ranking': central_ranking
        }
    
    return params

def sample_students_global_mixture(params, district, n_students):
    """
    Sample students from global mixture with district-specific sigma
    """
    
    # Global parameters
    phis = params['global_phis']
    weights = params['global_weights']
    K = len(phis)
    
    # District-specific parameters
    sigma_d = params['districts'][district]['central_ranking']
    schools = params['districts'][district]['schools']
    
    rankings = []
    
    for _ in range(n_students):
        # Choose type from global mixture
        k = np.random.choice(K, p=weights)
        
        # Sample ranking from Mallows(σ_d, φ_k)
        school_to_idx = {s: i for i, s in enumerate(schools)}
        sigma_indices = np.array([school_to_idx[s] for s in sigma_d])
        
        ranking = mallows_insertion_sampling(sigma_indices, phis[k])
        rankings.append(ranking)
    
    return rankings

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
    log_file = None
    if outfile:
        log_file = open(outfile, 'w', buffering=1)

    districts = sorted(observed_agg.keys())
    n_students_total = int(match_stats_df['Total Applicants'].sum())
    
    # Run M simulations, collecting stats for all districts
    simulated_samples = {d: [] for d in districts}

    total_filled = np.zeros(len(school_info_df))
    
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
    sim_util = mean_filled / school_info_df['Capacity'].values * 100

    obs_util = school_info_df['Utilization'].values 
    util_penalty = -0.1 * np.mean((obs_util - sim_util)**2)
    
    log_and_print()  # New line after progress indicator
    
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
                            school_info_df, M=20, seed=42, iteration=1):
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
                school_info_df, M=M, seed=seed, iteration=iteration
            )
            
            params['global_phis'][k] = original_phi
            return -total_log_lik
        
        result = minimize_scalar(
            objective_global_phi_k,
            bounds=(0.01, 0.99),
            method='bounded',
            options={'xatol': 0.01}
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
    # util_error: Positive = school needs more students (move UP in ranking)
    real_util = (school_info_df['Utilization'] / 100) * school_info_df['Capacity']
    util_error = real_util.values - final_agg['filled']
    
    all_schools = list(school_info_df['School DBN'])
    
    for d_id, d_data in params['districts'].items():
        if 'pop_scores' not in d_data:
            d_data['pop_scores'] = {s: (len(d_data['schools']) - i) 
                                   for i, s in enumerate(d_data['central_ranking'])}
        
        for i, s_dbn in enumerate(all_schools):
            if s_dbn in d_data['pop_scores']:
                d_data['pop_scores'][s_dbn] += eta * util_error[i]
        
        # Sort by nudged scores to get new sigma
        new_sigma = sorted(d_data['pop_scores'].items(), key=lambda x: x[1], reverse=True)
        d_data['central_ranking'] = [s[0] for s in new_sigma]
        
    return params

def run_synthetic_experiment_3_MoM_no_utilization(outfile=None):
    log_and_print("\n" + "="*60, log_file=outfile)
    log_and_print("EXPERIMENT 3 MoM, Match Stats, No Utilization", log_file=outfile)
    log_and_print("="*60, log_file=outfile)
    
    all_match_stats = []
    observed_stats = None

    for seed in range(40, 45):
        log_and_print(f"\nRunning synthetic experiment with seed {seed}...", log_file=outfile)
        df3, match_stats_df3, school_info_df3, true_params3 = create_synthetic_experiment(
            n_students=500, n_schools=20, capacity_per_school=30,
            k_ranking_length=10, true_K=2, seed=DATA_GENERATION_SEED
        )

        # Store observed once (same across seeds since data seed=DATA_GENERATION_SEED is fixed)
        if observed_stats is None:
            observed_stats = np.array([
                match_stats_df3['% Matches to Choice 1-3'].iloc[0],
                match_stats_df3['% Matches to Choice 1-5'].iloc[0],
                match_stats_df3['% Matches to Choice 1-10'].iloc[0],
                match_stats_df3['Unmatched'].iloc[0]
            ])

        params3, lottery3, log_liks3, agg = EM_algorithm(
            df3, match_stats_df3, school_info_df3,
            max_iter=10, M_simulations=10, K=3, seed=seed
        )

        # Run one final simulation with estimated params
        agg = run_single_simulation(params3, df3, match_stats_df3, school_info_df3, lottery3)
        all_match_stats.append(agg['match_stats'][0, :])

        out_lines = [
            f"\nSEED {seed} RESULTS:",
            f"  True phis: {true_params3['true_phis']}",
            f"  Estimated phis: {params3['global_phis']}",
            f"  Error: {np.abs(params3['global_phis'] - true_params3['true_phis'])}"
        ]
        for line in out_lines:
            log_and_print(line, log_file=outfile)
        with open(f"{EXP_OUT_FOLDER}experiment3_results.txt", "a+") as f:
            for line in out_lines:
                f.write(line + "\n")
                f.flush()

    all_match_stats = np.array(all_match_stats)  

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.boxplot(all_match_stats, labels=['Top-3', 'Top-5', 'Top-10', 'Unmatched'])

    for i, obs in enumerate(observed_stats):
        ax.scatter(i + 1, obs, color='red', zorder=5, marker='D', 
                label='Observed' if i == 0 else '')

    ax.set_ylabel('Percentage (%)')
    ax.set_title('Match Statistics: Simulated vs Observed (K=3, Seeds 40-49)')
    ax.legend()
    plt.tight_layout()
    plt.savefig(f"{EXP_OUT_FOLDER}match_stats_boxplot_v2_K2.png", dpi=150)
    plt.show()

def run_synthetic_experiment_3_MoM_yes_utilization(outfile=None):
    log_and_print("\n" + "="*60, log_file=outfile)
    log_and_print("EXPERIMENT 3 MoM, Match Stats, Yes Utilization", log_file=outfile)
    log_and_print("="*60, log_file=outfile)
    
    all_match_stats = []
    all_utilizations = []
    observed_stats = None

    for seed in range(40, 50):
        log_and_print(f"\nRunning synthetic experiment with seed {seed}...")
        df3, match_stats_df3, school_info_df3, true_params3 = create_synthetic_experiment(
            n_students=500, n_schools=20, capacity_per_school=30,
            k_ranking_length=10, true_K=3, district_ct=3, seed=DATA_GENERATION_SEED
        )

        # Store observed once (same across seeds since data seed=DATA_GENERATION_SEED is fixed)
        if observed_stats is None:
            observed_stats = np.array([
                match_stats_df3['% Matches to Choice 1-3'].iloc[0],
                match_stats_df3['% Matches to Choice 1-5'].iloc[0],
                match_stats_df3['% Matches to Choice 1-10'].iloc[0],
                match_stats_df3['Unmatched'].iloc[0]
            ])
            true_utilization = school_info_df3['Utilization'].values / 100.0

        params3, lottery3, log_liks3, agg = EM_algorithm(
            df3, match_stats_df3, school_info_df3,
            max_iter=10, M_simulations=10, K=3, seed=seed
        )

        all_match_stats.append(agg['match_stats'][0, :])
        sim_util = agg['filled'] / school_info_df3['Capacity'].values
        all_utilizations.append(sim_util)


        out_lines = [
            f"\nSEED {seed} RESULTS:",
            f"  True phis: {true_params3['true_phis']}",
            f"  Estimated phis: {params3['global_phis']}",
            f"  Error: {np.abs(params3['global_phis'] - true_params3['true_phis'])}"
        ]

        for d_id, true_sigma in true_params3['true_sigmas'].items():
            est_sigma = params3['districts'][d_id]['central_ranking']
            
            # Map schools to ranks for Kendall Tau (how similar is the ordering?)
            school_to_true_rank = {s: i for i, s in enumerate(true_sigma)}
            true_ranks = [school_to_true_rank[s] for s in true_sigma]
            est_ranks = [school_to_true_rank[s] for s in est_sigma]
            tau, _ = kendalltau(true_ranks, est_ranks)
            
            out_lines.append(f"  District {d_id} Sigma Kendall Tau: {tau:.4f}")
            out_lines.append(f"    True Top 3: {true_sigma[:3]}")
            out_lines.append(f"    Est  Top 3: {est_sigma[:3]}")

        for line in out_lines:
            log_and_print(line)
        with open(f"{EXP_OUT_FOLDER}experiment3_3_dists_utils_results.txt", "a+") as f:
            for line in out_lines:
                f.write(line + "\n")
                f.flush()

    all_match_stats = np.array(all_match_stats)  

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.boxplot(all_match_stats, labels=['Top-3', 'Top-5', 'Top-10', 'Unmatched'])

    for i, obs in enumerate(observed_stats):
        ax.scatter(i + 1, obs, color='red', zorder=5, marker='D', 
                label='Observed' if i == 0 else '')

    ax.set_ylabel('Percentage (%)')
    ax.set_title('Match Statistics: Simulated vs Observed (K=3, Seeds 40-49)')
    ax.legend()
    plt.tight_layout()
    plt.savefig(f"{EXP_OUT_FOLDER}match_stats_boxplot_v2_K3_3_dists_utils.png", dpi=150)
    plt.show()

    all_util_array = np.array(all_utilizations) # Shape: (Seeds, Schools)
    fig2, ax2 = plt.subplots(figsize=(12, 5))
    
    # Create boxplot for each school (columns of the array)
    bp = ax2.boxplot(all_util_array, patch_artist=True)
    
    # Customize boxes
    for patch in bp['boxes']:
        patch.set_facecolor('lightblue')
        patch.set_alpha(0.6)

    # Overlay True Values
    ax2.scatter(range(1, len(true_utilization) + 1), true_utilization, 
                color='red', marker='D', s=30, zorder=5, label='True Observed Util')

    ax2.set_xlabel('School Index')
    ax2.set_ylabel('Utilization (Fraction of Capacity)')
    ax2.set_title('School-Level Utilization: Simulated (Box) vs True (Red Diamond)')
    ax2.axhline(y=1.0, color='gray', linestyle='--', alpha=0.5, label='100% Capacity')
    ax2.legend()
    
    plt.tight_layout()
    plt.savefig(f"{EXP_OUT_FOLDER}school_utilization_boxplot.png", dpi=150)
    plt.show()

def run_real(outfile):
    df = read_data('data/master_data_03_residential_district.xlsx')
    match_stats_df = read_data('../Data-Analysis/raw-data/DATA3_fall-2024-high-school-offer-results-website-1.xlsx',
                                sheet='Match to Choice-District')
    school_info_df = read_data('../Data-Analysis/raw-data/DATA4_fall-2025---hs-directory-data.xlsx',
                            sheet='Data')
    addtl_school_info_df = read_data('../Data-Analysis/raw-data/DATA2_fall-2024-admissions_part-ii_suppressed.xlsx',
                            sheet='School')
    df, match_stats_df, school_info_df = preprocess_data(df, match_stats_df, school_info_df, addtl_school_info_df)

    log_and_print(f"df unique schools: {df['School DBN'].nunique()}", outfile)
    log_and_print(f"school_info_df rows: {len(school_info_df)}", outfile)
    log_and_print(f"school_info_df unique schools: {school_info_df['School DBN'].nunique()}", outfile)

    params, lottery, log_likelihoods, final_agg = EM_algorithm(
        df, match_stats_df, school_info_df,
        max_iter=args.max_iter,
        M_simulations=args.M,
        K=args.K,
        outfile=outfile
    )
    log_and_print(f"===== FINAL RESULTS =====", outfile)
    log_and_print(params, outfile=outfile)
    log_and_print(final_agg, outfile=outfile)
    log_and_print(log_likelihoods, outfile=outfile)



if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--synthetic', action='store_true', help='Run synthetic experiments')
    parser.add_argument('--K', type=int, default=12, help='Number of mixture components for real data')
    parser.add_argument('--M', type=int, default=5, help='Number of simulations per evaluation')
    parser.add_argument('--max_iter', type=int, default=5, help='Maximum EM iterations')
    
    args = parser.parse_args()
    
    if args.synthetic:
        # Run synthetic experiments
        #run_synthetic_experiment_3_MoM_no_utilization()
        run_synthetic_experiment_3_MoM_yes_utilization()
    
    else:
        # Run on real data
        run_real(outfile=f"{EXP_OUT_FOLDER}run_main_1.txt")