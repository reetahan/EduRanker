import pandas as pd
import numpy as np
from scipy.stats import  kendalltau
from scipy.optimize import minimize_scalar
import copy
import sys
import argparse




def read_data(file_path, sheet=0):
    """
    Reads data from the given file path and returns a pandas DataFrame.
    """
    if file_path.endswith('.csv'):
        data = pd.read_csv(file_path)
    else:
        data = pd.read_excel(file_path, sheet_name=sheet)
    return data


def preprocess_data(df, match_stats_df, school_info_df):

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
    school_info_df =  school_info_df[['dbn','Capacity']]
    school_info_df = school_info_df.rename(columns={'dbn': 'School DBN'})

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
    print(f"Average list length from data: {avg_list_length:.2f}")
     
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
                print(f"Warning: Student matched to {match} not in ranking: {ranking}")
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
                         lottery_global, k_ranking_length=12, M_val=1):
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
    
    for district in districts:
        
        n_students = int(match_stats_df[
            match_stats_df['Residential District'] == district
        ]['Total Applicants'].iloc[0])
        
        # Get district-specific info
        sigma_d = params['districts'][district]['central_ranking']
        schools_list = params['districts'][district]['schools']
        school_to_idx = {s: i for i, s in enumerate(schools_list)}
        
        #print(f" Generating rankings for {n_students} students of length {k_ranking_length} amongst {len(schools_list)} schools")
        
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
    #print(f"  Total schools: {len(all_schools)}, Total capacity: {capacities.sum()}, Total students: {len(all_rankings)}")

    matches_idx = gale_shapley(rankings_as_indices, lottery_global, capacities)
    matches_schools = np.array([all_schools[m] if m >= 0 else '-1' for m in matches_idx])

    num_matched = np.sum(matches_idx >= 0)
    num_unmatched = np.sum(matches_idx == -1)
    #print(f"    Matched: {num_matched}/{len(matches_idx)}, Unmatched: {num_unmatched}")

    if num_matched > 0:
        match_positions = []
        for i, ranking in enumerate(rankings_as_indices):
            if matches_idx[i] >= 0:
                match_pos = np.where(ranking == matches_idx[i])[0]
                if len(match_pos) > 0:
                    match_positions.append(match_pos[0])

    agg = compute_aggregates(all_rankings, matches_schools, 
                            np.array(all_district_assignments), all_schools)
  
    return agg


def create_synthetic_experiment(n_students=500, n_schools=20, capacity_per_school=30,
                                k_ranking_length=10, true_K=1, seed=42):
    """
    Create synthetic data with known ground truth parameters
    
    Args:
        n_students: Total students (default 500)
        n_schools: Total schools (default 20)
        capacity_per_school: Seats per school (default 30, gives 600 capacity)
        k_ranking_length: List length (default 10)
        true_K: Number of true mixture components (1, 2, or 3)
        seed: Random seed
    
    Returns:
        df, match_stats_df, school_info_df, true_params
    """
    
    np.random.seed(seed)
    
    
    # Define TRUE parameters (ground truth)
    if true_K == 1:
        true_phis = np.array([0.3])
        true_weights = np.array([1.0])
    elif true_K == 2:
        true_phis = np.array([0.2, 0.6])
        true_weights = np.array([0.6, 0.4])
    elif true_K == 3:
        true_phis = np.array([0.15, 0.4, 0.7])
        true_weights = np.array([0.5, 0.3, 0.2])
    
    # Create TRUE central ranking (by "desirability")
    # School 0 most desirable, School 19 least desirable
    schools_list = [f"SCHOOL_{i:02d}" for i in range(n_schools)]
    true_sigma = schools_list.copy()  # Already in desirability order
    
    print(f"\nGROUND TRUTH:")
    print(f"  True phis: {true_phis}")
    print(f"  True weights: {true_weights}")
    print(f"  True sigma (top 5): {true_sigma[:5]}")
    print(f"  Students: {n_students}")
    print(f"  Schools: {n_schools}")
    print(f"  Capacity per school: {capacity_per_school}")
    print(f"  Total capacity: {n_schools * capacity_per_school}")
    print(f"  List length: {k_ranking_length}\n")
    
    # Generate student rankings from TRUE model
    school_to_idx = {s: i for i, s in enumerate(schools_list)}
    sigma_indices = np.array([school_to_idx[s] for s in true_sigma])
    
    all_rankings = []
    
    for student in range(n_students):
        # Choose type from TRUE mixture
        k = np.random.choice(true_K, p=true_weights)
        
        # Sample from TRUE Mallows(sigma, phi_k)
        ranking = mallows_insertion_sampling(sigma_indices, true_phis[k])
        ranking = ranking[:k_ranking_length]
        
        all_rankings.append(ranking)
    
    # Convert to school names
    rankings_as_schools = [[schools_list[idx] for idx in r] for r in all_rankings]
    
    # Run Gale-Shapley with TRUE rankings
    lottery = np.random.permutation(n_students)
    
    rankings_as_indices = []
    for ranking in all_rankings:
        rankings_as_indices.append(ranking)
    
    capacities = np.array([capacity_per_school] * n_schools)
    
    matches_idx = gale_shapley(rankings_as_indices, lottery, capacities)
    matches_schools = np.array([schools_list[m] if m >= 0 else '-1' for m in matches_idx])
    
    # Compute aggregates (single district)
    district_assignments = np.array([1] * n_students)
    agg = compute_aggregates(rankings_as_schools, matches_schools, 
                            district_assignments, schools_list)
    

    app_data = []
    for school_idx, school in enumerate(schools_list):
        # Count applications
        total_apps = sum(school in ranking for ranking in rankings_as_schools)
        
        # Count "true" applications (school appears at or after match position)
        true_apps = 0
        for i, ranking in enumerate(rankings_as_schools):
            if matches_schools[i] != '-1' and matches_schools[i] in ranking:
                match_pos = ranking.index(matches_schools[i])
                if school in ranking[match_pos:]:
                    true_apps += 1
            elif matches_schools[i] == '-1' and school in ranking:
                true_apps += 1
        
        ratio = (true_apps ** 2) / max(total_apps, 1)
        
        app_data.append({
            'School DBN': school,
            'School Name': f'School {school}',
            'School District': 1,
            'Residential District': 1,
            'Total Applicants by Residential District': total_apps,
            'True Applicants by Residential District': true_apps,
            'Total Applicants School': total_apps,
            'Total True Applicants School': true_apps,
            'Ratio': ratio,
            'Rank': school_idx
        })
    
    df = pd.DataFrame(app_data)
    
    # match_stats_df: Observed match outcomes
    match_stats = agg['match_stats'][0, :]  # Single district
    
    match_stats_data = [{
        'Residential District': 1,
        'Total Applicants': n_students,
        '% Matches to Choice 1-3': match_stats[0],
        '% Matches to Choice 1-5': match_stats[1],
        '% Matches to Choice 1-10': match_stats[2],
        'Unmatched': match_stats[3]
    }]
    
    match_stats_df = pd.DataFrame(match_stats_data)
    
    # school_info_df: Capacities
    school_info_data = []
    for school in schools_list:
        school_info_data.append({
            'School DBN': school,
            'Capacity': capacity_per_school
        })
    
    school_info_df = pd.DataFrame(school_info_data)
    
    # Print observed statistics
    print("OBSERVED STATISTICS (from TRUE model):")
    print(f"  Top-3: {match_stats[0]:.1f}%")
    print(f"  Top-5: {match_stats[1]:.1f}%")
    print(f"  Top-10: {match_stats[2]:.1f}%")
    print(f"  Unmatched: {match_stats[3]:.1f}%")
    print("="*60 + "\n")
    
    true_params = {
        'true_K': true_K,
        'true_phis': true_phis,
        'true_weights': true_weights,
        'true_sigma': true_sigma
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
                 max_iter=10, tol=0.01, K=1, M_simulations=20, seed=42):
    """
    EM algorithm with GLOBAL MIXTURE
    """
    
    np.random.seed(seed)
    
    print("="*60)
    print("EM ALGORITHM - GLOBAL MIXTURE")
    print("="*60)
    
    districts = sorted(df['Residential District'].unique())
    n_total_students = int(match_stats_df['Total Applicants'].sum())
    lottery_global = np.random.permutation(n_total_students)
    
    print(f"\nInitialization:")
    print(f"  Districts: {len(districts)}")
    print(f"  Total students: {n_total_students}")
    print(f"  Global mixture components: K={K}")
    print(f"  Max iterations: {max_iter}")
    print(f"  Simulations per evaluation: M={M_simulations}\n")
    
    # Initialize with GLOBAL mixture
    params = initialize_parameters_global_mixture(districts, df, K)

    print("\n" + "="*60)
    print("PARAMETER STRUCTURE")
    print("="*60)
    print(f"Global phis: {params['global_phis']}")
    print(f"Global weights: {params['global_weights']}")
    print("\nDistrict-specific central rankings:")
    for district in list(params['districts'].keys())[:3]:
        sigma_d = params['districts'][district]['central_ranking']
        schools_d = params['districts'][district]['schools']
        print(f"\nDistrict {district}:")
        print(f"  # Schools: {len(schools_d)}")
        print(f"  Top 5 in ranking: {sigma_d[:5]}")
    print("="*60 + "\n")

    observed_agg = extract_observed_aggregates(df, match_stats_df)
    
    log_likelihoods = []
    
    # EM loop
    for iteration in range(max_iter):
        print(f"\n{'='*60}")
        print(f"EM ITERATION {iteration + 1}/{max_iter}")
        print(f"{'='*60}")
        
        old_params = copy.deepcopy(params)
        
        # M-STEP: Optimize global parameters
        params = optimize_global_mixture(
            params, observed_agg, df, match_stats_df, 
            school_info_df, M=M_simulations, seed=seed,
            iteration=iteration
        )

        # Sort them to remove indexing ambiguity
        sorted_indices = np.argsort(params['global_phis'])
        params['global_phis'] = params['global_phis'][sorted_indices]
        params['global_weights'] = params['global_weights'][sorted_indices]
        
        # Compute total log-likelihood
        print("\n  Computing final log-likelihood at optimized parameters...")
        total_log_lik = compute_log_likelihood_gaussian_all_districts(
            params, observed_agg, df, match_stats_df, 
            school_info_df, M=M_simulations, seed=seed,
            iteration=iteration
        )
        
        log_likelihoods.append(total_log_lik)
        print(f"\nTotal log-likelihood: {total_log_lik:.2f}")
        
        # Check convergence
        max_phi_change = max(
            abs(params['global_phis'][k] - old_params['global_phis'][k])
            for k in range(K)
        )
        
        print(f"Max phi change: {max_phi_change:.4f}")
        
        if iteration > 0:
            delta_log_lik = log_likelihoods[-1] - log_likelihoods[-2]
            print(f"Log-likelihood change: {delta_log_lik:.4f}")
        
        if max_phi_change < tol:
            print(f"\n{'='*60}")
            print("EM CONVERGED!")
            print(f"{'='*60}")
            break
    
    print(f"\nFinal global parameters:")
    print(f"  Global phis: {params['global_phis']}")
    print(f"  Global weights: {params['global_weights']}")
    
    return params, lottery_global, log_likelihoods

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
    
    print(f"Global mixture initialized:")
    print(f"  Global phis: {global_phis}")
    print(f"  Global weights: {global_weights}")
    
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
                                                   M=1, seed=42, iteration=1):
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
    
    for sim in range(M):
        print(f"      Simulation {sim+1}/{M}...", end='\r')
        
        # Create fresh lottery
        
        current_sim_seed = seed * sim
        
        # FIX: The lottery is now fixed for this 'world' across all phi evaluations
        rng_lottery = np.random.default_rng(seed=current_sim_seed)
        lottery_sim = rng_lottery.permutation(n_students_total)
        
        # FIX: The Mallows choices are also fixed for this 'world'
        np.random.seed(current_sim_seed)
        
        # Simulate ALL districts together (do this ONCE per M iteration)
        agg = run_single_simulation(
            params_global, df, match_stats_df, school_info_df, 
            lottery_sim, k_ranking_length=12, M_val=sim
        )
        
        # Extract stats for EACH district from this single simulation
        for d_idx, district in enumerate(districts):
            agg_vec = agg['match_stats'][d_idx, :]
            simulated_samples[district].append(agg_vec)
    
    print()  # New line after progress indicator
    
    print("\n" + "="*60)
    print(f"FIT DIAGNOSTICS | Seed: {seed} | Iteration: {iteration}")
    print("="*60)
    
    for d_idx, district in enumerate(districts):  
        obs = observed_agg[district]['match_stats']
        sim = agg['match_stats'][d_idx, :]
        
        print(f"\nDistrict {district}:")
        print(f"  Observed:  top3={obs[0]:5.1f}%, top5={obs[1]:5.1f}%, top10={obs[2]:5.1f}%, unmatched={obs[3]:5.1f}%")
        print(f"  Simulated: top3={sim[0]:5.1f}%, top5={sim[1]:5.1f}%, top10={sim[2]:5.1f}%, unmatched={sim[3]:5.1f}%")
        print(f"  Difference: top3={obs[0]-sim[0]:+5.1f}, top5={obs[1]-sim[1]:+5.1f}, top10={obs[2]-sim[2]:+5.1f}, unmatched={obs[3]-sim[3]:+5.1f}")
    
    print("="*60 + "\n")
    # Now compute likelihood for each district separately
    total_log_lik = 0
    
    for district in districts:
        X = np.array(simulated_samples[district])  # M × 4 array
        
        # Check for valid data
        if len(X) == 0 or np.any(np.isnan(X)) or np.any(np.isinf(X)):
            print(f"      Warning: Invalid data for district {district}")
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
                print(f"      Warning: Invalid log-likelihood for district {district}")
                log_lik = -1e10
                
        except Exception as e:
            print(f"      Warning: Likelihood computation failed for district {district}: {e}")
            
            # Fall back to simple MSE
            mse = np.mean((obs_vec - mu)**2)
            log_lik = -mse * 100
        
        total_log_lik += log_lik
    
    return total_log_lik

def optimize_global_mixture(params, observed_agg, df, match_stats_df, 
                            school_info_df, M=20, seed=42, iteration=1):
    
    K = len(params['global_phis'])
    
    print("\n  Optimizing global mixture parameters...")
    
    
    for k in range(K):
        print(f"\n    Optimizing global φ_{k}...")
        
        phi_k_initial = params['global_phis'][k]
        
        def objective_global_phi_k(phi):
            """Negative log-likelihood for global φ_k"""
            
            # IMPROVEMENT: Instead of deepcopying the whole dict, 
            # just temporarily swap the one value we are testing.
            original_phi = params['global_phis'][k]
            params['global_phis'][k] = phi
            
            # Compute log-likelihood for ALL districts at once
            total_log_lik = compute_log_likelihood_gaussian_all_districts(
                params, observed_agg, df, match_stats_df, 
                school_info_df, M=M, seed=seed, iteration=iteration
            )
            
            # Swap back so we don't permanently alter params until the result is final
            params['global_phis'][k] = original_phi
            
            print(f"      phi_{k}={phi:.4f}, log_lik={total_log_lik:.2f}")
            
            return -total_log_lik
        
        result = minimize_scalar(
            objective_global_phi_k,
            bounds=(0.01, 0.99),
            method='bounded',
            options={'xatol': 0.01}
        )
        
        phi_k_new = result.x
        
        # FIX: Update the parameter immediately. 
        # The next iteration of the 'for k in range(K)' loop will now use this new value.
        params['global_phis'][k] = phi_k_new
        
        opt_line = f"      Iteration {iteration} | Seed {seed} | Optimized: φ_{k} {phi_k_initial:.4f} → {phi_k_new:.4f}"
        print(opt_line)
        with open("experiment3_results.txt", "a+") as f:
            f.write(opt_line + "\n")
            f.flush()
    
    return params


if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--synthetic', action='store_true', help='Run synthetic experiments')
    parser.add_argument('--K', type=int, default=12, help='Number of mixture components for real data')
    parser.add_argument('--M', type=int, default=1, help='Number of simulations per evaluation')
    parser.add_argument('--max_iter', type=int, default=5, help='Maximum EM iterations')
    
    args = parser.parse_args()
    
    if args.synthetic:
        # Run synthetic experiments
        
        '''
        # Experiment 1: K=1 (single component)
        print("\n" + "="*60)
        print("EXPERIMENT 1: K=1")
        print("="*60)
        
        df1, match_stats_df1, school_info_df1, true_params1 = create_synthetic_experiment(
            n_students=600, n_schools=20, capacity_per_school=30,
            k_ranking_length=10, true_K=1, seed=42
        )
        
        params1, _, log_liks1 = EM_algorithm(
            df1, match_stats_df1, school_info_df1,
            max_iter=10, M_simulations=10, K=1, seed=42
        )
        
        print(f"\nEXPERIMENT 1 RESULTS:")
        print(f"  True phi: {true_params1['true_phis']}")
        print(f"  Estimated phi: {params1['global_phis']}")
        print(f"  Error: {np.abs(params1['global_phis'] - true_params1['true_phis'])}")
        
        
        # Experiment 2: K=2
        print("\n" + "="*60)
        print("EXPERIMENT 2: K=2")
        print("="*60)
        
        df2, match_stats_df2, school_info_df2, true_params2 = create_synthetic_experiment(
            n_students=600, n_schools=20, capacity_per_school=30,
            k_ranking_length=10, true_K=2, seed=43
        )
        
        params2, _, log_liks2 = EM_algorithm(
            df2, match_stats_df2, school_info_df2,
            max_iter=10, M_simulations=10, K=2, seed=43
        )
        
        print(f"\nEXPERIMENT 2 RESULTS:")
        print(f"  True phis: {true_params2['true_phis']}")
        print(f"  Estimated phis: {params2['global_phis']}")
        print(f"  Error: {np.abs(params2['global_phis'] - true_params2['true_phis'])}")
        '''
        # Experiment 3: K=3
        print("\n" + "="*60)
        print("EXPERIMENT 3: K=3")
        print("="*60)
        
        for seed in range(40, 50):
            print(f"\nRunning synthetic experiment with seed {seed}...")
            df3, match_stats_df3, school_info_df3, true_params3 = create_synthetic_experiment(
                n_students=500, n_schools=20, capacity_per_school=30,
                k_ranking_length=10, true_K=3, seed=44
            )

            true_sigma = true_params3['true_sigma']            
            print(f"Ground Truth: {true_sigma}")

            ratio_ranking = df3.sort_values('Ratio', ascending=True)['School DBN'].tolist()

            # 3. Calculate Kendall-Tau Correlation
            # We map the school names to their positions in the True ranking to compare
            true_pos = {school: i for i, school in enumerate(true_sigma)}
            true_ranks = [true_pos[s] for s in true_sigma]
            estimated_ranks = [true_pos[s] for s in ratio_ranking]

            tau, _ = kendalltau(true_ranks, estimated_ranks)
            norm_tau =  (1 - tau) / 2 

            # 4. Print the Comparison
            print("\n" + "="*40)
            print(f"True Popularity versus Derived Central Ranking")
            print("="*40)
            print(f"Kendall-Tau Score in [-1,1]: {tau:.4f}")
            print(f"Kendall-Tau Score Normalized to [0,1] : {norm_tau:.4f}")
            print(f"Ground Truth: {true_sigma}")
            print(f"Ratio Metric: {ratio_ranking}")
            print("-" * 40)
            exit(0)

     
            params3, _, log_liks3 = EM_algorithm(
                df3, match_stats_df3, school_info_df3,
                max_iter=10, M_simulations=10, K=3, seed=seed
            )
            
            out_lines = [
                    f"\nSEED {seed} RESULTS:", # Added this
                    f"  True phis: {true_params3['true_phis']}",
                    f"  Estimated phis: {params3['global_phis']}",
                    f"  Error: {np.abs(params3['global_phis'] - true_params3['true_phis'])}"
                ]
            for line in out_lines:
                print(line)
            with open("experiment3_results.txt", "a+") as f:
                for line in out_lines:
                    f.write(line + "\n")
                    f.flush()
    
    else:
        # Run on real data
        df = read_data('data/master_data_03_residential_district.xlsx')
        match_stats_df = read_data('../Data-Analysis/raw-data/DATA3_fall-2024-high-school-offer-results-website-1.xlsx',
                                    sheet='Match to Choice-District')
        school_info_df = read_data('../Data-Analysis/raw-data/DATA4_fall-2025---hs-directory-data.xlsx',
                                sheet='Data')
        df, match_stats_df, school_info_df = preprocess_data(df, match_stats_df, school_info_df)
        
        params, lottery, log_likelihoods = EM_algorithm(
            df, match_stats_df, school_info_df,
            max_iter=args.max_iter,
            M_simulations=args.M,
            K=args.K
        )