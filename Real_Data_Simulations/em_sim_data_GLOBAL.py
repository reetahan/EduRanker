import pandas as pd
import numpy as np
from scipy.stats import multivariate_normal
from scipy.optimize import minimize_scalar
import copy
import sys

GLOBAL_SEED = 42


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
    print(f"Running Gale-Shapley algorithm...")
    n_students = len(student_rankings)
    n_schools = len(school_capacities)
    
    student_order = np.argsort(student_lottery_numbers)
    
    matches = np.full(n_students, -1)
    school_tentative = [[] for _ in range(n_schools)]
    
    for student in student_order:
        if(student % 1000 == 0):
            print(f"  Processing student {student}...")
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
                         lottery_global, k_ranking_length=12):
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
        print(f"    Simulating district: {district}")
        
        n_students = int(match_stats_df[
            match_stats_df['Residential District'] == district
        ]['Total Applicants'].iloc[0])
        
        # Get district-specific info
        sigma_d = params['districts'][district]['central_ranking']
        schools_list = params['districts'][district]['schools']
        school_to_idx = {s: i for i, s in enumerate(schools_list)}
        
        print(f" Generating rankings for {n_students} students of length {k_ranking_length} amongst {len(schools_list)} schools")
        
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
    print(f"  Total schools: {len(all_schools)}, Total capacity: {capacities.sum()}, Total students: {len(all_rankings)}")

    matches_idx = gale_shapley(rankings_as_indices, lottery_global, capacities)
    matches_schools = np.array([all_schools[m] if m >= 0 else '-1' for m in matches_idx])

    num_matched = np.sum(matches_idx >= 0)
    num_unmatched = np.sum(matches_idx == -1)
    print(f"    Matched: {num_matched}/{len(matches_idx)}, Unmatched: {num_unmatched}")

    if num_matched > 0:
        match_positions = []
        for i, ranking in enumerate(rankings_as_indices):
            if matches_idx[i] >= 0:
                match_pos = np.where(ranking == matches_idx[i])[0]
                if len(match_pos) > 0:
                    match_positions.append(match_pos[0])
        if match_positions:
            print(f"    Match position distribution: 1st={sum(p==0 for p in match_positions)}, 2nd={sum(p==1 for p in match_positions)}, 3rd={sum(p==2 for p in match_positions)}")

        print(f"\n>>> MATCH POSITION DEBUG:")
        print(f">>> Position 0 (1st): {sum(p==0 for p in match_positions)}")
        print(f">>> Position 1 (2nd): {sum(p==1 for p in match_positions)}")
        print(f">>> Position 2 (3rd): {sum(p==2 for p in match_positions)}")
        print(f">>> Position 3-4: {sum(3<=p<5 for p in match_positions)}")
        print(f">>> Position 5-9: {sum(5<=p<10 for p in match_positions)}")
        print(f">>> Position 10-11: {sum(10<=p<12 for p in match_positions)}")
        print(f">>> Total matched: {len(match_positions)}")
        print(f">>> Total unmatched: {num_unmatched}\n")

    agg = compute_aggregates(all_rankings, matches_schools, 
                            np.array(all_district_assignments), all_schools)
    
    print(f"\n>>> SIMULATION SANITY CHECK:")
    print(f">>> Total students simulated: {len(all_rankings)}")
    print(f">>> Sample ranking from first student: {all_rankings[0]}")
    print(f">>> Sample ranking from the 42nd student: {all_rankings[41]}")
    print(f">>> Match stats shape: {agg['match_stats'].shape}")
    print(f">>> Match stats raw values:\n{agg['match_stats']}")
    return agg

def continuous_to_permutation(sigma_continuous, schools):
    """
    Convert continuous vector to permutation via argsort
    
    Args:
        sigma_continuous: np.array of continuous values
        schools: list of school IDs
    
    Returns:
        Permutation (list of schools in ranked order)
    """
    indices = np.argsort(sigma_continuous)
    return [schools[i] for i in indices]


def permutation_to_continuous(sigma, schools):
    """
    Convert permutation to continuous representation
    
    Args:
        sigma: Permutation (list of schools in ranked order)
        schools: list of all schools
    
    Returns:
        np.array of continuous values
    """
    sigma_continuous = np.zeros(len(schools))
    for i, school in enumerate(schools):
        position = sigma.index(school)
        sigma_continuous[i] = float(position)
    return sigma_continuous


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


def compute_log_likelihood_gaussian(params_global, observed_agg, district,
                                     df, match_stats_df, school_info_df,
                                     M=20):
    """
    Compute approximate log-likelihood using Gaussian assumption
    
    With improved numerical stability
    """
    
    # Create params dict with all districts, but use params_district for target district
    params_all = params_global.copy()
    
    # Get total number of students across ALL districts
    n_students_total = int(match_stats_df['Total Applicants'].sum())
    
    # Run M simulations
    simulated_samples = []
    
    for sim in range(M):
        if sim % 5 == 0:
            print(f"      Simulation {sim+1}/{M}...", end='\r')
        
        # Create fresh lottery for all students
        lottery_sim = np.random.permutation(n_students_total)
        
        # Simulate ALL districts together
        agg = run_single_simulation(
            params_all, df, match_stats_df, school_info_df, 
            lottery_sim,
            k_ranking_length=12
        )
        
        # Extract aggregates for the TARGET district only
        districts_sorted = sorted(match_stats_df['Residential District'].unique())
        district_idx = districts_sorted.index(district)
        
        # Flatten to vector (just match_stats for now)
        agg_vec = agg['match_stats'][district_idx, :]  # 4 values
        
        simulated_samples.append(agg_vec)
    
    print()
    
    # Convert to array
    X = np.array(simulated_samples)  # M × 4
    
    # Check for valid data
    if len(X) == 0 or np.any(np.isnan(X)) or np.any(np.isinf(X)):
        print(f"      Warning: Invalid simulation data")
        return -1e10
    
    # Estimate mean and covariance
    mu = np.mean(X, axis=0)
    
    # IMPROVED: Use more robust covariance estimation
    if M > 1:
        Sigma = np.cov(X, rowvar=False)
        
        # Handle different dimensionalities
        if Sigma.ndim == 0:  # Scalar
            Sigma = np.array([[Sigma]])
        elif Sigma.ndim == 1:  # 1D
            Sigma = np.diag(Sigma)
        
        # Add substantial regularization for numerical stability
        regularization = 1e-3 * np.eye(len(Sigma))
        Sigma = Sigma + regularization
        
        # Check for singularity
        try:
            np.linalg.cholesky(Sigma)
        except np.linalg.LinAlgError:
            print(f"      Warning: Singular covariance matrix, adding more regularization")
            Sigma = Sigma + 1e-2 * np.eye(len(Sigma))
    else:
        # Not enough samples for covariance
        Sigma = 1e-2 * np.eye(4)
    
    # Observed vector
    obs_vec = observed_agg['match_stats']
    
    # IMPROVED: Use simpler distance-based pseudo-likelihood instead of full multivariate normal
    # This is more numerically stable
    
    # Compute Mahalanobis distance
    try:
        diff = obs_vec - mu
        inv_Sigma = np.linalg.inv(Sigma)
        mahalanobis_sq = diff @ inv_Sigma @ diff
        
        # Log-likelihood (unnormalized, just the exponential term)
        # We ignore the normalizing constant since it doesn't depend on parameters
        log_lik = -0.5 * mahalanobis_sq
        
        # Sanity check
        if np.isnan(log_lik) or np.isinf(log_lik):
            print(f"      Warning: Invalid log-likelihood")
            log_lik = -1e10
            
    except Exception as e:
        print(f"      Warning: Likelihood computation failed: {e}")
        
        # Fall back to simple MSE-based pseudo-likelihood
        mse = np.mean((obs_vec - mu)**2)
        log_lik = -mse * 100  # Scale to reasonable range
    
    # Print diagnostics
    print(f"      mu={mu}, obs={obs_vec}")
    print(f"      diff={obs_vec - mu}")
    
    return log_lik

def EM_algorithm(df, match_stats_df, school_info_df,
                 max_iter=10, tol=0.01, K=1, M_simulations=20, seed=GLOBAL_SEED):
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
            school_info_df, M=M_simulations
        )
        
        # Compute total log-likelihood
        print("\n  Computing final log-likelihood at optimized parameters...")
        total_log_lik = compute_log_likelihood_gaussian_all_districts(
            params, observed_agg, df, match_stats_df, 
            school_info_df, M=M_simulations
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
        
        central_ranking = sorted(schools_list, key=lambda s: -obs_total.get(s, 0))
        
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
    Sample students from global mixture with district-specific σ
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
                                                   M=1):
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
        if sim % 5 == 0:
            print(f"      Simulation {sim+1}/{M}...", end='\r')
        
        # Create fresh lottery
        lottery_sim = np.random.permutation(n_students_total)
        
        # Simulate ALL districts together (do this ONCE per M iteration)
        agg = run_single_simulation(
            params_global, df, match_stats_df, school_info_df, 
            lottery_sim, k_ranking_length=12
        )
        
        # Extract stats for EACH district from this single simulation
        for d_idx, district in enumerate(districts):
            agg_vec = agg['match_stats'][d_idx, :]
            simulated_samples[district].append(agg_vec)
    
    print()  # New line after progress indicator
    
    print("\n" + "="*60)
    print("FIT DIAGNOSTICS")
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
                            school_info_df, M=20):
    
    K = len(params['global_phis'])
    
    print("\n  Optimizing global mixture parameters...")
    
    new_phis = []
    
    for k in range(K):
        print(f"\n    Optimizing global φ_{k}...")
        
        phi_k_current = params['global_phis'][k]
        
        def objective_global_phi_k(phi):
            """Negative log-likelihood for global φ_k"""
            
            params_test = copy.deepcopy(params)
            params_test['global_phis'][k] = phi
            
            # Compute log-likelihood for ALL districts at once
            total_log_lik = compute_log_likelihood_gaussian_all_districts(
                params_test, observed_agg, df, match_stats_df, 
                school_info_df, M=M
            )
            
            print(f"      phi_{k}={phi:.4f}, log_lik={total_log_lik:.2f}")
            
            return -total_log_lik
        
        result = minimize_scalar(
            objective_global_phi_k,
            bounds=(0.05, 0.95),
            method='bounded',
            options={'xatol': 0.05}
        )
        
        phi_k_new = result.x
        new_phis.append(phi_k_new)
        
        print(f"      Optimized: φ_{k} {phi_k_current:.4f} → {phi_k_new:.4f}")
    
    params['global_phis'] = np.array(new_phis)
    
    return params


if __name__ == "__main__":
    
    
    df = read_data('data/master_data_03_residential_district.xlsx')
    match_stats_df = read_data('../Data-Analysis/raw-data/DATA3_fall-2024-high-school-offer-results-website-1.xlsx',
                                sheet='Match to Choice-District')
    school_info_df = read_data('../Data-Analysis/raw-data/DATA4_fall-2025---hs-directory-data.xlsx',
                            sheet='Data')
    df, match_stats_df, school_info_df = preprocess_data(df, match_stats_df, school_info_df)
    
    # Run EM
    params, lottery, log_likelihoods = EM_algorithm(
        df, match_stats_df, school_info_df,
        max_iter=5,
        M_simulations=1,
        K=12
    )