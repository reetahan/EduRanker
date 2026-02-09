import pandas as pd
import numpy as np
from scipy.stats import multivariate_normal
from scipy.optimize import minimize_scalar
import copy

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


def sample_mallows_mixture(central_rankings, phis, mixture_weights, n_samples):
    K = len(central_rankings)
    rankings = []
    
    for _ in range(n_samples):
        component = np.random.choice(K, p=mixture_weights)
        ranking = mallows_insertion_sampling(central_rankings[component], phis[component])
        rankings.append(ranking)
    
    return rankings


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
        district_total = np.sum(match_stats[d, :])
        if district_total > 0:
            match_stats[d, :3] = (match_stats[d, :3] / district_total) * 100
    
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


def run_single_simulation(params_all_districts, df, match_stats_df, school_info_df, 
                         lottery_global, k_ranking_length=8):
    """
    Run one simulation with given parameters for all districts
    Returns: aggregated statistics
    """
    
    all_rankings = []
    all_district_assignments = []
    
    districts = [int(x) for x in list(params_all_districts.keys())]
    
    for district in districts:
        print(f"    Simulating district: {district}")
        params = params_all_districts[district]
        
        # Get number of students in this district
        n_students = int(match_stats_df[match_stats_df['Residential District'] == district]['Total Applicants'].iloc[0])
        
        # Sample rankings for this district
        schools_list = params['schools']
        school_to_global_idx = {s: i for i, s in enumerate(schools_list)}
        
        print(f" Generating rankings for {n_students} students of length {k_ranking_length} amongst {len(schools_list)} schools")
        rankings = sample_mallows_mixture(
            [np.array([school_to_global_idx[s] for s in cr]) for cr in params['central_rankings']],
            params['phis'],
            params['mixture_weights'],
            n_students
        )
        
        # Truncate to k schools
        rankings = [r[:k_ranking_length] for r in rankings]
        
        # Convert to school DBNs
        rankings_as_schools = [[schools_list[idx] for idx in r] for r in rankings]
        
        all_rankings.extend(rankings_as_schools)
        all_district_assignments.extend([district] * n_students)
    
    # Create global lottery
    n_total = len(all_rankings)
    district_offsets = {}
    offset = 0
    for district in districts:
        n_students = int(match_stats_df[match_stats_df['Residential District'] == district]['Total Applicants'].iloc[0])
        district_offsets[district] = offset
        offset += n_students
    
    # All unique schools
    all_schools = df['School DBN'].unique()
    school_to_idx = {s: i for i, s in enumerate(all_schools)}
    
    # Convert rankings to indices
    rankings_as_indices = []
    for ranking in all_rankings:
        rankings_as_indices.append(np.array([school_to_idx[s] for s in ranking]))
    
    # Get capacities
    capacities_dict = school_info_df.set_index('School DBN')['Capacity'].to_dict()
    capacities = np.array([capacities_dict.get(s, 0) for s in all_schools])
    print(f"  Total schools: {len(all_schools)}, Total capacity: {capacities.sum()}, Total students: {len(all_rankings)}")

    # Run Gale-Shapley
    matches_idx = gale_shapley(rankings_as_indices, lottery_global, capacities)
    matches_schools = np.array([all_schools[m] if m >= 0 else '-1' for m in matches_idx])

    num_matched = np.sum(matches_idx >= 0)
    num_unmatched = np.sum(matches_idx == -1)
    print(f"    Matched: {num_matched}/{len(matches_idx)}, Unmatched: {num_unmatched}")

    # Check distribution of matches
    if num_matched > 0:
        match_positions = []
        for i, ranking in enumerate(rankings_as_indices):
            if matches_idx[i] >= 0:
                match_pos = np.where(ranking == matches_idx[i])[0]
                if len(match_pos) > 0:
                    match_positions.append(match_pos[0])
        if match_positions:
            print(f"    Match position distribution: 1st={sum(p==0 for p in match_positions)}, 2nd={sum(p==1 for p in match_positions)}, 3rd={sum(p==2 for p in match_positions)}")
        
    # Compute aggregates
    agg = compute_aggregates(all_rankings, matches_schools, 
                            np.array(all_district_assignments), all_schools)
    
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


def initialize_parameters_for_em(districts, df, K=1):
    """
    Initialize Mallows parameters for EM algorithm
    
    Args:
        districts: list of district IDs
        df: admissions data
        K: number of mixture components
    
    Returns:
        dict with initial parameters per district
    """
    params = {}
    
    for district in districts:
        df_district = df[df['Residential District'] == district]
        schools_list = df_district['School DBN'].values
        n_schools = len(schools_list)
        
        # Initialize central ranking based on observed popularity
        obs_total = df_district.set_index('School DBN')['Total Applicants by Residential District'].to_dict()
        initial_ranking = sorted(schools_list, key=lambda s: -obs_total.get(s, 0))
        
        central_rankings = []
        for k in range(K):
            ranking = initial_ranking.copy()
            # Shuffle top 20 for variation
            top_n = min(20, len(ranking))
            top_schools = ranking[:top_n]
            np.random.shuffle(top_schools)
            ranking[:top_n] = top_schools
            central_rankings.append(ranking)
        
        # Initialize phis
        phis = np.random.beta(2, 8, K)  # Concentrates around 0.2
        phis = np.clip(phis, 0.1, 0.7)
        
        # Mixture weights
        mixture_weights = np.random.dirichlet([1]*K) if K > 1 else np.array([1.0])
        
        print(f"  District {district}: initialized with phis = {phis}")
        
        params[district] = {
            'schools': schools_list,
            'central_rankings': central_rankings,
            'phis': phis,
            'mixture_weights': mixture_weights
        }
    
    return params


def compute_log_likelihood_gaussian(params_district, observed_agg, district,
                                     df, match_stats_df, school_info_df,
                                     all_params, M=20):
    """
    Compute approximate log-likelihood using Gaussian assumption
    
    Key change: Simulate ALL districts together, only varying the target district's params
    
    Args:
        params_district: Parameters being tested for this district
        observed_agg: Observed aggregates for this district
        district: Which district we're optimizing
        all_params: Current parameters for ALL districts (we'll override one)
        M: Number of simulations
    
    Returns:
        log-likelihood (scalar)
    """
    
    # Create params dict with all districts, but use params_district for target district
    params_all = copy.deepcopy(all_params)
    params_all[district] = params_district
    
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
            k_ranking_length=8
        )
        
        # Extract aggregates for the TARGET district only
        districts_sorted = sorted(match_stats_df['Residential District'].unique())
        district_idx = districts_sorted.index(district)
        
        # Flatten to vector (just match_stats for now)
        agg_vec = agg['match_stats'][district_idx, :]  # 4 values
        
        simulated_samples.append(agg_vec)
    
    print()  # New line after progress
    
    # Convert to array
    X = np.array(simulated_samples)  # M × 4
    
    # Estimate mean and covariance
    mu = np.mean(X, axis=0)
    Sigma = np.cov(X, rowvar=False)
    
    # Regularize covariance (avoid singular matrix)
    if Sigma.ndim == 0:  # Scalar case
        Sigma = np.array([[Sigma]]) + 1e-6
    elif Sigma.ndim == 1:  # 1D case
        Sigma = np.diag(Sigma) + 1e-6 * np.eye(len(Sigma))
    else:
        Sigma = Sigma + 1e-6 * np.eye(len(Sigma))
    
    # Observed vector
    obs_vec = observed_agg['match_stats']
    
    # Compute log-likelihood
    try:
        log_lik = multivariate_normal.logpdf(obs_vec, mean=mu, cov=Sigma)
    except Exception as e:
        print(f"      Warning: Likelihood computation failed: {e}")
        log_lik = -1e10
    
    return log_lik

def optimize_district_parameters_simple(district, observed_agg, current_params,
                                        df, match_stats_df, school_info_df,
                                        all_params, M=20):
    """
    M-step: Optimize parameters for one district
    
    Args:
        all_params: Parameters for ALL districts (needed for simulation context)
    
    Returns:
        sigma_new, phi_new
    """
    
    print(f"\n  Optimizing district {district}...")
    
    schools = current_params['schools']
    sigma_current = current_params['central_rankings'][0]
    phi_current = current_params['phis'][0]
    
    def objective_phi(phi):
        """Negative log-likelihood as function of phi only"""
        
        # Build params with this phi
        params_test = {
            'schools': schools,
            'central_rankings': [sigma_current],
            'phis': np.array([phi]),
            'mixture_weights': np.array([1.0])
        }
        
        # Compute log-likelihood (simulating ALL districts)
        log_lik = compute_log_likelihood_gaussian(
            params_test, observed_agg, district,
            df, match_stats_df, school_info_df,
            all_params,  # ← Pass all params
            M=M
        )
        
        print(f"    phi={phi:.4f}, log_lik={log_lik:.2f}")
        
        return -log_lik
    
    # Optimize phi only
    from scipy.optimize import minimize_scalar
    
    result = minimize_scalar(
        objective_phi,
        bounds=(0.05, 0.95),
        method='bounded',
        options={'xatol': 0.05}
    )
    
    phi_new = result.x
    sigma_new = sigma_current
    
    print(f"  Optimized: phi {phi_current:.4f} → {phi_new:.4f}")
    
    return sigma_new, phi_new


def EM_algorithm(df, match_stats_df, school_info_df,
                 max_iter=10, tol=0.01, K=1, M_simulations=20, seed=GLOBAL_SEED):
    """
    EM algorithm to find maximum likelihood Mallows parameters
    """
    
    np.random.seed(seed)
    
    
    # Get districts
    districts = sorted(df['Residential District'].unique())
    
    # Create fixed global lottery
    n_total_students = int(match_stats_df['Total Applicants'].sum())
    lottery_global = np.random.permutation(n_total_students)
    
    print(f"\nInitialization:")
    print(f"  Districts: {len(districts)}")
    print(f"  Total students: {n_total_students}")
    print(f"  Mixture components: K={K}")
    print(f"  Max iterations: {max_iter}")
    print(f"  Simulations per evaluation: M={M_simulations}\n")
    
    # Initialize parameters
    print("Initializing parameters...")
    params = initialize_parameters_for_em(districts, df, K)
    
    # Extract observed aggregates
    observed_agg = extract_observed_aggregates(df, match_stats_df)
    
    # Track log-likelihoods
    log_likelihoods = []
    
    # EM loop
    for iteration in range(max_iter):
        print(f"\n{'='*60}")
        print(f"EM ITERATION {iteration + 1}/{max_iter}")
        print(f"{'='*60}")
        
        old_params = copy.deepcopy(params)
        
        total_log_lik = 0
        
        # M-STEP: Optimize each district
        for district in districts:
            
            # Optimize parameters
            sigma_new, phi_new = optimize_district_parameters_simple(
                district,
                observed_agg[district],
                params[district],
                df, match_stats_df, school_info_df,
                params,  # ← Pass ALL current params
                M=M_simulations
            )
            
            # Update parameters
            params[district]['central_rankings'][0] = sigma_new
            params[district]['phis'][0] = phi_new
            
            # Compute log-likelihood at new parameters
            district_log_lik = compute_log_likelihood_gaussian(
                params[district],
                observed_agg[district],
                district,
                df, match_stats_df, school_info_df,
                params,  # ← Use updated params
                M=M_simulations
            )
            
            total_log_lik += district_log_lik
            
            print(f"  District {district} log-likelihood: {district_log_lik:.2f}")
        
        log_likelihoods.append(total_log_lik)
        print(f"\nTotal log-likelihood: {total_log_lik:.2f}")
        
        # Check convergence
        max_phi_change = max(
            abs(params[d]['phis'][0] - old_params[d]['phis'][0])
            for d in districts
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
    
    print(f"\nFinal parameters:")
    for district in districts:
        print(f"  District {district}: phi = {params[district]['phis'][0]:.4f}")
    
    return params, lottery_global, log_likelihoods


if __name__ == "__main__":
    # Load data
    df = read_data('data/master_data_03_residential_district.xlsx')
    match_stats_df = read_data('../Data-Analysis/raw-data/DATA3_fall-2024-high-school-offer-results-website-1.xlsx',
                               sheet='Match to Choice-District')
    school_info_df = read_data('../Data-Analysis/raw-data/DATA4_fall-2025---hs-directory-data.xlsx',
                            sheet='Data')
    
    # Preprocess
    df, match_stats_df, school_info_df = preprocess_data(df, match_stats_df, school_info_df)
    
    # Run EM algorithm
    params, lottery, log_likelihoods = EM_algorithm(
        df, match_stats_df, school_info_df,
        max_iter=5,       
        tol=0.01,
        K=2,             
        M_simulations=10, 
        seed=GLOBAL_SEED
    )
    
    print(f"\nLog-likelihood progression: {log_likelihoods}")