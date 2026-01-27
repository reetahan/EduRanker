import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import differential_evolution
from scipy.special import softmax
from scipy.stats import chisquare

GLOBAL_SEED = 42

def compute_p_value(observed, simulated_distribution):
    if len(simulated_distribution) == 0:
        return 0.0
    
    # Two-tailed test
    mean_sim = np.mean(simulated_distribution)
    distance_obs = abs(observed - mean_sim)
    distances_sim = np.abs(simulated_distribution - mean_sim)
    
    p_value = np.mean(distances_sim >= distance_obs)
    
    return p_value

def goodness_of_fit_test(params_all_districts, df, match_stats_df, school_info_df, 
                        lottery_global, m=50):
    """
    Test if parameters are consistent with observed data
    Returns: dict with p-values
    """
    
    print(f"  Running {m} simulations...")
    
    districts = list(params_all_districts.keys())
    
    # Collect simulation results
    sim_match_stats = {d: [] for d in districts}
    sim_total_apps = {d: [] for d in districts}
    sim_true_apps = {d: [] for d in districts}
    sim_utilization = []
    
    for sim_num in range(m):
        if sim_num % 10 == 0:
            print(f"    Simulation {sim_num}/{m}...", end='\r')
        
        agg = run_single_simulation(params_all_districts, df, match_stats_df, 
                                    school_info_df, lottery_global)
        
        # Extract per-district stats
        district_to_idx = {d: i for i, d in enumerate(districts)}
        
        for district in districts:
            d_idx = district_to_idx[district]
            sim_match_stats[district].append(agg['match_stats'][d_idx, :])
            sim_total_apps[district].append(np.sum(agg['total_app'][d_idx, :]))
            sim_true_apps[district].append(np.sum(agg['true_app'][d_idx, :]))
        
        sim_utilization.append(agg['filled'])
    
    print(f"\n  Computing p-values...")
    
    # Get observed values
    p_values = {}
    
    for district in districts:
        
        obs_match = match_stats_df[match_stats_df['Residential District'] == district].iloc[0]
        obs_match_vec = np.array([
            obs_match['% Matches to Choice 1-3'],
            obs_match['% Matches to Choice 1-5'],
            obs_match['% Matches to Choice 1-10'],
            obs_match['Unmatched']
        ])
        
        sim_match_dist = np.array(sim_match_stats[district])
        
        # Use chi-squared test on the distribution
        mean_sim = np.mean(sim_match_dist, axis=0)
        
        # Avoid division by zero
        mean_sim = np.clip(mean_sim, 0.1, None)
        
        _, p_val = chisquare(obs_match_vec, mean_sim)
        
        p_values[f'{district}_match_distribution'] = p_val
        
        # Apps (scaled to simulation size)
        df_district = df[df['Residential District'] == district]
        obs_total = df_district['Total Applicants by Residential District'].sum()
        obs_true = df_district['True Applicants by Residential District'].sum()
        
        n_students_obs = int(obs_match['Total Applicants'])
        n_students_sim = n_students_obs  # Same size
        
        scale_factor = n_students_sim / n_students_obs
        obs_total_scaled = obs_total * scale_factor
        obs_true_scaled = obs_true * scale_factor
        
        p_val_total = compute_p_value(obs_total_scaled, np.array(sim_total_apps[district]))
        p_val_true = compute_p_value(obs_true_scaled, np.array(sim_true_apps[district]))
        
        p_values[f'{district}_total_apps'] = p_val_total
        p_values[f'{district}_true_apps'] = p_val_true
    
    # School utilization (global)
    obs_filled = df.groupby('School DBN')['Total True Applicants School'].first().values
    sim_util_array = np.array(sim_utilization)
    
    # Average p-value across schools
    util_p_values = []
    for school_idx in range(len(obs_filled)):
        if obs_filled[school_idx] > 0:
            p_val = compute_p_value(obs_filled[school_idx], sim_util_array[:, school_idx])
            util_p_values.append(p_val)
    
    p_values['utilization_avg'] = np.mean(util_p_values) if util_p_values else 1.0
    
    return p_values

def read_data(file_path, sheet=0):
    """
    Reads Excel data from the given file path and returns a pandas DataFrame.
    """
    data = pd.read_excel(file_path, sheet_name=sheet)
    return data


def preprocess_data(df, match_stats_df, school_info_df):

    df = df[['School DBN', 'School Name', 'School District', 'Residential District', 
         'Total Applicants by Residential District', 'True Applicants by Residential District',
         'Total Applicants School', 'Total True Applicants School', 'Ratio', 'Rank']]
    dtype_mapping = {}
    for i in range(len(df.columns.array)):
        if(i > 3):
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
        match = matches[student_id]
        
        for school in ranking:
            school_idx = school_to_idx[school]
            total_app[district_idx, school_idx] += 1
        
        if match != '-1':
            match_school_idx = school_to_idx[match]
            match_position = np.where(ranking == match)[0][0]
            
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

def sample_random_parameters(districts, df, K=2):
    """
    Sample random Mallows mixture parameters for all districts
    Returns: dict with params per district
    """
    params = {}
    
    for district in districts:
        df_district = df[df['Residential District'] == district]
        schools_list = df_district['School DBN'].values
        n_schools = len(schools_list)
        
        # Initialize central rankings based on observed popularity
        obs_total = df_district.set_index('School DBN')['Total Applicants by Residential District'].to_dict()
        initial_ranking = sorted(schools_list, key=lambda s: -obs_total.get(s, 0))
        
        central_rankings = []
        for k in range(K):
            # Add small random permutation
            perm_indices = np.arange(n_schools)
            np.random.shuffle(perm_indices[:20])  # Shuffle only top 20
            ranking = [initial_ranking[i] for i in perm_indices]
            central_rankings.append(ranking)
        
        # Sample phis from Beta(2, 8) - concentrates around 0.2
        phis = np.random.beta(2, 8, K)
        phis = np.clip(phis, 0.05, 0.5)
        
        # Sample mixture weights from Dirichlet
        mixture_weights = np.random.dirichlet([1]*K)
        
        params[district] = {
            'schools': schools_list,
            'central_rankings': central_rankings,
            'phis': phis,
            'mixture_weights': mixture_weights
        }
    
    return params


def run_single_simulation(params_all_districts, df, match_stats_df, school_info_df, 
                         lottery_global, k_ranking_length=8):
    """
    Run one simulation with given parameters for all districts
    Returns: aggregated statistics
    """
    
    all_rankings = []
    all_district_assignments = []
    all_lottery = []
    
    districts = list(params_all_districts.keys())
    
    for district in districts:
        params = params_all_districts[district]
        
        # Get number of students in this district
        n_students = int(match_stats_df[match_stats_df['Residential District'] == district]['Total Applicants'].iloc[0])
        
        # Sample rankings for this district
        schools_list = params['schools']
        school_to_global_idx = {s: i for i, s in enumerate(schools_list)}
        
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
    
    # Run Gale-Shapley
    matches_idx = gale_shapley(rankings_as_indices, lottery_global, capacities)
    matches_schools = np.array([all_schools[m] if m >= 0 else '-1' for m in matches_idx])
    
    # Compute aggregates
    agg = compute_aggregates(all_rankings, matches_schools, 
                            np.array(all_district_assignments), all_schools)
    
    return agg

def find_valid_parameters(df, match_stats_df, school_info_df, 
                         n_attempts=100, m=50, K=2, seed=GLOBAL_SEED):
    """
    Search for parameters where all p-values > 0.05
    """
    
    np.random.seed(seed)
    
    districts = df['Residential District'].unique()
    
    n_total_students = int(match_stats_df['Total Applicants'].sum())
    lottery_global = np.random.permutation(n_total_students)
    
    print(f"Searching for valid Mallows parameters...")
    print(f"  Districts: {len(districts)}")
    print(f"  Total students: {n_total_students}")
    print(f"  Components per district: K={K}")
    print(f"  Simulations per test: m={m}")
    print(f"  Maximum attempts: {n_attempts}\n")
    
    best_params = None
    best_min_p_value = 0.0
    
    for attempt in range(n_attempts):
        print(f"\nAttempt {attempt + 1}/{n_attempts}")
        
        # Sample random parameters
        params = sample_random_parameters(districts, df, K)
        
        # Test goodness of fit
        p_values = goodness_of_fit_test(params, df, match_stats_df, school_info_df, 
                                       lottery_global, m)
        
        min_p_value = min(p_values.values())
        print(f"  Min p-value: {min_p_value:.4f}")
        print(f"  Failed constraints: {sum(1 for p in p_values.values() if p < 0.05)}/{len(p_values)}")
        
        if min_p_value > best_min_p_value:
            best_min_p_value = min_p_value
            best_params = params
            print(f"  ✓ New best!")
        
        # Check if all p-values acceptable
        if all(p > 0.05 for p in p_values.values()):
            print(f"All {len(p_values)} constraints satisfied!")
            print(f"\nP-value summary:")
            for key, val in sorted(p_values.items()):
                print(f"  {key}: {val:.4f}")
            return params, p_values, lottery_global
    
    print(f"\n No fully valid parameters found in {n_attempts} attempts")
    print(f"Best min p-value achieved: {best_min_p_value:.4f}")
    
    return best_params, None, lottery_global

df = read_data('data/master_data_03_residential_district.xlsx')
match_stats_df = read_data('../Data-Analysis/raw-data/DATA3_fall-2024-high-school-offer-results-website-1.xlsx',
                           sheet='Match to Choice-District')

school_info_df = read_data('../Data-Analysis/raw-data/DATA4_fall-2025---hs-directory-data.xlsx',
                        sheet='Data')

df, match_stats_df, school_info_df = preprocess_data(df, match_stats_df, school_info_df)

params, p_values, lottery = find_valid_parameters(
    df, match_stats_df, school_info_df,
    n_attempts=20,
    m=30,
    K=2,
    seed=GLOBAL_SEED
)