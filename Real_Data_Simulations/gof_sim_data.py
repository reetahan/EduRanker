import pandas as pd
import numpy as np
from scipy.stats import chisquare

GLOBAL_SEED = 42


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
            print(f"    Simulation {sim_num+1}/{m}...", end='\r')
        
        agg = run_single_simulation(params_all_districts, df, match_stats_df, 
                                    school_info_df, lottery_global)
        
        # Extract per-district stats
        district_to_idx = {d: i for i, d in enumerate(districts)}
        
        for district in districts:
            d_idx = district_to_idx[district]
            sim_match_stats[district].append(agg['match_stats'][d_idx, :])
            sim_total_apps[district].append(agg['total_app'][d_idx, :])  
            sim_true_apps[district].append(agg['true_app'][d_idx, :])
        
        sim_utilization.append(agg['filled'])
    
    print(f"\n  Computing p-values...")
    
    # Get observed values
    p_values = {}
    
    for district in districts:
        print(f"\n  District {district}:")
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
        obs_match_vec = np.clip(obs_match_vec, 0.1, None)

        # Normalize both to sum to 100 (handle rounding errors)
        obs_sum = obs_match_vec.sum()
        sim_sum = mean_sim.sum()
        obs_match_vec = obs_match_vec / obs_sum * sim_sum  # Scale obs to match sim's sum
        
        _, p_val = chisquare(obs_match_vec, mean_sim)
        
        
        p_values[f'{district}_match_distribution'] = p_val
        print(f"    Match stats p-value: {p_val:.4f}")  
        if p_val < 0.05:  
            print(f"      FAILED: obs={obs_match_vec}, sim={mean_sim}")
        
        # Apps (scaled to simulation size)
        df_district = df[df['Residential District'] == district]
        
        obs_total_vec = df_district['Total Applicants by Residential District'].values
        obs_true_vec = df_district['True Applicants by Residential District'].values
        
        sim_total_array = np.array(sim_total_apps[district])  
        sim_true_array = np.array(sim_true_apps[district])   

        all_schools = df['School DBN'].unique()
        schools_in_district = df_district['School DBN'].values
        school_indices = [np.where(all_schools == s)[0][0] for s in schools_in_district]
        
        sim_total_array = sim_total_array[:, school_indices] 
        sim_true_array = sim_true_array[:, school_indices] 
        
        mean_sim_total = np.mean(sim_total_array, axis=0)
        mean_sim_true = np.mean(sim_true_array, axis=0)
        
        mean_sim_total = np.clip(mean_sim_total, 0.1, None)
        mean_sim_true = np.clip(mean_sim_true, 0.1, None)
        obs_total_vec = np.clip(obs_total_vec, 0.1, None)
        obs_true_vec = np.clip(obs_true_vec, 0.1, None)
        
        obs_total_vec = obs_total_vec / obs_total_vec.sum() * mean_sim_total.sum()
        obs_true_vec = obs_true_vec / obs_true_vec.sum() * mean_sim_true.sum()
        
        
        _, p_val_total = chisquare(obs_total_vec, mean_sim_total)
        _, p_val_true = chisquare(obs_true_vec, mean_sim_true)
        
        p_values[f'{district}_total_apps'] = p_val_total
        p_values[f'{district}_true_apps'] = p_val_true
        print(f"    Total apps p-value: {p_val_total:.4f}")  
        print(f"    True apps p-value: {p_val_true:.4f}")  
        if p_val_total < 0.05:  
            print(f"      FAILED total: top 5 obs={obs_total_vec[:5]}, sim={mean_sim_total[:5]}")  
        if p_val_true < 0.05:  
            print(f"      FAILED true: top 5 obs={obs_true_vec[:5]}, sim={mean_sim_true[:5]}")  
    
    
    obs_filled = df.groupby('School DBN')['Total True Applicants School'].first().values
    sim_util_array = np.array(sim_utilization)  # m × s
    
    mean_sim = np.mean(sim_util_array, axis=0)
    
    # Clip first
    mean_sim = np.clip(mean_sim, 0.1, None)
    obs_filled = np.clip(obs_filled, 0.1, None)
    
    # Normalize to match sums
    obs_filled = obs_filled / obs_filled.sum() * mean_sim.sum()
    
    _, p_val = chisquare(obs_filled, mean_sim)
    p_values['utilization'] = p_val
    print(f"\n  Utilization p-value: {p_val:.4f}")  
    if p_val < 0.05:  
        print(f"    FAILED: top 10 obs={obs_filled[:10]}, sim={mean_sim[:10]}")
    
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
            ranking = initial_ranking.copy() 
            top_20 = ranking[:20]
            np.random.shuffle(top_20)
            ranking[:20] = top_20
            central_rankings.append(ranking)
        
        # Sample phis from Beta(2, 8) - concentrates around 0.2
        phis = np.random.beta(4, 6, K)
        phis = np.clip(phis, 0.1, 0.8)
        
        # Sample mixture weights from Dirichlet
        mixture_weights = np.random.dirichlet([1]*K)
        print(f"  District {district}: sampled phis = {phis}")
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

def find_valid_parameters(df, match_stats_df, school_info_df, 
                         n_attempts=100, m=50, K=2, seed=GLOBAL_SEED):
    """
    Search for parameters where all p-values > 0.05
    """
    
    np.random.seed(seed)
    
    districts = sorted(df['Residential District'].unique())
    
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
        print(f"  Sampled parameters, running GoF test")
        
        # Test goodness of fit
        p_values = goodness_of_fit_test(params, df, match_stats_df, school_info_df, 
                                       lottery_global, m)
        
        min_p_value = min(p_values.values())
        print(f"  Min p-value: {min_p_value:.4f}")
        print(f"  Failed constraints: {sum(1 for p in p_values.values() if p < 0.05)}/{len(p_values)}")
        print(f"  Worst 3 constraints:")
        sorted_pvals = sorted(p_values.items(), key=lambda x: x[1]) 
        for name, pval in sorted_pvals[:3]: 
            print(f"    {name}: {pval:.4f}") 

        if min_p_value > best_min_p_value:
            best_min_p_value = min_p_value
            best_params = params
            print(f"New best min p-value found: {best_min_p_value:.4f}")
        
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
    n_attempts=1,
    m=1,
    K=1,
    seed=GLOBAL_SEED
)