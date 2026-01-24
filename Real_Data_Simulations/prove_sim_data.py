import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import differential_evolution
from scipy.special import softmax

GLOBAL_SEED = 42

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
    
def create_district_objective(district_id, df, match_stats_df, school_info_df, 
                              lottery_fixed, K=2, M=1):
    
    district_name = df['Residential District'].unique()[district_id]
    df_district = df[df['Residential District'] == district_name]
    
    obs_total_app = df_district.set_index('School DBN')['Total Applicants by Residential District'].to_dict()
    obs_true_app = df_district.set_index('School DBN')['True Applicants by Residential District'].to_dict()
    
    match_row = match_stats_df[match_stats_df['Residential District'] == district_name].iloc[0]
    obs_match_stats = np.array([
        match_row['% Matches to Choice 1-3'],
        match_row['% Matches to Choice 1-5'],
        match_row['% Matches to Choice 1-10'],
        match_row['Unmatched']
    ])
    
    n_students_district = int(match_row['Total Applicants'])
    schools_list = df_district['School DBN'].values
    school_to_idx = {s: i for i, s in enumerate(schools_list)}
    
    capacities_dict = school_info_df.set_index('School DBN')['Capacity'].to_dict()
    capacities = np.array([capacities_dict.get(s, 0) for s in schools_list])
    
    student_start = int(match_stats_df.iloc[:district_id]['Total Applicants'].sum())
    student_end = student_start + n_students_district
    lottery_district = lottery_fixed[student_start:student_end]
    
    eval_count = [0]
    found_solution = [False]
    
    def objective(params):
        if found_solution[0]:
            return 0.0
            
        n_schools = len(schools_list)
        eval_count[0] += 1
        if eval_count[0] % 10 == 0:
            print(f"    Evaluation {eval_count[0]}...", end='\r')

        central_rankings = []
        for k in range(K):
            start = k * n_schools
            end = (k + 1) * n_schools
            central_rankings.append(params[start:end].argsort())
        
        phis = params[K * n_schools : K * n_schools + K]
        phis = np.clip(phis, 0.01, 0.99)
        
        mixture_weights = softmax(params[K * n_schools + K : K * n_schools + 2*K])
        
        total_app_samples = []
        true_app_samples = []
        match_stats_samples = []
        
        for _ in range(M):
            rankings = sample_mallows_mixture(central_rankings, phis, mixture_weights, n_students_district)
            
            rankings_as_schools = []
            for r in rankings:
                rankings_as_schools.append(schools_list[r])
            
            matches_idx = gale_shapley(rankings, lottery_district, capacities)
            matches_schools = np.array([schools_list[m] if m >= 0 else '-1' for m in matches_idx])
            
            district_assignments = np.array([district_name] * n_students_district)
            
            agg = compute_aggregates(rankings_as_schools, matches_schools, 
                                    district_assignments, schools_list)
            
            total_app_samples.append(agg['total_app'][0, :])
            true_app_samples.append(agg['true_app'][0, :])
            match_stats_samples.append(agg['match_stats'][0, :])
        
        exp_total = np.mean(total_app_samples, axis=0)
        exp_true = np.mean(true_app_samples, axis=0)
        exp_match = np.mean(match_stats_samples, axis=0)
        
        pct_errors = []
        for s in obs_total_app.keys():
            if obs_total_app[s] > 0:
                pct_errors.append(abs(exp_total[school_to_idx[s]] - obs_total_app[s]) / obs_total_app[s])
        
        for s in obs_true_app.keys():
            if obs_true_app[s] > 0:
                pct_errors.append(abs(exp_true[school_to_idx[s]] - obs_true_app[s]) / obs_true_app[s])
        
        for i in range(4):
            if obs_match_stats[i] > 0:
                pct_errors.append(abs(exp_match[i] - obs_match_stats[i]) / obs_match_stats[i])
        
        max_pct_error = max(pct_errors) if pct_errors else 0
        
        if max_pct_error <= 0.05:
            found_solution[0] = True
            print(f"\n✓ FOUND SOLUTION within 5%!")
            print(f"  Central rankings:")
            for k in range(K):
                print(f"    Component {k+1}: {central_rankings[k][:10]}...")
            print(f"  Phis: {phis}")
            print(f"  Mixture weights: {mixture_weights}")
            print(f"  Max percentage error: {max_pct_error*100:.2f}%")
            return 0.0
        
        error_total = sum(abs(exp_total[school_to_idx[s]] - obs_total_app[s]) 
                         for s in obs_total_app.keys())
        error_true = sum(abs(exp_true[school_to_idx[s]] - obs_true_app[s]) 
                        for s in obs_true_app.keys())
        error_match = np.sum(np.abs(exp_match - obs_match_stats))
        
        total_error = error_total + error_true + error_match
        
        if eval_count[0] % 10 == 0:
            print(f"    Eval {eval_count[0]}: error={total_error:.2f}, max%err={max_pct_error*100:.1f}%")
        
        return total_error
    
    return objective, schools_list, found_solution

def optimize_district(district_id, df, match_stats_df, school_info_df, 
                     lottery_fixed, K=2):
    
    objective, schools_list, found_solution = create_district_objective(
        district_id, df, match_stats_df, school_info_df, lottery_fixed, K, M=1
    )
    
    n_schools = len(schools_list)
    district_name = df['Residential District'].unique()[district_id]
    df_district = df[df['Residential District'] == district_name]
    
    popularity = df_district.set_index('School DBN')['Rank'].to_dict()
    initial_ranking = sorted(schools_list, key=lambda s: popularity.get(s, 999))
    initial_ranking_idx = np.array([i for i, _ in enumerate(initial_ranking)])
    
    initial_params = []
    for k in range(K):
        noise = np.random.randn(n_schools) * 0.3
        initial_params.extend(initial_ranking_idx + noise)
    
    initial_params.extend([0.5] * K)
    initial_params.extend([1.0/K] * K)
    initial_params = np.array(initial_params)
    
    bounds = []
    for k in range(K):
        bounds.extend([(0, n_schools - 1)] * n_schools)
    bounds.extend([(0.01, 0.99)] * K)
    bounds.extend([(0.01, 10.0)] * K)
    
    result = differential_evolution(
        objective,
        bounds=bounds,
        maxiter=1000,
        popsize=1,
        seed=GLOBAL_SEED,
        workers=1,
        atol=0.0,
        tol=0.0,
        disp=True
    )
    
    if not found_solution[0]:
        print(f"  Warning: No solution within 5% found after {result.nfev} evaluations")
    
    best_params = result.x
    
    central_rankings = []
    for k in range(K):
        start = k * n_schools
        end = (k + 1) * n_schools
        ranking_continuous = best_params[start:end]
        ranking_permutation = ranking_continuous.argsort()
        central_rankings.append(ranking_permutation)
    
    phis = np.clip(best_params[K * n_schools : K * n_schools + K], 0.01, 0.99)
    mixture_weights = softmax(best_params[K * n_schools + K : K * n_schools + 2*K])
    
    return {
        'district': district_name,
        'schools': schools_list,
        'central_rankings': central_rankings,
        'phis': phis,
        'mixture_weights': mixture_weights,
        'error': result.fun
    }

def fit_mallows_to_data(df, match_stats_df, school_info_df, K=2, seed=GLOBAL_SEED):
    
    np.random.seed(seed)
    
    n_total_students = int(match_stats_df['Total Applicants'].sum())
    lottery_global = np.random.permutation(n_total_students)
    
    districts = df['Residential District'].unique()
    n_districts = len(districts)
    
    fitted_models = {}
    
    for district_id in range(n_districts):
        district_name = districts[district_id]
        print(f"\nOptimizing district {district_name} ({district_id + 1}/{n_districts})...")
        
        result = optimize_district(
            district_id, df, match_stats_df, school_info_df, 
            lottery_global, K=K
        )
        
        fitted_models[district_name] = result
        print(f"  Error: {result['error']:.2f}")
    
    return fitted_models, lottery_global


df = read_data('data/master_data_03_residential_district.xlsx')
match_stats_df = read_data('../Data-Analysis/raw-data/DATA3_fall-2024-high-school-offer-results-website-1.xlsx',
                           sheet='Match to Choice-District')

school_info_df = read_data('../Data-Analysis/raw-data/DATA4_fall-2025---hs-directory-data.xlsx',
                        sheet='Data')

df, match_stats_df, school_info_df = preprocess_data(df, match_stats_df, school_info_df)

fit_mallows_to_data(df, match_stats_df, school_info_df, K=2, seed=GLOBAL_SEED)