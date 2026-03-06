import numpy as np
import pandas as pd
from analysis import plot_capacity_and_sigmas
from gale_shapley import gale_shapley, compute_aggregates
from mallows import mallows_insertion_sampling

def extract_realistic_params_from_real_data(df, school_info_df, n_schools=20, n_students=500, if_plot=False):
    """
    Extract realistic central rankings and capacities from Manhattan districts 1-3.
    Returns sigmas dict, capacities array, and schools list.
    """
    # Get top 20 schools by Ratio in District 1
    df_d1 = df[df['Residential District'] == 1]
    top20 = df_d1.sort_values('Ratio', ascending=False).head(n_schools)
    top20_schools = top20['School DBN'].values.tolist()
    
    # Verify all 20 appear in districts 2 and 3
    for d in [2, 3]:
        df_d = df[df['Residential District'] == d]
        missing = set(top20_schools) - set(df_d['School DBN'].values)
        if missing:
            print(f"  Warning: {len(missing)} schools missing from district {d}: {missing}")
            # Fill missing schools at the bottom of that district's ranking
    
    # Build central rankings for each district based on Ratio ordering
    true_sigmas = {}
    for d in [1, 2, 3]:
        df_d = df[(df['Residential District'] == d) & (df['School DBN'].isin(top20_schools))]
        ranked = df_d.sort_values('Ratio', ascending=False)['School DBN'].values.tolist()
        # Append any missing schools at the end
        missing = [s for s in top20_schools if s not in ranked]
        true_sigmas[d] = ranked + missing
    
    # Get real capacities and scale to target student count
    cap_dict = school_info_df.set_index('School DBN')['Capacity'].to_dict()
    raw_caps = np.array([cap_dict.get(s, 0) for s in top20_schools])
    
    # Scale so total capacity ~ n_students * 1.2 (same ratio as 600/500)
    total_target_capacity = int(n_students * 1.2)
    scaled_caps = np.round(raw_caps / raw_caps.sum() * total_target_capacity).astype(int)

    if(if_plot):
        plot_capacity_and_sigmas(top20_schools, scaled_caps, true_sigmas)

    
    print(f"  Extracted {n_schools} schools from real data")
    print(f"  Total scaled capacity: {scaled_caps.sum()} for {n_students} students")
    print(f"  D1 top 3: {true_sigmas[1][:3]}")
    print(f"  D2 top 3: {true_sigmas[2][:3]}")
    print(f"  D3 top 3: {true_sigmas[3][:3]}")
    
    return true_sigmas, scaled_caps, top20_schools

def create_synthetic_experiment(n_students=500, n_schools=20, 
                                             capacity_per_school=30, k_ranking_length=10, 
                                             true_K=1, district_ct =3, seed=42,
                                             external_sigmas=None, external_capacities=None,
                                             external_schools=None):
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
    
    if external_schools is not None:
        schools_list = external_schools
        n_schools = len(schools_list)
    else:
        schools_list = [f"SCHOOL_{i:02d}" for i in range(n_schools)]
    school_to_idx = {s: i for i, s in enumerate(schools_list)}
    
    districts = list(range(1, district_ct + 1))

    if external_sigmas is not None:
        true_sigmas = external_sigmas
    else:
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
    if external_capacities is not None:
        capacities = external_capacities
    else:
        capacities = np.array([capacity_per_school] * n_schools)
    
    matches_idx = gale_shapley(all_rankings, lottery, capacities)
    matches_schools = np.array([schools_list[m] if m >= 0 else '-1' for m in matches_idx])

    utilization_counts = pd.Series(matches_schools).value_counts()
    school_info_df = pd.DataFrame([
        {'School DBN': s, 'Capacity': capacities[i], 
         'Utilization': (utilization_counts.get(s, 0) / capacities[i] * 100) if capacities[i] > 0 else 0} 
        for i, s in enumerate(schools_list)
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
                'Ratio': (total_apps / capacities[s_idx]) if capacities[s_idx] > 0 else 0
            })
    
    df = pd.DataFrame(app_data)
    
    true_params = {
        'true_K': true_K, 'true_phis': true_phis, 
        'true_weights': true_weights, 'true_sigmas': true_sigmas
    }
    
    return df, match_stats_df, school_info_df, true_params