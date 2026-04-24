import argparse
import os
import numpy as np
import pandas as pd
from datetime import datetime
from em import EM_algorithm, run_single_simulation
from data_ingestion import read_data, preprocess_chilean_data
from util import log_and_print
from file_config import EXP_OUT_FOLDER, CHILEAN_DATA_DIR
from welfare import evaluate_simulation_output
import json

def run_chilean_data_experiment(
    outfile,
    max_iter=5,
    M=5,
    K=12,
    sampling_n_jobs=32,
    max_iter_opt=5,
    seed=40,
    profile_timing=False,
):
    indv_df = read_data(f"{CHILEAN_DATA_DIR}/individual_level_preferences_and_result.xlsx")
    match_df = read_data(f"{CHILEAN_DATA_DIR}/matching_outcome_by_region.xlsx")
    school_cap_df = read_data(f"{CHILEAN_DATA_DIR}/school_capacity.xlsx")
    school_cap_reg_df = read_data(f"{CHILEAN_DATA_DIR}/school_capacity_by_region.xlsx")

    chile_config_path = f"{CHILEAN_DATA_DIR}/chile_priority_config.json"
    priority_config = None
    if os.path.exists(chile_config_path):
        with open(chile_config_path) as f:
            priority_config = json.load(f)
        log_and_print(f"Loaded priority config: {chile_config_path}", outfile)

    df, match_stats_df, school_info_df, district_to_region = preprocess_chilean_data(
        indv_df, match_df, school_cap_reg_df, school_cap_df
    )

    list_lengths = indv_df.groupby('mrun')['preference_number'].max().clip(upper=15)
    counts = list_lengths.value_counts().sort_index()
    empirical_probs = (counts / counts.sum()).to_dict()

    simulation_kwargs = {
        "list_length_mode": "empirical",
        "list_length_empirical_probs": empirical_probs,
        "profile_timing": profile_timing,
        "priority_config": priority_config,
        "district_to_region": district_to_region,
    }

    log_and_print(f"df unique schools: {df['School DBN'].nunique()}", outfile)
    log_and_print(f"df unique districts: {df['Residential District'].nunique()}", outfile)
    log_and_print(f"school_info_df rows: {len(school_info_df)}", outfile)
    log_and_print(f"school_info_df unique schools: {school_info_df['School DBN'].nunique()}", outfile)
    log_and_print(f"Total students: {int(match_stats_df['Total Applicants'].sum())}", outfile)

    params, lottery, log_likelihoods, final_agg = EM_algorithm(
        df, match_stats_df, school_info_df,
        max_iter=max_iter,
        M_simulations=M,
        K=K,
        outfile=outfile,
        sampling_n_jobs=sampling_n_jobs,
        max_iter_opt=max_iter_opt,
        seed=seed,
        per_school_lottery=True,
        simulation_kwargs=simulation_kwargs,
    )

    np.random.seed(seed)
    agg, syn_rankings, syn_rankings_idx, matches_idx, syn_districts, syn_attrs = run_single_simulation(
        params, df, match_stats_df, school_info_df,
        per_school_lottery=False, sampling_n_jobs=1,
        return_rankings=True,
        outfile=outfile,
        **simulation_kwargs,
    )

    rows = []
    for i, (ranking, district) in enumerate(zip(syn_rankings, syn_districts)):
        row = {'student_id': i, 'district': district}
        for j, school in enumerate(ranking[:10]):
            row[f'choice_{j+1}'] = school
        rows.append(row)

    syn_df = pd.DataFrame(rows)
    syn_path = outfile.replace('.txt', '_synthetic_rankings.csv')
    syn_df.to_csv(syn_path, index=False)
    log_and_print(f"Saved synthetic rankings ({len(syn_df)} students) to {syn_path}", log_file=outfile)

    attr_df = pd.DataFrame(syn_attrs) if syn_attrs is not None else pd.DataFrame()
    attr_df['district'] = list(syn_districts)

    welfare_results = evaluate_simulation_output(
        sim_output={
            'rankings_as_indices': syn_rankings_idx,
            'matches_idx': matches_idx,
            'student_attributes': attr_df,
        },
        categories=['district', 'female'],
        output_dir=outfile.replace('.txt', '_welfare'),
    )
    log_and_print(
        f"Welfare: avg rank={welfare_results.rank_stats['avg_rank']:.3f}, "
        f"pct_matched={welfare_results.rank_stats['pct_matched']:.1f}%",
        log_file=outfile,
    )

    log_and_print(f"===== RUN COMPLETE =====", log_file=outfile)
    log_and_print(f"Log-likelihood trajectory: {log_likelihoods}", log_file=outfile)


if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--K', type=int, default=5, help='Number of mixture components')
    parser.add_argument('--M', type=int, default=10, help='Number of simulations per evaluation')
    parser.add_argument('--max_iter', type=int, default=10, help='Maximum EM iterations')
    parser.add_argument('--max_iter_opt', type=int, default=10, help='Maximum Optimizer iterations')
    parser.add_argument('--seed', type=int, default=40, help='Random seed')
    parser.add_argument('--n_jobs', type=int, default=64, help='Number of parallel workers')
    parser.add_argument('--profile_timing', action='store_true', help='Enable detailed timing logs')
    args = parser.parse_args()
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    outfile = f'{EXP_OUT_FOLDER}/chile_res_logs/{timestamp}/chilean_experiment_K={args.K}_M={args.M}_iter={args.max_iter}_opt={args.max_iter_opt}_seed={args.seed}_{timestamp}.txt'
    run_chilean_data_experiment(outfile=outfile, max_iter=args.max_iter, 
                M=args.M, K=args.K, sampling_n_jobs=args.n_jobs, 
                max_iter_opt=args.max_iter_opt, seed=args.seed,
                profile_timing=args.profile_timing)