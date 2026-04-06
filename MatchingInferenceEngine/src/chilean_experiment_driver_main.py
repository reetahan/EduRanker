import argparse
import os
import numpy as np
import pandas as pd
from datetime import datetime
from em import EM_algorithm, run_single_simulation
from data_ingestion import read_data, preprocess_chilean_data
from analysis import log_and_print
from config import EXP_OUT_FOLDER, CHILEAN_DATA_DIR

def run_chilean_data_experiment(outfile, max_iter=5, M=5, K=12, sampling_n_jobs=32, max_iter_opt=5, seed=40):
    indv_df = read_data(f"{CHILEAN_DATA_DIR}/individual_level_preferences_and_result.xlsx")
    match_df = read_data(f"{CHILEAN_DATA_DIR}/matching_outcome_by_region.xlsx")
    school_cap_df = read_data(f"{CHILEAN_DATA_DIR}/school_capacity.xlsx")
    school_cap_reg_df = read_data(f"{CHILEAN_DATA_DIR}/school_capacity_by_region.xlsx")

    df, match_stats_df, school_info_df = preprocess_chilean_data(
        indv_df, match_df, school_cap_reg_df, school_cap_df
    )

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
        per_school_lottery=True
    )

    np.random.seed(seed)
    agg, syn_rankings, syn_districts = run_single_simulation(
        params, df, match_stats_df, school_info_df,
        per_school_lottery=True, sampling_n_jobs=1,
        return_rankings=True
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
    args = parser.parse_args()
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    outfile = f'{EXP_OUT_FOLDER}chilean_experiment_K={args.K}_M={args.M}_iter={args.max_iter}_opt={args.max_iter_opt}_seed={args.seed}_{timestamp}.txt'
    run_chilean_data_experiment(outfile=outfile, max_iter=args.max_iter, 
                M=args.M, K=args.K, sampling_n_jobs=args.n_jobs, 
                max_iter_opt=args.max_iter_opt, seed=args.seed)