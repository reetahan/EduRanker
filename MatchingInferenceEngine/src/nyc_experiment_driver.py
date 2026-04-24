import argparse
import os
from datetime import datetime
from em import EM_algorithm, run_single_simulation
from welfare import evaluate_simulation_output
import json
import pandas as pd
import numpy as np
import pickle
from data_ingestion import read_data, preprocess_data
from util import log_and_print
from file_config import EXP_OUT_FOLDER, RAW_DATA_DIR, POLISHED_DATA_DIR


def run_real(outfile, df_filepath=None, max_iter=5, M=5, K=12,
             sampling_n_jobs=32, max_iter_opt=5, seed=40, n_welfare_sims=5,
             profile_timing=False):
    if df_filepath is None:
        df_filepath = f"{POLISHED_DATA_DIR}/master_data_03_residential_district.xlsx"

    df = read_data(df_filepath)
    match_stats_df = read_data(
        f"{RAW_DATA_DIR}/DATA3_fall-2024-high-school-offer-results-website-1.xlsx",
        sheet='Match to Choice-District'
    )
    school_info_df = read_data(
        f"{RAW_DATA_DIR}/DATA4_fall-2025---hs-directory-data.xlsx",
        sheet='Data'
    )
    addtl_school_info_df = read_data(
        f"{RAW_DATA_DIR}/DATA2_fall-2024-admissions_part-ii_suppressed.xlsx",
        sheet='School'
    )

    nyc_config_path = f"{POLISHED_DATA_DIR}/nyc_priority_config.json"
    priority_config = None
    if os.path.exists(nyc_config_path):
        with open(nyc_config_path) as f:
            priority_config = json.load(f)
        log_and_print(f"Loaded NYC priority config", outfile)
    

    df, match_stats_df, school_info_df, district_to_borough = preprocess_data(
        df, match_stats_df, school_info_df, addtl_school_info_df
    )

    simulation_kwargs = {
            "list_length_mode": "gaussian",
            "list_length_mean": 7,
            "list_length_std": 2,
            "list_length_min": 1,
            "list_length_max": 15,
            "return_student_data": False,
            "profile_timing": profile_timing,
            "priority_config": priority_config,
            "district_to_borough" : district_to_borough
        }

    log_and_print(f"df unique schools: {df['School DBN'].nunique()}", outfile)
    log_and_print(f"school_info_df rows: {len(school_info_df)}", outfile)
    log_and_print(f"school_info_df unique schools: {school_info_df['School DBN'].nunique()}", outfile)

    run_simulation_kwargs = dict(simulation_kwargs)
    run_simulation_kwargs["profile_timing"] = profile_timing

    params, lottery, log_likelihoods, final_agg = EM_algorithm(
        df,
        match_stats_df,
        school_info_df,
        max_iter=max_iter,
        M_simulations=M,
        K=K,
        outfile=outfile,
        sampling_n_jobs=sampling_n_jobs,
        max_iter_opt=max_iter_opt,
        seed=seed,
        simulation_kwargs=run_simulation_kwargs
    )

    np.random.seed(seed)
    agg, syn_rankings, syn_rankings_idx, matches_idx, syn_districts, syn_attrs = run_single_simulation(
        params, df, match_stats_df, school_info_df,
        per_school_lottery=False, sampling_n_jobs=1,
        return_rankings=True, lottery_global=lottery,
        outfile=outfile,
        **simulation_kwargs,
    )

    params_path = outfile.replace('.txt', '_params.pkl')
    with open(params_path, 'wb') as f:
        pickle.dump(params, f)
    log_and_print(f"Saved params to {params_path}", log_file=outfile)

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
        categories=['district'],
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
    parser.add_argument('--df-filepath', type=str, default=None, help='Filepath to input dataframe (xlsx file)')
    parser.add_argument('--K', type=int, default=5, help='Number of mixture components for real data')
    parser.add_argument('--M', type=int, default=10, help='Number of simulations per evaluation')
    parser.add_argument('--max_iter', type=int, default=10, help='Maximum EM iterations')
    parser.add_argument('--max_iter_opt', type=int, default=10, help='Maximum Optimizer iterations')
    parser.add_argument('--seed', type=int, default=40, help='Random seed for synthetic experiments')
    parser.add_argument('--final-analysis', action='store_true', help='Run final aggregation and plotting step')
    parser.add_argument('--n_jobs', type=int, default=64, help='Number of parallel workers')
    parser.add_argument('--n_welfare_sims', type=int, default=5, help='Number of post-EM welfare simulations')
    parser.add_argument('--profile_timing', action='store_true', help='Enable detailed timing logs')
    args = parser.parse_args()
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    df_filename = os.path.splitext(os.path.basename(args.df_filepath))[0] if args.df_filepath else "default"
    outfile = f'{EXP_OUT_FOLDER}nyc_res_logs/{timestamp}/real_experiment_K={args.K}_M={args.M}_iter={args.max_iter}_opt={args.max_iter_opt}_{df_filename}_{timestamp}.txt'
    run_real(
        outfile=outfile,
        df_filepath=args.df_filepath,
        max_iter=args.max_iter,
        M=args.M,
        K=args.K,
        sampling_n_jobs=args.n_jobs,
        max_iter_opt=args.max_iter_opt,
        seed=args.seed,
        n_welfare_sims=args.n_welfare_sims,
        profile_timing=args.profile_timing,
    )
