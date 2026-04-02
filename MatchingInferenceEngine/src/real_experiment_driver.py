import argparse
import os
from datetime import datetime
from em import EM_algorithm
from data_ingestion import read_data, preprocess_data
from analysis import log_and_print
from config import EXP_OUT_FOLDER, RAW_DATA_DIR, POLISHED_DATA_DIR

def run_real(outfile, df_filepath=None, max_iter=5, M=5, K=12, sampling_n_jobs=32, max_iter_opt=5, seed=40):
    if df_filepath is None:
        df_filepath = f"{POLISHED_DATA_DIR}/master_data_03_residential_district.xlsx"
    df = read_data(df_filepath)
    match_stats_df = read_data(f"{RAW_DATA_DIR}/DATA3_fall-2024-high-school-offer-results-website-1.xlsx",
                                sheet='Match to Choice-District')
    school_info_df = read_data(f"{RAW_DATA_DIR}/DATA4_fall-2025---hs-directory-data.xlsx",
                            sheet='Data')
    addtl_school_info_df = read_data(f"{RAW_DATA_DIR}/DATA2_fall-2024-admissions_part-ii_suppressed.xlsx",
                            sheet='School')
    df, match_stats_df, school_info_df = preprocess_data(df, match_stats_df, school_info_df, addtl_school_info_df)

    log_and_print(f"df unique schools: {df['School DBN'].nunique()}", outfile)
    log_and_print(f"school_info_df rows: {len(school_info_df)}", outfile)
    log_and_print(f"school_info_df unique schools: {school_info_df['School DBN'].nunique()}", outfile)

    params, lottery, log_likelihoods, final_agg = EM_algorithm(
        df, match_stats_df, school_info_df,
        max_iter=max_iter,
        M_simulations=M,
        K=K,
        outfile=outfile,
        sampling_n_jobs=sampling_n_jobs,
        max_iter_opt=max_iter_opt,
        seed=seed
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
    args = parser.parse_args()
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    df_filename = os.path.splitext(os.path.basename(args.df_filepath))[0] if args.df_filepath else "default"
    outfile = f'{EXP_OUT_FOLDER}real_experiment_K={args.K}_M={args.M}_iter={args.max_iter}_opt={args.max_iter_opt}_{df_filename}_{timestamp}.txt'
    run_real(outfile=outfile, df_filepath=args.df_filepath, max_iter=args.max_iter, 
             M=args.M, K=args.K, sampling_n_jobs=args.n_jobs, max_iter_opt=args.max_iter_opt, seed=args.seed)