import argparse
from datetime import datetime
from em import EM_algorithm
from data_ingestion import read_data, preprocess_data
from analysis import log_and_print
from config import EXP_OUT_FOLDER, RAW_DATA_DIR, POLISHED_DATA_DIR

def run_real(outfile, max_iter=5, M=5, K=12):
    df = read_data(f"{POLISHED_DATA_DIR}/master_data_03_residential_district.xlsx")
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
        outfile=outfile
    )
    log_and_print(f"===== FINAL RESULTS =====", outfile)
    log_and_print(params, log_file=outfile)
    log_and_print(final_agg, log_file=outfile)
    log_and_print(log_likelihoods, log_file=outfile)


if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--K', type=int, default=12, help='Number of mixture components for real data')
    parser.add_argument('--M', type=int, default=5, help='Number of simulations per evaluation')
    parser.add_argument('--max_iter', type=int, default=5, help='Maximum EM iterations')
    parser.add_argument('--seed', type=int, default=40, help='Random seed for synthetic experiments')
    parser.add_argument('--final-analysis', action='store_true', help='Run final aggregation and plotting step')
    args = parser.parse_args()
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    outfile = f'{EXP_OUT_FOLDER}real_experiment_K={args.K}_M={args.M}_iter={args.max_iter}_{timestamp}.txt'
    run_real(outfile=outfile, max_iter=args.max_iter , M=args.M, K=args.K)