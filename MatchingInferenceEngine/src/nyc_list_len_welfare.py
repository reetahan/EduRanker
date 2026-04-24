
import argparse
import pickle
import os
import json
import numpy as np
import pandas as pd

from em import run_single_simulation
from data_ingestion import read_data, preprocess_data
from priority_attributes import sample_student_attributes
from welfare import evaluate_simulation_output
from em import sample_rankings, run_matching
from file_config import RAW_DATA_DIR, POLISHED_DATA_DIR


def run_sweep(params, lottery, df, match_stats_df, school_info_df,
              priority_config, district_to_borough,
              min_lengths, output_dir, seed, n_jobs):

    

    os.makedirs(output_dir, exist_ok=True)
    summary_rows = []
    list_length_max = 15

    # Sample preferences and attributes ONCE
    print("Sampling preferences (fixed across all min_len values)...")
    all_rankings, all_district_assignments, rng = sample_rankings(
        params=params,
        match_stats_df=match_stats_df,
        sampling_n_jobs=n_jobs,
        list_length_max=list_length_max
    )
    print(f"  Sampled {len(all_rankings)} student rankings")

    all_schools = df['School DBN'].unique()
    dbn_to_progs = {s: [s] for s in all_schools}
    student_attrs = None
    if priority_config is not None:
        student_attrs = sample_student_attributes(
            district_assignments=all_district_assignments,
            all_schools=all_schools,
            dbn_to_progs=dbn_to_progs,
            priority_config=priority_config,
            district_to_region={str(d): str(d) for d in set(all_district_assignments)},
            rng=rng,
            district_to_borough=district_to_borough,
        )

    for min_len in min_lengths:
        print(f"\n{'='*50}")
        print(f"Running list_length_min={min_len}")
        print(f"{'='*50}")

        agg, syn_rankings, syn_rankings_idx, matches_idx, syn_attrs = run_matching(
            all_rankings=all_rankings,
            all_district_assignments=all_district_assignments,
            df=df,
            school_info_df=school_info_df,
            lottery_global=lottery,
            list_length_min=min_len,
            list_length_mean=7,
            list_length_std=2,
            list_length_max=list_length_max,
            rng=rng,
            priority_config=priority_config,
            district_to_region=None,
            district_to_borough=district_to_borough,
            per_school_lottery=False,
            student_attrs=student_attrs
        )

        attr_df = pd.DataFrame(syn_attrs) if syn_attrs is not None else pd.DataFrame()
        attr_df['district'] = list(all_district_assignments)

        welfare_results = evaluate_simulation_output(
            sim_output={
                'rankings_as_indices': syn_rankings_idx,
                'matches_idx':         matches_idx,
                'student_attributes':  attr_df,
            },
            categories=['district'],
            output_dir=os.path.join(output_dir, f'min_len_{min_len}'),
        )

        stats = welfare_results.rank_stats
        matched = (matches_idx >= 0).sum()
        n_total = len(matches_idx)

        print(f"  pct_matched: {stats['pct_matched']:.2f}%")
        print(f"  avg_rank:    {stats['avg_rank']:.3f}")
        print(f"  rank_var:    {stats['rank_variance']:.3f}")

        summary_rows.append({
            'list_length_min': min_len,
            'pct_matched':     round(stats['pct_matched'], 4),
            'avg_rank':        round(stats['avg_rank'], 4),
            'rank_variance':   round(stats['rank_variance'], 4),
            'n_matched':       int(matched),
            'n_total':         int(n_total),
        })

    summary_df = pd.DataFrame(summary_rows)
    summary_path = os.path.join(output_dir, 'sweep_summary.csv')
    summary_df.to_csv(summary_path, index=False)
    print(f"\nSummary saved to {summary_path}")
    print(summary_df.to_string(index=False))
    return summary_df


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--params',      required=True)
    parser.add_argument('--output_dir',  required=True)
    parser.add_argument('--min_lengths', type=int, nargs='+', default=[1, 3, 5, 7, 10])
    parser.add_argument('--seed',        type=int, default=42)
    parser.add_argument('--n_jobs',      type=int, default=32)
    parser.add_argument('--df_filepath', type=str, default=None)
    args = parser.parse_args()

    # Load data
    if args.df_filepath is None:
        args.df_filepath = f"{POLISHED_DATA_DIR}/master_data_03_residential_district.xlsx"

    print("Loading data...")
    df_raw = read_data(args.df_filepath)
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
    df, match_stats_df, school_info_df, district_to_borough = preprocess_data(
        df_raw, match_stats_df, school_info_df, addtl_school_info_df
    )
    print(f"  Schools: {df['School DBN'].nunique()}, Students: {int(match_stats_df['Total Applicants'].sum())}")

    # Load priority config
    nyc_config_path = f"{RAW_DATA_DIR}/nyc_priority_config.json"
    priority_config = None
    if os.path.exists(nyc_config_path):
        with open(nyc_config_path) as f:
            priority_config = json.load(f)
        print(f"  Loaded priority config: {nyc_config_path}")

    # Load params
    print(f"Loading params from {args.params}...")
    with open(args.params, 'rb') as f:
        params = pickle.load(f)
    print(f"  K={len(params['global_phis'])} components")
    print(f"  phis: {params['global_phis']}")

    # Generate a fixed lottery
    np.random.seed(args.seed)
    n_students = int(match_stats_df['Total Applicants'].sum())
    lottery = np.random.permutation(n_students)

    run_sweep(
        params=params,
        lottery=lottery,
        df=df,
        match_stats_df=match_stats_df,
        school_info_df=school_info_df,
        priority_config=priority_config,
        district_to_borough=district_to_borough,
        min_lengths=args.min_lengths,
        output_dir=args.output_dir,
        seed=args.seed,
        n_jobs=args.n_jobs,
    )


if __name__ == '__main__':
    main()