import numpy as np
import argparse
import os
import matplotlib.pyplot as plt
from scipy.stats import kendalltau
from datetime import datetime
from analysis import log_and_print
from data_ingestion import read_data, preprocess_data
from em import EM_algorithm, run_single_simulation
from src.project_specific_scripts.synthetic_data_generator import create_synthetic_experiment, extract_realistic_params_from_real_data
from src.file_config import EXP_OUT_FOLDER, DATA_GENERATION_SEED, RAW_DATA_DIR, POLISHED_DATA_DIR

def run_synthetic_experiment_3_MoM_no_utilization(outfile=None):
    log_and_print("\n" + "="*60, log_file=outfile)
    log_and_print("EXPERIMENT 3 MoM, Match Stats, No Utilization", log_file=outfile)
    log_and_print("="*60, log_file=outfile)
    
    all_match_stats = []
    observed_stats = None

    for seed in range(40, 50):
        log_and_print(f"\nRunning synthetic experiment with seed {seed}...", log_file=outfile)
        df3, match_stats_df3, school_info_df3, true_params3 = create_synthetic_experiment(
            n_students=500, n_schools=20, capacity_per_school=30,
            k_ranking_length=10, true_K=2, seed=DATA_GENERATION_SEED
        )

        # Store observed once (same across seeds since data seed=DATA_GENERATION_SEED is fixed)
        if observed_stats is None:
            observed_stats = np.array([
                match_stats_df3['% Matches to Choice 1-3'].iloc[0],
                match_stats_df3['% Matches to Choice 1-5'].iloc[0],
                match_stats_df3['% Matches to Choice 1-10'].iloc[0],
                match_stats_df3['Unmatched'].iloc[0]
            ])

        params3, lottery3, log_liks3, agg = EM_algorithm(
            df3, match_stats_df3, school_info_df3,
            max_iter=10, M_simulations=10, K=3, seed=seed
        )

        # Run one final simulation with estimated params
        agg = run_single_simulation(params3, df3, match_stats_df3, school_info_df3, lottery3)
        all_match_stats.append(agg['match_stats'][0, :])

        out_lines = [
            f"\nSEED {seed} RESULTS:",
            f"  True phis: {true_params3['true_phis']}",
            f"  Estimated phis: {params3['global_phis']}",
            f"  Error: {np.abs(params3['global_phis'] - true_params3['true_phis'])}"
        ]
        for line in out_lines:
            log_and_print(line, log_file=outfile)
        with open(f"{EXP_OUT_FOLDER}experiment3_results.txt", "a+") as f:
            for line in out_lines:
                f.write(line + "\n")
                f.flush()

    all_match_stats = np.array(all_match_stats)  

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.boxplot(all_match_stats, labels=['Top-3', 'Top-5', 'Top-10', 'Unmatched'])

    for i, obs in enumerate(observed_stats):
        ax.scatter(i + 1, obs, color='red', zorder=5, marker='D', 
                label='Observed' if i == 0 else '')

    ax.set_ylabel('Percentage (%)')
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    ax.set_title('Match Statistics: Simulated vs Observed (K=2, Seeds 40-49)')
    ax.legend()
    plt.tight_layout()
    plt.savefig(f"{EXP_OUT_FOLDER}match_stats_boxplot_K2_seeds40-49_{timestamp}.png", dpi=150)
    plt.show()

def run_synthetic_experiment_3_MoM_yes_utilization(outfile=None):
    log_and_print("\n" + "="*60, log_file=outfile)
    log_and_print("EXPERIMENT 3 MoM, Match Stats, Yes Utilization", log_file=outfile)
    log_and_print("="*60, log_file=outfile)
    
    all_match_stats = []
    all_utilizations = []
    observed_stats = None

    for seed in range(40, 50):
        log_and_print(f"\nRunning synthetic experiment with seed {seed}...")
        df3, match_stats_df3, school_info_df3, true_params3 = create_synthetic_experiment(
            n_students=500, n_schools=20, capacity_per_school=30,
            k_ranking_length=10, true_K=3, district_ct=3, seed=DATA_GENERATION_SEED
        )

        # Store observed once (same across seeds since data seed=DATA_GENERATION_SEED is fixed)
        if observed_stats is None:
            observed_stats = np.array([
                match_stats_df3['% Matches to Choice 1-3'].iloc[0],
                match_stats_df3['% Matches to Choice 1-5'].iloc[0],
                match_stats_df3['% Matches to Choice 1-10'].iloc[0],
                match_stats_df3['Unmatched'].iloc[0]
            ])
            true_utilization = school_info_df3['Utilization'].values / 100.0

        params3, lottery3, log_liks3, agg = EM_algorithm(
            df3, match_stats_df3, school_info_df3,
            max_iter=10, M_simulations=10, K=3, seed=seed
        )

        all_match_stats.append(agg['match_stats'][0, :])
        sim_util = agg['filled'] / school_info_df3['Capacity'].values
        all_utilizations.append(sim_util)


        out_lines = [
            f"\nSEED {seed} RESULTS:",
            f"  True phis: {true_params3['true_phis']}",
            f"  Estimated phis: {params3['global_phis']}",
            f"  Error: {np.abs(params3['global_phis'] - true_params3['true_phis'])}"
        ]

        for d_id, true_sigma in true_params3['true_sigmas'].items():
            est_sigma = params3['districts'][d_id]['central_ranking']
            
            # Map schools to ranks for Kendall Tau (how similar is the ordering?)
            school_to_true_rank = {s: i for i, s in enumerate(true_sigma)}
            true_ranks = [school_to_true_rank[s] for s in true_sigma]
            est_ranks = [school_to_true_rank[s] for s in est_sigma]
            tau, _ = kendalltau(true_ranks, est_ranks)
            
            out_lines.append(f"  District {d_id} Sigma Kendall Tau: {tau:.4f}")
            out_lines.append(f"    True Top 3: {true_sigma[:3]}")
            out_lines.append(f"    Est  Top 3: {est_sigma[:3]}")

        for line in out_lines:
            log_and_print(line)
        with open(f"{EXP_OUT_FOLDER}experiment3_3_dists_utils_results.txt", "a+") as f:
            for line in out_lines:
                f.write(line + "\n")
                f.flush()

    all_match_stats = np.array(all_match_stats)  

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.boxplot(all_match_stats, labels=['Top-3', 'Top-5', 'Top-10', 'Unmatched'])

    for i, obs in enumerate(observed_stats):
        ax.scatter(i + 1, obs, color='red', zorder=5, marker='D', 
                label='Observed' if i == 0 else '')

    ax.set_ylabel('Percentage (%)')
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    ax.set_title('Match Statistics: Simulated vs Observed (K=3, 3 Districts, Seeds 40-49)')
    ax.legend()
    plt.tight_layout()
    plt.savefig(f"{EXP_OUT_FOLDER}match_stats_boxplot_K3_3dists_seeds40-49_{timestamp}.png", dpi=150)
    plt.show()

    all_util_array = np.array(all_utilizations) # Shape: (Seeds, Schools)
    fig2, ax2 = plt.subplots(figsize=(12, 5))
    
    # Create boxplot for each school (columns of the array)
    bp = ax2.boxplot(all_util_array, patch_artist=True)
    
    # Customize boxes
    for patch in bp['boxes']:
        patch.set_facecolor('lightblue')
        patch.set_alpha(0.6)

    # Overlay True Values
    ax2.scatter(range(1, len(true_utilization) + 1), true_utilization, 
                color='red', marker='D', s=30, zorder=5, label='True Observed Util')

    ax2.set_xlabel('School Index')
    ax2.set_ylabel('Utilization (Fraction of Capacity)')
    ax2.set_title('School-Level Utilization: Simulated (Box) vs True (Red Diamond)')
    ax2.axhline(y=1.0, color='gray', linestyle='--', alpha=0.5, label='100% Capacity')
    ax2.legend()
    ax2.set_title('School-Level Utilization: Simulated vs True (K=3, 3 Districts, Seeds 40-49)')
    
    plt.tight_layout()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    plt.savefig(f"{EXP_OUT_FOLDER}school_utilization_boxplot_K3_3dists_seeds40-49_{timestamp}.png", dpi=150)
    plt.show()

def run_synthetic_experiment_3_MoM_yes_utilization_relevant_caps_central_ranks(seed=40, single_seed=True, outfile=None):
    log_and_print("\n" + "="*60, log_file=outfile)
    log_and_print("EXPERIMENT 3 MoM, Match Stats, Yes Utilization, Relevant Capacities, Central Rankings", log_file=outfile)
    log_and_print("="*60, log_file=outfile)

    
    df = read_data(f"{POLISHED_DATA_DIR}/master_data_03_residential_district.xlsx")
    match_stats_df = read_data(f"{RAW_DATA_DIR}/DATA3_fall-2024-high-school-offer-results-website-1.xlsx",
                                sheet='Match to Choice-District')
    school_info_df = read_data(f"{RAW_DATA_DIR}/DATA4_fall-2025---hs-directory-data.xlsx",
                            sheet='Data')
    addtl_school_info_df = read_data(f"{RAW_DATA_DIR}/DATA2_fall-2024-admissions_part-ii_suppressed.xlsx",
                            sheet='School')
    df, match_stats_df, school_info_df = preprocess_data(df, match_stats_df, school_info_df, addtl_school_info_df)
    
    if not single_seed:
        all_match_stats = []
        all_utilizations = []
        true_utilization = None
        observed_stats = None
        
        start_seed = 40
        end_seed = 139
        for s in range(start_seed, end_seed + 1):
            match_stats_file = f"{EXP_OUT_FOLDER}temp_match_stats_seed_{s}.npy"
            util_file = f"{EXP_OUT_FOLDER}temp_utilizations_seed_{s}.npy"
            if os.path.isfile(match_stats_file) and os.path.isfile(util_file):
                all_match_stats.append(np.load(match_stats_file))
                all_utilizations.append(np.load(util_file))
                if observed_stats is None:
                    obs_file = f"{EXP_OUT_FOLDER}temp_observed_stats.npy"
                    util_true_file = f"{EXP_OUT_FOLDER}temp_true_utilization.npy"
                    if os.path.isfile(obs_file) and os.path.isfile(util_true_file):
                        observed_stats = np.load(obs_file)
                        true_utilization = np.load(util_true_file)
        
        if len(all_match_stats) == 0:
            log_and_print("No accumulated results found. Skipping final analysis.", log_file=outfile)
            return
        
        all_match_stats = np.array(all_match_stats)
        all_utilizations = np.array(all_utilizations)
        
        # Create final plots from accumulated data
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.boxplot(all_match_stats, labels=['Top-3', 'Top-5', 'Top-10', 'Unmatched'])

        if observed_stats is not None:
            for i, obs in enumerate(observed_stats):
                ax.scatter(i + 1, obs, color='red', zorder=5, marker='D', 
                        label='Observed' if i == 0 else '')

        ax.set_ylabel('Percentage (%)')
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        ax.set_title(f'Match Statistics: Simulated vs Observed (K=3, 3 Districts, Real Capacities/Rankings, Seeds {start_seed}-{end_seed})')
        ax.legend()
        plt.tight_layout()
        plt.savefig(f"{EXP_OUT_FOLDER}match_stats_boxplot_K3_3dists_realcaps_seeds{start_seed}-{end_seed}_{timestamp}.png", dpi=150)
        plt.close()

        if true_utilization is not None:
            fig2, ax2 = plt.subplots(figsize=(12, 5))
            bp = ax2.boxplot(all_utilizations, patch_artist=True)
            for patch in bp['boxes']:
                patch.set_facecolor('lightblue')
                patch.set_alpha(0.6)

            ax2.scatter(range(1, len(true_utilization) + 1), true_utilization, 
                        color='red', marker='D', s=30, zorder=5, label='True Observed Util')

            ax2.set_xlabel('School Index')
            ax2.set_ylabel('Utilization (Fraction of Capacity)')
            ax2.set_title('School-Level Utilization: Simulated (Box) vs True (Red Diamond)')
            ax2.axhline(y=1.0, color='gray', linestyle='--', alpha=0.5, label='100% Capacity')
            ax2.legend()
            ax2.set_title(f'School-Level Utilization: Simulated vs True (K=3, 3 Districts, Real Capacities/Rankings, Seeds {start_seed}-{end_seed})')
            
            plt.tight_layout()
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            plt.savefig(f"{EXP_OUT_FOLDER}school_utilization_boxplot_K3_3dists_realcaps_seeds{start_seed}-{end_seed}_{timestamp}.png", dpi=150)
            plt.close()
        
        log_and_print(f"Final analysis complete. Processed {len(all_match_stats)} seeds.", log_file=outfile)
        return
    
    # Single seed run: process and save results
    all_match_stats = []
    all_utilizations = []
    observed_stats = None

    real_sigmas, real_caps, real_schools = extract_realistic_params_from_real_data(
        df, school_info_df, n_schools=20, n_students=500, if_plot=False
    )

    log_and_print(f"\nRunning synthetic experiment with seed {seed}...")
    df3, match_stats_df3, school_info_df3, true_params3 = create_synthetic_experiment(
        n_students=500, n_schools=20, capacity_per_school=30,
        k_ranking_length=10, true_K=3, district_ct=3, seed=DATA_GENERATION_SEED,
        external_sigmas=real_sigmas, external_capacities=real_caps,
        external_schools=real_schools
    )

    # Store observed once (same across seeds since data seed=DATA_GENERATION_SEED is fixed)
    if observed_stats is None:
        observed_stats = np.array([
            match_stats_df3['% Matches to Choice 1-3'].iloc[0],
            match_stats_df3['% Matches to Choice 1-5'].iloc[0],
            match_stats_df3['% Matches to Choice 1-10'].iloc[0],
            match_stats_df3['Unmatched'].iloc[0]
        ])
        true_utilization = school_info_df3['Utilization'].values / 100.0

    params3, lottery3, log_liks3, agg = EM_algorithm(
        df3, match_stats_df3, school_info_df3,
        max_iter=10, M_simulations=10, K=3, seed=seed
    )

    all_match_stats.append(agg['match_stats'][0, :])
    sim_util = agg['filled'] / school_info_df3['Capacity'].values
    all_utilizations.append(sim_util)

    # Save observed stats on first run (when seed == 40)
    if seed == 40:
        np.save(f"{EXP_OUT_FOLDER}temp_observed_stats.npy", observed_stats)
        np.save(f"{EXP_OUT_FOLDER}temp_true_utilization.npy", true_utilization)

    # Save results to temp files for later aggregation
    np.save(f"{EXP_OUT_FOLDER}temp_match_stats_seed_{seed}.npy", agg['match_stats'][0, :])
    np.save(f"{EXP_OUT_FOLDER}temp_utilizations_seed_{seed}.npy", sim_util)

    out_lines = [
        f"\nSEED {seed} RESULTS:",
        f"  True phis: {true_params3['true_phis']}",
        f"  Estimated phis: {params3['global_phis']}",
        f"  Error: {np.abs(params3['global_phis'] - true_params3['true_phis'])}"
    ]

    for d_id, true_sigma in true_params3['true_sigmas'].items():
        est_sigma = params3['districts'][d_id]['central_ranking']
        
        # Map schools to ranks for Kendall Tau (how similar is the ordering?)
        school_to_true_rank = {s: i for i, s in enumerate(true_sigma)}
        true_ranks = [school_to_true_rank[s] for s in true_sigma]
        est_ranks = [school_to_true_rank[s] for s in est_sigma]
        tau, _ = kendalltau(true_ranks, est_ranks)
        
        out_lines.append(f"  District {d_id} Sigma Kendall Tau: {tau:.4f}")
        out_lines.append(f"    True Top 3: {true_sigma[:3]}")
        out_lines.append(f"    Est  Top 3: {est_sigma[:3]}")

    for line in out_lines:
        log_and_print(line)
    with open(f"{EXP_OUT_FOLDER}experiment3_3_dists_utils_results.txt", "a+") as f:
        for line in out_lines:
            f.write(line + "\n")
            f.flush()

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--K', type=int, default=12, help='Number of mixture components for real data')
    parser.add_argument('--M', type=int, default=5, help='Number of simulations per evaluation')
    parser.add_argument('--max_iter', type=int, default=5, help='Maximum EM iterations')
    parser.add_argument('--seed', type=int, default=40, help='Random seed for synthetic experiments')
    parser.add_argument('--final-analysis', action='store_true', help='Run final aggregation and plotting step')
    
    args = parser.parse_args()
    
    if args.final_analysis:
        # Run final analysis on accumulated results
        run_synthetic_experiment_3_MoM_yes_utilization_relevant_caps_central_ranks(single_seed=False)
    else:
        # Run synthetic experiments with single seed
        run_synthetic_experiment_3_MoM_yes_utilization_relevant_caps_central_ranks(seed=args.seed, single_seed=True)