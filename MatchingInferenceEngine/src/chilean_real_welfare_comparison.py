
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from direct_chile_welfare import load_chile_welfare_inputs
from gale_shapley import gale_shapley_per_school_numba_wrapper
from welfare import build_student_level_welfare, summarize_global_sweep


def top_p_curve_female(sim_output, max_p, matched_only=False):
    student_df = build_student_level_welfare(
        rankings_as_indices=sim_output['rankings_as_indices'],
        matches_idx=sim_output['matches_idx'],
        student_attributes=sim_output['student_attributes'],
    )
    female_df = student_df[student_df['female'] == 1]
    if matched_only:
        female_df = female_df[female_df['matched'] == True]
    return summarize_global_sweep(female_df, max_p=max_p)


def run_stb_curves(rankings_as_indices, capacities, student_attrs, n_runs, seed, max_p):
    rng = np.random.default_rng(seed)
    n_students = len(rankings_as_indices)
    n_schools = len(capacities)
    all_sweeps_uncond = []
    all_sweeps_cond = []

    for run in range(n_runs):
        print(f"  STB run {run+1}/{n_runs}...")
        lottery_1d = rng.random(n_students)
        school_lotteries = np.tile(lottery_1d, (n_schools, 1))
        matches = gale_shapley_per_school_numba_wrapper(
            rankings_as_indices, school_lotteries, capacities
        )
        matched_rate = (matches >= 0).mean() * 100
        print(f"  STB run {run+1} match rate: {matched_rate:.1f}%")

        sim_output = {
            'rankings_as_indices': rankings_as_indices,
            'matches_idx':         matches,
            'student_attributes':  student_attrs,
        }
        all_sweeps_uncond.append(
            top_p_curve_female(sim_output, max_p=max_p, matched_only=False)
            .set_index('p')['top_p_pct']
        )
        all_sweeps_cond.append(
            top_p_curve_female(sim_output, max_p=max_p, matched_only=True)
            .set_index('p')['top_p_pct']
        )

    def _agg(sweeps):
        avg = pd.concat(sweeps, axis=1).mean(axis=1).reset_index()
        avg.columns = ['p', 'top_p_pct']
        std = pd.concat(sweeps, axis=1).std(axis=1).reset_index()
        std.columns = ['p', 'top_p_std']
        return avg.merge(std, on='p')

    return _agg(all_sweeps_uncond), _agg(all_sweeps_cond)


def make_plot(mtb_uncond, mtb_cond, stb_uncond, stb_cond,
              n_female, n_female_matched, n_stb_runs, output_path):
    fig, ax = plt.subplots(figsize=(10, 6))

    ax.plot(mtb_uncond['p'], mtb_uncond['top_p_pct'],
            marker='o', color='#2980b9', linewidth=2,
            label=f'MTB real — all female (n={n_female:,})')
    ax.plot(mtb_cond['p'], mtb_cond['top_p_pct'],
            marker='o', color='#2980b9', linewidth=2, linestyle='--',
            label=f'MTB real — matched female only (n={n_female_matched:,})')

    ax.plot(stb_uncond['p'], stb_uncond['top_p_pct'],
            marker='s', color='#e74c3c', linewidth=2,
            label=f'STB counterfactual — all female (avg {n_stb_runs} runs)')
    ax.fill_between(stb_uncond['p'],
                    stb_uncond['top_p_pct'] - stb_uncond['top_p_std'],
                    stb_uncond['top_p_pct'] + stb_uncond['top_p_std'],
                    alpha=0.15, color='#e74c3c')

    ax.plot(stb_cond['p'], stb_cond['top_p_pct'],
            marker='s', color='#e74c3c', linewidth=2, linestyle='--',
            label=f'STB counterfactual — matched female only')
    ax.fill_between(stb_cond['p'],
                    stb_cond['top_p_pct'] - stb_cond['top_p_std'],
                    stb_cond['top_p_pct'] + stb_cond['top_p_std'],
                    alpha=0.15, color='#e74c3c')

    ax.set_xlabel("p (top-p threshold)", fontsize=13)
    ax.set_ylabel("Female students matched to top-p choice (%)", fontsize=13)
    ax.set_title(
        "Welfare comparison for female students — Chile SAE\n"
        "MTB (real) vs STB (counterfactual), same preference lists",
        fontsize=13,
    )
    ax.set_ylim(0, 100)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches='tight')
    print(f"Saved to {output_path}")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--individual",  required=True)
    parser.add_argument("--capacity",    required=True)
    parser.add_argument("--output",      default="female_welfare_comparison.png")
    parser.add_argument("--n_stb_runs",  type=int, default=10)
    parser.add_argument("--max_p",       type=int, default=10)
    parser.add_argument("--seed",        type=int, default=42)
    args = parser.parse_args()

    print("Loading real data...")
    sim_output_mtb, student_attrs, school_info = load_chile_welfare_inputs(
        args.individual, capacity_path=args.capacity
    )

    capacities = school_info['capacities']
    if capacities is None:
        raise ValueError("capacity_path is required for STB counterfactual")

    print("Computing MTB curves...")
    mtb_sweep_uncond = top_p_curve_female(sim_output_mtb, max_p=args.max_p, matched_only=False)
    mtb_sweep_cond   = top_p_curve_female(sim_output_mtb, max_p=args.max_p, matched_only=True)
    n_female = int((student_attrs['female'] == 1).sum())
    n_female_matched = int((((student_attrs['female'] == 1).values) & (sim_output_mtb['matches_idx'] >= 0)).sum())
    print(f"Female students total:   {n_female:,}")
    print(f"Female students matched: {n_female_matched:,}")

    print(f"Running {args.n_stb_runs} STB counterfactual runs...")
    stb_sweep_uncond, stb_sweep_cond = run_stb_curves(
        rankings_as_indices=sim_output_mtb['rankings_as_indices'],
        capacities=capacities,
        student_attrs=student_attrs,
        n_runs=args.n_stb_runs,
        seed=args.seed,
        max_p=args.max_p,
    )

    make_plot(
        mtb_sweep_uncond, mtb_sweep_cond,
        stb_sweep_uncond, stb_sweep_cond,
        n_female, n_female_matched, args.n_stb_runs, args.output,
    )

    print("\nMTB vs STB female top-p comparison (unconditional):")
    merged = mtb_sweep_uncond[['p', 'top_p_pct']].rename(columns={'top_p_pct': 'MTB_%'}).merge(
        stb_sweep_uncond[['p', 'top_p_pct']].rename(columns={'top_p_pct': 'STB_%'}), on='p'
    )
    merged['STB-MTB'] = (merged['STB_%'] - merged['MTB_%']).round(2)
    print(merged.to_string(index=False))

    print("\nMTB vs STB female top-p comparison (matched only):")
    merged_cond = mtb_sweep_cond[['p', 'top_p_pct']].rename(columns={'top_p_pct': 'MTB_%'}).merge(
        stb_sweep_cond[['p', 'top_p_pct']].rename(columns={'top_p_pct': 'STB_%'}), on='p'
    )
    merged_cond['STB-MTB'] = (merged_cond['STB_%'] - merged_cond['MTB_%']).round(2)
    print(merged_cond.to_string(index=False))


if __name__ == "__main__":
    main()