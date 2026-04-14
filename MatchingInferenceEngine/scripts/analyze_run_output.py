"""
Parse EduRanker experiment log files, generate reports and plots.

Usage:
    python parse_logs.py --results-dir /path/to/experiment-results
    python parse_logs.py --results-dir /path/to/experiment-results --min-date 20260324 --report-out report.txt --plots-dir ./plots
"""

import os
import re
import argparse
import numpy as np
from collections import defaultdict, Counter
from dataclasses import dataclass, field
from typing import Optional
import pandas as pd
from scipy.stats import kendalltau
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import random

TARGET_SUBSAMPLE_PARAM_KEY = (4, 4, 4, 4)
TARGET_SUBSAMPLE_RANDOM_SEED = 123


# ============================================================
# Data structures
# ============================================================

@dataclass
class DistrictFit:
    district: int
    obs_top3: float
    obs_top5: float
    obs_top10: float
    obs_unmatched: float
    sim_top3: float
    sim_top5: float
    sim_top10: float
    sim_unmatched: float

    @property
    def diff_top3(self):
        return self.obs_top3 - self.sim_top3

    @property
    def diff_top5(self):
        return self.obs_top5 - self.sim_top5

    @property
    def diff_top10(self):
        return self.obs_top10 - self.sim_top10

    @property
    def diff_unmatched(self):
        return self.obs_unmatched - self.sim_unmatched


@dataclass
class ExperimentResult:
    filename: str
    K: int
    M: int
    max_iter: int
    max_iter_opt: int
    seed: str
    imputed_seed: Optional[str] = None
    final_phis: list = field(default_factory=list)
    final_weights: list = field(default_factory=list)
    central_rankings: dict = field(default_factory=dict)
    best_log_likelihood: Optional[float] = None
    best_fit_diagnostics: list = field(default_factory=list)
    best_util_mae: Optional[float] = None

    @property
    def param_key(self):
        return (self.K, self.M, self.max_iter, self.max_iter_opt)

    @property
    def param_str(self):
        return f"K={self.K}_M={self.M}_iter={self.max_iter}_opt={self.max_iter_opt}"


# ============================================================
# Parsing
# ============================================================

def parse_filename(filename):
    basename = os.path.splitext(os.path.basename(filename))[0]

    def _extract(pattern, text, default=None):
        m = re.search(pattern, text)
        return m.group(1) if m else default

    K = int(_extract(r'K[_=](\d+)', basename, '0'))
    M = int(_extract(r'M[_=](\d+)', basename, '0'))
    max_iter = int(_extract(r'iter[_=](\d+)', basename, '0'))
    max_iter_opt = int(_extract(r'opt[_=](\d+)', basename, '0'))
    seed = _extract(r'seed[_=](\d+)', basename, None) or 'default'
    imputed_seed = _extract(r'imputed_seed[_=](\d+)', basename, None)
    date_match = re.search(r'(\d{8})_\d{6}', basename)
    date_str = date_match.group(1) if date_match else None

    return K, M, max_iter, max_iter_opt, seed, imputed_seed, date_str


def parse_log_file(filepath):
    filename = os.path.basename(filepath)
    K, M, max_iter, max_iter_opt, seed, imputed_seed, date_str = parse_filename(filename)

    result = ExperimentResult(
        filename=filename, K=K, M=M, max_iter=max_iter,
        max_iter_opt=max_iter_opt, seed=seed, imputed_seed=imputed_seed,
    )

    with open(filepath, 'r') as f:
        lines = f.readlines()
    lines = [l.rstrip('\n') for l in lines]

    # Find last "New best log-likelihood!"
    best_ll_line_idx = None
    for i, line in enumerate(lines):
        if 'New best log-likelihood!' in line:
            m = re.search(r'New best log-likelihood! - (-?[\d.]+)', line)
            if m:
                result.best_log_likelihood = float(m.group(1))
                best_ll_line_idx = i

    # Find FIT DIAGNOSTICS block preceding best
    if best_ll_line_idx is not None:
        diag_start = None
        for i in range(best_ll_line_idx, -1, -1):
            if 'FIT DIAGNOSTICS' in lines[i]:
                diag_start = i
                break
        if diag_start is not None:
            diag_end = None
            for i in range(diag_start + 2, len(lines)):
                if lines[i].startswith('=' * 10):
                    diag_end = i
                    break
            if diag_end is not None:
                block = lines[diag_start:diag_end]
                result.best_fit_diagnostics = _parse_fit_diagnostics_block(block)
                for line in block:
                    m = re.search(r'Mean Absolute Utilization Error:\s*([\d.]+)%', line)
                    if m:
                        result.best_util_mae = float(m.group(1))

    # Extract final phis and weights (search from end)
    for line in reversed(lines):
        if 'Global phis:' in line and not result.final_phis:
            m = re.search(r'Global phis: \[(.*?)\]', line)
            if m:
                result.final_phis = [float(x) for x in m.group(1).split()]
        if 'Global weights:' in line and not result.final_weights:
            m = re.search(r'Global weights: \[(.*?)\]', line)
            if m:
                result.final_weights = [float(x) for x in m.group(1).split()]
        if result.final_phis and result.final_weights:
            break

    # Extract central rankings
    in_rankings = False
    for line in lines:
        if 'Estimated central rankings (sigma) per district:' in line:
            in_rankings = True
            continue
        if in_rankings:
            m = re.match(r"\s*District (\d+): \[(.+)\]", line)
            if m:
                district = int(m.group(1))
                schools = [s.strip("' ") for s in m.group(2).split("',")]
                result.central_rankings[district] = schools
            elif line.startswith('===') or 'RUN COMPLETE' in line:
                in_rankings = False

    return result


def _parse_fit_diagnostics_block(block_lines):
    diagnostics = []
    current_district = None
    obs_vals = None
    sim_vals = None
    pct_pattern = re.compile(
        r'top3=\s*(-?[\d.]+)%.*top5=\s*(-?[\d.]+)%.*top10=\s*(-?[\d.]+)%.*unmatched=\s*(-?[\d.]+)%'
    )
    for line in block_lines:
        dm = re.match(r'\s*District (\d+):', line)
        if dm:
            current_district = int(dm.group(1))
            obs_vals = None
            sim_vals = None
            continue
        if current_district is not None:
            if 'Observed:' in line:
                m = pct_pattern.search(line)
                if m:
                    obs_vals = [float(m.group(i)) for i in range(1, 5)]
            elif 'Simulated:' in line:
                m = pct_pattern.search(line)
                if m:
                    sim_vals = [float(m.group(i)) for i in range(1, 5)]
            if obs_vals and sim_vals:
                diagnostics.append(DistrictFit(
                    district=current_district,
                    obs_top3=obs_vals[0], obs_top5=obs_vals[1],
                    obs_top10=obs_vals[2], obs_unmatched=obs_vals[3],
                    sim_top3=sim_vals[0], sim_top5=sim_vals[1],
                    sim_top10=sim_vals[2], sim_unmatched=sim_vals[3],
                ))
                current_district = None
                obs_vals = None
                sim_vals = None
    return diagnostics


def find_and_parse_logs(results_dir, min_date='20260324'):
    results = []
    for fname in sorted(os.listdir(results_dir)):
        if not fname.endswith('.txt') or not (fname.startswith('real_experiment') or fname.startswith('chilean_experiment')):
            continue
        _, _, _, _, _, _, date_str = parse_filename(fname)
        if date_str is None or date_str < min_date:
            continue
        filepath = os.path.join(results_dir, fname)
        with open(filepath, 'r') as f:
            content = f.read()
        if 'RUN COMPLETE' not in content:
            continue
        print(f"Parsing: {fname}")
        results.append(parse_log_file(filepath))
    print(f"\nParsed {len(results)} completed experiments.")
    return results


def subsample_one_per_imputed_seed(results, random_seed=0, param_key=None):
    """Deterministically pick one run per imputed seed.

    Args:
        results: Parsed experiment results.
        random_seed: Seed used to choose among duplicate files for the same imputed seed.
        param_key: Optional exact param_key tuple to restrict which runs are eligible.
    """
    if param_key is None:
        target_group = list(results)
        untouched = []
    else:
        target_group = [r for r in results if r.param_key == param_key]
        untouched = [r for r in results if r.param_key != param_key]

    eligible = [r for r in target_group if r.imputed_seed is not None]

    grouped = defaultdict(list)
    for result in eligible:
        grouped[result.imputed_seed].append(result)

    rng = random.Random(random_seed)
    selected = []
    for imputed_seed in sorted(grouped.keys()):
        group = sorted(grouped[imputed_seed], key=lambda r: r.filename)
        chosen = group[rng.randrange(len(group))]
        selected.append(chosen)

    passthrough = [r for r in target_group if r.imputed_seed is None]
    filtered = untouched + passthrough + selected
    filtered.sort(key=lambda r: r.filename)

    print(
        f"Subsampled imputed runs: kept {len(selected)} of {len(eligible)} eligible runs "
        f"across {len(grouped)} imputed seeds (seed={random_seed})."
    )
    if param_key is not None:
        print(
            f"Subsample applied only to param_key={param_key}; "
            f"kept {len(untouched)} runs from other configs unchanged."
        )
    return filtered


# ============================================================
# Report generation
# ============================================================

def generate_report(results, out_path=None):
    lines = []
    w = lines.append


    w("=" * 100)
    w("EDURANKER EXPERIMENT RESULTS REPORT")
    w("=" * 100)
    w(f"\nTotal completed experiments: {len(results)}")

    # --- Overview table ---
    w(f"\n{'='*100}")
    w(f"{'Seed':<10} {'K':<4} {'M':<4} {'Iter':<5} {'Opt':<5} {'Best LL':<16} {'Util MAE':<10} {'Phis'}")
    w(f"{'-'*100}")
    for r in sorted(results, key=lambda x: x.best_log_likelihood or float('-inf'), reverse=True):
        phis_str = ', '.join(f'{p:.3f}' for p in r.final_phis) if r.final_phis else 'N/A'
        mae_str = f'{r.best_util_mae:.2f}%' if r.best_util_mae is not None else 'N/A'
        ll_str = f'{r.best_log_likelihood:.2f}' if r.best_log_likelihood is not None else 'N/A'
        w(f"{r.seed:<10} {r.K:<4} {r.M:<4} {r.max_iter:<5} {r.max_iter_opt:<5} {ll_str:<16} {mae_str:<10} [{phis_str}]")
    w(f"{'='*100}")

    # --- Best run details ---
    best = max(results, key=lambda x: x.best_log_likelihood or float('-inf'))
    w(f"\n{'='*100}")
    w(f"BEST RUN: {best.filename}")
    w(f"{'='*100}")
    w(f"  Log-likelihood: {best.best_log_likelihood}")
    w(f"  Utilization MAE: {best.best_util_mae}%")
    w(f"  Phis: {[f'{p:.4f}' for p in best.final_phis]}")
    w(f"  Weights: {[f'{p:.4f}' for p in best.final_weights]}")
    w(f"  Params: K={best.K}, M={best.M}, max_iter={best.max_iter}, max_iter_opt={best.max_iter_opt}, seed={best.seed}")

    # Per-district match stats
    if best.best_fit_diagnostics:
        w(f"\n  Match Stats Differences (obs - sim):")
        w(f"  {'District':<10} {'dTop3':>8} {'dTop5':>8} {'dTop10':>8} {'dUnmatch':>8}  {'Obs Top3':>9} {'Sim Top3':>9}")
        w(f"  {'-'*70}")
        for d in sorted(best.best_fit_diagnostics, key=lambda x: x.district):
            w(f"  {d.district:<10} {d.diff_top3:>+8.1f} {d.diff_top5:>+8.1f} {d.diff_top10:>+8.1f} {d.diff_unmatched:>+8.1f}  {d.obs_top3:>8.1f}% {d.sim_top3:>8.1f}%")

        diffs_top3 = [d.diff_top3 for d in best.best_fit_diagnostics]
        diffs_top5 = [d.diff_top5 for d in best.best_fit_diagnostics]
        diffs_top10 = [d.diff_top10 for d in best.best_fit_diagnostics]
        diffs_unmatch = [d.diff_unmatched for d in best.best_fit_diagnostics]
        w(f"\n  Mean absolute diff:  top3={np.mean(np.abs(diffs_top3)):.1f}  top5={np.mean(np.abs(diffs_top5)):.1f}  top10={np.mean(np.abs(diffs_top10)):.1f}  unmatched={np.mean(np.abs(diffs_unmatch)):.1f}")
        w(f"  Mean diff:           top3={np.mean(diffs_top3):+.1f}  top5={np.mean(diffs_top5):+.1f}  top10={np.mean(diffs_top10):+.1f}  unmatched={np.mean(diffs_unmatch):+.1f}")

    # Top 10 sigma per district
    w(f"\n  Top 10 Central Rankings per District (best run):")
    w(f"  {'-'*80}")
    for district in sorted(best.central_rankings.keys()):
        sigma = best.central_rankings[district]
        top10 = sigma[:10]
        w(f"  District {district:>2}: {', '.join(top10)}")

    # --- Cross-seed aggregation (imputation runs only) ---
    imputation_runs = [r for r in results if r.seed != 'default']
    imputation_by_params = defaultdict(list)
    for r in imputation_runs:
        imputation_by_params[r.param_key].append(r)

    if any(len(v) > 1 for v in imputation_by_params.values()):
        w(f"\n{'='*100}")
        w("CROSS-SEED AGGREGATION (mean +/- std across imputation seeds)")
        w(f"{'='*100}")

        for param_key, group in sorted(imputation_by_params.items()):
            if len(group) < 2:
                continue
            K, M_val, max_iter, max_iter_opt = param_key
            w(f"\n  K={K}, M={M_val}, max_iter={max_iter}, max_iter_opt={max_iter_opt} ({len(group)} seeds)")

            lls = [r.best_log_likelihood for r in group if r.best_log_likelihood is not None]
            maes = [r.best_util_mae for r in group if r.best_util_mae is not None]

            if lls:
                w(f"    Log-likelihood:  mean={np.mean(lls):.2f}  std={np.std(lls):.2f}  best={max(lls):.2f}  worst={min(lls):.2f}")
            if maes:
                w(f"    Util MAE:        mean={np.mean(maes):.2f}%  std={np.std(maes):.2f}%")

            # Aggregate per-district diffs
            all_diffs = defaultdict(lambda: {'top3': [], 'top5': [], 'top10': [], 'unmatched': []})
            for r in group:
                for d in r.best_fit_diagnostics:
                    all_diffs[d.district]['top3'].append(d.diff_top3)
                    all_diffs[d.district]['top5'].append(d.diff_top5)
                    all_diffs[d.district]['top10'].append(d.diff_top10)
                    all_diffs[d.district]['unmatched'].append(d.diff_unmatched)

            if all_diffs:
                w(f"\n    Per-district mean diff (across seeds):")
                w(f"    {'District':<10} {'dTop3':>12} {'dTop5':>12} {'dTop10':>12} {'dUnmatch':>12}")
                w(f"    {'-'*60}")
                for district in sorted(all_diffs.keys()):
                    dd = all_diffs[district]
                    t3 = f"{np.mean(dd['top3']):>+6.1f}+/-{np.std(dd['top3']):4.1f}"
                    t5 = f"{np.mean(dd['top5']):>+6.1f}+/-{np.std(dd['top5']):4.1f}"
                    t10 = f"{np.mean(dd['top10']):>+6.1f}+/-{np.std(dd['top10']):4.1f}"
                    tu = f"{np.mean(dd['unmatched']):>+6.1f}+/-{np.std(dd['unmatched']):4.1f}"
                    w(f"    {district:<10} {t3} {t5} {t10} {tu}")

            # Phi aggregation
            all_phis = [r.final_phis for r in group if r.final_phis]
            if all_phis:
                max_k = max(len(p) for p in all_phis)
                w(f"\n    Phi estimates (across seeds):")
                for k_idx in range(max_k):
                    vals = [p[k_idx] for p in all_phis if len(p) > k_idx]
                    w(f"      phi[{k_idx+1}]: mean={np.mean(vals):.4f}  std={np.std(vals):.4f}  [{min(vals):.4f}, {max(vals):.4f}]")

    # --- Parameter sensitivity (averaged across seeds per config) ---
    all_by_params = defaultdict(list)
    for r in results:
        all_by_params[r.param_key].append(r)

    w(f"\n{'='*100}")
    w("PARAMETER SENSITIVITY (averaged across seeds per config)")
    w(f"{'='*100}")

    w(f"\n  All configs:")
    w(f"  {'K':<4} {'M':<4} {'Iter':<5} {'Opt':<5} {'n':<5} {'Best LL (mean+/-std)':<28} {'Util MAE (mean+/-std)':<24}")
    w(f"  {'-'*80}")
    for param_key, group in sorted(all_by_params.items()):
        K, M_val, max_iter, max_iter_opt = param_key
        lls = [r.best_log_likelihood for r in group if r.best_log_likelihood is not None]
        maes = [r.best_util_mae for r in group if r.best_util_mae is not None]
        n = len(group)
        ll_str = f"{np.mean(lls):.1f}+/-{np.std(lls):.1f}" if lls else "N/A"
        mae_str = f"{np.mean(maes):.1f}+/-{np.std(maes):.1f}%" if maes else "N/A"
        w(f"  {K:<4} {M_val:<4} {max_iter:<5} {max_iter_opt:<5} {n:<5} {ll_str:<28} {mae_str}")

    baseline_key = min(set(r.param_key for r in results))
    baseline_K, baseline_M, baseline_iter, baseline_opt = baseline_key
    baseline_vals = {'K': baseline_K, 'M': baseline_M, 'max_iter': baseline_iter, 'max_iter_opt': baseline_opt}
    w(f"\n  Baseline config: K={baseline_K}, M={baseline_M}, iter={baseline_iter}, opt={baseline_opt}")

    for vary_param, label in [('K', 'K'), ('M', 'M'), ('max_iter', 'max_iter'), ('max_iter_opt', 'max_iter_opt')]:
        other_params = [p for p in ['K', 'M', 'max_iter', 'max_iter_opt'] if p != vary_param]
        sweep_results = [r for r in results if all(getattr(r, p) == baseline_vals[p] for p in other_params)]
        val_groups = defaultdict(list)
        for r in sweep_results:
            val_groups[getattr(r, vary_param)].append(r)
        if len(val_groups) < 2:
            continue
        w(f"\n  Varying {label} (others held at baseline):")
        for val in sorted(val_groups.keys()):
            group = val_groups[val]
            lls = [r.best_log_likelihood for r in group if r.best_log_likelihood is not None]
            maes = [r.best_util_mae for r in group if r.best_util_mae is not None]
            n = len(group)
            ll_str = f"{np.mean(lls):.1f}+/-{np.std(lls):.1f}" if lls else "N/A"
            mae_str = f"{np.mean(maes):.1f}+/-{np.std(maes):.1f}%" if maes else "N/A"
            w(f"    {label}={val:<4}  n={n:<4}  LL={ll_str:<28}  MAE={mae_str}")

    report_text = '\n'.join(lines)

    if out_path:
        with open(out_path, 'w') as f:
            f.write(report_text)
        print(f"\nReport written to {out_path}")

    print(report_text)
    return report_text


# ============================================================
# Plots
# ============================================================

def generate_plots(results, plots_dir='.'):

    os.makedirs(plots_dir, exist_ok=True)

    imputation_runs = [r for r in results if r.seed != 'default']

    imputation_by_params = defaultdict(list)
    for r in imputation_runs:
        imputation_by_params[r.param_key].append(r)

    by_all_params = defaultdict(list)
    for r in results:
        by_all_params[r.param_key].append(r)

    print(f"  Imputation runs: {len(imputation_runs)}, Total runs: {len(results)}")

    # Param sensitivity: all results, averaged across seeds
    _plot_parameter_sensitivity(results, by_all_params, plots_dir)

    # Best run: from all results
    _plot_best_run_district_fit(results, plots_dir)

    # Cross-seed analysis: only from imputation runs
    if imputation_runs:
        _plot_cross_seed_variance(imputation_by_params, plots_dir)
        _plot_phi_comparison(imputation_by_params, plots_dir)
        _plot_central_ranking_comparison(imputation_by_params, plots_dir)


def _plot_parameter_sensitivity(results, by_params, plots_dir):
    """Plot parameter sensitivity, averaging across seeds per config."""
    if not results:
        return

    # Find baseline: config with smallest param tuple
    baseline_key = min(set(r.param_key for r in results))
    baseline_K, baseline_M, baseline_iter, baseline_opt = baseline_key
    baseline_vals = {'K': baseline_K, 'M': baseline_M, 'max_iter': baseline_iter, 'max_iter_opt': baseline_opt}

    for vary_param, label in [('K', 'K'), ('M', 'M'), ('max_iter', 'max_iter'), ('max_iter_opt', 'max_iter_opt')]:
        val_to_lls = defaultdict(list)
        val_to_maes = defaultdict(list)

        other_params = [p for p in ['K', 'M', 'max_iter', 'max_iter_opt'] if p != vary_param]

        for r in results:
            if all(getattr(r, p) == baseline_vals[p] for p in other_params):
                val = getattr(r, vary_param)
                if r.best_log_likelihood is not None:
                    val_to_lls[val].append(r.best_log_likelihood)
                if r.best_util_mae is not None:
                    val_to_maes[val].append(r.best_util_mae)

        if len(val_to_lls) < 2:
            continue

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

        vals = sorted(val_to_lls.keys())
        means = [np.mean(val_to_lls[v]) for v in vals]
        stds = [np.std(val_to_lls[v]) for v in vals]
        counts = [len(val_to_lls[v]) for v in vals]
        ax1.errorbar(vals, means, yerr=stds, marker='o', capsize=5)
        for v, m, n in zip(vals, means, counts):
            ax1.annotate(f'n={n}', (v, m), textcoords="offset points", xytext=(0, 10), fontsize=8, ha='center')
        ax1.set_xlabel(label)
        ax1.set_ylabel('Best Log-Likelihood (mean over seeds)')
        ax1.set_title(f'Log-Likelihood vs {label}\n(others held at K={baseline_K}, M={baseline_M}, iter={baseline_iter}, opt={baseline_opt})')
        ax1.grid(True, alpha=0.3)

        vals_mae = sorted(val_to_maes.keys())
        means_mae = [np.mean(val_to_maes[v]) for v in vals_mae]
        stds_mae = [np.std(val_to_maes[v]) for v in vals_mae]
        ax2.errorbar(vals_mae, means_mae, yerr=stds_mae, marker='s', capsize=5, color='orange')
        for v, m, n in zip(vals_mae, means_mae, [len(val_to_maes[v]) for v in vals_mae]):
            ax2.annotate(f'n={n}', (v, m), textcoords="offset points", xytext=(0, 10), fontsize=8, ha='center')
        ax2.set_xlabel(label)
        ax2.set_ylabel('Utilization MAE % (mean over seeds)')
        ax2.set_title(f'Utilization MAE vs {label}')
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(os.path.join(plots_dir, f'sensitivity_{label}.png'), dpi=150, bbox_inches='tight')
        plt.close()
        print(f"  Saved sensitivity_{label}.png")


def _plot_best_run_district_fit(results, plots_dir):
    best = max(results, key=lambda x: x.best_log_likelihood or float('-inf'))
    if not best.best_fit_diagnostics:
        return

    diagnostics = sorted(best.best_fit_diagnostics, key=lambda d: d.district)
    districts = [d.district for d in diagnostics]
    x = np.arange(len(districts))

    fig, axes = plt.subplots(2, 2, figsize=(16, 10))

    for ax, metric, title in zip(
        axes.flat,
        ['diff_top3', 'diff_top5', 'diff_top10', 'diff_unmatched'],
        ['Top-3 Diff', 'Top-5 Diff', 'Top-10 Diff', 'Unmatched Diff']
    ):
        vals = [getattr(d, metric) for d in diagnostics]
        colors = ['#e74c3c' if v > 0 else '#2ecc71' for v in vals]
        ax.bar(x, vals, color=colors, alpha=0.7)
        ax.set_xticks(x)
        ax.set_xticklabels(districts, fontsize=7, rotation=45)
        ax.set_xlabel('District')
        ax.set_ylabel('Obs - Sim (%)')
        ax.set_title(f'{title} (mean abs = {np.mean(np.abs(vals)):.1f})')
        ax.axhline(0, color='black', linewidth=0.5)
        ax.grid(True, alpha=0.2)

    plt.suptitle(f'Best Run District Fit: {best.param_str} seed={best.seed}', fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, 'best_run_district_fit.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved best_run_district_fit.png")


def _plot_cross_seed_variance(by_params, plots_dir):
    for param_key, group in sorted(by_params.items()):
        if len(group) < 2:
            continue

        K, M_val, max_iter, max_iter_opt = param_key
        tag = f"K{K}_M{M_val}_iter{max_iter}_opt{max_iter_opt}"

        all_diffs = defaultdict(lambda: {'top3': [], 'top5': [], 'top10': [], 'unmatched': []})
        for r in group:
            for d in r.best_fit_diagnostics:
                all_diffs[d.district]['top3'].append(abs(d.diff_top3))
                all_diffs[d.district]['top5'].append(abs(d.diff_top5))
                all_diffs[d.district]['top10'].append(abs(d.diff_top10))
                all_diffs[d.district]['unmatched'].append(abs(d.diff_unmatched))

        if not all_diffs:
            continue

        districts = sorted(all_diffs.keys())
        x = np.arange(len(districts))

        fig, ax = plt.subplots(figsize=(14, 6))
        width = 0.2
        for i, (metric, color) in enumerate([('top3', '#3498db'), ('top5', '#e74c3c'), ('top10', '#2ecc71'), ('unmatched', '#9b59b6')]):
            means = [np.mean(all_diffs[d][metric]) for d in districts]
            stds = [np.std(all_diffs[d][metric]) for d in districts]
            ax.bar(x + i * width, means, width, yerr=stds, label=metric, color=color, alpha=0.7, capsize=2)

        ax.set_xticks(x + 1.5 * width)
        ax.set_xticklabels(districts, fontsize=7, rotation=45)
        ax.set_xlabel('District')
        ax.set_ylabel('|Obs - Sim| (%)')
        ax.set_title(f'Cross-Seed Match Stat Error: {tag} ({len(group)} seeds)')
        ax.legend()
        ax.grid(True, alpha=0.2)

        plt.tight_layout()
        plt.savefig(os.path.join(plots_dir, f'cross_seed_{tag}.png'), dpi=150, bbox_inches='tight')
        plt.close()
        print(f"  Saved cross_seed_{tag}.png")


def _plot_phi_comparison(by_params, plots_dir):
    """For each param config with multiple seeds, plot phi values across seeds."""
    for param_key, group in sorted(by_params.items()):
        if len(group) < 2:
            continue

        K, M_val, max_iter, max_iter_opt = param_key
        tag = f"K{K}_M{M_val}_iter{max_iter}_opt{max_iter_opt}"

        all_phis = [r.final_phis for r in group if r.final_phis]
        if not all_phis:
            continue

        max_k = max(len(p) for p in all_phis)
        fig, ax = plt.subplots(figsize=(10, 6))
        x = np.arange(1, max_k + 1)

        phi_by_component = []
        for k_idx in range(max_k):
            phi_by_component.append([p[k_idx] for p in all_phis if len(p) > k_idx])

        bp = ax.boxplot(phi_by_component, positions=x, patch_artist=True, widths=0.4)
        for patch in bp['boxes']:
            patch.set_facecolor('lightblue')
            patch.set_alpha(0.6)

        ax.set_xticks(x)
        ax.set_xticklabels([f'phi[{i}]' for i in x])
        ax.set_xlabel('Mixture Component')
        ax.set_ylabel('Phi Value')
        ax.set_ylim(0, 1.05)
        ax.set_title(f'Phi Estimates Across Seeds: {tag} ({len(group)} seeds)')
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(plots_dir, f'phi_comparison_{tag}.png'), dpi=150, bbox_inches='tight')
        plt.close()
        print(f"  Saved phi_comparison_{tag}.png")


def _plot_central_ranking_comparison(by_params, plots_dir, top_n=10):
    """For each param config with multiple seeds, compare top-N central rankings across seeds.
    
    For each district, show a heatmap of how consistently each school appears in the top N.
    """
    for param_key, group in sorted(by_params.items()):
        if len(group) < 2:
            continue

        K, M_val, max_iter, max_iter_opt = param_key
        tag = f"K{K}_M{M_val}_iter{max_iter}_opt{max_iter_opt}"

        # Collect all districts
        all_districts = set()
        for r in group:
            all_districts.update(r.central_rankings.keys())
        all_districts = sorted(all_districts)

        if not all_districts:
            continue

        # For each district, count how often each school appears in top N
        n_seeds = len(group)

        # Summary plot: for each district, what fraction of seeds agree on top-1, top-3, top-5?
        agreement_top1 = []
        agreement_top3 = []
        agreement_top5 = []

        for district in all_districts:
            top1_schools = []
            top3_sets = []
            top5_sets = []
            for r in group:
                if district in r.central_rankings:
                    sigma = r.central_rankings[district]
                    if len(sigma) >= 1:
                        top1_schools.append(sigma[0])
                    if len(sigma) >= 3:
                        top3_sets.append(set(sigma[:3]))
                    if len(sigma) >= 5:
                        top5_sets.append(set(sigma[:5]))

            # Top-1 agreement: fraction of seeds that agree on the most common #1
            if top1_schools:
                from collections import Counter
                most_common_count = Counter(top1_schools).most_common(1)[0][1]
                agreement_top1.append(most_common_count / len(top1_schools))
            else:
                agreement_top1.append(0)

            # Top-3 agreement: average pairwise Jaccard similarity
            if len(top3_sets) >= 2:
                jaccards = []
                for i in range(len(top3_sets)):
                    for j in range(i + 1, len(top3_sets)):
                        inter = len(top3_sets[i] & top3_sets[j])
                        union = len(top3_sets[i] | top3_sets[j])
                        jaccards.append(inter / union if union > 0 else 0)
                agreement_top3.append(np.mean(jaccards))
            else:
                agreement_top3.append(1.0)

            # Top-5 agreement
            if len(top5_sets) >= 2:
                jaccards = []
                for i in range(len(top5_sets)):
                    for j in range(i + 1, len(top5_sets)):
                        inter = len(top5_sets[i] & top5_sets[j])
                        union = len(top5_sets[i] | top5_sets[j])
                        jaccards.append(inter / union if union > 0 else 0)
                agreement_top5.append(np.mean(jaccards))
            else:
                agreement_top5.append(1.0)

        # Plot agreement across districts
        fig, ax = plt.subplots(figsize=(14, 6))
        x = np.arange(len(all_districts))
        width = 0.25
        ax.bar(x - width, agreement_top1, width, label='Top-1 agreement', color='#e74c3c', alpha=0.7)
        ax.bar(x, agreement_top3, width, label='Top-3 Jaccard', color='#3498db', alpha=0.7)
        ax.bar(x + width, agreement_top5, width, label='Top-5 Jaccard', color='#2ecc71', alpha=0.7)

        ax.set_xticks(x)
        ax.set_xticklabels(all_districts, fontsize=7, rotation=45)
        ax.set_xlabel('District')
        ax.set_ylabel('Agreement (1.0 = identical across seeds)')
        ax.set_ylim(0, 1.1)
        ax.set_title(f'Central Ranking Stability Across Seeds: {tag} ({n_seeds} seeds)')
        ax.legend()
        ax.grid(True, alpha=0.2)

        plt.tight_layout()
        plt.savefig(os.path.join(plots_dir, f'sigma_stability_{tag}.png'), dpi=150, bbox_inches='tight')
        plt.close()
        print(f"  Saved sigma_stability_{tag}.png")

        # Detailed heatmap: for a subset of districts, show which schools appear in top N
        # Pick 6 districts spread across the range
        sample_districts = all_districts[::max(1, len(all_districts) // 6)][:6]

        fig, axes = plt.subplots(2, 3, figsize=(20, 12))
        for ax, district in zip(axes.flat, sample_districts):
            # Count school appearances in top N across seeds
            school_counts = defaultdict(int)
            for r in group:
                if district in r.central_rankings:
                    for s in r.central_rankings[district][:top_n]:
                        school_counts[s] += 1

            if not school_counts:
                ax.set_visible(False)
                continue

            # Sort by count descending, take top 15 for readability
            sorted_schools = sorted(school_counts.items(), key=lambda x: x[1], reverse=True)[:15]
            schools = [s[0] for s in sorted_schools]
            counts = [s[1] for s in sorted_schools]

            colors = ['#27ae60' if c == n_seeds else '#3498db' if c >= n_seeds * 0.5 else '#e74c3c' for c in counts]
            ax.barh(range(len(schools)), counts, color=colors, alpha=0.7)
            ax.set_yticks(range(len(schools)))
            ax.set_yticklabels(schools, fontsize=7, fontfamily='monospace')
            ax.invert_yaxis()
            ax.set_xlabel(f'Appearances in top {top_n} (out of {n_seeds} seeds)')
            ax.set_title(f'District {district}', fontweight='bold')
            ax.axvline(n_seeds, color='gray', linestyle='--', alpha=0.5)

        # Hide unused subplots
        for ax in axes.flat[len(sample_districts):]:
            ax.set_visible(False)

        plt.suptitle(f'Top-{top_n} Central Ranking Consistency: {tag}', fontweight='bold', fontsize=14)
        plt.tight_layout()
        plt.savefig(os.path.join(plots_dir, f'sigma_detail_{tag}.png'), dpi=150, bbox_inches='tight')
        plt.close()
        print(f"  Saved sigma_detail_{tag}.png")

def compare_synthetic_to_real_rankings(syn_csv_path, indv_df_path, plots_dir=None, report_path=None):
    """
    Compare synthetic rankings (from best EM run) to real individual-level rankings.
    
    Args:
        syn_csv_path: Path to synthetic rankings CSV (student_id, district, choice_1..choice_10)
        indv_df_path: Path to individual-level preferences Excel file
        plots_dir: Optional directory to save comparison plots
        report_path: Optional path to write comparison report
    """
    
    syn_df = pd.read_csv(syn_csv_path, dtype=str)
    indv_df = pd.read_excel(indv_df_path)
    indv_df['rbd'] = indv_df['rbd'].astype(str)
    indv_df['Region'] = indv_df['Region'].astype(str)
    

    regions = sorted(syn_df['district'].unique())
    
    lines = []
    w = lines.append
    w("=" * 90)
    w("RANKING COMPARISON: Synthetic vs Real")
    w("=" * 90)
    
    all_overlaps = []
    all_taus = []
    
    for region in regions:
        real_region = indv_df[indv_df['Region'] == region]
        real_top1 = Counter(real_region[real_region['preference_number'] == 1]['rbd'].astype(str))
        real_any = Counter(real_region['rbd'].astype(str))
        
        syn_region = syn_df[syn_df['district'] == region]
        syn_top1 = Counter(syn_region['choice_1'].dropna())
        syn_any = Counter()
        for col in [f'choice_{i}' for i in range(1, 11)]:
            if col in syn_region.columns:
                syn_any.update(syn_region[col].dropna())
        
        real_top10_schools = [s for s, _ in real_top1.most_common(10)]
        syn_top10_schools = [s for s, _ in syn_top1.most_common(10)]
        overlap = len(set(real_top10_schools) & set(syn_top10_schools))
        all_overlaps.append(overlap)
        
        shared = set(real_any.keys()) & set(syn_any.keys())
        tau = None
        if len(shared) > 5:
            real_rank = {s: i for i, (s, _) in enumerate(real_any.most_common())}
            syn_rank = {s: i for i, (s, _) in enumerate(syn_any.most_common())}
            shared_list = sorted(shared)
            r = [real_rank[s] for s in shared_list]
            s = [syn_rank[s] for s in shared_list]
            tau, _ = kendalltau(r, s)
            all_taus.append(tau)
        
        real_top1_set = set(real_top10_schools[:5])
        syn_top1_set = set(syn_top10_schools[:5])
        
        tau_str = f"{tau:.3f}" if tau is not None else "N/A"
        w(f"\n  {region}:")
        w(f"    Top-10 first-choice overlap: {overlap}/10")
        w(f"    Kendall tau (all shared schools): {tau_str}")
        w(f"    Real top 5 (by #1 choice):  {real_top10_schools[:5]}")
        w(f"    Synth top 5 (by #1 choice): {syn_top10_schools[:5]}")
    
    w(f"\n{'='*90}")
    w(f"SUMMARY")
    w(f"{'='*90}")
    w(f"  Mean top-10 overlap: {np.mean(all_overlaps):.1f}/10")
    w(f"  Median top-10 overlap: {np.median(all_overlaps):.0f}/10")
    if all_taus:
        w(f"  Mean Kendall tau: {np.mean(all_taus):.3f}")
        w(f"  Median Kendall tau: {np.median(all_taus):.3f}")
    
    report_text = '\n'.join(lines)
    print(report_text)
    
    if report_path:
        with open(report_path, 'w') as f:
            f.write(report_text)
        print(f"\nComparison report written to {report_path}")
    
    if plots_dir:
        os.makedirs(plots_dir, exist_ok=True)
 
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
        
        x = np.arange(len(regions))
        ax1.bar(x, all_overlaps, color='#3498db', alpha=0.7)
        ax1.set_xticks(x)
        ax1.set_xticklabels([r[:15] for r in regions], fontsize=7, rotation=45, ha='right')
        ax1.set_ylabel('Overlap (out of 10)')
        ax1.set_title('Top-10 First-Choice School Overlap per Region')
        ax1.axhline(np.mean(all_overlaps), color='red', linestyle='--', alpha=0.7, label=f'Mean={np.mean(all_overlaps):.1f}')
        ax1.legend()
        ax1.grid(True, alpha=0.2)
        
        if all_taus:
            ax2.bar(x[:len(all_taus)], all_taus, color='#e67e22', alpha=0.7)
            ax2.set_xticks(x[:len(all_taus)])
            ax2.set_xticklabels([r[:15] for r in regions[:len(all_taus)]], fontsize=7, rotation=45, ha='right')
            ax2.set_ylabel('Kendall Tau')
            ax2.set_title('Rank Correlation (All Shared Schools)')
            ax2.axhline(np.mean(all_taus), color='red', linestyle='--', alpha=0.7, label=f'Mean={np.mean(all_taus):.3f}')
            ax2.axhline(0, color='black', linewidth=0.5)
            ax2.legend()
            ax2.grid(True, alpha=0.2)
        
        plt.tight_layout()
        plt.savefig(os.path.join(plots_dir, 'ranking_comparison.png'), dpi=150, bbox_inches='tight')
        plt.close()
        print(f"  Saved ranking_comparison.png")
    
    return all_overlaps, all_taus


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Parse EduRanker experiment logs and generate reports')
    parser.add_argument('--results-dir', type=str, required=True, help='Path to experiment-results directory')
    parser.add_argument('--min-date', type=str, default='20260408', help='Minimum date in YYYYMMDD format')
    parser.add_argument('--report-out', type=str, default=None, help='Path to write text report')
    parser.add_argument('--plots-dir', type=str, default=None, help='Directory to save plots')
    parser.add_argument('--compare-rankings', action='store_true', help='Compare synthetic vs real rankings')
    parser.add_argument('--real-indv', type=str, default=None, help='Path to individual-level preferences Excel (for ranking comparison)')
    args = parser.parse_args()

    if args.compare_rankings:
        syn_csvs = sorted([f for f in os.listdir(args.results_dir) if f.endswith('_synthetic_rankings.csv')])
        if not syn_csvs:
            print("Error: no synthetic rankings CSV found in results directory")
            exit(1)
        if not args.real_indv:
            print("Error: --real-indv required for ranking comparison")
            exit(1)
        syn_csv = os.path.join(args.results_dir, syn_csvs[-1])
        
        print(f"Comparing: {syn_csv}")
        print(f"Against:   {args.real_indv}")
        compare_synthetic_to_real_rankings(
            syn_csv, args.real_indv,
            plots_dir=args.plots_dir,
            report_path=args.report_out
        )
    else:
        results = find_and_parse_logs(args.results_dir, min_date=args.min_date)
        if not results:
            print("No completed experiments found.")
            exit(0)
        results = subsample_one_per_imputed_seed(
            results,
            random_seed=TARGET_SUBSAMPLE_RANDOM_SEED,
            param_key=TARGET_SUBSAMPLE_PARAM_KEY,
        )
        generate_report(results, out_path=args.report_out)
        if args.plots_dir:
            print(f"\nGenerating plots...")
            generate_plots(results, plots_dir=args.plots_dir)