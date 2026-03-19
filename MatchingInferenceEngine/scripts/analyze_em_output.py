#!/usr/bin/env python3
"""Analyze EM experiment logs with repeated simulation and diagnostics blocks.

This parser is designed for logs produced by MatchingInferenceEngine/src/em.py.
It summarizes progress and fit metrics into a human-readable report and optional CSV.
"""

from __future__ import annotations

import argparse
import csv
import math
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from statistics import mean
from typing import Dict, Iterable, List, Optional, Tuple


EM_ITERATION_RE = re.compile(r"^EM ITERATION\s+(\d+)/(\d+)\s*$")
SIMULATION_RE = re.compile(r"^\s*Simulation\s+(\d+)/(\d+)\.\.\.\s*$")
MATCHED_RE = re.compile(r"^\s*Matched:\s*(\d+)/(\d+),\s*Unmatched:\s*(\d+)\s*$")
FIT_HEADER_RE = re.compile(r"^FIT DIAGNOSTICS\s*\|\s*Seed:\s*(\d+)\s*\|\s*Iteration:\s*(\d+)\s*$")
INIT_DISTRICTS_RE = re.compile(r"^Districts:\s*(\d+)\s*$")
INIT_K_RE = re.compile(r"^Global mixture components:\s*K=(\d+)\s*$")
INIT_MAX_ITER_RE = re.compile(r"^Max iterations:\s*(\d+)\s*$")
INIT_M_RE = re.compile(r"^Simulations per evaluation:\s*M=(\d+)\s*$")
COMPUTE_FINAL_LL_RE = re.compile(r"^Computing final log-likelihood at optimized parameters\.\.\.\s*$")
TOTAL_LOG_LIK_RE = re.compile(r"^Total log-likelihood:\s*([+-]?\d+(?:\.\d+)?)\s*$")
MAX_PHI_CHANGE_RE = re.compile(r"^Max phi change:\s*([+-]?\d+(?:\.\d+)?)\s*$")
EM_CONVERGED_RE = re.compile(r"^EM CONVERGED!\s*$")
FINAL_PARAMS_RE = re.compile(r"^Final global parameters:\s*$")
DISTRICT_RE = re.compile(r"^District\s+(\d+):\s*$")
METRIC_LINE_RE = re.compile(
    r"^(Observed|Simulated|Difference):\s*"
    r"top3=\s*([+-]?\d+(?:\.\d+)?)%,\s*"
    r"top5=\s*([+-]?\d+(?:\.\d+)?)%,\s*"
    r"top10=\s*([+-]?\d+(?:\.\d+)?)%,\s*"
    r"unmatched=\s*([+-]?\d+(?:\.\d+)?)%\s*$"
)
UTIL_MAE_RE = re.compile(r"^Mean Absolute Utilization Error:\s*(nan|[+-]?\d+(?:\.\d+)?)%\s*$", re.IGNORECASE)
TOP_MISMATCH_RE = re.compile(
    r"^\s*([A-Za-z0-9]+):\s*Obs=\s*(nan|[+-]?\d+(?:\.\d+)?)%,\s*"
    r"Sim=\s*(nan|[+-]?\d+(?:\.\d+)?)%,\s*Diff=\s*([+-]?nan|[+-]?\d+(?:\.\d+)?)%\s*$",
    re.IGNORECASE,
)


@dataclass
class MatchRecord:
    em_iteration: Optional[int]
    sim_idx: int
    sim_total: int
    matched: int
    total: int
    unmatched: int


@dataclass
class DistrictSnapshot:
    district: int
    top3_diff: float
    top5_diff: float
    top10_diff: float
    unmatched_diff: float


@dataclass
class FitBlock:
    seed: int
    iteration_label: int
    district_snapshots: List[DistrictSnapshot]
    util_mae: Optional[float]
    util_mae_is_nan: bool
    mismatch_nan_lines: int


@dataclass
class ParsedLog:
    em_iteration_headers: List[Tuple[int, int]]
    match_records: List[MatchRecord]
    fit_blocks: List[FitBlock]
    simulations_started: int
    declared_districts: Optional[int]
    declared_k: Optional[int]
    declared_max_iter: Optional[int]
    declared_m: Optional[int]
    compute_final_ll_count: int
    total_log_lik_count: int
    max_phi_change_count: int
    em_converged_count: int
    final_params_count: int


def _safe_float(text: str) -> float:
    if text.lower() in {"nan", "+nan", "-nan"}:
        return float("nan")
    return float(text)


def parse_log(lines: Iterable[str]) -> ParsedLog:
    em_iteration_headers: List[Tuple[int, int]] = []
    match_records: List[MatchRecord] = []
    fit_blocks: List[FitBlock] = []

    current_em_iteration: Optional[int] = None
    pending_sim: Optional[Tuple[int, int]] = None
    simulations_started = 0

    declared_districts: Optional[int] = None
    declared_k: Optional[int] = None
    declared_max_iter: Optional[int] = None
    declared_m: Optional[int] = None

    compute_final_ll_count = 0
    total_log_lik_count = 0
    max_phi_change_count = 0
    em_converged_count = 0
    final_params_count = 0

    in_fit = False
    fit_seed = 0
    fit_iteration_label = 0
    current_district: Optional[int] = None
    district_diffs: Dict[int, Tuple[float, float, float, float]] = {}
    fit_util_mae: Optional[float] = None
    fit_util_mae_is_nan = False
    fit_mismatch_nan_lines = 0

    def flush_fit_block() -> None:
        nonlocal in_fit
        nonlocal district_diffs
        nonlocal fit_util_mae
        nonlocal fit_util_mae_is_nan
        nonlocal fit_mismatch_nan_lines

        if not in_fit:
            return

        snapshots = [
            DistrictSnapshot(
                district=d,
                top3_diff=vals[0],
                top5_diff=vals[1],
                top10_diff=vals[2],
                unmatched_diff=vals[3],
            )
            for d, vals in sorted(district_diffs.items())
        ]

        fit_blocks.append(
            FitBlock(
                seed=fit_seed,
                iteration_label=fit_iteration_label,
                district_snapshots=snapshots,
                util_mae=fit_util_mae,
                util_mae_is_nan=fit_util_mae_is_nan,
                mismatch_nan_lines=fit_mismatch_nan_lines,
            )
        )

        in_fit = False
        district_diffs = {}
        fit_util_mae = None
        fit_util_mae_is_nan = False
        fit_mismatch_nan_lines = 0

    for raw in lines:
        line = raw.rstrip("\n")
        stripped = line.strip()

        m = EM_ITERATION_RE.match(stripped)
        if m:
            flush_fit_block()
            current_em_iteration = int(m.group(1))
            em_iteration_headers.append((int(m.group(1)), int(m.group(2))))
            pending_sim = None
            continue

        m = SIMULATION_RE.match(stripped)
        if m:
            simulations_started += 1
            pending_sim = (int(m.group(1)), int(m.group(2)))
            continue

        m = INIT_DISTRICTS_RE.match(stripped)
        if m:
            declared_districts = int(m.group(1))
            continue

        m = INIT_K_RE.match(stripped)
        if m:
            declared_k = int(m.group(1))
            continue

        m = INIT_MAX_ITER_RE.match(stripped)
        if m:
            declared_max_iter = int(m.group(1))
            continue

        m = INIT_M_RE.match(stripped)
        if m:
            declared_m = int(m.group(1))
            continue

        if COMPUTE_FINAL_LL_RE.match(stripped):
            compute_final_ll_count += 1
            continue

        if TOTAL_LOG_LIK_RE.match(stripped):
            total_log_lik_count += 1
            continue

        if MAX_PHI_CHANGE_RE.match(stripped):
            max_phi_change_count += 1
            continue

        if EM_CONVERGED_RE.match(stripped):
            em_converged_count += 1
            continue

        if FINAL_PARAMS_RE.match(stripped):
            final_params_count += 1
            continue

        m = MATCHED_RE.match(stripped)
        if m and pending_sim is not None:
            sim_idx, sim_total = pending_sim
            match_records.append(
                MatchRecord(
                    em_iteration=current_em_iteration,
                    sim_idx=sim_idx,
                    sim_total=sim_total,
                    matched=int(m.group(1)),
                    total=int(m.group(2)),
                    unmatched=int(m.group(3)),
                )
            )
            pending_sim = None
            continue

        m = FIT_HEADER_RE.match(stripped)
        if m:
            flush_fit_block()
            in_fit = True
            fit_seed = int(m.group(1))
            fit_iteration_label = int(m.group(2))
            current_district = None
            continue

        if in_fit:
            m = DISTRICT_RE.match(stripped)
            if m:
                current_district = int(m.group(1))
                continue

            m = METRIC_LINE_RE.match(stripped)
            if m and current_district is not None and m.group(1) == "Difference":
                district_diffs[current_district] = (
                    float(m.group(2)),
                    float(m.group(3)),
                    float(m.group(4)),
                    float(m.group(5)),
                )
                continue

            m = UTIL_MAE_RE.match(stripped)
            if m:
                value = _safe_float(m.group(1))
                fit_util_mae = value
                fit_util_mae_is_nan = math.isnan(value)
                continue

            m = TOP_MISMATCH_RE.match(stripped)
            if m:
                obs = _safe_float(m.group(2))
                sim = _safe_float(m.group(3))
                diff = _safe_float(m.group(4))
                if math.isnan(obs) or math.isnan(sim) or math.isnan(diff):
                    fit_mismatch_nan_lines += 1
                continue

    flush_fit_block()

    return ParsedLog(
        em_iteration_headers=em_iteration_headers,
        match_records=match_records,
        fit_blocks=fit_blocks,
        simulations_started=simulations_started,
        declared_districts=declared_districts,
        declared_k=declared_k,
        declared_max_iter=declared_max_iter,
        declared_m=declared_m,
        compute_final_ll_count=compute_final_ll_count,
        total_log_lik_count=total_log_lik_count,
        max_phi_change_count=max_phi_change_count,
        em_converged_count=em_converged_count,
        final_params_count=final_params_count,
    )


def summarize(parsed: ParsedLog, phi_optimizer_maxiter: int) -> Dict[str, object]:
    em_header_counter = Counter(iter_idx for iter_idx, _ in parsed.em_iteration_headers)
    fit_label_counter = Counter(block.iteration_label for block in parsed.fit_blocks)

    unmatched_rates = [rec.unmatched / rec.total * 100.0 for rec in parsed.match_records if rec.total > 0]
    matched_rates = [rec.matched / rec.total * 100.0 for rec in parsed.match_records if rec.total > 0]

    all_district_diff: Dict[int, List[Tuple[float, float, float, float]]] = defaultdict(list)
    for block in parsed.fit_blocks:
        for snap in block.district_snapshots:
            all_district_diff[snap.district].append(
                (
                    snap.top3_diff,
                    snap.top5_diff,
                    snap.top10_diff,
                    snap.unmatched_diff,
                )
            )

    district_mean_diff: Dict[int, Dict[str, float]] = {}
    district_mean_abs_diff: Dict[int, Dict[str, float]] = {}
    for district, rows in all_district_diff.items():
        district_mean_diff[district] = {
            "top3": mean(r[0] for r in rows),
            "top5": mean(r[1] for r in rows),
            "top10": mean(r[2] for r in rows),
            "unmatched": mean(r[3] for r in rows),
        }

        district_mean_abs_diff[district] = {
            "top3": mean(abs(r[0]) for r in rows),
            "top5": mean(abs(r[1]) for r in rows),
            "top10": mean(abs(r[2]) for r in rows),
            "unmatched": mean(abs(r[3]) for r in rows),
        }

    util_mae_values = [b.util_mae for b in parsed.fit_blocks if b.util_mae is not None and not b.util_mae_is_nan]
    util_mae_nan_count = sum(1 for b in parsed.fit_blocks if b.util_mae_is_nan)
    mismatch_nan_total = sum(b.mismatch_nan_lines for b in parsed.fit_blocks)

    m_declared = parsed.declared_m
    fit_blocks_completed = len(parsed.fit_blocks)
    match_records_completed = len(parsed.match_records)
    simulations_started = parsed.simulations_started

    sim_completed_in_current_eval: Optional[int] = None
    sim_started_in_current_eval: Optional[int] = None
    if m_declared and m_declared > 0:
        sim_completed_in_current_eval = max(0, match_records_completed - fit_blocks_completed * m_declared)
        sim_started_in_current_eval = max(0, simulations_started - fit_blocks_completed * m_declared)

    max_fit_evals_per_em_iter: Optional[int] = None
    max_fit_evals_total: Optional[int] = None
    if parsed.declared_k is not None:
        max_fit_evals_per_em_iter = parsed.declared_k * phi_optimizer_maxiter + 1
        if parsed.declared_max_iter is not None:
            max_fit_evals_total = parsed.declared_max_iter * max_fit_evals_per_em_iter

    return {
        "em_header_counter": em_header_counter,
        "fit_label_counter": fit_label_counter,
        "n_em_headers": len(parsed.em_iteration_headers),
        "n_simulations": len(parsed.match_records),
        "n_simulations_started": parsed.simulations_started,
        "n_fit_blocks": len(parsed.fit_blocks),
        "declared_districts": parsed.declared_districts,
        "declared_k": parsed.declared_k,
        "declared_max_iter": parsed.declared_max_iter,
        "declared_m": parsed.declared_m,
        "compute_final_ll_count": parsed.compute_final_ll_count,
        "total_log_lik_count": parsed.total_log_lik_count,
        "max_phi_change_count": parsed.max_phi_change_count,
        "em_converged_count": parsed.em_converged_count,
        "final_params_count": parsed.final_params_count,
        "fit_blocks_completed": fit_blocks_completed,
        "sim_completed_in_current_eval": sim_completed_in_current_eval,
        "sim_started_in_current_eval": sim_started_in_current_eval,
        "max_fit_evals_per_em_iter": max_fit_evals_per_em_iter,
        "max_fit_evals_total": max_fit_evals_total,
        "matched_rate_mean": mean(matched_rates) if matched_rates else None,
        "unmatched_rate_mean": mean(unmatched_rates) if unmatched_rates else None,
        "unmatched_rate_max": max(unmatched_rates) if unmatched_rates else None,
        "unmatched_rate_min": min(unmatched_rates) if unmatched_rates else None,
        "district_mean_diff": district_mean_diff,
        "district_mean_abs_diff": district_mean_abs_diff,
        "util_mae_mean": mean(util_mae_values) if util_mae_values else None,
        "util_mae_nan_count": util_mae_nan_count,
        "mismatch_nan_total": mismatch_nan_total,
    }


def write_district_csv(csv_path: Path, district_mean_abs_diff: Dict[int, Dict[str, float]]) -> None:
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["district", "mean_abs_top3_diff", "mean_abs_top5_diff", "mean_abs_top10_diff", "mean_abs_unmatched_diff"])
        for district in sorted(district_mean_abs_diff):
            row = district_mean_abs_diff[district]
            writer.writerow([
                district,
                f"{row['top3']:.4f}",
                f"{row['top5']:.4f}",
                f"{row['top10']:.4f}",
                f"{row['unmatched']:.4f}",
            ])


def _format_counter(counter: Counter) -> str:
    if not counter:
        return "none"
    return ", ".join(f"{k}:{v}" for k, v in sorted(counter.items()))


def print_summary(log_path: Path, summary: Dict[str, object], top_n: int) -> None:
    district_mean_diff = summary["district_mean_diff"]
    district_mean_abs_diff = summary["district_mean_abs_diff"]

    def _fmt(value: Optional[float], decimals: int = 2) -> str:
        if value is None:
            return "n/a"
        return f"{value:.{decimals}f}"

    print(f"Log: {log_path}")
    print("=" * 80)
    print(f"EM headers found: {summary['n_em_headers']} ({_format_counter(summary['em_header_counter'])})")
    print(f"FIT diagnostics blocks: {summary['n_fit_blocks']} ({_format_counter(summary['fit_label_counter'])})")
    print(f"Simulation headers seen: {summary['n_simulations_started']}")
    print(f"Simulation match records: {summary['n_simulations']}")
    print(f"Mean matched rate: {_fmt(summary['matched_rate_mean'])}%")
    print(
        "Unmatched rate (mean/min/max): "
        f"{_fmt(summary['unmatched_rate_mean'])}% / {_fmt(summary['unmatched_rate_min'])}% / {_fmt(summary['unmatched_rate_max'])}%"
    )
    print(f"Utilization MAE mean (non-NaN only): {_fmt(summary['util_mae_mean'])}%")
    print(f"Utilization MAE NaN blocks: {summary['util_mae_nan_count']}")
    print(f"Top-mismatch NaN lines: {summary['mismatch_nan_total']}")

    print("\nProgress diagnostics:")
    print(
        f"Declared setup: districts={summary['declared_districts']}, "
        f"K={summary['declared_k']}, M={summary['declared_m']}, max_iter={summary['declared_max_iter']}"
    )
    print(f"Internal optimizer fit evals completed (FIT blocks): {summary['fit_blocks_completed']}")
    print(f"EM-level final log-likelihood evaluations reached: {summary['compute_final_ll_count']}")
    print(f"EM iterations completed (Total log-likelihood lines): {summary['total_log_lik_count']}")
    print(f"Max-phi-change lines: {summary['max_phi_change_count']}")
    print(f"Converged markers: {summary['em_converged_count']}")
    print(f"Final-parameter blocks: {summary['final_params_count']}")

    if summary["sim_started_in_current_eval"] is not None:
        print(
            "Current fit eval progress (based on M): "
            f"started {summary['sim_started_in_current_eval']}/{summary['declared_m']}, "
            f"completed {summary['sim_completed_in_current_eval']}/{summary['declared_m']} simulations"
        )

    if summary["max_fit_evals_per_em_iter"] is not None:
        print(
            "Theoretical upper bound of fit evals: "
            f"{summary['max_fit_evals_per_em_iter']} per EM iteration"
            + (
                f", {summary['max_fit_evals_total']} total"
                if summary["max_fit_evals_total"] is not None
                else ""
            )
        )

    if summary["total_log_lik_count"] == 0 and summary["n_fit_blocks"] > 0:
        print(
            "Status: still inside optimize_global_mixture objective evaluations; "
            "EM-level post-optimization evaluation has not been reached yet."
        )

    if not district_mean_diff:
        print("No district difference blocks found.")
        return

    print("\nPer-district mean Obs-Sim difference (%):")
    print("district, top3_diff, top5_diff, top10_diff, unmatched_diff, combined_abs")
    for district in sorted(district_mean_diff):
        vals = district_mean_diff[district]
        abs_vals = district_mean_abs_diff[district]
        combined_abs = abs_vals["top3"] + abs_vals["top5"] + abs_vals["top10"] + abs_vals["unmatched"]
        print(
            f"{district}, {vals['top3']:.2f}, {vals['top5']:.2f}, {vals['top10']:.2f}, "
            f"{vals['unmatched']:.2f}, {combined_abs:.2f}"
        )

    district_ranked = sorted(
        district_mean_abs_diff.items(),
        key=lambda kv: kv[1]["top3"] + kv[1]["top5"] + kv[1]["top10"] + kv[1]["unmatched"],
        reverse=True,
    )

    print("\nWorst districts by combined mean absolute difference:")
    print("district, top3, top5, top10, unmatched, combined")
    for district, vals in district_ranked[:top_n]:
        combined = vals["top3"] + vals["top5"] + vals["top10"] + vals["unmatched"]
        print(
            f"{district}, {vals['top3']:.2f}, {vals['top5']:.2f}, {vals['top10']:.2f}, "
            f"{vals['unmatched']:.2f}, {combined:.2f}"
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze repeated EM experiment output logs.")
    parser.add_argument("logfile", type=Path, help="Path to EM output log file")
    parser.add_argument(
        "--district-csv",
        type=Path,
        default=None,
        help="Optional output CSV path for district-level mean absolute differences",
    )
    parser.add_argument(
        "--top-n",
        type=int,
        default=10,
        help="Number of worst districts to print",
    )
    parser.add_argument(
        "--phi-opt-maxiter",
        type=int,
        default=10,
        help="Max iterations used by the 1D bounded optimizer for each phi (default matches em.py)",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    if not args.logfile.exists():
        raise FileNotFoundError(f"Log file not found: {args.logfile}")

    with args.logfile.open("r", encoding="utf-8") as f:
        parsed = parse_log(f)

    summary = summarize(parsed, phi_optimizer_maxiter=args.phi_opt_maxiter)
    print_summary(args.logfile, summary, top_n=args.top_n)

    if args.district_csv is not None:
        write_district_csv(args.district_csv, summary["district_mean_abs_diff"])
        print(f"\nWrote district CSV: {args.district_csv}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
