from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Data containers

@dataclass
class WelfareResults:
    student_level: pd.DataFrame

    global_sweep: pd.DataFrame           
    rank_distribution: pd.DataFrame
    rank_stats: dict[str, float]         
    
    top_p_sweep_by_list_length: pd.DataFrame
    top_p_sweep_by_priority_percentile: pd.DataFrame | None
    top_p_sweep_by_category: dict[str, pd.DataFrame]
    top_p_sweep_by_conjunction: dict[tuple[str, ...], pd.DataFrame]
    saved_paths: dict[str, str] | None = None

# Internal helpers

def _match_rank(ranking: list[int], match_idx: int) -> int | None:
    if match_idx < 0:
        return None
    try:
        return ranking.index(match_idx) + 1
    except ValueError:
        return None


def _resolve_categories(student_df: pd.DataFrame, categories: list[str] | None) -> list[str]:
    if categories is not None:
        return [c for c in categories if c in student_df.columns]
    return [c for c in ["Residential District", "Home Language"] if c in student_df.columns]

def _resolve_priority(
    df: pd.DataFrame,
    priority_col: str | None,
    priority_matrix: np.ndarray | None,
    priority_higher_is_better: bool,
) -> pd.DataFrame:
    df = df.copy()

    if priority_matrix is not None:
        n, m = priority_matrix.shape
        if n != len(df):
            raise ValueError(
                f"priority_matrix has {n} rows but there are {len(df)} students."
            )
        scores = np.full(n, np.nan)
        for i, midx in enumerate(df["match_idx"].to_numpy()):
            if 0 <= midx < m:
                scores[i] = priority_matrix[i, midx]
        df["priority_score"] = scores

    elif priority_col is not None:
        if priority_col not in df.columns:
            raise ValueError(f"Missing priority column: {priority_col}")
        df["priority_score"] = pd.to_numeric(df[priority_col], errors="coerce")

    else:
        return df

    df["priority_percentile"] = (
        df["priority_score"]
        .rank(pct=True, ascending=priority_higher_is_better)
        * 100
    )
    return df

# Student-level build

def build_student_level_welfare(
    rankings_as_indices: list[list[int]],
    matches_idx: list[int] | np.ndarray,
    student_attributes: pd.DataFrame,
    priority_col: str | None = None,
    priority_matrix: np.ndarray | None = None,
    priority_higher_is_better: bool = True,
) -> pd.DataFrame:
   
    n = len(rankings_as_indices)
    if len(matches_idx) != n or len(student_attributes) != n:
        raise ValueError(
            "rankings_as_indices, matches_idx and student_attributes must have the same length"
        )

    df = student_attributes.reset_index(drop=True).copy()
    if "student_id" not in df.columns:
        df.insert(0, "student_id", np.arange(n))

    rankings  = [list(map(int, ranking)) for ranking in rankings_as_indices]
    match_idx = pd.Series(matches_idx).fillna(-1).astype(int).to_numpy()

    df["ranking"]     = rankings
    df["list_length"] = [len(r) for r in rankings]
    df["match_idx"]   = match_idx
    df["match_rank"]  = [_match_rank(r, m) for r, m in zip(rankings, match_idx)]
    df["matched"]     = df["match_rank"].notna()

    df = _resolve_priority(df, priority_col, priority_matrix, priority_higher_is_better)
    return df

# Shared helper: rank stats (avg, variance, std) on a match_rank Series.

def _rank_stats(match_rank_series: pd.Series) -> dict[str, float]:
    matched = match_rank_series.dropna()
    if len(matched) == 0:
        return {"avg_rank": float("nan"), "rank_variance": float("nan"), "rank_std": float("nan")}
    return {
        "avg_rank":      float(matched.mean()),
        "rank_variance": float(matched.var()),
        "rank_std":      float(matched.std()),
    }


def _add_rank_stats_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Round rank stat columns in a grouped summary DataFrame."""
    df = df.copy()
    for col in ("avg_rank", "rank_variance", "rank_std"):
        if col in df.columns:
            df[col] = df[col].round(4)
    return df

# Metric 1: top-p sweep — global unconditional

def _top_p_flag(df: pd.DataFrame, p: int) -> pd.Series:
    return df["match_rank"].le(p).fillna(False)


def summarize_global_sweep(student_df: pd.DataFrame, max_p: int | None = None) -> pd.DataFrame:
    """
    Top-p rate for every p from 1 to max_p, plus overall avg rank and variance.
    avg_rank / rank_variance are the same for every row (they don't depend on p).
    """
    k     = max_p or int(student_df["list_length"].max())
    stats = _rank_stats(student_df["match_rank"])
    rows  = []
    for p in range(1, k + 1):
        rate = _top_p_flag(student_df, p).mean()
        rows.append({"p": p, "top_p_rate": rate, "top_p_pct": 100 * rate, **stats})
    return pd.DataFrame(rows)

# Metric 2 (new): rank distribution

def summarize_rank_distribution(
    student_df: pd.DataFrame,
    max_p: int | None = None,
) -> pd.DataFrame:
    """
    Full distribution of match ranks across all students.
    Returns [match_rank (NaN for unmatched), count, pct].
    """
    k    = max_p or int(student_df["list_length"].max())
    n    = len(student_df)
    rows = []
    for r in range(1, k + 1):
        cnt = int((student_df["match_rank"] == r).sum())
        rows.append({"match_rank": r, "count": cnt, "pct": round(100 * cnt / n, 4)})
    unmatched = int(student_df["match_rank"].isna().sum())
    rows.append({"match_rank": float("nan"), "count": unmatched,
                 "pct": round(100 * unmatched / n, 4)})
    return pd.DataFrame(rows)


def summarize_rank_distribution_by_category(
    student_df: pd.DataFrame,
    categories: list[str] | None = None,
    max_p: int | None = None,
) -> dict[str, pd.DataFrame]:
    """
    Rank distribution conditioned on each single categorical attribute.
    Returns {category_name: DataFrame with [category, match_rank, count, pct]}.
    """
    resolved = _resolve_categories(student_df, categories)
    k        = max_p or int(student_df["list_length"].max())
    out: dict[str, pd.DataFrame] = {}
    for category in resolved:
        frames = []
        for val, grp in student_df.groupby(category, dropna=False):
            dist = summarize_rank_distribution(grp, max_p=k)
            dist.insert(0, category, val)
            frames.append(dist)
        out[category] = pd.concat(frames, ignore_index=True)
    return out

# Metric 3 : average rank + variance

def summarize_rank_stats_overall(student_df: pd.DataFrame) -> dict[str, float]:
    """
    Overall avg rank, rank variance (social inequity), and std across all
    matched students, plus match counts.

    avg_rank      : mean rank of matched school (lower = better)
    rank_variance : variance of rank distribution (higher = more inequity)
    rank_std      : standard deviation (same unit as ranks)
    """
    stats = _rank_stats(student_df["match_rank"])
    return {
        **stats,
        "n_matched":   int(student_df["matched"].sum()),
        "n_total":     len(student_df),
        "pct_matched": round(100 * student_df["matched"].mean(), 4),
    }


def summarize_rank_stats_by_category(
    student_df: pd.DataFrame,
    categories: list[str] | None = None,
) -> dict[str, pd.DataFrame]:
    """
    Avg rank and variance per category value.
    Returns {category: DataFrame with [category, avg_rank, rank_variance,
             rank_std, n_matched, students]}.
    """
    resolved = _resolve_categories(student_df, categories)
    out: dict[str, pd.DataFrame] = {}
    for category in resolved:
        rows = []
        for val, grp in student_df.groupby(category, dropna=False):
            stats = _rank_stats(grp["match_rank"])
            rows.append({
                category:        val,
                **stats,
                "n_matched": int(grp["matched"].sum()),
                "students":  len(grp),
            })
        out[category] = _add_rank_stats_columns(pd.DataFrame(rows))
    return out

# Per-dimension sweeps (top-p + rank stats embedded as columns)

def _grouped_sweep(
    student_df: pd.DataFrame,
    groupby_col: str | list[str],
    max_p: int,
    observed_groups: bool = False,
) -> pd.DataFrame:
    frames = []
    cols   = [groupby_col] if isinstance(groupby_col, str) else groupby_col
    for p in range(1, max_p + 1):
        df = student_df.copy()
        df["top_p"] = _top_p_flag(df, p)
        agg = (
            df.groupby(cols, dropna=False, observed=observed_groups)
            .apply(
                lambda g: pd.Series({
                    "students":   len(g),
                    "top_p_rate": g["top_p"].mean(),
                    **_rank_stats(g["match_rank"]),
                }),
                include_groups=False,
            )
            .reset_index()
        )
        agg.insert(0, "p", p)
        frames.append(agg)
    out = pd.concat(frames, ignore_index=True)
    out["top_p_pct"] = 100 * out["top_p_rate"]
    return _add_rank_stats_columns(out)


def summarize_top_p_sweep_by_list_length(
    student_df: pd.DataFrame, max_p: int | None = None
) -> pd.DataFrame:
    k = max_p or int(student_df["list_length"].max())
    return _grouped_sweep(student_df, "list_length", k).sort_values(["p", "list_length"])


def summarize_top_p_sweep_by_priority_percentile(
    student_df: pd.DataFrame,
    max_p: int | None = None,
    n_bins: int = 10,
) -> pd.DataFrame | None:
    if "priority_percentile" not in student_df.columns:
        return None
    k      = max_p or int(student_df["list_length"].max())
    edges  = np.linspace(0, 100, n_bins + 1)
    labels = [f"{int(edges[i])}-{int(edges[i + 1])}" for i in range(n_bins)]
    df     = student_df.copy()
    df["priority_bin"] = pd.cut(
        df["priority_percentile"].clip(0, 100),
        bins=edges, labels=labels, include_lowest=True,
    )

    out = _grouped_sweep(df, "priority_bin", k, observed_groups=False)
    avg_pct = (
        df.groupby("priority_bin", observed=False)["priority_percentile"]
        .mean()
        .reset_index()
        .rename(columns={"priority_percentile": "avg_priority_percentile"})
    )
    return out.merge(avg_pct, on="priority_bin", how="left")


def summarize_top_p_sweep_by_category(
    student_df: pd.DataFrame,
    categories: list[str] | None = None,
    max_p: int | None = None,
) -> dict[str, pd.DataFrame]:
    resolved = _resolve_categories(student_df, categories)
    k = max_p or int(student_df["list_length"].max())
    return {
        cat: _grouped_sweep(student_df, cat, k).sort_values(["p", "top_p_rate"])
        for cat in resolved
    }


def summarize_top_p_sweep_by_conjunction(
    student_df: pd.DataFrame,
    conjunctions: list[list[str]] | list[tuple[str, ...]],
    max_p: int | None = None,
) -> dict[tuple[str, ...], pd.DataFrame]:
    k = max_p or int(student_df["list_length"].max())
    out: dict[tuple[str, ...], pd.DataFrame] = {}
    for conjunction in conjunctions:
        cols = [c for c in conjunction if c in student_df.columns]
        if cols:
            out[tuple(cols)] = _grouped_sweep(student_df, cols, k)
    return out

# Plotting

def _save_plot(fig: plt.Figure, output_path: str | Path | None, show: bool) -> str | None:
    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, bbox_inches="tight", dpi=200)
    if show:
        plt.show()
    plt.close(fig)
    return None if output_path is None else str(output_path)


def plot_rank_distribution(
    dist: pd.DataFrame,
    output_path: str | Path | None = None,
    show: bool = False,
) -> str | None:
    """Bar chart of the full rank distribution (Metric 2)."""
    matched      = dist.dropna(subset=["match_rank"])
    unmatched_pct = float(dist.loc[dist["match_rank"].isna(), "pct"].iloc[0]) \
                    if dist["match_rank"].isna().any() else 0.0
    fig, ax = plt.subplots(figsize=(max(8, len(matched) * 0.4), 5))
    ax.bar(matched["match_rank"].astype(int), matched["pct"], color="#2980b9")
    ax.set_xlabel("Rank of matched school")
    ax.set_ylabel("Students (%)")
    ax.set_title(f"Rank distribution  (unmatched: {unmatched_pct:.1f}%)")
    ax.grid(True, axis="y", alpha=0.3)
    return _save_plot(fig, output_path, show)


def plot_global_sweep(
    summary: pd.DataFrame,
    output_path: str | Path | None = None,
    show: bool = False,
) -> str | None:
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(summary["p"], summary["top_p_pct"], marker="o")
    ax.set_xlabel("p")
    ax.set_ylabel("Top-p (%)")
    ax.set_title("Global top-p rate (all students)")
    ax.set_ylim(0, 100)
    ax.grid(True, alpha=0.3)
    return _save_plot(fig, output_path, show)


def plot_top_p_sweep_vs_list_length(
    summary: pd.DataFrame,
    p_values: list[int] | None = None,
    output_path: str | Path | None = None,
    show: bool = False,
) -> str | None:
    ps = p_values or sorted(summary["p"].unique())
    fig, ax = plt.subplots(figsize=(9, 5))
    for p in ps:
        sub = summary[summary["p"] == p].sort_values("list_length")
        ax.plot(sub["list_length"], sub["top_p_pct"], marker="o", label=f"top-{p}")
    ax.set_xlabel("List length")
    ax.set_ylabel("Top-p (%)")
    ax.set_title("Top-p by list length")
    ax.set_ylim(0, 100)
    ax.legend()
    ax.grid(True, alpha=0.3)
    return _save_plot(fig, output_path, show)


def plot_avg_rank_by_list_length(
    summary: pd.DataFrame,
    output_path: str | Path | None = None,
    show: bool = False,
) -> str | None:
    """Average rank ± 1 std vs list length (Metric 3)."""
    sub = summary[summary["p"] == summary["p"].min()].sort_values("list_length")
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(sub["list_length"], sub["avg_rank"], marker="o", color="#c0392b")
    ax.fill_between(
        sub["list_length"],
        sub["avg_rank"] - sub["rank_std"],
        sub["avg_rank"] + sub["rank_std"],
        alpha=0.2, color="#c0392b", label="±1 std",
    )
    ax.set_xlabel("List length")
    ax.set_ylabel("Average rank (lower = better)")
    ax.set_title("Average rank by list length")
    ax.legend()
    ax.grid(True, alpha=0.3)
    return _save_plot(fig, output_path, show)


def plot_top_p_sweep_vs_priority_percentile(
    summary: pd.DataFrame,
    p_values: list[int] | None = None,
    output_path: str | Path | None = None,
    show: bool = False,
) -> str | None:
    ps   = p_values or sorted(summary["p"].unique())
    bins = summary["priority_bin"].astype(str).unique()
    x    = np.arange(len(bins))
    fig, ax = plt.subplots(figsize=(9, 5))
    for p in ps:
        sub = summary[summary["p"] == p]
        ax.plot(x, sub["top_p_pct"].values, marker="o", label=f"top-{p}")
    ax.set_xticks(x)
    ax.set_xticklabels(bins, rotation=45, ha="right")
    ax.set_xlabel("Priority percentile bin")
    ax.set_ylabel("Top-p (%)")
    ax.set_title("Top-p by priority percentile")
    ax.set_ylim(0, 100)
    ax.legend()
    ax.grid(True, alpha=0.3)
    return _save_plot(fig, output_path, show)


def plot_top_p_sweep_by_category(
    summary: pd.DataFrame,
    category: str,
    p_values: list[int] | None = None,
    output_path: str | Path | None = None,
    show: bool = False,
) -> str | None:
    ps       = sorted(p_values or summary["p"].unique())
    cat_vals = summary[category].astype(str).unique()
    x        = np.arange(len(cat_vals))
    width    = 0.8 / max(len(ps), 1)
    fig, ax  = plt.subplots(figsize=(max(9, len(cat_vals) * 1.2), 5))
    for i, p in enumerate(ps):
        sub = summary[summary["p"] == p].copy()
        sub[category] = sub[category].astype(str)
        heights = [
            float(sub.loc[sub[category] == v, "top_p_pct"].values[0])
            if v in sub[category].values else 0.0
            for v in cat_vals
        ]
        ax.bar(x + i * width, heights, width, label=f"top-{p}")
    ax.set_xticks(x + width * (len(ps) - 1) / 2)
    ax.set_xticklabels(cat_vals, rotation=45, ha="right")
    ax.set_xlabel(category)
    ax.set_ylabel("Top-p (%)")
    ax.set_title(f"Top-p by {category}")
    ax.set_ylim(0, 100)
    ax.legend()
    ax.grid(True, axis="y", alpha=0.3)
    return _save_plot(fig, output_path, show)


def plot_rank_variance_by_category(
    rank_stats_by_cat: dict[str, pd.DataFrame],
    output_path_prefix: str | Path | None = None,
    show: bool = False,
) -> dict[str, str | None]:
    """Bar chart of rank variance (social inequity, Metric 3) per category."""
    saved: dict[str, str | None] = {}
    for category, df in rank_stats_by_cat.items():
        fig, ax = plt.subplots(figsize=(max(8, len(df) * 0.8), 5))
        ax.bar(df[category].astype(str), df["rank_variance"], color="#8e44ad")
        ax.set_xlabel(category)
        ax.set_ylabel("Rank variance (higher = more inequity)")
        ax.set_title(f"Rank variance by {category}")
        ax.tick_params(axis="x", rotation=45)
        ax.grid(True, axis="y", alpha=0.3)
        path = None
        if output_path_prefix is not None:
            slug = category.lower().replace(" ", "_")
            path = Path(str(output_path_prefix)) / f"rank_variance_by_{slug}.png"
        saved[category] = _save_plot(fig, path, show)
    return saved


def evaluate_simulation_output(
    sim_output: dict[str, Any],
    max_p: int | None = None,
    categories: list[str] | None = None,
    conjunctions: list[list[str]] | None = None,
    priority_col: str | None = None,
    priority_matrix: np.ndarray | None = None,
    priority_higher_is_better: bool = True,
    output_dir: str | Path | None = None,
    n_priority_bins: int = 10,
    plot_p_values: list[int] | None = None,
    show: bool = False,
) -> WelfareResults:

    required_keys = {"rankings_as_indices", "matches_idx", "student_attributes"}
    missing = required_keys - sim_output.keys()
    if missing:
        raise KeyError(f"sim_output is missing keys: {missing}")

    student_df = build_student_level_welfare(
        rankings_as_indices=sim_output["rankings_as_indices"],
        matches_idx=sim_output["matches_idx"],
        student_attributes=sim_output["student_attributes"],
        priority_col=priority_col,
        priority_matrix=priority_matrix,
        priority_higher_is_better=priority_higher_is_better,
    )

    k            = max_p or int(student_df["list_length"].max())
    highlight_ps = plot_p_values or [p for p in [1, 3, 5] if p <= k]

    # Metric 1 — top-p sweep (global)
    global_sweep = summarize_global_sweep(student_df, max_p=k)

    # Metric 2 — rank distribution
    rank_dist        = summarize_rank_distribution(student_df, max_p=k)
    rank_dist_by_cat = summarize_rank_distribution_by_category(
        student_df, categories=categories, max_p=k
    )

    # Metric 3 — average rank + variance
    rank_stats_overall = summarize_rank_stats_overall(student_df)
    rank_stats_by_cat  = summarize_rank_stats_by_category(student_df, categories=categories)

    # Per-dimension sweeps (all metrics embedded as columns)
    by_length     = summarize_top_p_sweep_by_list_length(student_df, max_p=k)
    by_priority   = summarize_top_p_sweep_by_priority_percentile(
        student_df, max_p=k, n_bins=n_priority_bins
    )
    by_category   = summarize_top_p_sweep_by_category(student_df, categories=categories, max_p=k)
    by_conjunction = (
        summarize_top_p_sweep_by_conjunction(student_df, conjunctions, max_p=k)
        if conjunctions else {}
    )

    # Saving
    saved_paths: dict[str, str] = {}
    base_dir = None if output_dir is None else Path(output_dir)

    if base_dir is not None:
        base_dir.mkdir(parents=True, exist_ok=True)

        def _csv(df: pd.DataFrame, name: str) -> str:
            p = base_dir / name
            df.to_csv(p, index=False)
            return str(p)

        saved_paths["student_level"]   = _csv(student_df, "student_level.csv")
        saved_paths["global_sweep"]    = _csv(global_sweep, "global_top_p_sweep.csv")
        saved_paths["rank_distribution"] = _csv(rank_dist, "rank_distribution.csv")
        saved_paths["rank_stats_overall"] = _csv(
            pd.DataFrame([rank_stats_overall]), "rank_stats_overall.csv"
        )

        saved_paths["global_sweep_plot"] = plot_global_sweep(
            global_sweep, base_dir / "global_top_p_sweep.png", show
        )
        saved_paths["rank_distribution_plot"] = plot_rank_distribution(
            rank_dist, base_dir / "rank_distribution.png", show
        )

        for category, dist_df in rank_dist_by_cat.items():
            slug = category.lower().replace(" ", "_")
            saved_paths[f"rank_dist_by_{slug}"] = _csv(
                dist_df, f"rank_distribution_by_{slug}.csv"
            )

        for category, rs_df in rank_stats_by_cat.items():
            slug = category.lower().replace(" ", "_")
            saved_paths[f"rank_stats_by_{slug}"] = _csv(rs_df, f"rank_stats_by_{slug}.csv")

        variance_paths = plot_rank_variance_by_category(
            rank_stats_by_cat, output_path_prefix=base_dir, show=show
        )
        for cat, vp in variance_paths.items():
            if vp:
                saved_paths[f"rank_variance_plot_{cat}"] = vp

        saved_paths["top_p_sweep_by_list_length"] = _csv(
            by_length, "top_p_sweep_by_list_length.csv"
        )
        saved_paths["list_length_plot"] = plot_top_p_sweep_vs_list_length(
            by_length, highlight_ps, base_dir / "top_p_sweep_vs_list_length.png", show
        )
        saved_paths["avg_rank_by_list_length_plot"] = plot_avg_rank_by_list_length(
            by_length, base_dir / "avg_rank_by_list_length.png", show
        )

        if by_priority is not None:
            saved_paths["top_p_sweep_by_priority_percentile"] = _csv(
                by_priority, "top_p_sweep_by_priority_percentile.csv"
            )
            saved_paths["priority_plot"] = plot_top_p_sweep_vs_priority_percentile(
                by_priority, highlight_ps,
                base_dir / "top_p_sweep_vs_priority_percentile.png", show
            )

        for category, summary in by_category.items():
            slug = category.lower().replace(" ", "_")
            saved_paths[f"top_p_sweep_by_{slug}"] = _csv(
                summary, f"top_p_sweep_by_{slug}.csv"
            )
            saved_paths[f"{slug}_plot"] = plot_top_p_sweep_by_category(
                summary, category, highlight_ps,
                base_dir / f"top_p_sweep_by_{slug}.png", show
            )

        for cols, summary in by_conjunction.items():
            slug = "_x_".join(c.lower().replace(" ", "_") for c in cols)
            saved_paths[f"top_p_sweep_by_{slug}"] = _csv(
                summary, f"top_p_sweep_by_{slug}.csv"
            )

    else:
        plot_global_sweep(global_sweep, show=show)
        plot_rank_distribution(rank_dist, show=show)
        plot_top_p_sweep_vs_list_length(by_length, highlight_ps, show=show)
        plot_avg_rank_by_list_length(by_length, show=show)
        if by_priority is not None:
            plot_top_p_sweep_vs_priority_percentile(by_priority, highlight_ps, show=show)
        for category, summary in by_category.items():
            plot_top_p_sweep_by_category(summary, category, highlight_ps, show=show)
        plot_rank_variance_by_category(rank_stats_by_cat, show=show)

    return WelfareResults(
        student_level=student_df,
        global_sweep=global_sweep,
        rank_distribution=rank_dist,
        rank_stats=rank_stats_overall,
        top_p_sweep_by_list_length=by_length,
        top_p_sweep_by_priority_percentile=by_priority,
        top_p_sweep_by_category=by_category,
        top_p_sweep_by_conjunction=by_conjunction,
        saved_paths=saved_paths or None,
    )