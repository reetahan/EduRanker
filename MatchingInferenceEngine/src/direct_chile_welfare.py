

import numpy as np
import pandas as pd
from welfare import evaluate_simulation_output

PRIORITY_ATTR_COLS = [
    'priority_student',
    'priority_sibling',
    'priority_parent_civil_servant',
    'priority_ex_student',
    'priority_already_registered',
    'high_performance_student',
    'integration_program_status_existing',
    'female',
]


def load_chile_welfare_inputs(individual_path, capacity_path=None):
    """
    Reads the individual-level Chilean preference data and returns inputs
    ready for welfare.evaluate_simulation_output.

    Args:
        individual_path  path to individual_level_preferences_and_result.xlsx
        capacity_path    optional path to school_capacity.xlsx — required if
                         you want to run a counterfactual STB DA

    Returns:
        sim_output:   dict with keys rankings_as_indices, matches_idx,
                      student_attributes
        student_attrs: pd.DataFrame with student-level attributes
        school_info:  dict with keys all_rbds, school_to_idx, capacities
                      (capacities is None if capacity_path not provided)
    """
    df = pd.read_excel(individual_path)

    all_rbds = sorted(df['rbd'].unique())
    school_to_idx = {rbd: i for i, rbd in enumerate(all_rbds)}

    student_order = df['mrun'].unique()
    n_students = len(student_order)
    student_to_idx = {mrun: i for i, mrun in enumerate(student_order)}

    rankings_as_indices = [None] * n_students
    for mrun, grp in df.sort_values(['mrun', 'preference_number']).groupby('mrun', sort=False):
        st_idx = student_to_idx[mrun]
        rankings_as_indices[st_idx] = np.array(
            [school_to_idx[r] for r in grp['rbd'].values],
            dtype=np.int32,
        )

    matched_rows = df[df['matched_first_round'] == 1][['mrun', 'rbd']].drop_duplicates('mrun')
    matches_idx = np.full(n_students, -1, dtype=np.int64)
    for _, row in matched_rows.iterrows():
        matches_idx[student_to_idx[row['mrun']]] = school_to_idx[row['rbd']]

    attr_cols = [c for c in PRIORITY_ATTR_COLS if c in df.columns]
    student_attrs = (
        df.groupby('mrun')[attr_cols + ['Region']]
        .first()
        .reindex(student_order)
        .reset_index()
    )
    student_attrs = student_attrs.rename(columns={
        'mrun':                                  'student_id',
        'Region':                                'district',
        'priority_student':                      'disadvantaged',
        'high_performance_student':              'high_performance',
        'integration_program_status_existing':   'special_needs',
        'priority_sibling':                      'sibling',
        'priority_parent_civil_servant':         'working_parent',
        'priority_ex_student':                   'returning_student',
        'priority_already_registered':           'already_registered',
    })

    capacities = None
    if capacity_path is not None:
        cap_df = pd.read_excel(capacity_path)
        school_caps = cap_df.groupby('rbd')['total_capacity'].sum()
        capacities = np.array(
            [int(school_caps.get(rbd, 0)) for rbd in all_rbds],
            dtype=np.int32,
        )

    matches_valid = (matches_idx >= 0).sum()
    print(f"Loaded {n_students:,} students, {len(all_rbds):,} schools")
    print(f"Matched:   {matches_valid:,} ({100*matches_valid/n_students:.1f}%)")
    print(f"Unmatched: {n_students - matches_valid:,}")

    sim_output = {
        'rankings_as_indices': rankings_as_indices,
        'matches_idx':         matches_idx,
        'student_attributes':  student_attrs,
    }

    school_info = {
        'all_rbds':      all_rbds,
        'school_to_idx': school_to_idx,
        'capacities':    capacities,
    }

    return sim_output, student_attrs, school_info


def run_real_welfare(individual_path, output_dir, categories=None):
    """
    Loads real Chilean data and runs welfare analysis directly.
    """

    if categories is None:
        categories = ['district', 'disadvantaged', 'sibling', 'female']

    sim_output, _, _ = load_chile_welfare_inputs(individual_path)

    results = evaluate_simulation_output(
        sim_output=sim_output,
        categories=categories,
        output_dir=output_dir,
    )

    print(f"avg_rank:    {results.rank_stats['avg_rank']:.3f}")
    print(f"pct_matched: {results.rank_stats['pct_matched']:.1f}%")
    print(f"rank_var:    {results.rank_stats['rank_variance']:.3f}")
    print(f"Outputs written to {output_dir}")

    return results


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--individual", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--categories", nargs="*",
                        default=["district", "disadvantaged", "sibling", "female"])
    args = parser.parse_args()

    run_real_welfare(args.individual, args.output_dir, args.categories)