import numpy as np

def compute_welfare_queries(student_df):
    df = student_df.copy()

    if 'unmatched' not in df.columns:
        df['unmatched'] = (
            (df['match'] == '-1') |
            (df['match'].isna())
        ).astype(int)

    p_unmatched_given_len = (
        df.groupby('list_length')['unmatched']
        .mean()
        .to_dict()
    )

    return {
        'student_level': df,
        'p_unmatched_given_list_length': p_unmatched_given_len
    }
