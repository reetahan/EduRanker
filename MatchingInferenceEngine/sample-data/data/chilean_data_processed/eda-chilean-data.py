import pandas as pd
from pathlib import Path


def _find_column(df, aliases):
    normalized = {c.lower().strip().replace(" ", "_"): c for c in df.columns}
    for alias in aliases:
        key = alias.lower().strip().replace(" ", "_")
        if key in normalized:
            return normalized[key]
    return None

def main():

    data_dir = Path("/scratch/rm6609/EduRanker/MatchingInferenceEngine/sample-data/data/chilean_data_processed")
    excel_files = sorted(list(data_dir.glob("*.xlsx")) + list(data_dir.glob("*.xls")))

    if not excel_files:
        print(f"No Excel files found in {data_dir}")
        return

    print("="*95)
    print(f"{'CHILEAN MATCHING DATASET ANALYSIS SUMMARY':^95}")
    print("="*95 + "\n")

    first_excel = excel_files[0]
    first_df = pd.read_excel(first_excel)
    print(f"FIRST FILE HEAD PREVIEW (all columns): {first_excel.name}")
    print("-" * 95)
    with pd.option_context(
        "display.max_columns", None,
        "display.width", None,
        "display.max_colwidth", None,
    ):
        print(first_df.head())

    mrun_col = _find_column(first_df, ["mrun", "m_run"])
    pref_col = _find_column(first_df, ["preference_number", "preference", "preference_num", "rank", "choice_rank"])
    matched_col = _find_column(first_df, ["matched_first_round", "matched_first", "first_round_matched"])

    print("\nFIRST FILE MATCHING CHECKS")
    print("-" * 95)
    if mrun_col is None or pref_col is None or matched_col is None:
        print(
            "Could not find required columns for matching checks. "
            f"Detected: mrun={mrun_col}, preference={pref_col}, matched_first_round={matched_col}"
        )
    else:
        tmp = first_df[[mrun_col, pref_col, matched_col]].copy()
        tmp[pref_col] = pd.to_numeric(tmp[pref_col], errors="coerce")
        tmp[matched_col] = pd.to_numeric(tmp[matched_col], errors="coerce").fillna(0)
        tmp["matched_flag"] = (tmp[matched_col] == 1).astype(int)

        # For each mrun, average matched preference among rows where matched_flag == 1.
        # If no row has matched_flag == 1, avg preference is NaN and unmatched_count captures it.
        by_mrun = (
            tmp.groupby(mrun_col)
            .apply(
                lambda g: pd.Series(
                    {
                        "rows": len(g),
                        "matched_rows": int(g["matched_flag"].sum()),
                        "unmatched_rows": int((g["matched_flag"] == 0).sum()),
                        "avg_preference_when_matched": g.loc[g["matched_flag"] == 1, pref_col].mean(),
                    }
                )
            )
            .reset_index()
            .sort_values(mrun_col)
        )

        print("Per-mrun summary (first 20 rows):")
        with pd.option_context("display.max_rows", 20, "display.max_columns", None, "display.width", None):
            print(by_mrun.head(20))

        all_zero_mrun = int((by_mrun["matched_rows"] == 0).sum())
        print(f"\nMRUNs with all matched_first_round=0 (unmatched by this indicator): {all_zero_mrun}")

        pref_summary = (
            tmp.groupby(pref_col)
            .agg(total_rows=("matched_flag", "size"), matched_rows=("matched_flag", "sum"))
            .reset_index()
            .sort_values(pref_col)
        )
        pref_summary["matched_percent"] = 100.0 * pref_summary["matched_rows"] / pref_summary["total_rows"]

        print("\nMatch percent by preference number:")
        with pd.option_context("display.max_columns", None, "display.width", None):
            print(pref_summary)
    print("\n" + "="*95 + "\n")

    for excel_path in excel_files:
        df = pd.read_excel(excel_path)
        
        print(f"FILE: {excel_path.name}")
        print(f"DIMENSIONS: {len(df):,} rows x {len(df.columns)} columns")
        print("-" * 95)
        
        # Header for the table
        print(f"{'COLUMN NAME':<35} | {'UNIQUE':<8} | {'DATA SUMMARY / RANGE'}")
        print("-" * 95)
        
        for col in df.columns:
            series = df[col]
            unique_vals = series.dropna().unique()
            u_count = len(unique_vals)
            
            # Logic to create a clean summary string for the third column
            if u_count == 0:
                summary = "All Null Values"
            elif u_count <= 4:
                # Show all values if few
                summary = "Values: " + ", ".join(str(v) for v in sorted(unique_vals, key=str))
            elif pd.api.types.is_numeric_dtype(series):
                # Show range for numeric data
                non_null = series.dropna()
                summary = f"Range: [{non_null.min()} to {non_null.max()}] | Median: {non_null.median()}"
            else:
                # Categorical count
                summary = f"Categorical ({u_count} unique entries)"
            
            print(f"{col:<35} | {u_count:<8} | {summary}")
        
        print("\n" + "="*95 + "\n")

if __name__ == "__main__":
    main()