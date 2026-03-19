from pathlib import Path
import pandas as pd


def main():
    data_dir = Path("/scratch/rm6609/EduRanker/MatchingInferenceEngine/sample-data/data/chilean_data_processed")
    excel_files = sorted(list(data_dir.glob("*.xlsx")) + list(data_dir.glob("*.xls")))

    if not excel_files:
        print(f"No Excel files found in {data_dir}")
        return

    print(f"Found {len(excel_files)} Excel file(s) in {data_dir}\n")

    for excel_path in excel_files:
        print("=" * 80)
        print(f"File: {excel_path.name}")
        print("-" * 80)
        df = pd.read_excel(excel_path)
        print(f"Total rows: {len(df)}")
        print(f"Total columns: {len(df.columns)}")
        print("Columns:")
        for idx, col in enumerate(df.columns, start=1):
            print(f"  {idx}. {col}")

        print("\nColumn summaries:")
        for col in df.columns:
            series = df[col]
            unique_vals = series.dropna().unique()
            unique_count = len(unique_vals)
            print(f"  - {col}")
            print(f"    Unique (non-null): {unique_count}")

            if unique_count < 20:
                values_display = ", ".join(str(v) for v in sorted(unique_vals, key=str))
                print(f"    Values: {values_display}")
            elif pd.api.types.is_numeric_dtype(series):
                non_null = series.dropna()
                if len(non_null) > 0:
                    print(f"    Min: {non_null.min()}")
                    print(f"    Max: {non_null.max()}")
                    print(f"    Median: {non_null.median()}")
                else:
                    print("    Min/Max/Median: n/a (all values null)")

        print("Head:")
        print(df.head())
        print()


main()