import pandas as pd
import numpy as np
from pathlib import Path

def substitute_suppressed_values(df, columns):
    """
    Substitute suppressed values: 's' -> 1, 's^' -> 6
    
    Args:
        df: DataFrame to process
        columns: List of column names to apply substitution
    """
    df = df.copy()
    for col in columns:
        if col in df.columns:
            df[col] = df[col].replace({'s': 1, 's^': 6})
            # Convert to numeric, coercing errors to NaN
            df[col] = pd.to_numeric(df[col], errors='coerce')
    return df


def add_ratio_and_rank(df, true_col, total_col, category_col, ratio_col_name='Ratio', rank_col_name='Rank'):
    """
    Add ratio and rank columns to a dataframe.
    
    Ratio = true / total for each category
    Rank = ranking within each category by ratio (1 = best)
    
    Special handling: Rows with both true=1 and total=1 (from suppressed values)
    are ranked at the bottom of their category since they're artificial.
    
    Args:
        df: DataFrame to process
        true_col: Name of the "true applicants" column
        total_col: Name of the "total applicants" column
        category_col: Name of the category column to group by for ranking
        ratio_col_name: Name for the new ratio column
        rank_col_name: Name for the new rank column
    
    Returns:
        DataFrame with ratio and rank columns added
    """
    df = df.copy()
    
    # Calculate ratio
    df[ratio_col_name] = df[true_col] / df[total_col]
    df[ratio_col_name] = df[ratio_col_name].fillna(0)  # Handle division by zero
    
    # Identify suppressed rows (both true and total are 1)
    df['_is_suppressed'] = (df[true_col] == 1) & (df[total_col] == 1)
    
    # Create a modified ratio for ranking purposes
    # Suppressed rows get -1 so they rank last
    df['_rank_ratio'] = df.apply(
        lambda row: -1 if row['_is_suppressed'] else row[ratio_col_name],
        axis=1
    )
    
    # Rank within each category (ascending=False means higher ratio = better rank)
    # method='min' means ties get the same rank
    df[rank_col_name] = df.groupby(category_col)['_rank_ratio'].rank(
        ascending=False, 
        method='min'
    ).astype(int)
    
    # Clean up temporary columns
    df = df.drop(columns=['_is_suppressed', '_rank_ratio'])
    
    return df


def get_borough_from_zip(zip_code):
    """Map NYC zip codes to boroughs"""
    # NYC zip code ranges by borough
    manhattan = list(range(10001, 10040)) + list(range(10044, 10048)) + [10055, 10060, 10065, 10069, 
                10075, 10103, 10110, 10111, 10112, 10115, 10119, 10128, 10152, 10153, 10154, 10162, 
                10165, 10167, 10168, 10169, 10170, 10171, 10172, 10173, 10174, 10177, 10199, 
                10271, 10278, 10279, 10280, 10282]
    bronx = list(range(10451, 10476)) + [10454, 10455, 10456, 10457, 10458, 10459, 10460, 10461, 
            10462, 10463, 10464, 10465, 10466, 10467, 10468, 10469, 10470, 10471, 10472, 10473, 10474, 10475]
    brooklyn = list(range(11201, 11240)) + [11241, 11242, 11243, 11245, 11247, 11249, 11251, 11252, 11256]
    queens = list(range(11004, 11006)) + list(range(11101, 11107)) + list(range(11351, 11380)) + \
             [11411, 11412, 11413, 11414, 11415, 11416, 11417, 11418, 11419, 11420, 11421, 11422, 
              11423, 11424, 11425, 11426, 11427, 11428, 11429, 11430, 11432, 11433, 11434, 11435, 
              11436, 11691, 11692, 11693, 11694, 11695, 11697]
    staten_island = list(range(10301, 10315))
    
    try:
        zip_int = int(zip_code)
        if zip_int in manhattan:
            return 'Manhattan'
        elif zip_int in bronx:
            return 'Bronx'
        elif zip_int in brooklyn:
            return 'Brooklyn'
        elif zip_int in queens:
            return 'Queens'
        elif zip_int in staten_island:
            return 'Staten Island'
        else:
            return 'Unknown'
    except:
        return 'Unknown'


def create_language_aggregates(data1_path, output_path):
    """
    Create aggregates by home language from DATA1
    Filters rows where Category starts with "Home Language is"
    Only includes high schools (schools with Grade 9 applicants)
    """
    print("Creating language aggregates...")
    
    # Read DATA1 School sheet
    df = pd.read_excel(data1_path, sheet_name='School')
    
    # Filter to only high schools (schools with any Grade 9 data in "All Students")
    df_all = df[df['Category'] == 'All Students'].copy()
    df_all['Grade 9 Total Applicants'] = df_all['Grade 9 Total Applicants'].replace({'s': 1, 's^': 6})
    df_all['Grade 9 Total Applicants'] = pd.to_numeric(df_all['Grade 9 Total Applicants'], errors='coerce')
    high_schools = df_all[df_all['Grade 9 Total Applicants'].notna()]['School DBN'].unique()
    print(f"Filtering to {len(high_schools)} high schools with Grade 9 applicants")
    
    # Filter to only language rows from high schools
    df_lang = df[(df['Category'].str.startswith('Home Language is', na=False)) & 
                 (df['School DBN'].isin(high_schools))].copy()
    print(f"Found {len(df_lang):,} language rows across {df_lang['School DBN'].nunique()} schools")
    
    # Extract language name from Category
    df_lang['Home Language'] = df_lang['Category'].str.replace('Home Language is ', '', regex=False)
    
    # Define columns to process
    numeric_cols = [
        'Grade 9 Total Applicants',
        'Grade 9 True Applicants', 
        'Grade 9 Seats Available',
        'Grade 9 Offers'
    ]
    
    # Substitute suppressed values
    df_lang = substitute_suppressed_values(df_lang, numeric_cols)
    
    # Get school totals from "All Students" rows
    df_totals = df[df['Category'] == 'All Students'][['School DBN', 'Grade 9 Total Applicants', 
                                                        'Grade 9 True Applicants', 'Grade 9 Seats Available', 
                                                        'Grade 9 Offers']].copy()
    df_totals = substitute_suppressed_values(df_totals, numeric_cols)
    df_totals.columns = ['School DBN', 'Total Applicants School', 'Total True Applicants School', 
                         'Seats Available School', 'Offers School']
    
    # Merge school totals
    df_lang = df_lang.merge(df_totals, on='School DBN', how='left')
    
    # Select final columns
    result_df = df_lang[[
        'School DBN', 'School Name', 'School District', 'Home Language',
        'Grade 9 Total Applicants', 'Grade 9 True Applicants',
        'Total Applicants School', 'Total True Applicants School',
        'Seats Available School', 'Offers School'
    ]].rename(columns={
        'Grade 9 Total Applicants': 'Total Applicants Language',
        'Grade 9 True Applicants': 'True Applicants Language'
    })
    
    # Add ratio and rank columns
    result_df = add_ratio_and_rank(
        result_df,
        true_col='True Applicants Language',
        total_col='Total Applicants Language',
        category_col='Home Language',
        ratio_col_name='Ratio',
        rank_col_name='Rank'
    )
    
    result_df.to_csv(output_path, index=False)
    print(f"✓ Saved language aggregates: {output_path} ({len(result_df):,} rows)")
    return result_df


def create_zip_code_aggregates(data1_path, output_path):
    """
    Create aggregates by home zip code from DATA1
    Filters rows where Category starts with "Zip Code"
    Only includes high schools (schools with Grade 9 applicants)
    """
    print("\nCreating zip code aggregates...")
    
    # Read DATA1 School sheet
    df = pd.read_excel(data1_path, sheet_name='School')
    
    # Filter to only high schools (schools with any Grade 9 data in "All Students")
    df_all = df[df['Category'] == 'All Students'].copy()
    df_all['Grade 9 Total Applicants'] = df_all['Grade 9 Total Applicants'].replace({'s': 1, 's^': 6})
    df_all['Grade 9 Total Applicants'] = pd.to_numeric(df_all['Grade 9 Total Applicants'], errors='coerce')
    high_schools = df_all[df_all['Grade 9 Total Applicants'].notna()]['School DBN'].unique()
    print(f"Filtering to {len(high_schools)} high schools with Grade 9 applicants")
    
    # Filter to only zip code rows from high schools
    df_zip = df[(df['Category'].str.startswith('Zip Code', na=False)) & 
                (df['School DBN'].isin(high_schools))].copy()
    print(f"Found {len(df_zip):,} zip code rows across {df_zip['School DBN'].nunique()} schools")
    
    # Extract zip code from Category (e.g., "Zip Code 11213" -> "11213")
    df_zip['Home Zip Code'] = df_zip['Category'].str.replace('Zip Code ', '', regex=False)
    
    # Define columns to process
    numeric_cols = [
        'Grade 9 Total Applicants',
        'Grade 9 True Applicants',
        'Grade 9 Seats Available',
        'Grade 9 Offers'
    ]
    
    # Substitute suppressed values
    df_zip = substitute_suppressed_values(df_zip, numeric_cols)
    
    # Get school totals from "All Students" rows (high schools only)
    df_totals = df[(df['Category'] == 'All Students') & 
                   (df['School DBN'].isin(high_schools))][['School DBN', 'Grade 9 Total Applicants', 
                                                            'Grade 9 True Applicants', 'Grade 9 Seats Available', 
                                                            'Grade 9 Offers']].copy()
    df_totals = substitute_suppressed_values(df_totals, numeric_cols)
    df_totals.columns = ['School DBN', 'Total Applicants School', 'Total True Applicants School', 
                         'Seats Available School', 'Offers School']
    
    # Merge school totals
    df_zip = df_zip.merge(df_totals, on='School DBN', how='left')
    
    # Select final columns
    result_df = df_zip[[
        'School DBN', 'School Name', 'School District', 'Home Zip Code',
        'Grade 9 Total Applicants', 'Grade 9 True Applicants',
        'Total Applicants School', 'Total True Applicants School',
        'Seats Available School', 'Offers School'
    ]].rename(columns={
        'Grade 9 Total Applicants': 'Total Applicants Zip',
        'Grade 9 True Applicants': 'True Applicants Zip'
    })
    
    # Add ratio and rank columns
    result_df = add_ratio_and_rank(
        result_df,
        true_col='True Applicants Zip',
        total_col='Total Applicants Zip',
        category_col='Home Zip Code',
        ratio_col_name='Ratio',
        rank_col_name='Rank'
    )
    
    result_df.to_csv(output_path, index=False)
    print(f"✓ Saved zip code aggregates: {output_path} ({len(result_df):,} rows)")
    return result_df


def create_borough_aggregates(data1_path, output_path):
    """
    Create aggregates by home borough from DATA1
    Borough is derived from zip code rows in the Category column
    Only includes high schools (schools with Grade 9 applicants)
    """
    print("\nCreating borough aggregates...")
    
    # Read DATA1 School sheet
    df = pd.read_excel(data1_path, sheet_name='School')
    
    # Filter to only high schools (schools with any Grade 9 data in "All Students")
    df_all = df[df['Category'] == 'All Students'].copy()
    df_all['Grade 9 Total Applicants'] = df_all['Grade 9 Total Applicants'].replace({'s': 1, 's^': 6})
    df_all['Grade 9 Total Applicants'] = pd.to_numeric(df_all['Grade 9 Total Applicants'], errors='coerce')
    high_schools = df_all[df_all['Grade 9 Total Applicants'].notna()]['School DBN'].unique()
    print(f"Filtering to {len(high_schools)} high schools with Grade 9 applicants")
    
    # Filter to only zip code rows from high schools
    df_zip = df[(df['Category'].str.startswith('Zip Code', na=False)) & 
                (df['School DBN'].isin(high_schools))].copy()
    print(f"Found {len(df_zip):,} zip code rows to derive borough from")
    
    # Extract zip code from Category
    df_zip['Home Zip Code'] = df_zip['Category'].str.replace('Zip Code ', '', regex=False)
    
    # Derive borough from zip code
    df_zip['Home Borough'] = df_zip['Home Zip Code'].apply(get_borough_from_zip)
    
    # Define columns to process
    numeric_cols = [
        'Grade 9 Total Applicants',
        'Grade 9 True Applicants',
        'Grade 9 Seats Available',
        'Grade 9 Offers'
    ]
    
    # Substitute suppressed values
    df_zip = substitute_suppressed_values(df_zip, numeric_cols)
    
    # Aggregate by school and borough
    aggregates = []
    for (school_dbn, school_name, school_district, borough), group in df_zip.groupby(
        ['School DBN', 'School Name', 'School District', 'Home Borough']
    ):
        row = {
            'School DBN': school_dbn,
            'School Name': school_name,
            'School District': school_district,
            'Home Borough': borough,
            'Total Applicants Borough': group['Grade 9 Total Applicants'].sum(),
            'True Applicants Borough': group['Grade 9 True Applicants'].sum(),
        }
        aggregates.append(row)
    
    result_df = pd.DataFrame(aggregates)
    
    # Get school totals from "All Students" rows (high schools only)
    df_totals = df[(df['Category'] == 'All Students') & 
                   (df['School DBN'].isin(high_schools))][['School DBN', 'Grade 9 Total Applicants', 
                                                            'Grade 9 True Applicants', 'Grade 9 Seats Available', 
                                                            'Grade 9 Offers']].copy()
    df_totals = substitute_suppressed_values(df_totals, numeric_cols)
    df_totals.columns = ['School DBN', 'Total Applicants School', 'Total True Applicants School', 
                         'Seats Available School', 'Offers School']
    
    # Merge school totals
    result_df = result_df.merge(df_totals, on='School DBN', how='left')
    
    # Add ratio and rank columns
    result_df = add_ratio_and_rank(
        result_df,
        true_col='True Applicants Borough',
        total_col='Total Applicants Borough',
        category_col='Home Borough',
        ratio_col_name='Ratio',
        rank_col_name='Rank'
    )
    
    result_df.to_csv(output_path, index=False)
    print(f"✓ Saved borough aggregates: {output_path} ({len(result_df):,} rows)")
    return result_df


def convert_master_to_csv(xlsx_path, csv_path):
    """Convert residential_district.xlsx to CSV and add ratio/rank columns"""
    print("\nConverting residential_district.xlsx to CSV...")
    df = pd.read_excel(xlsx_path)
    
    # Substitute suppressed values in the relevant columns
    numeric_cols = ['Total Applicants Residential District', 'True Applicants Residential District']
    df = substitute_suppressed_values(df, numeric_cols)
    
    # Add ratio and rank columns
    df = add_ratio_and_rank(
        df,
        true_col='True Applicants Residential District',
        total_col='Total Applicants Residential District',
        category_col='Residential District',
        ratio_col_name='Ratio',
        rank_col_name='Rank'
    )
    
    df.to_csv(csv_path, index=False)
    print(f"✓ Saved residential_district.csv: {csv_path} ({len(df):,} rows, {len(df.columns)} columns)")
    return df


if __name__ == "__main__":
    # Set up paths
    base_path = Path(__file__).parent
    data1_path = base_path / "raw-data/DATA1_fall-2024-admissions-72-suppressed.xlsx"
    output_dir = base_path / "output"
    output_dir.mkdir(exist_ok=True)
    
    print("="*80)
    print("GENERATING AGGREGATE DATASETS")
    print("="*80)
    
    # 1. Convert master_data to CSV
    master_xlsx = base_path / "residential_district.xlsx"
    master_csv = output_dir / "residential_district.csv"
    convert_master_to_csv(master_xlsx, master_csv)
    
    # 2. Create language aggregates
    language_csv = output_dir / "language_aggregates.csv"
    create_language_aggregates(data1_path, language_csv)
    
    # 3. Create zip code aggregates
    zip_csv = output_dir / "zip_code_aggregates.csv"
    create_zip_code_aggregates(data1_path, zip_csv)
    
    # 4. Create borough aggregates
    borough_csv = output_dir / "borough_aggregates.csv"
    create_borough_aggregates(data1_path, borough_csv)
    
    print("\n" + "="*80)
    print("COMPLETE!")
    print("="*80)
    print(f"\nGenerated files in '{output_dir}/':")
    print("  - master_data.csv")
    print("  - language_aggregates.csv")
    print("  - zip_code_aggregates.csv")
    print("  - borough_aggregates.csv")
