import pandas as pd
import numpy as np
from pathlib import Path

def substitute_suppressed_values(df, columns):
    df = df.copy()
    for col in columns:
        if col in df.columns:
            df[col] = df[col].replace({'s': 1, 's^': 6})
            df[col] = pd.to_numeric(df[col], errors='coerce')
    return df


def add_ratio_and_rank(df, true_col, total_col, category_col, ratio_col_name='Ratio', rank_col_name='Rank'):
    """Calculate ratio (true^2/total) and rank within categories. Suppressed rows (true=1, total=1) rank last."""
    df = df.copy()
    df[ratio_col_name] = (df[true_col] ** 2) / df[total_col]
    df[ratio_col_name] = df[ratio_col_name].fillna(0)
    
    df['_is_suppressed'] = (df[true_col] == 1) & (df[total_col] == 1)
    df['_rank_ratio'] = df.apply(lambda row: -1 if row['_is_suppressed'] else row[ratio_col_name], axis=1)
    df[rank_col_name] = df.groupby(category_col)['_rank_ratio'].rank(ascending=False, method='min').astype(int)
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


def create_language_aggregates(data1_path, output_dir):
    """Generate separate CSVs for top 7 most common home languages."""
    df = pd.read_excel(data1_path, sheet_name='School')
    
    df_all = df[df['Category'] == 'All Students'].copy()
    df_all['Grade 9 Total Applicants'] = df_all['Grade 9 Total Applicants'].replace({'s': 1, 's^': 6})
    df_all['Grade 9 Total Applicants'] = pd.to_numeric(df_all['Grade 9 Total Applicants'], errors='coerce')
    high_schools = df_all[df_all['Grade 9 Total Applicants'].notna()]['School DBN'].unique()
    
    df_lang = df[(df['Category'].str.startswith('Home Language is', na=False)) & 
                 (df['School DBN'].isin(high_schools))].copy()
    df_lang['Home Language'] = df_lang['Category'].str.replace('Home Language is ', '', regex=False)
    top_languages = df_lang['Home Language'].value_counts().head(7).index.tolist()
    
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
    
    output_dir.mkdir(exist_ok=True)
    for language in top_languages:
        lang_data = result_df[result_df['Home Language'] == language].copy().sort_values('Rank')
        safe_lang_name = language.replace(' ', '_').replace('/', '_').lower()
        lang_data.to_csv(output_dir / f"ranking_language_{safe_lang_name}.csv", index=False)
    
    return result_df


def create_zip_code_aggregates(data1_path, output_path):
    """Generate zip code aggregates."""
    df = pd.read_excel(data1_path, sheet_name='School')
    
    df_all = df[df['Category'] == 'All Students'].copy()
    df_all['Grade 9 Total Applicants'] = df_all['Grade 9 Total Applicants'].replace({'s': 1, 's^': 6})
    df_all['Grade 9 Total Applicants'] = pd.to_numeric(df_all['Grade 9 Total Applicants'], errors='coerce')
    high_schools = df_all[df_all['Grade 9 Total Applicants'].notna()]['School DBN'].unique()
    
    df_zip = df[(df['Category'].str.startswith('Zip Code', na=False)) & 
                (df['School DBN'].isin(high_schools))].copy()
    df_zip['Home Zip Code'] = df_zip['Category'].str.replace('Zip Code ', '', regex=False)
    
    numeric_cols = ['Grade 9 Total Applicants', 'Grade 9 True Applicants', 'Grade 9 Seats Available', 'Grade 9 Offers']
    df_zip = substitute_suppressed_values(df_zip, numeric_cols)
    
    df_totals = df[(df['Category'] == 'All Students') & 
                   (df['School DBN'].isin(high_schools))][['School DBN', 'Grade 9 Total Applicants', 
                                                            'Grade 9 True Applicants', 'Grade 9 Seats Available', 
                                                            'Grade 9 Offers']].copy()
    df_totals = substitute_suppressed_values(df_totals, numeric_cols)
    df_totals.columns = ['School DBN', 'Total Applicants School', 'Total True Applicants School', 
                         'Seats Available School', 'Offers School']
    df_zip = df_zip.merge(df_totals, on='School DBN', how='left')
    
    result_df = df_zip[[
        'School DBN', 'School Name', 'School District', 'Home Zip Code',
        'Grade 9 Total Applicants', 'Grade 9 True Applicants',
        'Total Applicants School', 'Total True Applicants School',
        'Seats Available School', 'Offers School'
    ]].rename(columns={
        'Grade 9 Total Applicants': 'Total Applicants Zip',
        'Grade 9 True Applicants': 'True Applicants Zip'
    })
    
    result_df = add_ratio_and_rank(result_df, 'True Applicants Zip', 'Total Applicants Zip', 'Home Zip Code')
    result_df.to_csv(output_path, index=False)
    return result_df


def create_borough_aggregates(data1_path, output_path):
    """Generate borough aggregates (derived from zip codes)."""
    df = pd.read_excel(data1_path, sheet_name='School')
    
    df_all = df[df['Category'] == 'All Students'].copy()
    df_all['Grade 9 Total Applicants'] = df_all['Grade 9 Total Applicants'].replace({'s': 1, 's^': 6})
    df_all['Grade 9 Total Applicants'] = pd.to_numeric(df_all['Grade 9 Total Applicants'], errors='coerce')
    high_schools = df_all[df_all['Grade 9 Total Applicants'].notna()]['School DBN'].unique()
    
    df_zip = df[(df['Category'].str.startswith('Zip Code', na=False)) & 
                (df['School DBN'].isin(high_schools))].copy()
    df_zip['Home Zip Code'] = df_zip['Category'].str.replace('Zip Code ', '', regex=False)
    df_zip['Home Borough'] = df_zip['Home Zip Code'].apply(get_borough_from_zip)
    
    numeric_cols = ['Grade 9 Total Applicants', 'Grade 9 True Applicants', 'Grade 9 Seats Available', 'Grade 9 Offers']
    df_zip = substitute_suppressed_values(df_zip, numeric_cols)
    
    aggregates = []
    for (school_dbn, school_name, school_district, borough), group in df_zip.groupby(
        ['School DBN', 'School Name', 'School District', 'Home Borough']
    ):
        aggregates.append({
            'School DBN': school_dbn,
            'School Name': school_name,
            'School District': school_district,
            'Home Borough': borough,
            'Total Applicants Borough': group['Grade 9 Total Applicants'].sum(),
            'True Applicants Borough': group['Grade 9 True Applicants'].sum(),
        })
    result_df = pd.DataFrame(aggregates)
    
    df_totals = df[(df['Category'] == 'All Students') & 
                   (df['School DBN'].isin(high_schools))][['School DBN', 'Grade 9 Total Applicants', 
                                                            'Grade 9 True Applicants', 'Grade 9 Seats Available', 
                                                            'Grade 9 Offers']].copy()
    df_totals = substitute_suppressed_values(df_totals, numeric_cols)
    df_totals.columns = ['School DBN', 'Total Applicants School', 'Total True Applicants School', 
                         'Seats Available School', 'Offers School']
    result_df = result_df.merge(df_totals, on='School DBN', how='left')
    result_df = add_ratio_and_rank(result_df, 'True Applicants Borough', 'Total Applicants Borough', 'Home Borough')
    result_df.to_csv(output_path, index=False)
    return result_df


def convert_master_to_csv(xlsx_path, csv_path):
    """Convert residential_district.xlsx to CSV with ratio/rank."""
    df = pd.read_excel(xlsx_path)
    df = substitute_suppressed_values(df, ['Total Applicants Residential District', 'True Applicants Residential District'])
    df = add_ratio_and_rank(df, 'True Applicants Residential District', 'Total Applicants Residential District', 'Residential District')
    df.to_csv(csv_path, index=False)
    return df


if __name__ == "__main__":
    # Set up paths
    base_path = Path(__file__).parent
    data1_path = base_path / "raw-data/DATA1_fall-2024-admissions-72-suppressed.xlsx"
    output_dir = base_path / "output"
    output_dir.mkdir(exist_ok=True)
    
    
    # 1. Convert master_data to CSV
    master_xlsx = base_path / "residential_district.xlsx"
    master_csv = output_dir / "residential_district.csv"
    convert_master_to_csv(master_xlsx, master_csv)
    
    # 2. Create language aggregates (outputs multiple CSVs, one per top 7 language)
    create_language_aggregates(data1_path, output_dir)
    
    # 3. Create zip code aggregates
    zip_csv = output_dir / "zip_code_aggregates.csv"
    create_zip_code_aggregates(data1_path, zip_csv)
    
    # 4. Create borough aggregates
    borough_csv = output_dir / "borough_aggregates.csv"
    create_borough_aggregates(data1_path, borough_csv)
    
    print(f"\nGenerated files in '{output_dir}/':")
