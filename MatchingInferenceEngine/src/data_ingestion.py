
import pandas as pd
import numpy as np
from util import log_and_print

def preprocess_data(df, match_stats_df, school_info_df, addtl_school_info_df):
    '''
    Fill in your custom preprocessing function here. The function should return the following, at
    the minimum. You may return additional dataframes or have additional columns in your data as needed,
    but this is what is required to run the experiments:

    1) A DataFrame with columns ['School ID', 'School Name', 'School District', 'Residential District', 
         'Total Applicants by {Aggregate}', 'Total Applicants School', '{Metric}', 'Rank (sorted by Metric)']
    2) A DataFrame with columns ['School ID', 'Capacity', 'Utilization'] 
    3) A DataFrame with columns ['Aggregate', 'Total Applicants (for Aggregate)', '% Matches to Choices 1 to k_1', 
                                    '% Matches to Choice 1 to k_2', '% Matches to Choice 1 to k_3',
                                    '% Matches to Choice 1 to k_4']
    '''
    pass

def read_data(file_path, sheet=0, is_first_row_header=False):
    """
    Reads data from the given file path and returns a pandas DataFrame.
    """
    if file_path.endswith('.csv'):
        data = pd.read_csv(file_path)
    else:
        data = pd.read_excel(file_path, sheet_name=sheet)
    
    if is_first_row_header:
        data.columns = data.iloc[0]
        data = data[1:]

    return data

def extract_observed_aggregates(df, match_stats_df):
    """
    Extract observed aggregates for each district
    
    Returns:
        dict mapping district -> observed statistics
    """
    observed = {}
    
    districts = sorted(df['Residential District'].unique())
    
    for district in districts:
        df_d = df[df['Residential District'] == district]
        match_d = match_stats_df[
            match_stats_df['Residential District'] == int(district)
        ].iloc[0]
        
        observed[district] = {
            'match_stats': np.array([
                match_d['% Matches to Choice 1-3'],
                match_d['% Matches to Choice 1-5'],
                match_d['% Matches to Choice 1-10'],
                match_d['Unmatched']
            ]),
            'total_app': df_d['Total Applicants by Residential District'].values,
            'true_app': df_d['True Applicants by Residential District'].values,
            'schools': df_d['School DBN'].values
        }
    
    return observed

def nyc_preprocess_data(df, match_stats_df, school_info_df, addtl_school_info_df):

    prog_rows = []
    for _, row in school_info_df.iterrows():
        dbn = str(row['dbn']).strip()
        for i in range(1, 12):
            seats_ge  = int(float(row[f'seats9ge{i}'])) if pd.notnull(row.get(f'seats9ge{i}')) else 0
            seats_swd = int(float(row[f'seats9swd{i}'])) if pd.notnull(row.get(f'seats9swd{i}')) else 0
            ge_apps = int(float(row[f'grade9geapplicants{i}'])) if pd.notnull(row.get(f'grade9geapplicants{i}')) else 0
            swd_apps = int(float(row[f'grade9swdapplicants{i}'])) if pd.notnull(row.get(f'grade9swdapplicants{i}')) else 0

            total_cap = seats_ge + seats_swd
            if total_cap == 0 or (ge_apps + swd_apps) == 0:
                continue
            
            prog_rows.append({
                'School DBN': f"{dbn}_prog{i}",
                'dbn': dbn,
                'Capacity': total_cap,
                'seats_ge': seats_ge,
                'seats_swd': seats_swd,
                'total_applicants': ge_apps + swd_apps,
            })

    prog_info_df = pd.DataFrame(prog_rows)
   
    school_total_apps = prog_info_df.groupby('dbn')['total_applicants'].sum().rename('school_total_apps')
    prog_info_df = prog_info_df.join(school_total_apps, on='dbn')
    prog_info_df['cap_share'] = prog_info_df['total_applicants'] / prog_info_df['school_total_apps'].replace(0, 1)

    addtl_school_info_df = addtl_school_info_df[(addtl_school_info_df['Category'] == 'All Students') & (pd.to_numeric(addtl_school_info_df['Grade 9 Students'], errors='coerce').notna())]
    addtl_school_info_df = addtl_school_info_df[['School DBN', 'Grade 9 Students']]
    addtl_school_info_df['Grade 9 Students'] = addtl_school_info_df['Grade 9 Students'].astype(int)
    school_util = addtl_school_info_df.set_index('School DBN')['Grade 9 Students'].to_dict()
    prog_info_df['school_grade9'] = prog_info_df['dbn'].map(school_util).fillna(0)
    prog_info_df['prog_grade9'] = prog_info_df['school_grade9'] * prog_info_df['cap_share']
    prog_info_df['Utilization'] = (prog_info_df['prog_grade9'] / prog_info_df['Capacity'] * 100).clip(upper=100)
    school_info_df = prog_info_df[['School DBN', 'Capacity', 'seats_ge', 'seats_swd', 'Utilization']].copy()

    df = df[df['Residential District'] != 'Unknown']
    prog_df_rows = []
    for _, prog_row in prog_info_df.iterrows():
        share = prog_row['cap_share']
        for _, s_row in df[df['School DBN'] == prog_row['dbn']].iterrows():
            prog_df_rows.append({
                'School DBN': prog_row['School DBN'],
                'School Name': s_row['School Name'],
                'School District': s_row['School District'],
                'Residential District': s_row['Residential District'],
                'Total Applicants by Residential District': max(1, round(s_row['Total Applicants by Residential District'] * share)),
                'True Applicants by Residential District': max(0, round(s_row['True Applicants by Residential District'] * share)),
                'Total Applicants School': round(s_row['Total Applicants School'] * share),
                'Total True Applicants School': round(s_row['Total True Applicants School'] * share),
            })
    df = pd.DataFrame(prog_df_rows)
    df['Ratio'] = (df['True Applicants by Residential District'] ** 2) / df['Total Applicants by Residential District'].replace(0, 1)
    df['Rank'] = df.groupby('Residential District')['Ratio'].rank(ascending=False, method='first')
    for col in ['Total Applicants by Residential District', 'True Applicants by Residential District',
                'Total Applicants School', 'Total True Applicants School']:
        df[col] = df[col].astype(int)

    match_stats_df = match_stats_df[~match_stats_df['Residential District'].isin(['Total', 'Unknown', 'Unknown '])]
    match_stats_df = match_stats_df[['Residential District', 'Total Applicants',
                                    '# Matches to Choice 1-3', '# Matches to Choice 1-5',
                                    '# Matches to Choice 1-10', '# Matches to Choice 1-12']]

    match_stats_df['Residential District'] = pd.to_numeric(match_stats_df['Residential District'], errors='coerce')
    match_stats_df['Total Applicants'] = match_stats_df['Total Applicants'].astype(str).str.replace(',', '').str.strip()
    match_stats_df['Total Applicants'] = pd.to_numeric(match_stats_df['Total Applicants'], errors='coerce')

    for col in ['# Matches to Choice 1-3', '# Matches to Choice 1-5',
                '# Matches to Choice 1-10', '# Matches to Choice 1-12']:
        match_stats_df[col] = match_stats_df[col].astype(str).str.replace(',', '').str.strip()
        match_stats_df[col] = pd.to_numeric(match_stats_df[col], errors='coerce')
        match_stats_df[col] = match_stats_df[col] / match_stats_df['Total Applicants'] * 100

    match_stats_df['Unmatched'] = 100.0 - match_stats_df['# Matches to Choice 1-12']
    match_stats_df = match_stats_df.drop(columns=['# Matches to Choice 1-12'])
    match_stats_df = match_stats_df.rename(columns={
        '# Matches to Choice 1-3':  '% Matches to Choice 1-3',
        '# Matches to Choice 1-5':  '% Matches to Choice 1-5',
        '# Matches to Choice 1-10': '% Matches to Choice 1-10',
    })
    match_stats_df = match_stats_df.dropna(subset=['Residential District', 'Total Applicants'])
    match_stats_df['Residential District'] = match_stats_df['Residential District'].astype(int)

    avg_list_length = df['Total Applicants by Residential District'].sum() / match_stats_df['Total Applicants'].sum()
    log_and_print(f"Average list length from data: {avg_list_length:.2f}")

    return df, match_stats_df, school_info_df

def preprocess_chilean_data(indv_df, match_df, school_cap_reg_df, school_cap_df):


    matched_rows = indv_df[indv_df['matched_first_round'] == 1][['mrun', 'rbd', 'program_code', 'preference_number']].copy()
    matched_rows.rename(columns={'preference_number': 'match_rank'}, inplace=True)

    tot_reg = indv_df.groupby(['Region', 'rbd', 'program_code'])['mrun'].nunique().reset_index()
    tot_reg.rename(columns={'mrun': 'Total Applicants by Residential District'}, inplace=True)
    
    merged = pd.merge(indv_df, matched_rows[['mrun', 'match_rank']], on='mrun', how='left')
    merged['match_rank'] = merged['match_rank'].fillna(9999)
    

    true_df = merged[(merged['preference_number'] >= merged['match_rank']) | (merged['match_rank'] == 9999)]
    true_reg = true_df.groupby(['Region', 'rbd', 'program_code'])['mrun'].nunique().reset_index()
    true_reg.rename(columns={'mrun': 'True Applicants by Residential District'}, inplace=True)
    
    tot_sch = indv_df.groupby(['rbd', 'program_code'])['mrun'].nunique().reset_index()
    tot_sch.rename(columns={'mrun': 'Total Applicants School'}, inplace=True)
    
    true_sch = true_df.groupby(['rbd', 'program_code'])['mrun'].nunique().reset_index()
    true_sch.rename(columns={'mrun': 'Total True Applicants School'}, inplace=True)
    
    df = pd.merge(tot_reg, true_reg, on=['Region', 'rbd', 'program_code'], how='left').fillna(0)
    df = pd.merge(df, tot_sch, on=['rbd', 'program_code'], how='left').fillna(0)
    df = pd.merge(df, true_sch, on=['rbd', 'program_code'], how='left').fillna(0)
    
    df['School DBN'] = df['rbd'].astype(str) + '_' + df['program_code'].astype(str)
    df['School Name'] = "School_" + df['rbd'].astype(str) + '_' + df['program_code'].astype(str)
    df['School District'] = df['Region'].astype(str)
    df['Residential District'] = df['Region'].astype(str)
    
    df['Ratio'] = (df['True Applicants by Residential District'] ** 2) / df['Total Applicants by Residential District'].replace(0, 1)
    df['Rank'] = df.groupby('Residential District')['Ratio'].rank(ascending=False, method='first')
    
    df = df[['School DBN', 'School Name', 'School District', 'Residential District', 
             'Total Applicants by Residential District', 'True Applicants by Residential District',
             'Total Applicants School', 'Total True Applicants School', 'Ratio', 'Rank']]
    
    for col in ['Total Applicants by Residential District', 'True Applicants by Residential District', 
                'Total Applicants School', 'Total True Applicants School']:
        df[col] = df[col].astype(int)

    stats = []
    for _, row in match_df.iterrows():
        region = row['Region']
        n_students = row['n_students']
        
        matched_fraction = (100 - row['pct_unmatched']) / 100
        pct_top3 = sum(row[f'pct_top{i}'] for i in range(1, 4)) * matched_fraction
        pct_top5 = sum(row[f'pct_top{i}'] for i in range(1, 6)) * matched_fraction
        pct_top10 = sum(row[f'pct_top{i}'] for i in range(1, 11)) * matched_fraction
        
        stats.append({
            'Residential District': str(region),
            'Total Applicants': int(n_students),
            '% Matches to Choice 1-3': pct_top3,
            '% Matches to Choice 1-5': pct_top5,
            '% Matches to Choice 1-10': pct_top10,
            'Unmatched': row['pct_unmatched'],
        })
        
    new_match_stats_df = pd.DataFrame(stats)
    
    school_caps = school_cap_df.groupby(['rbd', 'program_code'])['total_capacity'].sum().reset_index()
    school_caps['School DBN'] = school_caps['rbd'].astype(str) + '_' + school_caps['program_code'].astype(str)
    school_caps = school_caps[['School DBN', 'total_capacity']].rename(columns={'total_capacity': 'Capacity'})
   
    admitted = school_cap_reg_df.groupby(['rbd', 'program_code'])['n_admitted'].sum().reset_index()
    admitted['School DBN'] = admitted['rbd'].astype(str) + '_' + admitted['program_code'].astype(str)
    admitted = admitted[['School DBN', 'n_admitted']].rename(columns={'n_admitted': 'matched_count'})
    
    school_info_df = pd.merge(school_caps, admitted, on='School DBN', how='left')
    school_info_df['matched_count'] = school_info_df['matched_count'].fillna(0)
    school_info_df['Utilization'] = np.where(
        school_info_df['Capacity'] > 0,
        (school_info_df['matched_count'] / school_info_df['Capacity'] * 100).clip(upper=100),
        0.0
    )
    school_info_df = school_info_df[['School DBN', 'Capacity', 'Utilization']]
    
    return df, new_match_stats_df, school_info_df