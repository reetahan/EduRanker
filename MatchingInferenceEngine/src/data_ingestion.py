
import pandas as pd
import numpy as np
from util import log_and_print

def read_data(file_path, sheet=0):
    """
    Reads data from the given file path and returns a pandas DataFrame.
    """
    if file_path.endswith('.csv'):
        data = pd.read_csv(file_path)
    else:
        data = pd.read_excel(file_path, sheet_name=sheet)
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
            match_stats_df['Residential District'] == district
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

def preprocess_data(df, match_stats_df, school_info_df, addtl_school_info_df):

    df = df[['School DBN', 'School Name', 'School District', 'Residential District', 
         'Total Applicants by Residential District', 'True Applicants by Residential District',
         'Total Applicants School', 'Total True Applicants School', 'Ratio', 'Rank']]
    df = df[df['Residential District'] != 'Unknown']
    dtype_mapping = {}
    for i in range(len(df.columns.array)):
        if(i > 2):
            dtype_mapping[df.columns.array[i]] = 'int64'
    df = df.astype(dtype_mapping)

    school_cols_sum = [f"seats9ge{i}" for i in range(1,12)] + [f"seats9swd{i}" for i in range(1,12)] 
    school_info_df['Capacity'] = school_info_df.apply(lambda x: sum(x[col] if pd.notnull(x[col]) else 0 for col in school_cols_sum), axis=1)
    
    addtl_school_info_df = addtl_school_info_df[(addtl_school_info_df['Category'] == 'All Students') & (pd.to_numeric(addtl_school_info_df['Grade 9 Students'], errors='coerce').notna())]
    addtl_school_info_df  = addtl_school_info_df[['School DBN', 'Grade 9 Students']]
    addtl_school_info_df['Grade 9 Students'] = addtl_school_info_df['Grade 9 Students'].astype(int)
    school_info_df = school_info_df[['dbn','Capacity']]
    school_info_df = school_info_df.rename(columns={'dbn': 'School DBN'})
    school_info_df = school_info_df[school_info_df['School DBN'].isin(df['School DBN'].unique())]
    school_info_df = addtl_school_info_df.join(school_info_df.set_index('School DBN'), on='School DBN', how='inner')
    school_info_df['Utilization'] = (school_info_df['Grade 9 Students'] / school_info_df['Capacity'] * 100).clip(upper=100)
    school_info_df = school_info_df[['School DBN', 'Capacity', 'Utilization']]

    match_stats_df.columns = match_stats_df.iloc[0]
    match_stats_df = match_stats_df.drop(match_stats_df.index[0])
    match_stats_df = match_stats_df[['Residential District', 'Total Applicants', '% Matches to Choice 1-3', 
                                    '% Matches to Choice 1-5', '% Matches to Choice 1-10', '% Matches to Choice 1-12']]
    dtype_mapping = {}
    for i in range(len(match_stats_df.columns.array)):
        if(i > 0):
            match_stats_df[match_stats_df.columns.array[i]] = match_stats_df[match_stats_df.columns.array[i]].str.replace('%','').str.replace(',','')
            dtype_mapping[match_stats_df.columns.array[i]] = 'float64'
    match_stats_df = match_stats_df.astype(dtype_mapping)
    match_stats_df['Unmatched'] = 100.0 - match_stats_df['% Matches to Choice 1-12'].astype(float)
    match_stats_df = match_stats_df.drop(columns=['% Matches to Choice 1-12'])
    match_stats_df = match_stats_df[~match_stats_df['Residential District'].isin(['Total', 'Unknown '])]
    match_stats_df['Residential District'] = pd.to_numeric(match_stats_df['Residential District'])
    
    avg_list_length = df['Total Applicants by Residential District'].sum() / match_stats_df['Total Applicants'].sum()
    log_and_print(f"Average list length from data: {avg_list_length:.2f}")

    district_to_borough = {
        str(d): b for d, b in
        [(d,'M') for d in range(1,7)] +
        [(d,'X') for d in range(7,13)] +
        [(d,'K') for d in range(13,24)] +
        [(d,'Q') for d in range(24,31)] +
        [(31,'R'), (32,'K')]
    }
     
    return df, match_stats_df, school_info_df, district_to_borough

def preprocess_chilean_data(indv_df, match_df, school_cap_reg_df, school_cap_df):

    matched_rows = indv_df[indv_df['matched_first_round'] == 1][['mrun', 'rbd', 'preference_number']].copy()
    matched_rows.rename(columns={'preference_number': 'match_rank', 'rbd': 'matched_rbd'}, inplace=True)

    tot_reg = indv_df.groupby(['Region', 'rbd'])['mrun'].nunique().reset_index()
    tot_reg.rename(columns={'mrun': 'Total Applicants by Residential District'}, inplace=True)
    
    merged = pd.merge(indv_df, matched_rows[['mrun', 'match_rank']], on='mrun', how='left')
    merged['match_rank'] = merged['match_rank'].fillna(9999)
    

    true_df = merged[(merged['preference_number'] >= merged['match_rank']) | (merged['match_rank'] == 9999)]
    true_reg = true_df.groupby(['Region', 'rbd'])['mrun'].nunique().reset_index()
    true_reg.rename(columns={'mrun': 'True Applicants by Residential District'}, inplace=True)
    
    tot_sch = indv_df.groupby('rbd')['mrun'].nunique().reset_index()
    tot_sch.rename(columns={'mrun': 'Total Applicants School'}, inplace=True)
    
    true_sch = true_df.groupby('rbd')['mrun'].nunique().reset_index()
    true_sch.rename(columns={'mrun': 'Total True Applicants School'}, inplace=True)
    
    df = pd.merge(tot_reg, true_reg, on=['Region', 'rbd'], how='left').fillna(0)
    df = pd.merge(df, tot_sch, on='rbd', how='left').fillna(0)
    df = pd.merge(df, true_sch, on='rbd', how='left').fillna(0)
    
    df['School DBN'] = df['rbd'].astype(str)
    df['School Name'] = "School_" + df['rbd'].astype(str) 
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
    
    school_caps = school_cap_df.groupby('rbd')['total_capacity'].sum().reset_index()
    school_caps.rename(columns={'rbd': 'School DBN', 'total_capacity': 'Capacity'}, inplace=True)
    school_caps['School DBN'] = school_caps['School DBN'].astype(str)
   
    admitted = school_cap_reg_df.groupby('rbd')['n_admitted'].sum().reset_index()
    admitted.rename(columns={'rbd': 'School DBN', 'n_admitted': 'matched_count'}, inplace=True)
    admitted['School DBN'] = admitted['School DBN'].astype(str)
    
    school_info_df = pd.merge(school_caps, admitted, on='School DBN', how='left')
    school_info_df['matched_count'] = school_info_df['matched_count'].fillna(0)
    school_info_df['Utilization'] = np.where(
        school_info_df['Capacity'] > 0,
        (school_info_df['matched_count'] / school_info_df['Capacity'] * 100).clip(upper=100),
        0.0
    )
    school_info_df = school_info_df[['School DBN', 'Capacity', 'Utilization']]
    
    district_to_region = {str(r): str(r) for r in df['Residential District'].unique()}
    return df, new_match_stats_df, school_info_df, district_to_region