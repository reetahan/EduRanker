import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from collections import defaultdict

# Set style for better-looking plots
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)


def load_data():
    """Load all aggregate CSV files"""
    base_path = Path(__file__).parent / "output"
    
    # List of 8 specialized high schools (SHSAT schools) to exclude
    specialized_schools = [
        '02M475',  # Stuyvesant High School
        '13K430',  # Brooklyn Technical High School
        '05M692',  # High School for Mathematics, Science and Engineering at City College
        '10X445',  # Bronx High School of Science
        '28Q687',  # Queens High School for the Sciences at York College
        '14K449',  # Brooklyn Latin School
        '01M539',  # High School for Math, Science and Engineering at City College (different from 05M692)
        '19K505',  # Staten Island Technical High School
    ]
    
    data = {
        'residential': pd.read_csv(base_path / "residential_district.csv"),
        'language': pd.read_csv(base_path / "language_aggregates.csv"),
        'zip': pd.read_csv(base_path / "zip_code_aggregates.csv")
    }
    
    print("Loaded data:")
    for name, df in data.items():
        print(f"  {name}: {len(df):,} rows, {len(df.columns)} columns")
    
    # Filter out specialized high schools
    print(f"\nFiltering out {len(specialized_schools)} specialized high schools (SHSAT schools)...")
    for name, df in data.items():
        before = len(df)
        data[name] = df[~df['School DBN'].isin(specialized_schools)].copy()
        after = len(data[name])
        removed = before - after
        print(f"  {name}: removed {removed:,} rows ({before:,} → {after:,})")
    
    print("\n" + "="*80)
    print("DATA STRUCTURE EXPLANATION")
    print("="*80)
    print("\nEach aggregate file contains rankings PER CATEGORY:")
    print("  - Residential: Each district ranks all schools (rank 1-N per district)")
    print("  - Language: Each language group ranks all schools (rank 1-N per language)")
    print("  - Zip Code: Each zip code ranks all schools (rank 1-N per zip)")
    print("\nExample: If School X is ranked #3 by Spanish speakers and #5 by")
    print("         Mandarin speakers, it appears TWICE in the language file.")
    print("\nSo a popular school might appear hundreds of times across all")
    print("categories with varying ranks - that's what we're analyzing!")
    print("\nNote: Specialized high schools (SHSAT schools) are excluded from this analysis")
    print("      as they have separate admissions processes.")
    
    return data


def analyze_district_preference(residential_df):
    """
    Question 1: Do residential districts tend to rank schools in their own district higher?
    """
    print("\n" + "="*80)
    print("QUESTION 1: District Preference Analysis")
    print("="*80)
    
    # Compare rankings for in-district vs out-of-district schools
    residential_df['Same_District'] = residential_df['School District'] == residential_df['Residential District']
    
    in_district = residential_df[residential_df['Same_District'] == True]
    out_district = residential_df[residential_df['Same_District'] == False]
    
    print(f"\nIn-district schools: {len(in_district):,} rows")
    print(f"Out-of-district schools: {len(out_district):,} rows")
    
    print(f"\nAverage rank for in-district schools: {in_district['Rank'].mean():.2f}")
    print(f"Average rank for out-of-district schools: {out_district['Rank'].mean():.2f}")
    
    print(f"\nMedian rank for in-district schools: {in_district['Rank'].median():.0f}")
    print(f"Median rank for out-of-district schools: {out_district['Rank'].median():.0f}")
    
    print(f"\nAverage ratio for in-district schools: {in_district['Ratio'].mean():.4f}")
    print(f"Average ratio for out-of-district schools: {out_district['Ratio'].mean():.4f}")
    
    # Count how many times in-district schools are ranked #1
    top_ranked = residential_df[residential_df['Rank'] == 1]
    in_district_top = top_ranked['Same_District'].sum()
    print(f"\n# of times in-district schools are ranked #1: {in_district_top} / {len(top_ranked)} ({in_district_top/len(top_ranked)*100:.1f}%)")
    
    # Visualization
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # Box plot of ranks
    axes[0].boxplot([in_district['Rank'], out_district['Rank']], 
                     labels=['In-District', 'Out-of-District'])
    axes[0].set_ylabel('Rank')
    axes[0].set_title('Rank Distribution: In-District vs Out-of-District')
    axes[0].grid(True, alpha=0.3)
    
    # Box plot of ratios
    axes[1].boxplot([in_district['Ratio'], out_district['Ratio']], 
                     labels=['In-District', 'Out-of-District'])
    axes[1].set_ylabel('Ratio (True/Total)')
    axes[1].set_title('Ratio Distribution: In-District vs Out-of-District')
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(Path(__file__).parent / "output" / "district_preference.png", dpi=300, bbox_inches='tight')
    print(f"\n✓ Saved visualization: output/district_preference.png")
    plt.close()
    
    return residential_df


def analyze_borough_preference(borough_df):
    """
    Question 2: Do boroughs rank schools in their own borough higher?
    """
    print("\n" + "="*80)
    print("QUESTION 2: Borough Preference Analysis")
    print("="*80)
    
    # Extract borough from school district (first 2 chars typically indicate borough)
    # Note: This is a simplification; you may need to refine based on actual DBN structure
    def get_school_borough(dbn):
        borough_codes = {
            '01': 'Manhattan', '02': 'Bronx', '03': 'Brooklyn',
            '04': 'Queens', '05': 'Staten Island',
            '06': 'Manhattan', '07': 'Bronx', '08': 'Brooklyn',
            '09': 'Manhattan', '10': 'Bronx', '11': 'Brooklyn',
            '12': 'Bronx', '13': 'Brooklyn', '14': 'Brooklyn',
            '15': 'Brooklyn', '16': 'Brooklyn', '17': 'Brooklyn',
            '18': 'Brooklyn', '19': 'Brooklyn', '20': 'Brooklyn',
            '21': 'Brooklyn', '22': 'Brooklyn', '23': 'Brooklyn',
            '24': 'Queens', '25': 'Queens', '26': 'Queens',
            '27': 'Queens', '28': 'Queens', '29': 'Queens',
            '30': 'Queens', '31': 'Staten Island', '32': 'Brooklyn',
            '75': 'Bronx', '79': 'Manhattan', '84': 'Manhattan'
        }
        district = dbn[:2]
        return borough_codes.get(district, 'Unknown')
    
    borough_df['School_Borough'] = borough_df['School DBN'].apply(get_school_borough)
    borough_df['Same_Borough'] = borough_df['School_Borough'] == borough_df['Home Borough']
    
    same_borough = borough_df[borough_df['Same_Borough'] == True]
    diff_borough = borough_df[borough_df['Same_Borough'] == False]
    
    print(f"\nSame-borough schools: {len(same_borough):,} rows")
    print(f"Different-borough schools: {len(diff_borough):,} rows")
    
    print(f"\nAverage rank for same-borough schools: {same_borough['Rank'].mean():.2f}")
    print(f"Average rank for different-borough schools: {diff_borough['Rank'].mean():.2f}")
    
    print(f"\nAverage ratio for same-borough schools: {same_borough['Ratio'].mean():.4f}")
    print(f"Average ratio for different-borough schools: {diff_borough['Ratio'].mean():.4f}")
    
    # Per-borough analysis
    print("\nPer-borough breakdown:")
    print("(What % of rankings from each borough are for schools IN that borough)")
    for borough in ['Manhattan', 'Bronx', 'Brooklyn', 'Queens', 'Staten Island']:
        borough_data = borough_df[borough_df['Home Borough'] == borough]
        same_boro = borough_data[borough_data['Same_Borough'] == True]
        if len(borough_data) > 0:
            pct_same = len(same_boro) / len(borough_data) * 100
            avg_rank_same = same_boro['Rank'].mean() if len(same_boro) > 0 else 0
            n_total = len(borough_data)
            n_same = len(same_boro)
            print(f"  {borough}: {n_same}/{n_total} ({pct_same:.1f}%) rankings are for same-borough schools, avg rank: {avg_rank_same:.2f}")
    
    return borough_df


def analyze_zip_preference(zip_df):
    """
    Question 2 (continued): Do zip codes rank schools in nearby areas higher?
    """
    print("\n" + "="*80)
    print("QUESTION 2b: Zip Code Preference Analysis")
    print("="*80)
    
    # Get school's primary zip (most common zip for that school)
    school_primary_zip = zip_df.groupby('School DBN')['Home Zip Code'].agg(
        lambda x: x.value_counts().index[0] if len(x) > 0 else None
    ).to_dict()
    
    zip_df['School_Primary_Zip'] = zip_df['School DBN'].map(school_primary_zip)
    zip_df['Same_Zip'] = zip_df['School_Primary_Zip'] == zip_df['Home Zip Code']
    
    same_zip = zip_df[zip_df['Same_Zip'] == True]
    diff_zip = zip_df[zip_df['Same_Zip'] == False]
    
    print(f"\nSame-zip schools: {len(same_zip):,} rows")
    print(f"Different-zip schools: {len(diff_zip):,} rows")
    
    if len(same_zip) > 0:
        print(f"\nAverage rank for same-zip schools: {same_zip['Rank'].mean():.2f}")
        print(f"Average ratio for same-zip schools: {same_zip['Ratio'].mean():.4f}")
    
    if len(diff_zip) > 0:
        print(f"\nAverage rank for different-zip schools: {diff_zip['Rank'].mean():.2f}")
        print(f"Average ratio for different-zip schools: {diff_zip['Ratio'].mean():.4f}")
    
    return zip_df


def identify_top_schools(data):
    """
    Question 3: Identify schools commonly ranked highly across different aggregates
    
    For each school, we count how many times it appears in the top 10 ranking
    across ALL categories. For example:
    - A school ranked in top 10 by 50 different languages = count of 50
    - A school ranked in top 10 by 30 zip codes = count of 30
    - Total count = appearances across all aggregate types
    """
    print("\n" + "="*80)
    print("QUESTION 3: Commonly High-Ranked Schools")
    print("="*80)
    print("\nNote: 'Count' = number of times a school appears in top 10 across")
    print("      ALL categories (districts, languages, zip codes, boroughs combined)")
    print("      A high count means the school is consistently highly ranked.")
    
    # For each aggregate type, find schools that appear in top 10 ranks frequently
    top_schools = defaultdict(lambda: {'count': 0, 'aggregates': [], 'avg_rank': []})
    
    for agg_type, df in data.items():
        # Get schools ranked in top 10 for each category
        top_10 = df[df['Rank'] <= 10].copy()
        
        for school_dbn in top_10['School DBN'].unique():
            school_data = top_10[top_10['School DBN'] == school_dbn]
            top_schools[school_dbn]['count'] += len(school_data)
            top_schools[school_dbn]['aggregates'].append(agg_type)
            top_schools[school_dbn]['avg_rank'].extend(school_data['Rank'].tolist())
            
            if 'name' not in top_schools[school_dbn]:
                top_schools[school_dbn]['name'] = school_data['School Name'].iloc[0]
    
    # Sort by frequency of top-10 appearances
    sorted_schools = sorted(top_schools.items(), key=lambda x: x[1]['count'], reverse=True)
    
    print("\nTop 20 schools by frequency of top-10 rankings:")
    print(f"{'DBN':<10} {'School Name':<50} {'Count':<7} {'Avg Rank':<10} {'Aggregates'}")
    print("-" * 110)
    
    top_20_schools = []
    for i, (dbn, info) in enumerate(sorted_schools[:20]):
        avg_rank = np.mean(info['avg_rank'])
        agg_str = ', '.join(set(info['aggregates']))
        print(f"{dbn:<10} {info['name'][:48]:<50} {info['count']:<7} {avg_rank:<10.2f} {agg_str}")
        top_20_schools.append({
            'School DBN': dbn,
            'School Name': info['name'],
            'Top-10 Count': info['count'],
            'Average Rank': avg_rank,
            'Aggregates': agg_str
        })
    
    # Save to CSV
    top_schools_df = pd.DataFrame(top_20_schools)
    output_path = Path(__file__).parent / "output" / "top_ranked_schools.csv"
    top_schools_df.to_csv(output_path, index=False)
    print(f"\n✓ Saved: output/top_ranked_schools.csv")
    
    return sorted_schools


def visualize_rankings(data, top_schools):
    """
    Question 4: Visualize rankings for top schools across different aggregates
    
    The heatmap shows average rank for each school across ALL categories within
    each aggregate type. For example:
    - 'Language' column = average of a school's rank across all languages that ranked it
    - 'Zip' column = average of a school's rank across all zip codes that ranked it
    
    Lower numbers (green) = better average ranking
    Higher numbers (red) = worse average ranking
    """
    print("\n" + "="*80)
    print("QUESTION 4: Ranking Visualizations")
    print("="*80)
    print("\nHeatmap explanation:")
    print("  - Each cell shows the AVERAGE rank for that school across")
    print("    all categories in that aggregate type")
    print("  - Example: 'Language' = average rank across all languages")
    print("  - Lower numbers (green) = consistently highly ranked")
    print("  - Higher numbers (red) = lower average ranking\n")
    
    # Select top 10 schools by frequency
    top_10_dbns = [dbn for dbn, _ in top_schools[:10]]
    
    # Create a heatmap showing average rank per school per aggregate type
    rank_matrix = []
    school_names = []
    
    for dbn, info in top_schools[:15]:
        school_names.append(info['name'][:30])  # Truncate for display
        ranks = []
        
        for agg_type in ['residential', 'language', 'zip']:
            df = data[agg_type]
            school_data = df[df['School DBN'] == dbn]
            
            if len(school_data) > 0:
                avg_rank = school_data['Rank'].mean()
            else:
                avg_rank = np.nan
            
            ranks.append(avg_rank)
        
        rank_matrix.append(ranks)
    
    rank_matrix = np.array(rank_matrix)
    
    # Create heatmap
    fig, ax = plt.subplots(figsize=(10, 10))
    
    # Mask NaN values
    mask = np.isnan(rank_matrix)
    
    sns.heatmap(rank_matrix, 
                xticklabels=['Residential', 'Language', 'Zip'],
                yticklabels=school_names,
                annot=True, 
                fmt='.1f',
                cmap='RdYlGn_r',  # Red=high rank (bad), Green=low rank (good)
                mask=mask,
                cbar_kws={'label': 'Average Rank'},
                ax=ax)
    
    ax.set_title('Average Rank Across Aggregate Types\n(Top 15 Schools by Frequency)', fontsize=14, pad=20)
    ax.set_xlabel('Aggregate Type', fontsize=12)
    ax.set_ylabel('School', fontsize=12)
    
    plt.tight_layout()
    plt.savefig(Path(__file__).parent / "output" / "ranking_heatmap.png", dpi=300, bbox_inches='tight')
    print(f"✓ Saved: output/ranking_heatmap.png")
    plt.close()
    
    # Rank distribution by aggregate type
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    axes = axes.flatten()
    
    for idx, (agg_type, df) in enumerate(data.items()):
        axes[idx].hist(df['Rank'], bins=50, edgecolor='black', alpha=0.7)
        axes[idx].set_xlabel('Rank')
        axes[idx].set_ylabel('Frequency')
        axes[idx].set_title(f'Rank Distribution: {agg_type.capitalize()}')
        axes[idx].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(Path(__file__).parent / "output" / "rank_distributions.png", dpi=300, bbox_inches='tight')
    print(f"✓ Saved: output/rank_distributions.png")
    plt.close()
    
    # Ratio distribution by aggregate type
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    axes = axes.flatten()
    
    for idx, (agg_type, df) in enumerate(data.items()):
        # Filter out suppressed values (ratio of 1.0 with both true and total = 1)
        filtered = df[~((df['Ratio'] == 1.0) & (df.iloc[:, 5] == 1))].copy()
        
        axes[idx].hist(filtered['Ratio'], bins=50, edgecolor='black', alpha=0.7)
        axes[idx].set_xlabel('Ratio (True/Total)')
        axes[idx].set_ylabel('Frequency')
        axes[idx].set_title(f'Ratio Distribution: {agg_type.capitalize()}')
        axes[idx].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(Path(__file__).parent / "output" / "ratio_distributions.png", dpi=300, bbox_inches='tight')
    print(f"✓ Saved: output/ratio_distributions.png")
    plt.close()


def save_rankings_for_mallows(data):
    """
    Question 5: Save rankings in format suitable for Mallows mixture models
    
    For each category (district, language, zip, borough), create a ranking list
    of schools ordered by their rank within that category.
    """
    print("\n" + "="*80)
    print("QUESTION 5: Saving Rankings for Mallows Mixture")
    print("="*80)
    
    output_dir = Path(__file__).parent / "output" / "mallows_rankings"
    output_dir.mkdir(exist_ok=True)
    
    rankings_summary = []
    
    for agg_type, df in data.items():
        print(f"\nProcessing {agg_type} rankings...")
        
        # Determine the category column
        category_col = {
            'residential': 'Residential District',
            'language': 'Home Language',
            'zip': 'Home Zip Code',
            'borough': 'Home Borough'
        }[agg_type]
        
        # For each category, create a ranking
        categories = df[category_col].unique()
        
        category_rankings = []
        
        for category in categories:
            category_data = df[df[category_col] == category].copy()
            
            # Sort by rank
            category_data = category_data.sort_values('Rank')
            
            # Create ranking list (just school DBNs in order)
            ranking = category_data['School DBN'].tolist()
            
            category_rankings.append({
                'category': str(category),
                'n_schools': int(len(ranking)),
                'ranking': ranking,
                'ratios': [float(r) for r in category_data['Ratio'].tolist()],
                'school_names': category_data['School Name'].tolist()
            })
            
            rankings_summary.append({
                'Aggregate Type': agg_type,
                'Category': category,
                'Number of Schools': len(ranking),
                'Top School': category_data.iloc[0]['School Name'],
                'Top School Ratio': category_data.iloc[0]['Ratio']
            })
        
        # Save rankings for this aggregate type
        import json
        output_file = output_dir / f"{agg_type}_rankings.json"
        with open(output_file, 'w') as f:
            json.dump(category_rankings, f, indent=2)
        
        print(f"  ✓ Saved {len(categories)} category rankings to: {output_file}")
    
    # Save summary
    summary_df = pd.DataFrame(rankings_summary)
    summary_path = output_dir / "rankings_summary.csv"
    summary_df.to_csv(summary_path, index=False)
    print(f"\n✓ Saved summary: {summary_path}")
    
    # Also save as NumPy arrays for easy loading
    print("\nSaving as NumPy arrays...")
    for agg_type, df in data.items():
        category_col = {
            'residential': 'Residential District',
            'language': 'Home Language',
            'zip': 'Home Zip Code',
            'borough': 'Home Borough'
        }[agg_type]
        
        # Create a mapping of school DBN to integer ID
        all_schools = df['School DBN'].unique()
        school_to_id = {dbn: idx for idx, dbn in enumerate(sorted(all_schools))}
        
        # Save the mapping
        np.save(output_dir / f"{agg_type}_school_mapping.npy", 
                {'school_to_id': school_to_id, 'id_to_school': {v: k for k, v in school_to_id.items()}},
                allow_pickle=True)
        
        # Convert rankings to integer arrays
        rankings_by_category = []
        categories = sorted(df[category_col].unique())
        
        for category in categories:
            category_data = df[df[category_col] == category].sort_values('Rank')
            ranking_ids = [school_to_id[dbn] for dbn in category_data['School DBN']]
            rankings_by_category.append(ranking_ids)
        
        # Save as numpy array (list of lists)
        np.save(output_dir / f"{agg_type}_rankings_array.npy", 
                {'categories': categories, 'rankings': rankings_by_category},
                allow_pickle=True)
        
        print(f"  ✓ Saved NumPy arrays for {agg_type}")
    
    print(f"\n✓ All rankings saved to: {output_dir}/")
    
    return rankings_summary


def main():
    print("="*80)
    print("RANKING ANALYSIS")
    print("="*80)
    
    # Load data
    data = load_data()
    
    # Question 1: District preference
    residential_df = analyze_district_preference(data['residential'])
    
    # Question 2: Zip preference
    zip_df = analyze_zip_preference(data['zip'])
    
    # Question 3: Top schools
    top_schools = identify_top_schools(data)
    
    # Question 4: Visualizations
    visualize_rankings(data, top_schools)
    
    # Question 5: Save rankings for Mallows
    save_rankings_for_mallows(data)
    
    print("\n" + "="*80)
    print("ANALYSIS COMPLETE!")
    print("="*80)
    print("\nGenerated files in 'output/':")
    print("  - district_preference.png")
    print("  - ranking_heatmap.png")
    print("  - rank_distributions.png")
    print("  - ratio_distributions.png")
    print("  - top_ranked_schools.csv")
    print("  - mallows_rankings/ (directory with JSON and NumPy files)")


if __name__ == "__main__":
    main()
