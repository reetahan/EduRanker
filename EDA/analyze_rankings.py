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
    """Load aggregate CSVs and filter specialized schools."""
    base_path = Path(__file__).parent / "output"
    
    specialized_schools = [
        '02M475', '13K430', '05M692', '10X445', '28Q687', 
        '14K449', '01M539', '19K505'
    ]
    
    data = {
        'residential': pd.read_csv(base_path / "residential_district.csv"),
        'language': pd.read_csv(base_path / "language_aggregates.csv"),
        'zip': pd.read_csv(base_path / "zip_code_aggregates.csv")
    }
    
    for name in data:
        data[name] = data[name][~data[name]['School DBN'].isin(specialized_schools)].copy()
    
    for name, df in data.items():
        true_col = next((col for col in df.columns if 'True Applicants' in col and 'School' not in col), None)
        total_col = next((col for col in df.columns if 'Total Applicants' in col and 'School' not in col), None)
        if true_col and total_col:
            df['Ratio'] = (df[true_col] ** 2) / df[total_col]
            df['Ratio'] = df['Ratio'].fillna(0)
            data[name] = df

    return data


def analyze_district_preference(residential_df):
    """Compare in-district vs out-of-district rankings."""
    residential_df['Same_District'] = residential_df['School District'] == residential_df['Residential District']
    in_district = residential_df[residential_df['Same_District'] == True]
    out_district = residential_df[residential_df['Same_District'] == False]
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    axes[0].boxplot([in_district['Rank'], out_district['Rank']], labels=['In-District', 'Out-of-District'])
    axes[0].set_ylabel('Rank')
    axes[0].set_title('Rank Distribution: In-District vs Out-of-District')
    axes[0].grid(True, alpha=0.3)
    
    axes[1].boxplot([in_district['Ratio'], out_district['Ratio']], labels=['In-District', 'Out-of-District'])
    axes[1].set_ylabel('Ratio (True/Total)')
    axes[1].set_title('Ratio Distribution: In-District vs Out-of-District')
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(Path(__file__).parent / "output" / "district_preference.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    return residential_df


def analyze_zip_preference(zip_df):
    """Compare same-zip vs different-zip rankings."""
    school_primary_zip = zip_df.groupby('School DBN')['Home Zip Code'].agg(
        lambda x: x.value_counts().index[0] if len(x) > 0 else None
    ).to_dict()
    
    zip_df['School_Primary_Zip'] = zip_df['School DBN'].map(school_primary_zip)
    zip_df['Same_Zip'] = zip_df['School_Primary_Zip'] == zip_df['Home Zip Code']
    
    return zip_df


def identify_top_schools(data):
    """Identify schools frequently ranked in top 10 across categories."""
    top_schools = defaultdict(lambda: {'count': 0, 'aggregates': [], 'avg_rank': []})
    
    for agg_type, df in data.items():
        top_10 = df[df['Rank'] <= 10].copy()
        for school_dbn in top_10['School DBN'].unique():
            school_data = top_10[top_10['School DBN'] == school_dbn]
            top_schools[school_dbn]['count'] += len(school_data)
            top_schools[school_dbn]['aggregates'].append(agg_type)
            top_schools[school_dbn]['avg_rank'].extend(school_data['Rank'].tolist())
            if 'name' not in top_schools[school_dbn]:
                top_schools[school_dbn]['name'] = school_data['School Name'].iloc[0]
    
    sorted_schools = sorted(top_schools.items(), key=lambda x: x[1]['count'], reverse=True)
    
    top_20_schools = []
    for dbn, info in sorted_schools[:20]:
        top_20_schools.append({
            'School DBN': dbn,
            'School Name': info['name'],
            'Top-10 Count': info['count'],
            'Average Rank': np.mean(info['avg_rank']),
            'Aggregates': ', '.join(set(info['aggregates']))
        })
    
    pd.DataFrame(top_20_schools).to_csv(Path(__file__).parent / "output" / "top_ranked_schools.csv", index=False)
    return sorted_schools


def visualize_rankings(data, top_schools):
    """Generate heatmap and distribution plots."""
    rank_matrix = []
    school_names = []
    for dbn, info in top_schools[:15]:
        school_names.append(info['name'][:30])
        ranks = []
        for agg_type in ['residential', 'language', 'zip']:
            df = data[agg_type]
            school_data = df[df['School DBN'] == dbn]
            avg_rank = school_data['Rank'].mean() if len(school_data) > 0 else np.nan
            ranks.append(avg_rank)
        rank_matrix.append(ranks)

    rank_matrix = np.array(rank_matrix)
    fig, ax = plt.subplots(figsize=(10, 10))
    sns.heatmap(rank_matrix, xticklabels=['Residential', 'Language', 'Zip'], yticklabels=school_names,
                annot=True, fmt='.1f', cmap='RdYlGn_r', mask=np.isnan(rank_matrix),
                cbar_kws={'label': 'Average Rank'}, ax=ax)
    ax.set_title('Average Rank Across Aggregate Types (Top 15 Schools)', fontsize=14, pad=20)
    plt.tight_layout()
    plt.savefig(Path(__file__).parent / "output" / "ranking_heatmap.png", dpi=300, bbox_inches='tight')
    plt.close()

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    for idx, (agg_type, df) in enumerate(data.items()):
        axes[idx].hist(df['Rank'], bins=50, edgecolor='black', alpha=0.7, cumulative=True, density=True)
        axes[idx].set_xlabel('Rank')
        axes[idx].set_ylabel('Cumulative Probability')
        axes[idx].set_title(f'Rank CDF: {agg_type.capitalize()}')
        axes[idx].grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(Path(__file__).parent / "output" / "rank_distributions.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    for idx, (agg_type, df) in enumerate(data.items()):
        filtered = df[~((df['Ratio'] == 1.0) & (df.iloc[:, 5] == 1))].copy()
        axes[idx].hist(filtered['Ratio'], bins=50, edgecolor='black', alpha=0.7, cumulative=True, density=True)
        axes[idx].set_xlabel('Ratio (True^2 / Total)')
        axes[idx].set_ylabel('Cumulative Probability')
        axes[idx].set_title(f'Ratio CDF: {agg_type.capitalize()}')
        axes[idx].grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(Path(__file__).parent / "output" / "ratio_distributions.png", dpi=300, bbox_inches='tight')
    plt.close()

    all_ranks = [rank for df in data.values() for rank in df['Rank'].tolist()]
    all_ratios = [ratio for df in data.values() for ratio in df['Ratio'].tolist()]

    plt.figure(figsize=(8, 5))
    sorted_ratios = np.sort(all_ratios)
    plt.step(sorted_ratios, np.arange(1, len(sorted_ratios) + 1) / len(sorted_ratios), where='post')
    plt.xlabel('Ratio (True^2 / Total)')
    plt.ylabel('Cumulative Probability')
    plt.title('Overall CDF of Ratio')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(Path(__file__).parent / "output" / "overall_ratio_cdf.png", dpi=300, bbox_inches='tight')
    plt.close()

    plt.figure(figsize=(8, 5))
    sorted_ranks = np.sort(all_ranks)
    plt.step(sorted_ranks, np.arange(1, len(sorted_ranks) + 1) / len(sorted_ranks), where='post')
    plt.xlabel('Rank')
    plt.ylabel('Cumulative Probability')
    plt.title('Overall CDF of Ranks')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(Path(__file__).parent / "output" / "overall_rank_cdf.png", dpi=300, bbox_inches='tight')
    plt.close()


def save_rankings_for_mallows(data):
    """Save rankings as JSON and NumPy arrays for Mallows models."""
    import json
    output_dir = Path(__file__).parent / "output" / "mallows_rankings"
    output_dir.mkdir(exist_ok=True)
    
    rankings_summary = []
    category_cols = {
        'residential': 'Residential District',
        'language': 'Home Language',
        'zip': 'Home Zip Code',
        'borough': 'Home Borough'
    }
    
    for agg_type, df in data.items():
        category_col = category_cols[agg_type]
        categories = df[category_col].unique()
        category_rankings = []
        
        for category in categories:
            category_data = df[df[category_col] == category].sort_values('Rank')
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
        
        with open(output_dir / f"{agg_type}_rankings.json", 'w') as f:
            json.dump(category_rankings, f, indent=2)
    
    pd.DataFrame(rankings_summary).to_csv(output_dir / "rankings_summary.csv", index=False)
    
    for agg_type, df in data.items():
        category_col = category_cols[agg_type]
        all_schools = df['School DBN'].unique()
        school_to_id = {dbn: idx for idx, dbn in enumerate(sorted(all_schools))}
        
        np.save(output_dir / f"{agg_type}_school_mapping.npy", 
                {'school_to_id': school_to_id, 'id_to_school': {v: k for k, v in school_to_id.items()}},
                allow_pickle=True)
        
        rankings_by_category = []
        categories = sorted(df[category_col].unique())
        for category in categories:
            category_data = df[df[category_col] == category].sort_values('Rank')
            ranking_ids = [school_to_id[dbn] for dbn in category_data['School DBN']]
            rankings_by_category.append(ranking_ids)
        
        np.save(output_dir / f"{agg_type}_rankings_array.npy", 
                {'categories': categories, 'rankings': rankings_by_category},
                allow_pickle=True)
    
    return rankings_summary


def main():
    data = load_data()
    analyze_district_preference(data['residential'])
    analyze_zip_preference(data['zip'])
    top_schools = identify_top_schools(data)
    visualize_rankings(data, top_schools)
    save_rankings_for_mallows(data)


if __name__ == "__main__":
    main()
