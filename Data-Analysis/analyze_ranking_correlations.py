"""Analyze ranking correlations between categories."""

import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from scipy.stats import spearmanr, kendalltau

def load_rankings(aggregate_type='residential'):
    data_dir = Path('output/mallows_rankings')
    with open(data_dir / f"{aggregate_type}_rankings.json", 'r') as f:
        return json.load(f)

def create_ranking_matrix(rankings_data):
    """Convert rankings to matrix format (rows=categories, cols=schools)."""
    all_schools = set()
    for entry in rankings_data:
        all_schools.update(entry['ranking'])
    
    schools = sorted(all_schools)
    categories = [entry['category'] for entry in rankings_data]
    matrix = np.full((len(categories), len(schools)), np.nan)
    
    for i, entry in enumerate(rankings_data):
        ranking = entry['ranking']
        for rank_position, school_dbn in enumerate(ranking, start=1):
            matrix[i, schools.index(school_dbn)] = rank_position
    
    for i, entry in enumerate(rankings_data):
        num_ranked = len(entry['ranking'])
        for j in range(len(schools)):
            if np.isnan(matrix[i, j]):
                matrix[i, j] = num_ranked + 1
    
    return pd.DataFrame(matrix, index=categories, columns=schools), categories, schools

def compute_spearman_correlation(ranking_matrix):
    n_categories = len(ranking_matrix)
    categories = ranking_matrix.index
    corr_matrix = np.zeros((n_categories, n_categories))
    
    for i in range(n_categories):
        for j in range(n_categories):
            if i == j:
                corr_matrix[i, j] = 1.0
            else:
                corr, _ = spearmanr(ranking_matrix.iloc[i], ranking_matrix.iloc[j])
                corr_matrix[i, j] = corr
    
    return pd.DataFrame(corr_matrix, index=categories, columns=categories)

def compute_kendall_correlation(ranking_matrix):
    n_categories = len(ranking_matrix)
    categories = ranking_matrix.index
    corr_matrix = np.zeros((n_categories, n_categories))
    
    for i in range(n_categories):
        for j in range(n_categories):
            if i == j:
                corr_matrix[i, j] = 1.0
            else:
                corr, _ = kendalltau(ranking_matrix.iloc[i], ranking_matrix.iloc[j])
                corr_matrix[i, j] = corr
    
    return pd.DataFrame(corr_matrix, index=categories, columns=categories)

def compute_top_k_overlap(ranking_matrix, k=10):
    n_categories = len(ranking_matrix)
    categories = ranking_matrix.index
    top_k_sets = [set(ranking_matrix.iloc[i].nsmallest(k).index.tolist()) for i in range(n_categories)]
    overlap_matrix = np.zeros((n_categories, n_categories))
    
    for i in range(n_categories):
        for j in range(n_categories):
            if i == j:
                overlap_matrix[i, j] = 1.0
            else:
                intersection = len(top_k_sets[i] & top_k_sets[j])
                union = len(top_k_sets[i] | top_k_sets[j])
                overlap_matrix[i, j] = intersection / union if union > 0 else 0.0
    
    return pd.DataFrame(overlap_matrix, index=categories, columns=categories)

def plot_correlation_heatmap(corr_matrix, title, filename):
    plt.figure(figsize=(14, 12))
    sns.heatmap(corr_matrix, annot=False, cmap='RdBu_r', center=0, vmin=-1, vmax=1, square=True, cbar_kws={'label': 'Correlation'})
    plt.title(title, fontsize=14, pad=20)
    plt.xlabel('Category', fontsize=11)
    plt.ylabel('Category', fontsize=11)
    plt.xticks(rotation=90, fontsize=8)
    plt.yticks(rotation=0, fontsize=8)
    plt.tight_layout()
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()

def analyze_correlation_statistics(corr_matrix):
    n = len(corr_matrix)
    upper_triangle = [corr_matrix.iloc[i, j] for i in range(n) for j in range(i+1, n)]
    upper_triangle = np.array(upper_triangle)
    
    return {
        'mean': np.mean(upper_triangle),
        'median': np.median(upper_triangle),
        'std': np.std(upper_triangle),
        'min': np.min(upper_triangle),
        'max': np.max(upper_triangle),
        'q25': np.percentile(upper_triangle, 25),
        'q75': np.percentile(upper_triangle, 75)
    }, upper_triangle

def find_most_similar_pairs(corr_matrix, n=10):
    n_categories = len(corr_matrix)
    pairs = [{'category1': corr_matrix.index[i], 'category2': corr_matrix.index[j], 'correlation': corr_matrix.iloc[i, j]}
             for i in range(n_categories) for j in range(i+1, n_categories)]
    return pd.DataFrame(pairs).nlargest(n, 'correlation')

def find_most_different_pairs(corr_matrix, n=10):
    n_categories = len(corr_matrix)
    pairs = [{'category1': corr_matrix.index[i], 'category2': corr_matrix.index[j], 'correlation': corr_matrix.iloc[i, j]}
             for i in range(n_categories) for j in range(i+1, n_categories)]
    return pd.DataFrame(pairs).nsmallest(n, 'correlation')

def analyze_language_rankings():
    """Analyze ranking similarities across top 7 languages."""
    output_dir = Path('output')
    language_files = sorted(output_dir.glob('ranking_language_*.csv'))
    
    if not language_files:
        return
    
    language_rankings = {}
    for lang_file in language_files:
        lang_name = lang_file.stem.replace('ranking_language_', '')
        language_rankings[lang_name] = pd.read_csv(lang_file).sort_values('Rank')
    
    all_schools = sorted(set(school for df in language_rankings.values() for school in df['School DBN'].unique()))
    lang_names = sorted(language_rankings.keys())
    
    matrix_data = {}
    for lang in lang_names:
        df = language_rankings[lang]
        rank_dict = dict(zip(df['School DBN'], df['Rank']))
        matrix_data[lang] = {school: rank_dict.get(school, np.nan) for school in all_schools}
    
    lang_ranking_matrix = pd.DataFrame(matrix_data).T
    for lang in lang_names:
        max_rank = lang_ranking_matrix.loc[lang].max()
        lang_ranking_matrix.loc[lang] = lang_ranking_matrix.loc[lang].fillna(max_rank + 1)
    
    lang_spearman = compute_spearman_correlation(lang_ranking_matrix)
    similar_langs = find_most_similar_pairs(lang_spearman, n=10)
    different_langs = find_most_different_pairs(lang_spearman, n=10)
    
    rank_variance = []
    for school in all_schools:
        ranks = lang_ranking_matrix[school]
        valid_ranks = ranks[~ranks.isna()]
        if len(valid_ranks) > 1:
            rank_variance.append({
                'school': school,
                'mean_rank': valid_ranks.mean(),
                'rank_variance': valid_ranks.var(),
                'num_languages': len(valid_ranks),
                'min_rank': valid_ranks.min(),
                'max_rank': valid_ranks.max()
            })
    
    variance_df = pd.DataFrame(rank_variance).sort_values('rank_variance', ascending=False)
    school_sample = pd.read_csv(output_dir / 'ranking_language_english.csv')[['School DBN', 'School Name']].drop_duplicates()
    variance_df = variance_df.merge(school_sample, left_on='school', right_on='School DBN', how='left')
    
    lang_output_dir = Path('output/language_analysis')
    lang_output_dir.mkdir(exist_ok=True, parents=True)
    lang_spearman.to_csv(lang_output_dir / 'language_spearman_correlation.csv')
    similar_langs.to_csv(lang_output_dir / 'language_most_similar_pairs.csv', index=False)
    different_langs.to_csv(lang_output_dir / 'language_most_different_pairs.csv', index=False)
    variance_df.to_csv(lang_output_dir / 'school_ranking_variance.csv', index=False)
    plot_correlation_heatmap(lang_spearman, 'Spearman Rank Correlation - Home Languages', lang_output_dir / 'language_spearman_heatmap.png')


def generate_overall_school_ranking():
    """Generate overall school ranking from raw data using (true^2)/total."""
    data1_path = Path('raw-data/DATA1_fall-2024-admissions-72-suppressed.xlsx')
    df = pd.read_excel(data1_path, sheet_name='School')
    df_schools = df[df['Category'] == 'All Students'].copy()
    
    df_schools['Grade 9 Total Applicants'] = pd.to_numeric(df_schools['Grade 9 Total Applicants'].replace({'s': 1, 's^': 6}), errors='coerce')
    df_schools['Grade 9 True Applicants'] = pd.to_numeric(df_schools['Grade 9 True Applicants'].replace({'s': 1, 's^': 6}), errors='coerce')
    df_schools = df_schools.dropna(subset=['Grade 9 Total Applicants', 'Grade 9 True Applicants'])
    
    df_schools['Ratio'] = (df_schools['Grade 9 True Applicants'] ** 2) / df_schools['Grade 9 Total Applicants']
    df_schools['Rank'] = df_schools['Ratio'].rank(ascending=False, method='min').astype(int)
    
    result_df = df_schools[['School DBN', 'School Name', 'School District', 'Grade 9 Total Applicants', 'Grade 9 True Applicants', 'Ratio', 'Rank']].rename(
        columns={'Grade 9 Total Applicants': 'Total Applicants', 'Grade 9 True Applicants': 'True Applicants'}
    ).sort_values('Rank')
    
    output_dir = Path('output/language_analysis')
    output_dir.mkdir(exist_ok=True, parents=True)
    result_df.to_csv(output_dir / 'all_schools_ranked.csv', index=False)


def main():
    aggregate_type = 'residential'
    rankings_data = load_rankings(aggregate_type)
    ranking_matrix, categories, schools = create_ranking_matrix(rankings_data)
    
    mean_ranks = ranking_matrix.mean(axis=0)
    top12_schools = mean_ranks.nsmallest(12).index.tolist()
    ranking_matrix = ranking_matrix[top12_schools]
    
    spearman_corr = compute_spearman_correlation(ranking_matrix)
    stats_spearman, values_spearman = analyze_correlation_statistics(spearman_corr)
    kendall_corr = compute_kendall_correlation(ranking_matrix)
    stats_kendall, values_kendall = analyze_correlation_statistics(kendall_corr)
    top10_overlap = compute_top_k_overlap(ranking_matrix, k=10)
    stats_top10, values_top10 = analyze_correlation_statistics(top10_overlap)
    
    similar_pairs = find_most_similar_pairs(spearman_corr, n=10)
    different_pairs = find_most_different_pairs(spearman_corr, n=10)
    
    output_dir = Path('output/correlation_analysis')
    output_dir.mkdir(exist_ok=True, parents=True)
    
    spearman_corr.to_csv(output_dir / f'{aggregate_type}_spearman_correlation.csv')
    kendall_corr.to_csv(output_dir / f'{aggregate_type}_kendall_correlation.csv')
    top10_overlap.to_csv(output_dir / f'{aggregate_type}_top10_overlap.csv')
    similar_pairs.to_csv(output_dir / f'{aggregate_type}_most_similar_pairs.csv', index=False)
    different_pairs.to_csv(output_dir / f'{aggregate_type}_most_different_pairs.csv', index=False)
    
    plot_correlation_heatmap(spearman_corr, f'Spearman Rank Correlation - {aggregate_type.title()} Districts', output_dir / f'{aggregate_type}_spearman_heatmap.png')
    plot_correlation_heatmap(kendall_corr, f"Kendall's Tau Correlation - {aggregate_type.title()} Districts", output_dir / f'{aggregate_type}_kendall_heatmap.png')
    plot_correlation_heatmap(top10_overlap, f'Top-10 School Overlap (Jaccard) - {aggregate_type.title()} Districts', output_dir / f'{aggregate_type}_top10_overlap_heatmap.png')
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    axes[0].hist(values_spearman, bins=30, edgecolor='black', alpha=0.7)
    axes[0].axvline(stats_spearman['mean'], color='red', linestyle='--', label=f"Mean = {stats_spearman['mean']:.3f}")
    axes[0].set_xlabel('Spearman Correlation', fontsize=11)
    axes[0].set_ylabel('Frequency', fontsize=11)
    axes[0].set_title('Distribution of Spearman Correlations', fontsize=12)
    axes[0].legend()
    axes[0].grid(alpha=0.3)
    
    axes[1].hist(values_kendall, bins=30, edgecolor='black', alpha=0.7, color='orange')
    axes[1].axvline(stats_kendall['mean'], color='red', linestyle='--', label=f"Mean = {stats_kendall['mean']:.3f}")
    axes[1].set_xlabel("Kendall's Tau", fontsize=11)
    axes[1].set_ylabel('Frequency', fontsize=11)
    axes[1].set_title("Distribution of Kendall's Tau", fontsize=12)
    axes[1].legend()
    axes[1].grid(alpha=0.3)
    
    axes[2].hist(values_top10, bins=30, edgecolor='black', alpha=0.7, color='green')
    axes[2].axvline(stats_top10['mean'], color='red', linestyle='--', label=f"Mean = {stats_top10['mean']:.3f}")
    axes[2].set_xlabel('Jaccard Similarity (Top-10)', fontsize=11)
    axes[2].set_ylabel('Frequency', fontsize=11)
    axes[2].set_title('Distribution of Top-10 Overlaps', fontsize=12)
    axes[2].legend()
    axes[2].grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / f'{aggregate_type}_correlation_distributions.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    analyze_language_rankings()
    generate_overall_school_ranking()

if __name__ == '__main__':
    main()
