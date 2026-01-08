"""
Analyze correlation between residential district rankings.
Computes pairwise correlations to understand how similar different districts'
school preferences are.
"""

import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from scipy.stats import spearmanr, kendalltau

def load_rankings(aggregate_type='residential'):
    """Load rankings from JSON file."""
    data_dir = Path('output/mallows_rankings')
    json_file = data_dir / f"{aggregate_type}_rankings.json"
    
    with open(json_file, 'r') as f:
        rankings_data = json.load(f)
    
    return rankings_data

def create_ranking_matrix(rankings_data):
    """
    Convert rankings to a matrix format for correlation analysis.
    
    Returns:
    - ranking_matrix: DataFrame where rows are categories, columns are schools,
                     values are ranks (lower rank = more preferred)
    - categories: list of category names
    """
    # Get all unique schools
    all_schools = set()
    for entry in rankings_data:
        all_schools.update(entry['ranking'])
    
    schools = sorted(all_schools)
    categories = [entry['category'] for entry in rankings_data]
    
    # Create matrix: rows = categories, columns = schools
    # Value = rank of school in that category (1-indexed, lower is better)
    matrix = np.full((len(categories), len(schools)), np.nan)
    
    for i, entry in enumerate(rankings_data):
        ranking = entry['ranking']
        for rank_position, school_dbn in enumerate(ranking, start=1):
            school_idx = schools.index(school_dbn)
            matrix[i, school_idx] = rank_position
    
    # For schools not in a category's ranking, assign rank = len(ranking) + 1
    for i, entry in enumerate(rankings_data):
        num_ranked = len(entry['ranking'])
        for j in range(len(schools)):
            if np.isnan(matrix[i, j]):
                matrix[i, j] = num_ranked + 1
    
    df = pd.DataFrame(matrix, index=categories, columns=schools)
    return df, categories, schools

def compute_spearman_correlation(ranking_matrix):
    """
    Compute Spearman rank correlation between all pairs of categories.
    
    Returns:
    - corr_matrix: DataFrame with pairwise correlations
    """
    n_categories = len(ranking_matrix)
    categories = ranking_matrix.index
    
    corr_matrix = np.zeros((n_categories, n_categories))
    
    for i in range(n_categories):
        for j in range(n_categories):
            if i == j:
                corr_matrix[i, j] = 1.0
            else:
                # Spearman correlation
                corr, _ = spearmanr(ranking_matrix.iloc[i], ranking_matrix.iloc[j])
                corr_matrix[i, j] = corr
    
    return pd.DataFrame(corr_matrix, index=categories, columns=categories)

def compute_kendall_correlation(ranking_matrix):
    """
    Compute Kendall's tau correlation between all pairs of categories.
    
    Returns:
    - corr_matrix: DataFrame with pairwise correlations
    """
    n_categories = len(ranking_matrix)
    categories = ranking_matrix.index
    
    corr_matrix = np.zeros((n_categories, n_categories))
    
    for i in range(n_categories):
        for j in range(n_categories):
            if i == j:
                corr_matrix[i, j] = 1.0
            else:
                # Kendall's tau
                corr, _ = kendalltau(ranking_matrix.iloc[i], ranking_matrix.iloc[j])
                corr_matrix[i, j] = corr
    
    return pd.DataFrame(corr_matrix, index=categories, columns=categories)

def compute_top_k_overlap(ranking_matrix, k=10):
    """
    Compute Jaccard similarity of top-k schools between categories.
    
    Returns:
    - overlap_matrix: DataFrame with pairwise Jaccard similarities
    """
    n_categories = len(ranking_matrix)
    categories = ranking_matrix.index
    
    # Get top k schools for each category
    top_k_sets = []
    for i in range(n_categories):
        ranks = ranking_matrix.iloc[i]
        top_schools = ranks.nsmallest(k).index.tolist()
        top_k_sets.append(set(top_schools))
    
    overlap_matrix = np.zeros((n_categories, n_categories))
    
    for i in range(n_categories):
        for j in range(n_categories):
            if i == j:
                overlap_matrix[i, j] = 1.0
            else:
                # Jaccard similarity
                intersection = len(top_k_sets[i] & top_k_sets[j])
                union = len(top_k_sets[i] | top_k_sets[j])
                overlap_matrix[i, j] = intersection / union if union > 0 else 0.0
    
    return pd.DataFrame(overlap_matrix, index=categories, columns=categories)

def plot_correlation_heatmap(corr_matrix, title, filename):
    """Plot correlation heatmap."""
    plt.figure(figsize=(14, 12))
    
    sns.heatmap(corr_matrix, annot=False, cmap='RdBu_r', center=0, 
                vmin=-1, vmax=1, square=True, cbar_kws={'label': 'Correlation'})
    
    plt.title(title, fontsize=14, pad=20)
    plt.xlabel('Category', fontsize=11)
    plt.ylabel('Category', fontsize=11)
    plt.xticks(rotation=90, fontsize=8)
    plt.yticks(rotation=0, fontsize=8)
    plt.tight_layout()
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"Saved: {filename}")
    plt.close()

def analyze_correlation_statistics(corr_matrix):
    """Compute summary statistics of correlations."""
    # Get upper triangle (excluding diagonal)
    n = len(corr_matrix)
    upper_triangle = []
    for i in range(n):
        for j in range(i+1, n):
            upper_triangle.append(corr_matrix.iloc[i, j])
    
    upper_triangle = np.array(upper_triangle)
    
    stats = {
        'mean': np.mean(upper_triangle),
        'median': np.median(upper_triangle),
        'std': np.std(upper_triangle),
        'min': np.min(upper_triangle),
        'max': np.max(upper_triangle),
        'q25': np.percentile(upper_triangle, 25),
        'q75': np.percentile(upper_triangle, 75)
    }
    
    return stats, upper_triangle

def find_most_similar_pairs(corr_matrix, n=10):
    """Find n most similar category pairs."""
    n_categories = len(corr_matrix)
    pairs = []
    
    for i in range(n_categories):
        for j in range(i+1, n_categories):
            pairs.append({
                'category1': corr_matrix.index[i],
                'category2': corr_matrix.index[j],
                'correlation': corr_matrix.iloc[i, j]
            })
    
    pairs_df = pd.DataFrame(pairs)
    return pairs_df.nlargest(n, 'correlation')

def find_most_different_pairs(corr_matrix, n=10):
    """Find n most different category pairs."""
    n_categories = len(corr_matrix)
    pairs = []
    
    for i in range(n_categories):
        for j in range(i+1, n_categories):
            pairs.append({
                'category1': corr_matrix.index[i],
                'category2': corr_matrix.index[j],
                'correlation': corr_matrix.iloc[i, j]
            })
    
    pairs_df = pd.DataFrame(pairs)
    return pairs_df.nsmallest(n, 'correlation')

def main():
    aggregate_type = 'residential'
    
    print("="*80)
    print(f"RANKING CORRELATION ANALYSIS - {aggregate_type.upper()}")
    print("="*80)
    
    # Load rankings
    print("\nLoading rankings...")
    rankings_data = load_rankings(aggregate_type)
    print(f"Loaded {len(rankings_data)} {aggregate_type} categories")
    
    # Create ranking matrix
    print("\nCreating ranking matrix...")
    ranking_matrix, categories, schools = create_ranking_matrix(rankings_data)
    print(f"Matrix shape: {len(categories)} categories × {len(schools)} schools")
    
    # Compute correlations
    print("\n" + "="*80)
    print("COMPUTING CORRELATIONS")
    print("="*80)
    
    print("\n1. Spearman rank correlation...")
    spearman_corr = compute_spearman_correlation(ranking_matrix)
    stats_spearman, values_spearman = analyze_correlation_statistics(spearman_corr)
    
    print("\n2. Kendall's tau correlation...")
    kendall_corr = compute_kendall_correlation(ranking_matrix)
    stats_kendall, values_kendall = analyze_correlation_statistics(kendall_corr)
    
    print("\n3. Top-10 overlap (Jaccard similarity)...")
    top10_overlap = compute_top_k_overlap(ranking_matrix, k=10)
    stats_top10, values_top10 = analyze_correlation_statistics(top10_overlap)
    
    # Print statistics
    print("\n" + "="*80)
    print("CORRELATION STATISTICS")
    print("="*80)
    
    print("\nSpearman Correlation:")
    for key, value in stats_spearman.items():
        print(f"  {key:10s}: {value:7.4f}")
    
    print("\nKendall's Tau:")
    for key, value in stats_kendall.items():
        print(f"  {key:10s}: {value:7.4f}")
    
    print("\nTop-10 Jaccard Overlap:")
    for key, value in stats_top10.items():
        print(f"  {key:10s}: {value:7.4f}")
    
    # Find most similar and different pairs
    print("\n" + "="*80)
    print("MOST SIMILAR PAIRS (Spearman)")
    print("="*80)
    similar_pairs = find_most_similar_pairs(spearman_corr, n=10)
    print(similar_pairs.to_string(index=False))
    
    print("\n" + "="*80)
    print("MOST DIFFERENT PAIRS (Spearman)")
    print("="*80)
    different_pairs = find_most_different_pairs(spearman_corr, n=10)
    print(different_pairs.to_string(index=False))
    
    # Save results
    output_dir = Path('output/correlation_analysis')
    output_dir.mkdir(exist_ok=True, parents=True)
    
    print("\n" + "="*80)
    print("SAVING RESULTS")
    print("="*80)
    
    # Save correlation matrices
    spearman_corr.to_csv(output_dir / f'{aggregate_type}_spearman_correlation.csv')
    print(f"Saved: {aggregate_type}_spearman_correlation.csv")
    
    kendall_corr.to_csv(output_dir / f'{aggregate_type}_kendall_correlation.csv')
    print(f"Saved: {aggregate_type}_kendall_correlation.csv")
    
    top10_overlap.to_csv(output_dir / f'{aggregate_type}_top10_overlap.csv')
    print(f"Saved: {aggregate_type}_top10_overlap.csv")
    
    # Save similar/different pairs
    similar_pairs.to_csv(output_dir / f'{aggregate_type}_most_similar_pairs.csv', index=False)
    print(f"Saved: {aggregate_type}_most_similar_pairs.csv")
    
    different_pairs.to_csv(output_dir / f'{aggregate_type}_most_different_pairs.csv', index=False)
    print(f"Saved: {aggregate_type}_most_different_pairs.csv")
    
    # Plot heatmaps
    print("\nGenerating visualizations...")
    plot_correlation_heatmap(
        spearman_corr, 
        f'Spearman Rank Correlation - {aggregate_type.title()} Districts',
        output_dir / f'{aggregate_type}_spearman_heatmap.png'
    )
    
    plot_correlation_heatmap(
        kendall_corr, 
        f"Kendall's Tau Correlation - {aggregate_type.title()} Districts",
        output_dir / f'{aggregate_type}_kendall_heatmap.png'
    )
    
    plot_correlation_heatmap(
        top10_overlap, 
        f'Top-10 School Overlap (Jaccard) - {aggregate_type.title()} Districts',
        output_dir / f'{aggregate_type}_top10_overlap_heatmap.png'
    )
    
    # Plot distribution of correlations
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    
    axes[0].hist(values_spearman, bins=30, edgecolor='black', alpha=0.7)
    axes[0].axvline(stats_spearman['mean'], color='red', linestyle='--', 
                    label=f"Mean = {stats_spearman['mean']:.3f}")
    axes[0].set_xlabel('Spearman Correlation', fontsize=11)
    axes[0].set_ylabel('Frequency', fontsize=11)
    axes[0].set_title('Distribution of Spearman Correlations', fontsize=12)
    axes[0].legend()
    axes[0].grid(alpha=0.3)
    
    axes[1].hist(values_kendall, bins=30, edgecolor='black', alpha=0.7, color='orange')
    axes[1].axvline(stats_kendall['mean'], color='red', linestyle='--', 
                    label=f"Mean = {stats_kendall['mean']:.3f}")
    axes[1].set_xlabel("Kendall's Tau", fontsize=11)
    axes[1].set_ylabel('Frequency', fontsize=11)
    axes[1].set_title("Distribution of Kendall's Tau", fontsize=12)
    axes[1].legend()
    axes[1].grid(alpha=0.3)
    
    axes[2].hist(values_top10, bins=30, edgecolor='black', alpha=0.7, color='green')
    axes[2].axvline(stats_top10['mean'], color='red', linestyle='--', 
                    label=f"Mean = {stats_top10['mean']:.3f}")
    axes[2].set_xlabel('Jaccard Similarity (Top-10)', fontsize=11)
    axes[2].set_ylabel('Frequency', fontsize=11)
    axes[2].set_title('Distribution of Top-10 Overlaps', fontsize=12)
    axes[2].legend()
    axes[2].grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / f'{aggregate_type}_correlation_distributions.png', 
                dpi=300, bbox_inches='tight')
    print(f"Saved: {aggregate_type}_correlation_distributions.png")
    plt.close()
    
    print("\n" + "="*80)
    print("ANALYSIS COMPLETE")
    print("="*80)
    print(f"All results saved to: {output_dir}")

if __name__ == '__main__':
    main()
