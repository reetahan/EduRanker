import numpy as np
import matplotlib.pyplot as plt
import os
from file_config import EXP_OUT_FOLDER

def plot_capacity_and_sigmas(real_schools, real_caps, real_sigmas):
    ranks = {}
    for d in [1, 2, 3]:
        for rank, s in enumerate(real_sigmas[d], 1):
            ranks[(d, s)] = rank

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5), gridspec_kw={'width_ratios': [1, 1.4]})

    # Bar chart sorted by capacity
    sort_idx = np.argsort(real_caps)[::-1]
    sorted_schools = [real_schools[i] for i in sort_idx]
    sorted_caps = real_caps[sort_idx]
    colors = ['#c0392b' if c > 50 else '#2980b9' for c in sorted_caps]
    ax1.barh(range(len(sorted_schools)), sorted_caps, color=colors, edgecolor='white', linewidth=0.5)
    ax1.set_yticks(range(len(sorted_schools)))
    #ax1.set_yticklabels(sorted_schools, fontsize=8, fontfamily='monospace')
    sorted_indices = [real_schools.index(s) + 1 for s in sorted_schools]
    ax1.set_yticklabels([f'School {idx} |' for idx in sorted_indices], fontsize=8)
    ax1.invert_yaxis()
    ax1.set_xlabel('Scaled Capacity (seats)')
    ax1.set_title('Capacity Distribution', fontweight='bold')
    ax1.axvline(x=np.median(real_caps), color='gray', linestyle='--', alpha=0.7, label=f'Median={np.median(real_caps):.0f}')
    ax1.axvline(x=np.mean(real_caps), color='orange', linestyle='--', alpha=0.7, label=f'Mean={np.mean(real_caps):.0f}')
    ax1.legend(fontsize=8)

    # Rank heatmap
    n = len(real_schools)
    rank_matrix = np.zeros((n, 3))
    for i, s in enumerate(real_schools):
        for j, d in enumerate([1, 2, 3]):
            rank_matrix[i, j] = ranks[(d, s)]

    im = ax2.imshow(rank_matrix, cmap='RdYlGn_r', aspect='auto', vmin=1, vmax=20)
    ax2.set_xticks([0, 1, 2])
    ax2.set_xticklabels(['District 1', 'District 2', 'District 3'], fontweight='bold')
    ax2.set_yticks(range(n))
    ax2.set_yticklabels([f'{s}  (cap={real_caps[i]})' for i, s in enumerate(real_schools)],
                         fontsize=7.5, fontfamily='monospace')
    ax2.set_title('Preference Rank by District\n(1=most preferred, 20=least)', fontweight='bold')
    for i in range(n):
        for j in range(3):
            r = int(rank_matrix[i, j])
            color = 'white' if r <= 3 or r >= 18 else 'black'
            ax2.text(j, i, str(r), ha='center', va='center', fontsize=8, color=color, fontweight='bold')
    plt.colorbar(im, ax=ax2, shrink=0.8, label='Rank (1=top)')

    plt.tight_layout()
    plt.savefig(f'{EXP_OUT_FOLDER}capacity_and_rankings.png', dpi=200, bbox_inches='tight')
    plt.show()

def log_and_print(message, log_file=None):
    """Print to console and optionally write to file with immediate flush"""
    text = str(message)
    print(text, flush=True)
    if log_file is not None:
        log_dir = os.path.dirname(log_file)
        if log_dir:
            os.makedirs(log_dir, exist_ok=True)
        f = open(log_file, "a+", buffering=1)
        f.write(text + '\n')
        f.flush()
        f.close()