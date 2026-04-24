import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('sweep_summary.csv')

fig, axes = plt.subplots(1, 3, figsize=(14, 5))

axes[0].plot(df['list_length_min'], df['pct_matched'], marker='o', color="#DF34F1")
axes[0].set_xlabel('Minimum list length')
axes[0].set_ylabel('% Matched')
axes[0].set_title('Match rate')
axes[0].grid(True, alpha=0.3)
axes[0].set_ylim(85, 90)

axes[1].plot(df['list_length_min'], df['avg_rank'], marker='o', color="#E69F00")
axes[1].set_xlabel('Minimum list length')
axes[1].set_ylabel('Average rank')
axes[1].set_title('Average rank of match')
axes[1].grid(True, alpha=0.3)
axes[1].set_ylim(2, 3)

axes[2].plot(df['list_length_min'], df['rank_variance'], marker='o', color="#009E73")
axes[2].set_xlabel('Minimum list length')
axes[2].set_ylabel('Rank variance')
axes[2].set_title('Rank variance (inequity)')
axes[2].grid(True, alpha=0.3)
axes[2].set_ylim(2, 6)

plt.suptitle('Effect of minimum list length on welfare in NYC', fontsize=13)
plt.tight_layout()
plt.savefig('min_list_length_sweep.png', dpi=200, bbox_inches='tight')
print("Saved.")