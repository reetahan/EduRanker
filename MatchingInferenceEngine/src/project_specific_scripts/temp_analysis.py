"""
compare_mtb_stb_female.py

Loads student_level.csv from MTB and STB welfare runs and plots
top-p comparison for female students.
"""

import pandas as pd
import matplotlib.pyplot as plt
from welfare import summarize_global_sweep, summarize_rank_stats_by_category

MTB_PATH = '/scratch/rm6609/EduRanker/MatchingInferenceEngine/experiment-results/chile_res_logs/20260420_114057/student_level.csv'
STB_PATH = '/scratch/rm6609/EduRanker/MatchingInferenceEngine/experiment-results/chile_res_logs/20260420_114600/chilean_experiment_K=6_M=10_iter=10_opt=10_seed=40_20260420_114600_STB_COUNTERFACTUAL_welfare/student_level.csv'
mtb = pd.read_csv(MTB_PATH)
stb = pd.read_csv(STB_PATH)

max_p = 10

# Slice female
mtb_f = mtb[mtb['female'] == True]
stb_f = stb[stb['female'] == True]

mtb_sweep = summarize_global_sweep(mtb_f, max_p=max_p)
stb_sweep = summarize_global_sweep(stb_f, max_p=max_p)

merged = mtb_sweep[['p', 'top_p_pct']].rename(columns={'top_p_pct': 'MTB_%'}).merge(
    stb_sweep[['p', 'top_p_pct']].rename(columns={'top_p_pct': 'STB_%'}), on='p'
)
merged['STB-MTB'] = (merged['STB_%'] - merged['MTB_%']).round(2)
print(merged.to_string(index=False))

print(f"MTB female: {len(mtb_f):,} students, {mtb_f['matched'].mean()*100:.1f}% matched")
print(f"STB female: {len(stb_f):,} students, {stb_f['matched'].mean()*100:.1f}% matched")

fig, ax = plt.subplots(figsize=(9, 6))
ax.plot(mtb_sweep['p'], mtb_sweep['top_p_pct'], marker='o', color='#0072B2', linewidth=2, label='MTB (per-school lottery)')
ax.plot(stb_sweep['p'], stb_sweep['top_p_pct'], marker='s', color='#E69F00', linewidth=2, label='STB (global lottery)')
ax.set_xlabel("p (top-p threshold)", fontsize=13)
ax.set_ylabel("Female students matched to top-p choice (%)", fontsize=13)
ax.set_title("Synthetic welfare comparison for female students\nMTB vs STB — inferred Mallows params", fontsize=13)
ax.set_ylim(0, 100)
ax.legend(fontsize=11)
ax.grid(True, alpha=0.3)
plt.tight_layout()
fig.savefig('synthetic_female_mtb_stb.png', dpi=200, bbox_inches='tight')
print("Saved plot.")