import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 6)

# Load data
data_path = Path(__file__).parent / "master_data.xlsx"
print(f"Loading data from: {data_path}")
df = pd.read_excel(data_path)

print("\n" + "="*80)
print("EXPLORATORY DATA ANALYSIS - Master Data")
print("="*80)

# ============================================================================
# 1. BASIC INFORMATION
# ============================================================================
print("\n" + "-"*80)
print("1. DATASET OVERVIEW")
print("-"*80)
print(f"Shape: {df.shape[0]:,} rows × {df.shape[1]} columns")
print(f"\nColumn Names and Types:")
print(df.dtypes)

print(f"\nMemory Usage:")
print(df.memory_usage(deep=True).sum() / 1024**2, "MB")

# ============================================================================
# 2. FIRST/LAST ROWS
# ============================================================================
print("\n" + "-"*80)
print("2. FIRST 5 ROWS")
print("-"*80)
print(df.head())

print("\n" + "-"*80)
print("3. LAST 5 ROWS")
print("-"*80)
print(df.tail())

# ============================================================================
# 4. MISSING VALUES
# ============================================================================
print("\n" + "-"*80)
print("4. MISSING VALUES ANALYSIS")
print("-"*80)
missing = df.isnull().sum()
missing_pct = 100 * df.isnull().sum() / len(df)
missing_table = pd.DataFrame({
    'Missing_Count': missing,
    'Percent': missing_pct
})
missing_table = missing_table[missing_table['Missing_Count'] > 0].sort_values('Missing_Count', ascending=False)

if len(missing_table) > 0:
    print(missing_table)
    
    # Visualize missing data
    plt.figure(figsize=(12, 6))
    missing_table['Percent'].plot(kind='barh')
    plt.xlabel('Percentage Missing')
    plt.title('Missing Data by Column')
    plt.tight_layout()
    plt.savefig('output/missing_data.png', dpi=300, bbox_inches='tight')
    print("\n✓ Saved: output/missing_data.png")
    plt.close()
else:
    print("No missing values found!")

# ============================================================================
# 5. NUMERICAL COLUMNS STATISTICS
# ============================================================================
print("\n" + "-"*80)
print("5. NUMERICAL COLUMNS SUMMARY")
print("-"*80)
numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
print(f"Found {len(numeric_cols)} numerical columns")
if len(numeric_cols) > 0:
    print(df[numeric_cols].describe())
    
    # Distribution plots for numerical columns
    n_cols = len(numeric_cols)
    if n_cols > 0:
        n_rows = (n_cols + 2) // 3
        fig, axes = plt.subplots(n_rows, 3, figsize=(15, 5*n_rows))
        axes = axes.flatten() if n_cols > 1 else [axes]
        
        for idx, col in enumerate(numeric_cols):
            if idx < len(axes):
                df[col].hist(bins=50, ax=axes[idx], edgecolor='black', alpha=0.7)
                axes[idx].set_title(f'{col}\n(μ={df[col].mean():.2f}, σ={df[col].std():.2f})')
                axes[idx].set_xlabel(col)
                axes[idx].set_ylabel('Frequency')
        
        # Hide extra subplots
        for idx in range(len(numeric_cols), len(axes)):
            axes[idx].axis('off')
        
        plt.tight_layout()
        plt.savefig('output/numerical_distributions.png', dpi=300, bbox_inches='tight')
        print("\n✓ Saved: output/numerical_distributions.png")
        plt.close()

# ============================================================================
# 6. CATEGORICAL COLUMNS ANALYSIS
# ============================================================================
print("\n" + "-"*80)
print("6. CATEGORICAL COLUMNS SUMMARY")
print("-"*80)
categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
print(f"Found {len(categorical_cols)} categorical columns")

if len(categorical_cols) > 0:
    for col in categorical_cols:
        n_unique = df[col].nunique()
        print(f"\n{col}:")
        print(f"  Unique values: {n_unique}")
        if n_unique <= 20:
            print(f"  Value counts:")
            print(df[col].value_counts().head(20))
        else:
            print(f"  Top 10 values:")
            print(df[col].value_counts().head(10))

# ============================================================================
# 7. CORRELATION ANALYSIS
# ============================================================================
if len(numeric_cols) > 1:
    print("\n" + "-"*80)
    print("7. CORRELATION MATRIX")
    print("-"*80)
    
    corr = df[numeric_cols].corr()
    print(corr)
    
    # Heatmap
    plt.figure(figsize=(12, 10))
    sns.heatmap(corr, annot=True, cmap='coolwarm', center=0, 
                square=True, linewidths=1, cbar_kws={"shrink": 0.8},
                fmt='.2f')
    plt.title('Correlation Heatmap')
    plt.tight_layout()
    plt.savefig('output/correlation_heatmap.png', dpi=300, bbox_inches='tight')
    print("\n✓ Saved: output/correlation_heatmap.png")
    plt.close()

# ============================================================================
# 8. DUPLICATES
# ============================================================================
print("\n" + "-"*80)
print("8. DUPLICATE ROWS")
print("-"*80)
duplicates = df.duplicated().sum()
print(f"Number of duplicate rows: {duplicates}")
if duplicates > 0:
    print(f"Percentage: {100 * duplicates / len(df):.2f}%")

# ============================================================================
# 9. DATA QUALITY ISSUES
# ============================================================================
print("\n" + "-"*80)
print("9. DATA QUALITY CHECKS")
print("-"*80)

# Check for infinite values in numeric columns
if len(numeric_cols) > 0:
    inf_counts = {}
    for col in numeric_cols:
        inf_count = np.isinf(df[col]).sum()
        if inf_count > 0:
            inf_counts[col] = inf_count
    
    if inf_counts:
        print("Columns with infinite values:")
        for col, count in inf_counts.items():
            print(f"  {col}: {count}")
    else:
        print("✓ No infinite values found")

# Check for constant columns
constant_cols = [col for col in df.columns if df[col].nunique() <= 1]
if constant_cols:
    print(f"\nConstant columns (single value): {constant_cols}")
else:
    print("✓ No constant columns")

# ============================================================================
# 10. EXPORT SUMMARY
# ============================================================================
print("\n" + "-"*80)
print("10. EXPORTING SUMMARY")
print("-"*80)

# Create output directory
output_dir = Path(__file__).parent / "output"
output_dir.mkdir(exist_ok=True)

# Save summary statistics
summary_path = output_dir / "summary_statistics.csv"
df.describe(include='all').to_csv(summary_path)
print(f"✓ Saved: {summary_path}")

# Save column info
col_info = pd.DataFrame({
    'Column': df.columns,
    'Type': df.dtypes.values,
    'Non_Null_Count': df.count().values,
    'Null_Count': df.isnull().sum().values,
    'Null_Percentage': (100 * df.isnull().sum() / len(df)).values,
    'Unique_Values': [df[col].nunique() for col in df.columns]
})
col_info_path = output_dir / "column_info.csv"
col_info.to_csv(col_info_path, index=False)
print(f"✓ Saved: {col_info_path}")

print("\n" + "="*80)
print("EDA COMPLETE!")
print("="*80)
print(f"\nGenerated files in '{output_dir}/':")
print("  - summary_statistics.csv")
print("  - column_info.csv")
print("  - missing_data.png (if applicable)")
print("  - numerical_distributions.png (if applicable)")
print("  - correlation_heatmap.png (if applicable)")
