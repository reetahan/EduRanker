import pandas as pd
import numpy as np

# Load the spreadsheet
file_path = 'DATA4_fall-2025---hs-directory-data.xlsx'
df = pd.read_excel(file_path, sheet_name='Data')

# 1. Global Counts
num_schools = len(df)
# Identify program name columns (program1, program2, ... program11)
program_cols = [f'program{i}' for i in range(1, 12) if f'program{i}' in df.columns]
total_programs = df[program_cols].notna().sum().sum()

# 2. Programs per School Statistics
df['num_programs'] = df[program_cols].notna().sum(axis=1)
prog_per_school_stats = df['num_programs'].describe()[['mean', '50%', 'max', 'min']].rename({'50%': 'median'})

# 3. Seats Analysis (9th Grade General Education + Students with Disabilities)
program_seats_list = []
school_total_seats = pd.Series(0, index=df.index)

for i in range(1, 12):
    ge_col = f'seats9ge{i}'
    swd_col = f'seats9swd{i}'
    prog_col = f'program{i}'
    
    if ge_col in df.columns and swd_col in df.columns:
        # Sum GE and SWD seats, treating missing values as 0
        seats_in_prog = df[ge_col].fillna(0) + df[swd_col].fillna(0)
        school_total_seats += seats_in_prog
        
        # Only collect seat counts for programs that are actually listed (not NaN)
        valid_seats = seats_in_prog[df[prog_col].notna()]
        program_seats_list.append(valid_seats)

# Aggregate all program-level seats into one series
all_prog_seats = pd.concat(program_seats_list) if program_seats_list else pd.Series(dtype=float)

# 4. Categorical Distributions (Top 15)
def get_top_15(dataframe, search_term):
    # Dynamically find columns (e.g., all containing 'method', 'eligibility', or 'priority')
    relevant_cols = [col for col in dataframe.columns if search_term in col.lower()]
    return dataframe[relevant_cols].stack().value_counts().head(15)

method_dist = get_top_15(df, 'method')
eligibility_dist = get_top_15(df, 'eligibility')
priority_dist = get_top_15(df, 'priority')

# --- FINAL REPORT ---
print(f"Total Schools in Dataset: {num_schools}")
print(f"Total Programs in Dataset: {total_programs}")

print("\n--- Programs per School Statistics ---")
print(prog_per_school_stats.to_string())

print("\n--- Seats per Program Statistics ---")
print(all_prog_seats.describe()[['mean', '50%', 'max', 'min']].rename({'50%': 'median'}).to_string())

print("\n--- Seats per School Statistics ---")
print(school_total_seats.describe()[['mean', '50%', 'max', 'min']].rename({'50%': 'median'}).to_string())

print("\n--- Top 15 Admission Methods ---")
print(method_dist)

print("\n--- Top 15 Eligibilities ---")
print(eligibility_dist)

print("\n--- Top 15 Priorities ---")
print(priority_dist)

import pandas as pd
import numpy as np

# Load the data
file_path = 'DATA4_fall-2025---hs-directory-data.xlsx'
df = pd.read_excel(file_path, sheet_name='Data')

# 1. Individual Method Distribution (Top 15 per column)
method_cols = [f'method{i}' for i in range(1, 12) if f'method{i}' in df.columns]

print("--- Distribution per Method Column (Top 5 Results per Slot) ---")
for col in method_cols:
    counts = df[col].value_counts().head(5)
    if not counts.empty:
        print(f"\n[{col.upper()}]")
        print(counts.to_string())

# 2. Aggregated Pivot Report
# This creates a table where rows are method types and columns are Method1, Method2...
all_methods = []
for col in method_cols:
    all_methods.append(df[col].value_counts())

method_pivot = pd.concat(all_methods, axis=1).fillna(0).astype(int)
method_pivot.columns = [col.upper() for col in method_cols]

print("\n--- Summary Pivot Table (All Methods) ---")
# Showing the top rows for brevity
print(method_pivot.head(15))

# 3. Calculate Global totals from the pivot
print("\n--- Global Totals across all Method Columns ---")
print(method_pivot.sum(axis=1).sort_values(ascending=False).head(15))