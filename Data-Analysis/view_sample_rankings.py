import pandas as pd
from pathlib import Path

def view_zip_ranking(zip_code='11211', save_csv=True):
    """View ranking for a specific zip code"""
    print("="*80)
    print(f"RANKING FOR ZIP CODE: {zip_code}")
    print("="*80)
    
    df = pd.read_csv(Path(__file__).parent / "output" / "zip_code_aggregates.csv")
    
    # Ensure zip_code is a string for comparison
    zip_code = str(zip_code)
    
    # Filter to specific zip code
    zip_data = df[df['Home Zip Code'] == zip_code].copy()
    
    if len(zip_data) == 0:
        print(f"\nNo data found for zip code {zip_code}")
        return
    
    # Identify suppressed values and re-rank properly
    zip_data['_is_suppressed'] = (zip_data['True Applicants Zip'] == 1) & (zip_data['Total Applicants Zip'] == 1)
    zip_data['_rank_ratio'] = zip_data.apply(
        lambda row: -1 if row['_is_suppressed'] else row['Ratio'],
        axis=1
    )
    
    # Re-rank based on corrected ratio
    zip_data['Rank'] = zip_data['_rank_ratio'].rank(ascending=False, method='min').astype(int)
    
    # Sort by rank
    zip_data = zip_data.sort_values('Rank')
    
    # Clean up temporary columns
    zip_data = zip_data.drop(columns=['_is_suppressed', '_rank_ratio'])
    
    print(f"\nTotal schools ranked by zip code {zip_code}: {len(zip_data)}")
    print(f"\nInterpretation:")
    print(f"  - Rank: Lower number = more popular with students from this zip code")
    print(f"  - Ratio: True Applicants / Total Applicants (higher = more popular)")
    print(f"  - True Applicants: # of students from {zip_code} who ranked this school")
    print(f"  - Total Applicants: Total # of students from {zip_code} eligible to apply")
    
    print(f"\n{'Rank':<6} {'School DBN':<12} {'School Name':<55} {'Ratio':<8} {'True':<7} {'Total':<7}")
    print("-" * 110)
    
    # Show top 20
    for idx, row in zip_data.head(20).iterrows():
        print(f"{int(row['Rank']):<6} {row['School DBN']:<12} {row['School Name'][:53]:<55} {row['Ratio']:<8.4f} {int(row['True Applicants Zip']):<7} {int(row['Total Applicants Zip']):<7}")
    
    if len(zip_data) > 20:
        print(f"\n... and {len(zip_data) - 20} more schools")
    
    # Save to CSV
    if save_csv:
        output_path = Path(__file__).parent / "output" / f"ranking_zip_{zip_code}.csv"
        zip_data.to_csv(output_path, index=False)
        print(f"\n✓ Saved full ranking to: {output_path}")
    
    print(f"\n" + "="*80)


def view_language_ranking(language='BENGALI', save_csv=True):
    """View ranking for a specific language"""
    print("="*80)
    print(f"RANKING FOR HOME LANGUAGE: {language}")
    print("="*80)
    
    df = pd.read_csv(Path(__file__).parent / "output" / "language_aggregates.csv")
    
    # Filter to specific language (case-insensitive)
    lang_data = df[df['Home Language'].str.upper() == language.upper()].copy()
    
    if len(lang_data) == 0:
        print(f"\nNo data found for language '{language}'")
        print("\nAvailable languages:")
        languages = sorted(df['Home Language'].unique())
        for i, lang in enumerate(languages[:30]):
            print(f"  {lang}", end="")
            if (i + 1) % 3 == 0:
                print()
        if len(languages) > 30:
            print(f"\n  ... and {len(languages) - 30} more languages")
        return
    
    # Identify suppressed values and re-rank properly
    lang_data['_is_suppressed'] = (lang_data['True Applicants Language'] == 1) & (lang_data['Total Applicants Language'] == 1)
    lang_data['_rank_ratio'] = lang_data.apply(
        lambda row: -1 if row['_is_suppressed'] else row['Ratio'],
        axis=1
    )
    
    # Re-rank based on corrected ratio
    lang_data['Rank'] = lang_data['_rank_ratio'].rank(ascending=False, method='min').astype(int)
    
    # Sort by rank
    lang_data = lang_data.sort_values('Rank')
    
    # Clean up temporary columns
    lang_data = lang_data.drop(columns=['_is_suppressed', '_rank_ratio'])
    
    print(f"\nTotal schools ranked by {language} speakers: {len(lang_data)}")
    print(f"\nInterpretation:")
    print(f"  - Rank: Lower number = more popular with {language}-speaking students")
    print(f"  - Ratio: True Applicants / Total Applicants (higher = more popular)")
    print(f"  - True Applicants: # of {language} speakers who ranked this school")
    print(f"  - Total Applicants: Total # of {language} speakers eligible to apply")
    
    print(f"\n{'Rank':<6} {'School DBN':<12} {'School Name':<55} {'Ratio':<8} {'True':<7} {'Total':<7}")
    print("-" * 110)
    
    # Show top 20
    for idx, row in lang_data.head(20).iterrows():
        print(f"{int(row['Rank']):<6} {row['School DBN']:<12} {row['School Name'][:53]:<55} {row['Ratio']:<8.4f} {int(row['True Applicants Language']):<7} {int(row['Total Applicants Language']):<7}")
    
    if len(lang_data) > 20:
        print(f"\n... and {len(lang_data) - 20} more schools")
    
    # Save to CSV
    if save_csv:
        output_path = Path(__file__).parent / "output" / f"ranking_language_{language.lower()}.csv"
        lang_data.to_csv(output_path, index=False)
        print(f"\n✓ Saved full ranking to: {output_path}")
    
    print(f"\n" + "="*80)


def view_district_ranking(district='01', save_csv=True):
    """View ranking for a specific residential district"""
    print("="*80)
    print(f"RANKING FOR RESIDENTIAL DISTRICT: {district}")
    print("="*80)
    
    df = pd.read_csv(Path(__file__).parent / "output" / "residential_district.csv")
    
    # Filter to specific district
    district_data = df[df['Residential District'] == district].copy()
    
    if len(district_data) == 0:
        print(f"\nNo data found for district {district}")
        print("\nAvailable districts:")
        districts = sorted(df['Residential District'].unique())
        print(", ".join(str(d) for d in districts))
        return
    
    # Identify suppressed values and re-rank properly
    district_data['_is_suppressed'] = (district_data['True Applicants Residential District'] == 1) & (district_data['Total Applicants Residential District'] == 1)
    district_data['_rank_ratio'] = district_data.apply(
        lambda row: -1 if row['_is_suppressed'] else row['Ratio'],
        axis=1
    )
    
    # Re-rank based on corrected ratio
    district_data['Rank'] = district_data['_rank_ratio'].rank(ascending=False, method='min').astype(int)
    
    # Sort by rank
    district_data = district_data.sort_values('Rank')
    
    # Clean up temporary columns
    district_data = district_data.drop(columns=['_is_suppressed', '_rank_ratio'])
    
    print(f"\nTotal schools ranked by District {district} residents: {len(district_data)}")
    print(f"\nInterpretation:")
    print(f"  - Rank: Lower number = more popular with District {district} residents")
    print(f"  - Ratio: True Applicants / Total Applicants (higher = more popular)")
    print(f"  - True Applicants: # of District {district} students who ranked this school")
    print(f"  - Total Applicants: Total # of District {district} students eligible to apply")
    
    print(f"\n{'Rank':<6} {'School DBN':<12} {'School Name':<55} {'Ratio':<8} {'True':<7} {'Total':<7}")
    print("-" * 110)
    
    # Show top 20
    for idx, row in district_data.head(20).iterrows():
        print(f"{int(row['Rank']):<6} {row['School DBN']:<12} {row['School Name'][:53]:<55} {row['Ratio']:<8.4f} {int(row['True Applicants Residential District']):<7} {int(row['Total Applicants Residential District']):<7}")
    
    if len(district_data) > 20:
        print(f"\n... and {len(district_data) - 20} more schools")
    
    # Save to CSV
    if save_csv:
        output_path = Path(__file__).parent / "output" / f"ranking_district_{district}.csv"
        district_data.to_csv(output_path, index=False)
        print(f"\n✓ Saved full ranking to: {output_path}")
    
    print(f"\n" + "="*80)


if __name__ == "__main__":
    # Show sample rankings
    view_zip_ranking('11211')
    print("\n\n")
    view_language_ranking('FRENCH')
    print("\n\n")
    
    # You can also view other categories:
    # view_zip_ranking('10001')  # Different zip code
    # view_language_ranking('SPANISH')  # Different language
    # view_district_ranking('02')  # Residential district
