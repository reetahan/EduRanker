

import argparse
import json
import re
import pandas as pd
import numpy as np

from src.project_specific_scripts.config_priority_generator import make_tier, make_reserve, validate_config


BOROUGH_PRIORITY_STRINGS_INV = {
    "Manhattan students or residents": "M",
    "Bronx students or residents":     "X",
    "Brooklyn students or residents":  "K",
    "Queens students or residents":    "Q",
    "Staten Island students or residents": "R",
}

CONTINUING_KEYWORDS = [
    "Continuing 8th graders",
    "continuing 8th graders",
    "Priority to continuing 8th graders",
]

SPECIAL_PROGRAM_PATTERNS = {
    "D75":         "District 75",
    "ASD_nest":    "ASD (Autism Spectrum Disorder) Nest",
    "ASD_horizon": "ASD (Autism Spectrum Disorder) Horizon",
    "ACES":        "ACES",
}

MAX_PROGRAMS = 11


def build_continuing_fraction_lookup(data1_path, data2_path, data3_path):
    def to_numeric(x):
        try:
            return float(str(x).replace('s^', '').replace('s', '').replace(',', '').strip())
        except (ValueError, TypeError):
            return np.nan

    df3 = pd.read_excel(data3_path, sheet_name='Match to Choice-District', header=0)
    df3.columns = df3.iloc[0]
    df3 = df3.iloc[1:].reset_index(drop=True)
    total_row = df3[df3.iloc[:, 0].astype(str).str.strip() == 'Total']
    total_applicants = int(str(total_row.iloc[0, 1]).replace(',', '').strip())

    df2 = pd.read_excel(data2_path, sheet_name='School')
    df2_all = df2[df2['Category'] == 'All Students'][['School DBN', 'Grade 9 Students']].copy()
    df2_all.columns = ['School DBN', 'enrolled_grade9']
    df2_all['enrolled_grade9'] = df2_all['enrolled_grade9'].apply(to_numeric)

    df1 = pd.read_excel(data1_path, sheet_name='School')
    df1_all = df1[df1['Category'] == 'All Students'][['School DBN', 'Grade 9 Seats Available']].copy()
    df1_all.columns = ['School DBN', 'seats']
    df1_all['seats'] = df1_all['seats'].apply(to_numeric)

    merged = df2_all.merge(df1_all, on='School DBN')
    merged = merged.dropna(subset=['enrolled_grade9', 'seats'])
    merged = merged[merged['seats'] > 0]
    merged['p'] = merged['enrolled_grade9'] / total_applicants

    lookup = merged.set_index('School DBN')['p'].to_dict()
    fallback = float(merged[merged['enrolled_grade9'] > 0]['p'].mean())
    return lookup, fallback, total_applicants


def parse_dia_fraction(diadetails_str):
    if pd.isna(diadetails_str):
        return None
    m = re.search(r"(\d+)%", str(diadetails_str))
    return round(int(m.group(1)) / 100, 4) if m else None


def parse_priority_tiers(row, prog_idx, continuing_p):
    tiers = []
    tier_rank = 1

    for p in range(1, 4):
        val = str(row.get(f"priority{p}_prog{prog_idx}", "")).strip()
        if not val or val == "nan":
            continue

        if any(kw in val for kw in CONTINUING_KEYWORDS):
            tiers.append(make_tier(
                tier=tier_rank, group="continuing", description=val,
                fraction_eligible=continuing_p, school_dependent=True,
            ))
            tier_rank += 1
            continue

        matched_boro = next(
            (code for label, code in BOROUGH_PRIORITY_STRINGS_INV.items() if label in val),
            None
        )
        if matched_boro:
            tiers.append(make_tier(
                tier=tier_rank, group="borough", description=val,
                fraction_eligible=None, school_dependent=True,
                borough_code=matched_boro,
            ))
            tier_rank += 1
            continue

        matched_special = next(
            (key for key, pattern in SPECIAL_PROGRAM_PATTERNS.items() if pattern in val),
            None
        )
        if matched_special:
            tiers.append(make_tier(
                tier=tier_rank, group="special_program", description=val,
                fraction_eligible=None, school_dependent=True,
                special_type=matched_special,
            ))
            tier_rank += 1
            continue

        if "students" in val.lower() and matched_boro is None and "New York City" not in val:
            tiers.append(make_tier(
                tier=tier_rank, group="feeder_school", description=val,
                fraction_eligible=None, school_dependent=True,
            ))
            tier_rank += 1
            continue

        if "New York City residents" in val or "Then to New York City residents" in val:
            continue

    tiers.append(make_tier(
        tier=tier_rank, group="all_nyc",
        description="New York City residents",
        fraction_eligible=1.0, school_dependent=False,
    ))
    return tiers


def build_nyc_config(df, continuing_lookup, continuing_fallback, total_applicants):
    school_overrides = {}

    for _, row in df.iterrows():
        dbn = str(row["dbn"]).strip()
        boro_code = str(row.get("boro", "")).strip()
        dia_fraction = parse_dia_fraction(row.get("diadetails"))

        if dbn in continuing_lookup:
            continuing_p = round(continuing_lookup[dbn], 6)
            continuing_source = "enrolled_grade9_over_citywide"
        else:
            continuing_p = round(continuing_fallback, 6)
            continuing_source = "fallback_mean"

        for prog_idx in range(1, MAX_PROGRAMS + 1):
            method = str(row.get(f"method{prog_idx}", "")).strip()
            if not method or method == "nan":
                continue

            def safe_int(col):
                try:
                    return int(float(row.get(col, 0) or 0))
                except (ValueError, TypeError):
                    return 0

            seats_ge  = safe_int(f"seats9ge{prog_idx}")
            seats_swd = safe_int(f"seats9swd{prog_idx}")
            total_seats = seats_ge + seats_swd
            swd_fraction = round(seats_swd / total_seats, 4) if total_seats > 0 else None

            prog_has_continuing = any(
                any(kw in str(row.get(f"priority{p}_prog{prog_idx}", ""))
                    for kw in CONTINUING_KEYWORDS)
                for p in range(1, 4)
            )

            priority_tiers = parse_priority_tiers(
                row=row, prog_idx=prog_idx,
                continuing_p=continuing_p if prog_has_continuing else None,
            )

            reserves = {}
            if swd_fraction is not None:
                reserves["SWD"] = make_reserve(
                    group="SWD", fraction=swd_fraction, seats=seats_swd,
                    description="Students with disabilities reserved seats",
                )
            if dia_fraction is not None:
                dia_seats = round(dia_fraction * total_seats) if total_seats > 0 else None
                reserves["DIA"] = make_reserve(
                    group="DIA", fraction=dia_fraction, seats=dia_seats,
                    description=str(row.get("diadetails", "")).strip(),
                )

            school_overrides[f"{dbn}_prog{prog_idx}"] = {
                "dbn": dbn,
                "program_index": prog_idx,
                "borough": boro_code,
                "method": method,
                "seats_ge": seats_ge,
                "seats_swd": seats_swd,
                "total_seats": total_seats,
                "priority_tiers": priority_tiers,
                "reserves": reserves,
                "continuing_fraction_source": continuing_source if prog_has_continuing else None,
            }

    return {
        "__meta__": {
            "system": "NYC",
            "id_format": "dbn_prog",
            "legal_basis": "NYC DOE High School Admissions Policy",
            "total_citywide_applicants": total_applicants,
            "continuing_fallback_p": round(continuing_fallback, 6),
            "granularity": {
                "priority_tiers": "school",
                "reserves": "school",
                "student_fractions": "school",
            },
        },
        "system_defaults": {
            "priority_tiers": [],
            "reserves": {},
            "student_attribute_fractions": {},
        },
        "region_overrides": {},
        "school_overrides": school_overrides,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input",  required=True)
    parser.add_argument("--data1",  required=True)
    parser.add_argument("--data2",  required=True)
    parser.add_argument("--data3",  required=True)
    parser.add_argument("--output", default="nyc_priority_config.json")
    parser.add_argument("--sheet",  default="Data")
    args = parser.parse_args()

    print("Building continuing fraction lookup ...")
    continuing_lookup, continuing_fallback, total_applicants = \
        build_continuing_fraction_lookup(args.data1, args.data2, args.data3)
    print(f"  Total citywide applicants: {total_applicants:,}")
    print(f"  Schools with data: {len(continuing_lookup)}")
    print(f"  Fallback p: {continuing_fallback:.6f}")

    print(f"\nReading {args.input} ...")
    df = pd.read_excel(args.input, sheet_name=args.sheet)
    print(f"  {len(df)} schools loaded.")

    config = build_nyc_config(df, continuing_lookup, continuing_fallback, total_applicants)

    warnings = validate_config(config)
    if warnings:
        print("\nValidation warnings:")
        for w in warnings:
            print(f"  ! {w}")

    so = config["school_overrides"]
    print(f"\nConfig summary:")
    print(f"  Total programs:           {len(so)}")
    print(f"  With borough priority:    {sum(1 for v in so.values() if any(t['group'] == 'borough' for t in v['priority_tiers']))}")
    print(f"  With continuing priority: {sum(1 for v in so.values() if any(t['group'] == 'continuing' for t in v['priority_tiers']))}")
    print(f"  With SWD reserve:         {sum(1 for v in so.values() if 'SWD' in v['reserves'])}")
    print(f"  With DIA reserve:         {sum(1 for v in so.values() if 'DIA' in v['reserves'])}")

    with open(args.output, "w") as f:
        json.dump(config, f, indent=2, default=str)
    print(f"\nWritten to {args.output}")


if __name__ == "__main__":
    main()