"""
real_chile_priority_generator.py

Inputs:
  individual_level_preferences_and_result.xlsx  per-student priority flags
  school_capacity.xlsx                          per-school seat structure

Priority tiers per region (Ley 20.845 + Correa et al. 2019):
  1. sibling, 2. working_parent, 3. returning_student, 4. all

Quota processing order (before priority tiers):
  1. special_needs       <=2 seats/classroom, validated schools only
  2. academic_excellence 30-85% seats, MINEDUC-preselected schools
  3. disadvantaged       15% seats, per school

priority_already_registered is NOT a DA priority tier. The law guarantees
unmatched transfer students a spot at their current school post-match.

Usage:
    python real_chile_priority_generator.py \
        --individual  individual_level_preferences_and_result.xlsx \
        --capacity    school_capacity.xlsx \
        --output      chile_priority_config.json
"""

import argparse
import json
import pandas as pd
import numpy as np

from src.project_specific_scripts.config_priority_generator import make_tier, make_reserve, validate_config


PRIORITY_COLS = [
    'priority_student',
    'priority_sibling',
    'priority_parent_civil_servant',
    'priority_ex_student',
    'priority_already_registered',
    'high_performance_student',
    'integration_program_status_existing',
]

SYSTEM_DEFAULT_TIERS = [
    make_tier(tier=1, group="sibling",
              description="Blood sibling enrolled/admitted at school",
              fraction_eligible=None, school_dependent=True),
    make_tier(tier=2, group="working_parent",
              description="Parent works permanently at school",
              fraction_eligible=None, school_dependent=True),
    make_tier(tier=3, group="returning_student",
              description="Previously enrolled, not expelled",
              fraction_eligible=None, school_dependent=True),
    make_tier(tier=4, group="all",
              description="General pool, ordered by lottery",
              fraction_eligible=1.0, school_dependent=False),
]


def build_region_fractions(df):
    student_df = df.groupby(['mrun', 'Region'])[PRIORITY_COLS].max().reset_index()

    region_overrides = {}
    for region, grp in student_df.groupby('Region'):
        fracs = {col: round(float(grp[col].mean()), 6) for col in PRIORITY_COLS}
        female_frac = float(
            df[df['Region'] == region]
            .groupby('mrun')['female'].first()
            .mean()
        )
        region_overrides[region] = {
            "n_students": int(len(grp)),
            "priority_tiers": [
                make_tier(tier=1, group="sibling",
                          description="Blood sibling enrolled/admitted at school",
                          fraction_eligible=fracs['priority_sibling'],
                          school_dependent=True),
                make_tier(tier=2, group="working_parent",
                          description="Parent works permanently at school",
                          fraction_eligible=fracs['priority_parent_civil_servant'],
                          school_dependent=True),
                make_tier(tier=3, group="returning_student",
                          description="Previously enrolled, not expelled",
                          fraction_eligible=fracs['priority_ex_student'],
                          school_dependent=True),
                make_tier(tier=4, group="all",
                          description="General pool, ordered by lottery",
                          fraction_eligible=1.0, school_dependent=False),
            ],
            "student_attribute_fractions": {
                "disadvantaged":      fracs['priority_student'],
                "high_performance":   fracs['high_performance_student'],
                "special_needs":      fracs['integration_program_status_existing'],
                "already_registered": fracs['priority_already_registered'],
                "female": female_frac,
            },
            "reserves": {},
        }

    total_students = int(student_df['mrun'].nunique())
    return region_overrides, total_students


def build_school_overrides(capacity_path, rbd_to_region):
    df = pd.read_excel(capacity_path)
    school_overrides = {}

    for _, row in df.iterrows():
        try:
            rbd = int(float(row['rbd']))
            region = rbd_to_region.get(int(rbd), None)
            program_code = str(int(float(row['program_code'])))
            total = float(row.get('total_admission_seats', 0) or 0)
        except (ValueError, TypeError):
            continue
        if total <= 0:
            continue

        def safe_int(col):
            try:
                return int(float(row.get(col, 0) or 0))
            except (ValueError, TypeError):
                return 0

        reserves = {}

        integration_seats = safe_int('integration_student_seats')
        if integration_seats > 0:
            reserves["special_needs"] = make_reserve(
                group="special_needs",
                fraction=round(integration_seats / total, 6),
                seats=integration_seats,
                description="Special educational needs (PIE), <=2 seats/classroom",
            )

        excellence_seats = (safe_int('high_selectivity_seats_transitional')
                            + safe_int('high_selectivity_seats_ranking'))
        if excellence_seats > 0:
            reserves["academic_excellence"] = make_reserve(
                group="academic_excellence",
                fraction=round(excellence_seats / total, 6),
                seats=excellence_seats,
                description="High academic performance (30-85\% seats, MINEDUC-selected schools)",
            )

        priority_seats = safe_int('priority_student_seats')
        if priority_seats > 0:
            reserves["disadvantaged"] = make_reserve(
                group="disadvantaged",
                fraction=round(priority_seats / total, 6),
                seats=priority_seats,
                description="Disadvantaged students (bottom tercile RSH)",
                legal_fraction=0.15,
            )

        if reserves:
            school_overrides[f"{rbd}_{program_code}"] = {
                "rbd": rbd,
                "program_code": program_code,
                "total_admission_seats": int(total),
                "region": region,
                "regular_seats": safe_int('regular_seats'),
                "reserves": reserves,
            }

    return school_overrides


def build_chile_config(region_overrides, school_overrides, total_students):
    return {
        "__meta__": {
            "system": "Chile SAE",
            "id_format": "rbd_program_code",
            "legal_basis": "Ley 20.845 + Decreto 152",
            "reference": "Correa, Epstein, Escobar (2019)",
            "total_students": total_students,
            "n_regions": len(region_overrides),
            "granularity": {
                "priority_tiers": "region",
                "reserves": "school",
                "student_fractions": "region",
            },
            "quota_processing_order": [
                "special_needs",
                "academic_excellence",
                "disadvantaged",
            ],
        },
        "system_defaults": {
            "priority_tiers": SYSTEM_DEFAULT_TIERS,
            "reserves": {},
            "student_attribute_fractions": {},
        },
        "region_overrides": region_overrides,
        "school_overrides": school_overrides,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--individual", required=True)
    parser.add_argument("--capacity",   required=True)
    parser.add_argument("--output",     default="chile_priority_config.json")
    args = parser.parse_args()

    print("Building per-region student fractions ...")
    indv_df = pd.read_excel(args.individual)
    region_overrides, total_students = build_region_fractions(indv_df)
    rbd_to_region = indv_df.groupby('rbd')['Region'].first().to_dict()
    print(f"  Regions: {len(region_overrides)}, students: {total_students:,}")

    print("Building per-school reserves ...")
    school_overrides = build_school_overrides(args.capacity, rbd_to_region)
    print(f"  Schools with reserves: {len(school_overrides)}")
    print(f"    disadvantaged:       {sum(1 for v in school_overrides.values() if 'disadvantaged' in v['reserves'])}")
    print(f"    academic_excellence: {sum(1 for v in school_overrides.values() if 'academic_excellence' in v['reserves'])}")
    print(f"    special_needs:       {sum(1 for v in school_overrides.values() if 'special_needs' in v['reserves'])}")

    config = build_chile_config(region_overrides, school_overrides, total_students)

    warnings = validate_config(config)
    if warnings:
        print("\nValidation warnings:")
        for w in warnings:
            print(f"  ! {w}")

    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2, ensure_ascii=False)
    print(f"\nWritten to {args.output}")


if __name__ == "__main__":
    main()