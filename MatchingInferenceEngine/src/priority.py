"""
priority_config_schema.py - Unified priority config schema for NYC and Chile generators.

Schema structure:
  __meta__          system metadata and granularity declarations
  system_defaults   fallback tiers/reserves/fractions
  region_overrides  per-region tiers/reserves/fractions (overrides system_defaults)
  school_overrides  per-school reserves only (merged with region/system; school wins)

Resolution order:
  priority_tiers               region -> system
  reserves                     school MERGE region MERGE system (school wins)
  student_attribute_fractions  region -> system

TierObject fields:
  tier, group, description, fraction_eligible, seat_cap_fraction,
  school_dependent, borough_code, special_type

ReserveObject fields:
  group, fraction, seats, description, legal_fraction
"""


def make_tier(tier, group, description, fraction_eligible,
              seat_cap_fraction=None, school_dependent=True,
              borough_code=None, special_type=None):
    return {
        "tier": tier,
        "group": group,
        "description": description,
        "fraction_eligible": fraction_eligible,
        "seat_cap_fraction": seat_cap_fraction,
        "school_dependent": school_dependent,
        "borough_code": borough_code,
        "special_type": special_type,
    }


def make_reserve(group, fraction, seats, description, legal_fraction=None):
    return {
        "group": group,
        "fraction": fraction,
        "seats": seats,
        "description": description,
        "legal_fraction": legal_fraction,
    }


def resolve_config(school_id, region, config):
    system = config.get("system_defaults", {})
    region_data = config.get("region_overrides", {}).get(region, {})
    school_data = config.get("school_overrides", {}).get(school_id, {})

    priority_tiers = region_data.get("priority_tiers") or system.get("priority_tiers", [])

    reserves = {}
    reserves.update(system.get("reserves", {}))
    reserves.update(region_data.get("reserves", {}))
    reserves.update(school_data.get("reserves", {}))

    student_fractions = {
        **system.get("student_attribute_fractions", {}),
        **region_data.get("student_attribute_fractions", {}),
    }

    return {
        "priority_tiers": priority_tiers,
        "reserves": reserves,
        "student_attribute_fractions": student_fractions,
    }


def validate_config(config):
    warnings = []
    meta = config.get("__meta__", {})

    if "system" not in meta:
        warnings.append("__meta__.system is missing")
    if "id_format" not in meta:
        warnings.append("__meta__.id_format is missing")

    for region, data in config.get("region_overrides", {}).items():
        tiers = data.get("priority_tiers", [])
        tier_nums = [t["tier"] for t in tiers]
        if tier_nums != sorted(tier_nums):
            warnings.append(f"Region '{region}': priority_tiers not in ascending order")
        if not any(t["group"] in ("all", "all_nyc") for t in tiers):
            warnings.append(f"Region '{region}': no catch-all tier (all/all_nyc)")

    tier_granularity = meta.get("granularity", {}).get("priority_tiers", "region")
    if tier_granularity != "school":
        for sid, data in config.get("school_overrides", {}).items():
            if "priority_tiers" in data:
                warnings.append(
                    f"School '{sid}': priority_tiers in school_overrides — "
                    "move to region_overrides."
                )

    return warnings