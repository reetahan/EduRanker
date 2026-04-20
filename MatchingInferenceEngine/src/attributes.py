"""
attributes.py

Samples per-student priority attributes given a priority config and student districts.

For school-independent attributes (SWD, DIA/disadvantaged, high_performance):
    drawn once per student from regional/system fractions.

For school-dependent attributes (borough, sibling, continuing, working_parent,
returning_student, feeder_school, special_program):
    drawn once per student, yielding a single school key (or None) where
    the attribute applies. At DA time, the attribute is active only when
    the student proposes to that specific school.

Output: list of dicts, one per student, length = n_students.
"""

import numpy as np
from priority import resolve_config


SCHOOL_INDEPENDENT_GROUPS = {"SWD", "DIA", "disadvantaged", "high_performance", "special_needs", "academic_excellence"}
SCHOOL_DEPENDENT_GROUPS   = {"borough", "continuing", "sibling", "working_parent", "returning_student", "feeder_school", "special_program"}

# Per-district DIA (low-income/FRL) fraction for NYC.
# Approximated from NYSED ENI data and FRL patterns; citywide ~0.72.
NYC_DISTRICT_DIA_FRACTION = {
    1:  0.68, 2:  0.38, 3:  0.62, 4:  0.78, 5:  0.80, 6:  0.75,
    7:  0.88, 8:  0.85, 9:  0.90, 10: 0.82, 11: 0.78, 12: 0.87,
    13: 0.78, 14: 0.82, 15: 0.65, 16: 0.90, 17: 0.85, 18: 0.72,
    19: 0.82, 20: 0.42, 21: 0.58, 22: 0.62, 23: 0.85,
    24: 0.72, 25: 0.48, 26: 0.38, 27: 0.78, 28: 0.68, 29: 0.80,
    30: 0.68, 31: 0.60, 32: 0.82,
}
NYC_DIA_FRACTION_DEFAULT = 0.72

# Citywide sibling prior for NYC. Lower than Chile (~15%) because
# with 400+ schools a student's sibling is unlikely at any specific one.
NYC_SIBLING_FRACTION = 0.08


def _get_fractions(config, region):
    """Return student_attribute_fractions for a region, falling back to system_defaults."""
    region_data = config.get("region_overrides", {}).get(region, {})
    system_data = config.get("system_defaults", {})
    return {
        **system_data.get("student_attribute_fractions", {}),
        **region_data.get("student_attribute_fractions", {}),
    }


def _get_tiers(config, region):
    """Return priority_tiers for a region, falling back to system_defaults."""
    region_data = config.get("region_overrides", {}).get(region, {})
    system_data = config.get("system_defaults", {})
    return (
        region_data.get("priority_tiers")
        or system_data.get("priority_tiers", [])
    )


def _school_dependent_tier_groups(config, region):
    """Return set of school-dependent groups present in this region's tiers."""
    tiers = _get_tiers(config, region)
    return {t["group"] for t in tiers if t.get("school_dependent", False)}


def sample_student_attributes(
    district_assignments,
    all_schools,
    dbn_to_progs,
    priority_config,
    district_to_region,
    rng,
    q_continuing=0.55,
    district_to_borough=None,
):
    """
    Sample priority attributes for each student.

    Args:
        district_assignments  list/array of district labels, length n_students
        all_schools           array of program keys (e.g. '01M292_prog1')
        dbn_to_progs          dict {dbn -> [prog_key, ...]}
        priority_config       unified priority config dict
        district_to_region    dict {district -> region_name}
        rng                   np.random.Generator
        q_continuing          P(continuing student lists their own school)

    Returns:
        list of dicts, one per student:
        {
            'SWD':              bool,
            'DIA':              bool,   # NYC
            'disadvantaged':    bool,   # Chile
            'high_performance': bool,
            'special_needs':    bool,
            'borough':          str | None,   # borough code if applicable
            'continuing_school': str | None,  # prog_key where student is continuing
            'sibling_school':    str | None,
            'working_parent_school': str | None,
            'returning_school':  str | None,
        }
    """
    n_students = len(district_assignments)
    prog_key_list = list(all_schools)
    n_progs = len(prog_key_list)

    # Precompute per-region fractions and tiers
    regions = set(district_to_region.values())
    region_fracs  = {r: _get_fractions(priority_config, r) for r in regions}
    region_dep_groups = {r: _school_dependent_tier_groups(priority_config, r) for r in regions}

    # Build borough lookup: prog_key -> borough_code
    prog_borough = {}
    for pk in prog_key_list:
        so = priority_config.get("school_overrides", {}).get(pk, {})
        prog_borough[pk] = so.get("borough", None)

    # Build continuing lookup: prog_key -> fraction_eligible (p)
    prog_continuing_p = {}
    for pk in prog_key_list:
        so = priority_config.get("school_overrides", {}).get(pk, {})
        tiers = so.get("priority_tiers", [])
        cont = next((t for t in tiers if t["group"] == "continuing"), None)
        if cont:
            prog_continuing_p[pk] = cont.get("fraction_eligible") or 0.0

    attrs = []
    for i in range(n_students):
        district = district_assignments[i]
        region = district_to_region.get(str(district), None)
        fracs = region_fracs.get(region, {}) if region else {}
        dep_groups = region_dep_groups.get(region, set()) if region else set()

        def draw(key):
            p = fracs.get(key, 0.0)
            return bool(rng.random() < p) if p else False

        a = {
            'SWD':              draw('SWD') or draw('special_needs'),
            'DIA':              bool(rng.random() < NYC_DISTRICT_DIA_FRACTION.get(
                                    int(district), NYC_DIA_FRACTION_DEFAULT))
                                if priority_config.get('__meta__', {}).get('system') == 'NYC'
                                else draw('DIA'),
            'disadvantaged':    draw('disadvantaged'),
            'high_performance': draw('high_performance'),
            'special_needs':    draw('special_needs'),
            'borough':          district_to_borough.get(str(district)) if district_to_borough else None,
            'continuing_school':     None,
            'sibling_school':        None,
            'working_parent_school': None,
            'returning_school':      None,
        }

        is_nyc = priority_config.get('__meta__', {}).get('system') == 'NYC'

        # School-dependent draws: pick one school (or none) per attribute
        if "continuing" in dep_groups and prog_continuing_p:
            # Draw a program key with probability p, then apply q
            probs = np.array([prog_continuing_p.get(pk, 0.0) for pk in prog_key_list])
            total = probs.sum()
            if total > 0 and rng.random() < total:
                chosen_idx = rng.choice(n_progs, p=probs / total)
                if rng.random() < q_continuing:
                    a['continuing_school'] = prog_key_list[chosen_idx]

        if "sibling" in dep_groups:
            p_sib = NYC_SIBLING_FRACTION if is_nyc else fracs.get('priority_sibling', 0.0)
            if p_sib > 0 and rng.random() < p_sib:
                dbn = rng.choice(list(dbn_to_progs.keys()))
                progs = dbn_to_progs[dbn]
                a['sibling_school'] = rng.choice(progs)

        if "working_parent" in dep_groups:
            p_wp = fracs.get('priority_parent_civil_servant', 0.0)
            if p_wp > 0 and rng.random() < p_wp:
                dbn = rng.choice(list(dbn_to_progs.keys()))
                progs = dbn_to_progs[dbn]
                a['working_parent_school'] = rng.choice(progs)

        if "returning_student" in dep_groups:
            p_ret = fracs.get('priority_ex_student', 0.0)
            if p_ret > 0 and rng.random() < p_ret:
                dbn = rng.choice(list(dbn_to_progs.keys()))
                progs = dbn_to_progs[dbn]
                a['returning_school'] = rng.choice(progs)

        attrs.append(a)

    return attrs


def build_composite_rank_matrix(
    all_schools,
    student_attrs,
    priority_config,
    school_lotteries,
    district_to_region,
    district_assignments,
):
    """
    Builds the effective rank matrix used by the DA in place of raw lotteries.

    Composite rank = reserve_bucket * 1e8 + priority_tier * 1e4 + lottery * 1e0

    reserve_bucket:
        0 = student is eligible for a reserve at this school (processed first)
        1 = general seats

    priority_tier: 1-indexed tier from config (lower = higher priority)
    max_tier + 1 = general pool (all/all_nyc)

    Args:
        all_schools      array of program keys, shape (n_schools,)
        student_attrs    list of attr dicts, length n_students
        priority_config  unified priority config
        school_lotteries np.ndarray shape (n_schools, n_students), values in [0,1)
        district_to_region dict {district -> region}
        district_assignments list of districts, length n_students

    Returns:
        np.ndarray shape (n_schools, n_students), dtype float64
    """
    n_schools = len(all_schools)
    n_students = len(student_attrs)
    ranks = school_lotteries.copy().astype(np.float64)

    for s_idx, prog_key in enumerate(all_schools):
        so = priority_config.get("school_overrides", {}).get(prog_key, {})
        prog_borough = so.get("borough", None)

        # Get tiers: school-level for NYC, region-level for Chile
        tiers = so.get("priority_tiers", None)
        if not tiers:
            # Fall back to region/system
            # Use first student's district as proxy — tiers are region-uniform
            sample_district = str(district_assignments[0])
            region = district_to_region.get(sample_district, None)
            tiers = _get_tiers(priority_config, region)

        reserves = so.get("reserves", {})
        max_tier = max((t["tier"] for t in tiers), default=1)

        for st_idx, attrs in enumerate(student_attrs):
            lottery = school_lotteries[s_idx, st_idx]

            # --- Reserve bucket ---
            reserve_bucket = 1  # default: general seats
            if "SWD" in reserves and attrs.get("SWD"):
                reserve_bucket = 0
            elif "DIA" in reserves and attrs.get("DIA"):
                reserve_bucket = 0
            elif "disadvantaged" in reserves and attrs.get("disadvantaged"):
                reserve_bucket = 0
            elif "special_needs" in reserves and attrs.get("special_needs"):
                reserve_bucket = 0
            elif "academic_excellence" in reserves and attrs.get("high_performance"):
                reserve_bucket = 0

            # --- Priority tier ---
            priority_tier = max_tier  # default: general pool
            for t in sorted(tiers, key=lambda x: x["tier"]):
                group = t["group"]
                if group in ("all", "all_nyc"):
                    break
                matched = False
                if group == "borough":
                    matched = (attrs.get("borough") == prog_borough)
                elif group == "continuing":
                    matched = (attrs.get("continuing_school") == prog_key)
                elif group == "sibling":
                    matched = (attrs.get("sibling_school") == prog_key)
                elif group == "working_parent":
                    matched = (attrs.get("working_parent_school") == prog_key)
                elif group == "returning_student":
                    matched = (attrs.get("returning_school") == prog_key)
                if matched:
                    priority_tier = t["tier"]
                    break

            ranks[s_idx, st_idx] = reserve_bucket * 1e8 + priority_tier * 1e4 + lottery

    return ranks