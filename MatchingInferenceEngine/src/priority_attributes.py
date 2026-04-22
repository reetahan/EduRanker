
import numpy as np


SCHOOL_INDEPENDENT_GROUPS = {"SWD", "DIA", "disadvantaged", "high_performance", "special_needs", "academic_excellence"}
SCHOOL_DEPENDENT_GROUPS   = {"borough", "continuing", "sibling", "working_parent", "returning_student", "feeder_school", "special_program"}


NYC_DIA_FRACTION_DEFAULT = 0.773 
NYC_DISTRICT_DIA_FRACTION = {
    1:  0.661, 2:  0.562, 3:  0.527, 4:  0.845, 5:  0.823,
    6:  0.845, 7:  0.941, 8:  0.872, 9:  0.953, 10: 0.849,
    11: 0.842, 12: 0.939, 13: 0.599, 14: 0.730, 15: 0.552,
    16: 0.868, 17: 0.807, 18: 0.823, 19: 0.902, 20: 0.782,
    21: 0.768, 22: 0.731, 23: 0.917, 24: 0.802, 25: 0.707,
    26: 0.608, 27: 0.795, 28: 0.705, 29: 0.756, 30: 0.698,
    31: 0.633, 32: 0.888,
}
NYC_SIBLING_FRACTION = 0.0


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
   
    n_students = len(district_assignments)
    prog_key_list = list(all_schools)
    n_progs = len(prog_key_list)

    regions = set(district_to_region.values())
    region_fracs  = {r: _get_fractions(priority_config, r) for r in regions}
    region_dep_groups = {r: _school_dependent_tier_groups(priority_config, r) for r in regions}

    prog_borough = {}
    for pk in prog_key_list:
        so = priority_config.get("school_overrides", {}).get(pk, {})
        prog_borough[pk] = so.get("borough", None)

    prog_continuing_p = {}
    for pk in prog_key_list:
        so = priority_config.get("school_overrides", {}).get(pk, {})
        tiers = so.get("priority_tiers", [])
        cont = next((t for t in tiers if t["group"] == "continuing"), None)
        if cont:
            prog_continuing_p[pk] = cont.get("fraction_eligible") or 0.0

    attrs = []
    region_tiers_map = {r: _get_tiers(priority_config, r) for r in regions}
    for i in range(n_students):
        district = district_assignments[i]
        region = district_to_region.get(str(district), None)
        region_tiers = region_tiers_map.get(region, [])
        fracs = region_fracs.get(region, {}) if region else {}
        dep_groups = region_dep_groups.get(region, set()) if region else set()

        def draw(key):
            p = fracs.get(key, 0.0)
            return bool(rng.random() < p) if p else False

        a = {
            'SWD':              draw('SWD') or draw('special_needs'),         
            'DIA':              bool(rng.random() < NYC_DISTRICT_DIA_FRACTION.get(int(district), NYC_DIA_FRACTION_DEFAULT))
                        if priority_config.get('__meta__', {}).get('system') == 'NYC'
                        else draw('DIA'),
            'disadvantaged':    draw('disadvantaged'),
            'high_performance': draw('high_performance'),
            'special_needs':    draw('special_needs'),
            'female':           bool(rng.random() < fracs.get('female', 0.485)),
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
            
            p_sib = NYC_SIBLING_FRACTION if is_nyc else next(
                (t.get('fraction_eligible') or 0.0 for t in region_tiers if t['group'] == 'sibling'), 0.0
            )
            if p_sib > 0 and rng.random() < p_sib:
                dbn = rng.choice(list(dbn_to_progs.keys()))
                progs = dbn_to_progs[dbn]
                a['sibling_school'] = rng.choice(progs)


        if "working_parent" in dep_groups:
            p_wp =  next(
                (t.get('fraction_eligible') or 0.0 for t in region_tiers if t['group'] == 'working_parent'), 0.0
            )
            if p_wp > 0 and rng.random() < p_wp:
                dbn = rng.choice(list(dbn_to_progs.keys()))
                progs = dbn_to_progs[dbn]
                a['working_parent_school'] = rng.choice(progs)

        if "returning_student" in dep_groups:
            p_ret = next(
                (t.get('fraction_eligible') or 0.0 for t in region_tiers if t['group'] == 'returning_student'), 0.0
            )
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
    Vectorized composite rank matrix for DA.
    Composite rank = reserve_bucket * 1e8 + priority_tier * 1e4 + lottery
    Lower = higher priority.
    """
    n_schools = len(all_schools)
    n_students = len(student_attrs)

    a_SWD    = np.array([a.get("SWD", False)            for a in student_attrs], dtype=bool)
    a_DIA    = np.array([a.get("DIA", False)            for a in student_attrs], dtype=bool)
    a_disadv = np.array([a.get("disadvantaged", False)  for a in student_attrs], dtype=bool)
    a_sn     = np.array([a.get("special_needs", False)  for a in student_attrs], dtype=bool)
    a_hp     = np.array([a.get("high_performance", False) for a in student_attrs], dtype=bool)
    a_borough= np.array([a.get("borough", None)         for a in student_attrs], dtype=object)

    def _invert(attr_key):
        idx = {}
        for i, a in enumerate(student_attrs):
            v = a.get(attr_key)
            if v is not None:
                idx.setdefault(v, []).append(i)
        return {k: np.array(v) for k, v in idx.items()}

    continuing_idx  = _invert("continuing_school")
    sibling_idx     = _invert("sibling_school")
    wp_idx          = _invert("working_parent_school")
    returning_idx   = _invert("returning_school")

    # Used as fallback if school/region unspecified
    sample_district = str(district_assignments[0])
    sample_region = district_to_region.get(sample_district, None)
    default_fallback_tiers = _get_tiers(priority_config, sample_region)

    school_overrides = priority_config.get("school_overrides", {})
    ranks = school_lotteries.copy().astype(np.float64)

    for s_idx, prog_key in enumerate(all_schools):
        so = school_overrides.get(prog_key, {})
        prog_borough = so.get("borough", None)
        school_region = so.get("region", None)
        fallback_tiers = _get_tiers(priority_config, school_region) if school_region else default_fallback_tiers
        tiers = so.get("priority_tiers") or fallback_tiers
        reserves = so.get("reserves", {})
        max_tier = max((t["tier"] for t in tiers), default=1)

        reserve_bucket = np.ones(n_students, dtype=np.float64)
        if "SWD" in reserves:
            reserve_bucket[a_SWD] = 0.0
        if "DIA" in reserves:
            reserve_bucket[a_DIA & (reserve_bucket == 1)] = 0.0
        if "disadvantaged" in reserves:
            reserve_bucket[a_disadv & (reserve_bucket == 1)] = 0.0
        if "special_needs" in reserves:
            reserve_bucket[a_sn & (reserve_bucket == 1)] = 0.0
        if "academic_excellence" in reserves:
            reserve_bucket[a_hp & (reserve_bucket == 1)] = 0.0

        priority_tier = np.full(n_students, float(max_tier))
        assigned = np.zeros(n_students, dtype=bool)

        for t in sorted(tiers, key=lambda x: x["tier"]):
            group = t["group"]
            if group in ("all", "all_nyc"):
                break

            if group == "borough" and prog_borough is not None:
                matched = (~assigned) & (a_borough == prog_borough)
                priority_tier[matched] = t["tier"]
                assigned[matched] = True
            elif group == "continuing":
                idxs = continuing_idx.get(prog_key)
                if idxs is not None:
                    mask = ~assigned[idxs]
                    priority_tier[idxs[mask]] = t["tier"]
                    assigned[idxs[mask]] = True
            elif group == "sibling":
                idxs = sibling_idx.get(prog_key)
                if idxs is not None:
                    mask = ~assigned[idxs]
                    priority_tier[idxs[mask]] = t["tier"]
                    assigned[idxs[mask]] = True
            elif group == "working_parent":
                idxs = wp_idx.get(prog_key)
                if idxs is not None:
                    mask = ~assigned[idxs]
                    priority_tier[idxs[mask]] = t["tier"]
                    assigned[idxs[mask]] = True
            elif group == "returning_student":
                idxs = returning_idx.get(prog_key)
                if idxs is not None:
                    mask = ~assigned[idxs]
                    priority_tier[idxs[mask]] = t["tier"]
                    assigned[idxs[mask]] = True

        ranks[s_idx] = reserve_bucket * 1e8 + priority_tier * 1e4 + school_lotteries[s_idx]

    return ranks