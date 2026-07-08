from __future__ import annotations

import base64
import gzip
import hashlib
import io
import re

import numpy as np
import pandas as pd
import streamlit as st
from scipy.stats import binom, hypergeom

# ---------------------------------------------------------------------------
# Data columns
# ---------------------------------------------------------------------------
WISH_RANK    = "wish_rank"
PROGRAM      = "program"
LOTTERY      = "lottery_number"
HASH_INPUT   = "lottery_hash_input"
HASH_HEX     = "lottery_hash_hex"
HASH_PCT     = "lottery_hash_percentile"

CAPACITY     = "total_admission_seats"
TRUE_APP     = "true_applicants_last_year"
POP          = "program_lottery_population_2024"
IMPUTED      = "calibration_2024_imputed"
IMPUT_METHOD = "calibration_2024_imputation_method"

PRIORITY_STUDENT_QUOTA = 0.15   # 15% reserved for priority students
DEFAULT_THRESHOLD_MTB  = 0.025
DEFAULT_THRESHOLD_STB  = 0.675

PRIORITIES = [
    "priority_sibling",
    "priority_student",
    "priority_parent_civil_servant",
    "priority_ex_student",
]
SAFETY     = "priority_already_registered"
NO_PRIORITY = "no_priority"
TIERS       = PRIORITIES + [NO_PRIORITY]

MAX_SHA256 = 2 ** 256 - 1

REGION = "Region"
UNKNOWN_REGION = "Unknown region"

# Embedded RBD -> Region lookup built from the 2025 individual-level preferences file.
# This lets the app sort the program dropdown by region without asking users to upload
# the large individual-level file.
REGION_ORDER = [
    "Región de Arica y Parinacota",
    "Región de Tarapacá",
    "Región de Antofagasta",
    "Región de Atacama",
    "Región de Coquimbo",
    "Región de Valparaíso",
    "Región Metropolitana de Santiago",
    "Región del Libertador Bernardo O'Higgins",
    "Región del Maule",
    "Región de Ñuble",
    "Región del Bío-Bío",
    "Región de La Araucanía",
    "Región de Los Ríos",
    "Región de Los Lagos",
    "Región de Aysén del Gral.Ibañez del Campo",
    "Región de Magallanes y Antártica Chilena",
    UNKNOWN_REGION,
]


# Embedded program-characteristic lookup built from chile_programs_sorted_by_specialty.csv.
# Key: "rbd|program_code". Values: (track, specialty sector, specialty name, gender, school day).
PROGRAM_TRACK = "program_track"
PROGRAM_SPECIALTY_SECTOR = "program_specialty_sector"
PROGRAM_SPECIALTY_NAME = "program_specialty_name"
PROGRAM_GENDER = "program_gender"
PROGRAM_SCHOOL_DAY = "program_school_day"
UNKNOWN_FILTER_VALUE = "Unknown"

PROGRAM_RECONSTRUCTED_NAME = "program_reconstructed_name"
PROGRAM_DISPLAY_NAME = "program_display_name"
SCHOOL_NAME = "school_name"
SCHOOL_COMMUNE = "school_commune"
UNKNOWN_PROGRAM_NAME = "Program details unavailable"
UNKNOWN_SCHOOL_NAME = "School name unavailable"

PROGRAM_RURALITY = "program_rurality"
PROGRAM_PIE = "program_pie"
PROGRAM_PACE = "program_pace"
PROGRAM_ENROLLMENT_FEE = "program_enrollment_fee"
PROGRAM_MONTHLY_FEE = "program_monthly_fee"
PROGRAM_RELIGIOUS_ORIENTATION = "program_religious_orientation"
PROGRAM_RELIGIOUS_DETAIL = "program_religious_detail"

TRACK_GENERAL = "General"
TRACK_SPECIALIZED = "Specialized"

SPECIALTY_FILTER_OPTIONS = [
    "Agriculture",
    "Metalworking and mechanics",
    "Electricity",
    "Food services",
    "Construction",
    "Technology and communications",
]
GENDER_FILTER_OPTIONS = ["Mixed", "Boys", "Girls"]
SCHOOL_DAY_FILTER_OPTIONS = ["Full day", "Morning", "Afternoon"]
RURALITY_FILTER_OPTIONS = ["Urban", "Rural"]
PIE_FILTER_OPTIONS = ["With PIE", "Without PIE"]
PACE_FILTER_OPTIONS = ["With PACE", "Without PACE"]
PAYMENT_FILTER_OPTIONS = [
    "Free",
    "$1,000-$10,000",
    "$10,001-$25,000",
    "$25,001-$50,000",
    "$50,001-$100,000",
    "More than $100,000",
    "No information",
]
RELIGIOUS_FILTER_OPTIONS = ["Secular", "Catholic", "Evangelical", "Other", "No information"]



def region_sort_index(region: str) -> int:
    try:
        return REGION_ORDER.index(str(region).strip())
    except ValueError:
        return len(REGION_ORDER)


def attach_embedded_regions(calib: pd.DataFrame) -> pd.DataFrame:
    """Attach embedded region labels to the capacities/calibration file.

    This keeps every program from the capacities file. If an RBD is not found in
    the embedded lookup, it is still available under Unknown region.
    """
    out = calib.copy()
    out["rbd"] = norm_code(out["rbd"])
    out[REGION] = out["rbd"].map(RBD_REGION_MAP).fillna(UNKNOWN_REGION)
    return out


def program_filter_key(rbd, program_code) -> str:
    return f"{norm_code_value(rbd)}|{norm_code_value(program_code)}"


def attach_embedded_program_filters(calib: pd.DataFrame) -> pd.DataFrame:
    """Attach embedded program characteristics used by the sidebar filters.

    The capacities/calibration file remains the source of the available programs.
    The embedded metadata only adds filtering fields. Programs with missing
    metadata remain available when no filter is active.
    """
    out = calib.copy()
    out["rbd"] = norm_code(out["rbd"])
    out["program_code"] = norm_code(out["program_code"])

    keys = out.apply(lambda row: program_filter_key(row["rbd"], row["program_code"]), axis=1)
    metadata = keys.map(PROGRAM_FILTER_MAP)

    out[PROGRAM_TRACK] = metadata.map(lambda x: x[0] if isinstance(x, tuple) else UNKNOWN_FILTER_VALUE)
    out[PROGRAM_SPECIALTY_SECTOR] = metadata.map(lambda x: x[1] if isinstance(x, tuple) else UNKNOWN_FILTER_VALUE)
    out[PROGRAM_SPECIALTY_NAME] = metadata.map(lambda x: x[2] if isinstance(x, tuple) else UNKNOWN_FILTER_VALUE)
    out[PROGRAM_GENDER] = metadata.map(lambda x: x[3] if isinstance(x, tuple) else UNKNOWN_FILTER_VALUE)
    out[PROGRAM_SCHOOL_DAY] = metadata.map(lambda x: x[4] if isinstance(x, tuple) else UNKNOWN_FILTER_VALUE)
    return out


def program_matches_filters(row: pd.Series, filters: dict | None) -> bool:
    """Return True if a program row satisfies the sidebar filters.

    Empty filters mean no restriction. Selected existing wishes are preserved
    separately in filter_program_options().
    """
    if not filters:
        return True

    selected_tracks = filters.get("tracks") or []
    selected_specialties = filters.get("specialty_sectors") or []
    selected_genders = filters.get("genders") or []
    selected_school_days = filters.get("school_days") or []
    selected_rurality = filters.get("rurality") or []
    selected_pie = filters.get("pie") or []
    selected_pace = filters.get("pace") or []
    selected_enrollment_fee = filters.get("enrollment_fee") or []
    selected_monthly_fee = filters.get("monthly_fee") or []
    selected_religious_orientation = filters.get("religious_orientation") or []

    track = str(row.get(PROGRAM_TRACK, UNKNOWN_FILTER_VALUE)).strip()
    specialty_sector = str(row.get(PROGRAM_SPECIALTY_SECTOR, UNKNOWN_FILTER_VALUE)).strip()
    gender = str(row.get(PROGRAM_GENDER, UNKNOWN_FILTER_VALUE)).strip()
    school_day = str(row.get(PROGRAM_SCHOOL_DAY, UNKNOWN_FILTER_VALUE)).strip()
    rurality = str(row.get(PROGRAM_RURALITY, UNKNOWN_FILTER_VALUE)).strip()
    pie = str(row.get(PROGRAM_PIE, UNKNOWN_FILTER_VALUE)).strip()
    pace = str(row.get(PROGRAM_PACE, UNKNOWN_FILTER_VALUE)).strip()
    enrollment_fee = str(row.get(PROGRAM_ENROLLMENT_FEE, UNKNOWN_FILTER_VALUE)).strip()
    monthly_fee = str(row.get(PROGRAM_MONTHLY_FEE, UNKNOWN_FILTER_VALUE)).strip()
    religious_orientation = str(row.get(PROGRAM_RELIGIOUS_ORIENTATION, UNKNOWN_FILTER_VALUE)).strip()

    if selected_tracks and track not in selected_tracks:
        return False

    if selected_specialties:
        if track != TRACK_SPECIALIZED or specialty_sector not in selected_specialties:
            return False

    if selected_genders and gender not in selected_genders:
        return False

    if selected_school_days and school_day not in selected_school_days:
        return False

    if selected_rurality and rurality not in selected_rurality:
        return False

    if selected_pie and pie not in selected_pie:
        return False

    if selected_pace and pace not in selected_pace:
        return False

    if selected_enrollment_fee and enrollment_fee not in selected_enrollment_fee:
        return False

    if selected_monthly_fee and monthly_fee not in selected_monthly_fee:
        return False

    if selected_religious_orientation and religious_orientation not in selected_religious_orientation:
        return False

    return True


def filters_are_active(filters: dict | None) -> bool:
    if not filters:
        return False
    return any(bool(filters.get(k)) for k in [
        "tracks",
        "specialty_sectors",
        "genders",
        "school_days",
        "rurality",
        "pie",
        "pace",
        "enrollment_fee",
        "monthly_fee",
        "religious_orientation",
    ])


# ---------------------------------------------------------------------------
# CSV reading utilities
# ---------------------------------------------------------------------------

def read_csv(file_bytes: bytes, sep: str = "auto") -> pd.DataFrame:
    kwargs: dict = {"dtype": str, "encoding": "utf-8-sig"}
    if sep == "auto":
        kwargs |= {"sep": None, "engine": "python"}
    else:
        kwargs["sep"] = sep
    df = pd.read_csv(io.BytesIO(file_bytes), **kwargs)
    df.columns = [str(c).lstrip("\ufeff").strip() for c in df.columns]
    return df


def norm_code_value(x) -> str:
    x = str(x).strip()
    if x.startswith('="') and x.endswith('"'):
        x = x[2:-1].strip()
    try:
        return str(int(float(x.replace(",", "."))))
    except Exception:
        return x


def norm_code(s: pd.Series) -> pd.Series:
    return s.map(norm_code_value)


def as_bool(x) -> bool:
    if pd.isna(x):
        return False
    return str(x).strip().lower() in {"1", "true", "yes", "y", "x", "oui"}


def as_float(x, default: float = 0.0) -> float:
    try:
        if pd.isna(x):
            return default
        return float(str(x).strip().replace(",", "."))
    except Exception:
        return default


# ---------------------------------------------------------------------------
# Hash MTB (SHA-256 RUN/IPE + RBD)
# ---------------------------------------------------------------------------

def normalize_run(student_id: str) -> str:
    """Normalize a Chilean RUN/IPE before hashing.

    Removes dots and spaces, uppercases K, and keeps the hyphen.
    Raises ValueError if the identifier is empty or contains invalid characters.
    """
    cleaned = str(student_id).strip().upper().replace(".", "")
    cleaned = re.sub(r"\s+", "", cleaned)
    if not cleaned:
        raise ValueError("Enter the student RUN/IPE before running the MTB calculation.")
    if not re.fullmatch(r"[0-9K\-]+", cleaned):
        raise ValueError(
            "The RUN/IPE may contain only digits, one optional hyphen, and the letter K."
        )
    return cleaned


def mtb_hash(student_id: str, rbd) -> dict:
    """Compute the deterministic lottery percentile for a (student, school) pair.

    SHA-256 returns a value between 0 and MAX_SHA256.
    The official priority direction is larger = better; it is converted into a
    0-best/1-worst percentile to match the model convention.

    Returns a dict with HASH_INPUT, HASH_HEX, HASH_PCT, and priority_percentile.
    """
    norm_id  = normalize_run(student_id)
    norm_rbd = norm_code_value(rbd)
    hash_input = f"{norm_id}{norm_rbd}"
    hex_digest  = hashlib.sha256(hash_input.encode("utf-8")).hexdigest()
    decimal     = int(hex_digest, 16)

    priority_pct = decimal / MAX_SHA256          # 1 = best
    lottery_pct  = 1.0 - priority_pct            # 0 = best

    return {
        HASH_INPUT: hash_input,
        HASH_HEX:   hex_digest,
        HASH_PCT:   float(np.clip(lottery_pct, 0, 1)),
        "priority_percentile": float(np.clip(priority_pct, 0, 1)),
    }


def pct_to_rank(percentile: float, n: int) -> int:
    """Convert a 0-best/1-worst percentile into an integer rank among n candidates."""
    n = max(int(n), 1)
    return int(1 + np.floor(np.clip(percentile, 0, 1) * max(n - 1, 0)))


def attach_mtb_hashes(
    wishes: pd.DataFrame,
    mapping: dict[str, pd.Series],
    student_id: str,
) -> pd.DataFrame:
    """Compute and attach the MTB percentile to each valid wish."""
    out = wishes.copy()
    for col in (HASH_INPUT, HASH_HEX, HASH_PCT):
        if col not in out.columns:
            out[col] = np.nan if col == HASH_PCT else ""

    for idx, wish in out.iterrows():
        label = str(wish.get(PROGRAM, "")).strip()
        if not label or label not in mapping:
            continue
        program   = mapping[label]
        true_app  = max(round(as_float(program[TRUE_APP], 1)), 1)
        h         = mtb_hash(student_id, program["rbd"])

        out.at[idx, HASH_INPUT] = h[HASH_INPUT]
        out.at[idx, HASH_HEX]   = h[HASH_HEX]
        out.at[idx, HASH_PCT]   = h[HASH_PCT]
        # Indicative rank among last year's true applicants
        out.at[idx, LOTTERY]    = pct_to_rank(h[HASH_PCT], true_app)

    return out


def get_embedded_calibration_bytes() -> bytes:
    """Return the built-in capacities + 2024 calibration CSV as bytes."""
    return gzip.decompress(base64.b64decode(EMBEDDED_CAPACITIES_CSV_GZ_B64))


def get_embedded_program_names_bytes() -> bytes:
    """Return the built-in reconstructed program names CSV as bytes."""
    return gzip.decompress(base64.b64decode(EMBEDDED_PROGRAM_NAMES_CSV_GZ_B64))


def compact_program_name(name: str) -> str:
    """Convert the reconstructed program name into a short English label.

    The source file contains concise reconstructed names such as
    "1º medio — Général H-C — Mixte — jornada completa". The dropdown is
    easier to scan if the repeated grade is removed and the stable descriptors
    are translated.
    """
    text = str(name or "").strip()
    if not text:
        return UNKNOWN_PROGRAM_NAME

    replacements = {
        "1º medio": "1st grade secondary",
        "Général H-C": "General H-C",
        "Spécialité TP": "Technical-vocational",
        "Mixte": "Mixed",
        "garçons": "Boys",
        "filles": "Girls",
        "jornada completa": "Full day",
        "jornada mañana": "Morning",
        "jornada tarde": "Afternoon",
    }
    for old_value, new_value in replacements.items():
        text = text.replace(old_value, new_value)

    parts = [part.strip() for part in text.split("—") if part.strip()]
    if parts and parts[0].lower().startswith("1st grade"):
        parts = parts[1:]

    if not parts:
        return UNKNOWN_PROGRAM_NAME

    if parts[0] == "Technical-vocational" and len(parts) >= 2:
        main = f"Technical-vocational: {parts[1]}"
        rest = parts[2:]
        return " · ".join([main] + rest)

    return " · ".join(parts)


def compact_school_name(name: str) -> str:
    """Return a readable school name for the program dropdown."""
    text = " ".join(str(name or "").strip().split())
    if not text:
        return UNKNOWN_SCHOOL_NAME

    # The source file is mostly uppercase. Title case is easier to scan in a dropdown.
    if text.upper() == text:
        text = text.title()
        for old, new in {
            " De ": " de ",
            " Del ": " del ",
            " La ": " la ",
            " Las ": " las ",
            " Los ": " los ",
            " Y ": " y ",
        }.items():
            text = text.replace(old, new)

    return text


VALUE_TRANSLATIONS = {
    PROGRAM_RURALITY: {
        "Urbain": "Urban",
        "Rural": "Rural",
    },
    PROGRAM_PIE: {
        "Avec PIE": "With PIE",
        "Sans PIE": "Without PIE",
    },
    PROGRAM_PACE: {
        "Avec PACE": "With PACE",
        "Sans PACE": "Without PACE",
    },
    PROGRAM_ENROLLMENT_FEE: {
        "Gratuit": "Free",
        "$1.000 A $10.000": "$1,000-$10,000",
        "$10.001 A $25.000": "$10,001-$25,000",
        "$25.001 A $50.000": "$25,001-$50,000",
        "$50.001 A $100.000": "$50,001-$100,000",
        "MAS DE $100.000": "More than $100,000",
        "Sans information": "No information",
    },
    PROGRAM_MONTHLY_FEE: {
        "Gratuit": "Free",
        "$1.000 A $10.000": "$1,000-$10,000",
        "$10.001 A $25.000": "$10,001-$25,000",
        "$25.001 A $50.000": "$25,001-$50,000",
        "$50.001 A $100.000": "$50,001-$100,000",
        "MAS DE $100.000": "More than $100,000",
        "Sans information": "No information",
    },
    PROGRAM_RELIGIOUS_ORIENTATION: {
        "Laïque": "Secular",
        "Catholique": "Catholic",
        "Évangélique": "Evangelical",
        "Autre": "Other",
        "Sans information": "No information",
    },
}


def clean_optional_value(value, *, default: str = "No information") -> str:
    text = " ".join(str(value or "").strip().split())
    if not text or text.lower() == "nan":
        return default
    return text


def translate_filter_value(value, target_column: str, *, default: str = "No information") -> str:
    text = clean_optional_value(value, default=default)
    return VALUE_TRANSLATIONS.get(target_column, {}).get(text, text)


@st.cache_data(show_spinner=False)
def load_embedded_program_names(file_bytes: bytes) -> pd.DataFrame:
    df = read_csv(file_bytes, sep="auto")
    df.columns = [str(c).lstrip("﻿").strip() for c in df.columns]

    required = {"rbd", "program_code", "nom_programme_reconstruit", "nom_lycee"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError("Embedded program-name file is missing columns: " + ", ".join(sorted(missing)))

    optional_source_cols = [
        "commune",
        "ruralite",
        "convenio_pie",
        "pace",
        "paiement_matricula",
        "paiement_mensualite",
        "orientation_religieuse",
        "orientation_religieuse_autre_detail",
    ]
    keep_cols = ["rbd", "program_code", "nom_programme_reconstruit", "nom_lycee"]
    keep_cols += [c for c in optional_source_cols if c in df.columns]

    out = df[keep_cols].copy()
    out["rbd"] = norm_code(out["rbd"])
    out["program_code"] = norm_code(out["program_code"])
    out[PROGRAM_RECONSTRUCTED_NAME] = out["nom_programme_reconstruit"].astype(str).str.strip()
    out[PROGRAM_DISPLAY_NAME] = out[PROGRAM_RECONSTRUCTED_NAME].map(compact_program_name)
    out[SCHOOL_NAME] = out["nom_lycee"].map(compact_school_name)
    out[SCHOOL_COMMUNE] = (
        out["commune"].astype(str).str.strip().str.title()
        if "commune" in out.columns else ""
    )

    criteria_sources = {
        PROGRAM_RURALITY: "ruralite",
        PROGRAM_PIE: "convenio_pie",
        PROGRAM_PACE: "pace",
        PROGRAM_ENROLLMENT_FEE: "paiement_matricula",
        PROGRAM_MONTHLY_FEE: "paiement_mensualite",
        PROGRAM_RELIGIOUS_ORIENTATION: "orientation_religieuse",
    }
    for target_col, source_col in criteria_sources.items():
        if source_col in out.columns:
            out[target_col] = out[source_col].map(lambda x, c=target_col: translate_filter_value(x, c))
        else:
            out[target_col] = "No information"

    if "orientation_religieuse_autre_detail" in out.columns:
        out[PROGRAM_RELIGIOUS_DETAIL] = out["orientation_religieuse_autre_detail"].map(lambda x: clean_optional_value(x, default=""))
    else:
        out[PROGRAM_RELIGIOUS_DETAIL] = ""

    source_cols_to_drop = [
        "nom_programme_reconstruit",
        "nom_lycee",
        "commune",
        "ruralite",
        "convenio_pie",
        "pace",
        "paiement_matricula",
        "paiement_mensualite",
        "orientation_religieuse",
        "orientation_religieuse_autre_detail",
    ]
    out = out.drop(columns=[c for c in source_cols_to_drop if c in out.columns])
    out = out.drop_duplicates(["rbd", "program_code"])
    return out


def attach_embedded_program_names(calib: pd.DataFrame) -> pd.DataFrame:
    """Attach reconstructed program names, real school names, and additional choice criteria."""
    out = calib.copy()
    out["rbd"] = norm_code(out["rbd"])
    out["program_code"] = norm_code(out["program_code"])

    names = load_embedded_program_names(get_embedded_program_names_bytes())
    out = out.merge(names, on=["rbd", "program_code"], how="left")
    out[PROGRAM_RECONSTRUCTED_NAME] = out[PROGRAM_RECONSTRUCTED_NAME].fillna("")
    out[PROGRAM_DISPLAY_NAME] = out[PROGRAM_DISPLAY_NAME].fillna(UNKNOWN_PROGRAM_NAME)
    out[SCHOOL_NAME] = out[SCHOOL_NAME].fillna("")
    out[SCHOOL_COMMUNE] = out[SCHOOL_COMMUNE].fillna("")

    for col in [
        PROGRAM_RURALITY,
        PROGRAM_PIE,
        PROGRAM_PACE,
        PROGRAM_ENROLLMENT_FEE,
        PROGRAM_MONTHLY_FEE,
        PROGRAM_RELIGIOUS_ORIENTATION,
    ]:
        out[col] = out[col].fillna("No information")
    out[PROGRAM_RELIGIOUS_DETAIL] = out[PROGRAM_RELIGIOUS_DETAIL].fillna("")
    return out


# ---------------------------------------------------------------------------
# Calibration file loading and validation
# ---------------------------------------------------------------------------

@st.cache_data(show_spinner=False)
def load_calibration(file_bytes: bytes) -> pd.DataFrame:
    df = read_csv(file_bytes, sep=";")
    if len(df.columns) == 1:
        df = read_csv(file_bytes, sep=",")
    df["program_code"] = norm_code(df["program_code"])
    df["rbd"] = norm_code(df["rbd"])
    df = attach_embedded_regions(df)
    df = attach_embedded_program_filters(df)
    df = attach_embedded_program_names(df)
    return df


def required_cols(lottery_mode: str) -> list[str]:
    cols = ["rbd", "program_code", CAPACITY, TRUE_APP]
    if lottery_mode == "STB":
        cols.append(POP)
    for tier in TIERS:
        cols += [
            f"priority_share_{tier}_2024",
            f"cum_share_before_{tier}_2024",
            f"cum_share_through_{tier}_2024",
        ]
    return cols


def infer_stb_population(calib: pd.DataFrame) -> tuple[int | None, str, bool]:
    """Infer the STB population from the file if it is constant."""
    values = (
        pd.to_numeric(calib[POP], errors="coerce").dropna()
        if POP in calib.columns
        else pd.Series(dtype=float)
    )
    if values.empty:
        return None, "missing_program_lottery_population_2024", True
    unique = values.round().astype(int).drop_duplicates()
    if len(unique) == 1:
        return max(int(unique.iloc[0]), 1), f"constant_{POP}", False
    return None, f"non_constant_{POP}", True


# ---------------------------------------------------------------------------
# Build program options
# ---------------------------------------------------------------------------

def make_program_option_label(row: pd.Series, duplicate_count: int = 1) -> str:
    """Build a readable but still uniquely identifiable dropdown label."""
    rbd = str(row["rbd"]).strip()
    code = str(row["program_code"]).strip()
    school_name = str(row.get(SCHOOL_NAME, "")).strip()
    commune = str(row.get(SCHOOL_COMMUNE, "")).strip()
    display_name = str(row.get(PROGRAM_DISPLAY_NAME, "")).strip()

    if not school_name or school_name == UNKNOWN_SCHOOL_NAME:
        school_part = f"RBD {rbd}"
    elif commune and commune.lower() != "nan":
        school_part = f"{school_name} ({commune})"
    else:
        school_part = school_name

    if not display_name or display_name == UNKNOWN_PROGRAM_NAME:
        display_name = f"Program code {code}"

    label = f"{school_part} — {display_name} · RBD {rbd}"
    if duplicate_count > 1:
        label = f"{label} · code {code}"
    return label


def build_options(calib: pd.DataFrame) -> tuple[list[str], dict[str, pd.Series]]:
    options, mapping = [], {}

    unique_programs = calib.drop_duplicates(["rbd", "program_code"]).copy()
    unique_programs["_region_sort"] = unique_programs[REGION].map(region_sort_index)
    unique_programs["_rbd_sort"] = pd.to_numeric(unique_programs["rbd"], errors="coerce")
    unique_programs["_program_sort"] = pd.to_numeric(unique_programs["program_code"], errors="coerce")

    # A few schools can have multiple distinct program codes with the same readable
    # reconstructed name. In those cases only, append the code to keep labels unique.
    unique_programs["_base_display_label"] = unique_programs.apply(
        lambda row: make_program_option_label(row, duplicate_count=1),
        axis=1,
    )
    duplicate_counts = unique_programs["_base_display_label"].value_counts().to_dict()

    unique_programs = unique_programs.sort_values(
        ["_region_sort", "_rbd_sort", "_program_sort", REGION, "rbd", "program_code"]
    )

    for _, row in unique_programs.iterrows():
        base_label = row["_base_display_label"]
        label = make_program_option_label(row, duplicate_counts.get(base_label, 1))
        options.append(label)
        mapping[label] = row

    return options, mapping

def available_regions(calib: pd.DataFrame) -> list[str]:
    """Return regions present in the capacities file, in the official north-to-south order."""
    if REGION not in calib.columns:
        return [UNKNOWN_REGION]

    present = {str(x).strip() or UNKNOWN_REGION for x in calib[REGION].dropna().unique()}
    if not present:
        return [UNKNOWN_REGION]

    ordered = [r for r in REGION_ORDER if r in present]
    extra = sorted(r for r in present if r not in ordered)
    return ordered + extra


def filter_program_options(
    program_mapping: dict[str, pd.Series],
    selected_region: str,
    active_filters: dict | None = None,
    current_values: list[str] | None = None,
) -> list[str]:
    """Filter program options by region and characteristics while preserving existing values."""
    options = []
    for label, row in program_mapping.items():
        if selected_region != "All regions" and str(row.get(REGION, UNKNOWN_REGION)).strip() != selected_region:
            continue
        if not program_matches_filters(row, active_filters):
            continue
        options.append(label)

    for value in current_values or []:
        value = str(value).strip()
        if value and value in program_mapping and value not in options:
            options.append(value)

    return options


# ---------------------------------------------------------------------------
# Wish list handling (empty table + CSV import)
# ---------------------------------------------------------------------------

def empty_wishes() -> pd.DataFrame:
    df = pd.DataFrame({WISH_RANK: [1, 2, 3], PROGRAM: ["", "", ""], LOTTERY: [1, 1, 1]})
    for col in PRIORITIES + [SAFETY]:
        df[col] = False
    return df

def clean_wish_rows(df: pd.DataFrame) -> pd.DataFrame:
    """
    Keep only meaningful wish rows, then pad back to 3 default rows.
    This avoids storing empty dynamic rows created by Streamlit.
    """
    out = df.copy()

    for col in [WISH_RANK, PROGRAM, LOTTERY] + PRIORITIES + [SAFETY]:
        if col not in out.columns:
            if col in PRIORITIES + [SAFETY]:
                out[col] = False
            elif col == LOTTERY:
                out[col] = 1
            else:
                out[col] = ""

    out[PROGRAM] = out[PROGRAM].fillna("").astype(str).str.strip()

    priority_cols = PRIORITIES + [SAFETY]
    has_priority = out[priority_cols].apply(lambda row: any(as_bool(x) for x in row), axis=1)
    has_program = out[PROGRAM] != ""

    out = out[has_program | has_priority].copy()

    out[WISH_RANK] = range(1, len(out) + 1)

    while len(out) < 3:
        new_row = {
            WISH_RANK: len(out) + 1,
            PROGRAM: "",
            LOTTERY: 1,
        }
        for col in priority_cols:
            new_row[col] = False
        out = pd.concat([out, pd.DataFrame([new_row])], ignore_index=True)

    return out.reset_index(drop=True)

def parse_wishes(file_bytes: bytes, mapping: dict[str, pd.Series]) -> pd.DataFrame:
    df = read_csv(file_bytes, sep="auto")
    base_to_label = {
        f"{r['rbd']} || {r['program_code']}": label
        for label, r in mapping.items()
    }

    # Automatic column-format detection
    if {WISH_RANK, PROGRAM, LOTTERY}.issubset(df.columns):
        out = pd.DataFrame({WISH_RANK: df[WISH_RANK], PROGRAM: df[PROGRAM], LOTTERY: df[LOTTERY]})
    elif {WISH_RANK, PROGRAM}.issubset(df.columns):
        out = pd.DataFrame({WISH_RANK: df[WISH_RANK], PROGRAM: df[PROGRAM], LOTTERY: 1})
    elif {"rang_du_voeu", "programme", "numero_loterie"}.issubset(df.columns):
        out = pd.DataFrame({WISH_RANK: df["rang_du_voeu"], PROGRAM: df["programme"], LOTTERY: df["numero_loterie"]})
    elif {"rang_du_voeu", "programme"}.issubset(df.columns):
        out = pd.DataFrame({WISH_RANK: df["rang_du_voeu"], PROGRAM: df["programme"], LOTTERY: 1})
    elif {"rbd", "program_code", "preference_number"}.issubset(df.columns):
        labels = df["rbd"].astype(str).str.strip() + " || " + norm_code(df["program_code"])
        lottery_col = df["lottery"] if "lottery" in df.columns else 1
        out = pd.DataFrame({WISH_RANK: df["preference_number"], PROGRAM: labels, LOTTERY: lottery_col})
    else:
        raise ValueError(
            "Expected columns: wish_rank/program, rang_du_voeu/programme, "
            "or rbd/program_code/preference_number."
        )

    out[WISH_RANK] = pd.to_numeric(out[WISH_RANK], errors="coerce").fillna(1).astype(int)
    out[LOTTERY]   = pd.to_numeric(out[LOTTERY], errors="coerce").fillna(1).astype(int)
    out[PROGRAM]   = (
        out[PROGRAM].astype(str).str.strip()
        .map(lambda x: x if x in mapping else base_to_label.get(x, ""))
    )
    for col in PRIORITIES + [SAFETY]:
        out[col] = df[col].map(as_bool) if col in df.columns else False

    return out.sort_values(WISH_RANK).reset_index(drop=True) if not out.empty else empty_wishes()


# ---------------------------------------------------------------------------
# Priority logic
# ---------------------------------------------------------------------------

def resolve_priority_tier(
    wish: pd.Series,
    program: pd.Series,
    *,
    lottery_percentile: float | None = None,
    lottery_rank: int | None = None,
    reference_count: int | None = None,
) -> str:
    """Determine the priority tier for a wish.

    Important: priority_student is no longer filtered by rank or by
    floor(15% * capacity). The calibration file already encodes the size of the
    priority_student segment through priority_share_* and cum_share_* columns;
    if the student has this priority, their percentile is placed in that
    calibrated segment.

    lottery_percentile, lottery_rank, and reference_count are kept for
    compatibility with existing STB/MTB calls, but are no longer used to activate
    or deactivate the priority_student tier.
    """
    if as_bool(wish.get("priority_sibling")):
        return "priority_sibling"

    if as_bool(wish.get("priority_student")):
        return "priority_student"

    if as_bool(wish.get("priority_parent_civil_servant")):
        return "priority_parent_civil_servant"
    if as_bool(wish.get("priority_ex_student")):
        return "priority_ex_student"
    return NO_PRIORITY


# ---------------------------------------------------------------------------
# Availability calculation for one wish
# ---------------------------------------------------------------------------

def availability(
    wish: pd.Series,
    program: pd.Series,
    *,
    lottery_mode: str = "MTB",
    common_lottery: int | None = None,
    stb_population: int | None = None,
) -> dict:
    capacity = max(round(as_float(program[CAPACITY])), 0)
    true_app = max(round(as_float(program[TRUE_APP])), 0)

    if lottery_mode == "STB":
        # --- STB mode: common number, global cohort population ---
        lottery    = max(int(common_lottery or 1), 1)
        population = max(int(stb_population or 1), 1)
        pop_label  = "stb_cohort_population"

        raw_rank   = min(lottery, population)
        percentile = float(np.clip((raw_rank - 1) / max(population - 1, 1), 0, 1))

        tier = resolve_priority_tier(
            wish, program,
            lottery_rank=raw_rank,
            reference_count=population,
        )
        share  = as_float(program[f"priority_share_{tier}_2024"])
        before = as_float(program[f"cum_share_before_{tier}_2024"])
        eff_pct  = float(np.clip(before + share * percentile, 0, 1))
        eff_rank = int(1 + np.floor(eff_pct * max(population - 1, 0)))
        eff_rank = min(max(eff_rank, 1), population)

        if as_bool(wish.get(SAFETY)):
            p_avail = 1.0
        elif capacity <= 0:
            p_avail = 0.0
        else:
            M        = max(population - 1, 0)
            draws    = min(max(eff_rank - 1, 0), M)
            successes = min(max(true_app - 1, 0), M)
            p_avail  = (
                1.0 if draws == 0 or successes == 0
                else float(hypergeom.cdf(capacity - 1, M, successes, draws))
            )

    else:
        # --- MTB mode: SHA-256 percentile, population = true_app ---
        population = max(true_app, 1)
        pop_label  = TRUE_APP

        lottery  = max(round(as_float(wish.get(LOTTERY, 1), 1)), 1)
        raw_rank = min(lottery, population)

        hash_pct = as_float(wish.get(HASH_PCT), np.nan)
        if pd.isna(hash_pct):
            # Fallback (file without RUN): percentile inferred from integer rank
            percentile = float(np.clip((raw_rank - 1) / max(population - 1, 1), 0, 1))
        else:
            percentile = float(np.clip(hash_pct, 0, 1))

        tier = resolve_priority_tier(
            wish, program,
            lottery_percentile=percentile,
            reference_count=population,
        )
        share  = as_float(program[f"priority_share_{tier}_2024"])
        before = as_float(program[f"cum_share_before_{tier}_2024"])
        eff_pct  = float(np.clip(before + share * percentile, 0, 1))
        eff_rank = pct_to_rank(eff_pct, population)

        if as_bool(wish.get(SAFETY)):
            p_avail = 1.0
        elif capacity <= 0:
            p_avail = 0.0
        else:
            # Binomial model: among the (true_app - 1) other applicants,
            # each has probability eff_pct of being ahead of the simulated student.
            p_avail = float(binom.cdf(capacity - 1, max(true_app - 1, 0), eff_pct))

    return {
        "wish_rank":                        int(wish[WISH_RANK]),
        "program":                          wish[PROGRAM],
        "lottery_mode":                     lottery_mode,
        "lottery_number":                   lottery,
        "priority_tier":                    tier,
        "capacity":                         capacity,
        "true_applicants_last_year":        true_app,
        "lottery_population_used":          population,
        "lottery_population_source":        pop_label,
        "raw_lottery_rank":                 raw_rank,
        "lottery_percentile_used":          percentile,
        "priority_effective_percentile":    eff_pct,
        "priority_effective_rank":          eff_rank,
        "lottery_hash_input":               str(wish.get(HASH_INPUT, "")),
        "lottery_hash_hex":                 str(wish.get(HASH_HEX, "")),
        "lottery_hash_percentile":          as_float(wish.get(HASH_PCT), np.nan),
        "availability_probability":         float(np.clip(p_avail, 0, 1)),
        "calibration_2024_imputed":         as_bool(program.get(IMPUTED, False)),
        "calibration_2024_imputation_method": str(program.get(IMPUT_METHOD, "")),
    }


# ---------------------------------------------------------------------------
# Global calculation (wish list -> results DataFrame)
# ---------------------------------------------------------------------------

def compute(
    wishes: pd.DataFrame,
    mapping: dict[str, pd.Series],
    *,
    lottery_mode: str = "MTB",
    common_lottery: int | None = None,
    stb_population: int | None = None,
) -> pd.DataFrame:
    clean = wishes[wishes[PROGRAM].astype(str).str.strip() != ""].sort_values(WISH_RANK)
    if clean.empty:
        raise ValueError("Add at least one valid wish.")

    if lottery_mode == "STB":
        if common_lottery is None:
            common_lottery = max(round(as_float(clean.iloc[0][LOTTERY], 1)), 1)
        if stb_population is None:
            raise ValueError("STB mode requires a cohort population.")

    rows = [
        availability(
            wish,
            mapping[wish[PROGRAM]],
            lottery_mode=lottery_mode,
            common_lottery=common_lottery,
            stb_population=stb_population,
        )
        for _, wish in clean.iterrows()
        if wish[PROGRAM] in mapping
    ]

    choices = pd.DataFrame(rows)
    choices["cumulative_unavailable_before_choice"] = (
        (1 - choices["availability_probability"]).cumprod().shift(1).fillna(1)
    )
    choices["choice_assignment_probability"] = (
        choices["cumulative_unavailable_before_choice"] * choices["availability_probability"]
    )
    choices["cumulative_unavailable_after_choice"] = (
        (1 - choices["availability_probability"]).cumprod()
    )
    return choices


# ===========================================================================
# Interface Streamlit
# ===========================================================================

st.set_page_config(
    page_title="SAE simulation - unmatched risk",
    page_icon="🎓",
    layout="wide",
)
st.title("Simulation of the risk of remaining unmatched")
st.caption(
    "MTB mode (admission 2026): SHA-256(RUN/IPE+RBD) percentile by school. "
    "STB mode: one common lottery number across all schools."
)

# ── Sidebar ──────────────────────────────────────────────────────────
with st.sidebar:
    st.caption("Capacities + 2024 calibration data are built in.")
    use_stb    = st.checkbox("STB mode: same number for all schools", value=False)
    lottery_mode = "STB" if use_stb else "MTB"

    threshold = st.slider(
        "Alert threshold - unmatched risk",
        0.01, 1.0,
        DEFAULT_THRESHOLD_STB if use_stb else DEFAULT_THRESHOLD_MTB,
        0.005,
        key=f"threshold_{lottery_mode.lower()}",
    )

    national_student_id = ""
    if not use_stb:
        national_student_id = st.text_input(
            "Student RUN/IPE (MTB mode)",
            value="",
            placeholder="12.345.678-9",
            help=(
                "Used to compute the SHA-256 percentile specific to each "
                "school. RUN format: 12.345.678-9. Dots are optional. "
                "For foreign students, enter the IPE."
            ),
        )

    st.markdown("### Program filters")
    st.caption("Leave every filter empty to show all programs.")

    filter_general = st.checkbox("General academic programs", value=False)
    filter_specialized = st.checkbox("Specialized / technical programs", value=False)

    selected_specialty_sectors = []
    if filter_specialized:
        selected_specialty_sectors = st.multiselect(
            "Specialized area",
            SPECIALTY_FILTER_OPTIONS,
            default=[],
            help="Leave empty to include all specialized areas.",
        )

    selected_genders = st.multiselect(
        "Gender composition",
        GENDER_FILTER_OPTIONS,
        default=[],
        help="Leave empty to include mixed, boys-only, and girls-only programs.",
    )

    selected_school_days = st.multiselect(
        "School day",
        SCHOOL_DAY_FILTER_OPTIONS,
        default=[],
        help="Leave empty to include full-day, morning, and afternoon programs.",
    )

    st.markdown("#### Additional criteria")

    selected_rurality = st.multiselect(
        "Rurality",
        RURALITY_FILTER_OPTIONS,
        default=[],
        help="Leave empty to include both urban and rural schools.",
    )

    selected_pie = st.multiselect(
        "PIE integration program",
        PIE_FILTER_OPTIONS,
        default=[],
        help="Leave empty to include schools with and without PIE.",
    )

    selected_pace = st.multiselect(
        "PACE program",
        PACE_FILTER_OPTIONS,
        default=[],
        help="Leave empty to include schools with and without PACE.",
    )

    selected_enrollment_fee = st.multiselect(
        "Enrollment fee",
        PAYMENT_FILTER_OPTIONS,
        default=[],
        help="Leave empty to include every enrollment-fee category.",
    )

    selected_monthly_fee = st.multiselect(
        "Monthly fee",
        PAYMENT_FILTER_OPTIONS,
        default=[],
        help="Leave empty to include every monthly-fee category.",
    )

    selected_religious_orientation = st.multiselect(
        "Religious orientation",
        RELIGIOUS_FILTER_OPTIONS,
        default=[],
        help="Leave empty to include every orientation.",
    )

    program_filters = {
        "tracks": ([TRACK_GENERAL] if filter_general else []) + ([TRACK_SPECIALIZED] if filter_specialized else []),
        "specialty_sectors": selected_specialty_sectors,
        "genders": selected_genders,
        "school_days": selected_school_days,
        "rurality": selected_rurality,
        "pie": selected_pie,
        "pace": selected_pace,
        "enrollment_fee": selected_enrollment_fee,
        "monthly_fee": selected_monthly_fee,
        "religious_orientation": selected_religious_orientation,
    }

# ── Built-in capacities/calibration data ─────────────────────────────
calib = load_calibration(get_embedded_calibration_bytes())
missing = [c for c in required_cols(lottery_mode) if c not in calib.columns]
if missing:
    st.error("Missing columns: " + ", ".join(missing[:20]))
    st.stop()

# ── STB population ───────────────────────────────────────────────────────────
stb_population = None
if use_stb:
    stb_population, _, needs_manual = infer_stb_population(calib)
    if needs_manual:
        st.warning(
            f"Column `{POP}` varies across programs: it cannot be used "
            "as the global STB population. Enter it manually."
        )
        with st.sidebar:
            raw_pop = st.text_input(
                "Global STB population",
                value="",
                help="Number of students in the common lottery cohort. Not the maximum program-level population.",
            ).strip()
        if not raw_pop:
            st.stop()
        try:
            stb_population = int(float(raw_pop.replace(",", ".")))
            if stb_population <= 0:
                raise ValueError
        except Exception:
            st.error("Enter a positive integer for the global STB population.")
            st.stop()
    else:
        with st.sidebar:
            st.caption(f"STB population inferred from file: {int(stb_population):,}")

# ── Program options ─────────────────────────────────────────────────────
program_options, program_mapping = build_options(calib)

# ── Section 1: student wish list ────────────────────────────────────────────────
st.subheader("1. Student wish list")
wish_file = st.file_uploader(
    "Optional: import a wish-list CSV to pre-fill the table",
    type=["csv"],
)

if wish_file is not None:
    try:
        base_rows = parse_wishes(wish_file.getvalue(), program_mapping)
        st.success(f"Table pre-filled with {len(base_rows)} wish(es).")
    except Exception as exc:
        st.error(f"Could not import the CSV: {exc}")
        base_rows = empty_wishes()
else:
    base_rows = empty_wishes()

table_key = hashlib.md5(wish_file.getvalue()).hexdigest()[:8] if wish_file else "empty"
editor_state_key = f"wish_rows_{table_key}_{lottery_mode.lower()}"
editor_source_key = f"wish_rows_source_{lottery_mode.lower()}"
editor_widget_key_base = f"wishes_editor_{table_key}_{lottery_mode.lower()}"

# Keep the edited wish list in session state. This lets the student change the
# region filter and add programs from another region without losing the previous wishes.
if st.session_state.get(editor_source_key) != table_key or editor_state_key not in st.session_state:
    st.session_state[editor_source_key] = table_key
    st.session_state[editor_state_key] = clean_wish_rows(base_rows)

editor_rows = st.session_state[editor_state_key].copy()

# Drop values that no longer exist in the loaded capacities file, but preserve all
# valid selected programs even when the region filter changes.
if PROGRAM in editor_rows.columns:
    editor_rows[PROGRAM] = editor_rows[PROGRAM].map(
        lambda x: x if str(x).strip() in program_mapping or str(x).strip() == "" else ""
    )

region_options = ["All regions"] + available_regions(calib)
selected_program_region = st.selectbox(
    "Program region",
    region_options,
    index=0,
    help=(
        "Choose a region to make the program list shorter. Already selected "
        "programs from other regions are kept in the table, so the student can "
        "build a wish list across several regions."
    ),
)

current_program_values = (
    editor_rows.get(PROGRAM, pd.Series(dtype=str))
    .dropna()
    .astype(str)
    .str.strip()
    .tolist()
)
program_options_for_editor = filter_program_options(
    program_mapping,
    selected_program_region,
    active_filters=program_filters,
    current_values=current_program_values,
)

options_signature = hashlib.md5(
    "|".join(program_options_for_editor).encode("utf-8")
).hexdigest()[:8]

editor_widget_key = f"{editor_widget_key_base}_{options_signature}"

if selected_program_region != "All regions" or filters_are_active(program_filters):
    preserved = [
        p for p in current_program_values
        if p in program_mapping
        and not (
            (selected_program_region == "All regions" or str(program_mapping[p].get(REGION, UNKNOWN_REGION)).strip() == selected_program_region)
            and program_matches_filters(program_mapping[p], program_filters)
        )
    ]
    matching_count = max(len(program_options_for_editor) - len(preserved), 0)
    extra_note = (
        f" Existing selected program(s) outside the current filters are also kept available: "
        f"{len(preserved)}."
        if preserved else ""
    )
    region_text = selected_program_region if selected_program_region != "All regions" else "all regions"
    st.caption(
        f"Showing {matching_count} matching program option(s) for {region_text}."
        f"{extra_note}"
    )

col_config: dict = {
    WISH_RANK: st.column_config.NumberColumn("Wish rank", min_value=1, step=1, width=95),
    PROGRAM:   st.column_config.SelectboxColumn(
        "Program",
        options=[""] + program_options_for_editor,
        width="large",
        help=(
            "Use the Program region selector and the sidebar filters to shorten the list. "
            "Each option shows the school RBD and a readable program name. "
            "Selections already made outside the current filters stay in the table."
        ),
    ),
    "priority_sibling":              st.column_config.CheckboxColumn("Sibling priority", width="small"),
    "priority_student":              st.column_config.CheckboxColumn(
        "Priority student",
        width="medium",
        help=(
            "RSH means Registro Social de Hogares. This box should be checked when "
            "the student belongs to the lowest 40% socioeconomic-vulnerability group "
            "and is eligible for the Chilean priority-student criterion."
        ),
    ),
    "priority_parent_civil_servant": st.column_config.CheckboxColumn("Civil-servant parent priority", width="medium"),
    "priority_ex_student":           st.column_config.CheckboxColumn("Former-student priority", width="medium"),
    SAFETY: st.column_config.CheckboxColumn("Already enrolled", width="medium"),
}

if use_stb:
    col_config[LOTTERY] = st.column_config.NumberColumn(
        "Common lottery number", min_value=1, step=1, width="medium"
    )
else:
    # In MTB mode, the number is computed automatically through the hash, so the column is hidden
    editor_rows = editor_rows.drop(columns=[LOTTERY], errors="ignore")

edited = st.data_editor(
    editor_rows,
    num_rows="dynamic",
    width="stretch",
    hide_index=True,
    key=editor_widget_key,
    column_config=col_config,
    column_order=[
        WISH_RANK,
        PROGRAM,
        "priority_sibling",
        "priority_student",
        "priority_parent_civil_servant",
        "priority_ex_student",
        SAFETY,
    ] + ([LOTTERY] if use_stb else []),
)

# Persist edits so changing Program region does not reset previously entered wishes.
cleaned_edited = clean_wish_rows(edited)

old_state = clean_wish_rows(st.session_state[editor_state_key])

if not cleaned_edited.astype(str).equals(old_state.astype(str)):
    st.session_state[editor_state_key] = cleaned_edited
    st.rerun()

edited = cleaned_edited

# Imputed-calibration warning
selected = [p for p in edited[PROGRAM].dropna().astype(str).str.strip() if p]
imputed  = [
    p for p in selected
    if p in program_mapping and as_bool(program_mapping[p].get(IMPUTED, False))
]
if imputed:
    st.warning(
        "Less reliable estimate: at least one selected program uses "
        "mean-imputed 2024 calibration values."
    )

# ── MTB percentile preview ────────────────────────────────────────────────
if not use_stb and selected and national_student_id.strip():
    try:
        preview_w   = attach_mtb_hashes(edited, program_mapping, national_student_id)
        preview_cols = [WISH_RANK, PROGRAM, LOTTERY, HASH_PCT]
        preview     = (
            preview_w[preview_w[PROGRAM].astype(str).str.strip() != ""][preview_cols]
            .copy()
        )
        preview[HASH_PCT] = (
            pd.to_numeric(preview[HASH_PCT], errors="coerce")
            .map(lambda x: "" if pd.isna(x) else f"{x:.4f}")
        )
        with st.expander("Calculated MTB percentiles (RUN + RBD)", expanded=False):
            st.dataframe(preview, width="stretch", hide_index=True)
    except Exception as exc:
        st.warning(f"MTB preview unavailable: {exc}")

# ── Common STB number ─────────────────────────────────────────────────────────
common_lottery = None
if use_stb:
    valid_lots = pd.to_numeric(edited[LOTTERY], errors="coerce").dropna()
    default_lot = int(max(round(valid_lots.iloc[0]), 1)) if not valid_lots.empty else 1
    with st.sidebar:
        common_lottery = st.number_input(
            "Common STB lottery number",
            min_value=1, value=default_lot, step=1,
            help="Applied to all schools in STB mode.",
        )

# ── Section 2: simulation ────────────────────────────────────────────────────
st.subheader("2. Run the simulation")

if st.button("Calculate unmatched risk", type="primary"):
    if lottery_mode == "MTB" and not national_student_id.strip():
        st.error("Please enter the student's RUN/IPE before running the MTB simulation.")
        st.stop()

    try:
        wishes_for_compute = edited
        if lottery_mode == "MTB":
            wishes_for_compute = attach_mtb_hashes(edited, program_mapping, national_student_id)

        choices    = compute(
            wishes_for_compute,
            program_mapping,
            lottery_mode=lottery_mode,
            common_lottery=common_lottery,
            stb_population=stb_population,
        )
        p_unmatched = float(choices["cumulative_unavailable_after_choice"].iloc[-1])
        at_risk     = p_unmatched >= threshold

        # Columns to display
        display_cols = [
            "wish_rank",
            "program",
            "lottery_mode",
            "lottery_number",
            "priority_tier",
            "capacity",
            "true_applicants_last_year",
            "availability_probability",
            "choice_assignment_probability",
        ]

        table = choices[display_cols].copy()

        for prob_col in ("availability_probability", "choice_assignment_probability"):
            table[prob_col] = (
                table[prob_col].astype(float).map(lambda x: f"{x:.1%}")
            )

        table = table.rename(columns={
            "wish_rank": "Wish rank",
            "program": "Program",
            "lottery_mode": "Lottery mode",
            "lottery_number": "Lottery number",
            "priority_tier": "Priority tier",
            "capacity": "Seats",
            "true_applicants_last_year": "True applicants last year",
            "availability_probability": "Chance if considered",
            "choice_assignment_probability": "Final chance of assignment",
        })

        st.subheader("Wish-level details")
        st.caption(
            "Chance if considered is the chance of getting that program if the student "
            "reaches that wish. Final chance of assignment also accounts for all higher-ranked wishes."
        )
        st.dataframe(table, width="stretch", hide_index=True)

        st.subheader("Summary")

        positive = (
            choices[choices["choice_assignment_probability"] > 0]
            .sort_values("choice_assignment_probability", ascending=False)
            .reset_index(drop=True)
        )

        if at_risk:
            st.error(
                "The student is at risk of remaining unmatched. "
                "The list appears risky; adding safer options is recommended."
            )
            if positive.empty:
                st.markdown("**Most likely outcome:**")
                st.write("1. Unmatched")
            else:
                st.markdown("**Most likely outcomes:**")
                st.write("1. Unmatched")
                for i, row in positive.head(2).iterrows():
                    st.write(f"{i + 2}. {row['program']}")
        else:
            if positive.empty:
                st.error("No listed school appears realistically accessible.")
            else:
                best = positive.iloc[0]
                st.success(
                    f"The student is not flagged as at risk. "
                    f"The most likely assignment is: **{best['program']}**."
                )
                st.markdown("**Top 3 most likely schools:**")
                for i, row in positive.head(3).iterrows():
                    st.write(f"{i + 1}. {row['program']}")

    except ValueError as exc:
        st.error(str(exc))

    except Exception as exc:
        st.error("Unexpected error during the simulation.")
        st.exception(exc)
