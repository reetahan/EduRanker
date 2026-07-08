from __future__ import annotations

import hashlib
import io

import numpy as np
import pandas as pd
import streamlit as st
from scipy.stats import hypergeom

WISH_RANK = "wish_rank"
PROGRAM = "program"
LOTTERY = "lottery_number"
CAPACITY = "total_admission_seats"
TRUE_APP = "true_applicants_last_year"
POP = "program_lottery_population_2024"
IMPUTED = "calibration_2024_imputed"
IMPUT_METHOD = "calibration_2024_imputation_method"
PRIORITY_STUDENT_QUOTA_SHARE = 0.15

PRIORITIES = [
    "priority_sibling",
    "priority_student",
    "priority_parent_civil_servant",
    "priority_ex_student",
]
SAFETY = "priority_already_registered"
NO_PRIORITY = "no_priority"
TIERS = PRIORITIES + [NO_PRIORITY]


def read_csv(file_bytes: bytes, sep="auto") -> pd.DataFrame:
    kwargs = {"dtype": str, "encoding": "utf-8-sig"}
    if sep == "auto":
        kwargs |= {"sep": None, "engine": "python"}
    else:
        kwargs["sep"] = sep
    df = pd.read_csv(io.BytesIO(file_bytes), **kwargs)
    df.columns = [str(c).lstrip("\ufeff").strip() for c in df.columns]
    return df


def norm_code(s: pd.Series) -> pd.Series:
    def f(x):
        x = str(x).strip()
        if x.startswith('="') and x.endswith('"'):
            x = x[2:-1].strip()
        try:
            return str(int(float(x.replace(",", "."))))
        except Exception:
            return x
    return s.map(f)


def as_bool(x) -> bool:
    if pd.isna(x):
        return False
    return str(x).strip().lower() in {"1", "true", "yes", "y", "x", "oui"}


def as_float(x, default=0.0) -> float:
    try:
        if pd.isna(x):
            return default
        return float(str(x).strip().replace(",", "."))
    except Exception:
        return default


@st.cache_data(show_spinner=False)
def load_calibration(file_bytes: bytes) -> pd.DataFrame:
    df = read_csv(file_bytes, sep=";")
    if len(df.columns) == 1:
        df = read_csv(file_bytes, sep=",")
    df["program_code"] = norm_code(df["program_code"])
    return df


def required_cols() -> list[str]:
    cols = ["rbd", "program_code", CAPACITY, TRUE_APP, POP]
    for tier in TIERS:
        cols += [
            f"priority_share_{tier}_2024",
            f"cum_share_before_{tier}_2024",
            f"cum_share_through_{tier}_2024",
        ]
    return cols


def build_options(calib: pd.DataFrame):
    options, mapping = [], {}
    name_cols = [c for c in ["school_name", "program_name", "name", "label", "formation"] if c in calib.columns]
    for _, row in calib.drop_duplicates(["rbd", "program_code"]).iterrows():
        rbd, code = str(row["rbd"]).strip(), str(row["program_code"]).strip()
        extra = ""
        if name_cols:
            names = [str(row[c]).strip() for c in name_cols if pd.notna(row[c]) and str(row[c]).strip()]
            extra = " — " + " / ".join(names[:2]) if names else ""
        label = f"{rbd} || {code}{extra}"
        options.append(label)
        mapping[label] = row
    return sorted(options), mapping


def empty_wishes() -> pd.DataFrame:
    df = pd.DataFrame({WISH_RANK: [1, 2, 3], PROGRAM: ["", "", ""], LOTTERY: [1, 1, 1]})
    for col in PRIORITIES + [SAFETY]:
        df[col] = False
    return df


def parse_wishes(file_bytes: bytes, mapping: dict[str, pd.Series]) -> pd.DataFrame:
    df = read_csv(file_bytes, sep="auto")
    base_to_label = {f"{r['rbd']} || {r['program_code']}": label for label, r in mapping.items()}

    if {WISH_RANK, PROGRAM, LOTTERY}.issubset(df.columns):
        out = pd.DataFrame({WISH_RANK: df[WISH_RANK], PROGRAM: df[PROGRAM], LOTTERY: df[LOTTERY]})
    elif {"rang_du_voeu", "programme", "numero_loterie"}.issubset(df.columns):
        out = pd.DataFrame({WISH_RANK: df["rang_du_voeu"], PROGRAM: df["programme"], LOTTERY: df["numero_loterie"]})
    elif {"rbd", "program_code", "lottery", "preference_number"}.issubset(df.columns):
        labels = df["rbd"].astype(str).str.strip() + " || " + norm_code(df["program_code"])
        out = pd.DataFrame({WISH_RANK: df["preference_number"], PROGRAM: labels, LOTTERY: df["lottery"]})
    else:
        raise ValueError("Expected columns: wish_rank/program/lottery_number or rang_du_voeu/programme/numero_loterie.")

    out[WISH_RANK] = pd.to_numeric(out[WISH_RANK], errors="coerce").fillna(1).astype(int)
    out[LOTTERY] = pd.to_numeric(out[LOTTERY], errors="coerce").fillna(1).astype(int)
    out[PROGRAM] = out[PROGRAM].astype(str).str.strip().map(lambda x: x if x in mapping else base_to_label.get(x, ""))

    for col in PRIORITIES + [SAFETY]:
        out[col] = df[col].map(as_bool) if col in df.columns else False
    return out.sort_values(WISH_RANK).reset_index(drop=True) if not out.empty else empty_wishes()


def priority_tier(wish: pd.Series, program: pd.Series) -> str:
    if as_bool(wish.get("priority_sibling")):
        return "priority_sibling"

    if as_bool(wish.get("priority_student")):
        capacity = max(round(as_float(program[CAPACITY])), 0)
        quota_count = int(np.floor(PRIORITY_STUDENT_QUOTA_SHARE * capacity))
        lottery = max(round(as_float(wish[LOTTERY], 1)), 1)
        if lottery <= quota_count:
            return "priority_student"

    if as_bool(wish.get("priority_parent_civil_servant")):
        return "priority_parent_civil_servant"
    if as_bool(wish.get("priority_ex_student")):
        return "priority_ex_student"
    return NO_PRIORITY


def availability(wish: pd.Series, program: pd.Series) -> dict:
    capacity = max(round(as_float(program[CAPACITY])), 0)
    tier = priority_tier(wish, program)
    true_app = max(round(as_float(program[TRUE_APP])), 0)
    population = max(round(as_float(program[POP], 1)), 1)
    lottery = max(round(as_float(wish[LOTTERY], 1)), 1)

    share = as_float(program[f"priority_share_{tier}_2024"])
    before = as_float(program[f"cum_share_before_{tier}_2024"])
    raw_rank = min(lottery, population)
    percentile = np.clip((raw_rank - 1) / max(population - 1, 1), 0, 1)
    effective_rank = int(1 + np.floor(np.clip(before + share * percentile, 0, 1) * max(population - 1, 0)))
    effective_rank = min(max(effective_rank, 1), population)

    if as_bool(wish.get(SAFETY)):
        p_available = 1.0
    elif capacity <= 0:
        p_available = 0.0
    else:
        M = max(population - 1, 0)
        draws = min(max(effective_rank - 1, 0), M)
        successes = min(max(true_app - 1, 0), M)
        p_available = 1.0 if draws == 0 or successes == 0 else float(hypergeom.cdf(capacity - 1, M, successes, draws))

    return {
        "wish_rank": int(wish[WISH_RANK]),
        "program": wish[PROGRAM],
        "lottery_number": lottery,
        "priority_tier": tier,
        "capacity": capacity,
        "true_applicants_last_year": true_app,
        "program_lottery_population_2024": population,
        "raw_lottery_rank": raw_rank,
        "priority_effective_rank": effective_rank,
        "availability_probability": np.clip(p_available, 0, 1),
        "calibration_2024_imputed": as_bool(program.get(IMPUTED, False)),
        "calibration_2024_imputation_method": str(program.get(IMPUT_METHOD, "")),
    }


def compute(wishes: pd.DataFrame, mapping: dict[str, pd.Series]) -> pd.DataFrame:
    clean = wishes[wishes[PROGRAM].astype(str).str.strip() != ""].sort_values(WISH_RANK)
    if clean.empty:
        raise ValueError("Add at least one valid wish.")
    rows = [availability(wish, mapping[wish[PROGRAM]]) for _, wish in clean.iterrows() if wish[PROGRAM] in mapping]
    choices = pd.DataFrame(rows)
    choices["cumulative_unavailable_before_choice"] = (1 - choices["availability_probability"]).cumprod().shift(1).fillna(1)
    choices["choice_assignment_probability"] = choices["cumulative_unavailable_before_choice"] * choices["availability_probability"]
    choices["cumulative_unavailable_after_choice"] = (1 - choices["availability_probability"]).cumprod()
    return choices


def risk_label(p: float) -> str:
    if p < 0.05:
        return "Very low risk"
    if p < 0.15:
        return "Low to moderate risk"
    if p < 0.30:
        return "Notable risk"
    return "High risk"


st.set_page_config(page_title="MTB unmatched-risk simulation", page_icon="🎓", layout="wide")
st.title("Simulation of the risk of remaining unmatched")
st.caption("Uses the 2025 capacities file enriched with 2024 calibration data.")

with st.sidebar:
    calib_file = st.file_uploader("Capacities + 2024 calibration file", type=["csv"])
    threshold = st.slider("Unmatched-risk alert threshold", 0.01, 1.0, 0.025, 0.005)

if calib_file is None:
    st.info("Add the capacities + 2024 calibration CSV in the sidebar to start.")
    st.stop()

calib = load_calibration(calib_file.getvalue())
missing = [c for c in required_cols() if c not in calib.columns]
if missing:
    st.error("Missing columns: " + ", ".join(missing[:20]))
    st.stop()

program_options, program_mapping = build_options(calib)
st.subheader("1. Enter or import the student's wishes")
wish_file = st.file_uploader("Optional: import a student wishes CSV to pre-fill the table", type=["csv"])

if wish_file is not None:
    try:
        base_rows = parse_wishes(wish_file.getvalue(), program_mapping)
        st.success(f"Table pre-filled with {len(base_rows)} wish/wishes.")
    except Exception as e:
        st.error(f"Could not import the student CSV: {e}")
        base_rows = empty_wishes()
else:
    base_rows = empty_wishes()

key = hashlib.md5(wish_file.getvalue()).hexdigest()[:8] if wish_file else "empty"
edited = st.data_editor(
    base_rows,
    num_rows="dynamic",
    use_container_width=True,
    key=f"wishes_{key}",
    column_config={
        WISH_RANK: st.column_config.NumberColumn("Wish rank", min_value=1, step=1),
        PROGRAM: st.column_config.SelectboxColumn("Program", options=[""] + program_options),
        LOTTERY: st.column_config.NumberColumn("Lottery number", min_value=1, step=1),
        "priority_sibling": st.column_config.CheckboxColumn("Sibling priority"),
        "priority_student": st.column_config.CheckboxColumn("Student priority"),
        "priority_parent_civil_servant": st.column_config.CheckboxColumn("Civil-servant parent priority"),
        "priority_ex_student": st.column_config.CheckboxColumn("Former-student priority"),
        SAFETY: st.column_config.CheckboxColumn("Already enrolled / safety"),
    },
)

selected = [p for p in edited[PROGRAM].dropna().astype(str).str.strip() if p]
imputed = [p for p in selected if p in program_mapping and as_bool(program_mapping[p].get(IMPUTED, False))]
if imputed:
    st.warning("Less reliable estimate: at least one selected program uses average-imputed 2024 calibration values.")

st.subheader("2. Run the simulation")
if st.button("Calculate unmatched risk", type="primary"):
    try:
        choices = compute(edited, program_mapping)
        p_unmatched = float(choices["cumulative_unavailable_after_choice"].iloc[-1])
        at_risk = p_unmatched >= threshold

        display_cols = [
            "wish_rank",
            "program",
            "lottery_number",
            "priority_tier",
            "capacity",
            "true_applicants_last_year",
            "choice_assignment_probability",
        ]

        table = choices[display_cols].copy()
        table["choice_assignment_probability"] = (
            table["choice_assignment_probability"]
            .astype(float)
            .map(lambda x: f"{x:.1%}")
        )

        st.subheader("Wish-level details")
        st.dataframe(table, use_container_width=True, hide_index=True)

        st.subheader("Summary")

        positive_choices = (
            choices[choices["choice_assignment_probability"] > 0]
            .sort_values("choice_assignment_probability", ascending=False)
            .reset_index(drop=True)
        )

        if at_risk:
            st.error(
                "The student is at risk of remaining unmatched. "
                "The list appears risky, so the student should consider adding safer options."
            )

            if positive_choices.empty:
                st.markdown("**Most likely outcome:**")
                st.write("1. Unmatched")
            else:
                st.markdown("**Most likely outcomes:**")
                st.write("1. Unmatched")

                for i, row in positive_choices.head(2).iterrows():
                    st.write(f"{i + 2}. {row['program']}")

        else:
            if positive_choices.empty:
                st.error(
                    "The student is at risk of remaining unmatched. "
                    "No listed school appears realistically available."
                )
            else:
                best_choice = positive_choices.iloc[0]

                st.success(
                    f"The student is not flagged as at risk of remaining unmatched. "
                    f"The most likely assignment is: **{best_choice['program']}**."
                )

                st.markdown("**Top 3 most likely schools:**")
                for i, row in positive_choices.head(3).iterrows():
                    st.write(f"{i + 1}. {row['program']}")
    except Exception as e:
        st.error("Error during the simulation")
        st.exception(e)
