"""
Usability Survey Page (SUS)

Implements the System Usability Scale (SUS) — a validated, 10-question
questionnaire for measuring perceived usability.

SUS scoring formula
-------------------
- Odd questions  (1, 3, 5, 7, 9): converted score = response - 1
- Even questions (2, 4, 6, 8, 10): converted score = 5 - response
- Final SUS score = sum(converted scores) × 2.5  [range 0–100]

Adjective grades
----------------
> 80.3  → A  — Excellent
68–80.3 → B  — Good (acceptable threshold)
51–68   → C/D — OK / Poor
< 51    → F  — Unacceptable
"""

from __future__ import annotations

import sys
from datetime import datetime, timezone
from pathlib import Path

_p = Path(__file__).resolve().parent
_repo_root = _p.parent.parent if _p.name == "pages" else _p.parent
if str(_repo_root) not in sys.path:
    sys.path.insert(0, str(_repo_root))

import pandas as pd
import streamlit as st

from app.bootstrap import init_page

init_page()

_SUS_CSV = _repo_root / "data" / "processed" / "sus_responses.csv"
_SUS_CSV.parent.mkdir(parents=True, exist_ok=True)

_SUS_QUESTIONS = [
    "I think that I would like to use this system frequently.",
    "I found the system unnecessarily complex.",
    "I thought the system was easy to use.",
    "I think that I would need the support of a technical person to be able to use this system.",
    "I found the various functions in this system were well integrated.",
    "I thought there was too much inconsistency in this system.",
    "I would imagine that most people would learn to use this system very quickly.",
    "I found the system very cumbersome to use.",
    "I felt very confident using the system.",
    "I needed to learn a lot of things before I could get going with this system.",
]

_SCALE_LABELS = [
    "Strongly Disagree",
    "Disagree",
    "Neutral",
    "Agree",
    "Strongly Agree",
]
_SCALE_VALUES = {label: i + 1 for i, label in enumerate(_SCALE_LABELS)}


def _calculate_sus(responses: list[int]) -> float:
    """Compute SUS score from a list of 10 integer responses (1–5)."""
    total = 0
    for i, r in enumerate(responses):
        if (i + 1) % 2 == 1:
            total += r - 1
        else:
            total += 5 - r
    return total * 2.5


def _sus_grade(score: float) -> tuple[str, str]:
    """Return (grade letter, adjective) for a SUS score."""
    if score > 80.3:
        return "A", "Excellent"
    if score >= 68:
        return "B", "Good"
    if score >= 51:
        return "C", "OK / Poor"
    return "F", "Unacceptable"


def _load_responses() -> pd.DataFrame:
    if _SUS_CSV.exists():
        return pd.read_csv(_SUS_CSV)
    cols = (
        ["timestamp", "respondent", "role"]
        + [f"q{i+1}" for i in range(10)]
        + ["sus_score", "grade"]
    )
    return pd.DataFrame(columns=cols)


def _append_response(
    respondent: str, role: str, answers: list[int], score: float, grade: str
) -> None:
    df = _load_responses()
    row: dict = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "respondent": respondent,
        "role": role,
        **{f"q{i+1}": answers[i] for i in range(10)},
        "sus_score": round(score, 2),
        "grade": grade,
    }
    df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
    df.to_csv(_SUS_CSV, index=False)


# ── Page layout ──────────────────────────────────────────────────────────────
st.title("📋 Usability Survey (SUS)")
st.markdown(
    "Please complete the **System Usability Scale** questionnaire after exploring "
    "the PRISM dashboard. Your feedback helps measure and improve the system's usability."
)

tab_survey, tab_results = st.tabs(["Complete Survey", "Aggregate Results"])

# ── Survey tab ───────────────────────────────────────────────────────────────
with tab_survey:
    st.markdown("### Respondent Information *(optional)*")
    col_r1, col_r2 = st.columns(2)
    with col_r1:
        respondent_name = st.text_input("Name / Alias", placeholder="e.g. Evaluator 1")
    with col_r2:
        respondent_role = st.selectbox(
            "Role",
            ["", "Project Manager", "Developer", "Analyst", "Researcher", "Student", "Other"],
        )

    st.markdown("---")
    st.markdown(
        "### Instructions\n"
        "For each statement, select the response that best describes your experience "
        "with PRISM. There are no right or wrong answers — please respond based on your "
        "immediate impressions."
    )

    answers: list[int] = []
    with st.form("sus_form"):
        for i, question in enumerate(_SUS_QUESTIONS):
            st.markdown(f"**{i + 1}.** {question}")
            choice = st.select_slider(
                f"q{i + 1}",
                options=_SCALE_LABELS,
                value="Neutral",
                label_visibility="collapsed",
                key=f"sus_q{i + 1}",
            )
            answers.append(_SCALE_VALUES[choice])
            st.markdown("")

        submitted = st.form_submit_button("Submit Survey", type="primary")

    if submitted:
        score = _calculate_sus(answers)
        grade, adjective = _sus_grade(score)
        _append_response(
            respondent=respondent_name or "Anonymous",
            role=respondent_role or "Not specified",
            answers=answers,
            score=score,
            grade=f"{grade} – {adjective}",
        )

        colour = (
            "#00CC66"
            if grade == "A"
            else "#1E88E5" if grade == "B" else "#FFA500" if grade == "C" else "#FF4B4B"
        )
        st.markdown(
            f"<div style='background:{colour};color:#fff;padding:16px;border-radius:8px;"
            f"text-align:center;font-size:1.4em'>"
            f"Your SUS Score: <strong>{score:.1f} / 100</strong> &nbsp;|&nbsp; "
            f"Grade: <strong>{grade}</strong> — {adjective}"
            f"</div>",
            unsafe_allow_html=True,
        )
        st.markdown("")
        st.info(
            "Thank you for completing the survey. Your response has been recorded. "
            "Switch to the **Aggregate Results** tab to see overall scores."
        )

# ── Aggregate Results tab ────────────────────────────────────────────────────
with tab_results:
    df_resp = _load_responses()

    if df_resp.empty:
        st.info("No survey responses yet. Complete the survey above to see aggregate results.")
    else:
        df_resp["sus_score"] = pd.to_numeric(df_resp["sus_score"], errors="coerce")
        valid = df_resp.dropna(subset=["sus_score"])
        n = len(valid)
        mean_score = valid["sus_score"].mean()
        grade, adjective = _sus_grade(mean_score)

        st.markdown(f"### Results — {n} respondent{'s' if n != 1 else ''}")
        c1, c2, c3 = st.columns(3)
        c1.metric("Mean SUS Score", f"{mean_score:.1f} / 100")
        c2.metric("Overall Grade", f"{grade} — {adjective}")
        c3.metric("Respondents", n)

        st.markdown("#### Score Distribution")
        try:
            import plotly.express as px

            fig = px.histogram(
                valid,
                x="sus_score",
                nbins=10,
                range_x=[0, 100],
                labels={"sus_score": "SUS Score"},
                color_discrete_sequence=["#1E88E5"],
            )
            fig.add_vline(
                x=68,
                line_dash="dash",
                line_color="orange",
                annotation_text="Acceptable (68)",
                annotation_position="top right",
            )
            fig.add_vline(
                x=80.3,
                line_dash="dash",
                line_color="green",
                annotation_text="Excellent (80.3)",
                annotation_position="top right",
            )
            fig.update_layout(height=300, bargap=0.1)
            st.plotly_chart(fig, width="stretch")
        except ImportError:
            st.bar_chart(valid["sus_score"].value_counts().sort_index())

        st.markdown("#### All Responses")
        display_cols = ["timestamp", "respondent", "role", "sus_score", "grade"]
        st.dataframe(
            valid[[c for c in display_cols if c in valid.columns]],
            width="stretch",
            hide_index=True,
        )

        st.download_button(
            label="📥 Export Responses to CSV",
            data=valid.to_csv(index=False),
            file_name="sus_responses.csv",
            mime="text/csv",
            help="Download all responses for thesis documentation. "
            "Note: this file is not committed to version control.",
        )
