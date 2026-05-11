"""
Risk Trends Page

Visualises how project risk scores evolve over time using snapshots saved
by the ML Analysis and Rankings pages.
"""

from __future__ import annotations

import sys
from pathlib import Path

_p = Path(__file__).resolve().parent
_repo_root = _p.parent.parent if _p.name == "pages" else _p.parent
if str(_repo_root) not in sys.path:
    sys.path.insert(0, str(_repo_root))

import pandas as pd
import streamlit as st

from app.bootstrap import init_page

init_page()

from src.data.snapshot_store import SnapshotStore

st.title("📉 Risk Trends")
st.markdown("Track how project risk scores change over time across ML and MCDA analyses.")

store = SnapshotStore()
df = store.load_snapshots()

if df.empty:
    st.info(
        "No risk snapshots recorded yet.\n\n"
        "Run **ML Analysis** (page 3) or **Rankings** (page 5) to start capturing "
        "trend data. Each run appends a timestamped snapshot automatically."
    )
    st.stop()

# ── Filters ────────────────────────────────────────────────────────────────
st.markdown("---")
col_f1, col_f2 = st.columns([2, 1])

with col_f1:
    all_projects = sorted(df["project_name"].dropna().unique().tolist())
    selected_projects = st.multiselect(
        "Filter projects",
        options=all_projects,
        default=all_projects[:10] if len(all_projects) > 10 else all_projects,
        help="Select one or more projects to display trend lines.",
    )

with col_f2:
    source_opts = ["All"] + sorted(df["source"].dropna().unique().tolist())
    selected_source = st.selectbox("Snapshot source", source_opts)

filtered = df.copy()
if selected_projects:
    filtered = filtered[filtered["project_name"].isin(selected_projects)]
if selected_source != "All":
    filtered = filtered[filtered["source"] == selected_source]

if filtered.empty:
    st.warning("No data matches the current filters.")
    st.stop()

# ── ML Risk Score Trend ─────────────────────────────────────────────────────
ml_data = filtered.dropna(subset=["risk_score"])
if not ml_data.empty:
    st.markdown("### ML Risk Score Over Time")

    try:
        import plotly.graph_objects as go

        fig = go.Figure()

        fig.add_hrect(
            y0=0.6,
            y1=1.05,
            fillcolor="red",
            opacity=0.07,
            line_width=0,
            annotation_text="High ≥ 0.6",
            annotation_position="top left",
        )
        fig.add_hrect(
            y0=0.3,
            y1=0.6,
            fillcolor="orange",
            opacity=0.07,
            line_width=0,
            annotation_text="Medium 0.3–0.6",
            annotation_position="top left",
        )
        fig.add_hrect(
            y0=-0.05,
            y1=0.3,
            fillcolor="green",
            opacity=0.07,
            line_width=0,
            annotation_text="Low < 0.3",
            annotation_position="top left",
        )

        for project in ml_data["project_name"].unique():
            proj_df = ml_data[ml_data["project_name"] == project].sort_values("timestamp")
            fig.add_trace(
                go.Scatter(
                    x=proj_df["timestamp"],
                    y=proj_df["risk_score"],
                    mode="lines+markers",
                    name=project,
                )
            )

        fig.update_layout(
            xaxis_title="Snapshot Time",
            yaxis_title="ML Risk Score",
            yaxis_range=[-0.05, 1.05],
            legend_title="Project",
            height=450,
            hovermode="x unified",
        )
        st.plotly_chart(fig, width="stretch")

    except ImportError:
        st.line_chart(
            ml_data.pivot_table(index="timestamp", columns="project_name", values="risk_score")
        )

# ── MCDA Score Trend ────────────────────────────────────────────────────────
mcda_data = filtered.dropna(subset=["mcda_score"])
if not mcda_data.empty:
    st.markdown("### MCDA Score Over Time")

    try:
        import plotly.graph_objects as go

        fig2 = go.Figure()
        for project in mcda_data["project_name"].unique():
            proj_df = mcda_data[mcda_data["project_name"] == project].sort_values("timestamp")
            fig2.add_trace(
                go.Scatter(
                    x=proj_df["timestamp"],
                    y=proj_df["mcda_score"],
                    mode="lines+markers",
                    name=project,
                )
            )
        fig2.update_layout(
            xaxis_title="Snapshot Time",
            yaxis_title="MCDA Score",
            legend_title="Project",
            height=400,
            hovermode="x unified",
        )
        st.plotly_chart(fig2, width="stretch")

    except ImportError:
        st.line_chart(
            mcda_data.pivot_table(index="timestamp", columns="project_name", values="mcda_score")
        )

# ── Crisis Detection ─────────────────────────────────────────────────────────
st.markdown("---")
st.markdown("### Crisis Detection")
st.caption("Projects that moved Low → Medium → High across 3 or more consecutive snapshots.")

_LEVEL_ORDER = {"Low": 0, "Medium": 1, "High": 2}

crisis_rows: list[dict] = []
for project in filtered["project_name"].unique():
    proj_df = (
        filtered[filtered["project_name"] == project]
        .sort_values("timestamp")
        .dropna(subset=["risk_level"])
    )
    if len(proj_df) < 3:
        continue
    levels = proj_df["risk_level"].map(_LEVEL_ORDER).dropna().tolist()
    # sliding window of 3: check for strictly increasing sequence hitting High
    for i in range(len(levels) - 2):
        window = levels[i : i + 3]
        if window[0] < window[1] < window[2] and window[2] == _LEVEL_ORDER["High"]:
            row_high = proj_df.iloc[i + 2]
            crisis_rows.append(
                {
                    "Project": project,
                    "First snapshot": proj_df.iloc[i]["timestamp"],
                    "Crisis timestamp": row_high["timestamp"],
                    "Final risk score": row_high.get("risk_score", ""),
                }
            )
            break

if crisis_rows:
    st.dataframe(
        pd.DataFrame(crisis_rows),
        width="stretch",
        hide_index=True,
    )
else:
    st.success(
        "No escalating risk patterns (Low → Medium → High) detected in the current selection."
    )

# ── Raw snapshots ────────────────────────────────────────────────────────────
with st.expander("Raw snapshot data"):
    st.dataframe(
        filtered.sort_values("timestamp", ascending=False),
        width="stretch",
        hide_index=True,
    )
