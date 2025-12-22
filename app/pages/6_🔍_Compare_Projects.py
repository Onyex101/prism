"""
Compare Projects Page

Side-by-side project comparison.
"""

import streamlit as st
import pandas as pd
import plotly.graph_objects as go

st.set_page_config(page_title="Compare Projects - PRISM", page_icon="🔍", layout="wide")

st.title("🔍 Compare Projects")
st.markdown("Side-by-side comparison of project metrics and risk factors")

# Check if data is loaded
if "projects_df" not in st.session_state:
    st.warning("⚠️ No data loaded. Please upload data first.")
    if st.button("Go to Upload Page"):
        st.switch_page("pages/2_📁_Upload_Data.py")
    st.stop()

df = st.session_state["projects_df"]

# Project selection
st.markdown("### Select Projects to Compare")

name_col = "project_name" if "project_name" in df.columns else "project_id"
project_list = df[name_col].tolist()

col1, col2 = st.columns(2)

with col1:
    project_a = st.selectbox("Project A", project_list, index=0)

with col2:
    project_b = st.selectbox(
        "Project B",
        project_list,
        index=min(1, len(project_list) - 1),
    )

if project_a == project_b:
    st.warning("Please select two different projects to compare.")
    st.stop()

# Get project data
row_a = df[df[name_col] == project_a].iloc[0]
row_b = df[df[name_col] == project_b].iloc[0]

# Comparison section
st.markdown("---")
st.markdown("### Side-by-Side Comparison")

# Metrics comparison
metrics_to_compare = [
    ("completion_rate", "Completion Rate", "%"),
    ("budget", "Budget", "$"),
    ("spent", "Spent", "$"),
    ("team_size", "Team Size", ""),
    ("risk_score", "Risk Score", ""),
    ("mcda_score", "MCDA Score", ""),
]

col1, col2, col3 = st.columns([2, 1, 2])

with col1:
    st.markdown(f"#### {project_a}")

with col2:
    st.markdown("#### Metric")

with col3:
    st.markdown(f"#### {project_b}")

for metric, label, unit in metrics_to_compare:
    if metric in df.columns:
        col1, col2, col3 = st.columns([2, 1, 2])

        val_a = row_a[metric]
        val_b = row_b[metric]

        with col1:
            if unit == "$":
                st.write(f"${val_a:,.0f}")
            elif unit == "%":
                st.write(f"{val_a:.1f}%")
            else:
                st.write(f"{val_a:.2f}" if isinstance(val_a, float) else val_a)

        with col2:
            st.write(f"**{label}**")

        with col3:
            if unit == "$":
                st.write(f"${val_b:,.0f}")
            elif unit == "%":
                st.write(f"{val_b:.1f}%")
            else:
                st.write(f"{val_b:.2f}" if isinstance(val_b, float) else val_b)

# Radar chart comparison
st.markdown("---")
st.markdown("### Visual Comparison")

# Prepare radar data
radar_metrics = ["completion_rate", "budget_utilization", "team_size", "risk_score", "mcda_score"]
available_metrics = [m for m in radar_metrics if m in df.columns]

if len(available_metrics) >= 3:
    # Normalize values for radar
    values_a = []
    values_b = []

    for metric in available_metrics:
        col_min = df[metric].min()
        col_max = df[metric].max()
        range_val = col_max - col_min if col_max != col_min else 1

        val_a = (row_a[metric] - col_min) / range_val
        val_b = (row_b[metric] - col_min) / range_val

        values_a.append(val_a)
        values_b.append(val_b)

    # Close the radar
    available_metrics_closed = available_metrics + [available_metrics[0]]
    values_a_closed = values_a + [values_a[0]]
    values_b_closed = values_b + [values_b[0]]

    fig = go.Figure()

    fig.add_trace(
        go.Scatterpolar(
            r=values_a_closed,
            theta=available_metrics_closed,
            fill="toself",
            name=project_a,
            line_color="#1E88E5",
        )
    )

    fig.add_trace(
        go.Scatterpolar(
            r=values_b_closed,
            theta=available_metrics_closed,
            fill="toself",
            name=project_b,
            line_color="#5E35B1",
        )
    )

    fig.update_layout(
        polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
        showlegend=True,
        height=500,
    )

    st.plotly_chart(fig, use_container_width=True)

else:
    st.info("Not enough metrics available for radar chart comparison.")

# Text comparison
if "status_comments" in df.columns:
    st.markdown("---")
    st.markdown("### Status Comments Comparison")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown(f"**{project_a}**")
        st.text_area("", row_a["status_comments"], height=200, disabled=True, key="comments_a")

    with col2:
        st.markdown(f"**{project_b}**")
        st.text_area("", row_b["status_comments"], height=200, disabled=True, key="comments_b")
