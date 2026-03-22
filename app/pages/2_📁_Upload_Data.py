"""
Upload Data Page

Import project data from CSV or JSON files using DataLoader and DataValidator.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

_p = Path(__file__).resolve().parent
_repo_root = _p.parent.parent if _p.name == "pages" else _p.parent
if str(_repo_root) not in sys.path:
    sys.path.insert(0, str(_repo_root))

import pandas as pd
import streamlit as st
import yaml

from app.bootstrap import init_page

root = init_page()
config_path = root / "config" / "mcda_config.yaml"

from src.data.generator import SyntheticDataGenerator
from src.data.loader import DataLoader
from src.data.validator import DataValidator

# Wide layout: ``init_page()`` calls ``st.set_page_config(layout="wide")`` (see app/bootstrap.py).

st.title("📁 Upload Data")
st.markdown("Import your project data for risk analysis")

# File upload section
st.markdown("### Upload Project Data")

uploaded_file = st.file_uploader(
    "Choose a CSV, JSON, or Excel file",
    type=["csv", "json", "xlsx", "xls"],
    help="Upload project data. JIRA-processed CSV is auto-detected and normalized.",
)

col1, col2, col3 = st.columns(3)

with col1:
    template = """project_id,project_name,start_date,planned_end_date,budget,spent,planned_hours,actual_hours,team_size,completion_rate,status,priority,status_comments
PROJ-001,Example Project,2024-01-01,2024-06-30,100000,50000,1000,600,5,55.0,Active,High,"On track with minor issues. Team working well together."
"""
    st.download_button(
        label="📥 Download CSV Template",
        data=template,
        file_name="prism_template.csv",
        mime="text/csv",
    )

with col2:
    if st.button("📊 Load processed JIRA data"):
        loader = DataLoader()
        try:
            df = loader.load_jira_data()
            st.session_state["projects_df"] = df
            st.success(f"✅ Loaded {len(df)} projects from data/processed/jira_projects.csv")
            st.rerun()
        except FileNotFoundError as e:
            st.error(str(e))

with col3:
    if st.button("🧪 Generate synthetic demo"):
        gen = SyntheticDataGenerator(random_seed=42)
        df = gen.generate(n_projects=80, include_text=True)
        st.session_state["projects_df"] = df
        st.success(f"✅ Generated {len(df)} synthetic projects for demo.")
        st.rerun()

# Process uploaded file
if uploaded_file is not None:
    try:
        loader = DataLoader()
        df = loader.load_from_bytes(uploaded_file.read(), uploaded_file.name)

        st.success(
            f"✅ Parsed {len(df)} rows from {uploaded_file.name} "
            f"(detected source: **{getattr(loader, 'data_source', 'unknown')}**)"
        )

        st.markdown("### Data Preview")
        st.dataframe(df.head(10), width="stretch")

        st.markdown("### Data Validation")
        validator = DataValidator()
        vresult = validator.validate(df)
        if vresult.is_valid:
            st.success("✅ Validation passed")
        else:
            st.warning(f"Validation completed with {vresult.error_count} error(s). Review below.")
        if vresult.errors:
            st.dataframe(pd.DataFrame(vresult.errors), width="stretch", hide_index=True)
        if vresult.warnings:
            st.info(f"{vresult.warning_count} warning(s)")
            st.dataframe(pd.DataFrame(vresult.warnings), width="stretch", hide_index=True)

        col_a, col_b, col_c = st.columns(3)
        with col_a:
            completeness = (1 - df.isnull().sum().sum() / df.size) * 100 if df.size else 0
            st.metric("Data Completeness", f"{completeness:.1f}%")
        with col_b:
            duplicate_ids = df["project_id"].duplicated().sum() if "project_id" in df.columns else 0
            st.metric("Duplicate IDs", int(duplicate_ids))
        with col_c:
            if "status_comments" in df.columns:
                avg_comment_len = df["status_comments"].astype(str).str.len().mean()
                st.metric("Avg Comment Length", f"{avg_comment_len:.0f} chars")
            else:
                st.metric("Avg Comment Length", "N/A")

        st.markdown("---")
        if st.button("✅ Use This Data", type="primary"):
            st.session_state["projects_df"] = df
            st.success("Data loaded successfully! Navigate to Dashboard to explore.")
            st.balloons()

    except Exception as e:
        st.error(f"Error loading file: {e}")

if config_path.exists():
    with st.expander("MCDA weight profiles (reference)"):
        with open(config_path, encoding="utf-8") as f:
            cfg = yaml.safe_load(f)
        profiles = cfg.get("weight_profiles", {})
        st.json(profiles)

# Show current data status
st.markdown("---")
st.markdown("### Current Data Status")

if "projects_df" in st.session_state:
    df = st.session_state["projects_df"]
    st.success(f"✅ {len(df)} projects loaded and ready for analysis")

    if st.button("🗑️ Clear Current Data"):
        del st.session_state["projects_df"]
        st.rerun()
else:
    st.info("No data currently loaded. Upload a file or use the buttons above.")
