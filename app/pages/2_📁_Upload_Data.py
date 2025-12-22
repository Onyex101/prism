"""
Upload Data Page

Import project data from CSV or JSON files.
"""

import streamlit as st
import pandas as pd
from pathlib import Path
import sys

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

st.set_page_config(page_title="Upload Data - PRISM", page_icon="📁", layout="wide")

st.title("📁 Upload Data")
st.markdown("Import your project data for risk analysis")

# File upload section
st.markdown("### Upload Project Data")

uploaded_file = st.file_uploader(
    "Choose a CSV or JSON file",
    type=["csv", "json"],
    help="Upload your project data. See the template for required format.",
)

col1, col2 = st.columns(2)

with col1:
    if st.button("📥 Download CSV Template"):
        template = """project_id,project_name,start_date,planned_end_date,budget,spent,planned_hours,actual_hours,team_size,completion_rate,status,priority,status_comments
PROJ-001,Example Project,2024-01-01,2024-06-30,100000,50000,1000,600,5,55.0,Active,High,"On track with minor issues. Team working well together."
"""
        st.download_button(
            label="Download",
            data=template,
            file_name="prism_template.csv",
            mime="text/csv",
        )

with col2:
    if st.button("📊 Load Sample Data"):
        sample_path = Path(__file__).parent.parent.parent / "data" / "raw" / "sample_projects.csv"
        if sample_path.exists():
            df = pd.read_csv(sample_path)
            st.session_state["projects_df"] = df
            st.success(f"✅ Loaded {len(df)} sample projects!")
            st.rerun()
        else:
            st.error("Sample data not found.")

# Process uploaded file
if uploaded_file is not None:
    try:
        # Load data
        if uploaded_file.name.endswith(".csv"):
            df = pd.read_csv(uploaded_file)
        else:
            import json

            data = json.load(uploaded_file)
            if isinstance(data, list):
                df = pd.DataFrame(data)
            elif isinstance(data, dict) and "projects" in data:
                df = pd.DataFrame(data["projects"])
            else:
                df = pd.DataFrame([data])

        st.success(f"✅ Loaded {len(df)} projects from {uploaded_file.name}")

        # Data preview
        st.markdown("### Data Preview")
        st.dataframe(df.head(10), use_container_width=True)

        # Validation
        st.markdown("### Data Validation")

        required_cols = [
            "project_id",
            "project_name",
            "budget",
            "spent",
            "planned_hours",
            "actual_hours",
            "team_size",
            "completion_rate",
            "status",
            "status_comments",
        ]

        missing_cols = [col for col in required_cols if col not in df.columns]

        if missing_cols:
            st.warning(f"⚠️ Missing recommended columns: {', '.join(missing_cols)}")
        else:
            st.success("✅ All required columns present")

        # Data quality stats
        col1, col2, col3 = st.columns(3)

        with col1:
            completeness = (1 - df.isnull().sum().sum() / df.size) * 100
            st.metric("Data Completeness", f"{completeness:.1f}%")

        with col2:
            duplicate_ids = df["project_id"].duplicated().sum() if "project_id" in df.columns else 0
            st.metric("Duplicate IDs", duplicate_ids)

        with col3:
            if "status_comments" in df.columns:
                avg_comment_len = df["status_comments"].str.len().mean()
                st.metric("Avg Comment Length", f"{avg_comment_len:.0f} chars")
            else:
                st.metric("Avg Comment Length", "N/A")

        # Save button
        st.markdown("---")
        if st.button("✅ Use This Data", type="primary"):
            st.session_state["projects_df"] = df
            st.success("Data saved! Redirecting to dashboard...")
            st.balloons()

    except Exception as e:
        st.error(f"Error loading file: {e}")

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
    st.info("No data currently loaded. Upload a file or load sample data above.")
