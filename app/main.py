"""
PRISM - Predictive Risk Intelligence for Software Management

Main Streamlit application entry point.
"""

import sys
from pathlib import Path

# Streamlit puts the script directory on sys.path first; ensure repo root is present
# so ``import app`` resolves before any ``from app.bootstrap`` (see app/bootstrap.py).
_p = Path(__file__).resolve().parent
_repo_root = _p.parent.parent if _p.name == "pages" else _p.parent
if str(_repo_root) not in sys.path:
    sys.path.insert(0, str(_repo_root))

import streamlit as st

from app.bootstrap import init_page
from app.streamlit_theme import (
    ACCENT_BLUE,
    RISK_COLOR_HIGH,
    RISK_COLOR_LOW,
    RISK_COLOR_MEDIUM,
)

# Wide layout + sidebar styles (same as every multipage script; must be first st.* calls)
init_page()

# Custom CSS (colors from app.streamlit_theme for consistency with charts)
st.markdown(
    f"""
<style>
    .main-header {{
        display: flex;
        align-items: center;
        gap: 0.65rem;
        margin-bottom: 0.5rem;
        line-height: 1.1;
    }}
    .prism-logo {{
        font-size: 4.25rem;
        line-height: 1;
    }}
    .main-header-text {{
        font-size: 2.75rem;
        font-weight: bold;
        color: {ACCENT_BLUE};
    }}
    .sub-header {{
        font-size: 1.2rem;
        color: #666;
        margin-bottom: 2rem;
    }}
    .metric-card {{
        background-color: #f8f9fa;
        border-radius: 10px;
        padding: 1rem;
        text-align: center;
    }}
    .risk-high {{ color: {RISK_COLOR_HIGH}; }}
    .risk-medium {{ color: {RISK_COLOR_MEDIUM}; }}
    .risk-low {{ color: {RISK_COLOR_LOW}; }}
</style>
""",
    unsafe_allow_html=True,
)


def _load_demo_data():
    """Load processed JIRA CSV if present; otherwise generate synthetic PRISM-format data."""
    from src.data.generator import SyntheticDataGenerator
    from src.data.loader import DataLoader

    loader = DataLoader()
    try:
        return loader.load_jira_data(), "jira"
    except FileNotFoundError:
        gen = SyntheticDataGenerator(random_seed=42)
        df = gen.generate(n_projects=80, include_text=True)
        return df, "synthetic"


def main():
    """Main application function."""
    # Sidebar
    with st.sidebar:
        st.markdown("### 🔮 PRISM")
        st.markdown("---")
        st.markdown("### Navigation")
        st.markdown(
            """
        Use the pages in the sidebar to:
        - 📊 View Dashboard
        - 📁 Upload Data
        - 🤖 ML Analysis
        - 💬 LLM Insights
        - 📈 Rankings
        - 🔍 Compare Projects
        - 💭 Chat Assistant
        """
        )
        st.markdown("---")
        st.markdown("### Quick Stats")

        # Show stats if data is loaded
        if "projects_df" in st.session_state:
            df = st.session_state["projects_df"]
            st.metric("Total Projects", len(df))
        else:
            st.info("No data loaded yet")

    # Main content
    st.markdown(
        '<p class="main-header"><span class="prism-logo" aria-hidden="true">🔮</span>'
        '<span class="main-header-text">PRISM</span></p>',
        unsafe_allow_html=True,
    )
    st.markdown(
        '<p class="sub-header">Predictive Risk Intelligence for Software Management</p>',
        unsafe_allow_html=True,
    )

    # Welcome section
    col1, col2 = st.columns([2, 1])

    with col1:
        st.markdown(
            """
        ### Welcome to PRISM

        PRISM is an AI-powered system that helps project managers identify and prioritize
        software project risks before they become critical issues.

        **Key Features:**
        - 🤖 **Machine Learning** - Predicts risk from project metrics
        - 💬 **LLM Analysis** - Understands concerns in project comments
        - 📈 **MCDA Ranking** - Prioritizes projects objectively
        - 💭 **Chat Assistant** - Answers questions about your portfolio

        **Get Started:**
        1. Go to **📁 Upload Data** to import your project data
        2. View predictions on **🤖 ML Analysis** and **💬 LLM Insights**
        3. See prioritized rankings on **📈 Rankings**
        4. Ask questions using **💭 Chat Assistant**
        """
        )

    with col2:
        st.markdown(
            """
        ### Quick Actions

        """
        )
        if st.button("📁 Upload Data", width="stretch"):
            st.switch_page("pages/2_📁_Upload_Data.py")

        if st.button("📊 View Dashboard", width="stretch"):
            st.switch_page("pages/1_📊_Dashboard.py")

        if st.button("💭 Ask PRISM", width="stretch"):
            st.switch_page("pages/7_💭_Chat_Assistant.py")

    # Sample data section
    st.markdown("---")
    st.markdown("### Try with demo data")
    st.caption(
        "Loads `data/processed/jira_projects.csv` when available (notebook pipeline); "
        "otherwise generates synthetic PRISM-format projects."
    )

    if st.button("Load demo data", type="primary"):
        try:
            df, source = _load_demo_data()
            st.session_state["projects_df"] = df
            if source == "jira":
                st.success(f"✅ Loaded {len(df)} projects from processed JIRA data.")
            else:
                st.success(
                    f"✅ Generated {len(df)} synthetic demo projects (no "
                    "`jira_projects.csv` found — run preprocessing or place the file under "
                    "`data/processed/`)."
                )
            st.rerun()
        except Exception as e:
            st.error(f"Error loading demo data: {e}")

    # Footer
    st.markdown("---")
    st.markdown(
        """
        <div style='text-align: center; color: #888; font-size: 0.9rem;'>
        PRISM v1.0 | Predictive Risk Intelligence for Software Management | Built with Streamlit
        </div>
        """,
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()
