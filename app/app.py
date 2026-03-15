"""
PRISM - Predictive Risk Intelligence for Software Management

Main Streamlit application entry point.
"""

import streamlit as st

# Page configuration - must be first Streamlit command
st.set_page_config(
    page_title="PRISM - Project Risk Intelligence",
    page_icon="🔮",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS
st.markdown(
    """
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1E88E5;
        margin-bottom: 0.5rem;
    }
    .sub-header {
        font-size: 1.2rem;
        color: #666;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f8f9fa;
        border-radius: 10px;
        padding: 1rem;
        text-align: center;
    }
    .risk-high { color: #FF4B4B; }
    .risk-medium { color: #FFA500; }
    .risk-low { color: #00CC66; }
</style>
""",
    unsafe_allow_html=True,
)


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
    st.markdown('<p class="main-header">🔮 PRISM</p>', unsafe_allow_html=True)
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
        if st.button("📁 Upload Data", use_container_width=True):
            st.switch_page("pages/2_📁_Upload_Data.py")

        if st.button("📊 View Dashboard", use_container_width=True):
            st.switch_page("pages/1_📊_Dashboard.py")

        if st.button("💭 Ask PRISM", use_container_width=True):
            st.switch_page("pages/7_💭_Chat_Assistant.py")

    # Sample data section
    st.markdown("---")
    st.markdown("### Try with Sample Data")

    if st.button("Load Sample Data", type="primary"):
        try:
            import pandas as pd
            from pathlib import Path

            sample_path = Path(__file__).parent.parent / "data" / "raw" / "sample_projects.csv"
            if sample_path.exists():
                df = pd.read_csv(sample_path)
                st.session_state["projects_df"] = df
                st.success(f"✅ Loaded {len(df)} sample projects!")
                st.rerun()
            else:
                st.error("Sample data file not found.")
        except Exception as e:
            st.error(f"Error loading sample data: {e}")

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
