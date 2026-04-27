"""
Upload Data Page

Import project data from CSV, JSON, Excel, Jira Cloud (OAuth), or synthetic demo.
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
import yaml

from app.bootstrap import init_page
from app.jira_oauth_ui import build_jira_stack, query_param_first

root = init_page()
config_path = root / "config" / "mcda_config.yaml"

from config.settings import settings
from src.data.generator import SyntheticDataGenerator
from src.data.loader import DataLoader
from src.data.validator import DataValidator

st.title("📁 Upload Data")
st.markdown("Import your project data for risk analysis")

# OAuth callback (Upload Data page URL must match JIRA_OAUTH_REDIRECT_URI if using this page)
if settings.jira_oauth_configured():
    code = query_param_first(st.query_params, "code")
    state = query_param_first(st.query_params, "state")
    if code and state:
        flow_cb, _, _ = build_jira_stack(settings, st.session_state)
        try:
            flow_cb.exchange_code(code, state)
            st.session_state.pop("jira_oauth_url", None)
            st.query_params.clear()
            st.success("Logged in with Atlassian.")
            st.rerun()
        except Exception as exc:
            st.error(f"OAuth error: {exc}")

tab_file, tab_jira, tab_demo = st.tabs(["File upload", "Connect to Jira", "Synthetic / sample"])

with tab_file:
    st.markdown("### Upload Project Data")
    uploaded_file = st.file_uploader(
        "Choose a CSV, JSON, or Excel file",
        type=["csv", "json", "xlsx", "xls"],
        help="Upload project data. JIRA-processed CSV is auto-detected and normalized.",
        key="upload_file_widget",
    )

    col1, _ = st.columns(2)
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
                st.warning(
                    f"Validation completed with {vresult.error_count} error(s). Review below."
                )
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
                dup = df["project_id"].duplicated().sum() if "project_id" in df.columns else 0
                st.metric("Duplicate IDs", int(dup))
            with col_c:
                if "status_comments" in df.columns:
                    avg_len = df["status_comments"].astype(str).str.len().mean()
                    st.metric("Avg Comment Length", f"{avg_len:.0f} chars")
                else:
                    st.metric("Avg Comment Length", "N/A")

            st.markdown("---")
            if st.button("✅ Use This Data", type="primary", key="use_uploaded"):
                st.session_state["projects_df"] = df
                st.success("Data loaded successfully! Navigate to Dashboard to explore.")
                st.balloons()

        except Exception as e:
            st.error(f"Error loading file: {e}")

with tab_jira:
    st.markdown("### Jira Cloud (OAuth 2.0)")
    if not settings.jira_oauth_configured():
        st.warning(
            "Jira OAuth is not configured. Add **JIRA_OAUTH_CLIENT_ID**, "
            "**JIRA_OAUTH_CLIENT_SECRET**, and **JIRA_OAUTH_REDIRECT_URI** to your `.env` "
            "(see `env_template.txt`). Register an OAuth 2.0 (3LO) app at "
            "[Atlassian Developer Console](https://developer.atlassian.com/console/myapps)."
        )
    else:
        flow, store, sync = build_jira_stack(settings, st.session_state)
        token = store.load()

        if token is None:
            if st.button("🔗 Login with Atlassian", type="primary"):
                st.session_state["jira_oauth_url"] = flow.build_authorize_url()
                st.rerun()
            if "jira_oauth_url" in st.session_state:
                st.link_button(
                    "Open Atlassian to sign in",
                    st.session_state["jira_oauth_url"],
                    type="primary",
                )
            st.caption(
                "After you approve access, your browser returns to this app with a code. "
                "Ensure **JIRA_OAUTH_REDIRECT_URI** matches this page’s URL (or your app root) "
                "in the Atlassian developer console."
            )
        elif token.selected_cloud_id is None and len(token.available_sites) > 1:
            st.subheader("Choose a Jira site")
            labels = [f"{s.name} — {s.url}" for s in token.available_sites]
            choice = st.radio("Site", range(len(token.available_sites)), format_func=lambda i: labels[i])
            if st.button("Use this site"):
                sid = token.available_sites[int(choice)].id
                flow.select_site(sid)
                st.rerun()
            if st.button("Logout"):
                flow.logout()
                st.session_state.pop("jira_oauth_url", None)
                st.rerun()
        elif token.selected_cloud_id is None and len(token.available_sites) == 0:
            st.error("No Jira Cloud sites returned for this account. Log out and try again.")
            if st.button("Logout"):
                flow.logout()
                st.rerun()
        else:
            try:
                who = sync.test_connection()
                st.success(
                    f"Connected as **{who.get('display_name', 'User')}** "
                    f"({who.get('email', '')})"
                )
            except Exception as exc:
                st.warning(f"Could not verify session: {exc}")

            site_name = next(
                (s.name for s in token.available_sites if s.id == token.selected_cloud_id),
                token.selected_cloud_id or "",
            )
            st.caption(f"Site: **{site_name}**")

            if len(token.available_sites) > 1 and st.button("Switch site"):
                flow.clear_selected_site()
                st.rerun()

            if st.button("Logout from Atlassian"):
                flow.logout()
                st.session_state.pop("jira_oauth_url", None)
                st.rerun()

            projects = sync.list_projects()
            keys = [p.key for p in projects]
            labels_map = {p.key: f"{p.key} — {p.name}" for p in projects}
            selected = st.multiselect(
                "Projects to sync",
                options=keys,
                format_func=lambda k: labels_map.get(k, k),
            )
            cap = max(50, min(settings.JIRA_MAX_ISSUES_PER_PROJECT, 5000))
            max_issues = st.slider(
                "Max issues per project",
                min_value=50,
                max_value=cap,
                value=min(500, cap),
                step=50,
            )
            jql_extra = st.text_input(
                "Optional JQL filter (AND …)",
                value="",
                help="Example: created >= -180d",
            )
            persist = st.checkbox("Save CSV snapshot to data/processed/jira_from_jira_cloud.csv")

            if st.button("Sync from Jira", type="primary"):
                if not selected:
                    st.error("Select at least one project.")
                else:
                    progress = st.progress(0.0, text="Starting…")

                    def _prog(phase: str, cur: int, tot: int) -> None:
                        if tot <= 0:
                            return
                        progress.progress(min(cur / tot, 1.0), text=f"{phase}: {cur}/{tot}")

                    try:
                        save_path = None
                        if persist:
                            save_path = root / "data" / "processed" / "jira_from_jira_cloud.csv"
                        df = sync.sync(
                            selected,
                            max_issues_per_project=max_issues,
                            jql_extra=jql_extra,
                            progress=_prog,
                            save_path=save_path,
                        )
                        st.session_state["projects_df"] = df
                        progress.progress(1.0, text="Done")
                        st.success(f"Loaded {len(df)} project row(s) from Jira.")
                    except Exception as exc:
                        st.error(f"Sync failed: {exc}")

with tab_demo:
    st.markdown("### Demo data")
    c1, c2 = st.columns(2)
    with c1:
        if st.button("📊 Load processed JIRA data"):
            loader = DataLoader()
            try:
                df = loader.load_jira_data()
                st.session_state["projects_df"] = df
                st.success(f"✅ Loaded {len(df)} projects from data/processed/jira_projects.csv")
                st.rerun()
            except FileNotFoundError as e:
                st.error(str(e))
    with c2:
        if st.button("🧪 Generate synthetic demo"):
            gen = SyntheticDataGenerator(random_seed=42)
            df = gen.generate(n_projects=80, include_text=True)
            st.session_state["projects_df"] = df
            st.success(f"✅ Generated {len(df)} synthetic projects for demo.")
            st.rerun()

if config_path.exists():
    with st.expander("MCDA weight profiles (reference)"):
        with open(config_path, encoding="utf-8") as f:
            cfg = yaml.safe_load(f)
        profiles = cfg.get("weight_profiles", {})
        st.json(profiles)

st.markdown("---")
st.markdown("### Current Data Status")

if "projects_df" in st.session_state:
    df = st.session_state["projects_df"]
    st.success(f"✅ {len(df)} projects loaded and ready for analysis")

    if st.button("🗑️ Clear Current Data"):
        del st.session_state["projects_df"]
        st.rerun()
else:
    st.info("No data currently loaded. Use a tab above to load data.")
