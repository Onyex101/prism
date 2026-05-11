"""
ML Analysis Page

Machine learning risk predictions, feature importance, and optional SHAP explanations.
Aligned with notebooks 03–07 (reopen_rate cap, LightGBM fallback, feature_names.json).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

_p = Path(__file__).resolve().parent
_repo_root = _p.parent.parent if _p.name == "pages" else _p.parent
if str(_repo_root) not in sys.path:
    sys.path.insert(0, str(_repo_root))

from app.bootstrap import init_page
from app.components import require_projects_df

root = init_page()

from src.data import FeatureEngineer
from src.models.ml import MLPredictor, MLTrainer
from src.models.ml.predictor import positive_class_probability

st.title("🤖 ML Analysis")
st.markdown("Machine learning risk predictions, feature importance, and SHAP explanations")

df = require_projects_df().copy()
model_path = root / "models" / "ml" / "best_model.pkl"
feature_json_path = root / "models" / "ml" / "feature_names.json"


def _load_feature_list() -> list[str] | None:
    if not feature_json_path.exists():
        return None
    with open(feature_json_path, encoding="utf-8") as f:
        return json.load(f)


def _prepare_features(df_in: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """Feature engineering + notebook-aligned preprocessing."""
    fe = FeatureEngineer()
    df_fe = fe.create_features(df_in)

    if "reopen_rate" in df_fe.columns:
        df_fe["reopen_rate"] = df_fe["reopen_rate"].clip(upper=1.0)

    feature_list = _load_feature_list()
    exclude_cols = [
        "project_id",
        "project_name",
        "risk_level",
        "status_comments",
        "project_description",
        "team_feedback",
        "start_date",
        "planned_end_date",
        "actual_end_date",
        "technology_stack",
        "stakeholder_notes",
    ]

    if feature_list:
        for col in feature_list:
            if col not in df_fe.columns:
                df_fe[col] = 0
        X = df_fe[feature_list].copy()
    else:
        feature_cols = [
            col
            for col in df_fe.columns
            if col not in exclude_cols
            and df_fe[col].dtype in ["int64", "float64", "int32", "float32", "bool"]
        ]
        X = df_fe[feature_cols].copy()

    X = X.fillna(0).replace([np.inf, -np.inf], 0)
    return df_fe, list(X.columns)


st.markdown("### Run ML Analysis")

col_run, col_hist = st.columns([1, 2])

with col_run:
    run_clicked = st.button("🚀 Run Analysis", type="primary")

if run_clicked:
    with st.spinner("Running ML analysis..."):
        try:
            df_fe, feature_cols = _prepare_features(df)

            predictor = None
            importance_df = None
            trained_model = None

            if model_path.exists():
                predictor = MLPredictor(model_path=model_path)
                X = df_fe[feature_cols].copy()
                if predictor.feature_names:
                    for col in predictor.feature_names:
                        if col not in X.columns:
                            X[col] = 0
                    X = X[predictor.feature_names]
                scores_df = predictor.get_risk_scores(X)
                importance_df = predictor.get_feature_importance()
                trained_model = predictor.model
                st.session_state["ml_feature_matrix"] = X.copy()
                st.session_state["ml_trained_model"] = trained_model
                st.session_state["ml_feature_names"] = list(X.columns)
            else:
                feature_list = _load_feature_list()
                X = df_fe[feature_cols].copy()
                y = (
                    (df_fe["risk_level"] == "High").astype(int)
                    if "risk_level" in df_fe.columns
                    else (
                        (df_fe["completion_rate"] < 50).astype(int)
                        if "completion_rate" in df_fe.columns
                        else pd.Series([0] * len(df_fe))
                    )
                )

                trainer = MLTrainer(model_type="lightgbm")
                try:
                    trainer.train(X, y)
                except Exception:
                    trainer = MLTrainer(model_type="random_forest")
                    trainer.train(X, y)

                proba_mat = trainer.model.predict_proba(X)
                probas = positive_class_probability(trainer.model, proba_mat)
                risk_levels = [
                    "High" if p >= 0.6 else "Medium" if p >= 0.3 else "Low" for p in probas
                ]
                scores_df = pd.DataFrame({"risk_score": probas, "risk_level": risk_levels})
                trained_model = trainer.model
                st.session_state["ml_feature_matrix"] = X.copy()
                st.session_state["ml_trained_model"] = trained_model
                st.session_state["ml_feature_names"] = list(X.columns)
                imp = getattr(trainer.model, "feature_importances_", None)
                if imp is not None:
                    names = feature_list if feature_list else feature_cols
                    importance_df = pd.DataFrame(
                        {"feature": names[: len(imp)], "importance": imp}
                    ).sort_values("importance", ascending=False)

            rs_arr = np.asarray(
                pd.to_numeric(scores_df["risk_score"], errors="coerce").values,
                dtype=float,
            )
            if rs_arr.shape[0] != len(df_fe):
                raise ValueError(
                    f"Score length ({rs_arr.shape[0]}) does not match projects ({len(df_fe)})."
                )
            df_fe["risk_score"] = rs_arr
            df_fe["risk_level"] = scores_df["risk_level"].values
            df_fe["ml_risk_score"] = df_fe["risk_score"]

            st.session_state["projects_df"] = df_fe
            st.session_state["ml_importance_df"] = importance_df

            try:
                from src.data.snapshot_store import SnapshotStore

                SnapshotStore().save_snapshot(df_fe, source="ml")
            except Exception:
                pass

            st.success("✅ ML analysis complete!")
        except Exception as e:
            st.error(f"Analysis failed: {e}")
            import traceback

            st.code(traceback.format_exc())


def _format_avg_risk(mean_val: float) -> str:
    """Format mean P(high); small probabilities need extra decimals (not 0.00)."""
    if pd.isna(mean_val):
        return "N/A"
    if mean_val < 1e-8:
        return f"{mean_val:.2e}"
    if mean_val < 0.01:
        return f"{mean_val:.5f}"
    return f"{mean_val:.3f}"


def _risk_score_column_to_series(disp: pd.DataFrame, col: str) -> pd.Series:
    """
    Return a single 1D Series for ``col`` (duplicate column names → first column).

    Plotly and ``pd.to_numeric`` misbehave when ``df[col]`` is a DataFrame.
    """
    obj = disp[col]
    if isinstance(obj, pd.DataFrame):
        obj = obj.iloc[:, 0]
    s = obj.squeeze()
    if isinstance(s, pd.DataFrame):
        s = s.iloc[:, 0]
    return s


def _finite_risk_scores_for_hist(disp: pd.DataFrame) -> np.ndarray:
    """Collect finite P(high) values from ``risk_score`` or ``ml_risk_score``."""
    for name in ("risk_score", "ml_risk_score"):
        if name not in disp.columns:
            continue
        raw = _risk_score_column_to_series(disp, name)
        num = pd.to_numeric(raw, errors="coerce")
        vals = np.asarray(num, dtype=float)
        vals = vals[np.isfinite(vals)]
        if vals.size:
            return vals
    return np.array([], dtype=float)


def _tick_format_for_probability_range(lo: float, hi: float) -> str:
    """Choose tick format from the visible x range (not only raw vmin/vmax)."""
    span = hi - lo
    m = max(abs(lo), abs(hi))
    if span <= 0 or m < 1e-4 or (m < 0.02 and span < 0.01):
        return ".2e"
    if hi <= 0.05 or span < 0.05:
        return ".4f"
    if hi <= 0.5:
        return ".3f"
    return ".2f"


def _xaxis_for_probability_bars(vmin: float, vmax: float) -> dict:
    """
    X-axis range and ticks so bars stay visible for **any** plausible score distribution.

    - Values outside ``[0, 1]``: pad around ``[vmin, vmax]`` (mis-scaled / non-probability).
    - Values inside ``[0, 1]``: **zoom** when the data span only a small slice of the unit
      interval (bars would be invisible on a forced full 0–1 axis). Otherwise use full
      ``[0, 1]`` for standard probability interpretation.
    """
    spread = vmax - vmin
    cfg: dict = {"title": "Risk score (P(high), 0–1)"}

    if vmin < -1e-9 or vmax > 1.0 + 1e-9:
        pad = max(spread * 0.15, 1e-15)
        lo, hi = vmin - pad, vmax + pad
        cfg["range"] = [lo, hi]
        cfg["tickformat"] = _tick_format_for_probability_range(lo, hi)
        return cfg

    # [0, 1] probabilities: zoom if cluster is narrow relative to the full unit scale
    # (covers tiny absolute values ~1e-5 and tight clusters e.g. 0.45–0.48).
    narrow_on_unit = spread < 0.12 or (
        spread > 0 and (spread / max(vmax, 1e-15)) < 0.12 and vmax < 0.85
    )
    if narrow_on_unit or spread == 0:
        if spread == 0:
            pad = max(abs(vmin) * 0.25, 1e-15)
        else:
            pad = max(spread * 0.25, vmax * 0.08, 1e-15)
        lo = max(0.0, vmin - pad)
        hi = min(1.0, vmax + pad)
        if hi <= lo:
            hi = lo + max(pad, 1e-12)
        cfg["range"] = [lo, hi]
        cfg["tickformat"] = _tick_format_for_probability_range(lo, hi)
        return cfg

    cfg["range"] = [0, 1]
    cfg["tickformat"] = ".2f"
    return cfg


def _risk_distribution_figure(vals: np.ndarray):
    """
    Histogram as explicit bin counts (numpy + Bar).

    Plotly ``go.Histogram`` can render an empty chart when ``xaxis.range`` is [0, 1]
    but binning places samples on bin edges (e.g. all zeros / tiny probabilities).
    """
    import plotly.graph_objects as go

    vals = np.asarray(vals, dtype=float)
    vals = vals[np.isfinite(vals)]
    n = int(vals.size)
    if n == 0:
        return go.Figure()

    vmin, vmax = float(np.min(vals)), float(np.max(vals))
    # Bin count heuristic for ~10–100 projects
    nuniq = max(1, int(np.unique(vals).size))
    nbins = max(5, min(24, max(nuniq * 2, int(np.sqrt(n)) * 2)))

    # Single-value portfolios: widen bin so Bar has non-zero width
    if vmin == vmax:
        half = max(0.02, abs(vmin) * 0.1 + 0.01)
        edges = np.array([vmin - half, vmin + half])
        counts, edges = np.histogram(vals, bins=edges)
    else:
        counts, edges = np.histogram(vals, bins=nbins)

    xc = (edges[:-1] + edges[1:]) / 2.0
    widths = np.diff(edges)
    # Avoid zero width in pathological edge cases
    widths = np.maximum(widths, np.finfo(float).eps * 10)

    fig = go.Figure(
        data=[
            go.Bar(
                x=xc,
                y=counts,
                width=widths * 0.95,
                marker_color="#1E88E5",
            )
        ]
    )
    xaxis_cfg = _xaxis_for_probability_bars(vmin, vmax)
    fig.update_layout(
        title="Risk score distribution",
        xaxis=xaxis_cfg,
        yaxis_title="Count",
        height=300,
        bargap=0.02,
    )
    return fig


with col_hist:
    disp_check = st.session_state.get("projects_df", df)
    if "risk_score" in disp_check.columns or "ml_risk_score" in disp_check.columns:
        vals = _finite_risk_scores_for_hist(disp_check)
        if vals.size == 0:
            st.info(
                "No numeric **risk scores** to plot yet. Click **Run Analysis** so the model "
                "writes P(high) scores for each project. If you already ran analysis, check "
                "that `risk_score` / `ml_risk_score` are numbers (duplicate or bad columns can "
                "hide values)."
            )
        else:
            fig = _risk_distribution_figure(vals)
            st.plotly_chart(fig, width="stretch")
            vmax = float(np.max(vals))
            if vmax < 0.05:
                st.caption(
                    "The x-axis is **zoomed** to the score range so bars stay visible. "
                    "P(high) can be very small (e.g. ~10⁻⁵) while still being a valid 0–1 probability."
                )
            elif len(vals) <= 10:
                st.caption(
                    "Few projects: scores are still valid probabilities; averages can look small "
                    "because P(high) is often below 0.3 for low-risk portfolios."
                )

disp_df = st.session_state.get("projects_df", df)
if "risk_score" in disp_df.columns:
    st.markdown("---")
    st.markdown("### Prediction Results")

    rs_all = pd.to_numeric(disp_df["risk_score"], errors="coerce")
    mean_p = float(rs_all.mean())
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.metric("Average P(high)", _format_avg_risk(mean_p))
        if not pd.isna(mean_p) and mean_p < 0.05:
            st.caption(f"≈ **{mean_p * 100:.3f}%** if expressed as percent (scores stay in 0–1).")
    with c2:
        st.metric("High Risk Projects", int((disp_df["risk_level"] == "High").sum()))
    with c3:
        st.metric("Medium Risk Projects", int((disp_df["risk_level"] == "Medium").sum()))
    with c4:
        st.metric("Low Risk Projects", int((disp_df["risk_level"] == "Low").sum()))

    st.markdown("### Project Risk Scores")
    st.caption(
        "P(high) is the model’s estimated probability of the **high-risk** class (0–1). "
        "Uploaded/demo rows often look like “all Low” with **tiny** probabilities (e.g. 0.002); "
        "that is normal — Progress bars hid those values before."
    )
    base_cols = ["project_name", "risk_score", "risk_level"]
    display_cols = list(base_cols)
    if "completion_rate" in disp_df.columns:
        display_cols.append("completion_rate")
    if "status" in disp_df.columns:
        display_cols.append("status")

    results_df = disp_df[[c for c in display_cols if c in disp_df.columns]].copy()
    rs_num = pd.to_numeric(results_df["risk_score"], errors="coerce")
    results_df.insert(
        results_df.columns.get_loc("risk_score") + 1,
        "p_high_%",
        (rs_num * 100).round(4),
    )
    results_df = results_df.sort_values("risk_score", ascending=False)
    st.dataframe(
        results_df,
        width="stretch",
        hide_index=True,
        column_config={
            "risk_score": st.column_config.NumberColumn(
                "P(high)",
                format="%.5f",
                min_value=0.0,
                max_value=1.0,
                help="Probability of high-risk class (0–1). Use p_high_% for a percent scale.",
            ),
            "p_high_%": st.column_config.NumberColumn(
                "p_high_%",
                format="%.4f",
                help="P(high) × 100 (easier to read when probabilities are small).",
            ),
        },
    )

    st.markdown("---")
    st.markdown("### Feature Importance")
    importance_df = st.session_state.get("ml_importance_df")
    if importance_df is not None and len(importance_df) > 0:
        try:
            from src.visualization.risk_charts import RiskCharts

            fig = RiskCharts.feature_importance_bar(importance_df, top_n=10)
            st.plotly_chart(fig, width="stretch")
        except Exception:
            import plotly.graph_objects as go

            plot_df = importance_df.head(10)
            fig = go.Figure(
                data=[
                    go.Bar(
                        y=plot_df["feature"],
                        x=plot_df["importance"],
                        orientation="h",
                        marker_color="#1E88E5",
                    )
                ]
            )
            fig.update_layout(title="Feature Importance", height=300)
            st.plotly_chart(fig, width="stretch")
    else:
        st.info("Train a model (see `make train` / notebook 03) or run analysis to see importance.")

    st.markdown("---")
    st.markdown("### SHAP explanation (optional)")
    st.caption("Local explanations for a single project — requires the `shap` package.")

    model = st.session_state.get("ml_trained_model")
    X_stored = st.session_state.get("ml_feature_matrix")
    feat_names = st.session_state.get("ml_feature_names")

    if model is not None and X_stored is not None and feat_names:
        name_col = "project_name" if "project_name" in disp_df.columns else "project_id"
        options = disp_df[name_col].tolist()
        pick = st.selectbox("Select project", options)
        if st.button("Explain prediction with SHAP"):
            try:
                from src.explainability.shap_explainer import SHAPExplainer

                pos = disp_df[name_col].tolist().index(pick)

                explainer = SHAPExplainer(model, feature_names=feat_names)
                explainer.fit(X_stored)
                text = explainer.get_explanation_text(X_stored, index=pos, top_n=5)
                st.text(text)
                detail = explainer.explain_instance(X_stored, index=pos)
                st.json(detail["top_positive"][:5])
                st.json(detail["top_negative"][:5])
            except ImportError:
                st.warning("Install SHAP: pip install shap")
            except Exception as e:
                st.error(str(e))
    else:
        st.info("Run **Run Analysis** first to enable SHAP (model + feature matrix required).")

else:
    st.info("👆 Click **Run Analysis** to generate ML predictions for your projects.")
