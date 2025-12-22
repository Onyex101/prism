"""
Risk Charts Module

Generates visualizations for risk analysis results.
"""

from typing import Optional

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots


class RiskCharts:
    """Generate charts for risk visualization."""

    # Color scheme
    COLORS = {
        "high": "#FF4B4B",
        "medium": "#FFA500",
        "low": "#00CC66",
        "primary": "#1E88E5",
        "secondary": "#5E35B1",
    }

    @staticmethod
    def risk_distribution_pie(df: pd.DataFrame) -> go.Figure:
        """
        Create pie chart showing risk level distribution.

        Args:
            df: DataFrame with 'risk_level' column

        Returns:
            Plotly figure
        """
        counts = df["risk_level"].value_counts()

        colors = [RiskCharts.COLORS.get(level.lower(), "#808080") for level in counts.index]

        fig = go.Figure(
            data=[
                go.Pie(
                    labels=counts.index,
                    values=counts.values,
                    hole=0.4,
                    marker_colors=colors,
                    textinfo="label+percent",
                )
            ]
        )

        fig.update_layout(
            title="Risk Level Distribution",
            showlegend=True,
            height=400,
        )

        return fig

    @staticmethod
    def risk_score_bar(
        df: pd.DataFrame,
        top_n: int = 10,
        name_col: str = "project_name",
        score_col: str = "mcda_score",
    ) -> go.Figure:
        """
        Create horizontal bar chart of risk scores.

        Args:
            df: DataFrame with project data
            top_n: Number of projects to show
            name_col: Column for project names
            score_col: Column for scores

        Returns:
            Plotly figure
        """
        # Sort by score (ascending for risk - lower is riskier)
        plot_df = df.nsmallest(top_n, score_col)

        colors = [
            (
                RiskCharts.COLORS["high"]
                if s < 0.4
                else RiskCharts.COLORS["medium"] if s < 0.7 else RiskCharts.COLORS["low"]
            )
            for s in plot_df[score_col]
        ]

        fig = go.Figure(
            data=[
                go.Bar(
                    y=plot_df[name_col],
                    x=plot_df[score_col],
                    orientation="h",
                    marker_color=colors,
                    text=plot_df[score_col].round(2),
                    textposition="outside",
                )
            ]
        )

        fig.update_layout(
            title=f"Top {top_n} High-Risk Projects",
            xaxis_title="MCDA Score (lower = higher risk)",
            yaxis_title="Project",
            height=max(400, top_n * 40),
            yaxis={"categoryorder": "total ascending"},
        )

        return fig

    @staticmethod
    def risk_gauge(score: float, title: str = "Risk Score") -> go.Figure:
        """
        Create a gauge chart for risk score.

        Args:
            score: Risk score (0-1)
            title: Chart title

        Returns:
            Plotly figure
        """
        # Invert for gauge (higher = riskier)
        risk_value = 1 - score

        if risk_value >= 0.6:
            color = RiskCharts.COLORS["high"]
        elif risk_value >= 0.3:
            color = RiskCharts.COLORS["medium"]
        else:
            color = RiskCharts.COLORS["low"]

        fig = go.Figure(
            go.Indicator(
                mode="gauge+number",
                value=risk_value * 100,
                title={"text": title},
                number={"suffix": "%"},
                gauge={
                    "axis": {"range": [0, 100]},
                    "bar": {"color": color},
                    "steps": [
                        {"range": [0, 30], "color": "#E8F5E9"},
                        {"range": [30, 60], "color": "#FFF3E0"},
                        {"range": [60, 100], "color": "#FFEBEE"},
                    ],
                    "threshold": {
                        "line": {"color": "black", "width": 4},
                        "thickness": 0.75,
                        "value": risk_value * 100,
                    },
                },
            )
        )

        fig.update_layout(height=300)

        return fig

    @staticmethod
    def feature_importance_bar(
        importance_df: pd.DataFrame,
        top_n: int = 10,
    ) -> go.Figure:
        """
        Create bar chart of feature importance.

        Args:
            importance_df: DataFrame with 'feature' and 'importance' columns
            top_n: Number of features to show

        Returns:
            Plotly figure
        """
        plot_df = importance_df.head(top_n)

        fig = go.Figure(
            data=[
                go.Bar(
                    y=plot_df["feature"],
                    x=plot_df["importance"],
                    orientation="h",
                    marker_color=RiskCharts.COLORS["primary"],
                )
            ]
        )

        fig.update_layout(
            title="Feature Importance",
            xaxis_title="Importance Score",
            yaxis_title="Feature",
            height=max(400, top_n * 35),
            yaxis={"categoryorder": "total ascending"},
        )

        return fig

    @staticmethod
    def sentiment_distribution(df: pd.DataFrame) -> go.Figure:
        """
        Create histogram of sentiment scores.

        Args:
            df: DataFrame with 'sentiment_score' column

        Returns:
            Plotly figure
        """
        fig = px.histogram(
            df,
            x="sentiment_score",
            nbins=20,
            color_discrete_sequence=[RiskCharts.COLORS["primary"]],
        )

        fig.add_vline(x=0, line_dash="dash", line_color="gray")

        fig.update_layout(
            title="Sentiment Score Distribution",
            xaxis_title="Sentiment Score (-1 to 1)",
            yaxis_title="Count",
            height=400,
        )

        return fig

    @staticmethod
    def risk_category_radar(categories: dict) -> go.Figure:
        """
        Create radar chart of risk categories.

        Args:
            categories: Dict of category -> count

        Returns:
            Plotly figure
        """
        cats = list(categories.keys())
        values = list(categories.values())

        # Close the radar
        cats.append(cats[0])
        values.append(values[0])

        fig = go.Figure(
            data=go.Scatterpolar(
                r=values,
                theta=cats,
                fill="toself",
                fillcolor="rgba(30, 136, 229, 0.3)",
                line_color=RiskCharts.COLORS["primary"],
            )
        )

        fig.update_layout(
            polar=dict(radialaxis=dict(visible=True)),
            title="Risk Categories",
            height=400,
        )

        return fig

    @staticmethod
    def comparison_radar(
        projects: list[dict],
        metrics: list[str],
    ) -> go.Figure:
        """
        Create radar chart comparing multiple projects.

        Args:
            projects: List of project dicts with metrics
            metrics: List of metric names to compare

        Returns:
            Plotly figure
        """
        fig = go.Figure()

        colors = [
            RiskCharts.COLORS["primary"],
            RiskCharts.COLORS["secondary"],
            RiskCharts.COLORS["high"],
            RiskCharts.COLORS["medium"],
        ]

        for i, project in enumerate(projects[:4]):
            name = project.get("project_name", f"Project {i+1}")
            values = [project.get(m, 0) for m in metrics]
            values.append(values[0])  # Close the radar

            fig.add_trace(
                go.Scatterpolar(
                    r=values,
                    theta=metrics + [metrics[0]],
                    name=name,
                    line_color=colors[i % len(colors)],
                )
            )

        fig.update_layout(
            polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
            title="Project Comparison",
            height=500,
        )

        return fig
