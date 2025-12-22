"""
Feature Engineering Module

Creates derived features from raw project data for ML models.
"""

from datetime import datetime
from typing import Optional

import numpy as np
import pandas as pd
from loguru import logger


class FeatureEngineer:
    """Engineer features from raw project data."""

    def __init__(self):
        """Initialize the feature engineer."""
        self.feature_names: list[str] = []

    def create_features(
        self,
        df: pd.DataFrame,
        reference_date: Optional[datetime] = None,
    ) -> pd.DataFrame:
        """
        Create all derived features.

        Args:
            df: Input DataFrame with raw project data
            reference_date: Date to use for calculations (defaults to today)

        Returns:
            DataFrame with additional derived features
        """
        df = df.copy()

        if reference_date is None:
            reference_date = datetime.now()

        # Performance metrics
        df = self._create_performance_metrics(df)

        # Temporal features
        df = self._create_temporal_features(df, reference_date)

        # Variance features
        df = self._create_variance_features(df)

        # Risk indicators
        df = self._create_risk_indicators(df)

        logger.info(f"Created {len(self.feature_names)} derived features")

        return df

    def _create_performance_metrics(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create performance index features."""
        # Schedule Performance Index (SPI)
        # SPI > 1 means ahead of schedule
        if "planned_hours" in df.columns and "actual_hours" in df.columns:
            df["schedule_performance_index"] = np.where(
                df["actual_hours"] > 0,
                df["planned_hours"] / df["actual_hours"],
                1.0,
            )
            df["schedule_performance_index"] = df["schedule_performance_index"].clip(0, 3)
            self.feature_names.append("schedule_performance_index")

        # Cost Performance Index (CPI)
        # CPI > 1 means under budget
        if "budget" in df.columns and "spent" in df.columns:
            df["cost_performance_index"] = np.where(
                df["spent"] > 0,
                df["budget"] / df["spent"],
                1.0,
            )
            df["cost_performance_index"] = df["cost_performance_index"].clip(0, 3)
            self.feature_names.append("cost_performance_index")

        # Team productivity (completion rate per team member)
        if "completion_rate" in df.columns and "team_size" in df.columns:
            df["productivity_per_person"] = np.where(
                df["team_size"] > 0,
                df["completion_rate"] / df["team_size"],
                0,
            )
            self.feature_names.append("productivity_per_person")

        return df

    def _create_temporal_features(
        self,
        df: pd.DataFrame,
        reference_date: datetime,
    ) -> pd.DataFrame:
        """Create time-based features."""
        # Convert dates
        if "start_date" in df.columns:
            df["start_date"] = pd.to_datetime(df["start_date"], errors="coerce")

        if "planned_end_date" in df.columns:
            df["planned_end_date"] = pd.to_datetime(df["planned_end_date"], errors="coerce")

        # Project duration (planned)
        if "start_date" in df.columns and "planned_end_date" in df.columns:
            df["planned_duration_days"] = (
                df["planned_end_date"] - df["start_date"]
            ).dt.days
            df["planned_duration_days"] = df["planned_duration_days"].clip(lower=1)
            self.feature_names.append("planned_duration_days")

        # Days since start
        if "start_date" in df.columns:
            df["days_since_start"] = (
                pd.Timestamp(reference_date) - df["start_date"]
            ).dt.days
            df["days_since_start"] = df["days_since_start"].clip(lower=0)
            self.feature_names.append("days_since_start")

        # Days remaining
        if "planned_end_date" in df.columns:
            df["days_remaining"] = (
                df["planned_end_date"] - pd.Timestamp(reference_date)
            ).dt.days
            self.feature_names.append("days_remaining")

        # Percent of time elapsed
        if "planned_duration_days" in df.columns and "days_since_start" in df.columns:
            df["time_elapsed_pct"] = np.where(
                df["planned_duration_days"] > 0,
                (df["days_since_start"] / df["planned_duration_days"]) * 100,
                0,
            )
            df["time_elapsed_pct"] = df["time_elapsed_pct"].clip(0, 200)
            self.feature_names.append("time_elapsed_pct")

        return df

    def _create_variance_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create variance and deviation features."""
        # Budget variance (%)
        if "budget" in df.columns and "spent" in df.columns:
            df["budget_variance_pct"] = np.where(
                df["budget"] > 0,
                ((df["spent"] - df["budget"]) / df["budget"]) * 100,
                0,
            )
            self.feature_names.append("budget_variance_pct")

        # Hours variance (%)
        if "planned_hours" in df.columns and "actual_hours" in df.columns:
            df["hours_variance_pct"] = np.where(
                df["planned_hours"] > 0,
                ((df["actual_hours"] - df["planned_hours"]) / df["planned_hours"]) * 100,
                0,
            )
            self.feature_names.append("hours_variance_pct")

        # Schedule vs completion gap
        # If time_elapsed_pct > completion_rate, project is behind
        if "time_elapsed_pct" in df.columns and "completion_rate" in df.columns:
            df["schedule_gap"] = df["time_elapsed_pct"] - df["completion_rate"]
            self.feature_names.append("schedule_gap")

        # Budget utilization rate
        if "budget" in df.columns and "spent" in df.columns:
            df["budget_utilization"] = np.where(
                df["budget"] > 0,
                (df["spent"] / df["budget"]) * 100,
                0,
            )
            self.feature_names.append("budget_utilization")

        return df

    def _create_risk_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create risk indicator features."""
        # Team stability (inverse of turnover)
        if "team_turnover" in df.columns:
            df["team_stability"] = 1 - df["team_turnover"].clip(0, 1)
            self.feature_names.append("team_stability")

        # Complexity-adjusted progress
        if "completion_rate" in df.columns and "complexity_score" in df.columns:
            df["complexity_adjusted_progress"] = np.where(
                df["complexity_score"] > 0,
                df["completion_rate"] / df["complexity_score"],
                df["completion_rate"],
            )
            self.feature_names.append("complexity_adjusted_progress")

        # Burn rate (spending per day)
        if "spent" in df.columns and "days_since_start" in df.columns:
            df["burn_rate"] = np.where(
                df["days_since_start"] > 0,
                df["spent"] / df["days_since_start"],
                0,
            )
            self.feature_names.append("burn_rate")

        # Is over budget flag
        if "budget_variance_pct" in df.columns:
            df["is_over_budget"] = (df["budget_variance_pct"] > 0).astype(int)
            self.feature_names.append("is_over_budget")

        # Is behind schedule flag
        if "schedule_gap" in df.columns:
            df["is_behind_schedule"] = (df["schedule_gap"] > 10).astype(int)
            self.feature_names.append("is_behind_schedule")

        return df

    def get_feature_names(self) -> list[str]:
        """Return list of created feature names."""
        return self.feature_names.copy()
