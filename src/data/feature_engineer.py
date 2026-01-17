"""
Feature Engineering Module
==========================

This module creates derived features from raw project data for ML models.

It calculates performance metrics, temporal features, variance indicators,
and risk signals from base project attributes.

Example:
    >>> from src.data.feature_engineer import FeatureEngineer
    >>> engineer = FeatureEngineer()
    >>> df_with_features = engineer.create_features(df)
"""

from datetime import datetime
from typing import Optional

import numpy as np
import pandas as pd
from loguru import logger


class FeatureEngineer:
    """
    Engineer features from raw project data.

    This class follows the Single Responsibility Principle by focusing
    solely on feature creation. It creates interpretable features that
    align with project management domain knowledge.

    :ivar feature_names: List of created feature names.
    :vartype feature_names: list[str]

    Example:
        >>> engineer = FeatureEngineer()
        >>> df = engineer.create_features(projects_df)
        >>> print(engineer.get_feature_names())
    """

    def __init__(self) -> None:
        """Initialize the feature engineer."""
        self.feature_names: list[str] = []

    def create_features(
        self,
        df: pd.DataFrame,
        reference_date: Optional[datetime] = None,
    ) -> pd.DataFrame:
        """
        Create all derived features.

        This method orchestrates the creation of all feature types:
        performance metrics, temporal features, variance features,
        and risk indicators.

        :param df: Input DataFrame with raw project data.
        :type df: pd.DataFrame
        :param reference_date: Date to use for calculations.
            Defaults to current date.
        :type reference_date: Optional[datetime]
        :return: DataFrame with additional derived features.
        :rtype: pd.DataFrame

        Example:
            >>> engineer = FeatureEngineer()
            >>> df = engineer.create_features(raw_df)
        """
        df = df.copy()
        self.feature_names = []

        if reference_date is None:
            reference_date = datetime.now()

        df = self._create_performance_metrics(df)
        df = self._create_temporal_features(df, reference_date)
        df = self._create_variance_features(df)
        df = self._create_risk_indicators(df)

        logger.info(f"Created {len(self.feature_names)} derived features")

        return df

    def _create_performance_metrics(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create performance index features.

        Creates SPI (Schedule Performance Index), CPI (Cost Performance Index),
        and productivity metrics.

        :param df: Input DataFrame.
        :type df: pd.DataFrame
        :return: DataFrame with performance metrics.
        :rtype: pd.DataFrame
        """
        df = self._create_schedule_performance_index(df)
        df = self._create_cost_performance_index(df)
        df = self._create_productivity_metric(df)
        return df

    def _create_schedule_performance_index(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create Schedule Performance Index (SPI).

        SPI > 1 means ahead of schedule, SPI < 1 means behind schedule.

        :param df: Input DataFrame.
        :type df: pd.DataFrame
        :return: DataFrame with SPI feature.
        :rtype: pd.DataFrame
        """
        if "planned_hours" not in df.columns or "actual_hours" not in df.columns:
            return df

        df["schedule_performance_index"] = np.where(
            df["actual_hours"] > 0,
            df["planned_hours"] / df["actual_hours"],
            1.0,
        )
        df["schedule_performance_index"] = df["schedule_performance_index"].clip(0, 3)
        self.feature_names.append("schedule_performance_index")
        return df

    def _create_cost_performance_index(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create Cost Performance Index (CPI).

        CPI > 1 means under budget, CPI < 1 means over budget.

        :param df: Input DataFrame.
        :type df: pd.DataFrame
        :return: DataFrame with CPI feature.
        :rtype: pd.DataFrame
        """
        if "budget" not in df.columns or "spent" not in df.columns:
            return df

        df["cost_performance_index"] = np.where(
            df["spent"] > 0,
            df["budget"] / df["spent"],
            1.0,
        )
        df["cost_performance_index"] = df["cost_performance_index"].clip(0, 3)
        self.feature_names.append("cost_performance_index")
        return df

    def _create_productivity_metric(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create team productivity metric.

        Measures completion rate per team member.

        :param df: Input DataFrame.
        :type df: pd.DataFrame
        :return: DataFrame with productivity feature.
        :rtype: pd.DataFrame
        """
        if "completion_rate" not in df.columns or "team_size" not in df.columns:
            return df

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
        """
        Create time-based features.

        :param df: Input DataFrame.
        :type df: pd.DataFrame
        :param reference_date: Reference date for calculations.
        :type reference_date: datetime
        :return: DataFrame with temporal features.
        :rtype: pd.DataFrame
        """
        df = self._convert_dates(df)
        df = self._create_duration_features(df)
        df = self._create_elapsed_features(df, reference_date)
        return df

    def _convert_dates(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Convert date columns to datetime format.

        :param df: Input DataFrame.
        :type df: pd.DataFrame
        :return: DataFrame with converted dates.
        :rtype: pd.DataFrame
        """
        date_cols = ["start_date", "planned_end_date"]
        for col in date_cols:
            if col in df.columns:
                df[col] = pd.to_datetime(df[col], errors="coerce")
        return df

    def _create_duration_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create project duration features.

        :param df: Input DataFrame.
        :type df: pd.DataFrame
        :return: DataFrame with duration features.
        :rtype: pd.DataFrame
        """
        if "start_date" not in df.columns or "planned_end_date" not in df.columns:
            return df

        df["planned_duration_days"] = (
            df["planned_end_date"] - df["start_date"]
        ).dt.days
        df["planned_duration_days"] = df["planned_duration_days"].clip(lower=1)
        self.feature_names.append("planned_duration_days")
        return df

    def _create_elapsed_features(
        self, df: pd.DataFrame, reference_date: datetime
    ) -> pd.DataFrame:
        """
        Create features based on elapsed time.

        :param df: Input DataFrame.
        :type df: pd.DataFrame
        :param reference_date: Reference date for calculations.
        :type reference_date: datetime
        :return: DataFrame with elapsed time features.
        :rtype: pd.DataFrame
        """
        reference_ts = pd.Timestamp(reference_date)

        if "start_date" in df.columns:
            df["days_since_start"] = (reference_ts - df["start_date"]).dt.days
            df["days_since_start"] = df["days_since_start"].clip(lower=0)
            self.feature_names.append("days_since_start")

        if "planned_end_date" in df.columns:
            df["days_remaining"] = (df["planned_end_date"] - reference_ts).dt.days
            self.feature_names.append("days_remaining")

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
        """
        Create variance and deviation features.

        :param df: Input DataFrame.
        :type df: pd.DataFrame
        :return: DataFrame with variance features.
        :rtype: pd.DataFrame
        """
        df = self._create_budget_variance(df)
        df = self._create_hours_variance(df)
        df = self._create_schedule_gap(df)
        df = self._create_budget_utilization(df)
        return df

    def _create_budget_variance(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create budget variance percentage feature.

        :param df: Input DataFrame.
        :type df: pd.DataFrame
        :return: DataFrame with budget variance.
        :rtype: pd.DataFrame
        """
        if "budget" not in df.columns or "spent" not in df.columns:
            return df

        df["budget_variance_pct"] = np.where(
            df["budget"] > 0,
            ((df["spent"] - df["budget"]) / df["budget"]) * 100,
            0,
        )
        self.feature_names.append("budget_variance_pct")
        return df

    def _create_hours_variance(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create hours variance percentage feature.

        :param df: Input DataFrame.
        :type df: pd.DataFrame
        :return: DataFrame with hours variance.
        :rtype: pd.DataFrame
        """
        if "planned_hours" not in df.columns or "actual_hours" not in df.columns:
            return df

        df["hours_variance_pct"] = np.where(
            df["planned_hours"] > 0,
            ((df["actual_hours"] - df["planned_hours"]) / df["planned_hours"]) * 100,
            0,
        )
        self.feature_names.append("hours_variance_pct")
        return df

    def _create_schedule_gap(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create schedule gap feature.

        Positive gap means project is behind schedule.

        :param df: Input DataFrame.
        :type df: pd.DataFrame
        :return: DataFrame with schedule gap.
        :rtype: pd.DataFrame
        """
        if "time_elapsed_pct" not in df.columns or "completion_rate" not in df.columns:
            return df

        df["schedule_gap"] = df["time_elapsed_pct"] - df["completion_rate"]
        self.feature_names.append("schedule_gap")
        return df

    def _create_budget_utilization(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create budget utilization feature.

        :param df: Input DataFrame.
        :type df: pd.DataFrame
        :return: DataFrame with budget utilization.
        :rtype: pd.DataFrame
        """
        if "budget" not in df.columns or "spent" not in df.columns:
            return df

        df["budget_utilization"] = np.where(
            df["budget"] > 0,
            (df["spent"] / df["budget"]) * 100,
            0,
        )
        self.feature_names.append("budget_utilization")
        return df

    def _create_risk_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create risk indicator features.

        :param df: Input DataFrame.
        :type df: pd.DataFrame
        :return: DataFrame with risk indicators.
        :rtype: pd.DataFrame
        """
        df = self._create_team_stability(df)
        df = self._create_complexity_adjusted_progress(df)
        df = self._create_burn_rate(df)
        df = self._create_binary_flags(df)
        return df

    def _create_team_stability(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create team stability feature (inverse of turnover).

        :param df: Input DataFrame.
        :type df: pd.DataFrame
        :return: DataFrame with team stability.
        :rtype: pd.DataFrame
        """
        if "team_turnover" not in df.columns:
            return df

        df["team_stability"] = 1 - df["team_turnover"].clip(0, 1)
        self.feature_names.append("team_stability")
        return df

    def _create_complexity_adjusted_progress(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create complexity-adjusted progress feature.

        :param df: Input DataFrame.
        :type df: pd.DataFrame
        :return: DataFrame with complexity-adjusted progress.
        :rtype: pd.DataFrame
        """
        if "completion_rate" not in df.columns or "complexity_score" not in df.columns:
            return df

        df["complexity_adjusted_progress"] = np.where(
            df["complexity_score"] > 0,
            df["completion_rate"] / df["complexity_score"],
            df["completion_rate"],
        )
        self.feature_names.append("complexity_adjusted_progress")
        return df

    def _create_burn_rate(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create burn rate feature (spending per day).

        :param df: Input DataFrame.
        :type df: pd.DataFrame
        :return: DataFrame with burn rate.
        :rtype: pd.DataFrame
        """
        if "spent" not in df.columns or "days_since_start" not in df.columns:
            return df

        df["burn_rate"] = np.where(
            df["days_since_start"] > 0,
            df["spent"] / df["days_since_start"],
            0,
        )
        self.feature_names.append("burn_rate")
        return df

    def _create_binary_flags(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create binary flag features.

        :param df: Input DataFrame.
        :type df: pd.DataFrame
        :return: DataFrame with binary flags.
        :rtype: pd.DataFrame
        """
        if "budget_variance_pct" in df.columns:
            df["is_over_budget"] = (df["budget_variance_pct"] > 0).astype(int)
            self.feature_names.append("is_over_budget")

        if "schedule_gap" in df.columns:
            df["is_behind_schedule"] = (df["schedule_gap"] > 10).astype(int)
            self.feature_names.append("is_behind_schedule")

        return df

    def get_feature_names(self) -> list[str]:
        """
        Return list of created feature names.

        :return: List of feature names.
        :rtype: list[str]

        Example:
            >>> engineer = FeatureEngineer()
            >>> df = engineer.create_features(raw_df)
            >>> features = engineer.get_feature_names()
        """
        return self.feature_names.copy()
