"""
Jira issue-level to PRISM project-level aggregation
===================================================

Aggregates issue-level DataFrames (and optional comments, changelog, descriptions)
into one row per project with derived risk labels. Used by the Apache Kaggle
preprocessing script and by the live Jira Cloud connector.

Example:
    >>> from src.data.jira_aggregator import JiraAggregator
    >>> agg = JiraAggregator()
    >>> df = agg.aggregate(issues_df, comments_by_project, changelog_stats, descriptions_by_project)
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from loguru import logger
from tqdm import tqdm


class JiraAggregator:
    """
    Aggregate Jira-style issue metrics to PRISM project rows and derive risk labels.

    :ivar max_comments_per_project: Max comment bodies sampled per project for ``status_comments``.
    :vartype max_comments_per_project: int
    :ivar max_descriptions_per_project: Max issue descriptions sampled per project.
    :vartype max_descriptions_per_project: int
    """

    MAX_COMMENTS_PER_PROJECT = 500
    MAX_DESCRIPTIONS_PER_PROJECT = 50

    def __init__(
        self,
        max_comments_per_project: int = MAX_COMMENTS_PER_PROJECT,
        max_descriptions_per_project: int = MAX_DESCRIPTIONS_PER_PROJECT,
    ) -> None:
        """
        Initialize the aggregator with sampling limits.

        :param max_comments_per_project: Upper bound on comments merged into ``status_comments``.
        :type max_comments_per_project: int
        :param max_descriptions_per_project: Upper bound on descriptions merged into ``project_description``.
        :type max_descriptions_per_project: int

        Example:
            >>> JiraAggregator(max_comments_per_project=100)
        """
        self.max_comments_per_project = max_comments_per_project
        self.max_descriptions_per_project = max_descriptions_per_project

    def aggregate(
        self,
        issues: pd.DataFrame,
        comments_by_project: dict[str, list[str]],
        changelog_stats: dict[str, dict[str, int]],
        descriptions_by_project: dict[str, list[str]],
        *,
        show_progress: bool = True,
        progress_desc: str = "Aggregating",
    ) -> pd.DataFrame:
        """
        Build one PRISM-compatible row per ``project.key`` in ``issues``.

        :param issues: Issue-level rows with columns matching Apache export / API mapping.
        :type issues: pandas.DataFrame
        :param comments_by_project: Project key -> list of comment bodies.
        :type comments_by_project: dict[str, list[str]]
        :param changelog_stats: Project key -> ``status_changes`` and ``reopens`` counts.
        :type changelog_stats: dict[str, dict[str, int]]
        :param descriptions_by_project: Project key -> list of issue description strings.
        :type descriptions_by_project: dict[str, list[str]]
        :param show_progress: If ``True``, show a tqdm progress bar during grouping.
        :type show_progress: bool
        :param progress_desc: Label for the tqdm bar.
        :type progress_desc: str
        :return: One row per project with ``risk_level`` and ``risk_score_composite``.
        :rtype: pandas.DataFrame

        Example:
            >>> agg = JiraAggregator()
            >>> out = agg.aggregate(df, {}, {}, {}, show_progress=False)
        """
        logger.info("Aggregating to project level...")

        issues = issues.copy()
        issues["created"] = pd.to_datetime(issues["created"], errors="coerce")
        issues["updated"] = pd.to_datetime(issues["updated"], errors="coerce")
        issues["resolutiondate"] = pd.to_datetime(issues["resolutiondate"], errors="coerce")

        group_iter = issues.groupby("project.key", observed=True)
        if show_progress:
            group_iter = tqdm(group_iter, desc=progress_desc)

        results: list[dict] = []
        for project_key, group in group_iter:
            pkey = str(project_key)
            results.append(
                self._calculate_project_metrics(
                    project_key=pkey,
                    project_name=str(group["project.name"].iloc[0]),
                    issues=group,
                    comments=comments_by_project.get(pkey, []),
                    changelog=changelog_stats.get(pkey, {"status_changes": 0, "reopens": 0}),
                    descriptions=descriptions_by_project.get(pkey, []),
                )
            )

        out = pd.DataFrame(results)
        if len(out) == 0:
            logger.info("Created 0 project records")
            return out
        out = self._derive_risk_labels(out)
        logger.info(f"Created {len(out)} project records")
        return out

    def _calculate_project_metrics(
        self,
        project_key: str,
        project_name: str,
        issues: pd.DataFrame,
        comments: list[str],
        changelog: dict[str, int],
        descriptions: list[str],
    ) -> dict:
        """Compute metrics for one project group."""
        total_issues = len(issues)

        type_col = issues["issuetype.name"]
        status_col = issues["status.name"]
        priority_col = issues["priority.name"]

        n_bugs = int((type_col == "Bug").sum())
        n_improvements = int((type_col == "Improvement").sum())
        n_features = int(type_col.isin(["New Feature", "Feature"]).sum())
        n_tasks = int((type_col == "Task").sum())

        n_closed = int(status_col.isin(["Closed", "Resolved"]).sum())
        n_open = int(status_col.isin(["Open", "In Progress", "Reopened"]).sum())

        n_blockers = int((priority_col == "Blocker").sum())
        n_critical = int((priority_col == "Critical").sum())

        min_date = issues["created"].min()
        max_date = issues["updated"].max()
        duration = (max_date - min_date).days if pd.notna(min_date) and pd.notna(max_date) else 0
        duration = max(duration, 1)

        resolved = issues.loc[issues["resolutiondate"].notna() & issues["created"].notna()]
        if len(resolved) > 0:
            res_days = (resolved["resolutiondate"] - resolved["created"]).dt.days
            avg_res = float(res_days.mean())
            med_res = float(res_days.median())
        else:
            avg_res = med_res = 0.0

        assignees = issues["assignee"].dropna().unique()
        creators = issues["creator"].dropna().unique()
        reporters = issues["reporter"].dropna().unique()
        team_size = len(set(assignees) | set(creators))

        months = max(duration / 30, 1)
        velocity = n_closed / months
        defect_rate = n_bugs / max(total_issues, 1)
        completion_rate = (n_closed / max(total_issues, 1)) * 100

        if len(comments) > 50:
            rng = np.random.RandomState(42)
            sampled_comments = list(rng.choice(comments, 50, replace=False))
        else:
            sampled_comments = comments
        combined_comments = "\n\n".join(sampled_comments)

        combined_descriptions = "\n\n".join(descriptions[: self.max_descriptions_per_project])

        reopens = changelog.get("reopens", 0)
        status_changes = changelog.get("status_changes", 0)

        return {
            "project_id": project_key,
            "project_name": project_name,
            "project_type": "Development",
            "start_date": min_date.strftime("%Y-%m-%d") if pd.notna(min_date) else None,
            "planned_end_date": (
                issues["created"].max().strftime("%Y-%m-%d")
                if pd.notna(issues["created"].max())
                else None
            ),
            "actual_end_date": (
                resolved["resolutiondate"].max().strftime("%Y-%m-%d")
                if (
                    len(resolved) > 0
                    and pd.notna(resolved["resolutiondate"].max())
                    and n_open == 0
                )
                else None
            ),
            "total_issues": total_issues,
            "open_issues": n_open,
            "closed_issues": n_closed,
            "bug_count": n_bugs,
            "feature_count": n_features,
            "improvement_count": n_improvements,
            "task_count": n_tasks,
            "blocker_count": n_blockers,
            "critical_count": n_critical,
            "blocker_ratio": n_blockers / max(total_issues, 1),
            "critical_ratio": n_critical / max(total_issues, 1),
            "planned_hours": total_issues * 8,
            "actual_hours": n_closed * 8 + n_open * 4,
            "team_size": team_size,
            "unique_assignees": len(assignees),
            "unique_reporters": len(reporters),
            "completion_rate": round(completion_rate, 2),
            "velocity": round(velocity, 2),
            "defect_rate": round(defect_rate, 4),
            "avg_resolution_days": round(avg_res, 2),
            "median_resolution_days": round(med_res, 2),
            "reopen_count": reopens,
            "reopen_rate": reopens / max(n_closed, 1),
            "status_changes": status_changes,
            "churn_rate": status_changes / max(total_issues, 1),
            "project_duration_days": duration,
            "total_votes": (
                int(issues["votes.votes"].sum()) if "votes.votes" in issues.columns else 0
            ),
            "total_watchers": (
                int(issues["watches.watchCount"].sum())
                if "watches.watchCount" in issues.columns
                else 0
            ),
            "avg_watchers_per_issue": (
                float(issues["watches.watchCount"].mean())
                if "watches.watchCount" in issues.columns
                else 0.0
            ),
            "status": "Active" if n_open > 0 else "Completed",
            "priority": "Critical" if n_blockers > 5 else ("High" if n_critical > 10 else "Medium"),
            "methodology": "Agile",
            "department": "Apache Foundation",
            "client_type": "External",
            "status_comments": combined_comments,
            "project_description": combined_descriptions,
            "team_feedback": "",
            "complexity_score": min(10, max(1, int(team_size / 5 + duration / 365))),
            "dependencies": 0,
            "team_turnover": 0.0,
        }

    @staticmethod
    def _derive_risk_labels(df: pd.DataFrame) -> pd.DataFrame:
        """
        Assign risk labels using a composite score built from percentile ranks.

        :param df: Project-level metrics before ``risk_level`` / ``risk_score_composite``.
        :type df: pandas.DataFrame
        :return: ``df`` with ``risk_score_composite`` and ``risk_level`` columns.
        :rtype: pandas.DataFrame
        """
        risk_cols = {
            "avg_resolution_days": 1.0,
            "reopen_rate": 1.0,
            "blocker_ratio": 1.0,
            "defect_rate": 1.0,
            "churn_rate": 0.5,
            "completion_rate": -1.0,
        }

        score = pd.Series(0.0, index=df.index)
        total_weight = sum(abs(w) for w in risk_cols.values())

        for col, weight in risk_cols.items():
            if col not in df.columns:
                continue
            pct_rank = df[col].rank(pct=True, na_option="bottom")
            if weight < 0:
                pct_rank = 1.0 - pct_rank
            score += pct_rank * abs(weight)

        score /= total_weight
        df = df.copy()
        df["risk_score_composite"] = score.round(4)

        high_thresh = score.quantile(0.67)
        low_thresh = score.quantile(0.33)

        df["risk_level"] = np.where(
            score >= high_thresh, "High", np.where(score >= low_thresh, "Medium", "Low")
        )

        risk_counts = df["risk_level"].value_counts()
        logger.info(
            f"Risk distribution: {risk_counts.to_dict()}  "
            f"(thresholds — low: {low_thresh:.3f}, high: {high_thresh:.3f})"
        )
        return df
