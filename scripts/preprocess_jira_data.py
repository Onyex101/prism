#!/usr/bin/env python3
"""
Apache JIRA Data Preprocessing Script

Transforms issue-level JIRA data into PRISM-compatible project-level data.

Source: https://www.kaggle.com/datasets/tedlozzo/apaches-jira-issues

This script:
1. Reads large JIRA CSV files efficiently (in chunks)
2. Aggregates issue-level data to project-level metrics
3. Joins comments for LLM text analysis
4. Derives risk labels from issue outcomes
5. Outputs PRISM-compatible project data

Usage:
    python scripts/preprocess_jira_data.py [--sample N] [--projects P1,P2,P3]

Examples:
    # Process all projects (takes a while)
    python scripts/preprocess_jira_data.py

    # Process only top 50 projects by issue count
    python scripts/preprocess_jira_data.py --sample 50

    # Process specific projects
    python scripts/preprocess_jira_data.py --projects SPARK,KAFKA,FLINK
"""

import argparse
import sys
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from tqdm import tqdm

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from loguru import logger

# Configure logging
logger.remove()
logger.add(
    sys.stderr,
    level="INFO",
    format="<green>{time:HH:mm:ss}</green> | <level>{level: <8}</level> | {message}",
)


class JiraDataPreprocessor:
    """
    Preprocessor for Apache JIRA dataset.

    Transforms issue-level data into project-level aggregates
    compatible with PRISM's risk analysis framework.
    """

    # Default paths
    RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw"
    PROCESSED_DATA_DIR = PROJECT_ROOT / "data" / "processed"

    # File names
    ISSUES_FILE = "issues.csv"
    COMMENTS_FILE = "comments.csv"
    CHANGELOG_FILE = "changelog.csv"
    ISSUELINKS_FILE = "issuelinks.csv"

    # Chunk size for reading large files
    CHUNK_SIZE = 100_000

    # Columns to read from each file (for memory efficiency)
    ISSUES_COLS = [
        "id",
        "key",
        "summary",
        "description",
        "resolution.name",
        "priority.name",
        "status.name",
        "issuetype.name",
        "project.key",
        "project.name",
        "created",
        "updated",
        "resolutiondate",
        "assignee",
        "creator",
        "reporter",
        "votes.votes",
        "watches.watchCount",
    ]

    COMMENTS_COLS = ["key", "comment.id", "comment.body", "comment.created"]

    CHANGELOG_COLS = ["key", "field", "fromString", "toString", "created"]

    def __init__(
        self,
        raw_data_dir: Optional[Path] = None,
        output_dir: Optional[Path] = None,
    ):
        """
        Initialize the preprocessor.

        Args:
            raw_data_dir: Directory containing raw JIRA CSV files
            output_dir: Directory for processed output
        """
        self.raw_data_dir = Path(raw_data_dir) if raw_data_dir else self.RAW_DATA_DIR
        self.output_dir = Path(output_dir) if output_dir else self.PROCESSED_DATA_DIR

        # Ensure output directory exists
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Validate input files exist
        self._validate_input_files()

    def _validate_input_files(self):
        """Check that required input files exist."""
        required_files = [self.ISSUES_FILE, self.COMMENTS_FILE]

        for fname in required_files:
            fpath = self.raw_data_dir / fname
            if not fpath.exists():
                raise FileNotFoundError(
                    f"Required file not found: {fpath}\n"
                    f"Please download the Apache JIRA dataset from:\n"
                    f"https://www.kaggle.com/datasets/tedlozzo/apaches-jira-issues"
                )

        logger.info(f"Input files validated in {self.raw_data_dir}")

    def get_project_list(self, top_n: Optional[int] = None) -> list[str]:
        """
        Get list of all projects in the dataset.

        Args:
            top_n: If provided, return only top N projects by issue count

        Returns:
            List of project keys
        """
        logger.info("Scanning projects in dataset...")

        project_counts = {}

        for chunk in tqdm(
            pd.read_csv(
                self.raw_data_dir / self.ISSUES_FILE,
                usecols=["project.key"],
                chunksize=self.CHUNK_SIZE,
                low_memory=False,
            ),
            desc="Counting projects",
        ):
            for proj, count in chunk["project.key"].value_counts().items():
                project_counts[proj] = project_counts.get(proj, 0) + count

        # Sort by count
        sorted_projects = sorted(project_counts.items(), key=lambda x: x[1], reverse=True)

        if top_n:
            sorted_projects = sorted_projects[:top_n]

        logger.info(f"Found {len(project_counts)} projects, selected {len(sorted_projects)}")

        return [p[0] for p in sorted_projects]

    def load_issues_for_projects(self, project_keys: list[str]) -> pd.DataFrame:
        """
        Load issues for specified projects.

        Args:
            project_keys: List of project keys to load

        Returns:
            DataFrame with issue data
        """
        logger.info(f"Loading issues for {len(project_keys)} projects...")

        project_set = set(project_keys)
        chunks = []

        for chunk in tqdm(
            pd.read_csv(
                self.raw_data_dir / self.ISSUES_FILE,
                usecols=self.ISSUES_COLS,
                chunksize=self.CHUNK_SIZE,
                low_memory=False,
            ),
            desc="Loading issues",
        ):
            # Filter to selected projects
            filtered = chunk[chunk["project.key"].isin(project_set)]
            if len(filtered) > 0:
                chunks.append(filtered)

        if not chunks:
            raise ValueError("No issues found for specified projects")

        issues = pd.concat(chunks, ignore_index=True)
        logger.info(f"Loaded {len(issues):,} issues")

        return issues

    def load_comments_for_issues(
        self, issue_keys: set[str], sample_per_project: int = 100
    ) -> pd.DataFrame:
        """
        Load comments for specified issues.

        Args:
            issue_keys: Set of issue keys to load comments for
            sample_per_project: Max comments to sample per project for LLM

        Returns:
            DataFrame with comment data
        """
        logger.info(f"Loading comments for {len(issue_keys):,} issues...")

        chunks = []

        for chunk in tqdm(
            pd.read_csv(
                self.raw_data_dir / self.COMMENTS_FILE,
                usecols=self.COMMENTS_COLS,
                chunksize=self.CHUNK_SIZE,
                low_memory=False,
            ),
            desc="Loading comments",
        ):
            # Filter to selected issues
            filtered = chunk[chunk["key"].isin(issue_keys)]
            if len(filtered) > 0:
                chunks.append(filtered)

        if not chunks:
            logger.warning("No comments found for specified issues")
            return pd.DataFrame(columns=self.COMMENTS_COLS)

        comments = pd.concat(chunks, ignore_index=True)
        logger.info(f"Loaded {len(comments):,} comments")

        return comments

    def load_changelog_for_issues(self, issue_keys: set[str]) -> pd.DataFrame:
        """
        Load changelog entries for specified issues.

        Args:
            issue_keys: Set of issue keys to load changelog for

        Returns:
            DataFrame with changelog data
        """
        changelog_path = self.raw_data_dir / self.CHANGELOG_FILE

        if not changelog_path.exists():
            logger.warning("Changelog file not found, skipping")
            return pd.DataFrame(columns=self.CHANGELOG_COLS)

        logger.info(f"Loading changelog for {len(issue_keys):,} issues...")

        chunks = []

        for chunk in tqdm(
            pd.read_csv(
                changelog_path,
                usecols=self.CHANGELOG_COLS,
                chunksize=self.CHUNK_SIZE,
                low_memory=False,
            ),
            desc="Loading changelog",
        ):
            filtered = chunk[chunk["key"].isin(issue_keys)]
            if len(filtered) > 0:
                chunks.append(filtered)

        if not chunks:
            return pd.DataFrame(columns=self.CHANGELOG_COLS)

        changelog = pd.concat(chunks, ignore_index=True)
        logger.info(f"Loaded {len(changelog):,} changelog entries")

        return changelog

    def aggregate_to_project_level(
        self,
        issues: pd.DataFrame,
        comments: pd.DataFrame,
        changelog: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Aggregate issue-level data to project-level metrics.

        Args:
            issues: Issue data
            comments: Comment data
            changelog: Changelog data

        Returns:
            Project-level DataFrame compatible with PRISM schema
        """
        logger.info("Aggregating to project level...")

        # Parse dates
        issues["created"] = pd.to_datetime(issues["created"], errors="coerce")
        issues["updated"] = pd.to_datetime(issues["updated"], errors="coerce")
        issues["resolutiondate"] = pd.to_datetime(issues["resolutiondate"], errors="coerce")

        # Group by project
        projects = []

        for project_key, group in tqdm(issues.groupby("project.key"), desc="Processing projects"):
            project_data = self._calculate_project_metrics(
                project_key=project_key,
                project_name=group["project.name"].iloc[0],
                issues=group,
                comments=comments[comments["key"].str.startswith(project_key + "-")],
                changelog=(
                    changelog[changelog["key"].str.startswith(project_key + "-")]
                    if len(changelog) > 0
                    else pd.DataFrame()
                ),
            )
            projects.append(project_data)

        df = pd.DataFrame(projects)

        # Derive risk labels
        df = self._derive_risk_labels(df)

        logger.info(f"Created {len(df)} project records")

        return df

    def _calculate_project_metrics(
        self,
        project_key: str,
        project_name: str,
        issues: pd.DataFrame,
        comments: pd.DataFrame,
        changelog: pd.DataFrame,
    ) -> dict:
        """
        Calculate all metrics for a single project.

        Args:
            project_key: Project identifier
            project_name: Project name
            issues: Issues for this project
            comments: Comments for this project
            changelog: Changelog for this project

        Returns:
            Dictionary with project metrics
        """
        total_issues = len(issues)

        # Issue type breakdown
        bugs = issues[issues["issuetype.name"] == "Bug"]
        improvements = issues[issues["issuetype.name"] == "Improvement"]
        features = issues[issues["issuetype.name"].isin(["New Feature", "Feature"])]
        tasks = issues[issues["issuetype.name"] == "Task"]

        # Status breakdown
        closed = issues[issues["status.name"].isin(["Closed", "Resolved"])]
        open_issues = issues[issues["status.name"].isin(["Open", "In Progress", "Reopened"])]

        # Priority breakdown
        blockers = issues[issues["priority.name"] == "Blocker"]
        critical = issues[issues["priority.name"] == "Critical"]

        # Date calculations
        min_date = issues["created"].min()
        max_date = issues["updated"].max()
        project_duration_days = (
            (max_date - min_date).days if pd.notna(min_date) and pd.notna(max_date) else 0
        )
        project_duration_days = max(project_duration_days, 1)  # Avoid division by zero

        # Resolution time (for resolved issues)
        resolved_issues = issues[issues["resolutiondate"].notna() & issues["created"].notna()]
        if len(resolved_issues) > 0:
            resolution_times = (
                resolved_issues["resolutiondate"] - resolved_issues["created"]
            ).dt.days
            avg_resolution_time = resolution_times.mean()
            median_resolution_time = resolution_times.median()
        else:
            avg_resolution_time = 0
            median_resolution_time = 0

        # Team metrics (unique contributors)
        assignees = issues["assignee"].dropna().unique()
        creators = issues["creator"].dropna().unique()
        reporters = issues["reporter"].dropna().unique()
        team_size = len(set(assignees) | set(creators))

        # Velocity (issues resolved per month)
        months = max(project_duration_days / 30, 1)
        velocity = len(closed) / months

        # Defect rate
        defect_rate = len(bugs) / max(total_issues, 1)

        # Completion rate
        completion_rate = (len(closed) / max(total_issues, 1)) * 100

        # Combine comments for text analysis
        project_comments = comments["comment.body"].dropna().tolist()
        # Sample comments if too many (for LLM efficiency)
        if len(project_comments) > 50:
            np.random.seed(42)
            project_comments = list(np.random.choice(project_comments, 50, replace=False))

        combined_comments = " ".join(
            str(c)[:500] for c in project_comments[:20]
        )  # Limit text length

        # Issue descriptions for context
        descriptions = issues["description"].dropna().tolist()[:10]
        combined_descriptions = " ".join(str(d)[:300] for d in descriptions)

        # Changelog analysis
        reopens = 0
        status_changes = 0
        if len(changelog) > 0:
            status_changes_df = changelog[changelog["field"] == "status"]
            status_changes = len(status_changes_df)
            reopens = len(status_changes_df[status_changes_df["toString"] == "Reopened"])

        return {
            # Core identifiers
            "project_id": project_key,
            "project_name": project_name,
            "project_type": "Development",  # All are software projects
            # Dates
            "start_date": min_date.strftime("%Y-%m-%d") if pd.notna(min_date) else None,
            "planned_end_date": None,  # Not available in JIRA
            "actual_end_date": None,
            # Effort metrics (derived from issues instead of hours)
            "total_issues": total_issues,
            "open_issues": len(open_issues),
            "closed_issues": len(closed),
            "bug_count": len(bugs),
            "feature_count": len(features),
            "improvement_count": len(improvements),
            "task_count": len(tasks),
            # Priority metrics
            "blocker_count": len(blockers),
            "critical_count": len(critical),
            "blocker_ratio": len(blockers) / max(total_issues, 1),
            "critical_ratio": len(critical) / max(total_issues, 1),
            # Budget/cost placeholders (not available in JIRA)
            "budget": None,
            "spent": None,
            "planned_hours": total_issues * 8,  # Estimate: 8 hours per issue avg
            "actual_hours": int(len(closed) * 8 + len(open_issues) * 4),  # Partial for open
            # Team metrics
            "team_size": team_size,
            "unique_assignees": len(assignees),
            "unique_reporters": len(reporters),
            # Performance metrics
            "completion_rate": round(completion_rate, 2),
            "velocity": round(velocity, 2),  # Issues per month
            "defect_rate": round(defect_rate, 4),
            "avg_resolution_days": round(avg_resolution_time, 2),
            "median_resolution_days": round(median_resolution_time, 2),
            # Quality indicators
            "reopen_count": reopens,
            "reopen_rate": reopens / max(len(closed), 1),
            "status_changes": status_changes,
            "churn_rate": status_changes / max(total_issues, 1),
            # Duration
            "project_duration_days": project_duration_days,
            # Engagement metrics
            "total_votes": issues["votes.votes"].sum() if "votes.votes" in issues.columns else 0,
            "total_watchers": (
                issues["watches.watchCount"].sum() if "watches.watchCount" in issues.columns else 0
            ),
            "avg_watchers_per_issue": (
                issues["watches.watchCount"].mean() if "watches.watchCount" in issues.columns else 0
            ),
            # PRISM-required fields
            "status": "Active" if len(open_issues) > 0 else "Completed",
            "priority": (
                "Critical" if len(blockers) > 5 else ("High" if len(critical) > 10 else "Medium")
            ),
            "methodology": "Agile",  # Apache projects typically use agile
            "department": "Apache Foundation",
            "client_type": "External",  # Open source
            # Text for LLM analysis
            "status_comments": combined_comments[:2000] if combined_comments else "",
            "project_description": combined_descriptions[:1000] if combined_descriptions else "",
            "team_feedback": "",  # Not available
            # Complexity estimate based on team size and duration
            "complexity_score": min(10, max(1, int(team_size / 5 + project_duration_days / 365))),
            "dependencies": 0,  # Would need issuelinks analysis
            "team_turnover": 0.0,  # Not directly measurable
        }

    def _derive_risk_labels(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Derive risk labels from project metrics.

        Risk is determined by:
        - High: blocker_ratio > 0.05 OR reopen_rate > 0.1 OR defect_rate > 0.5
        - Medium: blocker_ratio > 0.02 OR reopen_rate > 0.05 OR defect_rate > 0.3
        - Low: Otherwise

        Args:
            df: Project DataFrame

        Returns:
            DataFrame with risk_level column added
        """

        def calculate_risk(row):
            # High risk indicators
            if (
                row["blocker_ratio"] > 0.05
                or row["reopen_rate"] > 0.1
                or row["defect_rate"] > 0.5
                or row["avg_resolution_days"] > 60
            ):
                return "High"

            # Medium risk indicators
            if (
                row["blocker_ratio"] > 0.02
                or row["reopen_rate"] > 0.05
                or row["defect_rate"] > 0.3
                or row["avg_resolution_days"] > 30
            ):
                return "Medium"

            return "Low"

        df["risk_level"] = df.apply(calculate_risk, axis=1)

        # Log distribution
        risk_counts = df["risk_level"].value_counts()
        logger.info(f"Risk distribution: {risk_counts.to_dict()}")

        return df

    def process(
        self,
        project_keys: Optional[list[str]] = None,
        top_n_projects: Optional[int] = None,
        output_filename: str = "jira_projects.csv",
    ) -> pd.DataFrame:
        """
        Main processing pipeline.

        Args:
            project_keys: Specific projects to process (optional)
            top_n_projects: Process top N projects by issue count (optional)
            output_filename: Name for output file

        Returns:
            Processed project-level DataFrame
        """
        # Determine which projects to process
        if project_keys:
            projects = project_keys
        elif top_n_projects:
            projects = self.get_project_list(top_n=top_n_projects)
        else:
            projects = self.get_project_list()

        logger.info(
            f"Processing {len(projects)} projects: {projects[:10]}{'...' if len(projects) > 10 else ''}"
        )

        # Load data
        issues = self.load_issues_for_projects(projects)
        issue_keys = set(issues["key"].unique())

        comments = self.load_comments_for_issues(issue_keys)
        changelog = self.load_changelog_for_issues(issue_keys)

        # Aggregate
        df = self.aggregate_to_project_level(issues, comments, changelog)

        # Save output
        output_path = self.output_dir / output_filename
        df.to_csv(output_path, index=False)
        logger.info(f"Saved processed data to {output_path}")

        # Also save a sample for quick testing
        sample_path = self.output_dir / "jira_projects_sample.csv"
        df.head(20).to_csv(sample_path, index=False)
        logger.info(f"Saved sample (20 projects) to {sample_path}")

        return df


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Preprocess Apache JIRA data for PRISM",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    parser.add_argument(
        "--sample", "-s", type=int, default=None, help="Process only top N projects by issue count"
    )

    parser.add_argument(
        "--projects",
        "-p",
        type=str,
        default=None,
        help="Comma-separated list of project keys to process (e.g., SPARK,KAFKA,FLINK)",
    )

    parser.add_argument(
        "--output",
        "-o",
        type=str,
        default="jira_projects.csv",
        help="Output filename (default: jira_projects.csv)",
    )

    parser.add_argument(
        "--raw-dir", type=str, default=None, help="Directory containing raw JIRA CSV files"
    )

    parser.add_argument(
        "--output-dir", type=str, default=None, help="Directory for processed output"
    )

    args = parser.parse_args()

    # Parse project list if provided
    project_keys = None
    if args.projects:
        project_keys = [p.strip().upper() for p in args.projects.split(",")]

    # Initialize and run preprocessor
    preprocessor = JiraDataPreprocessor(
        raw_data_dir=args.raw_dir,
        output_dir=args.output_dir,
    )

    df = preprocessor.process(
        project_keys=project_keys,
        top_n_projects=args.sample,
        output_filename=args.output,
    )

    # Print summary
    print("\n" + "=" * 60)
    print("PREPROCESSING COMPLETE")
    print("=" * 60)
    print(f"Projects processed: {len(df)}")
    print(f"Risk distribution:")
    print(df["risk_level"].value_counts().to_string())
    print(f"\nSample projects:")
    print(
        df[["project_id", "project_name", "total_issues", "completion_rate", "risk_level"]]
        .head(10)
        .to_string()
    )
    print("=" * 60)


if __name__ == "__main__":
    main()
