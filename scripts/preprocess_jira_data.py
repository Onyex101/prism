#!/usr/bin/env python3
"""
Apache JIRA Data Preprocessing Script

Transforms issue-level JIRA data into PRISM-compatible project-level data.

Source: https://www.kaggle.com/datasets/tedlozzo/apaches-jira-issues

This script:
1. Reads large JIRA CSV files with bounded memory
2. Aggregates issue-level data to project-level metrics
3. Samples full-text comments for LLM analysis (no truncation)
4. Derives risk labels from issue outcomes
5. Outputs PRISM-compatible project data

Designed to handle all 640+ projects without running out of memory.

Usage:
    python scripts/preprocess_jira_data.py [--sample N] [--projects P1,P2,P3]

Examples:
    python scripts/preprocess_jira_data.py                          # all projects
    python scripts/preprocess_jira_data.py --sample 50              # top 50
    python scripts/preprocess_jira_data.py --projects SPARK,KAFKA   # specific
"""

import argparse
import csv
import gc
import sys
from collections import defaultdict
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from loguru import logger

logger.remove()
logger.add(
    sys.stderr,
    level="INFO",
    format="<green>{time:HH:mm:ss}</green> | <level>{level: <8}</level> | {message}",
)

csv.field_size_limit(sys.maxsize)


def _project_from_key(issue_key: str) -> str:
    """Extract project key from a JIRA issue key (e.g. 'SPARK-1234' -> 'SPARK')."""
    idx = issue_key.find("-")
    return issue_key[:idx] if idx > 0 else issue_key


class JiraDataPreprocessor:
    """
    Memory-efficient preprocessor for the Apache JIRA dataset.

    Memory strategy per phase:
      1. Project list  — single-column pandas scan, ~50 MB peak
      2. Issue metrics  — pandas chunked, 15 lightweight columns w/ categories
      3. Descriptions   — csv.reader stream, ≤ 20 full-text per project
      4. Comments       — csv.reader stream, ≤ 200 full-text per project
      5. Changelog      — csv.reader stream, integer counters only
      6. Aggregation    — iterates grouped issues, produces one dict per project
    """

    RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw"
    PROCESSED_DATA_DIR = PROJECT_ROOT / "data" / "processed"

    ISSUES_FILE = "issues.csv"
    COMMENTS_FILE = "comments.csv"
    CHANGELOG_FILE = "changelog.csv"
    ISSUELINKS_FILE = "issuelinks.csv"

    CHUNK_SIZE = 50_000

    ISSUES_METRIC_COLS = [
        "key",
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

    # Convert these to pandas Categorical after loading to cut memory ~60 %
    CATEGORICAL_COLS = [
        "resolution.name",
        "priority.name",
        "status.name",
        "issuetype.name",
        "project.key",
        "project.name",
    ]

    MAX_COMMENTS_PER_PROJECT = 500
    MAX_DESCRIPTIONS_PER_PROJECT = 50

    def __init__(
        self,
        raw_data_dir: Optional[Path] = None,
        output_dir: Optional[Path] = None,
    ):
        self.raw_data_dir = Path(raw_data_dir) if raw_data_dir else self.RAW_DATA_DIR
        self.output_dir = Path(output_dir) if output_dir else self.PROCESSED_DATA_DIR
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self._validate_input_files()

    def _validate_input_files(self):
        required = [self.ISSUES_FILE, self.COMMENTS_FILE]
        for fname in required:
            fpath = self.raw_data_dir / fname
            if not fpath.exists():
                raise FileNotFoundError(
                    f"Required file not found: {fpath}\n"
                    f"Download from: https://www.kaggle.com/datasets/tedlozzo/apaches-jira-issues"
                )
        logger.info(f"Input files validated in {self.raw_data_dir}")

    # ------------------------------------------------------------------
    # Phase 1: project list
    # ------------------------------------------------------------------

    def get_project_list(self, top_n: Optional[int] = None) -> list[str]:
        logger.info("Scanning projects in dataset...")
        counts: dict[str, int] = {}
        for chunk in tqdm(
            pd.read_csv(
                self.raw_data_dir / self.ISSUES_FILE,
                usecols=["project.key"],
                chunksize=self.CHUNK_SIZE,
            ),
            desc="Counting projects",
        ):
            for proj, n in chunk["project.key"].value_counts().items():
                counts[proj] = counts.get(proj, 0) + n

        ranked = sorted(counts.items(), key=lambda x: x[1], reverse=True)
        if top_n:
            ranked = ranked[:top_n]
        logger.info(f"Found {len(counts)} projects, selected {len(ranked)}")
        return [p for p, _ in ranked]

    # ------------------------------------------------------------------
    # Phase 2: issue metrics (pandas, no heavy text columns)
    # ------------------------------------------------------------------

    def _load_issue_metrics(self, project_keys: set[str]) -> pd.DataFrame:
        logger.info(f"Loading issue metrics for {len(project_keys)} projects...")
        chunks = []
        for chunk in tqdm(
            pd.read_csv(
                self.raw_data_dir / self.ISSUES_FILE,
                usecols=self.ISSUES_METRIC_COLS,
                chunksize=self.CHUNK_SIZE,
            ),
            desc="Loading issues",
        ):
            mask = chunk["project.key"].isin(project_keys)
            filtered = chunk.loc[mask]
            if len(filtered) > 0:
                chunks.append(filtered)

        if not chunks:
            raise ValueError("No issues found for specified projects")

        issues = pd.concat(chunks, ignore_index=True)
        # Re-apply categorical dtype after concat (concat upcasts to object)
        for c in self.CATEGORICAL_COLS:
            if c in issues.columns:
                issues[c] = issues[c].astype("category")

        logger.info(
            f"Loaded {len(issues):,} issues ({issues.memory_usage(deep=True).sum() / 1e6:.0f} MB)"
        )
        return issues

    # ------------------------------------------------------------------
    # Phase 3: descriptions (csv.reader, one pass, bounded per project)
    # ------------------------------------------------------------------

    def _stream_descriptions(self, project_keys: set[str]) -> dict[str, list[str]]:
        logger.info("Streaming issue descriptions...")
        descs: dict[str, list[str]] = defaultdict(list)
        limit = self.MAX_DESCRIPTIONS_PER_PROJECT
        full_count = 0

        fpath = self.raw_data_dir / self.ISSUES_FILE
        with open(fpath, "r", encoding="utf-8", errors="replace") as f:
            reader = csv.reader(f)
            header = next(reader)
            try:
                idx_proj = header.index("project.key")
                idx_desc = header.index("description")
            except ValueError as e:
                logger.error(f"Missing column in issues.csv: {e}")
                return {}

            for row in reader:
                proj = row[idx_proj]
                if proj not in project_keys:
                    continue
                if len(descs[proj]) >= limit:
                    if full_count >= len(project_keys):
                        break
                    continue
                desc = row[idx_desc] if idx_desc < len(row) else ""
                if desc:
                    descs[proj].append(desc)
                    if len(descs[proj]) == limit:
                        full_count += 1

        total = sum(len(v) for v in descs.values())
        logger.info(f"Sampled {total:,} descriptions across {len(descs)} projects")
        return dict(descs)

    # ------------------------------------------------------------------
    # Phase 4: comments (csv.reader, one pass, bounded per project)
    # Full text is preserved — no truncation.
    # ------------------------------------------------------------------

    def _stream_comments(self, project_keys: set[str]) -> dict[str, list[str]]:
        logger.info("Streaming comments...")
        comments: dict[str, list[str]] = defaultdict(list)
        limit = self.MAX_COMMENTS_PER_PROJECT
        full_count = 0

        fpath = self.raw_data_dir / self.COMMENTS_FILE
        with open(fpath, "r", encoding="utf-8", errors="replace") as f:
            reader = csv.reader(f)
            header = next(reader)
            try:
                idx_key = header.index("key")
                idx_body = header.index("comment.body")
            except ValueError as e:
                logger.error(f"Missing column in comments.csv: {e}")
                return {}

            rows_read = 0
            for row in reader:
                rows_read += 1
                if rows_read % 10_000_000 == 0:
                    logger.info(f"  ...{rows_read:,} comment rows streamed")

                if idx_key >= len(row):
                    continue
                proj = _project_from_key(row[idx_key])
                if proj not in project_keys:
                    continue
                if len(comments[proj]) >= limit:
                    continue

                body = row[idx_body] if idx_body < len(row) else ""
                if body:
                    comments[proj].append(body)
                    if len(comments[proj]) == limit:
                        full_count += 1

        total = sum(len(v) for v in comments.values())
        logger.info(
            f"Collected {total:,} comments across {len(comments)} projects "
            f"({rows_read:,} rows streamed)"
        )
        return dict(comments)

    # ------------------------------------------------------------------
    # Phase 5: changelog (csv.reader, one pass, counters only)
    # ------------------------------------------------------------------

    def _stream_changelog_stats(self, project_keys: set[str]) -> dict[str, dict[str, int]]:
        changelog_path = self.raw_data_dir / self.CHANGELOG_FILE
        if not changelog_path.exists():
            logger.warning("Changelog file not found, skipping")
            return {}

        logger.info("Streaming changelog...")
        stats: dict[str, dict[str, int]] = defaultdict(lambda: {"status_changes": 0, "reopens": 0})

        with open(changelog_path, "r", encoding="utf-8", errors="replace") as f:
            reader = csv.reader(f)
            header = next(reader)
            try:
                idx_key = header.index("key")
                idx_field = header.index("field")
                idx_to = header.index("toString")
            except ValueError as e:
                logger.error(f"Missing column in changelog.csv: {e}")
                return {}

            rows_read = 0
            for row in reader:
                rows_read += 1
                if rows_read % 10_000_000 == 0:
                    logger.info(f"  ...{rows_read:,} changelog rows streamed")

                if idx_field >= len(row) or row[idx_field] != "status":
                    continue
                if idx_key >= len(row):
                    continue

                proj = _project_from_key(row[idx_key])
                if proj not in project_keys:
                    continue

                stats[proj]["status_changes"] += 1
                if idx_to < len(row) and row[idx_to] == "Reopened":
                    stats[proj]["reopens"] += 1

        logger.info(f"Changelog stats for {len(stats)} projects ({rows_read:,} rows streamed)")
        return dict(stats)

    # ------------------------------------------------------------------
    # Phase 6: aggregate
    # ------------------------------------------------------------------

    def _aggregate(
        self,
        issues: pd.DataFrame,
        comments_by_project: dict[str, list[str]],
        changelog_stats: dict[str, dict[str, int]],
        descriptions_by_project: dict[str, list[str]],
    ) -> pd.DataFrame:
        logger.info("Aggregating to project level...")

        issues["created"] = pd.to_datetime(issues["created"], errors="coerce")
        issues["updated"] = pd.to_datetime(issues["updated"], errors="coerce")
        issues["resolutiondate"] = pd.to_datetime(issues["resolutiondate"], errors="coerce")

        results: list[dict] = []
        for project_key, group in tqdm(
            issues.groupby("project.key", observed=True), desc="Aggregating"
        ):
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

        df = pd.DataFrame(results)
        df = self._derive_risk_labels(df)
        logger.info(f"Created {len(df)} project records")
        return df

    # ------------------------------------------------------------------
    # Per-project metric calculation
    # ------------------------------------------------------------------

    @staticmethod
    def _calculate_project_metrics(
        project_key: str,
        project_name: str,
        issues: pd.DataFrame,
        comments: list[str],
        changelog: dict[str, int],
        descriptions: list[str],
    ) -> dict:
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

        # Full-text LLM fields — sample then join, no per-item truncation
        if len(comments) > 50:
            rng = np.random.RandomState(42)
            sampled_comments = list(rng.choice(comments, 50, replace=False))
        else:
            sampled_comments = comments
        combined_comments = "\n\n".join(sampled_comments)

        combined_descriptions = "\n\n".join(descriptions)

        reopens = changelog.get("reopens", 0)
        status_changes = changelog.get("status_changes", 0)

        return {
            "project_id": project_key,
            "project_name": project_name,
            "project_type": "Development",
            "start_date": min_date.strftime("%Y-%m-%d") if pd.notna(min_date) else None,
            # planned_end_date: latest issue creation date (last point the project was actively planned)
            # actual_end_date:  latest resolution date for completed projects, None if still active
            "planned_end_date": (
                issues["created"].max().strftime("%Y-%m-%d")
                if pd.notna(issues["created"].max())
                else None
            ),
            "actual_end_date": (
                resolved["resolutiondate"].max().strftime("%Y-%m-%d")
                if (
                    len(resolved) > 0 and pd.notna(resolved["resolutiondate"].max()) and n_open == 0
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

    # ------------------------------------------------------------------
    # Risk labels
    # ------------------------------------------------------------------

    @staticmethod
    def _derive_risk_labels(df: pd.DataFrame) -> pd.DataFrame:
        """
        Assign risk labels using a composite score built from percentile ranks.

        Each indicator is ranked within the dataset (0–1), so thresholds are
        relative — they remain meaningful regardless of whether the data comes
        from small corporate projects or large open-source repos like Apache.

        Indicators and their direction:
          - avg_resolution_days  ↑ higher = worse
          - reopen_rate          ↑ higher = worse
          - blocker_ratio        ↑ higher = worse
          - defect_rate          ↑ higher = worse
          - churn_rate           ↑ higher = worse
          - completion_rate      ↓ lower  = worse  (inverted)

        Composite score = weighted average of percentile ranks (0 = best, 1 = worst).
        Projects in top 33 % → High, middle 33 % → Medium, bottom 33 % → Low.
        """
        risk_cols = {
            "avg_resolution_days": 1.0,
            "reopen_rate": 1.0,
            "blocker_ratio": 1.0,
            "defect_rate": 1.0,
            "churn_rate": 0.5,
            "completion_rate": -1.0,  # negative weight — lower completion = higher risk
        }

        score = pd.Series(0.0, index=df.index)
        total_weight = sum(abs(w) for w in risk_cols.values())

        for col, weight in risk_cols.items():
            if col not in df.columns:
                continue
            pct_rank = df[col].rank(pct=True, na_option="bottom")
            if weight < 0:
                # Invert: low completion_rate should rank high on the risk score
                pct_rank = 1.0 - pct_rank
            score += pct_rank * abs(weight)

        score /= total_weight
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

    # ------------------------------------------------------------------
    # Main pipeline
    # ------------------------------------------------------------------

    def process(
        self,
        project_keys: Optional[list[str]] = None,
        top_n_projects: Optional[int] = None,
        output_filename: str = "jira_projects.csv",
    ) -> pd.DataFrame:
        if project_keys:
            projects = project_keys
        elif top_n_projects:
            projects = self.get_project_list(top_n=top_n_projects)
        else:
            projects = self.get_project_list()

        project_set = set(projects)
        logger.info(
            f"Processing {len(projects)} projects: "
            f"{projects[:10]}{'...' if len(projects) > 10 else ''}"
        )

        # Step 1 — issue metrics (pandas, categorical dtypes, no text columns)
        issues = self._load_issue_metrics(project_set)
        gc.collect()

        # Step 2 — descriptions (csv.reader stream, bounded)
        descriptions = self._stream_descriptions(project_set)
        gc.collect()

        # Step 3 — comments (csv.reader stream, bounded, full text)
        comments = self._stream_comments(project_set)
        gc.collect()

        # Step 4 — changelog (csv.reader stream, counters only)
        changelog = self._stream_changelog_stats(project_set)
        gc.collect()

        # Step 5 — aggregate
        df = self._aggregate(issues, comments, changelog, descriptions)
        del issues, comments, changelog, descriptions
        gc.collect()

        # Save
        output_path = self.output_dir / output_filename
        df.to_csv(output_path, index=False)
        logger.info(f"Saved to {output_path}")

        sample_path = self.output_dir / "jira_projects_sample.csv"
        df.head(20).to_csv(sample_path, index=False)
        logger.info(f"Saved sample (20 projects) to {sample_path}")

        return df


def main():
    parser = argparse.ArgumentParser(
        description="Preprocess Apache JIRA data for PRISM",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--sample", "-s", type=int, default=None, help="Top N projects by issue count"
    )
    parser.add_argument(
        "--projects", "-p", type=str, default=None, help="Comma-separated project keys"
    )
    parser.add_argument(
        "--output", "-o", type=str, default="jira_projects.csv", help="Output filename"
    )
    parser.add_argument("--raw-dir", type=str, default=None)
    parser.add_argument("--output-dir", type=str, default=None)
    args = parser.parse_args()

    project_keys = [p.strip().upper() for p in args.projects.split(",")] if args.projects else None

    preprocessor = JiraDataPreprocessor(raw_data_dir=args.raw_dir, output_dir=args.output_dir)
    df = preprocessor.process(
        project_keys=project_keys,
        top_n_projects=args.sample,
        output_filename=args.output,
    )

    print("\n" + "=" * 60)
    print("PREPROCESSING COMPLETE")
    print("=" * 60)
    print(f"Projects processed: {len(df)}")
    print(f"Risk distribution:\n{df['risk_level'].value_counts().to_string()}")
    print(f"\nSample projects:")
    print(
        df[
            [
                "project_id",
                "project_name",
                "total_issues",
                "completion_rate",
                "avg_resolution_days",
                "defect_rate",
                "risk_score_composite",
                "risk_level",
            ]
        ]
        .head(10)
        .to_string()
    )
    print("=" * 60)


if __name__ == "__main__":
    main()
