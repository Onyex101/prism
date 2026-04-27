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

import pandas as pd
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from loguru import logger

from src.data.jira_aggregator import JiraAggregator

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
        aggregator = JiraAggregator(
            max_comments_per_project=self.MAX_COMMENTS_PER_PROJECT,
            max_descriptions_per_project=self.MAX_DESCRIPTIONS_PER_PROJECT,
        )
        df = aggregator.aggregate(issues, comments, changelog, descriptions)
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
