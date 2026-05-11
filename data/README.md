# PRISM Data Directory

This directory contains all data files used by the PRISM system.

## Directory Structure

```
data/
├── raw/                    # Original, unprocessed data files
│   ├── issues.csv          # Apache JIRA issues (18.5M records, 1.8GB)
│   ├── comments.csv        # Issue comments (62.4M records, 3.8GB)
│   ├── changelog.csv       # Issue history (40.5M records, 2.5GB)
│   └── issuelinks.csv      # Issue dependencies (390K records, 99MB)
├── processed/              # Cleaned and transformed data
│   ├── jira_projects.csv           # Full aggregated project-level data
│   ├── jira_projects_sample.csv    # First 20 projects (quick testing)
│   ├── risk_snapshots.csv          # Timestamped risk snapshots (auto-generated)
│   └── sus_responses.csv           # SUS usability survey responses (auto-generated)
├── synthetic/              # Generated test data
├── external/               # Third-party datasets
└── schemas/                # Data schema definitions
    ├── project_schema.json # JSON schema for validation
    └── validation_rules.yaml
```

## Primary Dataset: Apache JIRA Issues

### Source

**Kaggle Dataset:** [Apache JIRA Issues: Changelog, Comments, Links](https://www.kaggle.com/datasets/tedlozzo/apaches-jira-issues)

### Dataset Statistics

| Metric | Value |
|--------|-------|
| Total Issues | 18.5 million |
| Total Comments | 62.4 million |
| Changelog Entries | 40.5 million |
| Issue Links | 390,000 |
| Projects | 640 |
| Date Range | 2000 - Present |
| Total Size | ~8 GB |

### Top Projects by Issue Count

1. **SPARK** - 50,000 issues
2. **FLINK** - 37,000 issues
3. **FLEX** - 35,000 issues
4. **HBASE** - 29,000 issues
5. **HIVE** - 29,000 issues
6. **AMBARI** - 26,000 issues
7. **IGNITE** - 24,000 issues
8. **CAMEL** - 21,000 issues
9. **CASSANDRA** - 20,000 issues
10. **KAFKA** - 17,000 issues

## Download Instructions

### Option 1: Kaggle CLI (Recommended)

```bash
pip install kaggle

# Download kaggle.json from https://www.kaggle.com/settings
mkdir -p ~/.kaggle
mv ~/Downloads/kaggle.json ~/.kaggle/
chmod 600 ~/.kaggle/kaggle.json

cd /path/to/prism
kaggle datasets download -d tedlozzo/apaches-jira-issues -p data/raw/

cd data/raw
unzip apaches-jira-issues.zip
rm apaches-jira-issues.zip
```

### Option 2: Manual Download

1. Visit [https://www.kaggle.com/datasets/tedlozzo/apaches-jira-issues](https://www.kaggle.com/datasets/tedlozzo/apaches-jira-issues)
2. Click "Download" (requires Kaggle account)
3. Extract the ZIP file to `data/raw/`
4. Verify you have these files:
   - `issues.csv`
   - `comments.csv`
   - `changelog.csv`
   - `issuelinks.csv`

## Data Preprocessing

After downloading, run the preprocessing script to convert issue-level data to project-level metrics:

```bash
cd /path/to/prism

# Option A: Top 50 projects (recommended for testing, ~10 min)
python scripts/preprocess_jira_data.py --sample 50

# Option B: Specific projects
python scripts/preprocess_jira_data.py --projects SPARK,KAFKA,FLINK,HADOOP,HIVE

# Option C: All 640 projects
python scripts/preprocess_jira_data.py

# Check the output
ls -lh data/processed/
```

### How the Preprocessing Works

The script processes the four raw JIRA files in a single pass each, staying within memory limits regardless of dataset size:

| Phase | File | Method | Memory |
|-------|------|--------|--------|
| Issue metrics | `issues.csv` | pandas chunked, categorical dtypes | ~130 MB for 50 projects |
| Descriptions | `issues.csv` | csv.reader stream, max 20 per project | Bounded |
| Comments | `comments.csv` | csv.reader stream, max 500 per project (`MAX_COMMENTS_PER_PROJECT`) | Bounded |
| Changelog | `changelog.csv` | csv.reader stream, integer counters only | Near zero |

### Preprocessing Output

The script creates `data/processed/jira_projects.csv` with one row per project:

| Column | Description |
|--------|-------------|
| `project_id` | Project key (e.g., SPARK, KAFKA) |
| `project_name` | Full project name |
| `total_issues` | Total issue count |
| `open_issues` | Currently open issues |
| `closed_issues` | Resolved/closed issues |
| `bug_count` | Number of bug-type issues |
| `feature_count` | New feature issues |
| `improvement_count` | Improvement issues |
| `task_count` | Task issues |
| `blocker_count` | Blocker priority issues |
| `critical_count` | Critical priority issues |
| `blocker_ratio` | Blockers / total issues |
| `critical_ratio` | Critical / total issues |
| `planned_end_date` | Latest issue creation date (last point the project was actively planned) |
| `actual_end_date` | Latest resolution date (completed projects only; `null` if still active) |
| `planned_hours` | Estimated (issues × 8h) |
| `actual_hours` | Derived from open/closed counts |
| `team_size` | Unique assignees + creators |
| `unique_assignees` | Distinct assignees |
| `unique_reporters` | Distinct reporters |
| `completion_rate` | % of issues resolved |
| `velocity` | Issues resolved per month |
| `defect_rate` | Bugs / total issues |
| `avg_resolution_days` | Mean days from open → resolved |
| `median_resolution_days` | Median resolution time |
| `reopen_count` | Total issue reopens |
| `reopen_rate` | Reopens / closed issues |
| `status_changes` | Total status field changes |
| `churn_rate` | Status changes / total issues |
| `project_duration_days` | First issue → last update |
| `total_votes` | Sum of issue votes |
| `total_watchers` | Sum of issue watchers |
| `complexity_score` | Derived (team size + duration, 1–10) |
| `risk_level` | Derived: High / Medium / Low |
| `risk_score_composite` | Composite score (0–1, higher = riskier) |
| `status_comments` | Sampled comments for LLM (full text) |
| `project_description` | Sampled issue descriptions for LLM |

### Risk Label Methodology

Risk labels are derived using a **percentile-based composite score** so that results are meaningful across any project portfolio — whether corporate sprints or long-running open source repositories.

Six indicators are percentile-ranked within the dataset, weighted, and averaged into a composite score:

| Indicator | Weight | Direction |
|-----------|--------|-----------|
| `avg_resolution_days` | 1.0 | Higher = riskier |
| `reopen_rate` | 1.0 | Higher = riskier |
| `blocker_ratio` | 1.0 | Higher = riskier |
| `defect_rate` | 1.0 | Higher = riskier |
| `churn_rate` | 0.5 | Higher = riskier |
| `completion_rate` | 1.0 (inverted) | Lower = riskier |

Projects are then split into thirds:
- **High** — top 33% composite score
- **Medium** — middle 33%
- **Low** — bottom 33%

This ensures a balanced distribution regardless of sample size or the absolute scale of the metrics.

## Auto-Generated Files in `processed/`

### `risk_snapshots.csv`

Created and appended to automatically every time the **ML Analysis** (page 3) or **Rankings** (page 5) page completes a run. Stores one row per project per run.

| Column | Description |
|--------|-------------|
| `snapshot_id` | UUID shared by all rows in the same run |
| `timestamp` | ISO-8601 UTC timestamp of the run |
| `source` | `"ml"` (ML Analysis) or `"mcda"` (Rankings) |
| `project_id` | Project key or identifier |
| `project_name` | Project display name |
| `risk_score` | ML risk score (0–1); NaN for MCDA-only runs |
| `risk_level` | ML risk level (High/Medium/Low); empty for MCDA-only runs |
| `mcda_score` | TOPSIS MCDA score; NaN for ML-only runs |
| `mcda_risk_level` | MCDA risk level; empty for ML-only runs |

Used by the **Risk Trends** page (page 8) to plot per-project risk evolution over time and detect escalating risk patterns.

### `sus_responses.csv`

Appended to when a respondent submits the **Usability Survey** (page 9).

| Column | Description |
|--------|-------------|
| `timestamp` | ISO-8601 UTC submission time |
| `respondent` | Name or alias (optional, defaults to "Anonymous") |
| `role` | Respondent role (optional) |
| `q1`–`q10` | Raw SUS responses (1 = Strongly Disagree, 5 = Strongly Agree) |
| `sus_score` | Calculated SUS score (0–100) |
| `grade` | Letter grade + adjective (e.g., "B – Good") |

Use the **Export Responses to CSV** button on the Usability Survey page to download this file for thesis documentation. This file is **not** committed to version control (see `.gitignore`).

## Data Privacy & Gitignore

Large data files are **excluded from git** (see `.gitignore`):

```
data/raw/*.csv        # Ignored
data/raw/*.json       # Ignored
data/processed/*.csv  # Ignored
data/external/*       # Ignored
```

## Schema Validation

Project data should conform to `schemas/project_schema.json`. Validate using:

```python
from src.data.validator import DataValidator

validator = DataValidator()
is_valid, errors = validator.validate(df)
```

## License

The Apache JIRA dataset is sourced from publicly available Apache Foundation issue trackers. Usage is subject to Kaggle's terms of service.
