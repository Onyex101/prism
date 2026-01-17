# PRISM Data Directory

This directory contains all data files used by the PRISM system.

## Directory Structure

```
data/
├── raw/                    # Original, unprocessed data files
│   ├── issues.csv          # Apache JIRA issues (18.5M records, 1.8GB)
│   ├── comments.csv        # Issue comments (62.4M records, 3.8GB)
│   ├── changelog.csv       # Issue history (40.5M records, 2.5GB)
│   ├── issuelinks.csv      # Issue dependencies (390K records, 99MB)
│   └── sample_projects.csv # Small sample for testing (15 records)
├── processed/              # Cleaned and transformed data
│   ├── jira_projects.csv   # Aggregated project-level data
│   └── jira_projects_sample.csv  # Sample for quick testing
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
# Install Kaggle CLI if not installed
pip install kaggle

# Configure Kaggle API credentials
# Download kaggle.json from https://www.kaggle.com/settings
mkdir -p ~/.kaggle
mv ~/Downloads/kaggle.json ~/.kaggle/
chmod 600 ~/.kaggle/kaggle.json

# Download the dataset
cd /path/to/prism
kaggle datasets download -d tedlozzo/apaches-jira-issues -p data/raw/

# Unzip the files
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
# Navigate to project root
cd /path/to/prism

# Option A: Process top 50 projects (recommended for testing, ~5 min)
python scripts/preprocess_jira_data.py --sample 50

# Option B: Process specific projects
python scripts/preprocess_jira_data.py --projects SPARK,KAFKA,FLINK,HADOOP,HIVE

# Option C: Process all 640 projects (takes ~30-60 min)
python scripts/preprocess_jira_data.py

# Check the output
ls -la data/processed/
```

### Preprocessing Output

The script creates `data/processed/jira_projects.csv` with these columns:

| Column | Description |
|--------|-------------|
| `project_id` | Project key (e.g., SPARK, KAFKA) |
| `project_name` | Full project name |
| `total_issues` | Total issue count |
| `open_issues` | Currently open issues |
| `closed_issues` | Resolved issues |
| `bug_count` | Number of bugs |
| `blocker_count` | Critical blocker issues |
| `completion_rate` | % of issues resolved |
| `velocity` | Issues resolved per month |
| `defect_rate` | Bugs / total issues |
| `avg_resolution_days` | Mean time to resolve |
| `reopen_rate` | Issue reopen frequency |
| `team_size` | Unique contributors |
| `risk_level` | Derived: High/Medium/Low |
| `status_comments` | Sampled comments (for LLM) |
| `project_description` | Combined issue descriptions |

## Data Privacy & Gitignore

Large data files are **excluded from git** (see `.gitignore`):

```
data/raw/*.csv        # Ignored (except sample_projects.csv)
data/raw/*.json       # Ignored
data/processed/*.csv  # Ignored
data/external/*       # Ignored
```

Only `sample_projects.csv` is tracked in git for testing purposes.

## Sample Data

For quick testing without downloading the full dataset, use:

```python
from src.data.loader import DataLoader

loader = DataLoader()
df = loader.load_sample_data()  # Loads data/raw/sample_projects.csv
print(df.head())
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

