# PRISM Notebooks

Jupyter notebooks for exploratory analysis, model development, and results
visualisation. Run them **in order** — each notebook builds on artefacts
produced by the previous ones.

## Prerequisites

```bash
# Activate virtual environment
source venv/bin/activate

# Install dependencies (includes Jupyter)
pip install -r requirements.txt

# Optional: set your OpenAI API key for notebook 04
# Copy env_template.txt → .env and fill in OPENAI_API_KEY
cp env_template.txt .env
```

Before running any notebook, generate the processed data file:

```bash
python scripts/preprocess_jira_data.py
# Output: data/processed/jira_projects.csv
```

## Notebook Order

### 01 — Data Exploration (`01_data_exploration.ipynb`)
**Purpose**: Understand the raw JIRA data and the preprocessed project-level
summary before any modelling.

- Inspects the four raw JIRA exports (`issues.csv`, `changelog.csv`,
  `comments.csv`, `issuelinks.csv`) for schema and file size
- Loads `data/processed/jira_projects.csv` and reviews feature distributions,
  missing values, and zero-variance columns
- Analyses the risk-level target variable and its correlations with all
  numerical features
- Measures raw text lengths for `status_comments` and `project_description`
  as a context-window budget for the LLM notebook
- Performs word-frequency analysis by risk tier to identify vocabulary signals

**Output**: No files saved — exploration only.

---

### 02 — Feature Engineering (`02_feature_engineering.ipynb`)
**Purpose**: Apply and validate the 12 derived features computed by
`src/data/feature_engineer.py`.

- Runs `FeatureEngineer` on the raw JIRA project data
- Visualises SPI distributions, productivity per person, and team metrics
  by risk level to confirm each feature's discriminatory power
- Produces correlation plots (feature ↔ risk, inter-feature) to flag
  multicollinearity before ML training
- Validates binary indicator flags (`is_behind_schedule`, `is_high_complexity`)
- Prints feature-selection recommendations based on Pearson |r| thresholds
- Saves the engineered DataFrame to `data/processed/projects_with_features.csv`

**Output**: `data/processed/projects_with_features.csv`

---

### 03 — ML Modeling (`03_ml_modeling.ipynb`)
**Purpose**: Train, compare, and evaluate machine learning models for binary
risk classification (High vs Low/Medium).

- Loads `jira_projects.csv` and re-runs `FeatureEngineer` (keeping this
  notebook self-contained)
- Builds the feature matrix from `feature_names.json` if available, otherwise
  falls back to a hand-curated feature list with `reopen_rate` capping
- Stratified 70/30 train–test split
- Initial training with Random Forest via `MLTrainer`; model comparison across
  six algorithms (Random Forest, XGBoost, LightGBM, Logistic Regression,
  Extra Trees, SVC)
- Retrains with the best-performing model type
- Evaluation: confusion matrix, ROC curve, classification report, and
  misclassification analysis (false negatives highlighted as highest priority)
- Feature importance analysis with correct formatting for tree-based vs
  normalised importances
- Saves `best_model.pkl` and `feature_names.json` for downstream notebooks

**Output**: `models/ml/best_model.pkl`, `models/ml/feature_names.json`

---

### 04 — LLM Experiments (`04_llm_experiments.ipynb`)
**Purpose**: Analyse project risk using GPT-based sentiment and category
extraction, and compare LLM signals against ML predictions.

- Requires `OPENAI_API_KEY` in `.env`; falls back to mock data automatically
  if the key is absent
- Single project demo: `LLMAnalyzer.analyze_project()` returns sentiment score,
  risk level, risk categories, and a one-sentence summary
- Batch analysis of 50 stratified projects (17 per risk tier) via
  `analyze_batch()` with built-in rate limiting and NaN handling
- `RiskExtractor` normalises raw LLM dicts into typed `RiskAnalysis` dataclasses
- Summary statistics and three-panel visualisation (sentiment histogram, risk
  distribution bar, category bar)
- ML vs LLM agreement analysis: 42% agreement rate on the 50-project batch;
  disagreements are printed for manual review
- Prompt engineering demo: shows how injecting quantitative metrics as
  `additional_context` increases LLM confidence and category precision

**Output**: No files saved — results are hard-coded into notebook 07's
`LLM_SENTIMENT` dict to avoid re-calling the API on every run.

---

### 05 — MCDA Analysis (`05_mcda_analysis.ipynb`)
**Purpose**: Rank all 640 projects using TOPSIS multi-criteria decision
analysis, combining ML risk scores with JIRA-derived quality metrics.

- Generates ML risk scores from the saved `best_model.pkl`
- Default weight profile: `blocker_ratio` 40%, `defect_rate` 30%,
  `reopen_rate` 15%, `ml_risk_score` 10%, `sentiment_score` 5%
- Compares four stakeholder weight profiles (default, technical, quality,
  schedule) and measures Jaccard overlap between their top-20 riskiest sets
- Sensitivity analysis: each criterion perturbed ±10% to identify the most
  influential weight
- Visualisations: MCDA score distribution, top-15 riskiest projects bar chart
- ML vs MCDA risk-label cross-tabulation and agreement analysis
- Percentile-based thresholds for MCDA risk classification (bottom third =
  High, top third = Low) to produce a balanced 33/33/33 distribution

**Output**: No files saved — rankings are re-computed in notebooks 06 and 07.

---

### 06 — Hybrid Evaluation (`06_hybrid_evaluation.ipynb`)
**Purpose**: End-to-end pipeline validation — load the saved model, run MCDA,
and quantify the complementary value of the hybrid approach.

- Loads `best_model.pkl` and `feature_names.json`; falls back to retraining
  only if the files are missing
- Applies `reopen_rate` capping and `sentiment_score = 0.0` default before
  prediction
- Re-runs MCDA with `ml_risk_score` as input (prevents flat TOPSIS scores)
- Side-by-side and overlap bar charts comparing ML-only vs hybrid risk rankings
- Business metrics: estimated intervention costs for false negatives (missed
  High-risk projects) vs false positives under both approaches
- Documents the "bimodal ML ranking problem": the model assigns extreme
  probabilities to most projects, making the top-N selection arbitrary — MCDA
  breaks ties using continuous, multi-dimensional criteria

**Output**: No files saved — validation only.

---

### 07 — Results Visualization (`07_results_visualization.ipynb`)
**Purpose**: Produce export-ready Plotly HTML charts for the full portfolio.

- Runs the complete pipeline inline (FeatureEngineer → ML scoring → MCDA) to
  ensure all charts use consistent data
- Hard-coded `LLM_SENTIMENT` dict (49 real gpt-4.1 scores from notebook 04,
  run 2026-03-22); remaining 591 projects default to 0.0 (neutral)
- Charts saved to `reports/`:
  - `risk_distribution.html` — MCDA risk-tier donut chart
  - `risk_score_bar.html` — top-15 riskiest projects by MCDA score
  - `feature_importance.html` — top-15 ML features by split-count importance
  - `portfolio_risk_radar.html` — five JIRA risk dimensions as portfolio averages
  - `comparison_radar_risky.html` / `comparison_radar_safe.html` — normalised
    metric profiles for the top-3 riskiest vs top-3 safest projects
  - `sentiment_distribution.html` — per-project LLM sentiment for the 49
    analysed projects

**Output**: `reports/*.html`

---

## Data Flow

```
data/raw/
  issues.csv
  changelog.csv          ──► scripts/preprocess_jira_data.py
  comments.csv
  issuelinks.csv
                         ──► data/processed/jira_projects.csv
                                  │
                         01_data_exploration.ipynb (EDA)
                                  │
                         02_feature_engineering.ipynb
                                  │
                         03_ml_modeling.ipynb ──► models/ml/best_model.pkl
                                  │                            feature_names.json
                         04_llm_experiments.ipynb (LLM scores)
                                  │
                         05_mcda_analysis.ipynb (TOPSIS ranking)
                                  │
                         06_hybrid_evaluation.ipynb (pipeline validation)
                                  │
                         07_results_visualization.ipynb ──► reports/*.html
```

## Notes

- Notebooks are numbered for sequential execution; running them out of order
  will fail if upstream artefacts (`best_model.pkl`, `feature_names.json`) are
  missing
- All random operations use `random_state=42` for reproducibility
- Notebooks 05–07 re-run `FeatureEngineer` and load `best_model.pkl` rather
  than reading intermediate CSVs — this eliminates data-leakage risk from
  accidentally including derived columns as model inputs
- LLM API calls cost approximately $0.05–0.10 for the 50-project batch; the
  results are cached in notebook 07's `LLM_SENTIMENT` dict to avoid repeated
  charges
