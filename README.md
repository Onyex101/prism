# PRISM - Predictive Risk Intelligence for Software Management

**A Hybrid AI System Integrating Machine Learning and Large Language Models for Software Project Risk Analysis**

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![Tests](https://img.shields.io/badge/tests-88%20passed-brightgreen.svg)]()
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Status: Implementation Complete](https://img.shields.io/badge/Status-Implementation%20Complete-success.svg)]()

---

## 📋 Table of Contents

- [Overview](#overview)
- [Key Features](#key-features)
- [System Architecture](#system-architecture)
- [Project Structure](#project-structure)
- [Quick Start](#quick-start)
- [Data Sources](#data-sources)
- [Use Cases](#use-cases)
- [Technology Stack](#technology-stack)
- [Testing](#testing)
- [Documentation](#documentation)
- [License](#license)

---

## 🎯 Overview

PRISM (Predictive Risk Intelligence for Software Management) is an AI-powered system designed to help project managers identify and prioritize software project risks before they escalate into crises. Unlike traditional project management tools that only report current status, PRISM predicts future problems using a unique hybrid approach that combines:

- **Machine Learning (ML):** Analyzes structured project data (budgets, schedules, team metrics) to predict risk scores
- **Large Language Models (LLM):** Evaluates project comments and communications to extract risk indicators and sentiment
- **Multi-Criteria Decision Analysis (MCDA):** Ranks projects based on combined insights from ML, LLM, and performance factors

### The Problem

Software projects face alarmingly high failure rates:
- Only 29% succeed (on time, on budget, with required features)
- 52% are challenged (late, over budget, or missing features)
- 19% fail outright

Traditional risk management is reactive—problems are identified only after they appear in metrics. Project managers need predictive capabilities that forecast risks 2-4 weeks in advance.

### The Solution

PRISM provides:
- ✅ **Early Risk Detection:** Predicts high-risk projects before traditional metrics deteriorate
- ✅ **Hybrid Intelligence:** Combines quantitative metrics with qualitative team communications
- ✅ **Portfolio Prioritization:** Objectively ranks projects to focus manager attention
- ✅ **Explainable AI:** Natural language explanations via chat assistant
- ✅ **Actionable Insights:** Not just "what" but "why" and "what to do"

---

## ✨ Key Features

### 1. Machine Learning Risk Prediction
- **Ensemble models** (Random Forest, XGBoost) for risk classification
- **Feature engineering** from schedule performance, cost variance, velocity, and team metrics
- **SHAP explainability** showing which factors drive each project's risk score
- **Cross-validation** and hyperparameter tuning support

### 2. LLM-Powered Text Analysis
- **OpenAI GPT integration** to analyze project comments, status updates, team feedback
- **Risk extraction:** Identifies concerns, blockers, and warning signs in natural language
- **Sentiment analysis:** Detects team morale issues and stakeholder dissatisfaction
- **Risk categorization:** Technical, resource, schedule, and scope risks

### 3. MCDA Project Ranking
- **TOPSIS algorithm** ranks projects based on multiple weighted criteria:
  - ML risk score (40% weight)
  - LLM sentiment score (25%)
  - Schedule performance index (15%)
  - Cost performance index (10%)
  - Team stability (10%)
- **Configurable weights** via YAML configuration
- **Sensitivity analysis** validates ranking stability

### 4. Interactive Dashboard
- **Streamlit-based** web interface
- **File upload:** CSV/JSON from Jira, Azure DevOps, Monday.com, etc.
- **Visualizations:** Risk distribution charts, trend analysis, project comparisons
- **Export functionality:** Reports and data export

### 5. AI Chat Assistant
- **Natural language Q&A:** Ask questions about specific projects
- **Risk explanations:** "Why is Project X high risk?"
- **Recommendations:** Context-aware suggestions based on analysis results
- **Conversation memory:** Multi-turn dialogues

---

## 🏗️ System Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                         PRISM Dashboard                          │
│                      (Streamlit Interface)                       │
│  ┌─────────┐  ┌──────────┐  ┌─────────┐  ┌──────┐  ┌────────┐ │
│  │Dashboard│  │ ML       │  │   LLM   │  │MCDA  │  │  Chat  │ │
│  │Overview │  │ Analysis │  │Insights │  │Ranks │  │Assist. │ │
│  └─────────┘  └──────────┘  └─────────┘  └──────┘  └────────┘ │
└──────────────────────┬──────────────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────────────┐
│                   Core Processing Modules                        │
├──────────────────┬────────────────────┬──────────────────────────┤
│  ML Module       │   LLM Module       │   MCDA Module            │
│ ┌──────────────┐ │ ┌────────────────┐ │ ┌──────────────────────┐ │
│ │Random Forest │ │ │OpenAI GPT API  │ │ │TOPSIS Ranking        │ │
│ │XGBoost       │ │ │Risk Extraction │ │ │Criteria Weighting    │ │
│ │SHAP Explainer│ │ │Sentiment Score │ │ │Sensitivity Analysis  │ │
│ └──────────────┘ │ └────────────────┘ │ └──────────────────────┘ │
└──────────────────┴────────────────────┴──────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────────────┐
│              Data Processing & Feature Engineering               │
│  ┌──────────┐  ┌───────────┐  ┌──────────┐  ┌──────────────┐  │
│  │Data      │  │Validation │  │Feature   │  │Preprocessing │  │
│  │Loader    │  │& Cleaning │  │Engineer  │  │Pipeline      │  │
│  └──────────┘  └───────────┘  └──────────┘  └──────────────┘  │
└─────────────────────────────────────────────────────────────────┘
```

---

## 📁 Project Structure

```
prism/
├── app/                          # Streamlit application
│   ├── app.py                    # Main entry point
│   └── pages/                    # Dashboard pages
│       ├── 1_📊_Dashboard.py
│       ├── 2_📁_Upload_Data.py
│       ├── 3_🤖_ML_Analysis.py
│       ├── 4_💬_LLM_Insights.py
│       ├── 5_📈_Rankings.py
│       ├── 6_🔍_Compare_Projects.py
│       └── 7_💭_Chat_Assistant.py
│
├── src/                          # Core source code
│   ├── data/                     # Data processing modules
│   │   ├── loader.py             # Data loading (CSV, JSON, Excel)
│   │   ├── validator.py          # Data validation
│   │   ├── preprocessor.py       # Preprocessing pipeline
│   │   ├── feature_engineer.py   # Feature creation
│   │   └── generator.py          # Synthetic data (for testing)
│   │
│   ├── models/                   # ML and LLM models
│   │   ├── ml/                   # Machine learning
│   │   │   ├── trainer.py        # Model training
│   │   │   ├── predictor.py      # Predictions
│   │   │   └── evaluator.py      # Model evaluation
│   │   └── llm/                  # Large language models
│   │       ├── analyzer.py       # LLM text analysis
│   │       └── risk_extractor.py # Risk extraction
│   │
│   ├── mcda/                     # Multi-criteria decision analysis
│   │   ├── topsis.py             # TOPSIS algorithm
│   │   └── ranker.py             # Project ranking
│   │
│   ├── explainability/           # Model interpretation
│   │   └── shap_explainer.py     # SHAP explanations
│   │
│   ├── chat/                     # Chat assistant
│   │   └── assistant.py          # Conversational AI
│   │
│   ├── visualization/            # Charts and visualizations
│   │   └── risk_charts.py        # Plotly charts
│   │
│   └── utils/                    # Utilities
│       ├── logger.py             # Logging configuration
│       └── metrics.py            # Portfolio metrics
│
├── config/                       # Configuration files
│   ├── settings.py               # Application settings
│   ├── mcda_config.yaml          # MCDA weights
│   ├── model_config.yaml         # ML model parameters
│   └── llm_config.yaml           # LLM settings
│
├── data/                         # Data directory
│   ├── raw/                      # Raw project data files
│   ├── processed/                # Processed data
│   └── schemas/                  # Validation schemas
│
├── tests/                        # Test suite
│   ├── unit/                     # Unit tests
│   └── integration/              # Integration tests
│
├── scripts/                      # Utility scripts
│   ├── preprocess_jira_data.py   # Jira data preprocessing
│   └── train_models.py           # Model training script
│
├── docs/                         # Documentation
│   └── academic/                 # Academic chapters
│
├── requirements.txt              # Python dependencies
├── pyproject.toml                # Project configuration
├── Makefile                      # Build commands
└── README.md                     # This file
```

---

## 🚀 Quick Start

### Prerequisites

- Python 3.10 or higher
- pip package manager
- OpenAI API key ([get one here](https://platform.openai.com/api-keys)) - for LLM features
- macOS users: `brew install libomp` (for XGBoost support)

### Installation

```bash
# 1. Clone the repository
git clone https://github.com/yourusername/prism.git
cd prism

# 2. Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Configure environment (optional, for LLM features)
cp env_template.txt .env
# Edit .env and add your OPENAI_API_KEY=sk-...

# 5. Run tests to verify installation
pytest tests/

# 6. Launch dashboard
streamlit run app/app.py
```

### First-Time Usage

1. **Open your browser** to `http://localhost:8501`
2. **Click "Load Sample Data"** on the home page (or upload your own CSV)
3. **Navigate to different pages** to explore ML Analysis, LLM Insights, Rankings
4. **Use the Chat Assistant** to ask questions about your projects

---

## 📦 Data Sources

### Project Data Location

Real project data should be placed in `data/raw/`. The system supports:

- **CSV files** - Recommended format
- **JSON files** - For structured data
- **Excel files** - .xlsx format

### Required Fields

| Field | Type | Description |
|-------|------|-------------|
| `project_id` | Text | Unique identifier |
| `project_name` | Text | Project name |
| `start_date` | Date | Start date (YYYY-MM-DD) |
| `planned_end_date` | Date | Target end date |
| `budget` | Number | Total budget |
| `spent` | Number | Current spend |
| `planned_hours` | Number | Estimated hours |
| `actual_hours` | Number | Logged hours |
| `team_size` | Number | Team members |
| `completion_rate` | Number | Percent complete (0-100) |
| `status` | Text | Current status |
| `status_comments` | Text | Project updates (for LLM analysis) |

### Jira Data Integration

Use the preprocessing script to convert Jira exports:

```bash
# Process Jira data export
python scripts/preprocess_jira_data.py --input data/raw/jira_export.csv

# Process specific projects
python scripts/preprocess_jira_data.py --projects SPARK,KAFKA,FLINK
```

---

## 💼 Use Cases

### Portfolio Health Check
Upload all project data, view risk distribution, identify top concerns, and generate reports for stakeholder meetings.

### Single Project Deep-Dive
Analyze a specific project to understand risk drivers through ML feature importance and LLM-extracted concerns from team comments.

### Trend Analysis
Compare multiple time periods to assess if portfolio health is improving or degrading over time.

See [USER_APPLICATION_GUIDE.md](USER_APPLICATION_GUIDE.md) for detailed workflows and examples.

---

## 🛠️ Technology Stack

| Component | Technology | Purpose |
|-----------|-----------|---------|
| **Language** | Python 3.10+ | Core development |
| **ML Framework** | scikit-learn, XGBoost | Model training and prediction |
| **Interpretability** | SHAP | Model explanations |
| **LLM API** | OpenAI GPT | Text analysis |
| **MCDA** | Custom TOPSIS | Project ranking |
| **Dashboard** | Streamlit | Web interface |
| **Visualization** | Plotly | Interactive charts |
| **Data** | pandas, numpy | Data manipulation |
| **Testing** | pytest | Test framework |
| **Logging** | loguru | Application logging |

---

## 🧪 Testing

The project includes comprehensive tests:

```bash
# Run all tests
pytest tests/

# Run with verbose output
pytest tests/ -v

# Run specific test file
pytest tests/unit/test_mcda.py

# Run with coverage
pytest tests/ --cov=src
```

**Current Status:** 88 tests passing, 6 skipped (XGBoost environment dependency)

---

## 📚 Documentation

| Document | Description |
|----------|-------------|
| [USER_APPLICATION_GUIDE.md](USER_APPLICATION_GUIDE.md) | Complete user manual with workflows |
| [PROJECT_STRATEGY.md](PROJECT_STRATEGY.md) | Implementation strategy and planning |
| [DELIVERABLES_SUMMARY.md](DELIVERABLES_SUMMARY.md) | Project deliverables overview |
| [docs/academic/](docs/academic/) | Academic chapters and citations |
| [data/README.md](data/README.md) | Data schema and preparation guide |

---

## 📄 License

This project is licensed under the **MIT License** - see the [LICENSE](LICENSE) file for details.

---

## 🙏 Acknowledgments

- OpenAI for GPT API
- scikit-learn and XGBoost communities
- Streamlit for the dashboard framework
- SHAP library for model interpretability

---

<div align="center">

**Built with Python, scikit-learn, OpenAI GPT, and Streamlit**

*Transforming software project management from reactive to predictive*

</div>
