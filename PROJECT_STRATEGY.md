# PRISM Project Strategy & Implementation Plan

## Project Overview

**Title:** A Hybrid Predictive Framework Integrating Machine Learning and Language Models for Software Project Risk Analysis

**Timeline:** 3-4 months (Capstone Project)

**Goal:** Develop an AI-based system that predicts software project performance using a hybrid ML+LLM approach with MCDA ranking and interactive visualization.

---

## 1.1 Development Methodology

### Agile Approach Adapted for Capstone Timeline

**Sprint Structure:**
- **Sprint Duration:** 2 weeks
- **Total Sprints:** 6-8 sprints over 3-4 months
- **Sprint Ceremonies:**
  - Weekly progress reviews (self-assessment)
  - Bi-weekly advisor check-ins
  - End-of-sprint demos (working increments)

### Project Phases & Milestones

#### Phase 1: Foundation & Setup (Weeks 1-2)
**Deliverables:**
- ✅ Project repository setup with monorepo structure
- ✅ Development environment configuration
- ✅ Data schema design and sample dataset acquisition
- ✅ Technology stack integration testing
- ✅ Initial academic research and literature review

**Success Criteria:**
- All dependencies installed and tested
- At least 2 sample datasets loaded and validated
- Literature review outline completed with 15+ papers identified

---

#### Phase 2: Data Preparation & Exploration (Weeks 3-4)
**Deliverables:**
- ✅ Data preprocessing pipeline
- ✅ Exploratory data analysis (EDA) notebooks
- ✅ Data quality validation scripts
- ✅ Synthetic data generator for testing
- ✅ Feature engineering framework

**Success Criteria:**
- Clean, structured dataset with minimum 100 projects
- Feature correlation analysis completed
- Data pipeline handles CSV/JSON uploads
- Documentation of data quality metrics

---

#### Phase 3: ML Model Development (Weeks 5-7)
**Deliverables:**
- ✅ Baseline ML models (multiple algorithms)
- ✅ Feature selection and optimization
- ✅ Model evaluation framework
- ✅ Hyperparameter tuning pipeline
- ✅ Model persistence and versioning

**Success Criteria:**
- Minimum 70% accuracy on test set
- Model comparison with at least 4 algorithms
- Cross-validation implemented
- Model interpretability analysis (SHAP/LIME)

**Target ML Algorithms:**
1. Random Forest (primary)
2. Gradient Boosting (XGBoost/LightGBM)
3. Support Vector Regression
4. Neural Network (simple MLP)

---

#### Phase 4: LLM Integration (Weeks 8-9)
**Deliverables:**
- ✅ OpenAI API integration
- ✅ Prompt engineering for risk detection
- ✅ Comment sentiment analysis pipeline
- ✅ Risk indicator extraction
- ✅ LLM output validation and scoring

**Success Criteria:**
- Reliable risk categorization from project comments
- Response time < 3 seconds per project
- Structured output parsing (JSON format)
- Cost optimization (token usage monitoring)

**LLM Tasks:**
1. Extract risk indicators from project comments
2. Sentiment analysis (positive/negative/neutral)
3. Categorize risks (technical, resource, schedule, scope)
4. Generate natural language risk summaries

---

#### Phase 5: MCDA Implementation (Weeks 10-11)
**Deliverables:**
- ✅ MCDA algorithm implementation (TOPSIS or AHP)
- ✅ Criteria weighting mechanism
- ✅ Project ranking engine
- ✅ Sensitivity analysis for ranking stability
- ✅ Integration of ML + LLM outputs into MCDA

**Success Criteria:**
- Consistent ranking across multiple runs
- Configurable criteria weights
- Top 3 risk projects correctly identified
- Mathematical validation of MCDA calculations

**MCDA Criteria:**
1. ML risk score (40% weight)
2. LLM sentiment score (25% weight)
3. Schedule performance index (15% weight)
4. Budget variance (10% weight)
5. Resource utilization (10% weight)

---

#### Phase 6: Dashboard Development (Weeks 12-13)
**Deliverables:**
- ✅ Streamlit dashboard with multiple views
- ✅ Data upload interface
- ✅ Risk visualization charts
- ✅ Project comparison tables
- ✅ Export functionality (PDF/CSV)

**Success Criteria:**
- Intuitive UI validated with 3+ test users
- Load time < 5 seconds for 100 projects
- Responsive design (works on laptop screens)
- Error handling for invalid data

**Dashboard Pages:**
1. **Home:** Overview and quick stats
2. **Upload:** Data import interface
3. **Analysis:** ML predictions and LLM insights
4. **Rankings:** MCDA-based project prioritization
5. **Compare:** Side-by-side project comparison
6. **Chat Assistant:** AI-powered Q&A

---

#### Phase 7: Chat Assistant & Explanations (Week 14)
**Deliverables:**
- ✅ Chat interface for user queries
- ✅ Context-aware responses about predictions
- ✅ Drill-down explanations for risk scores
- ✅ Recommendation generation

**Success Criteria:**
- Natural conversation flow
- Accurate answers about specific projects
- Explanation of model decisions (SHAP integration)
- Response time < 5 seconds

---

#### Phase 8: Testing & Documentation (Weeks 15-16)
**Deliverables:**
- ✅ Unit tests for core modules
- ✅ Integration testing
- ✅ User acceptance testing
- ✅ Performance optimization
- ✅ Complete technical documentation
- ✅ User guide and demo videos
- ✅ Final academic report

**Success Criteria:**
- 80%+ code coverage for critical modules
- Zero critical bugs in core functionality
- Successful demo with stakeholders
- Academic report meets university standards

---

### Risk Mitigation Strategies

| Risk | Probability | Impact | Mitigation Strategy |
|------|-------------|--------|---------------------|
| **OpenAI API costs exceed budget** | Medium | High | - Implement response caching<br>- Use GPT-3.5-turbo for non-critical tasks<br>- Set hard spending limits<br>- Batch API requests |
| **Insufficient training data** | Medium | High | - Build synthetic data generator<br>- Use data augmentation techniques<br>- Leverage transfer learning<br>- Partner with industry for sample data |
| **ML model poor accuracy** | Low | High | - Test multiple algorithms<br>- Extensive feature engineering<br>- Use ensemble methods<br>- Adjust success criteria if needed |
| **LLM hallucinations/unreliable output** | Medium | Medium | - Structured output formats (JSON)<br>- Validation rules for responses<br>- Confidence scoring<br>- Fallback to rule-based extraction |
| **Timeline delays** | Medium | Medium | - Buffer time in final 2 weeks<br>- Prioritize MVP features first<br>- Parallel development where possible<br>- Weekly progress tracking |
| **Integration complexity** | Low | Medium | - Modular architecture<br>- API-first design<br>- Comprehensive testing at each phase<br>- Clear interface contracts |
| **Scope creep** | High | Medium | - Strict feature freeze after Phase 6<br>- Document "future work" instead<br>- Focus on core POC requirements<br>- Regular scope reviews with advisor |

---

## 1.2 Proof of Concept Approach

### POC Philosophy
**Focus:** Demonstrate core innovation (hybrid ML+LLM for risk prediction) rather than production-ready deployment.

**Design Principles:**
1. **Modularity:** Each component (ML, LLM, MCDA, UI) can be demonstrated independently
2. **Extensibility:** Architecture supports future API integration with PM tools
3. **Validation:** Clear metrics prove the hybrid approach outperforms single-method approaches
4. **Usability:** Project managers can use the system without technical training

### Phase Breakdown with Success Criteria

#### Phase 1: Data Preparation (20% of effort)
**Activities:**
- Design data schema compatible with Jira/Azure DevOps exports
- Acquire/generate 150-200 project records
- Create data validation pipeline
- Build EDA notebooks

**Success Criteria:**
- Dataset includes both structured metrics AND text comments
- Minimum 50 projects with "failed" or "at-risk" labels
- Data quality score > 90% (completeness, consistency)
- Documented data dictionary

**Deliverables:**
- `data_schema.json` - formal schema definition
- `sample_projects.csv` - curated dataset
- `data_generator.py` - synthetic data creation
- `eda_notebook.ipynb` - exploratory analysis

---

#### Phase 2: ML Model (25% of effort)
**Activities:**
- Feature engineering from structured data
- Train baseline models (4-5 algorithms)
- Hyperparameter optimization
- Model evaluation and selection
- Interpretability analysis

**Success Criteria:**
- Primary metric: ROC-AUC > 0.75 for risk classification
- Secondary metric: RMSE < 15% for completion rate prediction
- Model explainability using SHAP values
- Cross-validation score variance < 5%

**Deliverables:**
- `ml_pipeline.py` - training and prediction pipeline
- `model_comparison.ipynb` - algorithm benchmarking
- `best_model.pkl` - serialized production model
- `feature_importance.json` - ranked feature contributions

---

#### Phase 3: LLM Integration (20% of effort)
**Activities:**
- Design prompts for risk extraction from comments
- Implement OpenAI API calls with error handling
- Parse and structure LLM outputs
- Validate against human-labeled samples

**Success Criteria:**
- F1-score > 0.70 for risk category classification
- Agreement with human labels > 75%
- Average response time < 3 seconds per project
- API cost < $0.10 per project analysis

**Deliverables:**
- `llm_analyzer.py` - LLM integration module
- `prompts.yaml` - prompt templates
- `llm_evaluation.ipynb` - accuracy assessment
- `risk_taxonomy.json` - standardized risk categories

---

#### Phase 4: MCDA Integration (15% of effort)
**Activities:**
- Implement TOPSIS algorithm
- Define criteria and default weights
- Combine ML + LLM scores into MCDA framework
- Sensitivity analysis on weight variations

**Success Criteria:**
- Ranking stability: Kendall's tau > 0.8 with ±10% weight changes
- Top 5 high-risk projects include known problematic cases
- Computation time < 1 second for 100 projects
- Transparent scoring breakdown per project

**Deliverables:**
- `mcda_engine.py` - ranking algorithm
- `criteria_config.yaml` - configurable weights
- `ranking_validation.ipynb` - sensitivity analysis
- `ranking_explainer.py` - score breakdown generator

---

#### Phase 5: Dashboard Development (15% of effort)
**Activities:**
- Build Streamlit multi-page application
- Create visualizations (charts, tables, gauges)
- Implement file upload and data validation
- Design intuitive navigation and UX

**Success Criteria:**
- System Usability Scale (SUS) score > 70 from 5 test users
- All core features accessible within 3 clicks
- Visual design passes accessibility guidelines (contrast, fonts)
- Export functionality works for all result types

**Deliverables:**
- `main.py` - main Streamlit application
- `pages/` - individual page modules
- `components/` - reusable UI components
- `styles.css` - custom styling

---

#### Phase 6: Chat Assistant (5% of effort)
**Activities:**
- Implement conversational interface
- Connect chat to analysis results database
- Generate contextual explanations
- Add conversation memory

**Success Criteria:**
- Answers 90% of predefined test questions correctly
- Provides specific project details on request
- Explains model predictions in plain language
- Maintains conversation context for 5+ turns

**Deliverables:**
- `chat_assistant.py` - conversational AI module
- `chat_prompts.yaml` - system and user prompts
- `chat_memory.py` - conversation state management

---

### Minimum Viable Features for POC Validation

**Must Have (Core POC):**
1. ✅ CSV/JSON file upload for project data
2. ✅ ML model predicting project risk (classification)
3. ✅ LLM analyzing project comments for risk indicators
4. ✅ MCDA ranking combining both insights
5. ✅ Dashboard displaying top risk projects
6. ✅ Basic explanations for predictions

**Should Have (Enhanced POC):**
1. ✅ Multiple ML algorithms with comparison
2. ✅ Detailed risk categorization (technical, resource, schedule)
3. ✅ Interactive visualizations (charts, graphs)
4. ✅ Chat assistant for Q&A
5. ✅ Export results to PDF/CSV

**Nice to Have (Future Work):**
1. ⏸️ Real-time PM tool API integration
2. ⏸️ Historical trend analysis across time
3. ⏸️ Automated alerting system
4. ⏸️ Collaborative features (team annotations)
5. ⏸️ Mobile-responsive design

### Testing Checkpoints

**Checkpoint 1 (End of Week 4):**
- ✅ Data pipeline functional
- ✅ EDA reveals meaningful patterns
- **Gate:** Proceed only if data quality is sufficient

**Checkpoint 2 (End of Week 7):**
- ✅ ML model meets accuracy threshold
- ✅ Feature importance aligns with domain knowledge
- **Gate:** Proceed only if model is better than baseline

**Checkpoint 3 (End of Week 9):**
- ✅ LLM extracts risk indicators reliably
- ✅ Combined ML+LLM shows complementary value
- **Gate:** Validate hybrid approach superiority

**Checkpoint 4 (End of Week 13):**
- ✅ Dashboard fully functional
- ✅ End-to-end workflow tested
- **Gate:** UAT with 3-5 potential users

**Checkpoint 5 (End of Week 16):**
- ✅ All documentation complete
- ✅ Academic report ready
- **Gate:** Final advisor approval

---

## 1.3 Monorepo Folder Structure

```
prism/
├── README.md                          # Project overview, setup instructions, quick start
├── LICENSE                            # MIT or Academic License
├── .gitignore                         # Python, data files, API keys, models
├── requirements.txt                   # Python dependencies with versions
├── pyproject.toml                     # Poetry configuration (alternative)
├── .env.example                       # Template for environment variables
├── Makefile                           # Common commands (setup, test, run)
│
├── config/                            # Configuration files
│   ├── __init__.py
│   ├── settings.py                    # Application settings (paths, constants)
│   ├── logging_config.yaml            # Logging configuration
│   ├── model_config.yaml              # ML model hyperparameters
│   ├── mcda_config.yaml               # MCDA criteria and weights
│   └── ui_config.yaml                 # Dashboard themes and layouts
│
├── data/                              # Data storage (gitignored except samples)
│   ├── raw/                           # Original, immutable data
│   │   ├── sample_projects.csv        # Demo dataset (committed)
│   │   ├── jira_export_template.csv   # Template for Jira data
│   │   └── azure_export_template.json # Template for Azure DevOps
│   ├── processed/                     # Cleaned, transformed data
│   │   ├── train.csv
│   │   ├── test.csv
│   │   └── validation.csv
│   ├── synthetic/                     # Generated test data
│   │   └── generated_projects.csv
│   ├── external/                      # Third-party datasets
│   └── schemas/                       # Data schema definitions
│       ├── project_schema.json        # Required/optional fields
│       └── validation_rules.yaml      # Data quality rules
│
├── models/                            # Trained models (gitignored)
│   └── ml/                            # Machine learning models & feature_names.json
│       ├── random_forest_v1.pkl
│       ├── xgboost_v1.pkl
│       ├── best_model.pkl             # Production model
│       └── feature_names.json         # Feature columns aligned with training
│
├── src/                               # Source code
│   ├── __init__.py
│   │
│   ├── data/                          # Data processing modules
│   │   ├── __init__.py
│   │   ├── loader.py                  # Load CSV/JSON files
│   │   ├── validator.py               # Data quality checks
│   │   ├── preprocessor.py            # Cleaning and transformation
│   │   ├── feature_engineer.py        # Feature creation
│   │   └── generator.py               # Synthetic data generation
│   │
│   ├── models/                        # ML and LLM modules
│   │   ├── __init__.py
│   │   ├── ml/
│   │   │   ├── __init__.py
│   │   │   ├── random_forest.py       # RF implementation
│   │   │   ├── gradient_boosting.py   # XGBoost/LightGBM
│   │   │   ├── neural_network.py      # Simple MLP
│   │   │   ├── trainer.py             # Training pipeline
│   │   │   └── evaluator.py           # Model evaluation metrics
│   │   ├── llm/
│   │   │   ├── __init__.py
│   │   │   ├── openai_client.py       # OpenAI API wrapper
│   │   │   ├── prompt_builder.py      # Dynamic prompt generation
│   │   │   ├── risk_extractor.py      # Parse LLM responses
│   │   │   ├── sentiment_analyzer.py  # Sentiment scoring
│   │   │   └── cache.py               # Response caching
│   │   └── hybrid/
│   │       ├── __init__.py
│   │       ├── ensemble.py            # Combine ML + LLM predictions
│   │       └── calibration.py         # Score normalization
│   │
│   ├── mcda/                          # Multi-criteria decision analysis
│   │   ├── __init__.py
│   │   ├── topsis.py                  # TOPSIS algorithm
│   │   ├── ahp.py                     # Analytic Hierarchy Process
│   │   ├── criteria_manager.py        # Manage weights and criteria
│   │   ├── ranker.py                  # Project ranking engine
│   │   └── sensitivity.py             # Sensitivity analysis
│   │
│   ├── explainability/                # Model interpretation
│   │   ├── __init__.py
│   │   ├── shap_explainer.py          # SHAP values for ML
│   │   ├── lime_explainer.py          # LIME for local explanations
│   │   ├── feature_importance.py      # Feature contribution analysis
│   │   └── report_generator.py        # Natural language explanations
│   │
│   ├── chat/                          # Chat assistant
│   │   ├── __init__.py
│   │   ├── assistant.py               # Main chat logic
│   │   ├── context_manager.py         # Conversation memory
│   │   ├── query_handler.py           # Parse user questions
│   │   └── response_formatter.py      # Format answers
│   │
│   ├── visualization/                 # Plotting and charts
│   │   ├── __init__.py
│   │   ├── risk_charts.py             # Risk distribution plots
│   │   ├── comparison_plots.py        # Project comparison visuals
│   │   ├── trend_analysis.py          # Time series charts
│   │   └── dashboard_components.py    # Reusable Streamlit components
│   │
│   └── utils/                         # Utility functions
│       ├── __init__.py
│       ├── logger.py                  # Logging setup
│       ├── metrics.py                 # Custom metrics
│       ├── file_io.py                 # File operations
│       ├── validators.py              # Input validation
│       └── constants.py               # Application constants
│
├── app/                               # Streamlit application
│   ├── main.py                        # Main entry point
│   ├── pages/                         # Multi-page app
│   │   ├── 1_📊_Dashboard.py          # Overview dashboard
│   │   ├── 2_📁_Upload_Data.py        # Data upload page
│   │   ├── 3_🤖_ML_Analysis.py        # ML predictions page
│   │   ├── 4_💬_LLM_Insights.py       # LLM analysis page
│   │   ├── 5_📈_Rankings.py           # MCDA rankings page
│   │   ├── 6_🔍_Compare_Projects.py   # Comparison page
│   │   └── 7_💭_Chat_Assistant.py     # Chat interface
│   ├── components/                    # Reusable UI components
│   │   ├── __init__.py
│   │   ├── upload_widget.py           # File upload component
│   │   ├── risk_gauge.py              # Risk score gauge
│   │   ├── project_card.py            # Project summary card
│   │   └── data_table.py              # Interactive tables
│   ├── styles/                        # Custom CSS
│   │   ├── main.css
│   │   └── dark_theme.css
│   └── assets/                        # Static files
│       ├── logo.png
│       ├── favicon.ico
│       └── sample_data/               # Downloadable templates
│
├── notebooks/                         # Jupyter notebooks
│   ├── 01_data_exploration.ipynb      # EDA
│   ├── 02_feature_engineering.ipynb   # Feature creation and analysis
│   ├── 03_ml_modeling.ipynb           # ML model experiments
│   ├── 04_llm_experiments.ipynb       # LLM prompt testing
│   ├── 05_mcda_analysis.ipynb         # MCDA validation
│   ├── 06_hybrid_evaluation.ipynb     # Combined approach assessment
│   └── 07_results_visualization.ipynb # Final results and charts
│
├── tests/                             # Test suite
│   ├── __init__.py
│   ├── conftest.py                    # Pytest fixtures
│   ├── test_data/                     # Test datasets
│   │   ├── valid_project.csv
│   │   └── invalid_project.csv
│   ├── unit/                          # Unit tests
│   │   ├── test_data_loader.py
│   │   ├── test_preprocessor.py
│   │   ├── test_ml_models.py
│   │   ├── test_llm_analyzer.py
│   │   ├── test_mcda.py
│   │   └── test_utils.py
│   ├── integration/                   # Integration tests
│   │   ├── test_ml_pipeline.py
│   │   ├── test_hybrid_system.py
│   │   └── test_dashboard.py
│   └── performance/                   # Performance tests
│       └── test_response_time.py
│
├── docs/                              # Documentation
│   ├── academic/                      # Academic deliverables
│   │   ├── Chapter_1_Introduction.md
│   │   ├── Chapter_2_Literature_Review.md
│   │   ├── Chapter_3_Methodology.md
│   │   ├── Chapter_4_Implementation.md
│   │   ├── Chapter_5_Results.md
│   │   ├── Chapter_6_Conclusion.md
│   │   ├── references.bib             # BibTeX citations
│   │   └── figures/                   # Academic figures
│   ├── technical/                     # Technical documentation
│   │   ├── architecture.md            # System architecture
│   │   ├── api_reference.md           # Code API docs
│   │   ├── data_schema.md             # Data format specification
│   │   ├── model_details.md           # ML model documentation
│   │   └── deployment.md              # Deployment guide
│   ├── user_guide/                    # User documentation
│   │   ├── getting_started.md         # Quick start guide
│   │   ├── data_preparation.md        # How to prepare data
│   │   ├── using_dashboard.md         # Dashboard walkthrough
│   │   ├── interpreting_results.md    # Understanding outputs
│   │   └── faq.md                     # Frequently asked questions
│   ├── research/                      # Research notes
│   │   ├── literature_review_notes.md
│   │   ├── paper_summaries/           # Individual paper summaries
│   │   └── research_questions.md
│   └── presentations/                 # Slides and demos
│       ├── proposal_slides.pdf
│       ├── progress_presentation.pdf
│       └── final_defense.pdf
│
├── scripts/                           # Utility scripts
│   ├── setup_environment.sh           # Environment setup
│   ├── download_sample_data.py        # Fetch public datasets
│   ├── train_models.py                # Train all models
│   ├── evaluate_system.py             # Run full evaluation
│   ├── export_results.py              # Generate reports
│   └── deploy_app.sh                  # Deployment script
│
└── experiments/                       # Experimental code (not production)
    ├── alternative_algorithms/        # Algorithm experiments
    ├── prompt_variations/             # LLM prompt testing
    └── ui_prototypes/                 # UI mockups
```

### Directory Purpose Explanations

**Root Level Files:**
- `README.md`: Entry point for anyone discovering the project; includes setup, usage, and contribution guidelines
- `requirements.txt`: Pinned Python dependencies ensuring reproducible environments
- `.env.example`: Template showing required API keys and configuration (actual `.env` is gitignored)
- `Makefile`: Shortcuts for common tasks (`make setup`, `make test`, `make run`)

**`config/`**: Centralized configuration management
- Separates configuration from code
- YAML files for easy modification without code changes
- Version-controlled defaults with environment-specific overrides
- Enables experimentation by swapping config files

**`data/`**: All data assets (mostly gitignored except samples)
- `raw/`: Immutable source data; never modified after collection
- `processed/`: Cleaned data ready for modeling; reproducible from raw
- `schemas/`: Formal data contracts ensuring consistency
- Sample templates committed for user reference

**`models/`**: Serialized trained models (gitignored due to size)
- Organized by type (ML vs. LLM-related artifacts)
- Versioned with metadata tracking (performance, date, config used)
- Enables model rollback and A/B testing
- Registry file tracks all model versions

**`src/`**: Core application logic (production code)
- **`data/`**: ETL pipeline for all data operations
- **`models/`**: All ML and LLM logic; hybrid approach combines both
- **`mcda/`**: Decision analysis algorithms; pluggable design for multiple methods
- **`explainability/`**: Model interpretation; critical for trust
- **`chat/`**: Conversational interface; enhances usability
- **`visualization/`**: Plotting logic separated from UI
- **`utils/`**: Shared utilities; DRY principle

**`app/`**: User interface (Streamlit)
- `main.py`: Entry point that launches the dashboard
- `pages/`: Multi-page app structure; each page is self-contained
- `components/`: Reusable UI elements; consistent design
- `styles/`: Custom CSS for branding and UX
- `assets/`: Static files (images, templates)

**`notebooks/`**: Exploratory analysis and experimentation
- Numbered for sequential execution
- Bridges research and production code
- Documents decision-making process
- Useful for academic report figures

**`tests/`**: Comprehensive test suite
- Unit tests for individual functions
- Integration tests for workflows
- Performance benchmarks
- Test data fixtures for reproducibility

**`docs/`**: Multi-audience documentation
- **`academic/`**: Formal report chapters and thesis
- **`technical/`**: Developer-focused documentation
- **`user_guide/`**: End-user instructions
- **`research/`**: Literature review and research notes
- **`presentations/`**: Slides for defense and demos

**`scripts/`**: Automation and utilities
- One-command setup and deployment
- Batch processing scripts
- Evaluation and reporting automation

**`experiments/`**: Sandbox for trying new ideas
- Not production code; can be messy
- Quick prototyping without breaking main codebase
- Can be partially gitignored

---

## 1.4 Technology Stack

### Core Technologies

#### Programming Language
- **Python 3.10+**
  - Rationale: Rich ML/AI ecosystem, rapid development, excellent library support
  - Alternatives considered: R (limited LLM support), Julia (smaller ecosystem)

### Machine Learning Stack

#### Data Processing & Analysis
- **pandas 2.0+**: DataFrames for tabular data manipulation
- **numpy 1.24+**: Numerical computations and array operations
- **scipy 1.10+**: Statistical analysis and optimization

#### ML Frameworks & Algorithms
- **scikit-learn 1.3+**: Core ML algorithms, preprocessing, evaluation
  - Random Forest, SVM, Logistic Regression
  - Cross-validation, grid search
  - Pipeline and feature union
- **xgboost 2.0+**: Gradient boosting for high performance
- **lightgbm 4.0+**: Fast gradient boosting alternative
- **imbalanced-learn 0.11+**: Handle class imbalance (SMOTE, etc.)

#### Model Interpretability
- **shap 0.43+**: SHAP values for model explanations
- **lime 0.2+**: Local interpretable model-agnostic explanations
- **eli5 0.13+**: Feature importance visualization

#### Model Management
- **joblib 1.3+**: Model serialization and persistence
- **mlflow 2.8+** (optional): Experiment tracking and model registry

### Large Language Model Stack

#### LLM Integration
- **openai 1.3+**: OpenAI API client (GPT-4, GPT-3.5-turbo)
- **langchain 0.1+**: LLM orchestration and chaining
- **tiktoken 0.5+**: Token counting and cost estimation

#### NLP & Text Processing
- **nltk 3.8+**: Natural language toolkit (tokenization, sentiment)
- **spacy 3.7+**: Industrial-strength NLP (entity recognition, parsing)
- **textblob 0.17+**: Simple sentiment analysis baseline
- **transformers 4.35+** (optional): Hugging Face models for advanced NLP

#### Prompt Engineering
- **langchain-prompts**: Template management
- **guidance 0.1+** (optional): Structured LLM outputs

### MCDA (Multi-Criteria Decision Analysis)

#### MCDA Libraries
- **pymcdm 1.1+**: TOPSIS, VIKOR, PROMETHEE implementations
- **ahpy 0.3+**: Analytic Hierarchy Process
- **scipy.optimize**: Custom MCDA algorithm development

### Frontend & Visualization Stack

#### Dashboard Framework
- **streamlit 1.28+**: Interactive web dashboard
  - Multi-page apps
  - Session state management
  - File uploaders
  - Chat interface components

#### Visualization Libraries
- **plotly 5.17+**: Interactive charts and graphs (primary)
- **matplotlib 3.8+**: Static plots for reports
- **seaborn 0.13+**: Statistical visualizations
- **altair 5.1+**: Declarative visualizations
- **plotly-express**: Simplified plotly API

#### UI Components
- **streamlit-aggrid 0.3+**: Advanced data tables
- **streamlit-echarts 0.4+**: Additional chart types
- **streamlit-extras 0.3+**: Enhanced UI components

### Data Validation & Quality

#### Data Validation
- **pydantic 2.4+**: Data validation using type annotations
- **great_expectations 0.18+** (optional): Data quality testing
- **pandera 0.17+**: DataFrame schema validation

### Utility Libraries

#### Configuration Management
- **python-dotenv 1.0+**: Environment variable management
- **pyyaml 6.0+**: YAML configuration files
- **hydra-core 1.3+** (optional): Complex configuration management

#### Logging & Monitoring
- **loguru 0.7+**: Simplified logging with better formatting
- **tqdm 4.66+**: Progress bars for long operations
- **rich 13.6+**: Beautiful terminal output and logging

#### API & Networking
- **requests 2.31+**: HTTP requests
- **httpx 0.25+**: Async HTTP client
- **tenacity 8.2+**: Retry logic for API calls

### Testing Stack

#### Testing Frameworks
- **pytest 7.4+**: Primary testing framework
- **pytest-cov 4.1+**: Code coverage reporting
- **pytest-mock 3.12+**: Mocking for unit tests
- **hypothesis 6.92+**: Property-based testing

#### Performance Testing
- **locust 2.17+** (optional): Load testing for API endpoints
- **memory_profiler 0.61+**: Memory usage analysis

### Development Tools

#### Code Quality
- **black 23.11+**: Code formatting
- **flake8 6.1+**: Linting and style checks
- **pylint 3.0+**: Additional linting
- **mypy 1.7+**: Static type checking
- **isort 5.12+**: Import sorting

#### Version Control
- **git**: Version control system
- **pre-commit 3.5+**: Git hooks for code quality

#### Documentation
- **sphinx 7.2+**: API documentation generation
- **mkdocs 1.5+**: User documentation site
- **pdoc3 0.10+**: Lightweight API docs

### Deployment & Infrastructure

#### Containerization (Optional for POC)
- **docker 24.0+**: Containerization
- **docker-compose 2.23+**: Multi-container orchestration

#### Cloud Services (Optional)
- **streamlit-cloud**: Free hosting for Streamlit apps
- **heroku**: Alternative deployment platform
- **aws-s3** / **google-cloud-storage**: Model and data storage

### External APIs

#### Required APIs
- **OpenAI API**: GPT models for LLM analysis
  - Pricing: ~$0.002/1K tokens (GPT-3.5-turbo), ~$0.03/1K tokens (GPT-4)
  - Budget: ~$50-100 for POC development and testing

#### Optional APIs
- **Hugging Face Inference API**: Alternative to OpenAI
- **Cohere API**: Additional LLM option

---

### Technology Stack Summary Table

| Category | Primary Tool | Alternative | Reason for Choice |
|----------|-------------|-------------|-------------------|
| **Language** | Python 3.10+ | - | Best ML/AI ecosystem |
| **ML Framework** | scikit-learn | TensorFlow | Simpler for tabular data |
| **Boosting** | XGBoost | LightGBM | Better performance, popular |
| **LLM** | OpenAI GPT-4 | Anthropic Claude | Best accuracy, widespread |
| **LLM Orchestration** | LangChain | Custom | Faster development |
| **MCDA** | pymcdm | Custom | Validated algorithms |
| **Dashboard** | Streamlit | Dash, Gradio | Fastest development, intuitive |
| **Visualization** | Plotly | Matplotlib | Interactive charts |
| **Data Validation** | Pydantic | Cerberus | Type-safe, modern |
| **Testing** | pytest | unittest | Better fixtures, plugins |
| **Interpretability** | SHAP | LIME | More accurate, widely used |

---

### Dependency Management

#### requirements.txt Structure
```txt
# Core Data Science
pandas==2.0.3
numpy==1.24.4
scipy==1.10.1

# Machine Learning
scikit-learn==1.3.2
xgboost==2.0.3
lightgbm==4.1.0
imbalanced-learn==0.11.0

# Model Interpretability
shap==0.43.0
lime==0.2.0.1

# LLM and NLP
openai==1.3.0
langchain==0.1.0
tiktoken==0.5.1
nltk==3.8.1
spacy==3.7.2

# MCDA
pymcdm==1.1.4
ahpy==0.3.1

# Dashboard and Visualization
streamlit==1.28.2
plotly==5.17.0
matplotlib==3.8.2
seaborn==0.13.0
streamlit-aggrid==0.3.4

# Data Validation
pydantic==2.4.2
pandera==0.17.2

# Configuration and Utils
python-dotenv==1.0.0
pyyaml==6.0.1
loguru==0.7.2
tqdm==4.66.1
requests==2.31.0
tenacity==8.2.3

# Testing
pytest==7.4.3
pytest-cov==4.1.0
pytest-mock==3.12.0

# Code Quality
black==23.11.0
flake8==6.1.0
mypy==1.7.1
isort==5.12.0
```

#### Version Pinning Strategy
- **Exact versions** for critical dependencies (ML, LLM)
- **Compatible versions** (>=) for utilities
- **Lock files** (`requirements-lock.txt`) for production

---

### Setup and Installation

#### Quick Start Commands
```bash
# Clone repository
git clone https://github.com/yourusername/prism.git
cd prism

# Create virtual environment
python3.10 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install --upgrade pip
pip install -r requirements.txt

# Download NLTK and spaCy data
python -m nltk.downloader punkt vader_lexicon
python -m spacy download en_core_web_sm

# Setup environment variables
cp .env.example .env
# Edit .env and add your OPENAI_API_KEY

# Run tests
pytest tests/

# Launch dashboard
streamlit run app/main.py
```

#### Docker Setup (Optional)
```dockerfile
FROM python:3.10-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8501
CMD ["streamlit", "run", "app/main.py"]
```

---

## 1.5 Data Requirements

### Structured Data Schema

#### Project Core Attributes (Required Fields)

| Field Name | Data Type | Description | Example | Source System |
|------------|-----------|-------------|---------|---------------|
| `project_id` | String | Unique identifier | "PROJ-2024-001" | All systems |
| `project_name` | String | Descriptive name | "Mobile App Redesign" | All systems |
| `project_type` | Category | Project category | "Development", "Maintenance" | All systems |
| `start_date` | Date | Planned start date | "2024-01-15" | All systems |
| `planned_end_date` | Date | Target completion | "2024-06-30" | All systems |
| `actual_end_date` | Date/Null | Actual completion (if done) | "2024-07-15" or null | All systems |
| `budget` | Float | Total budget ($) | 150000.00 | All systems |
| `spent` | Float | Current spend ($) | 125000.00 | All systems |
| `planned_hours` | Integer | Estimated effort (hours) | 2000 | Jira, Azure DevOps |
| `actual_hours` | Integer | Logged time (hours) | 2300 | Jira, Azure DevOps |
| `team_size` | Integer | Number of members | 8 | All systems |
| `completion_rate` | Float | % complete (0-100) | 75.5 | All systems |
| `priority` | Category | Business priority | "High", "Medium", "Low" | All systems |
| `status` | Category | Current status | "Active", "On Hold", "Completed" | All systems |

#### Project Performance Metrics (Required for ML)

| Field Name | Data Type | Description | Calculation |
|------------|-----------|-------------|-------------|
| `schedule_variance` | Float | Days ahead/behind | actual_days - planned_days |
| `cost_variance` | Float | Budget difference ($) | spent - budget |
| `schedule_performance_index` | Float | Schedule efficiency | planned_hours / actual_hours |
| `cost_performance_index` | Float | Cost efficiency | budget / spent |
| `velocity` | Float | Story points per sprint | completed_points / sprints |
| `defect_rate` | Float | Bugs per feature | defects / features |
| `team_turnover` | Float | % team changes | departed / team_size |

#### Project Context (Optional but Valuable)

| Field Name | Data Type | Description | ML Impact |
|------------|-----------|-------------|-----------|
| `department` | String | Owning department | Medium |
| `client_type` | Category | Internal/External | High |
| `technology_stack` | String | Tech used | Medium |
| `methodology` | Category | Agile/Waterfall | High |
| `risk_level` | Category | Initial risk assessment | Very High |
| `complexity_score` | Integer | Subjective complexity (1-10) | High |
| `dependencies` | Integer | Number of dependent projects | Medium |

#### Text Data for LLM Analysis (Critical)

| Field Name | Data Type | Description | Min. Length | LLM Purpose |
|------------|-----------|-------------|-------------|-------------|
| `project_description` | Text | Initial project goals | 50 chars | Context understanding |
| `status_comments` | Text | Recent updates | 100 chars | Risk detection |
| `issue_summaries` | Text | Aggregated issues | 200 chars | Problem identification |
| `team_feedback` | Text | Team retrospectives | 100 chars | Sentiment analysis |
| `stakeholder_notes` | Text | Client/sponsor feedback | 50 chars | Expectation analysis |

**Text Quality Requirements:**
- Minimum 1 text field with 100+ characters per project
- Prefer `status_comments` as it contains real-time concerns
- Multiple text sources improve LLM accuracy
- Text should be recent (within last 30 days ideal)

#### Target Labels (For ML Training)

| Field Name | Data Type | Description | Values |
|------------|-----------|-------------|--------|
| `risk_category` | Category | Ground truth risk level | "High Risk", "Medium Risk", "Low Risk" |
| `success_outcome` | Binary | Did project succeed? | "Success", "Failure" |
| `final_status` | Category | Completion status | "Completed On-Time", "Delayed", "Cancelled" |

**Label Requirements:**
- Minimum 30% of dataset must have labels for supervised learning
- Class distribution: aim for 20-30% high-risk, 40-50% medium, 20-30% low
- Use domain expert labels or historical outcomes

---

### Data Volume Requirements

#### Minimum Dataset Sizes

**For POC Validation:**
- **Training set:** 100-150 projects
- **Validation set:** 30-50 projects
- **Test set:** 30-50 projects
- **Total:** 200-250 projects minimum

**For Production Quality:**
- **Training set:** 500+ projects
- **Test set:** 100+ projects
- **Total:** 600+ projects

#### Why These Numbers?

**ML Considerations:**
- 20-30 features requires 10x samples minimum (200+)
- Class imbalance handling needs sufficient minority samples
- Cross-validation (5-fold) needs 100+ for stable estimates
- Ensemble models benefit from larger datasets

**LLM Considerations:**
- LLMs work with small data (few-shot learning)
- 20-30 labeled examples for prompt calibration
- Text quality matters more than quantity

#### Class Distribution

**Ideal Balance:**
```
High Risk:    25% (50 projects)
Medium Risk:  45% (90 projects)
Low Risk:     30% (60 projects)
Total:        200 projects
```

**Handling Imbalance:**
- Use SMOTE for oversampling minority class
- Stratified sampling in train/test split
- Class weights in model training
- Ensemble methods (balances naturally)

---

### Sample Datasets and Sources

#### Primary Dataset: Apache JIRA Issues (Kaggle)

**Selected Dataset for PRISM Development:**

| Attribute | Value |
|-----------|-------|
| **Source** | [Apache JIRA Issues - Kaggle](https://www.kaggle.com/datasets/tedlozzo/apaches-jira-issues) |
| **Total Issues** | 18.5 million |
| **Total Comments** | 62.4 million (critical for LLM) |
| **Changelog Entries** | 40.5 million |
| **Projects** | 640 Apache Foundation projects |
| **Top Projects** | Spark (50K), Flink (37K), HBase (29K), Hive (29K), Kafka (17K) |
| **Date Range** | 2000 - Present |
| **Size** | ~8 GB (raw CSV files) |

**Why This Dataset?**
- ✅ **Rich text data** for LLM sentiment/risk extraction (62M developer comments)
- ✅ **Real software engineering** context from Apache Foundation
- ✅ **Issue tracking** with priorities, statuses, resolutions, blockers
- ✅ **Change history** for deriving velocity and churn metrics
- ✅ **Large scale** for robust ML training (18.5M issues)
- ✅ **Open source** and freely available on Kaggle
- ✅ **Well-structured** CSV format for easy processing

**Dataset Files:**
- `issues.csv` - Core issue data (status, priority, dates, descriptions)
- `comments.csv` - Developer comments (LLM input)
- `changelog.csv` - Issue history (status changes, reassignments)
- `issuelinks.csv` - Issue dependencies and relationships

**Preprocessing Script:** `scripts/preprocess_jira_data.py`
- Aggregates issue-level data to project-level metrics
- Derives risk labels from issue outcomes
- Joins comments for LLM text analysis
- Outputs PRISM-compatible CSV format

---

#### Alternative/Supplementary Datasets

**1. NASA Software Defect Datasets (MDP)**
- **Source:** [NASA Metrics Data Program](https://github.com/klainfo/NASADefectDataset)
- **Records:** 10,000+ modules across 13 projects
- **Features:** Code metrics, defect counts
- **Limitation:** Lacks text comments
- **Usage:** Baseline ML model training

**2. ISBSG Dataset**
- **Source:** [International Software Benchmarking Standards Group](https://isbsg.org/)
- **Records:** 9,000+ projects
- **Features:** Effort, duration, size, quality
- **Cost:** ~$500 (academic discount available)
- **Usage:** Gold standard for software project benchmarks

**3. SQuaD - Software Quality Dataset**
- **Source:** [arxiv.org/abs/2511.11265](https://arxiv.org/abs/2511.11265)
- **Projects:** 450 mature open-source projects (Apache, Mozilla, Linux)
- **Features:** 700+ software quality metrics
- **Releases:** 63,586 analyzed releases
- **Usage:** Code quality metrics, technical debt analysis

**4. GitHub Project Data**
- **Source:** [GitHub Archive](https://www.gharchive.org/)
- **Records:** Millions of projects
- **Features:** Commits, issues, PRs, discussions
- **Usage:** Text comments and activity metrics
- **Extraction:** GitHub GraphQL API

#### Synthetic Data Strategy

**Why Synthetic Data for POC:**
- ✅ Controlled risk distribution
- ✅ Privacy-compliant (no real client data)
- ✅ Adjustable complexity
- ✅ Unlimited volume for testing
- ✅ Known ground truth for validation

**Synthetic Data Generator Features:**

```python
# Example schema for generator
generate_projects(
    n_projects=200,
    risk_distribution={'high': 0.25, 'medium': 0.45, 'low': 0.30},
    date_range=('2022-01-01', '2024-11-01'),
    include_text=True,
    text_realism='high',  # Uses templates + variation
    correlation_strength=0.7  # Feature correlation
)
```

**Realistic Text Generation:**
- Template-based with variations
- Risk-aligned comments (high-risk → negative sentiment)
- Common PM terminology and phrases
- Mix of structured and unstructured text

**Validation of Synthetic Data:**
- Statistical similarity to real datasets (distribution tests)
- Feature correlations match domain knowledge
- Model trained on synthetic transfers to real data

#### Industry Partnerships (Aspirational)

**Potential Partners:**
- Local software companies (anonymized data)
- University IT projects
- Capstone project archives
- Open-source foundations

**Data Sharing Agreement:**
- Anonymize all identifiable information
- Aggregate metrics only
- Academic use only
- Results shared back with partner

---

### Data Preparation Pipeline

#### Data Collection Phase

**Step 1: Acquire Raw Data**
- Download public datasets
- Generate synthetic data
- (Optional) Collect from partners

**Step 2: Initial Assessment**
```python
# Data quality checks
- Row count and completeness
- Missing value analysis (target < 20% per column)
- Duplicate detection
- Date range validation
- Outlier detection (IQR method)
```

**Step 3: Schema Mapping**
- Map source fields to standard schema
- Handle naming variations (e.g., "Budget" vs "Planned_Cost")
- Document transformations

#### Data Cleaning Phase

**Step 4: Handle Missing Data**
```python
# Strategy by field type
- Numeric: Median imputation or regression imputation
- Categorical: Mode or 'Unknown' category
- Text: Empty string or 'No comment provided'
- Critical fields: Drop rows if missing (e.g., project_id)
```

**Step 5: Data Type Conversion**
```python
# Standardize types
- Dates: ISO 8601 format (YYYY-MM-DD)
- Currency: Float with 2 decimals
- Percentages: 0-100 scale
- Categories: Lowercase, standardized values
```

**Step 6: Outlier Treatment**
```python
# Handle extreme values
- Budget > $10M: Flag or cap
- Negative values (errors): Correct or drop
- Completion rate > 100%: Cap at 100
```

#### Feature Engineering Phase

**Step 7: Derived Features**
```python
# Calculate performance metrics
- schedule_variance = actual_days - planned_days
- cost_variance_pct = (spent - budget) / budget * 100
- spi = planned_hours / actual_hours if actual_hours > 0 else 1
- cpi = budget / spent if spent > 0 else 1
- completion_velocity = completion_rate / days_elapsed
- burn_rate = spent / days_elapsed
```

**Step 8: Temporal Features**
```python
# Time-based features
- project_duration_days = end_date - start_date
- days_since_start = current_date - start_date
- days_remaining = planned_end_date - current_date
- quarter = extract quarter from start_date
- is_year_end = True if month in [11, 12]
```

**Step 9: Categorical Encoding**
```python
# For ML models
- One-hot encoding: project_type, methodology
- Ordinal encoding: priority, risk_level
- Target encoding: department (if many categories)
```

#### Text Preprocessing Phase

**Step 10: Text Cleaning**
```python
# Standardize text data
- Lowercase conversion (optional, depends on use)
- Remove special characters
- Fix encoding issues (UTF-8)
- Combine multiple text fields into single column
```

**Step 11: Text Feature Extraction** (For ML, not LLM)
```python
# Traditional NLP features
- Sentiment score (-1 to 1)
- Text length
- Keyword counts (e.g., 'delay', 'risk', 'issue')
- Readability scores
```

#### Quality Assurance Phase

**Step 12: Validation Rules**
```python
# Business logic checks
assert end_date >= start_date
assert budget > 0
assert 0 <= completion_rate <= 100
assert actual_hours >= 0
assert team_size > 0
```

**Step 13: Final Quality Report**
- Missing value percentages
- Outlier counts
- Class distribution
- Statistical summaries
- Data lineage documentation

---

### Data Storage and Versioning

#### Directory Organization
```
data/
├── raw/
│   ├── nasa_mdp_v1.csv           # Original source
│   ├── apache_jira_projects.json # Original source
│   └── synthetic_v1.csv          # Generated data
├── processed/
│   ├── train.csv                 # 60% of data
│   ├── validation.csv            # 20% of data
│   └── test.csv                  # 20% of data
├── schemas/
│   └── project_schema.json       # Field definitions
└── metadata/
    └── data_provenance.yaml      # Data lineage
```

#### Version Control for Data
- Git LFS for large files (>100MB)
- DVC (Data Version Control) for dataset versioning
- SHA-256 checksums for data integrity
- Timestamp and version tags

#### Data Security
- No API keys or credentials in data files
- Anonymize client/person names
- `.gitignore` for sensitive data
- Encryption for data at rest (if needed)

---

## 1.6 Testing & Validation Strategy

### ML Model Evaluation Metrics

#### Classification Metrics (Risk Prediction)

**Primary Metrics:**

1. **ROC-AUC (Area Under ROC Curve)**
   - **Target:** ≥ 0.75 (Good), ≥ 0.80 (Excellent)
   - **Why:** Measures model's ability to distinguish risk levels
   - **Use:** Compare different algorithms

2. **F1-Score (Harmonic Mean of Precision/Recall)**
   - **Target:** ≥ 0.70 for high-risk class
   - **Why:** Balances false positives and false negatives
   - **Use:** Primary metric for deployment decision

3. **Precision (Positive Predictive Value)**
   - **Target:** ≥ 0.75 for high-risk class
   - **Why:** Minimizes false alarms (crying wolf)
   - **Use:** Ensure recommendations are trustworthy

4. **Recall (Sensitivity)**
   - **Target:** ≥ 0.70 for high-risk class
   - **Why:** Catch most actual high-risk projects
   - **Use:** Safety metric (don't miss critical risks)

**Secondary Metrics:**

5. **Accuracy**
   - **Target:** ≥ 0.75 overall
   - **Caveat:** Can be misleading with class imbalance
   - **Use:** Simple interpretability for stakeholders

6. **Confusion Matrix Analysis**
   - False Negatives (missed risks): Most critical
   - False Positives (false alarms): Less critical but affects trust

7. **Matthews Correlation Coefficient (MCC)**
   - **Target:** ≥ 0.50
   - **Why:** Robust to class imbalance
   - **Use:** Alternative overall metric

**Evaluation Approach:**

```python
# Cross-validation
- 5-fold stratified CV for model selection
- Report mean ± std for all metrics
- Ensure stability (low variance)

# Test set evaluation
- Hold-out test set (20%, never touched during training)
- Single evaluation after final model selection
- Generate classification report and confusion matrix
```

#### Regression Metrics (If Predicting Continuous Values)

If predicting completion rate or cost overrun as continuous:

1. **RMSE (Root Mean Squared Error)**
   - **Target:** < 15% of mean target value
   - **Why:** Penalizes large errors heavily

2. **MAE (Mean Absolute Error)**
   - **Target:** < 10% of mean target value
   - **Why:** Interpretable (average error magnitude)

3. **R² (Coefficient of Determination)**
   - **Target:** ≥ 0.60
   - **Why:** Variance explained by model

---

### LLM Output Quality Assessment

#### Quantitative Metrics

**1. Risk Classification Accuracy**
- **Method:** Compare LLM risk categorization to human labels
- **Metric:** Agreement rate (%)
- **Target:** ≥ 75% agreement with domain experts
- **Sample Size:** 50 manually labeled project comments

**2. Sentiment Correlation**
- **Method:** Compare LLM sentiment to VADER/TextBlob baseline
- **Metric:** Pearson correlation
- **Target:** r ≥ 0.70
- **Interpretation:** LLM captures similar sentiment patterns

**3. F1-Score for Risk Categories**
- **Method:** Treat LLM output as classifier
- **Metric:** F1-score per risk type (technical, resource, schedule, scope)
- **Target:** F1 ≥ 0.65 per category

**4. Response Consistency**
- **Method:** Run same prompt 3 times, measure variation
- **Metric:** Exact match rate or Jaccard similarity
- **Target:** ≥ 80% consistency
- **Note:** Use temperature=0 for consistency

#### Qualitative Evaluation

**5. Human Expert Review**
- **Method:** Domain experts rate LLM outputs (1-5 scale)
- **Criteria:**
  - Relevance: Are extracted risks actually present?
  - Completeness: Are key risks missed?
  - Accuracy: Are categorizations correct?
  - Utility: Are insights actionable?
- **Sample:** 30 projects rated by 2-3 experts
- **Target:** Average rating ≥ 3.5/5

**6. Hallucination Detection**
- **Method:** Check if LLM invents facts not in input
- **Metric:** Hallucination rate (%)
- **Target:** < 5% of responses contain hallucinations
- **Detection:** Manual review + fact-checking against source text

#### Cost and Performance Metrics

**7. Token Usage Efficiency**
- **Metric:** Average tokens per project
- **Target:** < 2,000 tokens per analysis
- **Cost:** < $0.10 per project (GPT-3.5-turbo)

**8. Response Time**
- **Metric:** 95th percentile latency
- **Target:** < 5 seconds per project
- **Includes:** API call + parsing

**9. API Reliability**
- **Metric:** Success rate (%)
- **Target:** ≥ 99% (with retries)
- **Handling:** Exponential backoff, fallback strategies

#### Validation Process

```python
# LLM Evaluation Pipeline

Step 1: Create gold standard dataset
- 50 projects with expert-labeled risks
- Diverse risk profiles
- Range of text quality

Step 2: Run LLM analysis
- Process each project with production prompts
- Record outputs and metadata (tokens, time)
- Retry failed calls

Step 3: Automated metrics
- Calculate agreement, F1, correlation
- Generate confusion matrix for risk types

Step 4: Manual review
- Expert review of 30 samples
- Rate relevance, completeness, accuracy
- Document hallucinations and errors

Step 5: Iterate on prompts
- Adjust prompts based on failure modes
- Re-evaluate with same gold standard
- Track improvement over iterations
```

---

### MCDA Ranking Validation

#### Mathematical Correctness

**1. Algorithm Verification**
- **Method:** Test TOPSIS/AHP against known examples from literature
- **Metric:** Exact match of ranking
- **Target:** 100% match on benchmark problems

**2. Criteria Weight Sensitivity**
- **Method:** Vary each criterion weight ±20%
- **Metric:** Kendall's Tau correlation of rankings
- **Target:** τ ≥ 0.80 (stable rankings)
- **Interpretation:** Small weight changes shouldn't drastically alter rankings

**3. Monotonicity Test**
- **Method:** Improve one criterion for a project, verify rank improves
- **Metric:** Pass/fail for 20 test cases
- **Target:** 100% pass rate
- **Interpretation:** Better scores → better ranks

#### Ranking Quality

**4. Top-K Precision**
- **Method:** Compare MCDA top-10 risks with expert top-10
- **Metric:** Overlap @ K (e.g., @ K=5, @ K=10)
- **Target:** ≥ 60% overlap for top-5
- **Use:** Validates most critical insights

**5. Rank Correlation with Ground Truth**
- **Method:** Correlate MCDA rank with actual project outcomes
- **Metric:** Spearman's rho
- **Target:** ρ ≥ 0.50
- **Dataset:** Historical projects with known outcomes

**6. Discriminative Power**
- **Method:** Measure score separation between risk levels
- **Metric:** Effect size (Cohen's d) between high/low risk
- **Target:** d ≥ 0.80 (large effect)
- **Interpretation:** Clear distinction between risk categories

#### Hybrid System Validation

**7. Incremental Value Test**
- **Method:** Compare three approaches:
  - ML only
  - LLM only  
  - Hybrid (ML + LLM + MCDA)
- **Metric:** F1-score, AUC, ranking correlation
- **Target:** Hybrid outperforms single methods by ≥5%
- **Critical:** Justifies hybrid approach

**8. Ablation Study**
- **Method:** Remove each MCDA criterion, measure ranking change
- **Metric:** Kendall's Tau drop
- **Interpretation:** Identifies most influential criteria

---

### User Acceptance Testing (UAT)

#### UAT Planning

**Participants:**
- 5-7 potential users (project managers, team leads)
- Mix of technical and non-technical backgrounds
- Preference for those with PM tool experience

**Timeline:**
- Week 14: Recruit participants
- Week 15: Conduct sessions (1 hour each)
- Week 16: Analyze feedback and iterate

#### UAT Scenarios

**Scenario 1: Upload and Analyze**
- **Task:** Upload sample CSV, view risk predictions
- **Success Criteria:** Complete without assistance
- **Metrics:** Time to completion, errors encountered

**Scenario 2: Interpret Results**
- **Task:** Identify top 3 high-risk projects, explain why
- **Success Criteria:** Correct interpretation
- **Metrics:** Accuracy of interpretation

**Scenario 3: Use Chat Assistant**
- **Task:** Ask "Why is Project X high risk?" and understand answer
- **Success Criteria:** Satisfactory explanation received
- **Metrics:** User satisfaction rating

**Scenario 4: Compare Projects**
- **Task:** Compare two projects side-by-side
- **Success Criteria:** User can articulate differences
- **Metrics:** Completeness of comparison

**Scenario 5: Export Results**
- **Task:** Export ranking table to CSV
- **Success Criteria:** File downloads correctly
- **Metrics:** File validity, contains expected data

#### UAT Metrics

**System Usability Scale (SUS)**
- **Method:** 10-question standardized survey
- **Target:** Score ≥ 70 (above average)
- **Interpretation:** Measures perceived usability

**Task Success Rate**
- **Metric:** % of tasks completed successfully
- **Target:** ≥ 80%

**Time on Task**
- **Metric:** Average time per scenario
- **Target:** < 5 minutes per task

**Error Rate**
- **Metric:** Number of user errors per session
- **Target:** < 3 errors per session

**Subjective Satisfaction**
- **Method:** 5-point Likert scale questions
  - "I found the system easy to use"
  - "The risk predictions seem accurate"
  - "I would use this in my work"
- **Target:** Average ≥ 4.0/5

#### Feedback Collection

**Qualitative Feedback:**
- Open-ended questions:
  - "What did you like most?"
  - "What was confusing or frustrating?"
  - "What features are missing?"
  - "Would you recommend this to colleagues?"

**Think-Aloud Protocol:**
- Users verbalize thoughts while using system
- Identify pain points and misunderstandings
- Record for analysis

**Feedback Analysis:**
- Categorize feedback themes
- Prioritize critical vs. nice-to-have improvements
- Document in issue tracker for future work

---

### Performance Benchmarks

#### Response Time Benchmarks

| Operation | Target Latency | Max Acceptable |
|-----------|----------------|----------------|
| **Upload CSV (100 projects)** | < 2 seconds | < 5 seconds |
| **ML prediction (100 projects)** | < 3 seconds | < 10 seconds |
| **LLM analysis (per project)** | < 3 seconds | < 5 seconds |
| **LLM batch (100 projects)** | < 2 minutes | < 5 minutes |
| **MCDA ranking (100 projects)** | < 1 second | < 3 seconds |
| **Dashboard page load** | < 2 seconds | < 5 seconds |
| **Chat response** | < 5 seconds | < 10 seconds |

#### Scalability Benchmarks

| Dataset Size | ML Prediction | LLM Analysis | MCDA Ranking | End-to-End |
|--------------|---------------|--------------|--------------|------------|
| **10 projects** | < 1s | < 30s | < 0.5s | < 40s |
| **50 projects** | < 2s | < 2min | < 0.5s | < 3min |
| **100 projects** | < 3s | < 4min | < 1s | < 6min |
| **500 projects** | < 10s | < 20min | < 3s | < 25min |

**Optimization Strategies:**
- Parallel LLM API calls (batch requests)
- Caching for repeated analyses
- Async processing for large uploads
- Progress indicators for long operations

#### Memory and Storage

| Component | Memory Usage | Storage |
|-----------|--------------|---------|
| **ML Model (loaded)** | < 100 MB | ~50 MB |
| **Dashboard (runtime)** | < 500 MB | - |
| **Data (100 projects)** | < 10 MB | ~5 MB |
| **LLM Cache** | - | ~50 MB/100 projects |
| **Total Application** | < 1 GB | ~500 MB |

---

### Testing Strategy Summary

#### Testing Pyramid

```
         /\
        /  \  End-to-End Tests (5%)
       /____\  
      /      \  Integration Tests (15%)
     /________\
    /          \ Unit Tests (80%)
   /____________\
```

**Unit Tests (80% of tests):**
- Individual functions in isolation
- Fast execution (< 1 second total)
- High coverage (≥ 80% code coverage)
- Mock external dependencies

**Integration Tests (15% of tests):**
- ML pipeline (data → prediction)
- LLM workflow (text → risk extraction)
- Dashboard → backend integration
- Mock external APIs (OpenAI)

**End-to-End Tests (5% of tests):**
- Full user workflows
- Actual Streamlit app interactions
- Slowest but most realistic

#### Test Coverage Targets

| Module | Target Coverage | Critical Sections |
|--------|----------------|-------------------|
| **Data processing** | 85% | Validation, cleaning |
| **ML models** | 80% | Training, prediction |
| **LLM integration** | 75% | API calls, parsing |
| **MCDA** | 90% | Algorithms, ranking |
| **Utils** | 85% | File I/O, metrics |
| **Dashboard** | 60% | Core functionality |

**Rationale:**
- Higher coverage for algorithmic code (MCDA, ML)
- Lower coverage for UI code (harder to test, less critical)
- 100% coverage not required (diminishing returns)

#### Continuous Testing

**Pre-commit Hooks:**
- Run linters (flake8, black)
- Run fast unit tests (< 10s)
- Check for API keys in commits

**CI/CD Pipeline** (GitHub Actions):
```yaml
on: [push, pull_request]
steps:
  - Run all unit tests
  - Run integration tests
  - Check code coverage (fail if < 75%)
  - Build documentation
  - (Optional) Deploy to staging
```

#### Testing Documentation

**Test Plan Document:**
- Test strategy and objectives
- Test environment setup
- Test case descriptions
- Expected vs. actual results
- Bug tracking and resolution

**Test Reports:**
- Automated: pytest HTML reports, coverage reports
- Manual: UAT session notes, expert review summaries
- Final: Comprehensive testing summary for academic report

---

### Validation Checklist (Pre-Submission)

**ML Model:**
- ✅ Test set AUC ≥ 0.75
- ✅ F1-score ≥ 0.70 for high-risk class
- ✅ 5-fold CV shows stable performance (std < 0.05)
- ✅ SHAP explanations make domain sense
- ✅ Model saved and reproducible

**LLM Integration:**
- ✅ Agreement with human labels ≥ 75%
- ✅ No hallucinations in 30-sample review
- ✅ Average cost < $0.10 per project
- ✅ Response time < 5 seconds (95th percentile)
- ✅ Handles edge cases (empty text, very long text)

**MCDA Ranking:**
- ✅ Passes benchmark problems (100% accuracy)
- ✅ Ranking stability: Kendall's tau ≥ 0.80
- ✅ Hybrid outperforms single methods by ≥ 5%
- ✅ Top-5 overlap with ground truth ≥ 60%

**Dashboard:**
- ✅ SUS score ≥ 70 from UAT
- ✅ Task success rate ≥ 80%
- ✅ All core features functional
- ✅ No critical bugs
- ✅ Responsive on 1920x1080 and 1366x768 screens

**Code Quality:**
- ✅ Code coverage ≥ 75%
- ✅ All tests passing
- ✅ No linter errors
- ✅ Documentation complete
- ✅ README with setup instructions

**Academic Deliverables:**
- ✅ All chapters complete
- ✅ Minimum 30 citations
- ✅ Figures and tables properly formatted
- ✅ Results section with quantitative evidence
- ✅ Discussion of limitations and future work

---

## Summary

This comprehensive project strategy provides:

1. **Clear roadmap:** 8 phases over 16 weeks with concrete deliverables
2. **Risk mitigation:** Identified risks with proactive strategies
3. **Technical foundation:** Complete tech stack and architecture
4. **Data strategy:** Schema, sources, and pipeline for POC
5. **Quality assurance:** Multi-level testing and validation
6. **Realistic scope:** Balances ambition with capstone timeline

**Next Steps:**
1. Setup development environment
2. Acquire/generate initial dataset
3. Begin Phase 1 (Foundation & Setup)
4. Weekly progress tracking against milestones

---

**Document Version:** 1.0  
**Last Updated:** November 9, 2024  
**Status:** Ready for Implementation

