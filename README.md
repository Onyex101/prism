# PRISM - Predictive Risk Intelligence for Software Management

**A Hybrid AI System Integrating Machine Learning and Large Language Models for Software Project Risk Analysis**

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Status: Capstone Project](https://img.shields.io/badge/Status-Capstone%20Project-green.svg)]()

---

## 📋 Table of Contents

- [Overview](#overview)
- [Key Features](#key-features)
- [System Architecture](#system-architecture)
- [Project Documentation](#project-documentation)
- [Quick Start](#quick-start)
- [Use Cases](#use-cases)
- [Technology Stack](#technology-stack)
- [Project Status](#project-status)
- [Contributing](#contributing)
- [License](#license)
- [Contact](#contact)

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
- Average cost overruns: 189% | Average time overruns: 222%

Traditional risk management is reactive—problems are identified only after they appear in metrics. Project managers need predictive capabilities that forecast risks 2-4 weeks in advance.

### The Solution

PRISM provides:
- ✅ **Early Risk Detection:** Predicts high-risk projects 2-4 weeks before traditional metrics deteriorate
- ✅ **Hybrid Intelligence:** Combines quantitative metrics with qualitative team communications
- ✅ **Portfolio Prioritization:** Objectively ranks projects to focus manager attention
- ✅ **Explainable AI:** Natural language explanations via chat assistant
- ✅ **Actionable Insights:** Not just "what" but "why" and "what to do"

---

## ✨ Key Features

### 1. Machine Learning Risk Prediction
- **Ensemble models** (Random Forest, XGBoost) trained on historical project data
- **Feature engineering** from schedule performance, cost variance, velocity, and team metrics
- **SHAP explainability** showing which factors drive each project's risk score
- **Target accuracy:** ROC-AUC ≥ 0.75, F1-score ≥ 0.70

### 2. LLM-Powered Text Analysis
- **GPT integration** (OpenAI API) to analyze project comments, status updates, team feedback
- **Risk extraction:** Identifies concerns, blockers, and warning signs in natural language
- **Sentiment analysis:** Detects team morale issues and stakeholder dissatisfaction
- **Risk categorization:** Technical, resource, schedule, and scope risks
- **Evidence-based:** Includes direct quotes from source text

### 3. MCDA Project Ranking
- **TOPSIS algorithm** ranks projects based on multiple weighted criteria:
  - ML risk score (40% weight)
  - LLM sentiment score (25%)
  - Schedule performance index (15%)
  - Cost performance index (10%)
  - Team stability (10%)
- **Configurable weights:** Customize to organizational priorities
- **Sensitivity analysis:** Validates ranking stability

### 4. Interactive Dashboard
- **Streamlit-based** web interface—no installation required
- **File upload:** CSV/JSON from Jira, Azure DevOps, Monday.com, etc.
- **Visualizations:** Risk distribution charts, trend analysis, project comparisons
- **Export functionality:** PDF reports, CSV data, presentation slides
- **Responsive design:** Works on laptop screens (1366x768 and up)

### 5. AI Chat Assistant
- **Natural language Q&A:** Ask questions about specific projects
- **Risk explanations:** "Why is Project X high risk?"
- **Recommendations:** "What should I do to reduce risk?"
- **Contextual:** Understands your portfolio and analysis results
- **Conversational memory:** Multi-turn dialogues

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
│ │XGBoost       │ │ │Prompt Engine   │ │ │Criteria Weighting    │ │
│ │SHAP Explainer│ │ │Risk Extraction │ │ │Sensitivity Analysis  │ │
│ └──────────────┘ │ │Sentiment Score │ │ └──────────────────────┘ │
│                  │ │Response Cache  │ │                          │
│                  │ └────────────────┘ │                          │
└──────────────────┴────────────────────┴──────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────────────┐
│              Data Processing & Feature Engineering               │
│  ┌──────────┐  ┌───────────┐  ┌──────────┐  ┌──────────────┐  │
│  │Data      │  │Validation │  │Feature   │  │Preprocessing │  │
│  │Loader    │  │& Cleaning │  │Engineer  │  │Pipeline      │  │
│  └──────────┘  └───────────┘  └──────────┘  └──────────────┘  │
└──────────────────────┬──────────────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────────────┐
│                         Data Sources                             │
│  ┌───────────┐  ┌──────────────┐  ┌──────────────────────────┐ │
│  │CSV Upload │  │JSON Upload   │  │Sample/Synthetic Data     │ │
│  │(PM Tools) │  │(PM Tools)    │  │(For Testing/Training)    │ │
│  └───────────┘  └──────────────┘  └──────────────────────────┘ │
└─────────────────────────────────────────────────────────────────┘
```

### Data Flow

1. **Input:** Project data (CSV/JSON) uploaded by user
2. **Preprocessing:** Data validated, cleaned, features engineered
3. **Parallel Analysis:**
   - ML models predict risk scores from structured metrics
   - LLM analyzes text comments for qualitative risks
4. **MCDA Integration:** Combines ML + LLM + metrics into unified ranking
5. **Presentation:** Dashboard displays results with visualizations
6. **Interaction:** User explores via charts and asks questions via chat

---

## 📚 Project Documentation

This repository contains comprehensive documentation organized for different audiences:

### For Developers & Implementation
- **[PROJECT_STRATEGY.md](PROJECT_STRATEGY.md)** - Complete implementation plan (16-week timeline, tech stack, testing strategy)
- **[Code Structure](#)** - See `src/` directory with detailed inline comments

### For End Users (Project Managers)
- **[USER_APPLICATION_GUIDE.md](USER_APPLICATION_GUIDE.md)** - Comprehensive user manual
  - Key use cases and workflows
  - Step-by-step instructions with screenshots
  - Data preparation guide for Jira, Azure DevOps, etc.
  - Example scenarios (portfolio assessment, single project deep-dive)
  - Troubleshooting and FAQs

### For Academic Review
- **[docs/academic/Chapter_1_Introduction.md](docs/academic/Chapter_1_Introduction.md)** - Research background, problem statement, objectives (2,450 words)
- **[docs/academic/Chapter_2_Literature_Review.md](docs/academic/Chapter_2_Literature_Review.md)** - Comprehensive literature review (5,200 words, 39 citations)
- **[docs/academic/RESEARCH_CITATIONS.md](docs/academic/RESEARCH_CITATIONS.md)** - Complete bibliography with IEEE-formatted citations, URLs, and relevance summaries

### Quick Reference
- **Installation Guide** - See [Quick Start](#quick-start) below
- **API Documentation** - Coming in Chapter 4 (Implementation)
- **Deployment Guide** - Coming in future release

---

## 🚀 Quick Start

### Prerequisites

- Python 3.10 or higher
- pip package manager
- OpenAI API key ([get one here](https://platform.openai.com/api-keys))
- Modern web browser (Chrome, Firefox, Safari, Edge)

### Installation

```bash
# 1. Clone the repository
git clone https://github.com/yourusername/prism.git
cd prism

# 2. Create virtual environment
python3.10 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# 3. Install dependencies
pip install --upgrade pip
pip install -r requirements.txt

# 4. Download NLP data
python -m nltk.downloader punkt vader_lexicon
python -m spacy download en_core_web_sm

# 5. Configure environment
cp .env.example .env
# Edit .env and add your OPENAI_API_KEY=sk-...

# 6. Run tests (optional but recommended)
pytest tests/

# 7. Launch dashboard
streamlit run app/app.py
```

### First-Time Usage

1. **Open your browser** to `http://localhost:8501`
2. **Download sample template** from the Upload Data page
3. **Upload the sample CSV** (or use your own project data)
4. **Click "Run Risk Analysis"** and wait 1-3 minutes
5. **Explore results** across different dashboard pages
6. **Ask questions** via the Chat Assistant

---

## 💼 Use Cases

### Use Case 1: Portfolio Health Check (Monthly)
**Scenario:** PMO Director manages 25 software projects, needs monthly steering committee report

**Workflow:**
1. Export all project data from Jira (5 minutes)
2. Upload to PRISM (30 seconds)
3. Review dashboard showing 4 high-risk, 11 medium-risk, 10 low-risk (2 minutes)
4. Drill into top 3 high-risk projects to understand drivers (5 minutes)
5. Generate PDF report for executives (1 minute)
6. **Total time: 15 minutes** (vs. 2-3 hours manually reviewing 25 projects)

**Value:** Data-driven prioritization, objective evidence for resource allocation, faster decisions

---

### Use Case 2: Single Project Deep-Dive (Weekly)
**Scenario:** Project Manager senses something is "off" with Project Falcon but can't pinpoint the issue

**Workflow:**
1. Export last 3 months of Falcon data including comments (2 minutes)
2. Upload to PRISM (30 seconds)
3. ML reveals: velocity declining 18%, defect rate up 40%, team turnover (+2 people left) (1 minute)
4. LLM detects: "legacy code integration issues," "knowledge gaps," "too many meetings" (1 minute)
5. Chat assistant recommends: technical debt sprint, pair programming, meeting consolidation (2 minutes)
6. Take actions, re-analyze in 3 weeks to validate improvement

**Value:** Early diagnosis before crisis, data-backed interventions, measurable validation

---

### Use Case 3: Trend Analysis (Quarterly)
**Scenario:** Program Manager wants to assess if portfolio health is improving or degrading over time

**Workflow:**
1. Upload 6 monthly snapshots (June-November)
2. Compare average risk scores: June 0.54 → November 0.42 (improving!)
3. Identify: 3 projects moved from medium to low risk (success stories)
4. Flag: 2 projects consistently high-risk (need strategic decisions)
5. Spot: 1 new emerging concern (scope creep in Project B)

**Value:** Evidence of risk mitigation effectiveness, early detection of trends, proactive intervention

---

## 🛠️ Technology Stack

### Core Technologies

| Component | Technology | Version | Purpose |
|-----------|-----------|---------|---------|
| **Language** | Python | 3.10+ | Core development |
| **ML Framework** | scikit-learn | 1.3+ | Random Forest, preprocessing, metrics |
| **Boosting** | XGBoost | 2.0+ | Gradient boosting alternative |
| **Interpretability** | SHAP | 0.43+ | Model explanations |
| **LLM API** | OpenAI | 1.3+ | GPT-3.5-turbo / GPT-4 |
| **LLM Orchestration** | LangChain | 0.1+ | Prompt management, chaining |
| **NLP** | NLTK, spaCy | 3.8+, 3.7+ | Text preprocessing, sentiment |
| **MCDA** | pymcdm | 1.1+ | TOPSIS algorithm |
| **Dashboard** | Streamlit | 1.28+ | Web UI framework |
| **Visualization** | Plotly | 5.17+ | Interactive charts |
| **Data** | pandas, numpy | 2.0+, 1.24+ | Data manipulation |
| **Testing** | pytest | 7.4+ | Unit and integration tests |

### Why These Choices?

- **Python:** Rich ML/AI ecosystem, rapid development
- **scikit-learn + XGBoost:** Industry-standard, well-validated algorithms
- **SHAP:** Most accurate and widely adopted interpretability method
- **OpenAI GPT:** Best-in-class LLM with reasonable cost ($5-20/month typical usage)
- **Streamlit:** Fastest path to interactive web dashboard without frontend expertise
- **Plotly:** Interactive charts enhance user exploration

---

## 📊 Project Status

### Current Phase: **Development & Documentation** (November 2024)

This repository currently contains:
- ✅ Complete project strategy and implementation plan
- ✅ Comprehensive user application guide
- ✅ Academic report (Introduction & Literature Review chapters)
- ✅ Research citations bibliography (39 papers)
- ✅ Detailed folder structure and architecture design
- ⏳ Code implementation (Phase 1-8, upcoming)
- ⏳ Testing and validation (upcoming)
- ⏳ User acceptance testing (upcoming)

### Development Timeline

| Phase | Duration | Deliverables | Status |
|-------|----------|-------------|--------|
| **Phase 0:** Planning & Documentation | Weeks 1-2 | ✅ Project strategy, User guide, Academic chapters 1-2 | **Complete** |
| **Phase 1:** Foundation & Setup | Weeks 3-4 | Environment setup, Data schema, Sample datasets | Upcoming |
| **Phase 2:** Data Preparation | Weeks 5-6 | EDA, Preprocessing pipeline, Synthetic data generator | Upcoming |
| **Phase 3:** ML Model Development | Weeks 7-9 | Model training, Feature engineering, Evaluation | Upcoming |
| **Phase 4:** LLM Integration | Weeks 10-11 | OpenAI API, Prompt engineering, Risk extraction | Upcoming |
| **Phase 5:** MCDA Implementation | Weeks 12-13 | TOPSIS algorithm, Ranking engine, Sensitivity analysis | Upcoming |
| **Phase 6:** Dashboard Development | Weeks 14-15 | Streamlit UI, Visualizations, File upload | Upcoming |
| **Phase 7:** Chat Assistant | Week 16 | Conversational AI, Explanations, Q&A | Upcoming |
| **Phase 8:** Testing & Documentation | Weeks 17-18 | UAT, Performance testing, Final report | Upcoming |

**Expected Completion:** January 2025

### Capstone Milestones

- ✅ **Milestone 1:** Project proposal approved
- ✅ **Milestone 2:** Literature review completed (39 citations)
- ⏳ **Milestone 3:** ML model achieving ≥75% accuracy (target: Week 9)
- ⏳ **Milestone 4:** LLM integration functional (target: Week 11)
- ⏳ **Milestone 5:** Dashboard MVP (target: Week 15)
- ⏳ **Milestone 6:** User acceptance testing (target: Week 17)
- ⏳ **Milestone 7:** Final defense (target: Week 18)

---

## 👥 Contributing

This is a final year capstone project developed by [Your Name]. While this is primarily an individual academic project, feedback and suggestions are welcome!

### How to Provide Feedback

- **Report bugs:** [Open an issue](https://github.com/yourusername/prism/issues)
- **Suggest features:** [Open a feature request](https://github.com/yourusername/prism/issues)
- **Ask questions:** [Start a discussion](https://github.com/yourusername/prism/discussions)
- **Share use cases:** Email [your.email@example.com](mailto:your.email@example.com)

### For Researchers

If you're interested in building upon this work:
- All documentation is open and detailed for reproducibility
- Code will be released under MIT license upon completion
- Research paper will be published in university repository
- Contact for collaboration opportunities: [your.email@example.com]

---

## 📄 License

This project is licensed under the **MIT License** - see the [LICENSE](LICENSE) file for details.

### What This Means

- ✅ Free to use for commercial and non-commercial purposes
- ✅ Modify and distribute as needed
- ✅ No warranty provided (as-is)
- ✅ Attribution appreciated but not required

---

## 📧 Contact

**Project Author:** [Your Name]  
**Institution:** [Your University], Department of [Computer Science / Information Systems]  
**Program:** Final Year Capstone Project (2024-2025)  
**Email:** [your.email@example.com]  
**LinkedIn:** [Your LinkedIn Profile]  
**GitHub:** [@yourusername](https://github.com/yourusername)

**Academic Supervisor:** [Supervisor Name]  
**Email:** [supervisor.email@university.edu]

---

## 🙏 Acknowledgments

### Academic Resources
- **OpenAI** for GPT API access and developer resources
- **IEEE Xplore** and **ACM Digital Library** for research paper access
- **Python & Open Source Communities** for excellent ML/AI libraries

### Inspiration
This project was inspired by:
- Personal experience managing troubled software projects
- Standish Group's CHAOS reports documenting persistent project failure rates
- Recent breakthroughs in LLMs (GPT-4, etc.) enabling new possibilities
- Desire to make advanced AI accessible to project managers

### Special Thanks
- [Supervisor Name] for guidance and support
- Project managers who provided insights into real-world pain points
- Open source contributors whose libraries power PRISM

---

## 📖 Citation

If you use PRISM or reference this work in academic publications, please cite:

```bibtex
@mastersthesis{prism2024,
  author = {Your Name},
  title = {A Hybrid Predictive Framework Integrating Machine Learning and Language Models for Software Project Risk Analysis},
  school = {Your University},
  year = {2024-2025},
  type = {Final Year Capstone Project},
  url = {https://github.com/yourusername/prism}
}
```

---

## 🗺️ Roadmap

### POC (Current Scope)
- ML risk prediction from structured data
- LLM text analysis of project comments
- MCDA-based project ranking
- Interactive dashboard with chat assistant
- File-based data upload (CSV/JSON)

### Future Enhancements (Beyond Capstone)

**Near-term (3-6 months post-graduation):**
- Direct API integration with Jira, Azure DevOps
- Continuous monitoring with automated email alerts
- Historical tracking database for longitudinal analysis
- Multi-user support with authentication

**Medium-term (6-12 months):**
- Mobile-responsive design for smartphone access
- Expanded language support (Spanish, French, German)
- Integration with Slack/Teams for in-app notifications
- Advanced visualizations (network graphs, dependency maps)

**Long-term (1-2 years):**
- Transfer learning for rapid organizational adaptation
- Federated learning across organizations (privacy-preserving)
- Integration with code analysis tools (SonarQube, etc.)
- Prescriptive analytics (not just "what" but "what to do")

---

## 🎓 Learning Outcomes

This capstone project demonstrates proficiency in:

**Technical Skills:**
- Machine learning model development and evaluation
- Large language model integration and prompt engineering
- Multi-criteria decision analysis implementation
- Full-stack web application development
- Data engineering and ETL pipelines
- Software testing and validation

**Domain Knowledge:**
- Software project management practices
- Risk management frameworks and methodologies
- Agile and traditional SDLC processes
- Project portfolio management

**Research Skills:**
- Comprehensive literature review (39 papers)
- Research methodology design
- Empirical evaluation and statistical analysis
- Academic writing and documentation

**Professional Skills:**
- Requirements analysis and system design
- User-centric design thinking
- Technical documentation
- Stakeholder communication

---

## 📌 Quick Links

- **[Project Strategy](PROJECT_STRATEGY.md)** - Implementation roadmap
- **[User Guide](USER_APPLICATION_GUIDE.md)** - End-user manual
- **[Chapter 1: Introduction](docs/academic/Chapter_1_Introduction.md)** - Research background
- **[Chapter 2: Literature Review](docs/academic/Chapter_2_Literature_Review.md)** - Related work
- **[Citations](docs/academic/RESEARCH_CITATIONS.md)** - Bibliography

---

<div align="center">

**Built with ❤️ using Python, scikit-learn, OpenAI GPT, and Streamlit**

*Transforming software project management from reactive to predictive*

[⬆ Back to Top](#prism---predictive-risk-intelligence-for-software-management)

</div>

