# PRISM User Application Guide

## Predictive Risk Intelligence for Software Management

**Target Audience:** Software Project Managers, Program Managers, PMO Directors, Team Leads

**Version:** 1.0  
**Last Updated:** November 9, 2024

---

## Table of Contents

1. [Introduction](#introduction)
2. [Key Use Cases](#key-use-cases)
3. [Getting Started](#getting-started)
4. [Workflows](#workflows)
5. [Integration into PM Processes](#integration-into-pm-processes)
6. [Example Scenarios](#example-scenarios)
7. [Data Preparation Guide](#data-preparation-guide)
8. [Troubleshooting](#troubleshooting)
9. [Best Practices](#best-practices)
10. [FAQs](#faqs)

---

## Introduction

### What is PRISM?

PRISM (Predictive Risk Intelligence for Software Management) is an AI-powered system that helps project managers identify and prioritize software project risks before they become critical issues. Unlike traditional project management tools that only report current status, PRISM predicts future problems using advanced machine learning and natural language understanding.

### How Does PRISM Work?

PRISM uses a **hybrid AI approach** combining three powerful techniques:

1. **Machine Learning (ML):** Analyzes structured project data (budgets, schedules, team metrics) to predict risk scores based on historical patterns.

2. **Large Language Models (LLM):** Reads project comments, status updates, and team feedback to detect hidden concerns, sentiment, and early warning signs that numbers alone can't capture.

3. **Multi-Criteria Decision Analysis (MCDA):** Combines ML predictions, LLM insights, and key performance indicators into a single, actionable project ranking.

### What Makes PRISM Different?

| Traditional PM Tools | PRISM |
|---------------------|-------|
| ✅ Track current status | ✅ **Predict future risks** |
| ✅ Store historical data | ✅ **Learn from patterns** |
| ❌ Miss subtle warning signs in comments | ✅ **Understand natural language concerns** |
| ❌ Require manual risk assessment | ✅ **Automated, data-driven insights** |
| ❌ No cross-project comparison | ✅ **Portfolio-wide risk ranking** |

### Who Should Use PRISM?

**Primary Users:**
- **Program Managers:** Oversee multiple projects, need to prioritize attention
- **PMO Directors:** Ensure portfolio health, allocate resources strategically  
- **Project Managers:** Want early risk detection for their projects
- **Stakeholders:** Need clear, evidence-based risk assessments

**When to Use PRISM:**
- Managing 3+ concurrent projects
- Need to prioritize limited resources across portfolio
- Want early warning system for project issues
- Preparing for stakeholder reviews
- Making go/no-go decisions on troubled projects

---

## Key Use Cases

### Use Case 1: Early Risk Detection

**Problem:** By the time a project shows obvious red flags in dashboards (budget overruns, missed milestones), it's often too late for easy intervention.

**PRISM Solution:**
- Analyzes subtle patterns in data (velocity trends, small schedule slips)
- Reads team comments for early concerns ("dependencies unclear", "requirements changing")
- Flags projects as "at risk" 2-4 weeks before traditional metrics turn red

**Business Value:**
- Proactive intervention before crisis
- Reduce project failure rate by 20-30%
- Lower cost of remediation (early fixes are cheaper)

**Example:**
> *Project Alpha shows 85% completion and is only 3 days behind schedule—looks fine. But PRISM detects:*
> - *ML: Velocity has dropped 15% over last 3 sprints (trend)*
> - *LLM: Recent comments mention "testing environment issues" and "unclear acceptance criteria"*
> - *Risk Score: High (0.82)*
> - *Recommendation: Immediate review of testing process and requirements*

---

### Use Case 2: Project Comparison and Prioritization

**Problem:** When managing 10-20 projects, it's hard to objectively decide which need urgent attention versus which are truly on track.

**PRISM Solution:**
- Ranks all projects by composite risk score
- Shows side-by-side comparison of key metrics
- Highlights specific risk factors for each project
- Provides data-driven justification for resource allocation

**Business Value:**
- Optimal allocation of PM time and resources
- Transparent, defensible prioritization decisions
- Prevent "squeaky wheel" projects from drowning out truly critical issues

**Example:**
> *You manage 15 projects. PRISM ranks them:*
> 1. *Project Eagle (Risk: 0.89) - Technical debt + team turnover*
> 2. *Project Phoenix (Risk: 0.81) - Budget overrun + scope creep*
> 3. *Project Atlas (Risk: 0.76) - Schedule delays + dependency issues*
> ...*
> 15. *Project Zen (Risk: 0.21) - All metrics healthy*
>
> *You focus your week on Eagle and Phoenix, delegating Atlas to senior lead.*

---

### Use Case 3: Resource Allocation Decisions

**Problem:** Budget and staffing decisions are often based on gut feel or whoever asks loudest, not actual risk.

**PRISM Solution:**
- Identifies projects that would benefit most from additional resources
- Quantifies risk reduction potential
- Helps build business case for budget requests
- Shows impact of resource constraints on portfolio

**Business Value:**
- Data-driven budget justifications
- Maximize ROI of resource investments
- Reduce political decision-making

**Example:**
> *You have budget for 2 additional developers. PRISM shows:*
> - *Project Beta: High risk (0.84), underestimated complexity, team size below benchmark*
> - *Project Gamma: Medium risk (0.63), but risk driven by external factors (client delays), adding staff won't help*
>
> *PRISM recommends: Assign both developers to Beta. Gamma needs client engagement, not developers.*

---

### Use Case 4: Stakeholder Reporting

**Problem:** Executives want concise, accurate risk assessments without reading 50 pages of status reports.

**PRISM Solution:**
- Generates executive summary dashboards
- Provides natural language explanations of risks
- Offers "traffic light" visualizations (red/yellow/green)
- Backs up assessments with data evidence

**Business Value:**
- Clear, credible communication with executives
- Faster decision-making in steering committees
- Improved stakeholder confidence in PM capabilities

**Example:**
> *Monthly steering committee: Instead of 15 project updates, you show:*
> - ***Portfolio Health:** 2 high-risk, 5 medium-risk, 8 low-risk projects*
> - ***Top 3 Concerns:** Eagle (technical), Phoenix (budget), Atlas (schedule)*
> - ***Recommended Actions:** [specific interventions for each]*
> - ***Supporting Data:** [ML scores, key comments, trend charts]*
>
> *Meeting takes 20 minutes instead of 2 hours. Decisions are data-driven.*

---

## Getting Started

### System Requirements

**For Users:**
- Modern web browser (Chrome, Firefox, Safari, Edge)
- Internet connection
- No software installation required (web-based dashboard)

**For Administrators:**
- Python 3.10+ (if running locally)
- 2GB RAM minimum
- OpenAI API key (for LLM analysis)

### Accessing PRISM

**Option 1: Cloud Deployment** (Recommended)
1. Navigate to PRISM web URL (provided by administrator)
2. Log in with your credentials
3. Start using immediately

**Option 2: Local Installation**
1. Clone repository: `git clone https://github.com/yourorg/prism.git`
2. Install dependencies: `pip install -r requirements.txt`
3. Configure API key: Create `.env` file with `OPENAI_API_KEY=your_key`
4. Run dashboard: `streamlit run app/app.py`
5. Open browser to `http://localhost:8501`

### Quick Start (5 Minutes)

**Step 1:** Launch PRISM dashboard

**Step 2:** Navigate to "Upload Data" page

**Step 3:** Download sample CSV template

**Step 4:** Upload the sample data file

**Step 5:** View results on "Dashboard" page

**Step 6:** Explore "Rankings" to see risk-sorted projects

**Step 7:** Ask the "Chat Assistant" a question like "Why is Project Alpha high risk?"

✅ **You're ready to use PRISM!**

---

## Workflows

### Workflow 1: Uploading Project Data

**When:** Weekly or bi-weekly (align with sprint cycles)

**Steps:**

1. **Export data from your PM tool**
   - Jira: Project → Export → CSV
   - Azure DevOps: Query → Export to Excel → Save as CSV
   - Monday.com: Board → Export → Excel/CSV
   - See [Data Preparation Guide](#data-preparation-guide) for details

2. **Open PRISM dashboard**
   - Navigate to "📁 Upload Data" page

3. **Upload your file**
   - Click "Browse Files" button
   - Select your exported CSV or JSON file
   - Wait for validation (2-5 seconds)

4. **Review data quality report**
   - PRISM shows:
     - ✅ Number of projects loaded
     - ✅ Completeness score (% of required fields filled)
     - ⚠️ Warnings (missing data, outliers)
     - ❌ Errors (invalid formats, missing required fields)

5. **Handle validation issues**
   - If errors: Download error report, fix data, re-upload
   - If warnings: Proceed with analysis (may affect accuracy)
   - If all green: Continue to analysis

6. **Trigger analysis**
   - Click "Run Risk Analysis" button
   - Progress bar shows:
     - Step 1/3: ML predictions (5-10 seconds)
     - Step 2/3: LLM text analysis (1-3 minutes for 50 projects)
     - Step 3/3: MCDA ranking (1-2 seconds)

7. **View results**
   - Automatically redirected to Dashboard page
   - All results available for exploration

**Tips:**
- ✅ Use consistent file format (same columns each time)
- ✅ Upload regularly for trend analysis
- ✅ Keep project IDs consistent across uploads
- ❌ Don't mix data from different time periods in one file
- ❌ Don't modify column names from template

**Troubleshooting:**
- **Error: "Missing required columns"** → Check your CSV has all required fields from template
- **Warning: "50% of projects missing comments"** → Analysis will work, but LLM insights will be limited
- **Slow upload (>30 seconds)** → File might be too large; try splitting into batches of 100 projects

---

### Workflow 2: Interpreting Risk Predictions

**When:** After running analysis, before taking action

**Steps:**

1. **Review Dashboard Overview**
   - Navigate to "📊 Dashboard" page
   - Key metrics displayed:
     - Total projects analyzed
     - Risk distribution (pie chart: high/medium/low)
     - Average risk score
     - Top 5 high-risk projects

2. **Understand Risk Scores**
   - **Scale:** 0.0 to 1.0
     - 0.0 - 0.3: Low risk (green) ✅
     - 0.3 - 0.6: Medium risk (yellow) ⚠️
     - 0.6 - 1.0: High risk (red) ❌
   - **Interpretation:**
     - Score is probability of adverse outcome
     - 0.75 = 75% chance of delay/overrun/failure
   - **Confidence Level:**
     - Shows model certainty (low/medium/high)
     - High confidence = model is sure about prediction
     - Low confidence = borderline case or unusual project

3. **Examine ML Predictions**
   - Navigate to "🤖 ML Analysis" page
   - For each project, view:
     - **Risk Score:** Primary ML model prediction
     - **Contributing Factors:** SHAP values showing which features drove the prediction
       - Example: "Budget variance (+0.15), Velocity decline (+0.12), Team size (-0.05)"
     - **Similar Projects:** Historical projects with similar profiles and their outcomes

4. **Read LLM Insights**
   - Navigate to "💬 LLM Insights" page
   - For each project, review:
     - **Sentiment Score:** Overall tone (positive/negative/neutral)
     - **Risk Indicators:** Specific concerns extracted from text
       - Example: ["Unclear requirements", "Resource constraints", "Technical debt"]
     - **Risk Categories:** Type of risk detected
       - Technical: Code quality, architecture, tech stack issues
       - Resource: Staffing, skills, availability
       - Schedule: Timeline concerns, dependencies, blockers
       - Scope: Requirement changes, feature creep, misalignment
     - **Key Quotes:** Actual text snippets that triggered risk detection
       - Example: *"Testing environment still not ready after 3 weeks"*

5. **Check MCDA Rankings**
   - Navigate to "📈 Rankings" page
   - Projects sorted by composite score (ML + LLM + metrics)
   - View score breakdown:
     - ML Risk Score: 40% weight
     - LLM Sentiment: 25% weight
     - Schedule Performance Index: 15% weight
     - Cost Performance Index: 10% weight
     - Team Stability: 10% weight
   - See calculation details for each project

6. **Compare Projects**
   - Navigate to "🔍 Compare Projects" page
   - Select 2-4 projects for side-by-side comparison
   - View differences in:
     - Risk scores and rankings
     - Key metrics (budget, schedule, team)
     - Risk factors (what's different?)
     - LLM insights (sentiment and concerns)
   - Use this to understand why one project is riskier than another

7. **Drill Down with Questions**
   - Navigate to "💭 Chat Assistant" page
   - Ask specific questions:
     - "Why is Project Alpha high risk?"
     - "What can I do to reduce risk in Project Beta?"
     - "Which projects have resource issues?"
     - "Show me all projects with schedule delays"
   - Chat provides context-aware answers based on analysis results

**Interpretation Checklist:**
- ✅ High risk score + high confidence → Immediate attention needed
- ✅ High risk score + low confidence → Monitor closely, may be false alarm
- ✅ Medium risk + negative LLM sentiment → Investigate team morale
- ✅ High ML score + positive LLM sentiment → Check if numbers are misleading
- ⚠️ Low risk overall but specific concern flagged → Don't ignore outliers

---

### Workflow 3: Using the Chat Assistant

**When:** Need quick answers or explanations about specific projects

**Steps:**

1. **Navigate to Chat Assistant**
   - Click "💭 Chat Assistant" in sidebar
   - Chat interface appears with greeting message

2. **Ask Open-Ended Questions**
   - Example questions:
     - "What are the biggest risks in my portfolio?"
     - "Why is Project Phoenix ranked #1 in risk?"
     - "Which projects should I focus on this week?"
     - "Are there any projects with team morale issues?"
     - "Show me projects over budget"

3. **Ask Project-Specific Questions**
   - Example:
     - "What is the risk score for Project Atlas?"
     - "What factors contribute to Project Beta's risk?"
     - "Has Project Gamma improved since last analysis?"
     - "What does the LLM say about Project Delta?"

4. **Request Recommendations**
   - Example:
     - "What should I do about Project Eagle's high risk?"
     - "How can I reduce risk in Project Phoenix?"
     - "Should I add more developers to Project Atlas?"

5. **Clarify Explanations**
   - Follow up with:
     - "Can you explain SHAP values?"
     - "What does schedule performance index mean?"
     - "Why is sentiment important?"
     - "How is the risk score calculated?"

6. **Export Insights**
   - Ask: "Generate a summary report for Project Alpha"
   - Copy/paste response into email or document

**Chat Best Practices:**
- ✅ Be specific: "Project Alpha" instead of "that project"
- ✅ Ask one thing at a time: Easier for AI to understand
- ✅ Use project names from your data: Exact match is best
- ❌ Don't ask about projects not in uploaded data
- ❌ Don't expect real-time data: Chat uses most recent analysis

**Example Conversation:**

> **You:** Why is Project Phoenix high risk?
>
> **PRISM:** Project Phoenix has a risk score of 0.81 (high risk) based on the following factors:
> 1. **Cost Performance:** Currently 23% over budget with 3 months remaining
> 2. **Schedule Delays:** 2 weeks behind schedule, velocity declining 
> 3. **LLM Analysis:** Recent comments show concerns about "scope creep" and "unclear requirements"
> 4. **Team Feedback:** Negative sentiment (score: -0.42) indicating frustration
>
> **You:** What can I do to reduce the risk?
>
> **PRISM:** Based on the analysis, I recommend:
> 1. **Immediate:** Hold requirements freeze meeting with stakeholders to stop scope creep
> 2. **Short-term:** Clarify acceptance criteria for all remaining features
> 3. **Medium-term:** Consider reducing scope or extending timeline to align budget
> 4. **Team:** Address team concerns through retrospective focused on process improvements

---

### Workflow 4: Exporting Results

**When:** Need to share findings with stakeholders or archive for records

**Export Options:**

**1. Export Rankings Table**
- Location: "📈 Rankings" page
- Format: CSV, Excel
- Contents:
  - Project ID, Name
  - Risk score, Rank
  - All contributing scores
  - Risk category
- Use case: Import into other tools, offline analysis

**2. Export Dashboard Charts**
- Location: Any page with visualizations
- Format: PNG, SVG, PDF
- Method: Click "📷" icon on each chart
- Use case: Include in presentations, reports

**3. Export Project Details**
- Location: "🤖 ML Analysis" or "💬 LLM Insights"
- Format: JSON, CSV
- Contents: Full analysis results for one or more projects
- Use case: Detailed project review, audit trail

**4. Generate PDF Report**
- Location: "📊 Dashboard" page
- Button: "Generate Report"
- Contents:
  - Executive summary
  - Risk distribution charts
  - Top 10 high-risk projects with details
  - Recommendations
- Use case: Stakeholder presentations, monthly reviews

**5. Export Chat Conversation**
- Location: "💭 Chat Assistant" page
- Button: "Export Conversation"
- Format: TXT, PDF
- Use case: Document Q&A session, share insights

**Export Best Practices:**
- ✅ Include timestamp in filename (e.g., `prism_analysis_2024-11-09.csv`)
- ✅ Export both summary and details for important decisions
- ✅ Version control exports for trend analysis over time
- ✅ Store securely (may contain sensitive project info)

---

## Integration into PM Processes

### When to Use PRISM in Project Lifecycle

#### 1. Project Initiation
**Timing:** At kickoff or within first sprint

**Use PRISM to:**
- Baseline initial risk assessment
- Compare to similar past projects
- Set realistic expectations with stakeholders

**Workflow:**
- Upload initial project plan data (even if incomplete)
- Review ML prediction based on project attributes
- Use "similar projects" feature to learn from history
- Document initial risk factors

**Value:**
- Early risk awareness
- Data-driven project planning
- Historical lessons applied

---

#### 2. Sprint Planning / Monthly Reviews
**Timing:** End of each sprint or monthly

**Use PRISM to:**
- Track risk trends (is it getting better or worse?)
- Identify emerging issues before they escalate
- Prioritize risk mitigation in next sprint

**Workflow:**
- Export updated data from PM tool
- Upload to PRISM (maintains history automatically)
- Compare current risk to previous sprint
- Review LLM insights from recent comments/retrospectives
- Adjust sprint priorities based on findings

**Value:**
- Continuous risk monitoring
- Data-driven sprint planning
- Early intervention

**Integration Tip:** Schedule this 1 day before sprint planning meeting so you have fresh insights.

---

#### 3. Portfolio Review / Resource Allocation
**Timing:** Quarterly or when making staffing decisions

**Use PRISM to:**
- Compare all projects objectively
- Identify where resources will have most impact
- Build business case for budget requests

**Workflow:**
- Upload data for all portfolio projects (one file)
- Review MCDA rankings
- Use comparison feature for top vs. bottom performers
- Identify resource-constrained high-risk projects
- Generate report for executive presentation

**Value:**
- Portfolio-wide visibility
- Objective prioritization
- Resource optimization

**Integration Tip:** Combine with financial data (ROI, strategic value) for holistic prioritization.

---

#### 4. Steering Committee / Stakeholder Reporting
**Timing:** Monthly or quarterly

**Use PRISM to:**
- Provide executive summary of portfolio health
- Explain risks with data evidence
- Recommend actions with confidence

**Workflow:**
- Generate PDF report from dashboard
- Customize focus on top 3-5 high-risk projects
- Use chat to practice explaining risks
- Export key charts for slides
- Bring data to support discussions

**Value:**
- Credible risk communication
- Faster decision-making
- Reduced meeting time

**Integration Tip:** Send report 2 days before meeting so stakeholders come prepared.

---

#### 5. Project Recovery / Intervention
**Timing:** When project shows signs of trouble

**Use PRISM to:**
- Diagnose root causes of issues
- Prioritize recovery actions
- Monitor recovery effectiveness

**Workflow:**
- Upload problematic project data
- Review ML feature importance (what's driving risk?)
- Read LLM insights for team perspective
- Ask chat for recommendations
- Implement interventions
- Re-analyze after 2-4 weeks to measure improvement

**Value:**
- Data-driven diagnosis
- Targeted interventions
- Measurable recovery tracking

**Integration Tip:** Use PRISM alongside project audit or health check.

---

#### 6. Post-Mortem / Lessons Learned
**Timing:** After project completion (success or failure)

**Use PRISM to:**
- Validate if risk predictions were accurate
- Understand what contributed to outcome
- Document lessons for future projects

**Workflow:**
- Review historical PRISM analyses for the project
- Compare predictions to actual outcome
- Analyze which risk factors materialized
- Document in lessons learned database
- Use insights to improve future risk management

**Value:**
- Continuous improvement
- Organizational learning
- Model accuracy feedback

---

### How to Combine PRISM with Existing Tools

PRISM is designed to **complement**, not replace, your existing PM tools.

| Your Tool | Purpose | PRISM's Role |
|-----------|---------|--------------|
| **Jira / Azure DevOps** | Task tracking, sprint management | Risk prediction from aggregated data |
| **Microsoft Project** | Detailed scheduling, resource planning | Risk scoring to inform plan adjustments |
| **Excel / Google Sheets** | Custom reporting, ad-hoc analysis | Advanced analytics and AI insights |
| **Confluence / SharePoint** | Documentation, collaboration | Risk reports and recommendations |
| **Slack / Teams** | Communication, alerts | (Future) Risk alerts and chat integration |
| **Power BI / Tableau** | Business intelligence dashboards | AI-powered risk metrics as data source |

**Integration Pattern:**

```
PM Tool → Export Data → PRISM Analysis → Insights → Actions in PM Tool
   ↑                                                         ↓
   └─────────────────── Update & Monitor ──────────────────┘
```

**Example Workflow:**
1. Track projects in Jira (normal work)
2. Weekly: Export Jira data to CSV
3. Upload to PRISM for analysis
4. Identify high-risk projects
5. Create intervention tasks in Jira based on PRISM recommendations
6. Update project status in Jira
7. Next week: Repeat to see if risk improved

---

### Best Practices for Data Preparation

#### Frequency of Analysis

**Recommended Schedule:**

| Portfolio Size | Analysis Frequency | Rationale |
|----------------|-------------------|-----------|
| **1-5 projects** | Bi-weekly | Weekly too frequent for meaningful changes |
| **6-20 projects** | Weekly | Good balance of freshness and effort |
| **21+ projects** | Weekly (batch) | Automation helpful; stagger if needed |

**Special Circumstances:**
- **High-risk project:** Daily or every other day
- **Project in crisis:** Before/after each intervention
- **Stable portfolio:** Monthly may suffice

#### Data Quality Checklist

Before uploading to PRISM, ensure:

**Required:**
- ✅ All required fields present (see Data Schema)
- ✅ Project IDs are unique and consistent over time
- ✅ Dates are valid (start < end, no future dates for actuals)
- ✅ Numeric fields are numeric (no text in budget column)
- ✅ At least one text field with 100+ characters per project

**Recommended:**
- ✅ Recent data (within last 7 days)
- ✅ Consistent units (all budgets in USD, all hours in same scale)
- ✅ Complete data (< 20% missing values per column)
- ✅ Multiple text sources (comments, issues, feedback)

**Nice to Have:**
- ✅ Historical data (3+ months) for trend analysis
- ✅ Project outcomes labeled (for model improvement)
- ✅ Custom fields specific to your organization

#### Common Data Quality Issues

| Issue | Impact | Fix |
|-------|--------|-----|
| **Missing comments** | LLM can't analyze | Include status updates or retrospective notes |
| **Negative values in budget/hours** | ML errors | Investigate source, correct or exclude |
| **Inconsistent project IDs** | Can't track trends | Create ID mapping table, standardize |
| **All projects same status** | No variance to learn | Include diverse project states |
| **Dates in wrong format** | Parsing errors | Use ISO format YYYY-MM-DD |
| **Very long text (>5000 chars)** | LLM slow/expensive | Summarize or use recent comments only |

---

## Example Scenarios

### Scenario 1: Multi-Project Portfolio Risk Assessment

**Context:**
- You're a PMO Director managing 25 software projects
- Monthly steering committee meeting next week
- Need to identify which projects need executive attention
- Limited time to deep-dive into all 25

**Using PRISM:**

**Step 1: Data Collection (1 hour)**
- Export project data from Jira for all 25 projects
- Include fields: budget, hours, completion rate, status comments
- Save as `portfolio_2024-11-09.csv`

**Step 2: Analysis (5 minutes)**
- Upload CSV to PRISM
- Validation passes: 25 projects, 90% completeness
- Run risk analysis
- ML completes in 8 seconds
- LLM analysis completes in 2 minutes (25 projects)
- MCDA ranking generated

**Step 3: Review Results (15 minutes)**
- Dashboard shows:
  - **Portfolio Health:** 4 high-risk, 11 medium-risk, 10 low-risk
  - **Top 3 Concerns:**
    1. **Project Phoenix** (Risk: 0.87) - Budget 28% over, negative team sentiment
    2. **Project Titan** (Risk: 0.83) - Schedule delays, technical debt mounting
    3. **Project Nova** (Risk: 0.79) - Resource constraints, dependency issues

- Drill into each:
  - **Phoenix:** LLM detected "requirements keep changing", "no clear product owner"
  - **Titan:** ML shows velocity dropped 40% in last 2 sprints, SHAP values indicate code complexity is primary driver
  - **Nova:** Comments mention "waiting on Platform team", "blocked for 2 weeks"

**Step 4: Compare to Last Month (5 minutes)**
- Project Titan was medium-risk last month, now high → Escalating
- Project Apollo was high-risk last month, now medium → Recovering
- New concern: Project Nova wasn't flagged last month → Emerging issue

**Step 5: Prepare Steering Committee Report (20 minutes)**
- Generate PDF report with top 10 projects
- Create custom slide deck with:
  - Portfolio risk distribution chart
  - Top 3 project details with supporting data
  - Trend analysis (Phoenix/Titan/Nova)
  - Recommended actions for each
- Practice explanations using chat assistant

**Step 6: Steering Committee Meeting (30 minutes)**
- Present: "4 of 25 projects need immediate attention"
- Focus discussion on Phoenix, Titan, Nova
- Show data evidence (ML scores, LLM insights, trends)
- Decisions made:
  - Phoenix: Assign dedicated product owner by end of week
  - Titan: Technical lead to conduct code review, create refactoring plan
  - Nova: Escalate dependency to Platform team manager
- 15 minutes for decisions, 15 for Q&A
- No time wasted on 22 healthy projects

**Outcome:**
- Clear priorities established
- Data-driven decisions
- Meeting time reduced from typical 2 hours to 30 minutes
- All stakeholders confident in assessment

**Time Investment:**
- Total: ~2 hours (data + analysis + prep + meeting)
- Traditional approach: 6+ hours (review 25 project reports manually)
- **Time saved: 4+ hours**

---

### Scenario 2: Single Project Deep-Dive Analysis

**Context:**
- You're a Project Manager for "Project Falcon" (mobile app rewrite)
- Project is 6 months in, 12 months planned
- You feel something is "off" but can't pinpoint it
- Sprint retrospectives show team frustration but no specific blockers identified
- Want to diagnose issue before it becomes critical

**Using PRISM:**

**Step 1: Gather Comprehensive Data**
- Export last 3 months of Falcon data from Azure DevOps
- Include:
  - Sprint velocity and burndown data
  - Budget and hour tracking
  - All sprint retrospective comments
  - Bug counts and resolution times
  - Team composition changes
- Save as `falcon_history_3mo.csv`

**Step 2: Initial Analysis**
- Upload to PRISM
- **ML Risk Score:** 0.68 (high-medium threshold)
- **LLM Sentiment:** -0.35 (moderately negative)
- **MCDA Rank:** (only one project, but components analyzed)

**Step 3: Examine ML Feature Importance**
- Navigate to ML Analysis page
- SHAP values show top risk drivers:
  1. **Velocity trend:** -18% over 3 months (+0.22 risk)
  2. **Defect rate:** Increased 40% in last month (+0.18 risk)
  3. **Team turnover:** 2 developers left in last 2 months (+0.15 risk)
  4. **Code complexity:** Above threshold for maintainability (+0.10 risk)

**Insight:** Velocity decline and quality issues are primary concerns, not budget or schedule yet.

**Step 4: Read LLM Insights**
- Navigate to LLM Insights page
- **Risk Categories Detected:**
  - Technical (High): "Legacy code integration harder than expected", "API documentation missing"
  - Resource (Medium): "New team members taking time to ramp up", "Knowledge gaps"
  - Process (Medium): "Too many meetings", "Frequent context switching"

- **Key Quotes:**
  - *"Spent 3 days debugging integration issue, no documentation from old team"*
  - *"Feel like we're always firefighting instead of building new features"*
  - *"New developer still not fully productive after 1 month"*

**Insight:** Team is struggling with technical debt and knowledge transfer, affecting productivity and morale.

**Step 5: Ask Chat for Recommendations**
- Question: "What should I do to reduce risk in Project Falcon?"
- Response:
  ```
  Based on the analysis, Project Falcon's main issues are:
  1. Technical debt slowing development
  2. Knowledge loss from team turnover
  3. Ineffective onboarding for new members
  
  Recommended actions:
  IMMEDIATE (this week):
  - Schedule technical debt sprint (1-2 weeks) to document legacy code and APIs
  - Pair new developers with experienced team members
  - Cancel non-essential meetings to reduce context switching
  
  SHORT-TERM (next month):
  - Improve onboarding process (documentation, buddy system)
  - Allocate 20% of each sprint to refactoring/documentation
  - Implement knowledge sharing sessions (weekly tech talks)
  
  MONITOR:
  - Track velocity recovery (should improve after tech debt sprint)
  - Measure defect rate (should decrease with better code understanding)
  - Survey team morale after changes
  
  Re-analyze with PRISM in 3-4 weeks to assess improvement.
  ```

**Step 6: Take Action**
- Present findings to sponsor and team
- Get approval for 2-week technical debt sprint
- Implement recommendations
- Document in project wiki

**Step 7: Follow-Up Analysis (4 weeks later)**
- Upload new Falcon data
- **ML Risk Score:** 0.52 (medium) - Improved!
- **LLM Sentiment:** -0.12 (slightly negative) - Much better!
- **Velocity:** Increased 12% after tech debt sprint
- **Defect rate:** Decreased 25%
- **Team comments:** *"Much clearer what we're working with now"*, *"Less firefighting"*

**Outcome:**
- Issue diagnosed early (before budget/schedule impact)
- Data-driven interventions implemented
- Measurable improvement in 4 weeks
- Team morale recovered
- Project back on track to succeed

**Time Investment:**
- Initial analysis and planning: 2 hours
- Technical debt sprint: 80 team hours (planned investment)
- Follow-up analysis: 30 minutes
- **Value:** Potentially saved project from failure, estimated $50K+ in recovery costs avoided

---

### Scenario 3: Trend Analysis Across Time Periods

**Context:**
- You're a Program Manager overseeing 8 projects in a product line
- Want to understand if portfolio risk is increasing or decreasing over time
- Need to report on effectiveness of risk mitigation efforts
- Quarterly business review in 2 weeks

**Using PRISM:**

**Step 1: Historical Data Collection**
- Gather project exports for last 6 months
- Create monthly snapshots:
  - `portfolio_2024-06.csv` (June)
  - `portfolio_2024-07.csv` (July)
  - `portfolio_2024-08.csv` (August)
  - `portfolio_2024-09.csv` (September)
  - `portfolio_2024-10.csv` (October)
  - `portfolio_2024-11.csv` (November - current)

**Step 2: Analyze Each Time Period**
- Upload each CSV separately
- Record results in tracking spreadsheet:

| Month | Avg Risk | High-Risk Count | Medium-Risk Count | Low-Risk Count |
|-------|----------|-----------------|-------------------|----------------|
| June | 0.54 | 3 | 4 | 1 |
| July | 0.58 | 4 | 3 | 1 |
| August | 0.61 | 4 | 3 | 1 |
| September | 0.55 | 3 | 4 | 1 |
| October | 0.48 | 2 | 4 | 2 |
| November | 0.42 | 2 | 3 | 3 |

**Step 3: Identify Trends**
- **Overall Trend:** Risk declining since August (0.61 → 0.42)
- **Improvement Visible:** After September interventions
- **Success Story:** 3 projects moved from medium to low risk
- **Ongoing Concern:** 2 projects consistently high-risk (Projects X and Y)

**Step 4: Drill into Specific Projects**
- **Project X (Chronically High-Risk):**
  - June-Nov: Consistently 0.75-0.80 risk score
  - LLM: Ongoing resource constraints theme
  - Recommendation: May need structural changes (additional budget, timeline extension)

- **Project A (Improvement Success):**
  - June: 0.72 (high-risk)
  - July: 0.65 (intervention: added senior developer)
  - Sept: 0.55 (medium)
  - Nov: 0.38 (low) - Success!
  - Lesson: Quick resource augmentation effective

- **Project B (New Concern):**
  - June-Oct: Stable at 0.40-0.45 (medium-low)
  - Nov: Jumped to 0.58 (medium)
  - LLM: Recent comments show scope expansion without timeline adjustment
  - Action needed: Scope review meeting

**Step 5: Create Trend Visualizations**
- Chart 1: Average portfolio risk over time (line chart showing decline)
- Chart 2: Risk distribution by month (stacked bar chart)
- Chart 3: Individual project trajectories (multi-line chart)
- Chart 4: Correlation between interventions and risk reduction

**Step 6: Prepare Quarterly Business Review**
- **Key Messages:**
  1. Portfolio health improving (22% risk reduction June→Nov)
  2. Risk mitigation efforts working (evidence: Project A recovery)
  3. 2 projects need strategic decisions (Projects X and Y)
  4. 1 new concern emerging (Project B scope issue)

- **Supporting Evidence:**
  - PRISM trend data (quantitative)
  - Specific project examples (qualitative)
  - Before/after comparisons for interventions

- **Recommendations:**
  - Continue current approach for majority of portfolio
  - Deep-dive review for Projects X and Y with executive sponsor
  - Immediate scope clarification for Project B

**Outcome:**
- Clear narrative: "We're improving, but have specific challenges"
- Data-driven evidence of PM effectiveness
- Proactive identification of emerging issue (Project B)
- Strategic conversation about chronic risks (X, Y)
- Stakeholder confidence in portfolio management

**Time Investment:**
- Historical data collection: 2 hours
- Multi-period analysis: 1 hour
- Trend analysis and visualization: 1 hour
- Report preparation: 1 hour
- **Total: 5 hours for comprehensive 6-month portfolio analysis**

---

## Data Preparation Guide

### Exporting from Common PM Tools

#### Jira Export

**Step-by-Step:**

1. **Navigate to your project** in Jira

2. **Open Issue Navigator:**
   - Click "Filters" → "View all issues" or "Advanced search"

3. **Filter for relevant issues:**
   - JQL query: `project = "YOUR_PROJECT" AND created >= -6M`
   - Adjust time range as needed

4. **Export data:**
   - Click "⋮" (more actions) → "Export" → "CSV (Current fields)"
   - OR for custom fields: "Export" → "CSV (All fields)"

5. **For portfolio export:**
   - Use filter: `project in (PROJ1, PROJ2, PROJ3) AND updated >= -1M`
   - Export all at once

6. **Required Jira fields to include:**
   - Issue Key (Project ID)
   - Summary (Project Name)
   - Status
   - Created Date (Start Date)
   - Due Date (Planned End Date)
   - Updated Date
   - Original Estimate (Planned Hours)
   - Time Spent (Actual Hours)
   - Priority
   - Description + Comments (Text for LLM)
   - Custom fields: Budget, Team Size (if you track these)

**Jira to PRISM Field Mapping:**

| Jira Field | PRISM Field | Notes |
|------------|-------------|-------|
| Issue Key | project_id | Use project key, not issue key |
| Summary | project_name | Project level, not issue level |
| Status | status | May need aggregation for project status |
| Created | start_date | |
| Due Date | planned_end_date | |
| Resolved | actual_end_date | If completed |
| Original Estimate (sum) | planned_hours | Aggregate all issues |
| Time Spent (sum) | actual_hours | Aggregate all issues |
| Priority | priority | |
| Story Points (sum) | completion_rate | Calculate % done |
| Description + Comments | status_comments | Combine text fields |

**Aggregation for Project-Level Data:**
- Jira exports issue-level data
- PRISM needs project-level data
- Use pivot table or script to aggregate:
  - Sum hours, story points
  - Concatenate comments
  - Latest status
  - Count team members (unique assignees)

---

#### Azure DevOps Export

**Step-by-Step:**

1. **Navigate to Queries** in your Azure DevOps project

2. **Create or open a query:**
   - New Query → "Flat list of work items"
   - Filter: `[System.TeamProject] = @Project AND [System.WorkItemType] = 'Epic'` (or Feature, depending on your hierarchy)
   - Or use existing project tracking query

3. **Select columns:**
   - Click "Column Options"
   - Add: Title, State, Start Date, Target Date, Effort, Completed Work, Priority, Description, Discussion
   - Save query

4. **Export:**
   - Click "⋮" → "Export to CSV" or "Export to Excel"
   - Choose Excel for better formatting

5. **For budget data:**
   - May need separate export from Azure DevOps Analytics or custom dashboard
   - Or maintain in separate spreadsheet and merge

**Azure DevOps to PRISM Field Mapping:**

| Azure DevOps Field | PRISM Field | Notes |
|--------------------|-------------|-------|
| Work Item ID | project_id | Use Epic/Feature ID |
| Title | project_name | |
| State | status | Active, Resolved, Closed |
| Start Date | start_date | |
| Target Date | planned_end_date | |
| Completed Date | actual_end_date | |
| Original Estimate | planned_hours | |
| Completed Work | actual_hours | |
| Priority | priority | |
| Effort / Story Points | completion_rate | Calculate % done |
| Description + Discussion | status_comments | Combine |

**Tips:**
- Use **Analytics views** for easier project-level aggregation
- Export from **Boards** vs. **Work Items** for hierarchy
- Include child items (stories/tasks) for accurate metrics

---

#### Monday.com Export

**Step-by-Step:**

1. **Open your board** in Monday.com

2. **Export board:**
   - Click board menu (⋮) → "Export board to Excel"
   - All columns included automatically

3. **For multiple boards:**
   - Export each board separately
   - Combine in Excel using Power Query or manual merge

4. **Ensure columns exist:**
   - Status column
   - Timeline column (start/end dates)
   - Numbers columns (budget, hours)
   - Long text column (for comments/updates)
   - People column (team members)

**Monday.com to PRISM Field Mapping:**

| Monday.com Column | PRISM Field | Notes |
|-------------------|-------------|-------|
| Item ID | project_id | |
| Item Name | project_name | |
| Status | status | |
| Timeline (start) | start_date | |
| Timeline (end) | planned_end_date | |
| Completion | completion_rate | Use % column or calculate |
| Budget | budget | |
| Hours Logged | actual_hours | |
| Priority | priority | |
| Updates / Notes | status_comments | Export updates separately if needed |

**Tips:**
- Use **formulas** in Monday.com to calculate fields before export (e.g., SPI, CPI)
- **Updates export:** Monday.com → Board menu → "Activity log" → Export
- Combine item export + updates export for complete data

---

#### Generic CSV/Excel Export

If you use a custom system or Excel tracking:

**Minimum Required Columns:**

```csv
project_id,project_name,start_date,planned_end_date,budget,spent,planned_hours,actual_hours,team_size,completion_rate,status,priority,status_comments
PROJ-001,Mobile App Redesign,2024-01-15,2024-06-30,150000,125000,2000,2300,8,75.5,Active,High,"Sprint 12 complete. Testing environment issues delaying QA. Team concerned about unclear acceptance criteria."
PROJ-002,API Gateway Migration,2024-03-01,2024-09-30,200000,90000,3000,1500,6,45.0,Active,Medium,"On track. Successfully migrated 3 of 7 services. Performance improvements visible. Team morale high."
```

**Template Downloads:**
- Available in PRISM dashboard: "Upload Data" → "Download Template"
- Includes all required and optional fields with descriptions
- Fill in your data, save as CSV

---

### Required Fields and Optional Fields

#### Required Fields (Must Include)

| Field | Type | Description | Example |
|-------|------|-------------|---------|
| `project_id` | Text | Unique identifier, consistent across uploads | "PROJ-2024-001" |
| `project_name` | Text | Descriptive name | "Mobile App Redesign" |
| `start_date` | Date | Project start date (YYYY-MM-DD) | "2024-01-15" |
| `planned_end_date` | Date | Target completion date | "2024-06-30" |
| `budget` | Number | Total budget in dollars | 150000 |
| `spent` | Number | Current spend in dollars | 125000 |
| `planned_hours` | Number | Estimated effort in hours | 2000 |
| `actual_hours` | Number | Logged time in hours | 2300 |
| `team_size` | Number | Number of team members | 8 |
| `completion_rate` | Number | Percent complete (0-100) | 75.5 |
| `status` | Text | Current status | "Active", "On Hold", "Completed" |
| `status_comments` | Text | Recent updates/comments (100+ chars) | "Sprint 12 complete. Testing issues..." |

**Without these fields, PRISM cannot perform analysis.**

---

#### Optional but Valuable Fields

| Field | Type | Impact on Analysis | Example |
|-------|------|-------------------|---------|
| `actual_end_date` | Date | Completion tracking | "2024-07-15" |
| `priority` | Text | Risk prioritization | "High", "Medium", "Low" |
| `project_type` | Text | Similarity comparison | "New Development", "Maintenance" |
| `methodology` | Text | Process risk factors | "Agile", "Waterfall", "Hybrid" |
| `complexity_score` | Number (1-10) | ML feature | 7 |
| `client_type` | Text | Context | "Internal", "External" |
| `technology_stack` | Text | Technical risk | "Python, React, PostgreSQL" |
| `dependencies` | Number | Coordination risk | 3 |
| `velocity` | Number | Performance metric | 42 (story points/sprint) |
| `defect_rate` | Number | Quality metric | 0.08 (defects per feature) |
| `team_turnover` | Number | Stability metric | 0.125 (12.5% turnover) |
| `department` | Text | Organizational context | "Engineering", "Product" |
| `risk_level` | Text | Initial assessment | "High", "Medium", "Low" |

**Additional Text Fields (for better LLM analysis):**
- `project_description`: Initial project goals
- `issue_summaries`: Known problems
- `team_feedback`: Retrospective notes
- `stakeholder_notes`: Client comments

**The more quality data you provide, the more accurate PRISM's predictions.**

---

### Data Quality Checklist

Before uploading, verify:

**File Format:**
- [ ] File is CSV or JSON format
- [ ] CSV uses comma delimiters (not semicolon or tab)
- [ ] UTF-8 encoding (not ASCII or Latin-1)
- [ ] No special characters in header row
- [ ] Consistent column names (match template)

**Data Completeness:**
- [ ] All required fields present
- [ ] < 20% missing values per column
- [ ] At least one text field with 100+ characters per project
- [ ] Minimum 10 projects for analysis (50+ recommended)

**Data Validity:**
- [ ] Project IDs are unique (no duplicates)
- [ ] Dates in YYYY-MM-DD format
- [ ] Start date < End date for all projects
- [ ] Budget and spent are positive numbers
- [ ] Hours are positive numbers
- [ ] Completion rate between 0 and 100
- [ ] Team size > 0
- [ ] Status is one of: Active, On Hold, Completed, Cancelled

**Data Quality:**
- [ ] Recent data (within last 7 days preferred)
- [ ] Consistent units (all budgets in USD, not mixed currencies)
- [ ] No placeholder values like "TBD", "N/A", "0" where real data expected
- [ ] Text fields contain real comments, not auto-generated boilerplate

**Business Logic:**
- [ ] Completion rate aligns with status (100% = Completed)
- [ ] Spent ≤ Budget for most projects (overruns OK but verify accuracy)
- [ ] Actual hours reasonable for team size and duration
- [ ] Projects span appropriate time ranges (not all 1 week or 5 years)

**PRISM will warn you about issues, but fixing proactively saves time!**

---

### Sample CSV/JSON Templates

#### Minimal Template (Required Fields Only)

```csv
project_id,project_name,start_date,planned_end_date,budget,spent,planned_hours,actual_hours,team_size,completion_rate,status,status_comments
PROJ-001,Mobile App Redesign,2024-01-15,2024-06-30,150000,125000,2000,2300,8,75.5,Active,"Sprint 12 complete. Testing environment setup taking longer than expected. Team expressed concerns about unclear acceptance criteria from product owner. Three developers raised technical debt issues in retrospective."
PROJ-002,API Gateway Migration,2024-03-01,2024-09-30,200000,90000,3000,1500,6,45.0,Active,"Good progress on migration. Successfully moved 3 out of 7 services to new gateway. Performance metrics showing 30% improvement. Team morale is high and velocity stable at 40 points per sprint."
PROJ-003,Legacy System Decommission,2023-10-01,2024-12-31,300000,280000,4000,4200,5,85.0,Active,"Nearly complete. Final data migration scheduled for next month. Minor scope additions from compliance team causing slight delay. Budget tracking well, only 7% variance. Team ready to move to next project."
```

#### Complete Template (All Fields)

Available for download in PRISM dashboard → "Upload Data" page → "Download Complete Template" button.

---

## Troubleshooting

### Common Issues and Solutions

#### Issue: Upload Fails with "Missing Required Columns"

**Cause:** CSV doesn't include all required fields

**Solution:**
1. Download template from PRISM
2. Compare your CSV headers to template headers
3. Add missing columns (can be empty if data not available)
4. Ensure exact spelling (case-sensitive: `project_id` not `Project_ID`)

---

#### Issue: "Invalid Date Format" Error

**Cause:** Dates not in expected format

**Solution:**
- Use YYYY-MM-DD format exclusively
- Bad: "11/9/2024", "Nov 9, 2024", "9-Nov-24"
- Good: "2024-11-09"
- Excel users: Format cells as Text before entering dates to prevent auto-conversion

---

#### Issue: Risk Scores Seem Inaccurate

**Possible Causes and Solutions:**

**1. Insufficient historical data:**
- ML models need 100+ projects ideally
- With small datasets, predictions less reliable
- Solution: Use synthetic data to supplement, or rely more on LLM insights

**2. Missing text data:**
- If `status_comments` empty, LLM can't contribute
- Risk scores will be based only on numbers
- Solution: Include at least one rich text field (100+ chars) per project

**3. Data quality issues:**
- Outliers or placeholder values (e.g., budget = 1, hours = 0)
- Solution: Review data quality report, clean data, re-upload

**4. Different domain:**
- Model trained on different types of projects
- Solution: Provide feedback; model will improve with your data over time

**5. Expected low accuracy:**
- Risk prediction is probabilistic, not certain
- 70-75% accuracy is typical for complex predictions
- Solution: Use PRISM as decision support, not absolute truth

---

#### Issue: LLM Analysis Takes Too Long (>5 minutes)

**Cause:** Large number of projects or very long text

**Solution:**
- Batch uploads: Analyze 50 projects at a time instead of 200
- Truncate long text: PRISM automatically limits to first 2000 chars per project
- Check API status: OpenAI might be experiencing delays
- Be patient on first run: Subsequent analyses use caching

---

#### Issue: Chat Assistant Gives Generic Answers

**Cause:** Question too vague or project not in uploaded data

**Solution:**
- Be specific: "Why is Project Alpha high risk?" vs. "Why are projects risky?"
- Use exact project names from your data
- Ensure analysis completed before asking (check Dashboard first)
- Rephrase question if response unsatisfactory

---

#### Issue: Can't Find Exported Report

**Cause:** Browser download settings or popup blockers

**Solution:**
- Check browser downloads folder
- Disable popup blocker for PRISM domain
- Try different browser (Chrome recommended)
- Use "Export to CSV" instead of PDF if issues persist

---

### Getting Help

**Self-Service Resources:**
1. Check this User Guide (you're reading it!)
2. Review FAQs section below
3. Watch tutorial videos (if available in PRISM dashboard)
4. Try sample data to verify PRISM is working correctly

**Contact Support:**
- Email: support@prism-ai.example.com (replace with actual)
- Include:
  - Description of issue
  - Screenshot of error message
  - Sample data file (if applicable, anonymize sensitive info)
  - Steps you've already tried

**Report Bugs:**
- GitHub Issues: [github.com/yourorg/prism/issues](https://github.com/yourorg/prism/issues)
- Provide reproducible example

---

## Best Practices

### Do's ✅

1. **Upload data regularly** (weekly/bi-weekly) for trend tracking
2. **Include rich text data** (comments, feedback) for better LLM analysis
3. **Start with sample data** to learn the system before using real projects
4. **Review both ML and LLM insights** – they complement each other
5. **Export and archive results** for historical reference
6. **Use chat assistant** to understand complex predictions
7. **Combine with domain expertise** – PRISM informs, you decide
8. **Validate predictions** with outcomes to build trust
9. **Keep project IDs consistent** across uploads for tracking
10. **Check data quality report** before running analysis

### Don'ts ❌

1. **Don't ignore low confidence scores** – they indicate uncertainty
2. **Don't mix data from different time periods** in one upload
3. **Don't use PRISM as only decision factor** – it's one input
4. **Don't expect 100% accuracy** – prediction is probabilistic
5. **Don't upload sensitive data** without proper authorization
6. **Don't skip data validation** – garbage in, garbage out
7. **Don't change project IDs** between uploads (breaks trend analysis)
8. **Don't use as blame tool** – focus on improvement, not fault
9. **Don't forget to act on insights** – analysis without action is waste
10. **Don't wait for crisis** – use proactively, not reactively

---

## FAQs

### General Questions

**Q: Is PRISM a replacement for my PM tool (Jira, Azure DevOps, etc.)?**

A: No. PRISM is a **complementary analytics layer** that analyzes data from your PM tools to provide risk predictions. Continue using your PM tool for day-to-day work; use PRISM for strategic risk assessment.

---

**Q: How accurate are the predictions?**

A: Typical accuracy is 70-80% for risk classification (high/medium/low). Accuracy improves with:
- More historical data
- Higher data quality
- Consistent use over time
- Your feedback on prediction outcomes

Risk prediction is inherently uncertain – PRISM provides probabilistic assessments, not guarantees.

---

**Q: How much does PRISM cost to run?**

A: POC version costs:
- OpenAI API: ~$0.05-0.10 per project analyzed (LLM usage)
- For 50 projects/week: ~$2.50-5.00/week = ~$10-20/month
- For 200 projects/week: ~$10-20/week = ~$40-80/month

Infrastructure costs minimal (can run on laptop or cheap cloud VM).

---

**Q: Is my project data secure?**

A: POC version:
- Data stored locally (not in cloud by default)
- OpenAI API: Data sent for LLM analysis, not used for model training (per OpenAI policy)
- No data shared with third parties
- Follow your organization's data handling policies

Production version should include:
- End-to-end encryption
- Access controls and audit logging
- On-premise deployment option
- Compliance with GDPR/SOC2/etc.

---

**Q: Can I use PRISM with non-software projects?**

A: Maybe. PRISM is optimized for software projects, but the hybrid approach (ML + LLM + MCDA) could work for:
- Product development projects
- IT infrastructure projects
- Digital transformation initiatives

May need retraining ML models with your domain data for best results.

---

### Data Questions

**Q: What if I don't have historical data?**

A: Options:
1. Use PRISM's synthetic data generator to train initial model
2. Start with LLM-only mode (doesn't require historical training)
3. Begin collecting data now; PRISM improves over time
4. Use public software project datasets as baseline

Even without history, LLM analysis provides value from day one.

---

**Q: How many projects do I need for PRISM to work?**

A: Minimum:
- **For LLM analysis:** 1 project (works on any amount of text)
- **For ML analysis:** 50+ projects for training (can use synthetic data)
- **For meaningful portfolio comparison:** 5+ projects

Ideal:
- 100+ projects for robust ML model
- Mix of successful and troubled projects
- 3+ months of historical data

---

**Q: What if my comments are in another language?**

A: POC version:
- English only (OpenAI models work best with English)
- May work with Spanish, French, German (not tested)

Production version could support:
- Multi-language LLM analysis
- Translation layer before analysis
- Language-specific sentiment models

---

**Q: Can I edit the data after uploading?**

A: Not directly in PRISM. Workflow:
1. Download your current data from PRISM (if you didn't keep original)
2. Edit in Excel/CSV editor
3. Re-upload corrected version
4. PRISM will replace previous analysis

Future enhancement: In-app data editing.

---

### Technical Questions

**Q: What machine learning algorithms does PRISM use?**

A: Primary:
- **Random Forest** or **XGBoost** (ensemble decision trees)
- Selected based on accuracy during model comparison phase

Supporting:
- **SHAP** for explanations
- **SMOTE** for class imbalance
- **TOPSIS** for MCDA ranking

All implemented in scikit-learn and related libraries.

---

**Q: Which GPT model does PRISM use?**

A: Default:
- **GPT-3.5-turbo** for most LLM analysis (cost-effective)
- **GPT-4** for complex reasoning (optional, more expensive)

Configurable in settings based on budget/accuracy tradeoff.

---

**Q: How long are my analysis results stored?**

A: POC version:
- Results stored until you close browser (session-based)
- No persistent database by default

Production version:
- Results stored in database indefinitely
- Enables historical trend analysis
- Can configure retention policy

---

**Q: Can I run PRISM offline?**

A: Partially:
- ML analysis: Yes (works offline)
- LLM analysis: No (requires OpenAI API, which needs internet)
- Dashboard: Yes if running locally

Fully offline mode possible with:
- Local LLM (e.g., Llama 2, but lower quality)
- Pre-computed cache of LLM responses

---

### Usage Questions

**Q: How often should I run PRISM analysis?**

A: Recommended frequency:
- **Regular portfolio monitoring:** Weekly
- **High-risk project deep-dive:** Every 2-3 days
- **Stable projects:** Monthly
- **Before important meetings:** Day before steering committee, sprint planning, etc.

Balance: Fresh insights vs. time spent analyzing.

---

**Q: Can multiple people use PRISM at the same time?**

A: POC version:
- Local installation: One user at a time
- Cloud deployment: Multiple users, but may see each other's uploads (no user accounts)

Production version should include:
- Multi-user support with authentication
- User workspaces (isolated data)
- Role-based access control

---

**Q: What do I do if PRISM flags a project as high-risk but I disagree?**

A: Good question! Options:
1. **Investigate why:** Review ML feature importance and LLM insights. You might find concerns you missed.
2. **Check data quality:** Is the data accurate? Errors lead to wrong predictions.
3. **Consider false positive:** ~20-30% of high-risk predictions may not materialize. Monitor the project.
4. **Provide feedback:** Document your assessment and actual outcome. Helps improve model.

PRISM is a tool, not a dictator. Your expertise matters.

---

**Q: Can I customize the risk factors and weights?**

A: POC version:
- MCDA weights are configurable (see `config/mcda_config.yaml`)
- ML features are fixed (based on model training)

Production version could include:
- UI for weight adjustment
- Custom feature engineering
- Organization-specific risk models

---

**Q: Does PRISM learn from my feedback?**

A: Not automatically in POC version.

Manual improvement:
- Note prediction accuracy over time
- Retrain ML model with your historical outcomes
- Adjust LLM prompts based on error patterns

Production version could include:
- Active learning (model improves from feedback automatically)
- A/B testing of model versions
- Continuous improvement pipeline

---

### Troubleshooting Questions

**Q: Why is my file upload failing?**

A: Common causes:
1. **Format:** Not CSV or JSON → Save as CSV from Excel
2. **Size:** >10MB → Split into smaller files
3. **Encoding:** Not UTF-8 → Re-save with UTF-8 encoding
4. **Columns:** Missing required fields → Download template, compare
5. **Data:** Invalid values (negative budget, etc.) → Clean data first

Check PRISM error message for specific reason.

---

**Q: Why don't I see any LLM insights?**

A: Possible reasons:
1. **Missing text:** `status_comments` field empty → Add project comments
2. **API error:** OpenAI API key invalid or quota exceeded → Check API status
3. **Very short text:** < 20 characters → LLM needs meaningful content
4. **Processing:** Analysis still running → Wait for completion (can take 2-5 min for large portfolios)

---

**Q: Why are all my projects showing the same risk score?**

A: Causes:
1. **No variance in data:** All projects have similar metrics → Normal if portfolio is homogeneous
2. **Missing features:** Critical columns empty → Fill in more data
3. **Model not trained:** Using default model not calibrated to your data → Retrain with your projects
4. **Bug:** Submit issue with sample data

---

## Conclusion

PRISM is designed to make your job as a project manager easier and more effective by:
- Identifying risks before they become crises
- Providing objective, data-driven insights
- Saving time through automated analysis
- Improving communication with stakeholders

**Remember:** PRISM is a tool to augment your expertise, not replace it. Combine AI insights with your domain knowledge, intuition, and context for best results.

**Get Started Today:**
1. Download the sample data template
2. Export data from your PM tool
3. Upload to PRISM
4. Explore the insights
5. Take action on high-risk projects

For questions, feedback, or support: [support@prism-ai.example.com]

---

**Document Version:** 1.0  
**Last Updated:** November 9, 2024  
**For PRISM v1.0 POC**

