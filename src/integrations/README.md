# Integrations

This package connects PRISM to external project-management systems. The first implementation is **Jira Cloud** via OAuth 2.0 (3LO), `JiraOAuthFlow`, `JiraHttpClient`, and `JiraCloudConnector`.

## Adding another connector

1. Subclass `ProjectManagementConnector` in `base.py` and implement:

   - `test_connection()` — verify credentials and return a small identity dict.
   - `list_projects()` — return `list[ProjectSummary]`.
   - `fetch_project_data(project_key, max_issues, jql_extra="", progress=None)` — return `RawProjectData` (issues DataFrame + comments + changelog stats + descriptions) in the same shape expected by `JiraAggregator`.

2. Put **authentication in its own module** (for example `azure_devops_oauth.py` or PAT-based client), and inject it into the HTTP/connector layer. Do not mix OAuth/PAT logic with aggregation or Streamlit UI.

3. Add a thin orchestration class (like `JiraSyncService`) that takes `ProjectManagementConnector` and `JiraAggregator` so the Streamlit page stays presentation-only.

4. Register new tests with a fake connector that returns canned `RawProjectData` to preserve the Liskov contract of the ABC.
