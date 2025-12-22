"""
LLM Analyzer Module

Analyzes project text using Large Language Models.
"""

import json
from typing import Any, Optional

from loguru import logger

try:
    from openai import OpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False


class LLMAnalyzer:
    """Analyze project text using LLMs for risk extraction."""

    DEFAULT_SYSTEM_PROMPT = """You are an expert project risk analyst. Your task is to analyze project status 
comments and identify potential risks. Be objective and evidence-based.

Focus on these risk categories:
1. Technical: Code quality, architecture, technical debt, technology issues
2. Resource: Staffing, skills, availability, team capacity
3. Schedule: Timeline concerns, delays, dependencies, blockers
4. Scope: Requirement changes, feature creep, unclear specifications
5. Budget: Cost overruns, resource allocation, financial concerns

Always provide structured output in the exact JSON format requested."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "gpt-3.5-turbo",
        temperature: float = 0.0,
        max_tokens: int = 1000,
    ):
        """
        Initialize the LLM analyzer.

        Args:
            api_key: OpenAI API key
            model: Model to use
            temperature: Temperature for generation
            max_tokens: Maximum tokens in response
        """
        if not OPENAI_AVAILABLE:
            raise ImportError("OpenAI package not installed. Run: pip install openai")

        self.api_key = api_key
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens

        self.client: Optional[OpenAI] = None
        if api_key:
            self.client = OpenAI(api_key=api_key)

    def analyze_project(
        self,
        project_name: str,
        status_comments: str,
        additional_context: Optional[str] = None,
    ) -> dict:
        """
        Analyze a single project's text for risks.

        Args:
            project_name: Name of the project
            status_comments: Status comments/updates
            additional_context: Any additional context

        Returns:
            Dict with analysis results
        """
        if not self.client:
            raise ValueError("OpenAI client not initialized. Provide API key.")

        if not status_comments or len(status_comments.strip()) < 10:
            return self._empty_result("Insufficient text for analysis")

        user_prompt = self._build_user_prompt(
            project_name, status_comments, additional_context
        )

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": self.DEFAULT_SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=self.temperature,
                max_tokens=self.max_tokens,
            )

            content = response.choices[0].message.content
            result = self._parse_response(content)

            # Add metadata
            result["project_name"] = project_name
            result["tokens_used"] = response.usage.total_tokens

            logger.debug(f"Analyzed {project_name}: {result.get('risk_level', 'unknown')}")

            return result

        except Exception as e:
            logger.error(f"Error analyzing {project_name}: {e}")
            return self._empty_result(f"Analysis error: {str(e)}")

    def _build_user_prompt(
        self,
        project_name: str,
        status_comments: str,
        additional_context: Optional[str],
    ) -> str:
        """Build the user prompt for analysis."""
        prompt = f"""Analyze the following project status comments and extract risk indicators.

Project: {project_name}
Comments: {status_comments}
"""
        if additional_context:
            prompt += f"\nAdditional Context: {additional_context}\n"

        prompt += """
Respond with a JSON object containing:
{
    "sentiment_score": <float between -1.0 and 1.0>,
    "sentiment_label": "<positive|neutral|negative>",
    "risk_level": "<low|medium|high>",
    "risk_categories": ["<list of detected risk categories>"],
    "risk_indicators": ["<specific concerns extracted from text>"],
    "key_quotes": ["<relevant quotes from the comments>"],
    "confidence": <float between 0.0 and 1.0>,
    "summary": "<one sentence summary of overall project health>"
}"""
        return prompt

    def _parse_response(self, content: str) -> dict:
        """Parse the LLM response into structured data."""
        try:
            # Try to extract JSON from response
            content = content.strip()

            # Handle markdown code blocks
            if "```json" in content:
                content = content.split("```json")[1].split("```")[0]
            elif "```" in content:
                content = content.split("```")[1].split("```")[0]

            result = json.loads(content)

            # Validate and clean result
            result["sentiment_score"] = float(result.get("sentiment_score", 0.0))
            result["confidence"] = float(result.get("confidence", 0.5))
            result["risk_level"] = str(result.get("risk_level", "medium")).lower()
            result["sentiment_label"] = str(result.get("sentiment_label", "neutral")).lower()

            # Ensure lists
            for field in ["risk_categories", "risk_indicators", "key_quotes"]:
                if not isinstance(result.get(field), list):
                    result[field] = []

            return result

        except json.JSONDecodeError as e:
            logger.warning(f"Failed to parse LLM response: {e}")
            return self._empty_result("Failed to parse response")

    def _empty_result(self, reason: str = "") -> dict:
        """Return an empty/default result."""
        return {
            "sentiment_score": 0.0,
            "sentiment_label": "neutral",
            "risk_level": "medium",
            "risk_categories": [],
            "risk_indicators": [reason] if reason else [],
            "key_quotes": [],
            "confidence": 0.0,
            "summary": reason or "No analysis available",
        }

    def analyze_batch(
        self,
        projects: list[dict],
        text_field: str = "status_comments",
        name_field: str = "project_name",
    ) -> list[dict]:
        """
        Analyze multiple projects.

        Args:
            projects: List of project dicts
            text_field: Field containing text to analyze
            name_field: Field containing project name

        Returns:
            List of analysis results
        """
        results = []

        for project in projects:
            name = project.get(name_field, "Unknown")
            text = project.get(text_field, "")

            result = self.analyze_project(name, text)
            result["project_id"] = project.get("project_id", "")
            results.append(result)

        logger.info(f"Analyzed {len(results)} projects")

        return results
