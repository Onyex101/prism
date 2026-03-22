"""
LLM Analyzer Module
===================

This module analyzes project text using Large Language Models.

It extracts risk indicators, sentiment, and categorizes risks from
project status comments and documentation.

Example:
    >>> from src.models.llm.analyzer import LLMAnalyzer
    >>> analyzer = LLMAnalyzer(api_key="sk-...")
    >>> result = analyzer.analyze_project("Project X", "Status update text...")
"""

import json
import time
from typing import Any, Callable, Optional

from loguru import logger

try:
    from openai import OpenAI, OpenAIError

    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False
    OpenAIError = type("OpenAIError", (Exception,), {})  # noqa: invalid-name


class LLMAnalyzer:
    """
    Analyze project text using LLMs for risk extraction.

    This class follows the Single Responsibility Principle by focusing
    solely on LLM-based text analysis for risk detection.

    :cvar DEFAULT_SYSTEM_PROMPT: Default system prompt for risk analysis.
    :vartype DEFAULT_SYSTEM_PROMPT: str

    :ivar api_key: OpenAI API key.
    :vartype api_key: Optional[str]
    :ivar model: Model name to use.
    :vartype model: str
    :ivar temperature: Generation temperature.
    :vartype temperature: float
    :ivar max_tokens: Maximum tokens in response.
    :vartype max_tokens: int
    :ivar client: OpenAI client instance.
    :vartype client: Optional[OpenAI]

    Example:
        >>> analyzer = LLMAnalyzer(api_key="sk-...", model="gpt-4")
        >>> result = analyzer.analyze_project("My Project", "Behind schedule...")
        >>> print(result["risk_level"])
    """

    DEFAULT_SYSTEM_PROMPT: str = """You are an expert project risk analyst. Your task is to analyze project status 
comments and identify potential risks. Be objective and evidence-based.

Focus on these risk categories:
1. Technical: Code quality, architecture, technical debt, technology issues
2. Resource: Staffing, skills, availability, team capacity
3. Schedule: Timeline concerns, delays, dependencies, blockers
4. Scope: Requirement changes, feature creep, unclear specifications

Always provide structured output in the exact JSON format requested."""

    # ~3000 tokens of input budget for status_comments (leaves room for prompt + response)
    MAX_COMMENT_CHARS: int = 12_000

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "gpt-4.1",
        temperature: float = 0.0,
        max_tokens: int = 1000,
        request_delay: float = 0.5,
    ) -> None:
        """
        Initialize the LLM analyzer.

        :param api_key: OpenAI API key.
        :type api_key: Optional[str]
        :param model: Model to use (e.g., "gpt-4o-mini", "gpt-4o").
        :type model: str
        :param temperature: Temperature for generation (0.0 for deterministic).
        :type temperature: float
        :param max_tokens: Maximum tokens in response.
        :type max_tokens: int
        :param request_delay: Seconds to wait between API calls (rate limiting).
        :type request_delay: float
        :raises ImportError: If OpenAI package not installed.
        """
        if not OPENAI_AVAILABLE:
            raise ImportError("OpenAI package not installed. Run: pip install openai")

        self.api_key = api_key
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.request_delay = request_delay

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

        :param project_name: Name of the project.
        :type project_name: str
        :param status_comments: Status comments/updates to analyze.
        :type status_comments: str
        :param additional_context: Any additional context.
        :type additional_context: Optional[str]
        :return: Dict with analysis results including sentiment, risk level,
            categories, indicators, quotes, confidence, and summary.
        :rtype: dict
        :raises ValueError: If OpenAI client not initialized.

        Example:
            >>> result = analyzer.analyze_project(
            ...     "Mobile App",
            ...     "Delays due to API integration issues. Team morale low."
            ... )
            >>> print(result["risk_categories"])
        """
        self._validate_client()

        if not self._is_valid_text(status_comments):
            return self._empty_result("Insufficient text for analysis")

        # Truncate to MAX_COMMENT_CHARS to stay within model context limits.
        # status_comments can reach 784k chars for large Apache projects.
        truncated = self._truncate_text(status_comments, self.MAX_COMMENT_CHARS)

        user_prompt = self._build_user_prompt(
            project_name, truncated, additional_context
        )

        try:
            return self._call_api(project_name, user_prompt)
        except (OpenAIError, json.JSONDecodeError) as e:
            logger.error(f"Error analyzing {project_name}: {e}")
            return self._empty_result(f"Analysis error: {str(e)}")

    def _validate_client(self) -> None:
        """
        Validate that OpenAI client is initialized.

        :raises ValueError: If client not initialized.
        """
        if not self.client:
            raise ValueError("OpenAI client not initialized. Provide API key.")

    def _is_valid_text(self, text: str) -> bool:
        """
        Check if text is valid for analysis.

        :param text: Text to validate.
        :type text: str
        :return: Whether text is valid.
        :rtype: bool
        """
        # Coerce to str first — pandas reads missing status_comments as float NaN,
        # which has no .strip() method and would raise AttributeError.
        if not isinstance(text, str):
            text = "" if text != text else str(text)  # NaN check via self-inequality
        return bool(text and len(text.strip()) >= 10)

    def _truncate_text(self, text: str, max_chars: int) -> str:
        """
        Truncate text to a maximum character length.

        Truncates at a sentence boundary when possible to preserve coherence.

        :param text: Text to truncate.
        :type text: str
        :param max_chars: Maximum number of characters.
        :type max_chars: int
        :return: Truncated text.
        :rtype: str
        """
        if len(text) <= max_chars:
            return text

        truncated = text[:max_chars]
        last_sentence = truncated.rfind(". ")
        if last_sentence > max_chars * 0.8:
            truncated = truncated[: last_sentence + 1]

        logger.debug(
            f"Truncated text from {len(text):,} to {len(truncated):,} chars"
        )
        return truncated + " [truncated]"

    def _call_api(self, project_name: str, user_prompt: str) -> dict:
        """
        Call the OpenAI API and process response.

        :param project_name: Project name for metadata.
        :type project_name: str
        :param user_prompt: User prompt to send.
        :type user_prompt: str
        :return: Parsed analysis result.
        :rtype: dict
        """
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

        result["project_name"] = project_name
        result["tokens_used"] = response.usage.total_tokens

        logger.debug(f"Analyzed {project_name}: {result.get('risk_level', 'unknown')}")

        return result

    def _build_user_prompt(
        self,
        project_name: str,
        status_comments: str,
        additional_context: Optional[str],
    ) -> str:
        """
        Build the user prompt for analysis.

        :param project_name: Project name.
        :type project_name: str
        :param status_comments: Status comments.
        :type status_comments: str
        :param additional_context: Additional context.
        :type additional_context: Optional[str]
        :return: Formatted user prompt.
        :rtype: str
        """
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
        """
        Parse the LLM response into structured data.

        :param content: Raw response content.
        :type content: str
        :return: Parsed and validated result.
        :rtype: dict
        """
        try:
            content = self._extract_json(content)
            result = json.loads(content)
            return self._validate_result(result)
        except json.JSONDecodeError as e:
            logger.warning(f"Failed to parse LLM response: {e}")
            return self._empty_result("Failed to parse response")

    def _extract_json(self, content: str) -> str:
        """
        Extract JSON from response, handling markdown code blocks.

        :param content: Raw content.
        :type content: str
        :return: Extracted JSON string.
        :rtype: str
        """
        content = content.strip()

        if "```json" in content:
            content = content.split("```json")[1].split("```")[0]
        elif "```" in content:
            content = content.split("```")[1].split("```")[0]

        return content

    def _validate_result(self, result: dict) -> dict:
        """
        Validate and clean the parsed result.

        :param result: Raw parsed result.
        :type result: dict
        :return: Validated result.
        :rtype: dict
        """
        result["sentiment_score"] = float(result.get("sentiment_score", 0.0))
        result["confidence"] = float(result.get("confidence", 0.5))
        result["risk_level"] = str(result.get("risk_level", "medium")).lower()
        result["sentiment_label"] = str(result.get("sentiment_label", "neutral")).lower()

        for field in ["risk_categories", "risk_indicators", "key_quotes"]:
            if not isinstance(result.get(field), list):
                result[field] = []

        return result

    def _empty_result(self, reason: str = "") -> dict:
        """
        Return an empty/default result.

        :param reason: Reason for empty result.
        :type reason: str
        :return: Default result dictionary.
        :rtype: dict
        """
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
        *,
        on_progress: Optional[Callable[[int, int, str, dict[str, Any]], None]] = None,
    ) -> list[dict]:
        """
        Analyze multiple projects.

        :param projects: List of project dicts.
        :type projects: list[dict]
        :param text_field: Field containing text to analyze.
        :type text_field: str
        :param name_field: Field containing project name.
        :type name_field: str
        :param on_progress: Optional callback ``(done_count, total, project_name, result)``
            invoked after each project completes (for UI progress bars).
        :type on_progress: Optional[Callable[[int, int, str, dict[str, Any]], None]]
        :return: List of analysis results.
        :rtype: list[dict]

        Example:
            >>> projects = [{"project_name": "A", "status_comments": "..."}]
            >>> results = analyzer.analyze_batch(projects)
        """
        results = []
        total = len(projects)

        for i, project in enumerate(projects):
            name = project.get(name_field, "Unknown")
            raw = project.get(text_field, "")
            # Convert NaN / None to empty string so analyze_project always receives str
            text = "" if (raw != raw or raw is None) else str(raw)

            result = self.analyze_project(name, text)
            result["project_id"] = project.get("project_id", "")
            results.append(result)

            if on_progress is not None:
                on_progress(i + 1, total, name, result)

            if self.request_delay > 0 and i < len(projects) - 1:
                time.sleep(self.request_delay)

        logger.info(f"Analyzed {len(results)} projects")

        return results
