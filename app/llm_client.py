import json
import re
from typing import List, Dict, Optional
from langchain_groq import ChatGroq
from tenacity import retry, stop_after_attempt, wait_exponential

from app.config import settings
from app.models import FileHunk, Finding, FileReview

# Map file extensions to language name hints
LANGUAGE_HINTS = {
    "py": "Python", "js": "JavaScript", "jsx": "JavaScript (React)",
    "ts": "TypeScript", "tsx": "TypeScript (React)", "go": "Go", "rs": "Rust",
    "java": "Java", "kt": "Kotlin", "rb": "Ruby", "php": "PHP", "cs": "C#",
    "c": "C", "h": "C header", "cpp": "C++", "swift": "Swift", "scala": "Scala",
    "sh": "Shell", "bash": "Shell", "sql": "SQL", "html": "HTML", "css": "CSS",
    "yml": "YAML", "yaml": "YAML", "tf": "Terraform", "json": "JSON",
}

class LLMReviewer:
    def __init__(self) -> None:
        self.llm = ChatGroq(
            groq_api_key=settings.groq_api_key,
            model_name=settings.llm_model,
            temperature=0.1
        )
        self.max_json_retries = 2

    def build_system_prompt(self) -> str:
        return """You are a senior code reviewer. Examine the DIFF and reference context below.

REQUIREMENTS:
1. Examine the numbered target file changes. Report actionable issues.
2. Only report issues on lines prefixed with a line number and a "+".
3. Use the line number shown at the start of the line.
4. If there are no issues, output the literal JSON array: []
5. Respond with ONLY a JSON array of objects. Do not write any natural language, code block tags, or introductory text.

JSON Schema format:
[
  {
    "line": 42,
    "severity": "bug" | "warning" | "suggestion",
    "category": "security" | "bug" | "performance" | "style",
    "message": "A short, single-line explanation of the issue (max 140 chars)",
    "suggestion": "Optional exact python/code fix recommendation"
  }
]"""

    def build_user_prompt(self, hunk: FileHunk, context_text: str) -> str:
        ext = hunk.path.rsplit(".", 1)[-1].lower() if "." in hunk.path else ""
        language = LANGUAGE_HINTS.get(ext, "")
        lang_line = f"Language: {language}\n" if language else ""
        new_file_note = "This is a NEW file.\n" if hunk.is_new_file else ""
        truncated_note = "Note: the diff was truncated.\n" if hunk.is_truncated else ""

        return f"""File: {hunk.path}
{lang_line}{new_file_note}{truncated_note}
=== DIFF ===
{hunk.annotated_diff}

=== REFERENCE CONTEXT ===
{context_text}

Return the JSON array of findings now."""

    def extract_json_array(self, text: str) -> List:
        """Extract a valid JSON array out of conversational LLM response."""
        text = text.strip()
        # Clean markdown code blocks if present
        fence = re.search(r"```(?:json)?\s*(.*?)```", text, re.DOTALL)
        if fence:
            text = fence.group(1).strip()

        try:
            parsed = json.loads(text)
        except json.JSONDecodeError:
            # Fallback: attempt to find the first '[' and last ']'
            start = text.find("[")
            end = text.rfind("]")
            if start == -1 or end <= start:
                raise ValueError("Response contains no JSON array")
            try:
                parsed = json.loads(text[start : end + 1])
            except json.JSONDecodeError as exc:
                raise ValueError(f"Response JSON is malformed: {exc}") from exc

        if isinstance(parsed, dict):
            # If wrapped: {"findings": [...]}
            for key in ("findings", "issues", "results", "comments"):
                if isinstance(parsed.get(key), list):
                    return parsed[key]
            raise ValueError("Response is a JSON object, not a list")

        if not isinstance(parsed, list):
            raise ValueError("Response is not a JSON list")

        return parsed

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10), reraise=True)
    def _call_llm(self, messages: List[Dict]) -> str:
        """Perform raw API invocation with backoff."""
        resp = self.llm.invoke(messages)
        return resp.content.strip()

    def review_file(self, hunk: FileHunk, context_text: str) -> FileReview:
        """Reviews a single FileHunk. Implements JSON recovery loop."""
        messages = [
            {"role": "system", "content": self.build_system_prompt()},
            {"role": "user", "content": self.build_user_prompt(hunk, context_text)}
        ]

        last_error = "unknown error"
        for attempt in range(1 + self.max_json_retries):
            try:
                content = self._call_llm(messages)
            except Exception as exc:
                return FileReview(path=hunk.path, skipped=True, skip_reason=f"LLM API failure: {exc}")

            try:
                raw_findings = self.extract_json_array(content)
                findings = []
                for item in raw_findings:
                    if not isinstance(item, dict):
                        continue
                    # Pydantic validates and normalizes severity/category
                    try:
                        finding = Finding.model_validate({**item, "path": hunk.path})
                        # Validate that the line number reported is actually in the diff
                        if finding.line in hunk.commentable_lines:
                            findings.append(finding)
                    except Exception:
                        continue  # drop bad individual findings, keep valid ones

                return FileReview(path=hunk.path, findings=findings)

            except ValueError as exc:
                last_error = str(exc)
                # Conversational JSON nudge retry pattern
                messages = messages[:2] + [
                    {"role": "assistant", "content": content[:2000]},
                    {
                        "role": "user",
                        "content": "That was not a valid JSON array. Please respond with ONLY the valid JSON array of findings."
                    }
                ]
                continue

        return FileReview(
            path=hunk.path,
            skipped=True,
            skip_reason=f"LLM returned malformed JSON after {1 + self.max_json_retries} attempts. Error: {last_error}"
        )