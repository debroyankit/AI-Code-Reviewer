from enum import Enum
from typing import List, Optional, Set
from pydantic import BaseModel, Field, field_validator

class Severity(str, Enum):
    BUG = "bug"
    WARNING = "warning"
    SUGGESTION = "suggestion"

    @property
    def emoji(self) -> str:
        return {
            Severity.BUG: "🔴",
            Severity.WARNING: "🟡",
            Severity.SUGGESTION: "🔵",
        }[self]

    @property
    def label(self) -> str:
        return {
            Severity.BUG: "Bug",
            Severity.WARNING: "Warning",
            Severity.SUGGESTION: "Suggestion",
        }[self]

class Finding(BaseModel):
    line: int = Field(gt=0, description="Line number in the new version of the file")
    severity: Severity
    category: str = Field(min_length=1)
    message: str = Field(min_length=1)
    suggestion: Optional[str] = None
    path: str = ""

    @field_validator("severity", mode="before")
    @classmethod
    def _normalize_severity(cls, v: object) -> object:
        """Tolerates creative LLM classifications."""
        if isinstance(v, str):
            lowered = v.strip().lower()
            aliases = {
                "bugs": "bug",
                "error": "bug",
                "critical": "bug",
                "major": "bug",
                "warn": "warning",
                "warnings": "warning",
                "minor": "warning",
                "suggestions": "suggestion",
                "style": "suggestion",
                "info": "suggestion",
                "nit": "suggestion",
            }
            return aliases.get(lowered, lowered)
        return v

    @field_validator("category", mode="before")
    @classmethod
    def _normalize_category(cls, v: object) -> object:
        if isinstance(v, str):
            return v.strip().lower().replace(" ", "_").replace("-", "_")
        return v

    def comment_body(self) -> str:
        """Helper to format inline GitHub comment markdown."""
        body = f"{self.severity.emoji} **{self.severity.label}** — {self.message}"
        if self.suggestion:
            body += f"\n\n```suggestion\n{self.suggestion}\n```"
        return body

class FileHunk(BaseModel):
    path: str
    annotated_diff: str
    commentable_lines: Set[int]
    is_new_file: bool = False
    is_truncated: bool = False
    added_line_count: int = 0

class FileReview(BaseModel):
    path: str
    findings: List[Finding] = Field(default_factory=list)
    skipped: bool = False
    skip_reason: Optional[str] = None

class ReviewResult(BaseModel):
    file_reviews: List[FileReview] = Field(default_factory=list)
    files_reviewed: int = 0
    files_skipped: int = 0
    model: str = ""

    @property
    def findings(self) -> List[Finding]:
        return [f for fr in self.file_reviews for f in fr.findings]

    def count(self, severity: Severity) -> int:
        return sum(1 for f in self.findings if f.severity == severity)
