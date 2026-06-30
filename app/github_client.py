import sys
from typing import List, Optional, Tuple, Dict
from github import Github, Auth
from github.PullRequest import PullRequest
from github.File import File
from tenacity import retry, stop_after_attempt, wait_exponential

from app.config import settings
from app.models import FileHunk, Finding
from app.parser import build_file_hunk

class GitHubClient:
    def __init__(self) -> None:
        auth = Auth.Token(settings.github_token)
        self._gh = Github(auth=auth)
        self._repo = self._gh.get_repo(settings.repo_name)

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10), reraise=True)
    def get_pr(self, pr_number: int) -> Optional[PullRequest]:
        """Fetch the PR object. Checks draft/WIP status."""
        try:
            pr = self._repo.get_pull(pr_number)
        except Exception as e:
            print(f"❌ Failed to fetch PR #{pr_number}: {e}", file=sys.stderr)
            raise e

        if pr.draft:
            print("⚠️ PR is a Draft. Skipping review.")
            return None

        title_lower = pr.title.lower() if pr.title else ""
        if "wip" in title_lower or "do not merge" in title_lower:
            print("⚠️ PR is marked WIP. Skipping review.")
            return None

        return pr

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10), reraise=True)
    def fetch_pr_files(self, pr: PullRequest) -> List[File]:
        """Fetch files modified in the PR."""
        return list(pr.get_files())

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10), reraise=True)
    def fetch_file_content(self, filename: str, ref: str) -> str:
        """Fetch full file content from a specific git ref."""
        try:
            blob = self._repo.get_contents(filename, ref=ref)
            if isinstance(blob, list):
                return ""
            return blob.decoded_content.decode("utf-8")
        except Exception:
            return ""

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10), reraise=True)
    def get_existing_review_comments(self, pr: PullRequest) -> List[str]:
        """Returns a list of comments already posted by the current user to avoid duplicates."""
        try:
            current_user = self._gh.get_user().login
        except Exception:
            current_user = "github-actions[bot]"
        existing = []
        for comment in pr.get_review_comments():
            if comment.user.login == current_user:
                # Safely extract line number to avoid AttributeError
                line = None
                try:
                    line = comment.line
                except AttributeError:
                    pass
                if line is None:
                    line = comment.raw_data.get("line") or comment.position or 0
                existing.append(f"{comment.path}:{line}:{comment.body}")
        return existing


    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10), reraise=True)
    def submit_review(self, pr: PullRequest, body: str, event: str, comments: List[Dict]) -> bool:
        """Submits a single review using the public PyGithub API."""
        try:
            # comments param shape: [{"path": str, "line": int, "body": str}]
            # PyGithub converts this internally to the right format.
            review_comments = []
            for c in comments:
                review_comments.append({
                    "path": c["path"],
                    "line": c["line"],
                    "body": c["body"]
                })

            pr.create_review(
                body=body,
                event=event,
                comments=review_comments
            )
            print("✅ PR Review submitted successfully.")
            return True
        except Exception as e:
            print(f"❌ Failed to submit PR Review: {e}", file=sys.stderr)
            return False

    def get_current_user(self) -> str:
        try:
            return self._gh.get_user().login
        except Exception:
            return "github-actions[bot]"

    def should_process_file(self, path: str) -> bool:
        import os
        parts = path.split(os.sep)
        for p in parts:
            if p in settings.ignore_dirs:
                return False
        _, ext = os.path.splitext(path)
        return ext in settings.supported_extensions

    def prepare_hunks(self, files: List[File]) -> Tuple[List[FileHunk], List[Tuple[str, str]]]:
        """Parses modified files into FileHunks, skipping non-code/deleted/ignored files."""
        hunks = []
        skipped = []

        for f in files:
            if f.status == "removed":
                continue

            if not self.should_process_file(f.filename):
                continue

            if len(hunks) >= settings.max_files_per_pr:
                skipped.append((f.filename, f"PR exceeds limit of {settings.max_files_per_pr} files"))
                continue

            try:
                hunk = build_file_hunk(
                    f.filename,
                    f.patch,
                    max_lines=settings.max_lines_per_file,
                    is_new_file=f.status == "added"
                )
                if hunk:
                    hunks.append(hunk)
            except Exception as e:
                skipped.append((f.filename, f"Failed to parse diff: {e}"))

        return hunks, skipped