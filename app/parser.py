from typing import Optional
from unidiff import PatchSet
from unidiff.errors import UnidiffParseError

from app.models import FileHunk

class DiffParseError(Exception):
    """Raised when a patch cannot be parsed as a unified diff."""
    pass

def build_file_hunk(
    path: str,
    patch: Optional[str],
    max_lines: int = 500,
    is_new_file: bool = False,
) -> Optional[FileHunk]:
    """
    Parses a GitHub-style per-file patch text into an annotated FileHunk object.
    Returns None if the file is not reviewable (e.g. binary or pure deletion).
    """
    if not patch or not patch.strip():
        return None

    # GitHub format doesn't always contain the standard diff file headers,
    # so we add them manually if missing to make unidiff happy.
    has_headers = patch.lstrip().startswith(("---", "diff --git"))
    text = patch if has_headers else f"--- a/{path}\n+++ b/{path}\n{patch}"
    if not text.endswith("\n"):
        text += "\n"

    try:
        patch_set = PatchSet(text)
    except UnidiffParseError as exc:
        raise DiffParseError(f"Could not parse diff for {path}: {exc}") from exc

    if not patch_set:
        raise DiffParseError(f"Diff for {path} contained no files")

    patched_file = patch_set[0]
    if len(patched_file) == 0:
        raise DiffParseError(f"Diff for {path} contained no hunks")

    annotated_lines = []
    commentable_lines = set()
    added_count = 0
    truncated = False

    for hunk in patched_file:
        header = (
            f"@@ -{hunk.source_start},{hunk.source_length} "
            f"+{hunk.target_start},{hunk.target_length} @@"
        )
        annotated_lines.append(header)
        for line in hunk:
            content = line.value.rstrip("\n")
            if line.is_added:
                annotated_lines.append(f"{line.target_line_no:>6} + {content}")
                commentable_lines.add(line.target_line_no)
                added_count += 1
            elif line.is_context:
                annotated_lines.append(f"{line.target_line_no:>6}   {content}")
                commentable_lines.add(line.target_line_no)
            elif line.is_removed:
                annotated_lines.append(f"       - {content}")

        if len(annotated_lines) >= max_lines:
            truncated = True
            break

    if added_count == 0:
        # A rename with no changes, or a pure deletion
        return None

    if len(annotated_lines) > max_lines:
        annotated_lines = annotated_lines[:max_lines]
        truncated = True

    if truncated:
        annotated_lines.append("... (diff truncated)")

    return FileHunk(
        path=path,
        annotated_diff="\n".join(annotated_lines),
        commentable_lines=commentable_lines,
        is_new_file=is_new_file,
        is_truncated=truncated,
        added_line_count=added_count,
    )
