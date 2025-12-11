def extract_diff_positions(patch_text):
    """
    Converts NEW file line numbers → DIFF positions for GitHub API.
    Example:
      new line 42 appears at diff line 13 → mapping[42] = 13
    """
    mapping = {}                     # final result: {new_line: diff_position}
    if not patch_text:
        return mapping               # no diff → no mapping

    diff_position = 0                # counts lines inside the patch (1,2,3…)
    new_file_line = None             # tracks current NEW file line number

    for line in patch_text.splitlines():    
        diff_position += 1           # every line in patch increments diff pos

        # --------------------------
        # HUNK HEADER: @@ -old +new @@
        # Example: @@ -10,7 +20,8 @@
        # We only care about +20,8 → new file starts at line 20
        # --------------------------
        if line.startswith("@@"):
            try:
                parts = line.split()              # ["@@", "-10,7", "+20,8", "@@"]
                new_header = parts[2]             # "+20,8"
                new_start = int(new_header.split(",")[0][1:])  # 20
                new_file_line = new_start - 1     # set to 19 so next line becomes 20
            except Exception:
                new_file_line = None              # if parsing fails
            continue                               # move to next line

        # --------------------------
        # ADDED LINE (starts with "+")
        # Example: +    print("hello")
        # This line EXISTS in the NEW file → increment new_file_line
        # And map: new_file_line → diff_position
        # --------------------------
        if line.startswith("+") and not line.startswith("+++"):
            if new_file_line is None:
                continue
            new_file_line += 1
            mapping[new_file_line] = diff_position
            continue

        # --------------------------
        # REMOVED LINE ("-")
        # Exists only in OLD file → do NOT increment new_file_line
        # --------------------------
        elif line.startswith("-"):
            continue

        # --------------------------
        # CONTEXT LINE (unchanged)
        # Exists in NEW file → increment new_file_line
        # but don't store mapping
        # Example: "     print(result)"
        # --------------------------
        else:
            if new_file_line is None:
                continue
            new_file_line += 1

    return mapping   # final dictionary used to connect LLM line numbers → GitHub diff positions


def dedupe_comments(comments):
    """Remove exact duplicates and merge same path+line comments."""
    seen = set()
    out = []
    for c in comments:
        key = (c.get("path"), int(c.get("line")), c.get("body"))
        if key in seen:
            continue
        seen.add(key)
        out.append(c)
    return out

def map_to_diff_positions(comments, patch_positions):
    """
    Returns two lists:
      inline_comments_for_batch (list of dicts with path, position, body)
      fallback_comments (list of dicts to post as issue comments if position missing)
    """
    inline = []
    fallback = []
    for c in comments:
        path = c.get("path")
        line = int(c.get("line"))
        body = c.get("body")
        mapping = patch_positions.get(path, {})
        if mapping and line in mapping:
            inline.append({"path": path, "position": mapping[line], "body": body})
        else:
            fallback.append({"path": path, "line": line, "body": body})
    return inline, fallback

def build_summary(comments):
    if not comments:
        return "✅ LGTM — No issues found by AI."
    major = [c for c in comments if c.get("severity") == "major"]
    minor = [c for c in comments if c.get("severity") == "minor"]
    style = [c for c in comments if c.get("severity") == "style"]

    parts = []
    if major:
        parts.append(f"❌ {len(major)} major issue(s)")
    if minor:
        parts.append(f"⚠️ {len(minor)} minor suggestion(s)")
    if style:
        parts.append(f"ℹ️ {len(style)} style note(s)")

    summary = "🤖 **AI Code Review** — " + ", ".join(parts) + "."
    return summary