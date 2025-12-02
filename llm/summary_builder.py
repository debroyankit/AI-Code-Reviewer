def build_summary(comments):
    if not comments:
        return "LGTM — No issues found."

    major = len([c for c in comments if c["severity"] == "major"])
    minor = len([c for c in comments if c["severity"] == "minor"])
    style = len([c for c in comments if c["severity"] == "style"])

    parts = []
    if major: parts.append(f"{major} major issue(s)")
    if minor: parts.append(f"{minor} minor")
    if style: parts.append(f"{style} style")

    return "AI Code Review — " + ", ".join(parts)
