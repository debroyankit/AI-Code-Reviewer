def extract_diff_positions(patch_text):
    mapping = {}
    if not patch_text: return mapping

    diff_position = 0
    new_file_line = None

    for line in patch_text.splitlines():
        diff_position += 1

        if line.startswith("@@"):
            parts = line.split()
            new_start = int(parts[2].split(",")[0][1:])
            new_file_line = new_start - 1
            continue

        if line.startswith("+") and not line.startswith("+++"):
            new_file_line += 1
            mapping[new_file_line] = diff_position
        elif not line.startswith("-"):
            new_file_line += 1

    return mapping


def map_to_diff_positions(comments, patch_positions):
    inline, fallback = [], []
    for c in comments:
        path, line, body = c["path"], c["line"], c["body"]
        mapping = patch_positions.get(path, {})

        if line in mapping:
            inline.append({"path": path, "position": mapping[line], "body": body})
        else:
            fallback.append(c)
    return inline, fallback
