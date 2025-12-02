import os

SUPPORTED_EXTENSIONS = {...}
IGNORE_DIRS = {...}

def should_process_file(path):
    parts = path.split(os.sep)
    return not any(p in IGNORE_DIRS for p in parts) and os.path.splitext(path)[1] in SUPPORTED_EXTENSIONS

def get_pr_diff_and_files(env):
    pr = env.pr
    repo = env.repo

    diff_text = ""
    search_query = ""
    file_contents = {}
    patch_positions = {}

    from .diff_mapper import extract_diff_positions

    for f in pr.get_files():
        filename = f.filename
        if not should_process_file(filename):
            continue

        patch = f.patch
        diff_text += f"\n\n--- FILE: {filename} ---\n{patch}\n"
        patch_positions[filename] = extract_diff_positions(patch)

        added_lines = [line[1:] for line in patch.splitlines() if line.startswith("+") and not line.startswith("+++")]
        search_query += "\n".join(added_lines) + "\n"

        blob = repo.get_contents(filename, ref=pr.head.ref)
        file_contents[filename] = blob.decoded_content.decode("utf-8")

    return pr, diff_text, search_query, file_contents, patch_positions
