# ai_reviewer.py
import os
import json
import time
from dotenv import load_dotenv
import requests
from github import Github

# LangChain / Groq / RAG imports
from langchain_groq import ChatGroq
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import TextLoader
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings

# -------------------------------------------------------------------
# Load environment
# -------------------------------------------------------------------
load_dotenv()

GITHUB_TOKEN = os.getenv("GITHUB_TOKEN")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
REPO_NAME = os.getenv("GITHUB_REPOSITORY")  # owner/repo
PR_NUMBER = int(os.getenv("PR_NUMBER", "0"))

if not (GITHUB_TOKEN and GROQ_API_KEY and REPO_NAME and PR_NUMBER):
    raise SystemExit("Missing required env vars: GITHUB_TOKEN, GROQ_API_KEY, GITHUB_REPOSITORY, PR_NUMBER")

# -------------------------------------------------------------------
# Config
# -------------------------------------------------------------------
SUPPORTED_EXTENSIONS = {
    ".py", ".js", ".jsx", ".ts", ".tsx", ".java", ".kt", ".scala",
    ".c", ".cpp", ".h", ".cs", ".go", ".rs", ".swift",
    ".php", ".rb", ".lua", ".pl", ".sh", ".bat",
    ".html", ".css", ".sql", ".json", ".yaml", ".yml", ".toml",
}
IGNORE_DIRS = {"node_modules", "venv", "env", ".git", "__pycache__", "dist", "build", "target"}

# GitHub + LLM clients
gh = Github(GITHUB_TOKEN)
repo = gh.get_repo(REPO_NAME)
pr = repo.get_pull(PR_NUMBER)

llm = ChatGroq(groq_api_key=GROQ_API_KEY, model_name="llama-3.1-8b-instant", temperature=0)

# -------------------------------------------------------------------
# Helpers: diff -> diff_position mapping
# -------------------------------------------------------------------
def extract_diff_positions(patch_text):
    """
    Map new-file absolute line numbers -> diff position (the position index in patch)
    Returns dict: { line_in_new_file (int) : diff_position (int) }
    """
    mapping = {}
    if not patch_text:
        return mapping

    diff_position = 0
    new_file_line = None

    for line in patch_text.splitlines():
        diff_position += 1
        if line.startswith("@@"):
            # parse header: @@ -old_start,old_count +new_start,new_count @@
            try:
                parts = line.split()
                new_header = parts[2]  # +<start>,<count>
                new_start = int(new_header.split(",")[0][1:])
                new_file_line = new_start - 1
            except Exception:
                new_file_line = None
            continue

        # add-line in patch
        if line.startswith("+") and not line.startswith("+++"):
            if new_file_line is None:
                continue
            new_file_line += 1
            mapping[new_file_line] = diff_position
        elif line.startswith("-"):
            # removed line: doesn't advance new-file line count
            continue
        else:
            # context line
            if new_file_line is None:
                continue
            new_file_line += 1

    return mapping

# -------------------------------------------------------------------
# PR / diff fetching
# -------------------------------------------------------------------
def should_process_file(path):
    parts = path.split(os.sep)
    for p in parts:
        if p in IGNORE_DIRS:
            return False
    _, ext = os.path.splitext(path)
    return ext in SUPPORTED_EXTENSIONS

def get_pr_diff_and_files():
    """
    Return:
      pr (obj),
      diff_text (str) aggregated,
      search_query (str) aggregate of added lines,
      file_contents (dict: path -> full new file content string),
      patch_positions (dict: path -> {new_line: diff_position})
    """
    diff_text = ""
    search_query = ""
    file_contents = {}
    patch_positions = {}

    for f in pr.get_files():
        filename = f.filename
        _, ext = os.path.splitext(filename)
        if ext not in SUPPORTED_EXTENSIONS:
            continue

        diff_text += f"\n\n--- FILE: {filename} ---\n"
        patch = f.patch

        if patch:
            diff_text += patch + "\n"
            # compute mapping positions
            patch_positions[filename] = extract_diff_positions(patch)

            added_lines = [line[1:] for line in patch.splitlines() if line.startswith("+") and not line.startswith("+++")]
            if added_lines:
                search_query += "\n".join(added_lines) + "\n"
        else:
            # fallback: whole file is new or GitHub couldn't provide patch
            patch_positions[filename] = {}
            try:
                blob = repo.get_contents(filename, ref=pr.head.ref)
                content = blob.decoded_content.decode("utf-8")
            except Exception:
                content = ""
            diff_text += content + "\n"
            search_query += content + "\n"

        # always fetch new file contents (for line numbers)
        try:
            blob = repo.get_contents(filename, ref=pr.head.ref)
            file_contents[filename] = blob.decoded_content.decode("utf-8")
        except Exception:
            file_contents[filename] = ""

    return pr, diff_text, search_query, file_contents, patch_positions

# -------------------------------------------------------------------
# RAG index builder
# -------------------------------------------------------------------
def build_vector_store():
    print("🔄 Building RAG index (FAISS)...")
    documents = []
    for root, dirs, files in os.walk("."):
        dirs[:] = [d for d in dirs if d not in IGNORE_DIRS]
        for fname in files:
            fpath = os.path.join(root, fname)
            if not should_process_file(fpath):
                continue
            try:
                docs = TextLoader(fpath, encoding="utf-8").load()
                documents.extend(docs)
            except Exception:
                pass

    if not documents:
        print("⚠️ No documents found for RAG. Skipping vector DB.")
        return None

    splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=150)
    chunks = splitter.split_documents(documents)
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    db = FAISS.from_documents(chunks, embeddings)
    return db

def get_relevant_context(db, search_query):
    if not db or not search_query.strip():
        return ""
    docs = db.similarity_search(search_query, k=4)
    out = ""
    for d in docs:
        out += f"\n--- REFERENCE: {d.metadata.get('source','')} ---\n{d.page_content}\n"
    return out

# -------------------------------------------------------------------
# LLM review: request JSON list of issues
# -------------------------------------------------------------------
def review_with_groq(diff_text, context_text, file_contents):
    """
    Ask Groq to return a JSON array of items like:
    [
      {"path":"agent.py", "line":8, "body":"message", "severity":"major"},
      ...
    ]
    """
    # Provide numbered file contents so LLM can map line numbers reliably
    numbered = ""
    for p, text in file_contents.items():
        numbered += f"\n--- FILE: {p} ---\n"
        for i, l in enumerate(text.splitlines(), start=1):
            numbered += f"{i}: {l}\n"

    system_prompt = f"""
You are a senior code reviewer. Examine the DIFF and the NEW FILE CONTENTS below.

REQUIREMENTS (strict):
- Only review the DIFF and numbered NEW FILE CONTENTS.
- Return a JSON array (only JSON) of comment objects. Do NOT output any natural language outside the JSON.
- Each object must have:
  - path (string): file path relative to repo (e.g., 'agent.py')
  - line (integer): line number in the NEW file (1-based)
  - body (string): a short single-line comment (<= 140 chars)
  - severity (string): one of "major", "minor", or "style"

- Deduplicate similar issues: if the same typo occurs many times, report it ONCE.
- Group related typos in the same object if appropriate (comma-separated names).
- Keep comments concise.

If there are NO issues, output the literal JSON: []
"""

    user_message = f"""
=== DIFF ===
{diff_text}

=== CONTEXT ===
{context_text}

=== NEW FILE CONTENTS (NUMBERED) ===
{numbered}
"""

    # call Groq
    try:
        resp = llm.invoke(
            [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_message}
            ]
        )
        content = resp.content.strip()
        # Expect content to be JSON array. Try to parse.
        data = json.loads(content)
        if isinstance(data, list):
            return data
        else:
            # fallback: empty
            return []
    except Exception as e:
        print(f"⚠️ LLM invocation or JSON parse failed: {e}")
        # Try a safer fallback: attempt to extract JSON substring
        try:
            raw = resp.content
            start = raw.find("[")
            end = raw.rfind("]") + 1
            if start != -1 and end != -1:
                data = json.loads(raw[start:end])
                if isinstance(data, list):
                    return data
        except Exception:
            pass
        return []

# -------------------------------------------------------------------
# Deduplicate + map line -> diff position for GitHub
# -------------------------------------------------------------------
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
    For each comment, map its 'line' to a diff 'position' using patch_positions.
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
            # position unknown — fallback to file-level comment
            fallback.append({"path": path, "line": line, "body": body})
    return inline, fallback

# -------------------------------------------------------------------
# Submit batch review using GitHub Reviews API (requests)
# -------------------------------------------------------------------
def submit_batch_review(pr_number, inline_comments, summary, event="COMMENT"):
    """
    inline_comments: list of {path, position, body}
    summary: text for the review body
    event: COMMENT | REQUEST_CHANGES | APPROVE
    """
    url = f"https://api.github.com/repos/{REPO_NAME}/pulls/{pr_number}/reviews"
    headers = {
        "Authorization": f"Bearer {GITHUB_TOKEN}",
        "Accept": "application/vnd.github.v3+json"
    }

    payload = {
        "body": summary,
        "event": event,
        "comments": inline_comments
    }

    resp = requests.post(url, json=payload, headers=headers)
    if resp.status_code in (200, 201):
        print("✅ Batch review submitted.")
        return True
    else:
        print(f"❌ Batch review error {resp.status_code}: {resp.text}")
        return False

# -------------------------------------------------------------------
# Post fallback issue comments (file-level) via PyGithub
# -------------------------------------------------------------------
def post_fallback_comments(pr_obj, fallback_comments):
    for c in fallback_comments:
        path = c.get("path")
        line = c.get("line")
        body = c.get("body")
        try:
            comment_body = f"[Line {line}] {body}"
            pr_obj.create_issue_comment(f"In `{path}`: {comment_body}")
            print(f"💬 Fallback comment posted on {path} (line {line})")
        except Exception as e:
            print(f"⚠️ Failed to post fallback comment: {e}")

# -------------------------------------------------------------------
# Build concise review summary from severities
# -------------------------------------------------------------------
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

# -------------------------------------------------------------------
# Main flow
# -------------------------------------------------------------------
def main():
    print(f"🚀 Starting review for {REPO_NAME} PR #{PR_NUMBER}")

    pr_obj, diff_text, search_query, file_contents, patch_positions = get_pr_diff_and_files()

    if not diff_text.strip():
        print("⚠️ No supported code changes detected. Exiting.")
        return

    db = build_vector_store()
    context = get_relevant_context(db, search_query)

    print("🤖 Running LLM analysis...")
    raw_comments = review_with_groq(diff_text, context, file_contents)

    if not raw_comments:
        # No comments -> approve
        print("✅ LLM returned no issues. Posting APPROVE review.")
        submit_batch_review(PR_NUMBER, [], "✅ LGTM — No issues found by AI.", event="APPROVE")
        return

    # dedupe and normalize
    normalized = []
    for r in raw_comments:
        try:
            normalized.append({
                "path": r["path"],
                "line": int(r["line"]),
                "body": str(r["body"])[:700],  # truncate to be safe
                "severity": r.get("severity", "minor")
            })
        except Exception:
            continue

    normalized = dedupe_comments(normalized)

    # map to diff positions (for inline comments)
    inline_to_post, fallback_comments = map_to_diff_positions(normalized, patch_positions)

    # Build summary and decide event
    summary = build_summary(normalized)
    has_major = any(c["severity"] == "major" for c in normalized)
    event = "REQUEST_CHANGES" if has_major else "COMMENT"

    # Submit batch review with inline comments (if any)
    print(f"🚀 Submitting batch review ({len(inline_to_post)} inline comments, {len(fallback_comments)} fallbacks)...")
    success = submit_batch_review(PR_NUMBER, inline_to_post, summary, event=event)

    # Post fallback comments as file-level issue comments (if batch failed for some)
    if not success and inline_to_post:
        print("⚠️ Batch submission failed; posting inline comments individually as fallback.")
        for ic in inline_to_post:
            try:
                # Attempt PyGithub single inline comment (may require diff position)
                pr_obj.create_review_comment(body=ic["body"], commit_id=pr_obj.head.sha, path=ic["path"], position=ic["position"])
            except Exception as e:
                print(f"⚠️ Fallback single inline failed: {e}")

    # If there are fallback comments (no diff position), post as file-level comments
    if fallback_comments:
        post_fallback_comments(pr_obj, fallback_comments)

    print("✅ Review flow complete.")

if __name__ == "__main__":
    main()
