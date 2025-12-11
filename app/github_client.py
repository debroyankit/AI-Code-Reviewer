import os
import requests
from github import Github
from app.config import GITHUB_TOKEN, REPO_NAME, IGNORE_DIRS, SUPPORTED_EXTENSIONS
from app.utils import extract_diff_positions

# Initialize GitHub Client
gh = Github(GITHUB_TOKEN)
repo = gh.get_repo(REPO_NAME)

def get_current_user_login():
    """Get the username of the authenticated token owner."""
    return gh.get_user().login

def should_process_file(path): ## takes path from rag_engine where we are passing the changed file's path here
    parts = path.split(os.sep) ##if path = "src/components/Button.tsx", parts = ["src", "components", "Button.tsx"]
    for p in parts:
        if p in IGNORE_DIRS: ##IGNORE_DIRS = ["node_modules", "dist", ".venv"]
            return False
    _, ext = os.path.splitext(path)## os.path.splitext("src/utils/helpers.py")->("src/utils/helpers", ".py"), where _ = unused part (filename) and ext = extension (".py")
    return ext in SUPPORTED_EXTENSIONS

def get_pr_diff_and_files(pr_number):
    pr = repo.get_pull(pr_number)                
    # Example: PR #5 on repo. Lets you read changed files, patches, etc.

    diff_text = ""                                
    # This will contain ALL DIFFS across all files.
    # Example output:
    # --- FILE: utils.py ---
    # @@ -10,3 +10,4 @@
    # - old line
    # + new line

    search_query = ""                             
    # Used ONLY for RAG. Contains ONLY added lines (i.e., "+ new code").

    file_contents = {}                            
    # Full NEW file content for each changed file.
    # Example: file_contents["utils.py"] = "def foo():\n  return 5"

    patch_positions = {}                          
    # Stores: {filename: {new_file_line: diff_position}}
    # Example: {"utils.py": {12: 5, 13: 6}}

    for f in pr.get_files():                      
        # Loop through each file changed in the PR.
        # Example: f.filename = "src/utils/helpers.py"

        filename = f.filename                     

        _, ext = os.path.splitext(filename)       
        # Example: "utils.py" → ext = ".py"
        if ext not in SUPPORTED_EXTENSIONS:       
            continue                               # skip json, png, lock files, etc.

        diff_text += f"\n\n--- FILE: {filename} ---\n"  
        # Add a readable file header for the LLM.

        patch = f.patch                           
        # Unified diff for this file. Example:
        # @@ -10,3 +10,4 @@
        # - old = 1
        # + new = 2

        # --------------------------- CASE: patch exists (normal case) ---------------------------
        if patch:
            diff_text += patch + "\n"             
            # Append raw diff to diff_text which will be sent to the LLM.

            patch_positions[filename] = extract_diff_positions(patch)
            # Example output: {11: 5, 12: 6}
            # Meaning: new file line 11 → diff position 5 inside the patch.

            # Grab ONLY added lines from the patch to feed RAG.
            added_lines = [
                line[1:]                           # remove the '+' sign
                for line in patch.splitlines()
                if line.startswith("+") and not line.startswith("+++")
                # skip the diff header "+++ b/utils.py"
            ]

            if added_lines:
                search_query += "\n".join(added_lines) + "\n"
                # Example added_lines:
                # ["new = compute_v2(a)", "print('v2 enabled')"]

        # --------------------------- CASE: NO patch (rare case) ---------------------------
        else:
            patch_positions[filename] = {}        
            # No diff → cannot calculate new_line → diff_position.

            try:
                blob = repo.get_contents(filename, ref=pr.head.ref)
                content = blob.decoded_content.decode("utf-8")
                # Fetch full file content from PR head branch.
            except Exception:
                content = ""

            diff_text += content + "\n"           
            # If no patch, give the LLM the full file instead.

            search_query += content + "\n"        
            # Also send entire file content for RAG.

        # --------------------------- ALWAYS fetch full new file content ---------------------------
        try:
            blob = repo.get_contents(filename, ref=pr.head.ref)
            file_contents[filename] = blob.decoded_content.decode("utf-8")
            # Example:
            # file_contents["main.py"] = "def run():\n   print('Hello')"
        except Exception:
            file_contents[filename] = ""

    # --------------------------- RETURN EVERYTHING NEEDED ---------------------------
    return pr, diff_text, search_query, file_contents, patch_positions


def submit_batch_review(pr_number, inline_comments, summary, event="COMMENT"):
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

def create_review_comment_fallback(pr_obj, path, position, body):
    """
    Fallback: Post a single review comment using PyGithub.
    Must fetch the specific Commit object first.
    """
    try:
        # Fetch the commit object for the HEAD of the PR
        commit = repo.get_commit(pr_obj.head.sha)
        pr_obj.create_review_comment(body=body, commit=commit, path=path, position=position)
        print(f"✅ Fallback inline comment posted: {path}")
    except Exception as e:
        print(f"⚠️ Fallback single inline failed: {e}")

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