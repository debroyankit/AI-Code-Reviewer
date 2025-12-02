"""
DEBUG VERSION OF AI REVIEWER
This script prints EVERYTHING so you can see what is failing.
"""

from config.settings import load_env
from github_client.pr_fetcher import get_pr_diff_and_files
from llm.rag_builder import build_vector_store
from llm.context_retriever import get_relevant_context
from llm.reviewer import review_with_groq
from llm.summary_builder import build_summary
from github_client.diff_mapper import map_to_diff_positions
from github_client.reviewer_submit import submit_batch_review, post_fallback_comments

def main():
    print("\n================= DEBUG MODE START =================\n")

    # 1) Load environment
    env = load_env()

    print("[DEBUG] Loaded ENV:")
    print("  GITHUB_REPOSITORY =", env.REPO_NAME)
    print("  PR_NUMBER         =", env.PR_NUMBER)
    print("  TOKEN begins with =", env.GITHUB_TOKEN[:6], "...")

    # 2) Fetch PR files + diff
    print("\n[DEBUG] Fetching PR info...\n")
    pr_obj, diff_text, search_query, file_contents, patch_positions = get_pr_diff_and_files(env)

    print("[DEBUG] PR Title:", pr_obj.title)
    print("[DEBUG] Total files in PR:", len(list(pr_obj.get_files())))

    filenames = list(file_contents.keys())
    print("[DEBUG] Processed files:", filenames)

    print("\n[DEBUG] RAW DIFF TEXT:")
    print("---------------------------------------------------")
    print(diff_text if diff_text.strip() else "(EMPTY)")
    print("---------------------------------------------------\n")

    if not diff_text.strip():
        print("❌ DEBUG: diff_text is EMPTY → Script will not post review!")
        print("Possible reasons:")
        print(" - Unsupported file types")
        print(" - Patch missing")
        print(" - Incorrect PR_NUMBER or REPO_NAME")
        print(" - GitHub API permission issue")
        print("\n================= DEBUG MODE END =================\n")
        return

    # 3) Build RAG
    print("[DEBUG] Building FAISS vector store...")
    db = build_vector_store()

    print("[DEBUG] Vector store:", "Built" if db else "None")

    # 4) Retrieve context
    print("\n[DEBUG] Searching RAG context for added code...")
    print("[DEBUG] search_query =", search_query[:200], "...")
    context = get_relevant_context(db, search_query)

    print("[DEBUG] Context Retrieved:")
    print(context if context.strip() else "(EMPTY)")

    # 5) LLM Review
    print("\n[DEBUG] Sending diff + context to LLM...\n")
    comments = review_with_groq(diff_text, context, file_contents, env)

    print("[DEBUG] Raw LLM comments:")
    print(comments)

    if not comments:
        print("⚠️ DEBUG: LLM returned empty list → Will APPROVE with no review")
        print("\n================= DEBUG MODE END =================\n")
        submit_batch_review(env.PR_NUMBER, [], "LGTM — No issues found.", event="APPROVE", env=env)
        return

    # 6) Diff → Position mapping
    print("\n[DEBUG] Mapping LLM 'line' numbers to GitHub 'diff positions'...")
    inline, fallback = map_to_diff_positions(comments, patch_positions)

    print("[DEBUG] Inline comments =", inline)
    print("[DEBUG] Fallback comments =", fallback)

    # 7) Build summary
    summary = build_summary(comments)
    print("\n[DEBUG] Summary to send:", summary)

    # 8) Submit review
    print("\n[DEBUG] Attempting batch review submission...\n")
    success = submit_batch_review(env.PR_NUMBER, inline, summary, event="COMMENT", env=env)

    print("[DEBUG] Batch submit success:", success)

    # 9) Fallback
    if fallback:
        print("[DEBUG] Posting fallback comments...")
        post_fallback_comments(pr_obj, fallback)

    print("\n================= DEBUG MODE END =================\n")


if __name__ == "__main__":
    main()
