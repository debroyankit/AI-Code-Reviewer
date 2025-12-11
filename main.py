from app.config import REPO_NAME, PR_NUMBER, check_env_vars
from app.github_client import (
    get_pr_diff_and_files, 
    submit_batch_review, 
    post_fallback_comments, 
    get_current_user_login,
    create_review_comment_fallback
)
from app.rag_engine import build_vector_store, get_relevant_context
from app.llm_client import review_with_groq
from app.utils import dedupe_comments, map_to_diff_positions, build_summary

def main():
    check_env_vars()
    print(f"🚀 Starting review for {REPO_NAME} PR #{PR_NUMBER}")

    pr_obj, diff_text, search_query, file_contents, patch_positions = get_pr_diff_and_files(PR_NUMBER)

    if not diff_text.strip():
        print("⚠️ No supported code changes detected. Exiting.")
        return

    db = build_vector_store()
    context = get_relevant_context(db, search_query)

    print("🤖 Running LLM analysis...")
    raw_comments = review_with_groq(diff_text, context, file_contents)

    if not raw_comments:
        print("✅ LLM returned no issues. Posting APPROVE review.")
        # If self-review, we cannot APPROVE. Use COMMENT.
        current_user = get_current_user_login()
        event = "APPROVE"
        if current_user == pr_obj.user.login:
            print(f"ℹ️ User {current_user} is PR author. Switching 'APPROVE' to 'COMMENT'.")
            event = "COMMENT"
            
        submit_batch_review(PR_NUMBER, [], "✅ LGTM — No issues found by AI.", event=event)
        return

    # Normalize comments
    normalized = []
    for r in raw_comments:
        try:
            normalized.append({
                "path": r["path"],
                "line": int(r["line"]),
                "body": str(r["body"])[:700],
                "severity": r.get("severity", "minor")
            })
        except Exception:
            continue

    normalized = dedupe_comments(normalized)

    # map to diff positions
    inline_to_post, fallback_comments = map_to_diff_positions(normalized, patch_positions)

    # Build summary
    summary = build_summary(normalized)
    has_major = any(c["severity"] == "major" for c in normalized)
    
    # Determine Event Type (APPROVE vs REQUEST_CHANGES vs COMMENT)
    event = "REQUEST_CHANGES" if has_major else "COMMENT"
    
    # Check if self-review (cannot request changes on own PR)
    current_user = get_current_user_login()
    if current_user == pr_obj.user.login:
        print(f"ℹ️ User {current_user} is PR author. Forcing event to 'COMMENT' (cannot Request Changes on own PR).")
        event = "COMMENT"

    print(f"🚀 Submitting batch review ({len(inline_to_post)} inline comments, {len(fallback_comments)} fallbacks)...")
    success = submit_batch_review(PR_NUMBER, inline_to_post, summary, event=event)

    # Fallbacks for Batch Failures
    if not success and inline_to_post:
        print("⚠️ Batch submission failed; posting inline comments individually as fallback.")
        for ic in inline_to_post:
            # Uses the new helper that fetches the Commit object correctly
            create_review_comment_fallback(pr_obj, ic["path"], ic["position"], ic["body"])

    if fallback_comments:
        post_fallback_comments(pr_obj, fallback_comments)

    print("✅ Review flow complete.")

if __name__ == "__main__":
    main()