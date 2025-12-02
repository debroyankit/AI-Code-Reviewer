from config.settings import load_env
from github_client.pr_fetcher import get_pr_diff_and_files
from github_client.diff_mapper import extract_diff_positions, map_to_diff_positions
from github_client.reviewer_submit import submit_batch_review, post_fallback_comments
from llm.rag_builder import build_vector_store
from llm.context_retriever import get_relevant_context
from llm.reviewer import review_with_groq
from llm.summary_builder import build_summary

def main():
    env = load_env()
    print(f"🚀 Starting review for {env.REPO_NAME} PR #{env.PR_NUMBER}")

    pr_obj, diff_text, search_query, file_contents, patch_positions = get_pr_diff_and_files(env)

    db = build_vector_store()
    context = get_relevant_context(db, search_query)

    raw_comments = review_with_groq(diff_text, context, file_contents, env)

    if not raw_comments:
        submit_batch_review(env.PR_NUMBER, [], "LGTM", event="APPROVE", env=env)
        return

    inline_comments, fallback = map_to_diff_positions(raw_comments, patch_positions)

    summary = build_summary(raw_comments)
    has_major = any(c["severity"] == "major" for c in raw_comments)
    event = "REQUEST_CHANGES" if has_major else "COMMENT"

    submit_batch_review(env.PR_NUMBER, inline_comments, summary, event=event, env=env)

    if fallback:
        post_fallback_comments(pr_obj, fallback)

if __name__ == "__main__":
    main()
