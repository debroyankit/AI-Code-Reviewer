import sys
from typing import List

from app.config import settings
from app.github_client import GitHubClient
from app.llm_client import LLMReviewer
from app.rag_engine import build_vector_store, get_relevant_context
from app.models import ReviewResult, FileReview, Severity
from app.utils import dedupe_findings, build_summary

def main():
    print(f"🚀 Starting AI Code Reviewer for {settings.repo_name} PR #{settings.pr_number}")
    
    # 1. Initialize API Clients
    gh_client = GitHubClient()
    llm_reviewer = LLMReviewer()

    # 2. Fetch Pull Request and check eligibility
    pr = gh_client.get_pr(settings.pr_number)
    if not pr:
        print("✅ Review skipped based on draft/WIP criteria.")
        sys.exit(0)

    # 3. Fetch changed files
    files = gh_client.fetch_pr_files(pr)
    if not files:
        print("⚠️ No modified files detected in the PR. Exiting.")
        sys.exit(0)

    # 4. Prepare File Hunks (filtering out ignored directories and non-code files)
    hunks, parse_skips = gh_client.prepare_hunks(files)
    if not hunks:
        print("✅ No supported code changes to review. Exiting.")
        sys.exit(0)

    # 5. Build RAG Vector Store from codebase checkout
    db = build_vector_store(gh_client)

    # 6. Perform file-by-file reviews
    file_reviews: List[FileReview] = []
    skipped_files = list(parse_skips)

    for i, hunk in enumerate(hunks, 1):
        print(f"🔍 Analyzing file ({i}/{len(hunks)}): {hunk.path}...")
        
        # Build search query for context using added lines
        search_query = "\n".join(
            [line for line in hunk.annotated_diff.splitlines() if "+" in line and not "@@" in line]
        )
        context = get_relevant_context(db, search_query)

        # Call LLM with JSON recovery retry logic
        review = llm_reviewer.review_file(hunk, context)
        if review.skipped:
            print(f"⚠️ Skipped {hunk.path} review: {review.skip_reason}")
            skipped_files.append((hunk.path, review.skip_reason or "LLM Error"))
        else:
            file_reviews.append(review)
            print(f"✅ Analysis for {hunk.path} complete. Found {len(review.findings)} issues.")

    # 7. Aggregate and validate all findings
    result = ReviewResult(
        file_reviews=file_reviews,
        files_reviewed=len(file_reviews),
        files_skipped=len(skipped_files),
        model=settings.llm_model
    )

    all_findings = dedupe_findings(result.findings)

    # 8. Check for existing review comments to prevent double-posting (idempotency)
    existing_comment_keys = gh_client.get_existing_review_comments(pr)
    
    inline_to_post = []
    for f in all_findings:
        comment_key = f"{f.path}:{f.line}:{f.comment_body()}"
        # Only post if the exact comment hasn't been posted already
        if comment_key not in existing_comment_keys:
            inline_to_post.append({
                "path": f.path,
                "line": f.line,
                "body": f.comment_body()
            })

    # 9. Format the high-level review summary comment
    summary = build_summary(result, skipped_files)

    # Determine Review Action (REQUEST_CHANGES if major bugs found, else COMMENT)
    bugs_count = result.count(Severity.BUG)
    event = "COMMENT"
    
    # Author cannot Request Changes on their own PR
    current_user = gh_client.get_current_user()
    pr_author = pr.user.login
    
    if bugs_count > 0:
        if current_user == pr_author:
            print(f"ℹ️ Authenticated user is PR author ({pr_author}). Submitting review as COMMENT.")
            event = "COMMENT"
        else:
            print(f"🔴 Found {bugs_count} major bugs. Submitting review as REQUEST_CHANGES.")
            event = "REQUEST_CHANGES"

    # 10. Submit the Review
    print(f"🚀 Submitting PR review ({len(inline_to_post)} inline comments)...")
    success = gh_client.submit_review(
        pr=pr,
        body=summary,
        event=event,
        comments=inline_to_post
    )

    if not success:
        sys.exit(1)

    print("🎉 Code review complete.")
    
    # Return non-zero exit code if major bugs were identified to block merge in CI
    if bugs_count > 0 and current_user != pr_author:
        sys.exit(1)
    else:
        sys.exit(0)

if __name__ == "__main__":
    main()