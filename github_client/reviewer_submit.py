import requests

def submit_batch_review(pr_number, inline_comments, summary, event, env):
    url = f"https://api.github.com/repos/{env.REPO_NAME}/pulls/{pr_number}/reviews"
    headers = {"Authorization": f"Bearer {env.GITHUB_TOKEN}"}

    payload = {
        "body": summary,
        "event": event,
        "comments": inline_comments
    }

    resp = requests.post(url, json=payload, headers=headers)
    return resp.status_code in (200, 201)


def post_fallback_comments(pr_obj, fallback_comments):
    for c in fallback_comments:
        body = f"[Line {c['line']}] {c['body']}"
        pr_obj.create_issue_comment(f"In `{c['path']}`: {body}")
