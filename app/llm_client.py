import json
from langchain_groq import ChatGroq
from app.config import GROQ_API_KEY

llm = ChatGroq(groq_api_key=GROQ_API_KEY, model_name="llama-3.1-8b-instant", temperature=0)

def review_with_groq(diff_text, context_text, file_contents):
    """
    Ask Groq to return a JSON array of issues.
    """
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

    try:
        resp = llm.invoke(
            [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_message}
            ]
        )
        content = resp.content.strip()
        data = json.loads(content)
        if isinstance(data, list):
            return data
        else:
            return []
    except Exception as e:
        print(f"⚠️ LLM invocation or JSON parse failed: {e}")
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