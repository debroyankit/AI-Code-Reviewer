import json

def review_with_groq(diff_text, context_text, file_contents, env):
    numbered = ""
    for p, text in file_contents.items():
        numbered += f"\n--- FILE: {p} ---\n"
        for i, l in enumerate(text.splitlines(), start=1):
            numbered += f"{i}: {l}\n"

    messages = [
        {"role": "system", "content": "You are a senior code reviewer..."},
        {"role": "user", "content": diff_text + context_text + numbered}
    ]

    resp = env.llm.invoke(messages)

    try:
        return json.loads(resp.content)
    except:
        return []
