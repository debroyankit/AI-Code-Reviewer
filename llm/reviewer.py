import json, re

def review_with_groq(diff_text, context_text, file_contents, env):
    numbered = ""
    for p, text in file_contents.items():
        numbered += f"\n--- FILE: {p} ---\n"
        for i, l in enumerate(text.splitlines(), start=1):
            numbered += f"{i}: {l}\n"

    system_prompt = """
You are a strict JSON-generating code review agent.
You MAY output explanations or text.
But EVERY issue must be represented inside one or more JSON arrays like:

[
  {"path":"file.py","line":10,"body":"msg","severity":"major"}
]

I will merge them myself.
"""

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": f"{diff_text}\n{context_text}\n{numbered}"}
    ]

    resp = env.llm.invoke(messages)
    raw = resp.content

    print("\n[DEBUG RAW LLM OUTPUT]:\n", raw)

    # 1️⃣ Extract ALL JSON arrays
    arrays = re.findall(r'\[[\s\S]*?\]', raw)
    if not arrays:
        print("[DEBUG] No JSON arrays found.")
        return []

    print("\n[DEBUG] FOUND", len(arrays), "JSON ARRAYS")

    all_comments = []

    # 2️⃣ For each JSON array found → clean → parse
    for idx, arr in enumerate(arrays):
        print(f"\n[DEBUG] Processing array #{idx+1}:")
        print(arr)

        # Escape internal unescaped quotes inside string values
        cleaned = re.sub(
            r'("body"\s*:\s*")(.*?)(")',
            lambda m: m.group(1) + m.group(2).replace('"', '\\"') + m.group(3),
            arr
        )

        print("\n[DEBUG CLEANED ARRAY]:\n", cleaned)

        try:
            parsed = json.loads(cleaned)
            if isinstance(parsed, list):
                all_comments.extend(parsed)
                print("[DEBUG] Parsed OK.")
            else:
                print("[DEBUG] Not a list, skipping.")
        except Exception as e:
            print("[DEBUG] Parse error:", e)
            continue

    return all_comments
