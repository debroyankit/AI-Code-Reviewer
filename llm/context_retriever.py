def get_relevant_context(db, query):
    if not db or not query.strip():
        return ""
    docs = db.similarity_search(query, k=4)
    return "\n".join([f"--- REFERENCE: {d.metadata['source']} ---\n{d.page_content}" for d in docs])
