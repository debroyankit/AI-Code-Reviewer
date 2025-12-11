import os
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import TextLoader
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from app.config import IGNORE_DIRS
from app.github_client import should_process_file

def build_vector_store():
    print("🔄 Building RAG index (FAISS)...")
    documents = []
    for root, dirs, files in os.walk("."):
        dirs[:] = [d for d in dirs if d not in IGNORE_DIRS]
        for fname in files:
            fpath = os.path.join(root, fname)
            if not should_process_file(fpath):
                continue
            try:
                docs = TextLoader(fpath, encoding="utf-8").load()
                documents.extend(docs)
            except Exception:
                pass

    if not documents:
        print("⚠️ No documents found for RAG. Skipping vector DB.")
        return None

    splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=150)
    chunks = splitter.split_documents(documents)
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    db = FAISS.from_documents(chunks, embeddings)
    return db

def get_relevant_context(db, search_query):
    if not db or not search_query.strip():
        return ""
    docs = db.similarity_search(search_query, k=4)
    out = ""
    for d in docs:
        out += f"\n--- REFERENCE: {d.metadata.get('source','')} ---\n{d.page_content}\n"
    return out