import os
import sys
from typing import Optional
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import TextLoader
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings

from app.config import settings

# Directory where the FAISS index is cached between workflow runs.
# GitHub Actions `actions/cache` restores/saves this folder automatically.
CACHE_DIR = ".faiss_cache"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


def _get_embeddings() -> HuggingFaceEmbeddings:
    """Returns the embedding model instance (downloaded once, then cached by the library)."""
    return HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)


def _load_cached_index() -> Optional[FAISS]:
    """
    Attempts to load a previously saved FAISS index from the cache directory.
    Returns None if the cache doesn't exist or is corrupted.
    """
    index_path = os.path.join(CACHE_DIR, "index.faiss")
    if not os.path.isfile(index_path):
        return None

    try:
        embeddings = _get_embeddings()
        db = FAISS.load_local(CACHE_DIR, embeddings, allow_dangerous_deserialization=True)
        print("⚡ Loaded FAISS index from cache (skipping full rebuild).")
        return db
    except Exception as e:
        print(f"⚠️ Cache exists but failed to load: {e}. Rebuilding...", file=sys.stderr)
        return None


def _save_index_to_cache(db: FAISS) -> None:
    """Saves the FAISS index to the cache directory for future runs."""
    try:
        os.makedirs(CACHE_DIR, exist_ok=True)
        db.save_local(CACHE_DIR)
        print(f"💾 FAISS index saved to {CACHE_DIR}/ for future runs.")
    except Exception as e:
        print(f"⚠️ Failed to save FAISS cache: {e}", file=sys.stderr)


def build_vector_store(github_client) -> Optional[FAISS]:
    """
    Builds (or loads from cache) a FAISS vector store of the repository codebase.
    
    Flow:
    1. Check if a cached index exists in .faiss_cache/ → load it instantly.
    2. If no cache → scan repo, chunk files, embed, build index, save to cache.
    """
    # --- Step 1: Try loading from cache ---
    cached = _load_cached_index()
    if cached is not None:
        return cached

    # --- Step 2: Build from scratch ---
    print("🔄 Building local codebase vector index (first run — this will be cached)...")
    documents = []

    for root, dirs, files in os.walk("."):
        dirs[:] = [d for d in dirs if d not in settings.ignore_dirs]
        for fname in files:
            fpath = os.path.join(root, fname)
            if not github_client.should_process_file(fpath):
                continue
            try:
                loader = TextLoader(fpath, encoding="utf-8", autodetect_encoding=True)
                docs = loader.load()
                documents.extend(docs)
            except Exception as e:
                print(f"⚠️ Failed to index file {fpath}: {e}", file=sys.stderr)

    if not documents:
        print("⚠️ No valid code documents found. Skipping vector index.")
        return None

    try:
        splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=150)
        chunks = splitter.split_documents(documents)

        embeddings = _get_embeddings()
        db = FAISS.from_documents(chunks, embeddings)
        print(f"✅ Vector index built from {len(documents)} files ({len(chunks)} chunks).")

        # --- Step 3: Save to cache for next PR ---
        _save_index_to_cache(db)

        return db
    except Exception as e:
        print(f"❌ Failed to construct FAISS index: {e}", file=sys.stderr)
        return None

def get_relevant_context(db: Optional[FAISS], search_query: str) -> str:
    """Retrieves top-4 relevant chunks match search query."""
    if not db or not search_query.strip():
        return ""
    try:
        docs = db.similarity_search(search_query, k=4)
        out = []
        for d in docs:
            source = d.metadata.get("source", "unknown")
            out.append(f"--- REFERENCE SOURCE: {source} ---\n{d.page_content}\n")
        return "\n".join(out)
    except Exception as e:
        print(f"⚠️ RAG context retrieval failed: {e}", file=sys.stderr)
        return ""