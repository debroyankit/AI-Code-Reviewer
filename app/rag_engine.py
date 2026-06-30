"""
RAG Engine — Manages the FAISS vector store with incremental update support.

Architecture:
- load_index(): Load pre-built FAISS index from .faiss_cache/
- build_full_index(): Full rebuild from scratch (first-time bootstrap only)
- incremental_update(): Add/remove specific file chunks without rebuilding
- get_relevant_context(): Query the index for context during PR review
"""

import os
import sys
from typing import Dict, List, Optional, Set, Tuple

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import TextLoader
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document

from app.config import settings
from app.manifest import (
    Manifest,
    FileEntry,
    compute_file_hash,
    load_manifest,
    save_manifest,
    diff_manifest,
)

# Directory where the FAISS index is cached between workflow runs.
CACHE_DIR = ".faiss_cache"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

# Chunking settings
CHUNK_SIZE = 2000
CHUNK_OVERLAP = 150


def _get_embeddings() -> HuggingFaceEmbeddings:
    """Returns the embedding model instance."""
    return HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)


def _get_splitter() -> RecursiveCharacterTextSplitter:
    """Returns the text splitter for chunking code files."""
    return RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP
    )


def _load_file_documents(filepath: str) -> List[Document]:
    """Load a single file into LangChain Documents. Returns [] on failure."""
    try:
        loader = TextLoader(filepath, encoding="utf-8", autodetect_encoding=True)
        return loader.load()
    except Exception as e:
        print(f"⚠️ Failed to load {filepath}: {e}", file=sys.stderr)
        return []


def _chunk_documents(documents: List[Document]) -> List[Document]:
    """Split documents into chunks."""
    if not documents:
        return []
    splitter = _get_splitter()
    return splitter.split_documents(documents)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def load_index() -> Optional[FAISS]:
    """
    Load a pre-built FAISS index from .faiss_cache/.
    Returns None if no cache exists.
    Used by the PR review workflow (main.py).
    """
    index_path = os.path.join(CACHE_DIR, "index.faiss")
    if not os.path.isfile(index_path):
        print("ℹ️ No FAISS cache found. Reviews will proceed without RAG context.")
        return None

    try:
        embeddings = _get_embeddings()
        db = FAISS.load_local(CACHE_DIR, embeddings, allow_dangerous_deserialization=True)
        manifest = load_manifest(CACHE_DIR)
        file_count = len(manifest.files) if manifest else "unknown"
        print(f"⚡ Loaded FAISS index from cache ({file_count} files indexed).")
        return db
    except Exception as e:
        print(f"⚠️ Cache exists but failed to load: {e}. Reviews will proceed without RAG context.", file=sys.stderr)
        return None


def build_full_index(github_client=None) -> Optional[FAISS]:
    """
    Build the FAISS index from scratch by scanning all code files in the repo.
    Used by sync_cache.py for first-time bootstrap.

    Args:
        github_client: Optional GitHubClient instance for file filtering.
                       If None, uses settings.ignore_dirs and settings.supported_extensions directly.
    """
    print("🔄 Building full FAISS index (first-time bootstrap)...")
    documents = []
    file_hashes: Dict[str, str] = {}

    for root, dirs, files in os.walk("."):
        dirs[:] = [d for d in dirs if d not in settings.ignore_dirs]
        for fname in files:
            fpath = os.path.join(root, fname)
            # Normalize path separators
            fpath_normalized = fpath.replace("\\", "/")

            if github_client and not github_client.should_process_file(fpath):
                continue
            elif not github_client:
                _, ext = os.path.splitext(fpath)
                if ext not in settings.supported_extensions:
                    continue

            docs = _load_file_documents(fpath)
            if docs:
                documents.extend(docs)
                file_hashes[fpath_normalized] = compute_file_hash(fpath)

    if not documents:
        print("⚠️ No valid code documents found. Skipping vector index.")
        return None

    try:
        chunks = _chunk_documents(documents)
        embeddings = _get_embeddings()
        db = FAISS.from_documents(chunks, embeddings)

        # Build manifest: map each file to its doc_ids
        manifest = Manifest()
        # Map chunks back to their source files
        file_doc_ids: Dict[str, List[str]] = {}
        for doc_id, doc in db.docstore._dict.items():
            source = doc.metadata.get("source", "").replace("\\", "/")
            if source not in file_doc_ids:
                file_doc_ids[source] = []
            file_doc_ids[source].append(doc_id)

        for fpath, content_hash in file_hashes.items():
            manifest.files[fpath] = FileEntry(
                content_hash=content_hash,
                doc_ids=file_doc_ids.get(fpath, []),
            )

        # Save both the FAISS index and manifest
        os.makedirs(CACHE_DIR, exist_ok=True)
        db.save_local(CACHE_DIR)
        save_manifest(CACHE_DIR, manifest)

        print(f"✅ Full index built: {len(file_hashes)} files, {len(chunks)} chunks.")
        print(f"💾 Saved to {CACHE_DIR}/")
        return db

    except Exception as e:
        print(f"❌ Failed to build FAISS index: {e}", file=sys.stderr)
        return None


def incremental_update(
    changed_files: Set[str],
    deleted_files: Set[str],
) -> bool:
    """
    Incrementally update the FAISS index by adding/removing specific files.
    Used by sync_cache.py after a PR is merged.

    Args:
        changed_files: Set of file paths that were added or modified.
        deleted_files: Set of file paths that were deleted.

    Returns:
        True if the update succeeded, False otherwise.
    """
    if not changed_files and not deleted_files:
        print("ℹ️ No file changes detected. Cache is already up to date.")
        return True

    # Load existing index and manifest
    index_path = os.path.join(CACHE_DIR, "index.faiss")
    if not os.path.isfile(index_path):
        print("⚠️ No existing cache found. Run a full build first.")
        return False

    manifest = load_manifest(CACHE_DIR)
    if manifest is None:
        print("⚠️ Manifest missing or corrupted. Run a full build first.")
        return False

    try:
        embeddings = _get_embeddings()
        db = FAISS.load_local(CACHE_DIR, embeddings, allow_dangerous_deserialization=True)
    except Exception as e:
        print(f"❌ Failed to load existing index: {e}", file=sys.stderr)
        return False

    # --- Step 1: Delete old chunks for changed and deleted files ---
    ids_to_delete = []
    files_to_remove_from_manifest = set()

    for fpath in changed_files | deleted_files:
        fpath_normalized = fpath.replace("\\", "/")
        if fpath_normalized in manifest.files:
            ids_to_delete.extend(manifest.files[fpath_normalized].doc_ids)
            files_to_remove_from_manifest.add(fpath_normalized)

    if ids_to_delete:
        try:
            db.delete(ids_to_delete)
            print(f"🗑️ Removed {len(ids_to_delete)} old chunks from {len(files_to_remove_from_manifest)} files.")
        except Exception as e:
            print(f"⚠️ Failed to delete old chunks: {e}", file=sys.stderr)

    # Remove from manifest
    for fpath in files_to_remove_from_manifest:
        del manifest.files[fpath]

    # --- Step 2: Add new chunks for changed/added files ---
    new_doc_count = 0
    for fpath in changed_files:
        fpath_normalized = fpath.replace("\\", "/")
        if not os.path.isfile(fpath):
            continue  # File was listed as changed but doesn't exist (edge case)

        docs = _load_file_documents(fpath)
        if not docs:
            continue

        chunks = _chunk_documents(docs)
        if not chunks:
            continue

        # Add to FAISS — returns list of new doc IDs
        new_ids = db.add_documents(chunks)

        # Update manifest
        manifest.files[fpath_normalized] = FileEntry(
            content_hash=compute_file_hash(fpath),
            doc_ids=[str(id) for id in new_ids],
        )
        new_doc_count += len(chunks)

    print(f"➕ Added {new_doc_count} new chunks from {len(changed_files)} files.")

    # --- Step 3: Save updated index and manifest ---
    try:
        db.save_local(CACHE_DIR)
        save_manifest(CACHE_DIR, manifest)
        total_files = len(manifest.files)
        print(f"💾 Cache updated. Total indexed files: {total_files}")
        return True
    except Exception as e:
        print(f"❌ Failed to save updated cache: {e}", file=sys.stderr)
        return False


def scan_current_files(github_client=None) -> Dict[str, str]:
    """
    Scan the current working directory and return a dict of
    {normalized_filepath: content_hash} for all supported code files.
    """
    file_hashes: Dict[str, str] = {}

    for root, dirs, files in os.walk("."):
        dirs[:] = [d for d in dirs if d not in settings.ignore_dirs]
        for fname in files:
            fpath = os.path.join(root, fname)
            fpath_normalized = fpath.replace("\\", "/")

            if github_client and not github_client.should_process_file(fpath):
                continue
            elif not github_client:
                _, ext = os.path.splitext(fpath)
                if ext not in settings.supported_extensions:
                    continue

            content_hash = compute_file_hash(fpath)
            if content_hash:
                file_hashes[fpath_normalized] = content_hash

    return file_hashes


def get_relevant_context(db: Optional[FAISS], search_query: str) -> str:
    """Retrieves top-4 relevant chunks matching the search query."""
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