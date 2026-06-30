"""
Manifest — tracks which files are indexed in the FAISS vector store.

The manifest is a JSON file stored at .faiss_cache/manifest.json.
It maps each indexed file path to its content hash and the FAISS
document IDs (docstore keys) produced when that file was chunked and embedded.

This enables incremental updates: when a file changes, we look up its
old doc_ids, delete them from FAISS, re-chunk the new content, and
record the new doc_ids.
"""

import hashlib
import json
import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple

MANIFEST_FILENAME = "manifest.json"
MANIFEST_VERSION = 1


@dataclass
class FileEntry:
    """One indexed file in the manifest."""
    content_hash: str
    doc_ids: List[str] = field(default_factory=list)


@dataclass
class Manifest:
    """Full manifest state — serializable to/from JSON."""
    version: int = MANIFEST_VERSION
    commit_sha: str = ""
    files: Dict[str, FileEntry] = field(default_factory=dict)


def compute_file_hash(filepath: str) -> str:
    """Compute SHA-256 hash of a file's content."""
    h = hashlib.sha256()
    try:
        with open(filepath, "rb") as f:
            for chunk in iter(lambda: f.read(8192), b""):
                h.update(chunk)
    except (OSError, IOError):
        return ""
    return f"sha256:{h.hexdigest()}"


def load_manifest(cache_dir: str) -> Optional[Manifest]:
    """Load manifest from cache directory. Returns None if not found or corrupted."""
    manifest_path = os.path.join(cache_dir, MANIFEST_FILENAME)
    if not os.path.isfile(manifest_path):
        return None

    try:
        with open(manifest_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        manifest = Manifest(
            version=data.get("version", MANIFEST_VERSION),
            commit_sha=data.get("commit_sha", ""),
        )
        for path, entry_data in data.get("files", {}).items():
            manifest.files[path] = FileEntry(
                content_hash=entry_data.get("content_hash", ""),
                doc_ids=entry_data.get("doc_ids", []),
            )
        return manifest
    except (json.JSONDecodeError, KeyError, TypeError) as e:
        print(f"⚠️ Manifest corrupted, will rebuild: {e}")
        return None


def save_manifest(cache_dir: str, manifest: Manifest) -> None:
    """Save manifest to cache directory."""
    os.makedirs(cache_dir, exist_ok=True)
    manifest_path = os.path.join(cache_dir, MANIFEST_FILENAME)

    data = {
        "version": manifest.version,
        "commit_sha": manifest.commit_sha,
        "files": {
            path: {
                "content_hash": entry.content_hash,
                "doc_ids": entry.doc_ids,
            }
            for path, entry in manifest.files.items()
        },
    }

    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


def diff_manifest(
    old_manifest: Optional[Manifest],
    current_files: Dict[str, str],
) -> Tuple[Set[str], Set[str], Set[str]]:
    """
    Compare the old manifest against the current set of files on disk.

    Args:
        old_manifest: Previously saved manifest (None if first run).
        current_files: Dict of {filepath: content_hash} for all current code files.

    Returns:
        (added, modified, deleted) — three sets of file paths.
        - added: files in current_files but not in old_manifest
        - modified: files in both but with different content_hash
        - deleted: files in old_manifest but not in current_files
    """
    if old_manifest is None:
        # Everything is new
        return set(current_files.keys()), set(), set()

    old_paths = set(old_manifest.files.keys())
    new_paths = set(current_files.keys())

    added = new_paths - old_paths
    deleted = old_paths - new_paths
    modified = set()

    for path in old_paths & new_paths:
        if old_manifest.files[path].content_hash != current_files[path]:
            modified.add(path)

    return added, modified, deleted
