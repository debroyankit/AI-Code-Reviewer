"""
sync_cache.py — Entry point for the cache sync workflow.

This script is triggered by the cache-sync.yml workflow whenever code is
pushed to main/master (i.e., after a PR is merged) or via manual dispatch.

Flow:
1. Load existing manifest from .faiss_cache/manifest.json
2. If no manifest exists → full build (bootstrap)
3. If manifest exists:
   a. Scan all current code files in the repo
   b. Compare with manifest to find added/modified/deleted files
   c. Run incremental_update() on the FAISS index
4. Save updated index + manifest
5. Always exit 0 (so GitHub Actions saves the cache)
"""

import sys

from app.config import settings
from app.manifest import load_manifest, diff_manifest
from app.rag_engine import (
    CACHE_DIR,
    build_full_index,
    incremental_update,
    scan_current_files,
)


def main():
    print(f"🔄 Cache Sync for {settings.repo_name}")
    print(f"   Cache directory: {CACHE_DIR}")

    # Step 1: Load existing manifest
    manifest = load_manifest(CACHE_DIR)

    if manifest is None:
        # No existing cache — do a full bootstrap build
        print("📦 No existing cache found. Running full bootstrap build...")
        db = build_full_index()
        if db is None:
            print("⚠️ Bootstrap build produced no index (empty repo?).")
        else:
            print("✅ Bootstrap build complete.")
        sys.exit(0)

    # Step 2: Scan current files and compute hashes
    print("🔍 Scanning current codebase...")
    current_files = scan_current_files()
    print(f"   Found {len(current_files)} code files on disk.")
    print(f"   Manifest has {len(manifest.files)} previously indexed files.")

    # Step 3: Diff against manifest
    added, modified, deleted = diff_manifest(manifest, current_files)

    if not added and not modified and not deleted:
        print("✅ Cache is already up to date. No changes detected.")
        sys.exit(0)

    print(f"   📊 Changes detected:")
    print(f"      Added:    {len(added)} files")
    print(f"      Modified: {len(modified)} files")
    print(f"      Deleted:  {len(deleted)} files")

    # Step 4: Incremental update
    # changed_files = added + modified (both need new chunks)
    changed = added | modified
    success = incremental_update(changed_files=changed, deleted_files=deleted)

    if success:
        print("✅ Incremental cache update complete.")
    else:
        # If incremental update fails, fall back to full rebuild
        print("⚠️ Incremental update failed. Falling back to full rebuild...")
        db = build_full_index()
        if db:
            print("✅ Full rebuild complete.")
        else:
            print("❌ Full rebuild also failed.", file=sys.stderr)

    # Always exit 0 so GitHub Actions saves the cache
    sys.exit(0)


if __name__ == "__main__":
    main()
