"""
mlops/registry/utils.py

Standalone helper functions used across the registry package.
No external dependencies — standard library only.
"""

import os
import re
import json
import hashlib


# ---------------------------------------------------------------------------
# Path-safe naming
# ---------------------------------------------------------------------------

def sanitize_path_component(name: str) -> str:
    """
    Replace filesystem-unsafe characters so a user-facing name can be used
    as a folder name without creating unintended nested paths.

    Replaces: / \\ : * ? " < > |  →  underscore
    Strips leading/trailing dots and spaces.
    Falls back to 'unnamed' for empty results.
    """
    sanitized = re.sub(r'[<>:"/\\|?*]', '_', name)
    sanitized = sanitized.strip('. ')
    return sanitized or 'unnamed'


# ---------------------------------------------------------------------------
# Version folder creation
# ---------------------------------------------------------------------------

def create_version_folder(registry_root: str, project: str, version_id: str) -> str:
    """
    Create {registry_root}/{project}/{version_id}/ including all intermediate directories.

    `project` may be a forward-slash key like "unet/SnipeX/SLXCAP/CRC/cam0"
    or a plain name — both are handled correctly.

    Returns the absolute path to the created folder.
    Raises OSError if creation fails.
    """
    parts = project.replace("\\", "/").split("/")
    folder = os.path.join(registry_root, *parts, version_id)
    os.makedirs(folder, exist_ok=True)
    return os.path.abspath(folder)


# ---------------------------------------------------------------------------
# Dataset hashing
# ---------------------------------------------------------------------------

def compute_sha256(
    folder_path: str,
    extensions: tuple = (".png", ".jpg", ".jpeg"),
    progress_callback=None,
) -> str:
    """
    Compute a single SHA-256 hash representing all image files under folder_path.

    Files are collected recursively and sorted by relative path for determinism.
    Each file contributes its relative path (UTF-8) and raw bytes to the running hash.
    Calls progress_callback(current, total) after each file if provided.
    Returns the final hex digest string.
    """
    # Collect all matching files
    all_files = []
    for dirpath, _, filenames in os.walk(folder_path):
        for fname in filenames:
            if os.path.splitext(fname)[1].lower() in extensions:
                full = os.path.join(dirpath, fname)
                rel = os.path.relpath(full, folder_path)
                all_files.append((rel, full))

    all_files.sort(key=lambda x: x[0])  # deterministic ordering

    hasher = hashlib.sha256()
    total = len(all_files)

    for idx, (rel_path, full_path) in enumerate(all_files):
        hasher.update(rel_path.encode("utf-8"))
        with open(full_path, "rb") as fh:
            hasher.update(fh.read())
        if progress_callback is not None:
            progress_callback(idx + 1, total)

    return hasher.hexdigest()


# ---------------------------------------------------------------------------
# mtime cache helpers
# ---------------------------------------------------------------------------

def load_mtime_cache(cache_path: str) -> dict:
    """
    Read a JSON cache file and return its contents as a dict.
    Returns {} if the file is missing or contains malformed JSON.
    """
    if not os.path.isfile(cache_path):
        return {}
    try:
        with open(cache_path, "r", encoding="utf-8") as fh:
            return json.load(fh)
    except (json.JSONDecodeError, OSError):
        return {}


def save_mtime_cache(cache_path: str, data: dict):
    """Write data as JSON to cache_path with indent=2."""
    with open(cache_path, "w", encoding="utf-8") as fh:
        json.dump(data, fh, indent=2)


def is_hash_cache_valid(
    folder_path: str,
    cache: dict,
    extensions: tuple = (".png", ".jpg", ".jpeg"),
) -> bool:
    """
    Quick mtime check to determine whether the cached SHA-256 is still valid.

    Scans folder_path for matching files and compares (relative_path, mtime) pairs
    against cache['mtimes']. Returns True only if nothing has been added, removed,
    or modified. Returns False if cache is empty or missing the 'mtimes' key.
    """
    cached_mtimes = cache.get("mtimes")
    if not cached_mtimes:
        return False

    current_mtimes = {}
    for dirpath, _, filenames in os.walk(folder_path):
        for fname in filenames:
            if os.path.splitext(fname)[1].lower() in extensions:
                full = os.path.join(dirpath, fname)
                rel = os.path.relpath(full, folder_path)
                current_mtimes[rel] = os.path.getmtime(full)

    return current_mtimes == cached_mtimes


def get_dataset_hash_cached(folder_path: str, progress_callback=None) -> str:
    """
    Return the SHA-256 hash of the dataset at folder_path, using a local .hash_cache.json
    file to avoid recomputing when files have not changed.

    Workflow:
      1. Load .hash_cache.json from folder_path
      2. If mtime check passes → return the cached hash immediately
      3. Otherwise: compute hash, update cache, save, return the hash
    """
    cache_path = os.path.join(folder_path, ".hash_cache.json")
    cache = load_mtime_cache(cache_path)

    if is_hash_cache_valid(folder_path, cache) and "sha256" in cache:
        return cache["sha256"]

    # Recompute
    hash_str = compute_sha256(folder_path, progress_callback=progress_callback)

    # Build new mtime snapshot
    mtimes = {}
    extensions = (".png", ".jpg", ".jpeg")
    for dirpath, _, filenames in os.walk(folder_path):
        for fname in filenames:
            if os.path.splitext(fname)[1].lower() in extensions:
                full = os.path.join(dirpath, fname)
                rel = os.path.relpath(full, folder_path)
                mtimes[rel] = os.path.getmtime(full)

    save_mtime_cache(cache_path, {"sha256": hash_str, "mtimes": mtimes})
    return hash_str


# ---------------------------------------------------------------------------
# Path helpers
# ---------------------------------------------------------------------------

def write_project_csv(registry_root: str, project: str) -> None:
    """
    Write {registry_root}/{project}/models_registry.csv with one row per version.
    Called automatically after training finishes and after ONNX export completes.
    Failures are silently swallowed by callers — never crash training over CSV.
    """
    import csv
    from .scanner import RegistryScanner

    project_dir = os.path.join(registry_root, project)
    if not os.path.isdir(project_dir):
        return

    versions = RegistryScanner(registry_root).get_versions(project)

    csv_path = os.path.join(project_dir, "models_registry.csv")
    tmp_path = csv_path + ".tmp"

    fieldnames = [
        "version_id", "project", "annotator", "model_type",
        "status", "architecture", "encoder", "epochs",
        "best_val_loss", "best_epoch", "has_onnx", "has_weights",
        "sync_complete", "started_at", "commit_message",
    ]

    with open(tmp_path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for v in versions:
            writer.writerow(v)

    os.replace(tmp_path, csv_path)


def make_relative_path(absolute_path: str, datasets_root: str) -> str:
    """
    Return absolute_path expressed relative to datasets_root, using forward slashes.

    Example:
      make_relative_path('D:/Data/NestleS/v1/', 'D:/Data/') → 'NestleS/v1'

    If absolute_path is not under datasets_root, returns absolute_path unchanged.
    """
    try:
        rel = os.path.relpath(absolute_path, datasets_root)
        # If relpath goes outside (starts with '..'), return original
        if rel.startswith(".."):
            return absolute_path
        return rel.replace("\\", "/")
    except ValueError:
        # On Windows, relpath raises ValueError for paths on different drives
        return absolute_path
