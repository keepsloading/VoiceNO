"""Small, dependency-free helpers for identifying experiment artifacts."""

import hashlib
import json
import subprocess
from pathlib import Path


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def manifest_sha256(items):
    payload = json.dumps(items, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def code_revision(root):
    """Record tracked edits as well as HEAD; local recordings are never read."""
    try:
        revision = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=root, stderr=subprocess.DEVNULL,
        ).decode().strip()
        diff = subprocess.check_output(
            ["git", "diff", "HEAD", "--", "*.py", "*.ipynb", "requirements*.txt"],
            cwd=root, stderr=subprocess.DEVNULL,
        )
        status = subprocess.check_output(
            ["git", "status", "--porcelain"], cwd=root, stderr=subprocess.DEVNULL,
        ).decode()
        untracked = subprocess.check_output(
            ["git", "ls-files", "--others", "--exclude-standard", "-z", "--", "*.py", "*.ipynb"],
            cwd=root, stderr=subprocess.DEVNULL,
        ).decode().split("\0")
        return {
            "revision": revision,
            "working_tree_dirty": bool(status),
            "tracked_code_diff_sha256": hashlib.sha256(diff).hexdigest(),
            "untracked_code": {
                name: file_sha256(Path(root) / name) for name in untracked if name
            },
        }
    except (OSError, subprocess.CalledProcessError):
        return {"revision": None, "working_tree_dirty": None, "tracked_code_diff_sha256": None}
