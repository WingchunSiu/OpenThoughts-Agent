"""HuggingFace utilities for HPC launchers.

This module provides common utilities for working with HuggingFace Hub:
- Repository ID validation and sanitization
- Dataset path detection
"""

from __future__ import annotations

import hashlib
import re


def is_hf_dataset_path(path: str) -> bool:
    """Check if path looks like a HuggingFace dataset identifier.

    HF identifiers have format: org/repo-name or username/repo-name
    They contain exactly one "/" and no path separators like "./" or "../"

    Args:
        path: Path string to check

    Returns:
        True if path appears to be an HF dataset identifier
    """
    if not path:
        return False

    # Must contain exactly one "/"
    if path.count("/") != 1:
        return False

    # Must not look like a filesystem path
    if path.startswith(("./", "../", "/", "~")):
        return False

    # Must not contain backslashes (Windows paths)
    if "\\" in path:
        return False

    # Both parts must be non-empty
    parts = path.split("/")
    if not all(p.strip() for p in parts):
        return False

    return True


def sanitize_hf_repo_id(repo_id: str, max_length: int = 96) -> str:
    """Sanitize a HuggingFace repo_id to comply with naming rules.

    Keeps org prefix (e.g. 'mlfoundations-dev/') and cleans up the rest.
    Used when deriving HF dataset repo names from job names or model paths.

    Args:
        repo_id: The repository ID to sanitize (e.g., 'org/some-name').
        max_length: Maximum allowed length for the full repo_id.

    Returns:
        Sanitized repo_id that complies with HuggingFace naming rules.
    """

    def collapse(value: str) -> str:
        prev = None
        while value != prev:
            prev = value
            value = value.replace("--", "-").replace("..", ".")
        return value

    org, name = repo_id.split("/", 1) if "/" in repo_id else (None, repo_id)
    name = re.sub(r"[^A-Za-z0-9._-]", "-", name)
    name = collapse(name).strip("-.")
    if not name:
        name = "repo"
    limit = max_length - (len(org) + 1 if org else 0)
    if len(name) > limit > 8:
        digest = hashlib.sha1(name.encode()).hexdigest()[:8]
        keep = max(1, limit - len(digest))
        base = name[:keep].rstrip("-.") or "r"
        name = collapse(f"{base}{digest}").strip("-.")
    if name[0] in "-.":
        name = f"r{name[1:]}"
    if name[-1] in "-.":
        name = f"{name[:-1]}0"
    return f"{org}/{name}" if org else name


__all__ = [
    "is_hf_dataset_path",
    "sanitize_hf_repo_id",
]
