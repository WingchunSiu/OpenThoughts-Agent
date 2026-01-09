"""HuggingFace utilities for HPC launchers.

This module provides common utilities for working with HuggingFace Hub:
- Repository ID validation and sanitization
- Dataset path detection
- HF repo ID derivation for eval uploads
"""

from __future__ import annotations

import hashlib
import re
from typing import Optional


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


def derive_default_hf_repo_id(
    job_name: str,
    harbor_dataset: Optional[str] = None,
    dataset_path: Optional[str] = None,
    explicit_repo: Optional[str] = None,
) -> str:
    """Derive default HF repo ID from benchmark repo.

    Used by both local and HPC eval runners to auto-derive an HF repo ID
    when --upload_to_database is set but --upload_hf_repo is not provided.

    Args:
        job_name: Name of the eval job
        harbor_dataset: Harbor dataset slug (e.g., "terminal-bench@2.0")
        dataset_path: Path to dataset directory
        explicit_repo: Explicitly specified benchmark repo

    Returns:
        HF repo ID in format "<org>/<job_name>"
    """
    # Import here to avoid circular imports
    from hpc.launch_utils import derive_benchmark_repo

    benchmark_repo = derive_benchmark_repo(
        harbor_dataset=harbor_dataset,
        dataset_path=dataset_path,
        explicit_repo=explicit_repo,
    )
    if "/" in benchmark_repo:
        org = benchmark_repo.split("/", 1)[0]
    else:
        org = benchmark_repo or "openthoughts-agent"
    return f"{org}/{job_name}"


def resolve_hf_repo_id(
    explicit_repo: Optional[str],
    upload_to_database: bool,
    job_name: str,
    harbor_dataset: Optional[str] = None,
    dataset_path: Optional[str] = None,
    eval_benchmark_repo: Optional[str] = None,
) -> Optional[str]:
    """Resolve HF repo ID for eval upload.

    If explicit_repo is provided, use it.
    If upload_to_database is True but no explicit repo, auto-derive from benchmark.
    Otherwise return None.

    Used by both local and HPC eval runners to determine the HF repo ID.

    Args:
        explicit_repo: Explicitly specified HF repo ID (--upload_hf_repo)
        upload_to_database: Whether database upload is enabled
        job_name: Name of the eval job
        harbor_dataset: Harbor dataset slug
        dataset_path: Path to dataset directory
        eval_benchmark_repo: Explicit benchmark repo

    Returns:
        Sanitized HF repo ID, or None if HF upload should be skipped
    """
    if explicit_repo:
        return sanitize_hf_repo_id(explicit_repo)

    if upload_to_database:
        # Auto-derive HF repo ID when database upload is enabled
        derived_repo = derive_default_hf_repo_id(
            job_name=job_name,
            harbor_dataset=harbor_dataset,
            dataset_path=dataset_path,
            explicit_repo=eval_benchmark_repo,
        )
        return sanitize_hf_repo_id(derived_repo)

    return None


__all__ = [
    "is_hf_dataset_path",
    "sanitize_hf_repo_id",
    "derive_default_hf_repo_id",
    "resolve_hf_repo_id",
]
