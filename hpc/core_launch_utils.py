"""Shared launch utilities that are lightweight enough for cross-module imports."""

from __future__ import annotations

from os import PathLike
from pathlib import Path
from typing import Union

PathInput = Union[str, PathLike[str], Path, None]


def cleanup_endpoint_file(path_like: PathInput, *, descriptor: str = "endpoint file") -> None:
    """Remove a stale endpoint JSON if it exists."""

    if not path_like:
        return
    try:
        candidate = Path(path_like).expanduser()
    except Exception:
        return
    if not candidate.exists():
        return
    try:
        candidate.unlink()
        print(f"Removed {descriptor}: {candidate}")
    except OSError as exc:
        print(f"Warning: failed to remove {descriptor} {candidate}: {exc}")


__all__ = [
    "cleanup_endpoint_file",
]
