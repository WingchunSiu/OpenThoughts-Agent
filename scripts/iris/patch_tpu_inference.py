#!/usr/bin/env python3
"""Apply runtime patches to the iris worker's pip-installed tpu-inference.

Invoked from the launcher's bash bootstrap *after* ``uv sync`` and
*before* the workload starts. Each patch is idempotent (skipped if
already applied) and prints a one-line status so the iris-job log makes
the applied state obvious.

Pin-and-fork is the long-term answer for any patch that lives here for
more than a few weeks; this script is the "ship now, deal upstream
later" hatch.

Patches currently applied
-------------------------

* **hbm_usage_bytes non-addressable device skip** (tpu_inference/utils.py)

  ``hbm_usage_bytes()`` iterates over every device in the JAX mesh and
  calls ``device.memory_stats()``. On multi-host TPU slices >v6e-8
  (v6e-16 = 4 hosts, v6e-32 = 8 hosts) each host can only address its
  4 local chips; calling ``memory_stats()`` on a non-addressable device
  raises ``jax.errors.JaxRuntimeError: INVALID_ARGUMENT: MemoryStats is
  only supported for addressable PjRt devices.``

  The patch adds a ``is_addressable`` guard to the non-Ray branch of
  the iteration so non-local chips are skipped instead of raising.
  This matches the Ray branch's behavior (which already short-circuits
  after the first successful device).
"""

from __future__ import annotations

import re
import sys
from pathlib import Path


PATCHES: list[tuple[str, str, str, str]] = [
    # (relative_path, old_text, new_text, label)
    (
        "tpu_inference/utils.py",
        "    else:\n"
        "        for device in devices:\n"
        "            hbm_used = device.memory_stats()[\"bytes_in_use\"]\n"
        "            hbm_limit = device.memory_stats()[\"bytes_limit\"]\n"
        "            usage.append((hbm_used, hbm_limit))",
        "    else:\n"
        "        for device in devices:\n"
        "            # ot-agent patch (patch_tpu_inference.py):\n"
        "            # skip non-addressable devices on multi-host slices >v6e-8.\n"
        "            if not getattr(device, \"is_addressable\", True):\n"
        "                continue\n"
        "            try:\n"
        "                hbm_used = device.memory_stats()[\"bytes_in_use\"]\n"
        "                hbm_limit = device.memory_stats()[\"bytes_limit\"]\n"
        "            except Exception:\n"
        "                continue\n"
        "            usage.append((hbm_used, hbm_limit))",
        "hbm_usage_bytes: skip non-addressable devices on multi-host slices",
    ),
]


def _site_packages_root() -> Path:
    """Return ``site-packages/`` for the current Python's environment.

    Prefer the venv at /app/.venv (iris worker convention); fall back
    to whichever directory hosts the running interpreter's site-packages.
    """
    candidate = Path("/app/.venv/lib/python3.12/site-packages")
    if candidate.is_dir():
        return candidate
    # Fallback: derive from sys.executable.
    for path in sys.path:
        p = Path(path)
        if p.name == "site-packages":
            return p
    raise RuntimeError(
        "could not locate site-packages on PYTHONPATH; aborting patch"
    )


def _apply_one(site_pkg: Path, rel_path: str, old: str, new: str, label: str) -> str:
    """Return a status string describing what the patch did."""
    target = site_pkg / rel_path
    if not target.is_file():
        return f"SKIP (file not found: {target}): {label}"
    src = target.read_text()
    if new in src:
        return f"ALREADY-PATCHED: {label}"
    if old not in src:
        return f"WARN (patch site not found, file may have drifted): {label}"
    target.write_text(src.replace(old, new))
    return f"APPLIED: {label}"


def main() -> int:
    site_pkg = _site_packages_root()
    print(f"[tpu-inference-patch] site-packages = {site_pkg}", flush=True)
    for rel_path, old, new, label in PATCHES:
        status = _apply_one(site_pkg, rel_path, old, new, label)
        print(f"[tpu-inference-patch] {status}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
