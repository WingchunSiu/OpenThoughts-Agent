"""WandB utilities for HPC launchers.

This module provides utilities for setting up WandB directories and
permissions on HPC systems where directory permissions can be problematic.
"""

from __future__ import annotations

import os
import subprocess
from pathlib import Path
from typing import Optional


def ensure_wandb_dir(
    wandb_dir: Optional[str] = None,
    experiments_dir: Optional[str] = None,
    verbose: bool = True,
) -> str:
    """Ensure WandB directory exists and is writable.

    Creates the wandb directory if needed and fixes permissions to ensure
    WandB can write logs. Falls back to a temp directory if the primary
    location isn't writable.

    Args:
        wandb_dir: Explicit wandb directory path. If None, derives from experiments_dir.
        experiments_dir: Base experiments directory. Used if wandb_dir is None.
        verbose: Whether to print status messages.

    Returns:
        Path to the writable wandb directory.
    """
    # Determine wandb directory
    if wandb_dir:
        target_dir = Path(wandb_dir)
    elif experiments_dir:
        target_dir = Path(experiments_dir) / "wandb"
    else:
        # Fall back to environment variable or temp
        target_dir = Path(os.environ.get("WANDB_DIR", "/tmp/wandb"))

    # Create directory structure
    target_dir.mkdir(parents=True, exist_ok=True)

    # Also create the nested wandb/ subdirectory that wandb creates
    nested_dir = target_dir / "wandb"
    nested_dir.mkdir(parents=True, exist_ok=True)

    # Fix permissions
    _fix_wandb_permissions(target_dir, verbose=verbose)

    # Verify writable
    if _is_writable(target_dir):
        if verbose:
            print(f"[wandb_utils] WandB directory ready: {target_dir}")
        return str(target_dir)

    # Fall back to temp directory
    fallback_dir = Path("/tmp") / "wandb" / os.environ.get("USER", "unknown")
    fallback_dir.mkdir(parents=True, exist_ok=True)

    if verbose:
        print(f"[wandb_utils] Warning: {target_dir} not writable, using fallback: {fallback_dir}")

    return str(fallback_dir)


def _fix_wandb_permissions(wandb_dir: Path, verbose: bool = True) -> None:
    """Fix permissions on WandB directory.

    Args:
        wandb_dir: Path to wandb directory.
        verbose: Whether to print status messages.
    """
    if not wandb_dir.exists():
        return

    if verbose:
        print(f"[wandb_utils] Fixing permissions on: {wandb_dir}")

    try:
        # Make directory and all contents writable by owner
        subprocess.run(
            ["chmod", "-R", "u+rwX", str(wandb_dir)],
            check=True,
            capture_output=True,
        )
    except subprocess.CalledProcessError as e:
        if verbose:
            print(f"[wandb_utils] Warning: chmod failed on {wandb_dir}: {e.stderr}")


def _is_writable(path: Path) -> bool:
    """Check if a directory is writable.

    Args:
        path: Path to check.

    Returns:
        True if directory is writable, False otherwise.
    """
    if not path.exists():
        return False

    # Try to create a test file
    test_file = path / ".wandb_write_test"
    try:
        test_file.touch()
        test_file.unlink()
        return True
    except (OSError, PermissionError):
        return False


def setup_wandb_env(
    wandb_dir: Optional[str] = None,
    experiments_dir: Optional[str] = None,
    project_name: Optional[str] = None,
    run_name: Optional[str] = None,
    verbose: bool = True,
) -> dict:
    """Set up WandB environment variables.

    Ensures the wandb directory is writable and sets appropriate
    environment variables for WandB logging.

    Args:
        wandb_dir: Explicit wandb directory path.
        experiments_dir: Base experiments directory.
        project_name: WandB project name.
        run_name: WandB run name.
        verbose: Whether to print status messages.

    Returns:
        Dictionary of environment variables that were set.
    """
    env_vars = {}

    # Ensure wandb directory is writable
    resolved_dir = ensure_wandb_dir(
        wandb_dir=wandb_dir,
        experiments_dir=experiments_dir,
        verbose=verbose,
    )
    os.environ["WANDB_DIR"] = resolved_dir
    env_vars["WANDB_DIR"] = resolved_dir

    # Set project and run name if provided
    if project_name:
        os.environ["WANDB_PROJECT"] = project_name
        env_vars["WANDB_PROJECT"] = project_name

    if run_name:
        os.environ["WANDB_RUN_NAME"] = run_name
        env_vars["WANDB_RUN_NAME"] = run_name

    return env_vars


__all__ = [
    "ensure_wandb_dir",
    "setup_wandb_env",
]
