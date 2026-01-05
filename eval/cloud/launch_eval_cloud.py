#!/usr/bin/env python3
"""
Launch OpenThoughts evals on a cloud VM via SkyPilot.

This wrapper mirrors the key arguments from eval/local/run_eval.py, then wraps the
whole invocation inside a SkyPilot Task so we can bring up short-lived GPU nodes.
"""

from __future__ import annotations

import argparse
import os
import shlex
import sys
from pathlib import Path
from typing import List, Optional, Sequence

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.append(str(REPO_ROOT))

try:
    from eval.cloud.sync_utils import sync_eval_outputs
except ImportError:
    from sync_utils import sync_eval_outputs  # type: ignore

try:
    import sky
except ImportError as exc:  # pragma: no cover - optional dependency
    print(
        "SkyPilot is required for cloud launches. Install with "
        "`pip install '.[cloud]'` or `uv pip install '.[cloud]'`.",
        file=sys.stderr,
    )
    raise SystemExit(1) from exc


DEFAULT_REMOTE_OUTPUT_DIR = "/opt/openthoughts/cloud_runs"
DEFAULT_LOCAL_SYNC_DIR = (REPO_ROOT / "cloud_runs").as_posix()

# GitHub Container Registry images (build with docker/build_and_push.sh)
GHCR_IMAGE_BASE = "ghcr.io/open-thoughts/openthoughts-agent"
DEFAULT_DOCKER_IMAGE = f"{GHCR_IMAGE_BASE}:gpu-1x"


def _repo_relative(path_str: str) -> str:
    abs_path = Path(path_str).expanduser().resolve()
    try:
        relative = abs_path.relative_to(REPO_ROOT)
    except ValueError as exc:  # pragma: no cover - sanity guard
        raise ValueError(f"Path '{abs_path}' must live inside the repo ({REPO_ROOT})") from exc
    return relative.as_posix()


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Launch eval/local/run_eval.py on a cloud GPU node via SkyPilot.")

    # Mirrors run_eval arguments
    parser.add_argument("--harbor-config", required=True, help="Path (within repo) to Harbor YAML.")
    parser.add_argument("--datagen-config", help="Optional datagen config to seed defaults.")
    parser.add_argument("--dataset", help="Harbor dataset slug (exclusive with --dataset-path).")
    parser.add_argument("--dataset-path", help="Path to tasks directory (exclusive with --dataset).")
    parser.add_argument("--model", required=True, help="Model identifier used by run_eval.")
    parser.add_argument("--agent", default="terminus-2", help="Harbor agent to run.")
    parser.add_argument("--eval-benchmark-repo", required=True, help="Supabase repo id for eval bookkeeping.")
    parser.add_argument("--harbor-extra-arg", action="append", default=[], help="Extra --harbor jobs start args.")
    parser.add_argument("--agent-kwarg", action="append", default=[], help="Additional --agent-kwarg entries.")
    parser.add_argument("--n-concurrent", type=int, default=16)
    parser.add_argument("--n-attempts", type=int, default=3)
    parser.add_argument("--expected-trials", type=int, help="Optional Harbor expected-trials hint.")
    parser.add_argument("--gpus", type=int, default=1, help="run_eval --gpus value.")
    parser.add_argument("--dry-run", action="store_true", help="Pass --dry-run to run_eval.")
    parser.add_argument("--job-name", help="Optional override for Harbor job name.")

    # Cloud specific options
    parser.add_argument("--cloud-provider", choices=["gcp", "aws", "azure"], default="gcp")
    parser.add_argument("--region", help="Preferred region.")
    parser.add_argument("--zone", help="Preferred zone.")
    parser.add_argument("--accelerator", default="A100-40GB:1", help="SkyPilot accelerator spec (e.g., A100-40GB:1).")
    parser.add_argument("--use-spot", action="store_true", help="Use spot/preemptible instances.")
    parser.add_argument(
        "--docker-image",
        default=DEFAULT_DOCKER_IMAGE,
        help="Pre-built Docker image (default: auto-selects gpu-1x/4x/8x based on accelerator count). "
        "Build images with: ./docker/build_and_push.sh",
    )
    parser.add_argument("--task-name", default="ot-eval-cloud", help="SkyPilot task name.")
    parser.add_argument("--cluster-name", help="Optional SkyPilot cluster name override.")
    parser.add_argument("--remote-output-dir", default=DEFAULT_REMOTE_OUTPUT_DIR)
    parser.add_argument("--local-sync-dir", default=DEFAULT_LOCAL_SYNC_DIR)
    parser.add_argument("--secrets-env", help="Path to secrets.env to source inside the container.")
    parser.add_argument(
        "--no-sync",
        action="store_true",
        help="Skip syncing local codebase to VM (use code baked into Docker image).",
    )

    return parser.parse_args()


def _ensure_mutually_exclusive(dataset: Optional[str], dataset_path: Optional[str]) -> None:
    if dataset and dataset_path:
        raise ValueError("Specify either --dataset or --dataset-path (not both).")
    if not dataset and not dataset_path:
        raise ValueError("Must provide --dataset or --dataset-path for eval workloads.")


def _resolve_cloud(name: str):
    if name == "gcp":
        return sky.clouds.GCP()
    if name == "aws":
        return sky.clouds.AWS()
    if name == "azure":
        return sky.clouds.Azure()
    raise ValueError(f"Unsupported cloud provider '{name}'")


def _build_run_eval_command(args: argparse.Namespace) -> List[str]:
    cmd: List[str] = ["python", "eval/local/run_eval.py", "--harbor-config", args.harbor_config, "--model", args.model]
    if args.datagen_config:
        cmd.extend(["--datagen-config", args.datagen_config])
    if args.dataset:
        cmd.extend(["--dataset", args.dataset])
    elif args.dataset_path:
        cmd.extend(["--dataset-path", args.dataset_path])
    cmd.extend(
        [
            "--agent",
            args.agent,
            "--n-concurrent",
            str(args.n_concurrent),
            "--n-attempts",
            str(args.n_attempts),
            "--gpus",
            str(args.gpus),
            "--eval-benchmark-repo",
            args.eval_benchmark_repo,
            "--experiments-dir",
            args.remote_output_dir,
        ]
    )
    if args.expected_trials:
        cmd.extend(["--expected-trials", str(args.expected_trials)])
    if args.job_name:
        cmd.extend(["--job-name", args.job_name])
    if args.dry_run:
        cmd.append("--dry-run")
    for kwarg in args.agent_kwarg:
        cmd.extend(["--agent-kwarg", kwarg])
    for extra in args.harbor_extra_arg:
        cmd.extend(["--harbor-extra-arg", extra])
    return cmd


def _normalize_docker_image(image: str) -> str:
    """Ensure docker image has the 'docker:' prefix required by SkyPilot."""
    if not image.startswith("docker:"):
        return f"docker:{image}"
    return image


def _select_docker_image(args: argparse.Namespace) -> str:
    """Select appropriate Docker image variant based on GPU count.

    If user specified a custom --docker-image, use it as-is.
    Otherwise, auto-select gpu-1x/gpu-4x/gpu-8x based on accelerator count.
    """
    # If user provided a custom image (not our default), use it directly
    if args.docker_image != DEFAULT_DOCKER_IMAGE:
        return args.docker_image

    # Parse GPU count from accelerator spec (e.g., "H100-80GB:2" -> 2)
    accelerator = args.accelerator
    if ":" in accelerator:
        try:
            count = int(accelerator.split(":", 1)[1])
        except ValueError:
            count = 1
    else:
        count = 1

    # Select appropriate image variant
    if count <= 1:
        return f"{GHCR_IMAGE_BASE}:gpu-1x"
    elif count <= 4:
        return f"{GHCR_IMAGE_BASE}:gpu-4x"
    else:
        return f"{GHCR_IMAGE_BASE}:gpu-8x"


def main() -> None:
    args = _parse_args()
    _ensure_mutually_exclusive(args.dataset, args.dataset_path)

    # Normalize repo-relative paths so the container can access them.
    args.harbor_config = _repo_relative(args.harbor_config)
    if args.datagen_config:
        args.datagen_config = _repo_relative(args.datagen_config)
    if args.dataset_path:
        args.dataset_path = _repo_relative(args.dataset_path)

    # Select and normalize docker image (auto-selects variant based on GPU count)
    docker_image = _normalize_docker_image(_select_docker_image(args))

    run_eval_cmd = _build_run_eval_command(args)
    run_eval_str = " ".join(shlex.quote(part) for part in run_eval_cmd)
    remote_cmds = ["cd /opt/openthoughts", "set -euo pipefail"]

    remote_secret_path = None
    if args.secrets_env:
        secret_src = Path(args.secrets_env).expanduser().resolve()
        if not secret_src.exists():
            raise FileNotFoundError(f"secrets env file not found: {secret_src}")
        remote_secret_path = "/tmp/openthoughts_secrets.env"
        remote_cmds.append(f"set -a && source {remote_secret_path} && set +a")

    remote_cmds.append(run_eval_str)
    if remote_secret_path:
        remote_cmds.append(f"rm -f {remote_secret_path}")
    final_cmd = " && ".join(remote_cmds)

    task = sky.Task(args.task_name)
    resources = sky.Resources(
        cloud=_resolve_cloud(args.cloud_provider),
        accelerators=args.accelerator,
        use_spot=args.use_spot,
        region=args.region,
        zone=args.zone,
        image_id=docker_image,
    )
    task.set_resources(resources)

    # Build file mounts
    file_mounts = {}
    if not args.no_sync:
        # Sync local codebase to remote VM (overwrites stale code baked into Docker image)
        file_mounts["/opt/openthoughts"] = REPO_ROOT.as_posix()
    if remote_secret_path:
        file_mounts[remote_secret_path] = os.path.abspath(args.secrets_env)
    if file_mounts:
        task.set_file_mounts(file_mounts)
    task.set_run(final_cmd)

    sync_status = "disabled (--no-sync)" if args.no_sync else "enabled"
    print(f"[cloud] Launching SkyPilot task '{args.task_name}'")
    print(f"[cloud]   Accelerator: {args.accelerator}")
    print(f"[cloud]   Image: {docker_image}")
    print(f"[cloud]   Code sync: {sync_status}")
    handle = sky.launch(
        task,
        cluster_name=args.cluster_name,
        detach_run=False,
        stream_logs=True,
    )
    if isinstance(handle, Sequence):
        handle = handle[0] if handle else None
    cluster_for_sync = getattr(handle, "cluster_name", None) or args.cluster_name
    sync_eval_outputs(
        cluster_name=cluster_for_sync,
        remote_path=args.remote_output_dir,
        local_dir=args.local_sync_dir,
    )


if __name__ == "__main__":
    main()
