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
    from eval.cloud.providers import (
        get_all_provider_names,
        get_provider_config,
        resolve_cloud,
        check_provider_credentials,
        get_setup_instructions,
        list_providers,
    )
except ImportError:
    from providers import (  # type: ignore
        get_all_provider_names,
        get_provider_config,
        resolve_cloud,
        check_provider_credentials,
        get_setup_instructions,
        list_providers,
    )

# Handle --list-providers before importing sky (so it works without skypilot installed)
if "--list-providers" in sys.argv:
    print(list_providers(verbose=True))
    sys.exit(0)

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


# Output directory is relative to workdir - gets resolved at runtime based on sync mode
DEFAULT_REMOTE_OUTPUT_SUBDIR = "cloud_runs"
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
    parser.add_argument("--gpus", type=int, default=1, help="run_eval --gpus value.")
    parser.add_argument("--dry-run", action="store_true", help="Pass --dry-run to run_eval.")
    parser.add_argument("--job-name", help="Optional override for Harbor job name.")

    # Cloud specific options
    parser.add_argument(
        "--cloud-provider",
        default="gcp",
        help="Cloud provider(s) to use. Comma-separated for fallbacks (e.g., 'gcp,aws,lambda'). "
        "Run with --list-providers for details.",
    )
    parser.add_argument(
        "--list-providers",
        action="store_true",
        help="List supported cloud providers and exit.",
    )
    parser.add_argument(
        "--region",
        help="Preferred region(s). Comma-separated for fallbacks (e.g., 'us-central1,us-west1,europe-west1').",
    )
    parser.add_argument("--zone", help="Preferred zone.")
    parser.add_argument(
        "--accelerator",
        default="A100:1",
        help="SkyPilot accelerator spec(s). Comma-separated for fallback options "
        "(e.g., 'H100:1,H200:1,A100-80GB:1'). Run 'sky show-gpus' to list options.",
    )
    parser.add_argument("--use-spot", action="store_true", help="Use spot/preemptible instances.")
    parser.add_argument(
        "--docker-image",
        default=DEFAULT_DOCKER_IMAGE,
        help="Pre-built Docker image (default: auto-selects gpu-1x/4x/8x based on accelerator count). "
        "Build images with: ./docker/build_and_push.sh",
    )
    parser.add_argument("--task-name", default="ot-eval-cloud", help="SkyPilot task name.")
    parser.add_argument("--cluster-name", help="Optional SkyPilot cluster name override.")
    parser.add_argument(
        "--remote-output-subdir",
        default=DEFAULT_REMOTE_OUTPUT_SUBDIR,
        help="Subdirectory under workdir for outputs (default: 'cloud_runs').",
    )
    parser.add_argument("--local-sync-dir", default=DEFAULT_LOCAL_SYNC_DIR)
    parser.add_argument("--secrets-env", help="Path to secrets.env to source inside the container.")
    parser.add_argument(
        "--no-sync",
        action="store_true",
        help="Skip syncing local codebase to VM (use code baked into Docker image).",
    )
    parser.add_argument(
        "--autostop",
        type=int,
        default=30,
        metavar="MINUTES",
        help="Auto-stop cluster after N minutes of idle time (default: 30). Set to -1 to disable.",
    )
    parser.add_argument(
        "--down",
        action="store_true",
        help="Tear down cluster after task completes (default: keep cluster for reuse).",
    )

    return parser.parse_args()


def _ensure_mutually_exclusive(dataset: Optional[str], dataset_path: Optional[str]) -> None:
    if dataset and dataset_path:
        raise ValueError("Specify either --dataset or --dataset-path (not both).")
    if not dataset and not dataset_path:
        raise ValueError("Must provide --dataset or --dataset-path for eval workloads.")


def _build_run_eval_command(args: argparse.Namespace, remote_output_dir: str) -> List[str]:
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
            remote_output_dir,
        ]
    )
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

    # --list-providers is handled early (before sky import), but handle it here too for completeness
    if args.list_providers:
        print(list_providers(verbose=True))
        return

    _ensure_mutually_exclusive(args.dataset, args.dataset_path)

    # Parse comma-separated providers
    provider_names = [p.strip() for p in args.cloud_provider.split(",")]
    provider_configs = [get_provider_config(p) for p in provider_names]

    # Check credentials for all providers
    for pname, pconfig in zip(provider_names, provider_configs):
        creds_ok, creds_msg = check_provider_credentials(pname)
        if not creds_ok:
            print(f"[cloud] Warning ({pconfig.display_name}): {creds_msg}", file=sys.stderr)

    # Warn about provider limitations (use first provider's config for compatibility checks)
    primary_config = provider_configs[0]
    if args.use_spot and not all(pc.supports_spot for pc in provider_configs):
        no_spot = [pc.display_name for pc in provider_configs if not pc.supports_spot]
        print(
            f"[cloud] Warning: {', '.join(no_spot)} do not support spot instances. "
            "Spot will only be used where supported.",
            file=sys.stderr,
        )

    if args.region and not all(pc.supports_regions for pc in provider_configs):
        no_region = [pc.display_name for pc in provider_configs if not pc.supports_regions]
        print(
            f"[cloud] Warning: {', '.join(no_region)} do not support region selection. "
            "Region will only be used where supported.",
            file=sys.stderr,
        )

    # Normalize repo-relative paths so the container can access them.
    args.harbor_config = _repo_relative(args.harbor_config)
    if args.datagen_config:
        args.datagen_config = _repo_relative(args.datagen_config)
    if args.dataset_path:
        args.dataset_path = _repo_relative(args.dataset_path)

    # Select and normalize docker image (auto-selects variant based on GPU count)
    # Some providers (like RunPod) don't support docker as runtime environment
    if all(pc.supports_docker_runtime for pc in provider_configs):
        docker_image = _normalize_docker_image(_select_docker_image(args))
    else:
        no_docker = [pc.display_name for pc in provider_configs if not pc.supports_docker_runtime]
        if len(no_docker) == len(provider_configs):
            # None support docker
            docker_image = None
            print(
                f"[cloud] Note: Selected providers do not support Docker as runtime. "
                "Using provider's default environment.",
                file=sys.stderr,
            )
        else:
            # Some support, some don't - use docker where supported
            docker_image = _normalize_docker_image(_select_docker_image(args))
            print(
                f"[cloud] Note: {', '.join(no_docker)} do not support Docker as runtime. "
                "Docker will only be used where supported.",
                file=sys.stderr,
            )

    # Remote working directory - use /sky/workdir when syncing to avoid conflict with Docker image's /opt/openthoughts
    remote_workdir = "/sky/workdir" if not args.no_sync else "/opt/openthoughts"

    # Compute the actual output directory (under workdir)
    remote_output_dir = f"{remote_workdir}/{args.remote_output_subdir}"

    run_eval_cmd = _build_run_eval_command(args, remote_output_dir)
    run_eval_str = " ".join(shlex.quote(part) for part in run_eval_cmd)
    remote_cmds = [f"cd {remote_workdir}", "set -euo pipefail"]

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

    # Build file mounts
    file_mounts = {}
    if not args.no_sync:
        # Sync local codebase to remote VM at /sky/workdir (avoids conflict with Docker's /opt/openthoughts)
        file_mounts[remote_workdir] = REPO_ROOT.as_posix()
    if remote_secret_path:
        file_mounts[remote_secret_path] = os.path.abspath(args.secrets_env)

    # Build Resources with provider-specific settings
    # Support comma-separated accelerators, providers, and regions for fallback options
    accelerator_options = [a.strip() for a in args.accelerator.split(",")]
    region_options = [r.strip() for r in args.region.split(",")] if args.region else [None]

    def build_resource(provider_name: str, pconfig, accel: str, region: Optional[str]) -> sky.Resources:
        kwargs = {
            "cloud": resolve_cloud(provider_name),
            "accelerators": accel,
            "use_spot": args.use_spot if pconfig.supports_spot else False,
        }
        if region and pconfig.supports_regions:
            kwargs["region"] = region
        if args.zone:
            kwargs["zone"] = args.zone
        if docker_image and pconfig.supports_docker_runtime:
            kwargs["image_id"] = docker_image
        return sky.Resources(**kwargs)

    # Build all combinations of providers, accelerators, and regions
    all_resources = []
    for pname, pconfig in zip(provider_names, provider_configs):
        for accel in accelerator_options:
            for region in region_options:
                all_resources.append(build_resource(pname, pconfig, accel, region))

    if len(all_resources) == 1:
        resources = all_resources[0]
    else:
        # Multiple options: SkyPilot will try in order of cost
        resources = set(all_resources)

    task = sky.Task(name=args.task_name, run=final_cmd)
    task.set_resources(resources)
    if file_mounts:
        task.set_file_mounts(file_mounts)

    sync_status = f"enabled -> {remote_workdir}" if not args.no_sync else "disabled (using Docker image)"
    image_status = docker_image if docker_image else "(provider default)"
    provider_status = " | ".join(pc.display_name for pc in provider_configs)
    accel_status = " | ".join(accelerator_options)
    region_status = " | ".join(r for r in region_options if r) if any(region_options) else "auto"
    autostop_status = f"{args.autostop} min" if args.autostop > 0 else "disabled"
    print(f"[cloud] Launching SkyPilot task '{args.task_name}'")
    print(f"[cloud]   Provider(s): {provider_status}")
    print(f"[cloud]   Region(s): {region_status}")
    print(f"[cloud]   Accelerator(s): {accel_status}")
    print(f"[cloud]   Image: {image_status}")
    print(f"[cloud]   Code sync: {sync_status}")
    print(f"[cloud]   Autostop: {autostop_status}")
    if len(all_resources) > 1:
        print(f"[cloud]   Candidate resources: {len(all_resources)} combinations")
    if args.down:
        print(f"[cloud]   Teardown: enabled (cluster will be deleted after task)")

    # Launch returns a request ID; use stream_and_get to wait for provisioning
    # Note: Don't pass down=True to launch() - we need to sync outputs first, then tear down
    launch_kwargs = {"cluster_name": args.cluster_name}
    if args.autostop > 0:
        launch_kwargs["idle_minutes_to_autostop"] = args.autostop

    request_id = sky.launch(task, **launch_kwargs)
    launch_result = sky.stream_and_get(request_id)

    job_id: Optional[int] = None
    handle = None
    if isinstance(launch_result, tuple) and len(launch_result) == 2:
        potential_job_id, potential_handle = launch_result
        if isinstance(potential_job_id, int) or potential_job_id is None:
            job_id = potential_job_id
            handle = potential_handle
        else:
            handle = launch_result[0]
    elif isinstance(launch_result, Sequence):
        handle = launch_result[0] if launch_result else None
    else:
        handle = launch_result

    cluster_for_sync = getattr(handle, "cluster_name", None) or args.cluster_name

    # Wait for the job to complete by tailing logs
    # sky.launch() only waits for submission, not completion
    if cluster_for_sync:
        print(f"[cloud] Waiting for job to complete on cluster '{cluster_for_sync}'...")
        try:
            # tail_logs blocks until the job finishes and streams output
            if job_id is not None:
                sky.tail_logs(cluster_for_sync, job_id=job_id, follow=True)
            else:
                print("[cloud] Job ID unavailable; streaming latest job logs (may include older runs).")
                sky.tail_logs(cluster_for_sync, job_id=None, follow=True)
        except Exception as e:
            print(f"[cloud] Warning: Failed to tail logs: {e}", file=sys.stderr)
            print(f"[cloud] You can manually check status with: sky queue {cluster_for_sync}", file=sys.stderr)

    # Sync outputs from remote cluster
    # 1. Sync the experiments directory (logs, endpoint JSON, etc.)
    sync_eval_outputs(
        cluster_name=cluster_for_sync,
        remote_path=remote_output_dir,
        local_dir=args.local_sync_dir,
    )

    # 2. Sync trace_jobs directory (Harbor job outputs - the actual results)
    # Harbor puts outputs in ./trace_jobs relative to workdir
    trace_jobs_remote = f"{remote_workdir}/trace_jobs"
    trace_jobs_local = str(Path(args.local_sync_dir).parent / "trace_jobs")
    print(f"[cloud-sync] Also syncing Harbor trace_jobs from {trace_jobs_remote}...")
    sync_eval_outputs(
        cluster_name=cluster_for_sync,
        remote_path=trace_jobs_remote,
        local_dir=trace_jobs_local,
    )

    # Tear down cluster if requested (after syncing)
    if args.down and cluster_for_sync:
        print(f"[cloud] Tearing down cluster '{cluster_for_sync}'...")
        try:
            down_request = sky.down(cluster_for_sync)
            sky.stream_and_get(down_request)
            print(f"[cloud] Cluster '{cluster_for_sync}' terminated.")
        except Exception as e:
            print(f"[cloud] Warning: Failed to tear down cluster: {e}", file=sys.stderr)
            print(f"[cloud] Run manually: sky down {cluster_for_sync}", file=sys.stderr)


if __name__ == "__main__":
    main()
