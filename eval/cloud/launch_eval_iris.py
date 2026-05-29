#!/usr/bin/env python3
"""Launch OpenThoughts evals on Marin's Iris TPU cluster.

Iris analog of ``eval/cloud/launch_eval_cloud.py``. Shape mirrors the SkyPilot
launcher exactly so muscle memory carries over — same arg names, same flow.
The differences are all behind the IrisLauncher base.

Output handling: by default outputs are rsync'd back to ``--local-sync-dir``
periodically while the job runs, so downstream eval-analysis tooling sees
local files. Pass ``--output-mode gcs --gcs-output-dir gs://...`` to skip
the rsync layer and have the workload write straight to GCS instead.

Harbor environment: defaults to ``daytona`` (the only sandbox backend that
works on iris workers without DinD). Passing ``--harbor_env docker`` is not
gated — the job will fail at runtime because iris doesn't mount
/var/run/docker.sock into task containers.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import List, Optional

# Add repo root to sys.path for imports
_repo_root = Path(__file__).resolve().parents[2]
if str(_repo_root) not in sys.path:
    sys.path.append(str(_repo_root))

from hpc.iris_launch_utils import IrisLauncher
from hpc.cloud_launch_utils import repo_relative, parse_gpu_count, infer_harbor_env_from_config
from hpc.arg_groups import (
    add_harbor_args,
    add_harbor_env_arg,
    add_model_compute_args,
    add_hf_upload_args,
    add_database_upload_args,
)
from hpc.harbor_utils import load_harbor_config
from hpc.datagen_config_utils import parse_datagen_config
from hpc.launch_utils import PROJECT_ROOT


class EvalIrisLauncher(IrisLauncher):
    """Iris launcher for eval/local/run_eval.py."""

    task_name = "ot-eval-iris"
    job_name_prefix = "eval-iris"
    default_n_concurrent = 16

    def add_task_specific_args(self, parser: argparse.ArgumentParser) -> None:
        """Mirror EvalCloudLauncher's args exactly so users don't have to relearn flags."""
        add_harbor_args(parser, config_required=True)

        add_model_compute_args(
            parser,
            model_required=False,  # Can be inferred from datagen_config
            default_n_concurrent=self.default_n_concurrent,
            default_n_attempts=3,
            n_attempts_help="Times to run each task for standard error calculation (default: 3).",
        )

        # Default to daytona; docker passes through and fails organically on iris.
        add_harbor_env_arg(
            parser,
            default=self.default_harbor_env,
            legacy_names=["--eval-env", "--eval_env"],
        )

        parser.add_argument("--datagen_config",
                            help="Optional datagen config to seed defaults.")
        parser.add_argument("--datagen-config", dest="datagen_config", help=argparse.SUPPRESS)

        parser.add_argument("--dataset",
                            help="Harbor dataset slug (exclusive with --dataset_path).")
        parser.add_argument("--dataset_path",
                            help="Path to tasks directory (exclusive with --dataset).")
        parser.add_argument("--dataset-path", dest="dataset_path", help=argparse.SUPPRESS)

        parser.add_argument("--ray_object_store_gb", "--ray-object-store-gb",
                            type=float, default=None,
                            help="Ray object store (plasma) size in GB.")

        # NOTE: --job_name comes from add_harbor_args above.

        add_hf_upload_args(parser)
        add_database_upload_args(parser)

    def normalize_paths(self, args: argparse.Namespace) -> None:
        if args.dataset and args.dataset_path:
            raise ValueError("Specify either --dataset or --dataset-path (not both).")
        if not args.dataset and not args.dataset_path:
            raise ValueError("Must provide --dataset or --dataset-path for eval workloads.")

        # --gpus is the downstream run_eval.py knob for vLLM tensor_parallel_size.
        # On TPU, derive it from the TPU variant's chip count.
        if args.gpus is None:
            try:
                chips = int(args.tpu.rsplit("-", 1)[-1])
                args.gpus = chips
            except (ValueError, AttributeError):
                args.gpus = parse_gpu_count(getattr(args, "accelerator", "") or "")

        args.harbor_config = repo_relative(args.harbor_config, self.repo_root)
        if args.datagen_config:
            args.datagen_config = repo_relative(args.datagen_config, self.repo_root)
        if args.dataset_path and not args.dataset_path.startswith("/"):
            args.dataset_path = repo_relative(args.dataset_path, self.repo_root)

        infer_harbor_env_from_config(args, args.harbor_config, log_prefix="[eval-iris]")

        if not args.agent:
            harbor_cfg = load_harbor_config(args.harbor_config)
            agents = harbor_cfg.get("agents", [])
            if agents and isinstance(agents, list) and len(agents) > 0:
                inferred_agent = agents[0].get("name")
                if inferred_agent:
                    args.agent = inferred_agent
                    print(f"[eval-iris] Inferred --agent={inferred_agent} from harbor config")

        if not args.model and args.datagen_config:
            try:
                parsed = parse_datagen_config(args.datagen_config)
                if parsed.model:
                    args.model = parsed.model
                    print(f"[eval-iris] Inferred --model={parsed.model} from datagen config")
            except Exception as e:
                print(f"[eval-iris] Warning: Could not parse datagen config for model: {e}")

        if not args.model:
            raise ValueError("Must provide --model or --datagen_config (to infer model from engine.model)")
        if not args.agent:
            raise ValueError("Must provide --agent or ensure harbor config has agents[0].name")

        if args.harbor_env == "docker":
            print(
                "[eval-iris] WARNING: --harbor_env=docker on an iris worker requires "
                "/var/run/docker.sock mounted into the task container; iris workers don't "
                "do that by default. Job will likely fail. Use --harbor_env=daytona.",
                file=sys.stderr,
            )

    def build_task_command(self, args: argparse.Namespace, remote_output_dir: str) -> List[str]:
        cmd: List[str] = [
            "python", "eval/local/run_eval.py",
            "--harbor_config", args.harbor_config,
            "--model", args.model,
        ]

        if args.datagen_config:
            cmd.extend(["--datagen_config", args.datagen_config])
        if args.dataset:
            cmd.extend(["--dataset", args.dataset])
        elif args.dataset_path:
            cmd.extend(["--dataset_path", args.dataset_path])

        cmd.extend([
            "--agent", args.agent,
            "--n_concurrent", str(args.n_concurrent),
            "--n_attempts", str(args.n_attempts),
            "--gpus", str(args.gpus),
            "--experiments_dir", remote_output_dir,
        ])

        if args.harbor_env:
            cmd.extend(["--harbor_env", args.harbor_env])

        if args.job_name:
            cmd.extend(["--job_name", args.job_name])
        if args.dry_run:
            cmd.append("--dry_run")

        if args.ray_object_store_gb is not None:
            cmd.extend(["--ray_object_store_gb", str(args.ray_object_store_gb)])

        for kwarg in args.agent_kwarg:
            cmd.extend(["--agent_kwarg", kwarg])
        # Auto-inject --jobs-dir so harbor writes outputs to the same GCS
        # prefix the workload's --experiments_dir points at. With harbor's
        # UPath patch (penfever/otagent-latest @ dc41d295a4) this routes
        # all per-job/per-trial writes through fsspec to GCS instead of
        # local /app/trace_jobs/. User --harbor_extra_arg entries follow
        # below so an explicit --harbor_extra_arg=--jobs-dir=... wins.
        cmd.append(f"--harbor_extra_arg=--jobs-dir={remote_output_dir}")
        for extra in args.harbor_extra_arg:
            # Use the `=` form so argparse on the worker side accepts values
            # that start with `-` (e.g. --harbor_extra_arg=--n-tasks). The
            # space form `--harbor_extra_arg --n-tasks` trips argparse's
            # "looks like an option" heuristic and gets rejected with
            # "argument --harbor_extra_arg: expected one argument".
            cmd.append(f"--harbor_extra_arg={extra}")

        if args.upload_to_database:
            cmd.append("--upload_to_database")
        if args.upload_username:
            cmd.extend(["--upload_username", args.upload_username])
        if args.upload_error_mode:
            cmd.extend(["--upload_error_mode", args.upload_error_mode])
        if args.upload_hf_repo:
            cmd.extend(["--upload_hf_repo", args.upload_hf_repo])
        if args.upload_hf_token:
            cmd.extend(["--upload_hf_token", args.upload_hf_token])
        if args.upload_hf_private:
            cmd.append("--upload_hf_private")
        if args.upload_hf_episodes:
            cmd.extend(["--upload_hf_episodes", args.upload_hf_episodes])
        if args.upload_forced_update:
            cmd.append("--upload_forced_update")

        return cmd


def main() -> None:
    launcher = EvalIrisLauncher(PROJECT_ROOT)
    parser = launcher.create_argument_parser(
        description="Launch eval/local/run_eval.py on a Marin Iris TPU worker."
    )
    args = parser.parse_args()
    sys.exit(launcher.run(args))


if __name__ == "__main__":
    main()
