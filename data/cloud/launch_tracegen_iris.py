#!/usr/bin/env python3
"""Launch OpenThoughts trace generation on Marin's Iris TPU cluster.

Iris analog of ``data/cloud/launch_tracegen_cloud.py``. See
``eval/cloud/launch_eval_iris.py`` and ``hpc/iris_launch_utils.py`` for
the shared design notes (rsync vs gcs outputs, daytona-default Harbor env,
docker-not-gated, multi-host TPU is scaffolded but untested).
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

import yaml

from hpc.iris_launch_utils import IrisLauncher
from hpc.cloud_launch_utils import repo_relative, parse_gpu_count, infer_harbor_env_from_config
from hpc.arg_groups import (
    add_harbor_args,
    add_harbor_env_arg,
    add_model_compute_args,
    add_hf_upload_args,
    add_tasks_input_arg,
)
from hpc.launch_utils import PROJECT_ROOT


def _env_vars_from_datagen_yaml(path: Path) -> dict:
    """Lift a top-level ``env_vars:`` block from a datagen YAML, if present.

    Lets a per-config knob override launcher-side env defaults (e.g.
    `MODEL_IMPL_TYPE=auto` for models that prefer the JAX-native flax_nnx
    path). Returns ``{}`` if no env_vars block is declared.
    """
    if not path or not path.exists():
        return {}
    try:
        with path.open() as f:
            cfg = yaml.safe_load(f) or {}
    except Exception as e:
        print(f"[tracegen-iris] WARNING: failed to parse {path} for env_vars: {e}", file=sys.stderr)
        return {}
    env = cfg.get("env_vars") or {}
    if not isinstance(env, dict):
        return {}
    return {str(k): str(v) for k, v in env.items()}


class TracegenIrisLauncher(IrisLauncher):
    """Iris launcher for data/local/run_tracegen.py."""

    task_name = "ot-tracegen-iris"
    job_name_prefix = "tracegen-iris"
    default_n_concurrent = 64

    def add_task_specific_args(self, parser: argparse.ArgumentParser) -> None:
        add_harbor_args(parser, config_required=True)

        add_model_compute_args(
            parser,
            model_required=False,
            default_n_concurrent=self.default_n_concurrent,
            default_n_attempts=1,
            n_attempts_help="Times to run each task for repeated trials (default: 1).",
        )

        add_harbor_env_arg(
            parser,
            default=self.default_harbor_env,
            legacy_names=["--trace-env", "--trace_env"],
        )

        parser.add_argument("--datagen_config", required=True,
                            help="Datagen config with vLLM settings (required).")
        parser.add_argument("--datagen-config", dest="datagen_config", help=argparse.SUPPRESS)

        add_tasks_input_arg(parser, required=True)

        # NOTE: --job_name comes from add_harbor_args above.

        add_hf_upload_args(parser)

    def normalize_paths(self, args: argparse.Namespace) -> None:
        # On TPU, --gpus drives vLLM tensor_parallel_size — derive from TPU chip count.
        if args.gpus is None:
            try:
                chips = int(args.tpu.rsplit("-", 1)[-1])
                args.gpus = chips
            except (ValueError, AttributeError):
                args.gpus = parse_gpu_count(getattr(args, "accelerator", "") or "")

        args.harbor_config = repo_relative(args.harbor_config, self.repo_root)
        args.datagen_config = repo_relative(args.datagen_config, self.repo_root)
        # --tasks_input_path is overloaded: local FS path | HF dataset id | harbor slug.
        # Only run it through repo_relative when it's actually a local FS path that
        # exists under the repo. HF ids like ``mlfoundations-dev/foo`` would otherwise
        # be resolved against CWD and either rewritten into a bogus repo-relative
        # path or raise ValueError if CWD escapes the repo.
        if not args.tasks_input_path.startswith("/"):
            if (self.repo_root / args.tasks_input_path).exists():
                args.tasks_input_path = repo_relative(args.tasks_input_path, self.repo_root)

        infer_harbor_env_from_config(args, args.harbor_config, log_prefix="[tracegen-iris]")

        if args.harbor_env == "docker":
            print(
                "[tracegen-iris] WARNING: --harbor_env=docker on an iris worker requires "
                "/var/run/docker.sock mounted into the task container; iris workers don't "
                "do that by default. Job will likely fail. Use --harbor_env=daytona.",
                file=sys.stderr,
            )

    def build_task_command(self, args: argparse.Namespace, remote_output_dir: str) -> List[str]:
        cmd: List[str] = [
            "python", "data/local/run_tracegen.py",
            "--harbor_config", args.harbor_config,
            "--datagen_config", args.datagen_config,
            "--tasks_input_path", args.tasks_input_path,
        ]

        if args.model:
            cmd.extend(["--model", args.model])

        cmd.extend([
            "--agent", args.agent,
            "--n_concurrent", str(args.n_concurrent),
            "--n_attempts", str(args.n_attempts),
            "--gpus", str(args.gpus),
            "--experiments_dir", remote_output_dir,
        ])

        if args.harbor_env:
            cmd.extend(["--harbor_env", args.harbor_env])

        # When --resume-from is active, the iris-level job_name is fresh
        # (timestamped for iris uniqueness) but the harbor identity must be
        # the OLD job's name so harbor's _maybe_init_existing_job picks up
        # the existing config.json / trials. IrisLauncher.run() stashes the
        # old name on args._harbor_job_name_override.
        harbor_job_name = getattr(args, "_harbor_job_name_override", None) or args.job_name
        if harbor_job_name:
            cmd.extend(["--job_name", harbor_job_name])
        if args.dry_run:
            cmd.append("--dry_run")

        for kwarg in args.agent_kwarg:
            cmd.extend(["--agent_kwarg", kwarg])
        # Auto-inject --jobs-dir — see comment in eval/cloud/launch_eval_iris.py.
        cmd.append(f"--harbor_extra_arg=--jobs-dir={remote_output_dir}")
        for extra in args.harbor_extra_arg:
            # `=` form for argparse accept of `-`-prefixed values; see
            # the same comment in eval/cloud/launch_eval_iris.py.
            cmd.append(f"--harbor_extra_arg={extra}")

        if args.upload_hf_repo:
            cmd.extend(["--upload_hf_repo", args.upload_hf_repo])
        if args.upload_hf_token:
            cmd.extend(["--upload_hf_token", args.upload_hf_token])
        if args.upload_hf_private:
            cmd.append("--upload_hf_private")

        return cmd

    def build_env(self, args: argparse.Namespace) -> dict:
        """Lift a per-config ``env_vars:`` block from the datagen YAML.

        Returned entries are merged BEFORE the launcher's setdefaults in
        IrisLauncher.run(), so they win — e.g. setting
        ``env_vars: {MODEL_IMPL_TYPE: auto}`` in a gemma4 YAML overrides
        the launcher-wide default of `MODEL_IMPL_TYPE=vllm` (which is
        the right default only for AWQ workloads). Other env defaults
        the launcher provides (UV_LINK_MODE, OT_AGENT_SKIP_SFT_SYNC,
        etc) remain in effect via setdefault.
        """
        env = super().build_env(args)
        # args.datagen_config has been normalized to a repo-relative string in
        # normalize_paths(); resolve it against repo_root for file open().
        if args.datagen_config:
            yaml_path = self.repo_root / args.datagen_config
            env.update(_env_vars_from_datagen_yaml(yaml_path))
        return env


def main() -> None:
    launcher = TracegenIrisLauncher(PROJECT_ROOT)
    parser = launcher.create_argument_parser(
        description="Launch data/local/run_tracegen.py on a Marin Iris TPU worker."
    )
    args = parser.parse_args()
    sys.exit(launcher.run(args))


if __name__ == "__main__":
    main()
