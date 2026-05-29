#!/usr/bin/env python3
"""Submit the GCS->S3 mirror script as an iris job.

Companion to ``launch_hf_mirror.py`` — same shape, different transfer
direction. Pulls from our GCS staging at
``gs://marin-eu-west4/ot-agent/models/`` and pushes into an S3 bucket
(e.g. the LAION ``mmlaion`` bucket at Jülich) so that vLLM's
``runai_streamer`` can read the weights via real S3 protocol.

Usage::

    python -m scripts.iris.launch_gcs_to_s3 \\
        --repo cyankiwi/MiniMax-M2.7-AWQ-4bit \\
        --repo google/gemma-4-31B-it \\
        --repo QuantTrio/Qwen3.5-397B-A17B-AWQ \\
        --gcs-prefix gs://marin-eu-west4/ot-agent/models \\
        --s3-bucket mmlaion \\
        --s3-prefix ot-agent/models \\
        --s3-endpoint https://just-object.fz-juelich.de:9000 \\
        --secrets-env ~/Documents/secrets.env
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_repo_root = Path(__file__).resolve().parents[2]
if str(_repo_root) not in sys.path:
    sys.path.append(str(_repo_root))

from hpc.iris_launch_utils import IrisLauncher
from hpc.launch_utils import PROJECT_ROOT


class GcsToS3Launcher(IrisLauncher):
    task_name = "ot-gcs2s3"
    job_name_prefix = "gcs2s3"
    default_tpu = "v6e-4"

    def add_task_specific_args(self, parser: argparse.ArgumentParser) -> None:
        parser.add_argument("--repo", action="append", required=True,
                            help="HF model repo id (repeatable).")
        parser.add_argument("--gcs-prefix", "--gcs_prefix", required=True,
                            help="Source GCS prefix; src is <prefix>/<repo>/...")
        parser.add_argument("--s3-bucket", "--s3_bucket", required=True)
        parser.add_argument("--s3-prefix", "--s3_prefix", required=True)
        parser.add_argument("--s3-endpoint", "--s3_endpoint", default=None,
                            help="S3-compatible endpoint URL "
                                 "(e.g. MinIO). Falls back to AWS_ENDPOINT_URL "
                                 "on the worker if omitted.")
        parser.add_argument("--job_name",
                            help="Override the auto-generated iris job name.")
        parser.add_argument("--dry_run", action="store_true")

    def normalize_paths(self, args: argparse.Namespace) -> None:
        if not args.gcs_prefix.startswith("gs://"):
            raise SystemExit("--gcs-prefix must start with gs://")

    def build_task_command(self, args: argparse.Namespace,
                           remote_output_dir: str) -> list[str]:
        cmd = ["python", "scripts/iris/mirror_gcs_to_s3.py",
               "--gcs-prefix", args.gcs_prefix,
               "--s3-bucket", args.s3_bucket,
               "--s3-prefix", args.s3_prefix]
        if args.s3_endpoint:
            cmd.extend(["--s3-endpoint", args.s3_endpoint])
        for r in args.repo:
            cmd.extend(["--repo", r])
        return cmd


def main() -> None:
    launcher = GcsToS3Launcher(PROJECT_ROOT)
    parser = launcher.create_argument_parser(
        description="Re-mirror HF models from GCS staging into an S3 bucket.",
    )
    args = parser.parse_args()
    sys.exit(launcher.run(args))


if __name__ == "__main__":
    main()
