"""Generate TaskTrove-ready Harbor tasks from local terminal trajectories.

Examples:
    python -m data.terminal_trajectories.generate \
        --input recordings/ \
        --source-type human_terminal \
        --source-license CC-BY-4.0 \
        --engine-config hpc/datagen_yaml/example.yaml \
        --output-dir outputs/terminal-trajectories

    python -m data.terminal_trajectories.generate \
        --input agent-trajectories.jsonl \
        --source-type agent_trajectory \
        --source-license Apache-2.0 \
        --engine-config hpc/datagen_yaml/example.yaml \
        --output-dir outputs/agent-trajectories \
        --filter-mode annotate
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

from data.generation.engines import create_inference_engine
from data.generation.utils import load_datagen_config, resolve_engine_runtime
from data.terminal_trajectories.ingest import load_trajectories
from data.terminal_trajectories.pipeline import generate_tasks
from data.terminal_trajectories.schemas import FilterMode, SourceType


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate Harbor tasks from human or agent terminal trajectories"
    )
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--source-type",
        type=SourceType,
        choices=list(SourceType),
        required=True,
    )
    parser.add_argument(
        "--source-license",
        help="Fallback license when input records do not declare one",
    )
    parser.add_argument(
        "--filter-mode",
        type=FilterMode,
        choices=list(FilterMode),
        default=FilterMode.ANNOTATE,
        help="annotate keeps flagged records; enforce excludes them",
    )
    parser.add_argument("--engine-config", type=Path, required=True)
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Maximum inputs for a smoke run; non-positive values mean all",
    )
    parser.add_argument(
        "--no-package",
        action="store_true",
        help="Write task directories and manifest without tasks.parquet",
    )
    parser.add_argument(
        "--log-level",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        default="INFO",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s %(levelname)s %(message)s",
    )

    records = load_trajectories(
        args.input,
        source_type=args.source_type,
        source_license=args.source_license,
    )
    if args.limit is not None and args.limit > 0:
        records = records[: args.limit]

    loaded = load_datagen_config(args.engine_config)
    runtime = resolve_engine_runtime(loaded.config)
    if runtime.type == "none":
        raise ValueError("trajectory synthesis requires an inference engine")
    engine = create_inference_engine(runtime.type, **runtime.engine_kwargs)
    if engine.requires_initial_healthcheck and not engine.healthcheck():
        raise RuntimeError("inference engine healthcheck failed")

    generation_kwargs = dict(runtime.request_params)
    if runtime.max_output_tokens is not None:
        generation_kwargs.setdefault("max_tokens", runtime.max_output_tokens)

    summary = generate_tasks(
        records,
        output_dir=args.output_dir,
        engine=engine,
        filter_mode=args.filter_mode,
        generation_kwargs=generation_kwargs,
        package_parquet=not args.no_package,
    )
    print(
        json.dumps(
            {
                "tasks_dir": str(summary.tasks_dir),
                "manifest_path": str(summary.manifest_path),
                "parquet_path": (
                    str(summary.parquet_path) if summary.parquet_path else None
                ),
                "num_input": summary.num_input,
                "num_generated": summary.num_generated,
                "num_excluded": summary.num_excluded,
                "num_snapshots": summary.num_snapshots,
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
