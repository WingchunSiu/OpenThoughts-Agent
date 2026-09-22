"""End-to-end local trajectory-to-Harbor task pipeline."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from data.terminal_trajectories.audit import audit_trajectory, redacted_trajectory
from data.terminal_trajectories.harbor_task import (
    snapshot_count,
    validate_task_directory,
    write_task,
)
from data.terminal_trajectories.schemas import (
    FilterMode,
    GenerationSummary,
    TrajectoryRecord,
)
from data.terminal_trajectories.synthesis import (
    SynthesisSettings,
    TextGenerator,
    synthesize_task,
)
from scripts.harbor.tasks_parquet_converter import find_tasks, to_parquet


def _manifest_row(
    record: TrajectoryRecord,
    *,
    audit_reasons: tuple[str, ...],
    status: str,
    task_path: str | None,
) -> dict[str, Any]:
    return {
        "trajectory_id": record.trajectory_id,
        "source_type": record.source_type.value,
        "source_license": record.source_license,
        "source_url": record.source_url,
        "audit_reasons": list(audit_reasons),
        "status": status,
        "task_path": task_path,
    }


def _write_manifest(path: Path, rows: list[dict[str, Any]]) -> None:
    path.write_text(
        "".join(
            json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows
        ),
        encoding="utf-8",
    )


def generate_tasks(
    records: list[TrajectoryRecord],
    *,
    output_dir: Path,
    engine: TextGenerator,
    filter_mode: FilterMode,
    generation_kwargs: dict[str, Any] | None = None,
    package_parquet: bool = True,
) -> GenerationSummary:
    """Generate, validate, and optionally package tasks from normalized records."""

    output_dir.mkdir(parents=True, exist_ok=True)
    tasks_dir = output_dir / "tasks"
    tasks_dir.mkdir(exist_ok=False)
    manifest_path = output_dir / "manifest.jsonl"
    task_dirs: list[Path] = []
    manifest_rows: list[dict[str, Any]] = []
    excluded = 0
    settings = SynthesisSettings(generation_kwargs=generation_kwargs or {})

    for record in records:
        audit = audit_trajectory(record)
        if filter_mode is FilterMode.ENFORCE and not audit.clean:
            excluded += 1
            manifest_rows.append(
                _manifest_row(
                    record,
                    audit_reasons=audit.reasons,
                    status="excluded",
                    task_path=None,
                )
            )
            continue

        generated = synthesize_task(redacted_trajectory(record), engine, settings)
        task_dir = write_task(tasks_dir, record, audit, generated)
        validate_task_directory(task_dir)
        task_dirs.append(task_dir)
        manifest_rows.append(
            _manifest_row(
                record,
                audit_reasons=audit.reasons,
                status="generated",
                task_path=task_dir.name,
            )
        )

    _write_manifest(manifest_path, manifest_rows)
    parquet_path: Path | None = None
    if package_parquet:
        parquet_path = output_dir / "tasks.parquet"
        to_parquet(
            tasks_dir,
            parquet_path,
            find_tasks(tasks_dir, recursive=True),
            compression="gz",
        )

    return GenerationSummary(
        tasks_dir=tasks_dir,
        manifest_path=manifest_path,
        parquet_path=parquet_path,
        num_input=len(records),
        num_generated=len(task_dirs),
        num_excluded=excluded,
        num_snapshots=snapshot_count(task_dirs),
    )
