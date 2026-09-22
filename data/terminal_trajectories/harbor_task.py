"""Materialize current Harbor tasks with bounded environment profiles."""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

from harbor.models.task.task import Task

from data.terminal_trajectories.schemas import (
    EnvironmentProfile,
    GeneratedTask,
    TrajectoryAudit,
    TrajectoryRecord,
)

UPSTREAM_METHOD_REVISION = (
    "EuniAI/TerminalWorld@784698ba93735470ce1664bff2ec44bcd7b28e15"
)

_BASE_INSTALL = """\
RUN apt-get update && apt-get install -y --no-install-recommends \\
    bash \\
    build-essential \\
    ca-certificates \\
    curl \\
    git \\
    jq \\
    python3 \\
    python3-pip \\
    python3-venv \\
"""

_DOCKERFILES = {
    EnvironmentProfile.BASE: f"""\
FROM ubuntu:24.04

ARG DEBIAN_FRONTEND=noninteractive
{_BASE_INSTALL} \
    && rm -rf /var/lib/apt/lists/*
RUN python3 -m venv /opt/verifier \\
    && /opt/verifier/bin/pip install --no-cache-dir pytest==8.4.1
WORKDIR /app
""",
    EnvironmentProfile.NODE: f"""\
FROM ubuntu:24.04

ARG DEBIAN_FRONTEND=noninteractive
{_BASE_INSTALL} \
    nodejs \\
    npm \\
    && rm -rf /var/lib/apt/lists/*
RUN python3 -m venv /opt/verifier \\
    && /opt/verifier/bin/pip install --no-cache-dir pytest==8.4.1
WORKDIR /app
""",
    EnvironmentProfile.SYSTEM: f"""\
FROM ubuntu:24.04

ARG DEBIAN_FRONTEND=noninteractive
{_BASE_INSTALL} \
    iproute2 \\
    openssh-client \\
    procps \\
    sqlite3 \\
    unzip \\
    zip \\
    && rm -rf /var/lib/apt/lists/*
RUN python3 -m venv /opt/verifier \\
    && /opt/verifier/bin/pip install --no-cache-dir pytest==8.4.1
WORKDIR /app
""",
}

_TEST_SH = """\
#!/usr/bin/env bash
set -uo pipefail

mkdir -p /logs/verifier
set +e
/opt/verifier/bin/python -m pytest -q /tests/test_state.py
status=$?
set -e

case "$status" in
  0) printf '1\n' > /logs/verifier/reward.txt ;;
  1) printf '0\n' > /logs/verifier/reward.txt ;;
  *) exit "$status" ;;
esac
"""


def _toml_string(value: str) -> str:
    return json.dumps(value, ensure_ascii=False)


def task_slug(trajectory_id: str) -> str:
    """Return a stable Harbor-safe task name for a source trajectory ID."""

    normalized = re.sub(r"[^a-z0-9-]+", "-", trajectory_id.lower()).strip("-")
    if not normalized:
        normalized = "trajectory"
    digest = hashlib.sha256(trajectory_id.encode()).hexdigest()[:8]
    return f"trajectory-{normalized[:48]}-{digest}"


def render_task_toml(
    record: TrajectoryRecord,
    audit: TrajectoryAudit,
    generated: GeneratedTask,
    slug: str,
) -> str:
    """Render canonical Harbor 1.4 configuration with source provenance."""

    metadata_lines = [
        f"trajectory_id = {_toml_string(record.trajectory_id)}",
        f"source_type = {_toml_string(record.source_type.value)}",
        f"source_license = {_toml_string(record.source_license)}",
        f"environment_profile = {_toml_string(generated.environment_profile.value)}",
        f"audit_reasons = {json.dumps(list(audit.reasons), ensure_ascii=False)}",
        f"method_reference = {_toml_string(UPSTREAM_METHOD_REVISION)}",
    ]
    if record.source_url:
        metadata_lines.append(f"source_url = {_toml_string(record.source_url)}")

    return (
        'schema_version = "1.4"\n\n'
        "[task]\n"
        f'name = "trajectory-tasks/{slug}"\n'
        'version = "1.0.0"\n'
        'description = "Terminal task synthesized from a licensed source trajectory"\n'
        'keywords = ["terminal", "trajectory-derived"]\n\n'
        "[metadata]\n" + "\n".join(metadata_lines) + "\n\n[agent]\n"
        "timeout_sec = 900.0\n\n"
        "[verifier]\n"
        "timeout_sec = 120.0\n\n"
        "[environment]\n"
        'network_mode = "public"\n'
        "build_timeout_sec = 900.0\n"
        "cpus = 2\n"
        "memory_mb = 4096\n"
        "storage_mb = 10240\n"
    )


def write_task(
    tasks_dir: Path,
    record: TrajectoryRecord,
    audit: TrajectoryAudit,
    generated: GeneratedTask,
) -> Path:
    """Write one task without overwriting an existing directory."""

    slug = task_slug(record.trajectory_id)
    task_dir = tasks_dir / slug
    task_dir.mkdir(parents=True, exist_ok=False)
    (task_dir / "environment").mkdir()
    (task_dir / "solution").mkdir()
    (task_dir / "tests").mkdir()

    (task_dir / "instruction.md").write_text(generated.instruction_md, encoding="utf-8")
    (task_dir / "task.toml").write_text(
        render_task_toml(record, audit, generated, slug), encoding="utf-8"
    )
    (task_dir / "environment" / "Dockerfile").write_text(
        _DOCKERFILES[generated.environment_profile], encoding="utf-8"
    )
    solve_path = task_dir / "solution" / "solve.sh"
    solve_path.write_text(generated.solve_sh, encoding="utf-8")
    solve_path.chmod(0o755)
    test_path = task_dir / "tests" / "test.sh"
    test_path.write_text(_TEST_SH, encoding="utf-8")
    test_path.chmod(0o755)
    (task_dir / "tests" / "test_state.py").write_text(
        generated.test_state_py, encoding="utf-8"
    )
    return task_dir


def validate_task_directory(task_dir: Path) -> Task:
    """Load a task through Harbor and enforce this pipeline's file contract."""

    task = Task(task_dir)
    if not task.instruction.strip():
        raise ValueError(f"{task_dir}: instruction is empty")
    for relative_path in (
        "solution/solve.sh",
        "tests/test.sh",
        "tests/test_state.py",
        "environment/Dockerfile",
    ):
        path = task_dir / relative_path
        if not path.is_file() or path.stat().st_size == 0:
            raise ValueError(f"{task_dir}: missing or empty {relative_path}")
    return task


def snapshot_count(task_dirs: list[Path]) -> int:
    """Count distinct Dockerfile contents in generated tasks."""

    digests = {
        hashlib.sha256((task_dir / "environment" / "Dockerfile").read_bytes()).digest()
        for task_dir in task_dirs
    }
    return len(digests)
