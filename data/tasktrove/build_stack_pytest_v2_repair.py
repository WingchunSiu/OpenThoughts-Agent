#!/usr/bin/env python3
"""Repair DCAgent/exp_rpt_stack-pytest-v2 with the v4.13 pytest reward contract.

Every source verifier uses `trap cleanup EXIT` and therefore writes reward 0 or 1
for pip failures, pytest crashes, collection errors, and zero-test sessions.
Harbor cannot retry those as infrastructure. This builder keeps the packaged
tests, installs extra dependencies without `|| true`, and maps ordinary
collection / zero-test / test-failure outcomes to reward 0 while leaving
dependency, malformed-test, and unexpected runner failures without a reward.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import re
import subprocess
import warnings
from collections import Counter
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
from harbor_config.models.task.config import TaskConfig
from huggingface_hub import hf_hub_download

from data.tasktrove.build_storage_repair import (
    MAX_BATCH_ROWS,
    MIN_TASKS,
    REQUIRED_MEMBERS,
    TASK_SCHEMA,
    file_sha256,
    read_task,
    write_task,
)

TASKTROVE_REPO = "open-thoughts/TaskTrove"
SOURCE_DATASET = "DCAgent__exp_rpt_stack-pytest-v2"
SOURCE_SHA256 = "8bfac7f44ff1ea23db6f516802a4072e549b780ec63f94b159980da2313f91b2"
OUTPUT_DATASET = "DCAgent__exp_rpt_stack-pytest-v3"
REQUIRED_SOURCE_MEMBERS = REQUIRED_MEMBERS | {"tests/test_solution.py"}
CHANGED_MEMBER_ALLOWLIST = frozenset(
    {
        "environment/Dockerfile",
        "tests/test.sh",
        "tests/requirements.txt",
    }
)
STDLIB_PIP_PACKAGES = frozenset({"__future__"})
PIP_NAME_FIXES = {
    "PIL": "Pillow",
    "bs4": "beautifulsoup4",
    "sklearn": "scikit-learn",
    "yaml": "PyYAML",
}
PIP_INSTALL_RE = re.compile(r"pip install --quiet ([^\n]+)")

PYTEST_DOCKERFILE = """FROM python:3.12-slim-bookworm

WORKDIR /app
RUN mkdir -p /output && chmod 777 /output
RUN apt-get update \\
    && apt-get install -y --no-install-recommends bsdutils git \\
    && rm -rf /var/lib/apt/lists/*
RUN python3 -m venv /app/.venv \\
    && /app/.venv/bin/pip install --no-cache-dir pytest
ENV PATH=/app/.venv/bin:$PATH
"""

# Byte-compatible with the TaskTrove v4.13 stack-pytest-large-v3 wrapper.
PYTEST_TEST_SH = r"""#!/bin/bash
set -euo pipefail

LOGS_DIR=/logs/verifier
REWARD="$LOGS_DIR/reward.txt"
mkdir -p "$LOGS_DIR"
rm -f "$REWARD"

# Dependency setup is infrastructure. Leave no reward if it fails so Harbor retries.
source /app/.venv/bin/activate
if [ -s /tests/requirements.txt ]; then
    pip install --quiet --disable-pip-version-check -r /tests/requirements.txt
fi
python3 -m pytest --version >/dev/null
python3 - <<'PY'
import ast
import pathlib

ast.parse(pathlib.Path("/tests/test_solution.py").read_text())
PY

export PYTHONPATH="/app${PYTHONPATH:+:$PYTHONPATH}"
cd /app
set +e
pytest /tests/test_solution.py -v --tb=short \
    --junitxml="$LOGS_DIR/pytest.xml" 2>&1 | tee "$LOGS_DIR/pytest_output.txt"
PYTEST_EXIT=${PIPESTATUS[0]}

SUMMARY=$(python3 - <<'PY'
import pathlib
import xml.etree.ElementTree as ET

report = pathlib.Path("/logs/verifier/pytest.xml")
if not report.is_file():
    raise SystemExit(2)
root = ET.parse(report).getroot()
cases = root.findall(".//testcase")
print(len(cases), sum(case.find("skipped") is not None for case in cases))
PY
)
SUMMARY_EXIT=$?
set -e

if [ "$SUMMARY_EXIT" -ne 0 ]; then
    if [ "$PYTEST_EXIT" -eq 2 ] || [ "$PYTEST_EXIT" -eq 5 ]; then
        echo 0 > "$REWARD"
        exit 1
    fi
    rm -f "$REWARD"
    exit "$SUMMARY_EXIT"
fi
read -r TOTAL SKIPPED <<< "$SUMMARY"
if [ "$TOTAL" -lt 1 ]; then
    echo 0 > "$REWARD"
    exit 1
fi
if [ "$PYTEST_EXIT" -ne 0 ] && [ "$PYTEST_EXIT" -ne 1 ]; then
    if [ "$PYTEST_EXIT" -eq 2 ] || [ "$PYTEST_EXIT" -eq 5 ]; then
        echo 0 > "$REWARD"
        exit 1
    fi
    rm -f "$REWARD"
    exit "$PYTEST_EXIT"
fi

if [ "$PYTEST_EXIT" -eq 0 ] && [ "$TOTAL" -ge 1 ] && [ "$SKIPPED" -eq 0 ]; then
    echo 1 > "$REWARD"
    exit 0
fi
echo 0 > "$REWARD"
exit 1
"""


def score_pytest_run(
    *,
    pytest_exit: int,
    total: int,
    skipped: int,
    summary_ok: bool,
) -> int | None:
    """Map a pytest invocation to a Harbor reward.

    Returns 1 or 0 for a scoreable agent outcome. Returns None when the
    verifier should leave no reward file so Harbor retries the trial.
    """
    if not summary_ok:
        if pytest_exit in (2, 5):
            return 0
        return None
    if total < 1:
        return 0
    if pytest_exit not in (0, 1):
        if pytest_exit in (2, 5):
            return 0
        return None
    if pytest_exit == 0 and skipped == 0:
        return 1
    return 0


def extract_requirements(test_sh: str) -> list[str]:
    """Return extra pip requirements from a v2 trap-based verifier."""
    packages: list[str] = []
    seen: set[str] = set()
    for match in PIP_INSTALL_RE.finditer(test_sh):
        rest = match.group(1)
        rest = rest.replace("2>/dev/null", "").replace("|| true", "")
        for raw in rest.split():
            if raw in {"pytest", *STDLIB_PIP_PACKAGES}:
                continue
            name = PIP_NAME_FIXES.get(raw, raw)
            if name not in seen:
                seen.add(name)
                packages.append(name)
    return packages


def drop_reasons(files: dict[str, bytes]) -> list[str]:
    """Return reasons a packaged pytest task cannot be repaired in place."""
    source = files["tests/test_solution.py"].decode(errors="replace")
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", SyntaxWarning)
            tree = ast.parse(source)
    except SyntaxError:
        return ["syntax_error"]
    tests = [
        node
        for node in ast.walk(tree)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        and node.name.startswith("test")
    ]
    if not tests:
        return ["no_tests"]
    return []


def _changed_members(
    original: dict[str, bytes], transformed: dict[str, bytes]
) -> set[str]:
    return {
        name
        for name in original.keys() | transformed.keys()
        if original.get(name) != transformed.get(name)
    }


def patch_task(files: dict[str, bytes]) -> dict[str, bytes]:
    """Replace the trap-always-reward verifier with the v4.13 pytest wrapper."""
    script = files["tests/test.sh"].decode()
    if "trap cleanup EXIT" not in script:
        raise ValueError("source verifier does not use the trap-always-reward contract")
    output = dict(files)
    requirements = extract_requirements(script)
    output["environment/Dockerfile"] = PYTEST_DOCKERFILE.encode()
    output["tests/test.sh"] = PYTEST_TEST_SH.encode()
    output["tests/requirements.txt"] = (
        ("\n".join(requirements) + "\n").encode() if requirements else b""
    )
    changed = _changed_members(files, output)
    unexpected = changed - CHANGED_MEMBER_ALLOWLIST
    if unexpected:
        raise ValueError(f"unexpected changed members: {sorted(unexpected)}")
    return output


def transform_task(
    files: dict[str, bytes],
) -> tuple[dict[str, bytes] | None, list[str]]:
    missing = REQUIRED_SOURCE_MEMBERS - files.keys()
    if missing:
        raise ValueError(f"incomplete task: {sorted(missing)}")
    reasons = drop_reasons(files)
    if reasons:
        return None, reasons
    return patch_task(files), []


def _validate_shell(script: bytes, validated: set[str]) -> None:
    digest_hex = hashlib.sha256(script).hexdigest()
    if digest_hex in validated:
        return
    result = subprocess.run(
        ["bash", "-n"], input=script, capture_output=True, check=False
    )
    if result.returncode != 0:
        raise ValueError(f"bash syntax failed: {result.stderr.decode()}")
    validated.add(digest_hex)


def validate_transformed_task(
    original: dict[str, bytes],
    transformed: dict[str, bytes],
    validated_shells: set[str],
) -> None:
    missing = REQUIRED_SOURCE_MEMBERS - transformed.keys()
    if missing:
        raise ValueError(f"transformed task missing members: {sorted(missing)}")
    changed = _changed_members(original, transformed)
    unexpected = changed - CHANGED_MEMBER_ALLOWLIST
    if unexpected:
        raise ValueError(f"unexpected changed members: {sorted(unexpected)}")
    TaskConfig.model_validate_toml(transformed["task.toml"].decode("utf-8"))
    _validate_shell(transformed["tests/test.sh"], validated_shells)
    if transformed["tests/test.sh"] != PYTEST_TEST_SH.encode():
        raise ValueError("transformed verifier is not the v4.13 pytest wrapper")
    if b"trap cleanup EXIT" in transformed["tests/test.sh"]:
        raise ValueError("transformed verifier still writes reward from EXIT")
    if b"|| true" in transformed["tests/test.sh"]:
        raise ValueError("transformed verifier swallows dependency failures")


def source_parquet(explicit: Path | None) -> Path:
    if explicit is not None:
        return explicit.resolve()
    return Path(
        hf_hub_download(
            TASKTROVE_REPO,
            f"{SOURCE_DATASET}/tasks.parquet",
            repo_type="dataset",
        )
    )


def build(source: Path, output: Path) -> dict[str, object]:
    if file_sha256(source) != SOURCE_SHA256:
        raise ValueError(f"source hash mismatch: {source}")
    parquet = pq.ParquetFile(source)
    if parquet.schema_arrow != TASK_SCHEMA:
        raise ValueError(f"unexpected source schema: {parquet.schema_arrow}")
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        raise FileExistsError(output)
    writer = pq.ParquetWriter(
        output, TASK_SCHEMA, compression="zstd", use_dictionary=False
    )
    source_rows = 0
    kept_rows = 0
    paths: set[str] = set()
    drop_counts: Counter[str] = Counter()
    dropped_paths: list[str] = []
    validated_shells: set[str] = set()
    try:
        for batch in parquet.iter_batches(batch_size=MAX_BATCH_ROWS):
            transformed_rows = []
            for row in batch.to_pylist():
                source_rows += 1
                path = row["path"]
                if path in paths:
                    raise ValueError(f"duplicate path: {path}")
                paths.add(path)
                files = read_task(row["task_binary"])
                transformed, reasons = transform_task(files)
                if transformed is None:
                    dropped_paths.append(path)
                    drop_counts.update(reasons)
                    continue
                validate_transformed_task(files, transformed, validated_shells)
                transformed_rows.append(
                    {"path": path, "task_binary": write_task(transformed)}
                )
                kept_rows += 1
            if transformed_rows:
                writer.write_table(
                    pa.Table.from_pylist(transformed_rows, schema=TASK_SCHEMA)
                )
    finally:
        writer.close()
    if kept_rows < MIN_TASKS:
        raise ValueError(f"standing-order minimum violated: {kept_rows}")
    dropped_file = output.parent / "dropped_paths.txt"
    dropped_file.write_text("".join(f"{path}\n" for path in dropped_paths))
    return {
        "source_dataset": SOURCE_DATASET,
        "output_dataset": OUTPUT_DATASET,
        "source_rows": source_rows,
        "kept_rows": kept_rows,
        "dropped_rows": source_rows - kept_rows,
        "drop_reasons": dict(sorted(drop_counts.items())),
        "source_sha256": SOURCE_SHA256,
        "output_sha256": file_sha256(output),
        "dropped_paths": dropped_paths,
        "dropped_paths_sha256": file_sha256(dropped_file),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", type=Path, required=True)
    parser.add_argument("--source", type=Path)
    args = parser.parse_args()
    output = args.stage / "datasets" / OUTPUT_DATASET / "tasks.parquet"
    report = build(source_parquet(args.source), output)
    (args.stage / "manifest.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
