from __future__ import annotations

import io
import json
import tarfile
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq
import pytest

from data.terminal_trajectories.audit import audit_trajectory
from data.terminal_trajectories.harbor_task import validate_task_directory
from data.terminal_trajectories.ingest import load_trajectories
from data.terminal_trajectories.pipeline import generate_tasks
from data.terminal_trajectories.schemas import FilterMode, SourceType, TrajectoryRecord
from data.terminal_trajectories.synthesis import SynthesisSettings, synthesize_task


class ScriptedEngine:
    def __init__(self, responses: list[dict[str, str]]) -> None:
        self.responses = list(responses)
        self.prompts: list[str] = []

    def generate(self, prompt: str, **generation_kwargs: Any) -> str:
        self.prompts.append(prompt)
        return json.dumps(self.responses.pop(0))


def test_agent_trajectory_generates_loadable_tasktrove_row(tmp_path: Path) -> None:
    source_path = tmp_path / "trajectories.jsonl"
    source_path.write_text(
        json.dumps(
            {
                "trajectory_id": "agent/run:1",
                "source_type": "agent_trajectory",
                "source_license": "Apache-2.0",
                "source_url": "https://example.test/run/1",
                "steps": [
                    {
                        "source": "agent",
                        "message": "$ export CONTACT_EMAIL=person@example.com",
                    },
                    {
                        "source": "agent",
                        "message": "$ printf 'hello' > /app/result.txt",
                    },
                    {
                        "source": "environment",
                        "observation": {"exit_code": 0, "stdout": ""},
                    },
                ],
            }
        )
        + "\n",
        encoding="utf-8",
    )
    records = load_trajectories(
        source_path,
        source_type=SourceType.AGENT_TRAJECTORY,
        source_license=None,
    )
    engine = ScriptedEngine(
        [
            {"solve_sh": "printf 'hello' > /app/result.txt"},
            {
                "instruction_md": (
                    "Create `/app/result.txt` containing exactly the text `hello`."
                )
            },
            {"environment_profile": "base"},
            {
                "test_state_py": (
                    "from pathlib import Path\n\n"
                    "def test_result():\n"
                    "    assert Path('/app/result.txt').read_text() == 'hello'\n"
                )
            },
        ]
    )

    summary = generate_tasks(
        records,
        output_dir=tmp_path / "output",
        engine=engine,
        filter_mode=FilterMode.ANNOTATE,
    )

    assert summary.num_input == 1
    assert summary.num_generated == 1
    assert summary.num_excluded == 0
    assert summary.num_snapshots == 1
    assert len(engine.prompts) == 4
    assert "person@example.com" not in engine.prompts[0]
    assert "<redacted:pii_email>" in engine.prompts[0]

    task_dirs = list(summary.tasks_dir.iterdir())
    assert len(task_dirs) == 1
    task = validate_task_directory(task_dirs[0])
    assert (
        task.instruction
        == "Create `/app/result.txt` containing exactly the text `hello`.\n"
    )
    assert task.config.schema_version == "1.4"

    manifest = summary.manifest_path.read_text(encoding="utf-8")
    assert "https://example.test/run/1" in manifest
    assert "pii_email" in manifest
    assert "person@example.com" not in manifest
    assert "printf 'hello'" not in manifest

    assert summary.parquet_path is not None
    table = pq.read_table(summary.parquet_path)
    assert table.num_rows == 1
    row = table.to_pylist()[0]
    assert row["path"] == task_dirs[0].name
    with tarfile.open(fileobj=io.BytesIO(row["task_binary"]), mode="r:gz") as archive:
        assert "instruction.md" in archive.getnames()
        instruction = archive.extractfile("instruction.md")
        assert instruction is not None
        assert b"/app/result.txt" in instruction.read()


def test_filter_modes_keep_annotations_separate_from_selection(tmp_path: Path) -> None:
    record = TrajectoryRecord(
        trajectory_id="flagged",
        transcript="$ export API_TOKEN=abcdefghijklmnopqrstuvwxyz123456\n$ echo done\n",
        source_type=SourceType.HUMAN_TERMINAL,
        source_license="CC-BY-4.0",
    )
    audit = audit_trajectory(record)
    assert "credential_assignment" in audit.reasons

    engine = ScriptedEngine([])
    summary = generate_tasks(
        [record],
        output_dir=tmp_path / "enforced",
        engine=engine,
        filter_mode=FilterMode.ENFORCE,
        package_parquet=False,
    )

    assert summary.num_generated == 0
    assert summary.num_excluded == 1
    assert summary.num_snapshots == 0
    assert engine.prompts == []
    manifest_row = json.loads(summary.manifest_path.read_text(encoding="utf-8"))
    assert manifest_row["status"] == "excluded"
    assert manifest_row["audit_reasons"] == ["credential_assignment"]


def test_recording_directory_requires_and_preserves_source_license(
    tmp_path: Path,
) -> None:
    recording_dir = tmp_path / "recordings" / "human-42"
    recording_dir.mkdir(parents=True)
    (recording_dir / "recording.txt").write_text(
        "$ mkdir -p /app/demo\n$ touch /app/demo/ready\n", encoding="utf-8"
    )
    (recording_dir / "info.json").write_text(
        json.dumps(
            {
                "title": "Create demo marker",
                "license": "CC-BY-4.0",
                "url": "https://example.test/recording/42",
            }
        ),
        encoding="utf-8",
    )

    records = load_trajectories(
        tmp_path / "recordings",
        source_type=SourceType.HUMAN_TERMINAL,
        source_license=None,
    )

    assert records == [
        TrajectoryRecord(
            trajectory_id="human-42",
            transcript="$ mkdir -p /app/demo\n$ touch /app/demo/ready\n",
            source_type=SourceType.HUMAN_TERMINAL,
            source_license="CC-BY-4.0",
            source_url="https://example.test/recording/42",
            metadata={
                "title": "Create demo marker",
                "license": "CC-BY-4.0",
                "url": "https://example.test/recording/42",
            },
        )
    ]


def test_long_trajectory_merges_chunk_solutions() -> None:
    record = TrajectoryRecord(
        trajectory_id="long-run",
        transcript="$ printf a > /app/a\n$ printf b > /app/b\n",
        source_type=SourceType.AGENT_TRAJECTORY,
        source_license="Apache-2.0",
    )
    engine = ScriptedEngine(
        [
            {"solve_sh": "printf a > /app/a"},
            {"solve_sh": "printf b > /app/b"},
            {"solve_sh": "printf a > /app/a\nprintf b > /app/b"},
            {"instruction_md": "Create `/app/a` and `/app/b`."},
            {"environment_profile": "base"},
            {
                "test_state_py": (
                    "from pathlib import Path\n\n"
                    "def test_outputs():\n"
                    "    assert Path('/app/a').read_text() == 'a'\n"
                    "    assert Path('/app/b').read_text() == 'b'\n"
                )
            },
        ]
    )

    task = synthesize_task(
        record,
        engine,
        SynthesisSettings(generation_kwargs={}, max_lines_per_chunk=1),
    )

    assert len(engine.prompts) == 6
    assert "printf a > /app/a" in task.solve_sh
    assert "printf b > /app/b" in task.solve_sh


def test_generated_verifier_cannot_mutate_task_state() -> None:
    record = TrajectoryRecord(
        trajectory_id="unsafe-verifier",
        transcript="$ printf ok > /app/result.txt\n",
        source_type=SourceType.AGENT_TRAJECTORY,
        source_license="Apache-2.0",
    )
    engine = ScriptedEngine(
        [
            {"solve_sh": "printf ok > /app/result.txt"},
            {"instruction_md": "Create `/app/result.txt`."},
            {"environment_profile": "base"},
            {
                "test_state_py": (
                    "from pathlib import Path\n\n"
                    "def test_result():\n"
                    "    Path('/app/result.txt').write_text('ok')\n"
                    "    assert Path('/app/result.txt').exists()\n"
                )
            },
        ]
    )

    with pytest.raises(ValueError, match="may not mutate"):
        synthesize_task(record, engine, SynthesisSettings(generation_kwargs={}))
