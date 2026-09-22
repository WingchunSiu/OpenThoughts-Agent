"""Typed records shared by the trajectory-to-task pipeline."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path
from typing import Any


class SourceType(StrEnum):
    """Supported trajectory sources."""

    HUMAN_TERMINAL = "human_terminal"
    AGENT_TRAJECTORY = "agent_trajectory"


class FilterMode(StrEnum):
    """How deterministic audit findings affect task generation."""

    ANNOTATE = "annotate"
    ENFORCE = "enforce"


class EnvironmentProfile(StrEnum):
    """Bounded environment families used to keep snapshot counts small."""

    BASE = "base"
    NODE = "node"
    SYSTEM = "system"


@dataclass(frozen=True)
class TrajectoryRecord:
    """One normalized source trajectory with explicit provenance."""

    trajectory_id: str
    transcript: str
    source_type: SourceType
    source_license: str
    source_url: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class TrajectoryAudit:
    """Deterministic safety and reproducibility annotations for a trajectory."""

    trajectory_id: str
    reasons: tuple[str, ...]
    command_count: int
    line_count: int

    @property
    def clean(self) -> bool:
        return not self.reasons


@dataclass(frozen=True)
class GeneratedTask:
    """LLM-produced contents needed to materialize one Harbor task."""

    instruction_md: str
    solve_sh: str
    test_state_py: str
    environment_profile: EnvironmentProfile


@dataclass(frozen=True)
class GenerationSummary:
    """Paths and counts produced by one pipeline invocation."""

    tasks_dir: Path
    manifest_path: Path
    parquet_path: Path | None
    num_input: int
    num_generated: int
    num_excluded: int
    num_snapshots: int
