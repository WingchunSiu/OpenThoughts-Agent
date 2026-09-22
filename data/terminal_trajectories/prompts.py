"""Prompt builders for trajectory-derived task synthesis."""

from __future__ import annotations

import json

from data.terminal_trajectories.schemas import TrajectoryRecord


def _provenance_context(record: TrajectoryRecord) -> str:
    selected_metadata = {
        key: value
        for key, value in record.metadata.items()
        if key in {"title", "description", "command", "shell", "os"}
    }
    return json.dumps(selected_metadata, ensure_ascii=False, sort_keys=True)


def solution_prompt(record: TrajectoryRecord) -> str:
    """Ask a teacher to distill a successful reference workflow."""

    return f"""You are reconstructing a self-contained terminal task from a noisy trajectory.

Infer the final successful workflow and return a clean Bash reference solution. Remove
prompts, command output, failed attempts, exploratory reads, secrets, user-specific
paths, and irrelevant repetition. Preserve only actions supported by the trajectory;
do not invent a different task. The script must be deterministic, use paths under
/app for persistent outputs, start with '#!/usr/bin/env bash' and 'set -euo pipefail',
and must not access credentials or authenticated services.

Return exactly one JSON object with this schema:
{{"solve_sh": "<complete Bash script>"}}

Source metadata: {_provenance_context(record)}

Trajectory:
---
{record.transcript}
---
"""


def merge_solution_prompt(scripts: list[str]) -> str:
    """Ask a teacher to merge independently distilled transcript chunks."""

    joined = "\n\n".join(
        f"--- CHUNK {index} ---\n{script}"
        for index, script in enumerate(scripts, start=1)
    )
    return f"""Merge these partial Bash workflows into one deterministic reference solution.

Preserve the successful ordering, remove duplicates and failed alternatives, and do not
invent actions unsupported by the partial scripts. The result must start with
'#!/usr/bin/env bash' and 'set -euo pipefail', and write persistent outputs under /app.

Return exactly one JSON object with this schema:
{{"solve_sh": "<complete merged Bash script>"}}

{joined}
"""


def instruction_prompt(record: TrajectoryRecord, solve_sh: str) -> str:
    """Ask a teacher for an outcome-oriented instruction."""

    return f"""Write the agent-facing instruction for a terminal task.

Describe the required final observable state, not the commands used to reach it. Do
not mention the trajectory, recording, reference solution, hidden tests, or benchmark.
Do not provide a step-by-step recipe. Mention every output path and exact format that a
correct verifier must inspect. Use one to three concise paragraphs.

Return exactly one JSON object with this schema:
{{"instruction_md": "<Markdown instruction>"}}

Source metadata: {_provenance_context(record)}

Reference solution used only to infer the outcome:
---
{solve_sh}
---
"""


def environment_prompt(instruction_md: str, solve_sh: str) -> str:
    """Choose one bounded environment profile."""

    return f"""Choose the smallest fixed environment profile capable of running this
terminal task. Task-specific dependencies may be installed by the agent/reference
solution at runtime; do not request a custom image.

Profiles:
- base: Bash, Git, curl, jq, Python, compiler toolchain.
- node: base plus Node.js and npm.
- system: base plus networking/process/archive/SQLite administration tools.

Return exactly one JSON object with this schema:
{{"environment_profile": "base|node|system"}}

Instruction:
---
{instruction_md}
---

Reference solution:
---
{solve_sh}
---
"""


def verifier_prompt(instruction_md: str, solve_sh: str) -> str:
    """Ask a teacher for a state-based pytest verifier."""

    return f"""Create a deterministic state-based pytest verifier for this terminal task.

The verifier runs after the agent in the same /app filesystem. Check the requested final
state rather than command history or stdout. Use only Python's standard library and
pytest. Do not execute or read /solution/solve.sh, do not mutate /app, do not access the
network, and do not encode arbitrary implementation details that are absent from the
instruction. Include enough independent assertions that a no-op state fails.

Return exactly one JSON object with this schema:
{{"test_state_py": "<complete Python source>"}}

Instruction:
---
{instruction_md}
---

Reference solution, provided only to understand the expected state:
---
{solve_sh}
---
"""
