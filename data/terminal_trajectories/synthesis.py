"""Teacher-driven trajectory-to-task synthesis."""

from __future__ import annotations

import ast
import json
from dataclasses import dataclass, replace
from typing import Any, Protocol

from data.terminal_trajectories.prompts import (
    environment_prompt,
    instruction_prompt,
    merge_solution_prompt,
    solution_prompt,
    verifier_prompt,
)
from data.terminal_trajectories.schemas import (
    EnvironmentProfile,
    GeneratedTask,
    TrajectoryRecord,
)


class TextGenerator(Protocol):
    """The OT-Agent inference-engine surface used by this pipeline."""

    def generate(self, prompt: str, **generation_kwargs: Any) -> str:
        """Return generated text for one prompt."""


@dataclass(frozen=True)
class SynthesisSettings:
    """Generation arguments shared by all four teacher calls."""

    generation_kwargs: dict[str, Any]
    line_char_limit: int = 200
    max_lines_per_chunk: int = 500


def _json_object(response: str) -> dict[str, Any]:
    text = response.strip()
    if text.startswith("```"):
        first_newline = text.find("\n")
        last_fence = text.rfind("```")
        if first_newline >= 0 and last_fence > first_newline:
            text = text[first_newline + 1 : last_fence].strip()
    start = text.find("{")
    end = text.rfind("}")
    if start < 0 or end < start:
        raise ValueError("teacher response does not contain a JSON object")
    parsed = json.loads(text[start : end + 1])
    if not isinstance(parsed, dict):
        raise TypeError("teacher response must be a JSON object")
    return parsed


def _required_text(payload: dict[str, Any], key: str) -> str:
    value = payload.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"teacher response is missing non-empty {key!r}")
    return value.strip()


def _normalized_solution(script: str) -> str:
    lines = script.replace("\r\n", "\n").splitlines()
    if lines and lines[0].startswith("#!"):
        lines = lines[1:]
    while lines and not lines[0].strip():
        lines.pop(0)
    if lines and lines[0].strip() in {"set -e", "set -eu", "set -euo pipefail"}:
        lines = lines[1:]
    body = "\n".join(lines).strip()
    if not body:
        raise ValueError("reference solution contains no commands")
    return f"#!/usr/bin/env bash\nset -euo pipefail\n\n{body}\n"


def _trajectory_chunks(
    transcript: str,
    *,
    line_char_limit: int,
    max_lines_per_chunk: int,
) -> list[str]:
    lines = [
        line
        if len(line) <= line_char_limit
        else line[:line_char_limit] + " …[truncated]"
        for line in transcript.splitlines()
    ]
    return [
        "\n".join(lines[start : start + max_lines_per_chunk])
        for start in range(0, len(lines), max_lines_per_chunk)
    ] or [""]


def _validated_test(source: str) -> str:
    normalized = source.replace("\r\n", "\n").strip() + "\n"
    tree = ast.parse(normalized)
    lowered = normalized.lower()
    if "/solution" in lowered or "solve.sh" in lowered:
        raise ValueError("verifier may not inspect or execute the reference solution")
    if not any(isinstance(node, ast.Assert) for node in ast.walk(tree)):
        raise ValueError("verifier must contain at least one assertion")
    blocked_modules = {"http", "requests", "socket", "subprocess", "urllib"}
    mutating_methods = {
        "chmod",
        "mkdir",
        "remove",
        "rename",
        "replace",
        "rmdir",
        "rmtree",
        "touch",
        "unlink",
        "write_bytes",
        "write_text",
    }
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            modules = {alias.name.split(".", maxsplit=1)[0] for alias in node.names}
            if modules & blocked_modules:
                raise ValueError("verifier may not import network or process modules")
        elif isinstance(node, ast.ImportFrom) and node.module:
            if node.module.split(".", maxsplit=1)[0] in blocked_modules:
                raise ValueError("verifier may not import network or process modules")
        elif isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
            if node.func.attr in mutating_methods:
                raise ValueError("verifier may not mutate the task filesystem")
    return normalized


def synthesize_task(
    record: TrajectoryRecord,
    engine: TextGenerator,
    settings: SynthesisSettings,
) -> GeneratedTask:
    """Run the four TerminalWorld-style synthesis stages for one trajectory."""

    chunks = _trajectory_chunks(
        record.transcript,
        line_char_limit=settings.line_char_limit,
        max_lines_per_chunk=settings.max_lines_per_chunk,
    )
    partial_scripts: list[str] = []
    for chunk in chunks:
        chunk_record = replace(record, transcript=chunk)
        solve_payload = _json_object(
            engine.generate(solution_prompt(chunk_record), **settings.generation_kwargs)
        )
        partial_scripts.append(_required_text(solve_payload, "solve_sh"))
    if len(partial_scripts) > 1:
        merged_payload = _json_object(
            engine.generate(
                merge_solution_prompt(partial_scripts), **settings.generation_kwargs
            )
        )
        solve_source = _required_text(merged_payload, "solve_sh")
    else:
        solve_source = partial_scripts[0]
    solve_sh = _normalized_solution(solve_source)

    instruction_payload = _json_object(
        engine.generate(
            instruction_prompt(record, solve_sh), **settings.generation_kwargs
        )
    )
    instruction_md = _required_text(instruction_payload, "instruction_md") + "\n"

    environment_payload = _json_object(
        engine.generate(
            environment_prompt(instruction_md, solve_sh),
            **settings.generation_kwargs,
        )
    )
    profile = EnvironmentProfile(
        _required_text(environment_payload, "environment_profile")
    )

    verifier_payload = _json_object(
        engine.generate(
            verifier_prompt(instruction_md, solve_sh),
            **settings.generation_kwargs,
        )
    )
    test_state_py = _validated_test(_required_text(verifier_payload, "test_state_py"))

    return GeneratedTask(
        instruction_md=instruction_md,
        solve_sh=solve_sh,
        test_state_py=test_state_py,
        environment_profile=profile,
    )
