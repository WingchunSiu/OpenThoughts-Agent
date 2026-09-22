"""Normalize local terminal recordings and agent trajectories."""

from __future__ import annotations

import json
from collections.abc import Iterable, Iterator
from pathlib import Path
from typing import Any

from data.terminal_trajectories.schemas import SourceType, TrajectoryRecord


def _content_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, dict):
        nested = content.get("content") or content.get("text")
        if nested is not None:
            return _content_text(nested)
        return json.dumps(content, ensure_ascii=False, sort_keys=True)
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict) and isinstance(item.get("text"), str):
                parts.append(item["text"])
        return "\n".join(parts)
    if content is None:
        return ""
    return json.dumps(content, ensure_ascii=False, sort_keys=True)


def _messages_transcript(messages: Iterable[dict[str, Any]]) -> str:
    lines: list[str] = []
    for message in messages:
        role = str(message.get("role") or "unknown")
        content = _content_text(message.get("content"))
        if content:
            lines.append(f"[{role}]\n{content}")
        tool_calls = message.get("tool_calls")
        if tool_calls:
            lines.append(
                "[tool_calls]\n"
                + json.dumps(tool_calls, ensure_ascii=False, sort_keys=True)
            )
    return "\n\n".join(lines)


def _steps_transcript(steps: Iterable[dict[str, Any]]) -> str:
    lines: list[str] = []
    for step in steps:
        source = str(step.get("source") or "agent")
        message = _content_text(step.get("message"))
        if message:
            lines.append(f"[{source}]\n{message}")
        tool_calls = step.get("tool_calls")
        if tool_calls:
            lines.append(
                "[tool_calls]\n"
                + json.dumps(tool_calls, ensure_ascii=False, sort_keys=True)
            )
        observation = step.get("observation")
        if observation:
            lines.append(
                "[observation]\n"
                + json.dumps(observation, ensure_ascii=False, sort_keys=True)
            )
    return "\n\n".join(lines)


def _row_transcript(row: dict[str, Any]) -> str:
    transcript = row.get("transcript")
    if isinstance(transcript, str) and transcript.strip():
        return transcript

    steps = row.get("steps")
    if isinstance(steps, list):
        rendered = _steps_transcript(step for step in steps if isinstance(step, dict))
        if rendered:
            return rendered

    messages = row.get("messages") or row.get("conversations")
    if isinstance(messages, list):
        rendered = _messages_transcript(
            message for message in messages if isinstance(message, dict)
        )
        if rendered:
            return rendered

    raise ValueError("trajectory row has no transcript, steps, or messages")


def _required_license(row: dict[str, Any], fallback: str | None) -> str:
    value = row.get("source_license") or row.get("license") or fallback
    if not isinstance(value, str) or not value.strip():
        raise ValueError("trajectory source license is required")
    return value.strip()


def load_jsonl(
    path: Path,
    *,
    source_type: SourceType,
    source_license: str | None,
) -> list[TrajectoryRecord]:
    """Load normalized or Harbor-style trajectory rows from JSONL."""

    records: list[TrajectoryRecord] = []
    with path.open(encoding="utf-8") as source:
        for line_number, line in enumerate(source, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row, dict):
                raise TypeError(f"{path}:{line_number}: expected a JSON object")
            trajectory_id = row.get("trajectory_id") or row.get("id")
            if not isinstance(trajectory_id, str) or not trajectory_id.strip():
                raise ValueError(
                    f"{path}:{line_number}: trajectory_id or id is required"
                )
            row_source_type = SourceType(row.get("source_type", source_type.value))
            metadata = row.get("metadata") or {}
            if not isinstance(metadata, dict):
                raise TypeError(f"{path}:{line_number}: metadata must be an object")
            source_url = row.get("source_url")
            if source_url is not None and not isinstance(source_url, str):
                raise ValueError(f"{path}:{line_number}: source_url must be a string")
            records.append(
                TrajectoryRecord(
                    trajectory_id=trajectory_id.strip(),
                    transcript=_row_transcript(row),
                    source_type=row_source_type,
                    source_license=_required_license(row, source_license),
                    source_url=source_url,
                    metadata=metadata,
                )
            )
    return records


def _recording_directories(path: Path) -> Iterator[Path]:
    if (path / "recording.txt").is_file():
        yield path
        return
    for child in sorted(path.iterdir()):
        if child.is_dir() and (child / "recording.txt").is_file():
            yield child


def load_recording_directory(
    path: Path,
    *,
    source_type: SourceType,
    source_license: str | None,
) -> list[TrajectoryRecord]:
    """Load Asciinema-style ``recording.txt`` plus optional ``info.json`` dirs."""

    records: list[TrajectoryRecord] = []
    for recording_dir in _recording_directories(path):
        info_path = recording_dir / "info.json"
        info: dict[str, Any] = {}
        if info_path.is_file():
            loaded = json.loads(info_path.read_text(encoding="utf-8"))
            if not isinstance(loaded, dict):
                raise ValueError(f"{info_path}: expected a JSON object")
            info = loaded
        source_url = info.get("source_url") or info.get("url")
        if source_url is not None and not isinstance(source_url, str):
            raise ValueError(f"{info_path}: source URL must be a string")
        records.append(
            TrajectoryRecord(
                trajectory_id=recording_dir.name,
                transcript=(recording_dir / "recording.txt").read_text(
                    encoding="utf-8", errors="replace"
                ),
                source_type=source_type,
                source_license=_required_license(info, source_license),
                source_url=source_url,
                metadata=info,
            )
        )
    if not records:
        raise ValueError(f"no recording.txt files found under {path}")
    return records


def load_trajectories(
    path: Path,
    *,
    source_type: SourceType,
    source_license: str | None,
) -> list[TrajectoryRecord]:
    """Load a JSONL file or directory of terminal recordings."""

    if path.is_file():
        return load_jsonl(
            path,
            source_type=source_type,
            source_license=source_license,
        )
    if path.is_dir():
        return load_recording_directory(
            path,
            source_type=source_type,
            source_license=source_license,
        )
    raise FileNotFoundError(path)
