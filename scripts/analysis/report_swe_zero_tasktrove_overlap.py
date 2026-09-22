#!/usr/bin/env python3
"""Estimate SWE-Zero task overlap with active TaskTrove SWE sources.

The report uses a deterministic, evenly spaced SWE-Zero sample. It reads only
TaskTrove Parquet ``path`` columns for cheap ID matching. Sources whose paths do
not preserve upstream IDs can be instruction-indexed explicitly; by default the
active SWE-Gym source is indexed because SWE-Zero directly cites SWE-Gym-Raw.

Example:
    python -m scripts.analysis.report_swe_zero_tasktrove_overlap \
        --sample-size 100 \
        --output-dir outputs/swe-zero-tasktrove-overlap
"""

from __future__ import annotations

import argparse
import hashlib
import html
import io
import json
import logging
import re
import tarfile
import unicodedata
from collections import Counter, defaultdict
from collections.abc import Iterable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any
from urllib.parse import urlencode
from urllib.request import urlopen

import pyarrow.parquet as pq
from huggingface_hub import HfApi, HfFileSystem, hf_hub_download
from rapidfuzz import fuzz, process

SWE_ZERO_REPO = "nvidia/SWE-Zero-openhands-trajectories"
TASKTROVE_REPO = "open-thoughts/TaskTrove"
DATASETS_SERVER = "https://datasets-server.huggingface.co"
DATASET_INSTANCE_RE = re.compile(r"(?im)^-\s*Dataset instance:\s*`?([^`\s]+)`?\s*$")
ISSUE_DESCRIPTION_RE = re.compile(
    r"<issue_description>\s*(.*?)\s*</issue_description>", re.IGNORECASE | re.DOTALL
)
TASKTROVE_PROBLEM_RE = re.compile(
    r"^## Problem Statement\s*$", re.IGNORECASE | re.MULTILINE
)
TASKTROVE_PROBLEM_END = re.compile(
    r"^(?:## Tests (?:that must keep passing|currently failing).*|"
    r"Apply code changes directly inside\b)",
    re.IGNORECASE | re.MULTILINE,
)
ARCHIVE_SUFFIXES = (".tar.gz", ".tgz", ".tar", ".gz")
LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class SampleTask:
    """One sampled SWE-Zero task."""

    row_index: int
    instance_id: str
    source_dataset: str
    repo: str
    license: str
    instruction: str
    normalized_instruction: str
    instruction_sha256: str


@dataclass(frozen=True)
class TaskTroveTask:
    """TaskTrove identity and optional instruction evidence."""

    source: str
    path: str
    instance_id: str | None = None
    normalized_instruction: str | None = None
    instruction_sha256: str | None = None


@dataclass(frozen=True)
class MatchResult:
    """Best overlap evidence for one sampled task."""

    row_index: int
    instance_id: str
    source_dataset: str
    repo: str
    license: str
    match_type: str
    tasktrove_source: str | None
    tasktrove_path: str | None
    instruction_similarity: float | None
    instruction_sha256: str


def normalized_instruction(text: str) -> str:
    """Return a stable representation for exact hashes and fuzzy matching."""

    text = unicodedata.normalize("NFKC", html.unescape(text)).casefold()
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def instruction_sha256(text: str) -> str:
    """Hash normalized instruction text."""

    return hashlib.sha256(text.encode()).hexdigest()


def canonical_task_id(value: str) -> str:
    """Normalize IDs without destroying the common ``owner__repo-number`` form."""

    normalized = value.strip().strip("`").casefold().replace("/", "__")
    for suffix in ARCHIVE_SUFFIXES:
        if normalized.endswith(suffix):
            normalized = normalized[: -len(suffix)]
            break
    return normalized


def swe_zero_instruction(row: dict[str, Any]) -> str:
    """Extract the original issue description from one SWE-Zero row."""

    trajectory = row.get("trajectory")
    if not isinstance(trajectory, list):
        raise TypeError("SWE-Zero row is missing a trajectory list")
    user_content = next(
        (
            turn.get("content")
            for turn in trajectory
            if isinstance(turn, dict)
            and turn.get("role") == "user"
            and isinstance(turn.get("content"), str)
        ),
        None,
    )
    if not user_content:
        raise ValueError("SWE-Zero trajectory is missing a user instruction")
    issue_match = ISSUE_DESCRIPTION_RE.search(user_content)
    if issue_match:
        return issue_match.group(1).strip()
    return re.sub(
        r"<uploaded_files>.*?</uploaded_files>",
        "",
        user_content,
        flags=re.IGNORECASE | re.DOTALL,
    ).strip()


def tasktrove_problem_statement(instruction: str) -> str:
    """Remove TaskTrove setup and test-installation wrappers from an SWE task."""

    problem_match = TASKTROVE_PROBLEM_RE.search(instruction)
    problem = instruction[problem_match.end() :] if problem_match else instruction
    end_match = TASKTROVE_PROBLEM_END.search(problem)
    if end_match:
        problem = problem[: end_match.start()]
    return problem.strip()


def archive_instruction(archive_bytes: bytes) -> str:
    """Read ``instruction.md`` without extracting an untrusted task archive."""

    with tarfile.open(fileobj=io.BytesIO(archive_bytes), mode="r:*") as archive:
        members = [
            member
            for member in archive.getmembers()
            if member.isfile()
            and (
                member.name == "instruction.md"
                or member.name.endswith("/instruction.md")
            )
        ]
        if len(members) != 1:
            raise ValueError(
                f"expected one instruction.md in task archive, found {len(members)}"
            )
        source = archive.extractfile(members[0])
        if source is None:
            raise ValueError("could not read instruction.md from task archive")
        return source.read().decode("utf-8", errors="replace")


def sample_windows(
    total_rows: int, sample_size: int, window_count: int
) -> list[tuple[int, int]]:
    """Return evenly spaced ``(offset, length)`` windows."""

    if total_rows <= 0:
        raise ValueError("total_rows must be positive")
    if sample_size <= 0:
        raise ValueError("sample_size must be positive")
    if window_count <= 0:
        raise ValueError("window_count must be positive")
    sample_size = min(sample_size, total_rows)
    window_count = min(window_count, sample_size)
    quotient, remainder = divmod(sample_size, window_count)
    lengths = [quotient + (index < remainder) for index in range(window_count)]
    if window_count == 1:
        return [(0, lengths[0])]
    return [
        (
            round(index * (total_rows - length) / (window_count - 1)),
            length,
        )
        for index, length in enumerate(lengths)
    ]


def _dataset_server_json(endpoint: str, **params: str | int) -> dict[str, Any]:
    url = f"{DATASETS_SERVER}/{endpoint}?{urlencode(params)}"
    with urlopen(url, timeout=60) as response:
        payload = json.load(response)
    if not isinstance(payload, dict):
        raise TypeError(f"datasets-server returned non-object payload for {endpoint}")
    return payload


def dataset_num_rows(repo_id: str, config: str, split: str) -> int:
    """Read a split size without downloading the dataset."""

    payload = _dataset_server_json("size", dataset=repo_id)
    split_sizes = payload.get("size", {}).get("splits", [])
    for item in split_sizes:
        if item.get("config") == config and item.get("split") == split:
            return int(item["num_rows"])
    raise ValueError(f"datasets-server has no size for {repo_id}:{config}:{split}")


def fetch_swe_zero_sample(
    *,
    repo_id: str,
    config: str,
    split: str,
    sample_size: int,
    window_count: int,
    request_workers: int,
) -> tuple[list[SampleTask], list[tuple[int, int]], int]:
    """Fetch a systematic sample through bounded datasets-server requests."""

    total_rows = dataset_num_rows(repo_id, config, split)
    windows = sample_windows(total_rows, sample_size, window_count)
    if request_workers <= 0:
        raise ValueError("request_workers must be positive")
    with ThreadPoolExecutor(max_workers=min(request_workers, len(windows))) as pool:
        batches = list(
            pool.map(
                lambda window: _fetch_swe_zero_window(
                    repo_id=repo_id,
                    config=config,
                    split=split,
                    offset=window[0],
                    length=window[1],
                ),
                windows,
            )
        )
    tasks = sorted(
        (task for batch in batches for task in batch), key=lambda task: task.row_index
    )
    if len(tasks) != sum(length for _, length in windows):
        raise ValueError(
            f"requested {sum(length for _, length in windows)} rows, received {len(tasks)}"
        )
    return tasks, windows, total_rows


def _fetch_swe_zero_window(
    *,
    repo_id: str,
    config: str,
    split: str,
    offset: int,
    length: int,
) -> list[SampleTask]:
    payload = _dataset_server_json(
        "rows",
        dataset=repo_id,
        config=config,
        split=split,
        offset=offset,
        length=length,
    )
    tasks: list[SampleTask] = []
    for wrapped_row in payload.get("rows", []):
        row = wrapped_row.get("row")
        if not isinstance(row, dict):
            raise TypeError("datasets-server row is not an object")
        instruction = swe_zero_instruction(row)
        normalized = normalized_instruction(instruction)
        tasks.append(
            SampleTask(
                row_index=int(wrapped_row["row_idx"]),
                instance_id=str(row["instance_id"]),
                source_dataset=str(row["dataset"]),
                repo=str(row["repo"]),
                license=str(row["license"]),
                instruction=instruction,
                normalized_instruction=normalized,
                instruction_sha256=instruction_sha256(normalized),
            )
        )
    return tasks


def active_tasktrove_swe_sources(
    api: HfApi, repo_id: str, revision: str
) -> tuple[list[str], str]:
    """Return active top-level TaskTrove sources whose names contain ``swe``."""

    info = api.dataset_info(repo_id, revision=revision)
    if not info.sha:
        raise ValueError(f"could not resolve revision for {repo_id}@{revision}")
    files = api.list_repo_files(repo_id, repo_type="dataset", revision=info.sha)
    sources = sorted(
        path.split("/", maxsplit=1)[0]
        for path in files
        if path.count("/") == 1
        and path.endswith("/tasks.parquet")
        and "swe" in path.casefold()
    )
    if not sources:
        raise ValueError(f"no active SWE sources found in {repo_id}@{info.sha}")
    return sources, info.sha


def _remote_parquet_path(repo_id: str, revision: str, source: str) -> str:
    return f"hf://datasets/{repo_id}@{revision}/{source}/tasks.parquet"


def tasktrove_path_matches(
    *,
    repo_id: str,
    revision: str,
    sources: Iterable[str],
    sampled_instance_ids: set[str],
    filesystem: HfFileSystem,
) -> list[TaskTroveTask]:
    """Read only remote path columns and retain sampled-ID matches."""

    matches: list[TaskTroveTask] = []
    for source in sources:
        remote_path = _remote_parquet_path(repo_id, revision, source)
        with filesystem.open(remote_path, "rb") as stream:
            parquet = pq.ParquetFile(stream)
            for row_group in range(parquet.num_row_groups):
                for row in parquet.read_row_group(
                    row_group, columns=["path"]
                ).to_pylist():
                    path = str(row["path"])
                    candidate_id = canonical_task_id(path)
                    if candidate_id in sampled_instance_ids:
                        matches.append(
                            TaskTroveTask(
                                source=source,
                                path=path,
                                instance_id=candidate_id,
                            )
                        )
    return matches


def instruction_index(
    *,
    repo_id: str,
    revision: str,
    sources: Iterable[str],
) -> list[TaskTroveTask]:
    """Build ID and instruction evidence for selected TaskTrove sources."""

    tasks: list[TaskTroveTask] = []
    for source in sources:
        parquet_path = hf_hub_download(
            repo_id,
            f"{source}/tasks.parquet",
            repo_type="dataset",
            revision=revision,
        )
        parquet = pq.ParquetFile(parquet_path)
        for batch in parquet.iter_batches(columns=["path", "task_binary"]):
            for row in batch.to_pylist():
                instruction = archive_instruction(row["task_binary"])
                problem = normalized_instruction(
                    tasktrove_problem_statement(instruction)
                )
                instance_match = DATASET_INSTANCE_RE.search(instruction)
                tasks.append(
                    TaskTroveTask(
                        source=source,
                        path=str(row["path"]),
                        instance_id=(
                            canonical_task_id(instance_match.group(1))
                            if instance_match
                            else None
                        ),
                        normalized_instruction=problem,
                        instruction_sha256=instruction_sha256(problem),
                    )
                )
    return tasks


def match_tasks(
    samples: list[SampleTask],
    candidates: list[TaskTroveTask],
    *,
    near_threshold: float,
) -> list[MatchResult]:
    """Match by source ID, then exact instruction hash, then fuzzy instruction."""

    by_id: dict[str, list[TaskTroveTask]] = defaultdict(list)
    by_hash: dict[str, list[TaskTroveTask]] = defaultdict(list)
    instruction_choices: dict[int, str] = {}
    instruction_candidates: dict[int, TaskTroveTask] = {}
    for index, candidate in enumerate(candidates):
        if candidate.instance_id:
            by_id[canonical_task_id(candidate.instance_id)].append(candidate)
        if candidate.instruction_sha256:
            by_hash[candidate.instruction_sha256].append(candidate)
        if candidate.normalized_instruction:
            instruction_choices[index] = candidate.normalized_instruction
            instruction_candidates[index] = candidate

    results: list[MatchResult] = []
    for sample in samples:
        id_matches = by_id.get(canonical_task_id(sample.instance_id), [])
        if id_matches:
            candidate = id_matches[0]
            same_hash = candidate.instruction_sha256 == sample.instruction_sha256
            results.append(
                _match_result(
                    sample,
                    candidate,
                    match_type="id+instruction_hash" if same_hash else "id",
                    similarity=100.0 if same_hash else None,
                )
            )
            continue

        hash_matches = by_hash.get(sample.instruction_sha256, [])
        if hash_matches:
            results.append(
                _match_result(
                    sample,
                    hash_matches[0],
                    match_type="instruction_hash",
                    similarity=100.0,
                )
            )
            continue

        fuzzy_match = process.extractOne(
            sample.normalized_instruction,
            instruction_choices,
            scorer=fuzz.ratio,
            score_cutoff=near_threshold,
        )
        if fuzzy_match:
            _, score, candidate_index = fuzzy_match
            results.append(
                _match_result(
                    sample,
                    instruction_candidates[candidate_index],
                    match_type="near_instruction",
                    similarity=round(float(score), 2),
                )
            )
            continue

        results.append(
            _match_result(
                sample,
                None,
                match_type="none",
                similarity=None,
            )
        )
    return results


def _match_result(
    sample: SampleTask,
    candidate: TaskTroveTask | None,
    *,
    match_type: str,
    similarity: float | None,
) -> MatchResult:
    return MatchResult(
        row_index=sample.row_index,
        instance_id=sample.instance_id,
        source_dataset=sample.source_dataset,
        repo=sample.repo,
        license=sample.license,
        match_type=match_type,
        tasktrove_source=candidate.source if candidate else None,
        tasktrove_path=candidate.path if candidate else None,
        instruction_similarity=similarity,
        instruction_sha256=sample.instruction_sha256,
    )


def report_payload(
    *,
    results: list[MatchResult],
    swe_zero_repo: str,
    swe_zero_revision: str,
    swe_zero_total_rows: int,
    windows: list[tuple[int, int]],
    tasktrove_repo: str,
    tasktrove_revision: str,
    tasktrove_sources: list[str],
    instruction_sources: list[str],
    near_threshold: float,
) -> dict[str, Any]:
    """Build a JSON-serializable report with source-level counts."""

    matched_types = {
        "id",
        "id+instruction_hash",
        "instruction_hash",
        "near_instruction",
    }
    by_source: dict[str, dict[str, int]] = {}
    grouped: dict[str, list[MatchResult]] = defaultdict(list)
    for result in results:
        grouped[result.source_dataset].append(result)
    for source, source_results in sorted(grouped.items()):
        type_counts = Counter(result.match_type for result in source_results)
        by_source[source] = {
            "sampled": len(source_results),
            "matched": sum(type_counts[kind] for kind in matched_types),
            "unmatched": type_counts["none"],
            **dict(sorted(type_counts.items())),
        }
    type_counts = Counter(result.match_type for result in results)
    return {
        "metadata": {
            "swe_zero_repo": swe_zero_repo,
            "swe_zero_revision": swe_zero_revision,
            "swe_zero_total_rows": swe_zero_total_rows,
            "sample_windows": [
                {"offset": offset, "length": length} for offset, length in windows
            ],
            "tasktrove_repo": tasktrove_repo,
            "tasktrove_revision": tasktrove_revision,
            "tasktrove_sources": tasktrove_sources,
            "instruction_indexed_sources": instruction_sources,
            "near_instruction_threshold": near_threshold,
        },
        "summary": {
            "sampled": len(results),
            "matched": sum(type_counts[kind] for kind in matched_types),
            "unmatched": type_counts["none"],
            "match_types": dict(sorted(type_counts.items())),
            "by_swe_zero_source": by_source,
        },
        "tasks": [asdict(result) for result in results],
    }


def markdown_report(payload: dict[str, Any]) -> str:
    """Render the compact human-readable part of an overlap report."""

    metadata = payload["metadata"]
    summary = payload["summary"]
    lines = [
        "# SWE-Zero / TaskTrove overlap sample",
        "",
        f"- SWE-Zero: `{metadata['swe_zero_repo']}@{metadata['swe_zero_revision']}`",
        f"- TaskTrove: `{metadata['tasktrove_repo']}@{metadata['tasktrove_revision']}`",
        f"- Sampled: {summary['sampled']} / {metadata['swe_zero_total_rows']}",
        f"- Matched: {summary['matched']}",
        f"- Unmatched: {summary['unmatched']}",
        "",
        "## By SWE-Zero source",
        "",
        "| Source | Sampled | Matched | Unmatched |",
        "| --- | ---: | ---: | ---: |",
    ]
    for source, counts in summary["by_swe_zero_source"].items():
        lines.append(
            f"| {source} | {counts['sampled']} | {counts['matched']} | {counts['unmatched']} |"
        )
    lines.extend(
        [
            "",
            "## Matches",
            "",
            "| SWE-Zero instance | Evidence | TaskTrove source | TaskTrove path | Similarity |",
            "| --- | --- | --- | --- | ---: |",
        ]
    )
    for task in payload["tasks"]:
        if task["match_type"] == "none":
            continue
        similarity = (
            f"{task['instruction_similarity']:.2f}"
            if task["instruction_similarity"] is not None
            else ""
        )
        lines.append(
            f"| {task['instance_id']} | {task['match_type']} | "
            f"{task['tasktrove_source']} | {task['tasktrove_path']} | {similarity} |"
        )
    lines.extend(
        [
            "",
            "## Coverage note",
            "",
            (
                "All active TaskTrove sources with `swe` in their top-level name were "
                "checked by their lightweight Parquet `path` columns. Instruction hashes "
                "and fuzzy matching cover only the sources listed in "
                "`instruction_indexed_sources`; the default is the active SWE-Gym source "
                "because SWE-Zero directly cites SWE-Gym-Raw."
            ),
            "",
        ]
    )
    return "\n".join(lines)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Estimate SWE-Zero overlap with active TaskTrove SWE tasks"
    )
    parser.add_argument("--sample-size", type=int, default=100)
    parser.add_argument("--sample-windows", type=int, default=20)
    parser.add_argument("--request-workers", type=int, default=8)
    parser.add_argument("--near-threshold", type=float, default=92.0)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--swe-zero-repo", default=SWE_ZERO_REPO)
    parser.add_argument("--swe-zero-config", default="default")
    parser.add_argument("--swe-zero-split", default="train")
    parser.add_argument("--swe-zero-revision", default="main")
    parser.add_argument("--tasktrove-repo", default=TASKTROVE_REPO)
    parser.add_argument("--tasktrove-revision", default="main")
    parser.add_argument(
        "--instruction-source",
        action="append",
        default=None,
        help=(
            "TaskTrove source whose task binaries should be instruction-indexed; "
            "repeatable. Defaults to active sources containing 'swegym'."
        ),
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
    logging.getLogger("httpx").setLevel(logging.WARNING)
    api = HfApi()
    swe_zero_info = api.dataset_info(
        args.swe_zero_repo, revision=args.swe_zero_revision
    )
    samples, windows, total_rows = fetch_swe_zero_sample(
        repo_id=args.swe_zero_repo,
        config=args.swe_zero_config,
        split=args.swe_zero_split,
        sample_size=args.sample_size,
        window_count=args.sample_windows,
        request_workers=args.request_workers,
    )
    sources, tasktrove_revision = active_tasktrove_swe_sources(
        api, args.tasktrove_repo, args.tasktrove_revision
    )
    instruction_sources = (
        sorted(set(args.instruction_source))
        if args.instruction_source
        else [source for source in sources if "swegym" in source.casefold()]
    )
    unknown_sources = sorted(set(instruction_sources) - set(sources))
    if unknown_sources:
        raise ValueError(
            f"instruction sources are not active TaskTrove SWE sources: {unknown_sources}"
        )

    sample_ids = {canonical_task_id(task.instance_id) for task in samples}
    LOGGER.info("Reading TaskTrove path columns from %d SWE sources", len(sources))
    path_candidates = tasktrove_path_matches(
        repo_id=args.tasktrove_repo,
        revision=tasktrove_revision,
        sources=sources,
        sampled_instance_ids=sample_ids,
        filesystem=HfFileSystem(),
    )
    LOGGER.info(
        "Instruction-indexing %d TaskTrove source(s): %s",
        len(instruction_sources),
        ", ".join(instruction_sources),
    )
    instruction_candidates = instruction_index(
        repo_id=args.tasktrove_repo,
        revision=tasktrove_revision,
        sources=instruction_sources,
    )
    results = match_tasks(
        samples,
        path_candidates + instruction_candidates,
        near_threshold=args.near_threshold,
    )
    payload = report_payload(
        results=results,
        swe_zero_repo=args.swe_zero_repo,
        swe_zero_revision=swe_zero_info.sha or args.swe_zero_revision,
        swe_zero_total_rows=total_rows,
        windows=windows,
        tasktrove_repo=args.tasktrove_repo,
        tasktrove_revision=tasktrove_revision,
        tasktrove_sources=sources,
        instruction_sources=instruction_sources,
        near_threshold=args.near_threshold,
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    json_path = args.output_dir / "overlap.json"
    markdown_path = args.output_dir / "overlap.md"
    json_path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    markdown_path.write_text(markdown_report(payload), encoding="utf-8")
    print(
        json.dumps(
            {
                "json_report": str(json_path),
                "markdown_report": str(markdown_path),
                **payload["summary"],
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
