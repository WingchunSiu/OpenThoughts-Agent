"""Audit LiteCoder Terminal RL tasks and sample SFT-to-RL overlap.

The SFT source is a 1.2 GB JSON array.  This script uses deterministic HTTP
byte ranges so an overlap audit does not need to download every trajectory.
It compares the task description in the first human turn with all 602 public
RL task instructions.

Example:
    python -m scripts.analysis.report_litecoder_terminal_overlap \
        --rl-root /path/to/LiteCoder-Terminal-RL-preview \
        --sft-sample-size 128 \
        --output-dir outputs/litecoder-terminal-audit
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import logging
import re
import tarfile
import tomllib
import unicodedata
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass
from pathlib import Path, PurePosixPath
from typing import Any
from urllib.request import Request, urlopen

import pyarrow.parquet as pq
from harbor.models.task.task import Task
from huggingface_hub import HfApi, hf_hub_download
from rapidfuzz import fuzz, process

from scripts.harbor.tasks_parquet_converter import find_tasks, to_parquet

RL_REPO = "Lite-Coder/LiteCoder-Terminal-RL-preview"
RL_REVISION = "6fe7e994ff12d678de9b803da5c9907c8394a89c"
SFT_REPO = "Lite-Coder/LiteCoder-Terminal-SFT"
SFT_REVISION = "6acdbbdb29979e4b8ea717b12accc8214606d087"
SFT_FILENAME = "litecoder-sft.json"
TASKTROVE_REPO = "open-thoughts/TaskTrove"
TASK_DESCRIPTION_MARKER = "Task Description:"
TERMINAL_STATE_MARKER = "Current terminal state:"
JSON_RECORD_MARKER = b'{\n    "id":'
LFS_POINTER_MARKER = b"version https://git-lfs.github.com/spec/v1"
REQUIRED_TASK_FILES = (
    "instruction.md",
    "task.toml",
    "environment/Dockerfile",
    "solution/solve.sh",
    "tests/test.sh",
)
PACKAGE_RELATIVE_PATH = Path(
    "tasktrove/Lite-Coder__LiteCoder-Terminal-RL-preview/tasks.parquet"
)
NETWORK_INSTALL_RE = re.compile(
    r"(?:curl|wget|apt-get|apt\s+install|pip\s+install|uv\s+add|npm\s+install)",
    re.IGNORECASE,
)
TITLE_RE = re.compile(r"(?m)^#{1,3}\s+(?:Task(?:\s+Title)?:\s*)?(.+?)\s*$")
LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class RlTask:
    task_id: str
    name: str
    title: str
    instruction: str
    instruction_hash: str
    normalized_title: str


@dataclass(frozen=True)
class SftTask:
    row_id: int
    instruction: str
    instruction_hash: str
    title: str
    normalized_title: str
    byte_offset: int


@dataclass(frozen=True)
class OverlapMatch:
    sft_id: int
    sft_title: str
    rl_task_id: str
    rl_title: str
    match_type: str
    title_similarity: float
    instruction_similarity: float


def normalized_text(text: str) -> str:
    """Return stable text for hashes and similarity matching."""

    text = unicodedata.normalize("NFKC", text).casefold()
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def text_hash(text: str) -> str:
    return hashlib.sha256(normalized_text(text).encode()).hexdigest()


def file_starts_with(path: Path, marker: bytes) -> bool:
    with path.open("rb") as stream:
        return stream.read(len(marker)) == marker


def instruction_title(instruction: str) -> str:
    match = TITLE_RE.search(instruction)
    if match:
        return match.group(1).strip()
    first_line = next(
        (line.strip() for line in instruction.splitlines() if line.strip()), ""
    )
    return first_line[:200]


def extract_sft_instruction(first_human_turn: str) -> str:
    """Strip the terminal-agent wrapper and current-screen suffix."""

    instruction = first_human_turn
    if TASK_DESCRIPTION_MARKER in instruction:
        instruction = instruction.split(TASK_DESCRIPTION_MARKER, maxsplit=1)[1]
    if TERMINAL_STATE_MARKER in instruction:
        instruction = instruction.split(TERMINAL_STATE_MARKER, maxsplit=1)[0]
    instruction = instruction.strip()
    if not instruction:
        raise ValueError("SFT task description is empty")
    return instruction


def load_rl_tasks(repo_id: str = RL_REPO, revision: str = RL_REVISION) -> list[RlTask]:
    """Load the compact 602-row metadata Parquet, not task binaries."""

    parquet_path = hf_hub_download(
        repo_id,
        "data/train.parquet",
        repo_type="dataset",
        revision=revision,
    )
    rows = pq.read_table(
        parquet_path, columns=["task_id", "name", "title", "instruction"]
    ).to_pylist()
    return [
        RlTask(
            task_id=str(row["task_id"]),
            name=str(row["name"]),
            title=str(row["title"]),
            instruction=str(row["instruction"]),
            instruction_hash=text_hash(str(row["instruction"])),
            normalized_title=normalized_text(str(row["title"])),
        )
        for row in rows
    ]


def _content_length(url: str) -> int:
    request = Request(url, headers={"Range": "bytes=0-0"})
    with urlopen(request, timeout=120) as response:
        content_range = response.headers.get("Content-Range")
        if not content_range:
            raise ValueError("source did not honor byte-range request")
        return int(content_range.rsplit("/", maxsplit=1)[1])


def byte_windows(
    total_bytes: int, sample_size: int, records_per_window: int, window_bytes: int
) -> list[tuple[int, int]]:
    """Return evenly spaced byte ranges for a systematic JSON sample."""

    if min(total_bytes, sample_size, records_per_window, window_bytes) <= 0:
        raise ValueError("byte sampling arguments must be positive")
    window_count = (sample_size + records_per_window - 1) // records_per_window
    max_offset = max(0, total_bytes - window_bytes)
    if window_count == 1:
        return [(0, min(window_bytes, total_bytes))]
    return [
        (
            round(index * max_offset / (window_count - 1)),
            min(
                window_bytes,
                total_bytes - round(index * max_offset / (window_count - 1)),
            ),
        )
        for index in range(window_count)
    ]


def _range_records(
    url: str, offset: int, length: int, records_per_window: int
) -> list[SftTask]:
    request = Request(url, headers={"Range": f"bytes={offset}-{offset + length - 1}"})
    with urlopen(request, timeout=120) as response:
        payload = response.read()
    text = payload.decode("utf-8", errors="ignore")
    marker = JSON_RECORD_MARKER.decode()
    decoder = json.JSONDecoder()
    tasks: list[SftTask] = []
    search_at = 0
    while len(tasks) < records_per_window:
        start = text.find(marker, search_at)
        if start < 0:
            break
        try:
            row, end = decoder.raw_decode(text, start)
        except json.JSONDecodeError:
            search_at = start + len(marker)
            continue
        conversations = row.get("conversations")
        if not isinstance(conversations, list) or not conversations:
            raise ValueError(f"SFT row {row.get('id')} has no conversations")
        first_turn = conversations[0]
        if first_turn.get("from") != "human" or not isinstance(
            first_turn.get("value"), str
        ):
            raise ValueError(f"SFT row {row.get('id')} has no first human turn")
        try:
            instruction = extract_sft_instruction(first_turn["value"])
        except ValueError as error:
            excerpt = first_turn["value"][-500:].replace("\n", "\\n")
            raise ValueError(
                f"SFT row {row.get('id')} cannot yield a task description: {excerpt}"
            ) from error
        title = instruction_title(instruction)
        tasks.append(
            SftTask(
                row_id=int(row["id"]),
                instruction=instruction,
                instruction_hash=text_hash(instruction),
                title=title,
                normalized_title=normalized_text(title),
                byte_offset=offset + len(text[:start].encode("utf-8")),
            )
        )
        search_at = end
    return tasks


def fetch_sft_sample(
    *,
    repo_id: str,
    revision: str,
    sample_size: int,
    records_per_window: int,
    window_bytes: int,
    workers: int,
) -> tuple[list[SftTask], int, list[tuple[int, int]]]:
    """Fetch a deterministic byte-stratified sample of SFT task prompts."""

    url = f"https://huggingface.co/datasets/{repo_id}/resolve/{revision}/{SFT_FILENAME}"
    total_bytes = _content_length(url)
    windows = byte_windows(total_bytes, sample_size, records_per_window, window_bytes)
    with ThreadPoolExecutor(max_workers=min(workers, len(windows))) as pool:
        batches = list(
            pool.map(
                lambda window: _range_records(
                    url, window[0], window[1], records_per_window
                ),
                windows,
            )
        )
    by_id = {task.row_id: task for batch in batches for task in batch}
    tasks = sorted(by_id.values(), key=lambda task: task.row_id)[:sample_size]
    minimum_sample_size = max(1, int(sample_size * 0.8))
    if len(tasks) < minimum_sample_size:
        raise ValueError(
            f"byte ranges yielded {len(tasks)} unique rows, expected at least "
            f"{minimum_sample_size}"
        )
    return tasks, total_bytes, windows


def match_sft_to_rl(
    sft_tasks: list[SftTask], rl_tasks: list[RlTask], near_title_threshold: float
) -> list[OverlapMatch]:
    """Match exact instructions/titles first, then report best fuzzy candidate."""

    by_instruction = {task.instruction_hash: task for task in rl_tasks}
    by_title = {task.normalized_title: task for task in rl_tasks}
    title_choices = {
        index: task.normalized_title for index, task in enumerate(rl_tasks)
    }
    matches: list[OverlapMatch] = []
    for sft_task in sft_tasks:
        rl_task = by_instruction.get(sft_task.instruction_hash)
        match_type = "exact_instruction" if rl_task else ""
        if rl_task is None:
            rl_task = by_title.get(sft_task.normalized_title)
            match_type = "exact_title" if rl_task else ""
        title_match = process.extractOne(
            sft_task.normalized_title, title_choices, scorer=fuzz.WRatio
        )
        assert title_match is not None
        _, title_score, title_index = title_match
        if rl_task is None:
            rl_task = rl_tasks[int(title_index)]
            match_type = "near_title" if title_score >= near_title_threshold else "none"
        instruction_score = fuzz.ratio(
            normalized_text(sft_task.instruction),
            normalized_text(rl_task.instruction),
        )
        matches.append(
            OverlapMatch(
                sft_id=sft_task.row_id,
                sft_title=sft_task.title,
                rl_task_id=rl_task.task_id,
                rl_title=rl_task.title,
                match_type=match_type,
                title_similarity=round(float(title_score), 2),
                instruction_similarity=round(float(instruction_score), 2),
            )
        )
    return matches


def audit_rl_root(root: Path, rl_tasks: list[RlTask]) -> dict[str, Any]:
    """Run Harbor loading and static file checks over a local RL checkout."""

    task_dirs = sorted(path.parent for path in root.glob("*/instruction.md"))
    required_missing: Counter[str] = Counter()
    load_errors: list[dict[str, str]] = []
    schema_versions: Counter[str] = Counter()
    task_names: list[str] = []
    lfs_pointers: list[str] = []
    verifier_network_installs = 0
    docker_network_installs = 0
    docker_images: Counter[str] = Counter()
    floating_latest_images = 0
    instruction_hashes: Counter[str] = Counter()

    for task_dir in task_dirs:
        for relative_path in REQUIRED_TASK_FILES:
            path = task_dir / relative_path
            if not path.is_file() or path.stat().st_size == 0:
                required_missing[relative_path] += 1
        for path in task_dir.rglob("*"):
            if path.is_file() and file_starts_with(path, LFS_POINTER_MARKER):
                lfs_pointers.append(path.relative_to(root).as_posix())
        try:
            task = Task(task_dir)
            schema_versions[str(task.config.schema_version)] += 1
            task_names.append(task.config.task.name)
            instruction_hashes[text_hash(task.instruction)] += 1
        except (OSError, ValueError) as error:
            load_errors.append(
                {"task_id": task_dir.name, "error": f"{type(error).__name__}: {error}"}
            )
        task_toml = tomllib.loads((task_dir / "task.toml").read_text())
        image = task_toml.get("environment", {}).get("docker_image")
        if image:
            docker_images[str(image)] += 1
            floating_latest_images += str(image).endswith(":latest")
        test_sh = (task_dir / "tests/test.sh").read_text(errors="replace")
        dockerfile = (task_dir / "environment/Dockerfile").read_text(errors="replace")
        verifier_network_installs += bool(NETWORK_INSTALL_RE.search(test_sh))
        docker_network_installs += bool(NETWORK_INSTALL_RE.search(dockerfile))

    manifest_ids = {task.task_id for task in rl_tasks}
    local_ids = {path.name for path in task_dirs}
    duplicate_instructions = sum(
        count - 1 for count in instruction_hashes.values() if count > 1
    )
    return {
        "task_directories": len(task_dirs),
        "manifest_tasks": len(rl_tasks),
        "manifest_missing_locally": sorted(manifest_ids - local_ids),
        "local_missing_from_manifest": sorted(local_ids - manifest_ids),
        "required_missing": dict(required_missing),
        "harbor_load_errors": load_errors,
        "schema_versions": dict(schema_versions),
        "unique_task_names": len(set(task_names)),
        "duplicate_instruction_rows": duplicate_instructions,
        "unresolved_lfs_pointers": lfs_pointers,
        "verifier_scripts_with_network_installs": verifier_network_installs,
        "dockerfiles_with_network_installs": docker_network_installs,
        "unique_declared_docker_images": len(docker_images),
        "floating_latest_images": floating_latest_images,
        "declared_docker_images": dict(docker_images),
    }


def audit_tasktrove_package(parquet_path: Path) -> dict[str, Any]:
    """Validate every packaged task archive and return import metadata."""

    parquet = pq.ParquetFile(parquet_path)
    paths: set[str] = set()
    rows = 0
    archive_errors: list[dict[str, Any]] = []
    for batch in parquet.iter_batches(columns=["path", "task_binary"]):
        for row in batch.to_pylist():
            rows += 1
            path = str(row["path"])
            paths.add(path)
            try:
                with tarfile.open(
                    fileobj=io.BytesIO(row["task_binary"]), mode="r:gz"
                ) as archive:
                    missing = sorted(set(REQUIRED_TASK_FILES) - set(archive.getnames()))
                    if missing:
                        archive_errors.append({"path": path, "missing": missing})
            except tarfile.TarError as error:
                archive_errors.append({"path": path, "error": str(error)})
    digest = hashlib.sha256()
    with parquet_path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return {
        "path": str(parquet_path.resolve()),
        "bytes": parquet_path.stat().st_size,
        "sha256": digest.hexdigest(),
        "rows": rows,
        "unique_paths": len(paths),
        "archive_errors": archive_errors,
    }


def tasktrove_catalog(
    *, repo_id: str, revision: str
) -> tuple[dict[str, Any], dict[str, int]]:
    """Resolve the current catalog and Parquet sizes without task binaries."""

    api = HfApi()
    info = api.dataset_info(repo_id, revision=revision, files_metadata=True)
    if not info.sha:
        raise ValueError(f"could not resolve {repo_id}@{revision}")
    sizes = {
        sibling.rfilename.split("/", maxsplit=1)[0]: int(sibling.size or 0)
        for sibling in info.siblings
        if sibling.rfilename.count("/") == 1
        and sibling.rfilename.endswith("/tasks.parquet")
    }
    readme_path = hf_hub_download(
        repo_id, "README.md", repo_type="dataset", revision=info.sha
    )
    readme = Path(readme_path).read_text(errors="replace").casefold()
    return (
        {
            "repo": repo_id,
            "revision": info.sha,
            "root_sources": len(sizes),
            "readme_mentions_litecoder": "litecoder" in readme
            or "lite-coder" in readme,
            "source_name_mentions_litecoder": any(
                "litecoder" in source.casefold() or "lite-coder" in source.casefold()
                for source in sizes
            ),
        },
        sizes,
    )


def scan_tasktrove_paths(
    *,
    repo_id: str,
    revision: str,
    source_sizes: dict[str, int],
    rl_task_ids: set[str],
    max_file_bytes: int,
) -> dict[str, Any]:
    """Scan exact task IDs in bounded-size TaskTrove Parquets."""

    matches: list[dict[str, str]] = []
    scanned_rows = 0
    scanned_sources: list[str] = []
    skipped_sources: list[str] = []
    eligible_sources = [
        source
        for source, size in sorted(source_sizes.items())
        if not max_file_bytes or size <= max_file_bytes
    ]
    for source, size in sorted(source_sizes.items()):
        if max_file_bytes and size > max_file_bytes:
            skipped_sources.append(source)
            continue
        local_path = hf_hub_download(
            repo_id,
            f"{source}/tasks.parquet",
            repo_type="dataset",
            revision=revision,
        )
        parquet = pq.ParquetFile(local_path)
        for batch in parquet.iter_batches(columns=["path"], batch_size=65_536):
            paths = batch.column(0).to_pylist()
            scanned_rows += len(paths)
            for raw_path in paths:
                path = str(raw_path).rstrip("/")
                task_id = PurePosixPath(path).name
                if task_id in rl_task_ids:
                    matches.append({"source": source, "path": path, "task_id": task_id})
        scanned_sources.append(source)
        LOGGER.info(
            "Scanned TaskTrove source %s (%d/%d)",
            source,
            len(scanned_sources),
            len(eligible_sources),
        )
    return {
        "scanned_sources": len(scanned_sources),
        "scanned_rows": scanned_rows,
        "max_source_bytes": max_file_bytes,
        "skipped_sources": skipped_sources,
        "exact_path_matches": matches,
    }


def report_markdown(payload: dict[str, Any]) -> str:
    audit = payload["rl_static_audit"]
    overlap = payload["sft_rl_overlap"]
    tasktrove = payload["tasktrove"]
    counts = Counter(match["match_type"] for match in overlap["matches"])
    candidate_matches = [
        match for match in overlap["matches"] if match["match_type"] != "none"
    ]
    lines = [
        "# LiteCoder Terminal audit",
        "",
        "## Result",
        "",
        f"- RL source: `{payload['sources']['rl_repo']}@{payload['sources']['rl_revision']}`",
        f"- SFT source: `{payload['sources']['sft_repo']}@{payload['sources']['sft_revision']}`",
        f"- RL tasks: {audit['task_directories']} local / {audit['manifest_tasks']} manifest",
        f"- Harbor static loads: {audit['task_directories'] - len(audit['harbor_load_errors'])}/{audit['task_directories']}",
        f"- Missing required files: {sum(audit['required_missing'].values())}",
        f"- Unresolved Git LFS pointers: {len(audit['unresolved_lfs_pointers'])}",
        f"- Duplicate RL instruction rows: {audit['duplicate_instruction_rows']}",
        f"- SFT sample: {overlap['sample_size']} rows from {overlap['total_rows']} total rows",
        f"- SFT/RL exact instruction matches: {counts['exact_instruction']}",
        f"- SFT/RL exact title matches: {counts['exact_title']}",
        f"- SFT/RL near-title candidates: {counts['near_title']}",
        f"- TaskTrove: `{tasktrove['repo']}@{tasktrove['revision']}`",
        f"- TaskTrove directly names LiteCoder: {tasktrove['readme_mentions_litecoder'] or tasktrove['source_name_mentions_litecoder']}",
        f"- TaskTrove exact task-ID/path matches in bounded scan: {len(tasktrove['path_scan']['exact_path_matches'])}",
    ]
    package = payload.get("tasktrove_package")
    if package:
        lines.extend(
            [
                f"- Packaged TaskTrove rows: {package['rows']} ({package['bytes']} bytes)",
                f"- Packaged archive errors: {len(package['archive_errors'])}",
                f"- Packaged Parquet SHA-256: `{package['sha256']}`",
            ]
        )
    lines.extend(
        [
            "",
            "## Import readiness",
            "",
            (
                "The source is already organized as Harbor tasks and can be packaged "
                "into the TaskTrove `path` + `task_binary` Parquet schema without task "
                "reconstruction. Static compatibility is necessary but does not establish "
                "verifier correctness."
            ),
            "",
            (
                f"All {audit['task_directories']} task configs allow internet. "
                f"{audit['verifier_scripts_with_network_installs']} verifier scripts "
                "install or download dependencies at verification time. These are "
                "reproducibility and reward-availability risks to audit before publication."
            ),
            "",
            (
                f"{audit['floating_latest_images']} task configs reference a mutable "
                "`:latest` Docker image tag. Pin or remove those image references before "
                "a stable TaskTrove release."
            ),
            "",
            "Dynamic oracle/no-op/partial-solution gates were not run by this static report.",
            "",
            "## SFT/RL overlap candidates",
            "",
        ]
    )
    if package:
        lines.insert(
            lines.index("## SFT/RL overlap candidates"),
            (
                "The packaged Parquet is a local generated artifact. Recreate it with "
                "`--package-parquet`; this report records its checksum."
            ),
        )
        lines.insert(lines.index("## SFT/RL overlap candidates"), "")
    if not candidate_matches:
        lines.append(
            "No exact or thresholded near-title match was found in the sample."
        )
    else:
        lines.extend(
            [
                "| SFT id | Match | SFT title | RL task | Title score | Instruction score |",
                "|---:|---|---|---|---:|---:|",
            ]
        )
        for match in candidate_matches:
            lines.append(
                f"| {match['sft_id']} | {match['match_type']} | "
                f"{match['sft_title'].replace('|', '\\|')} | "
                f"{match['rl_task_id']} | {match['title_similarity']:.2f} | "
                f"{match['instruction_similarity']:.2f} |"
            )
    lines.extend(
        [
            "",
            "## Limits",
            "",
            (
                "- The SFT result is a deterministic byte-stratified sample, not a full "
                "11,255-row scan."
            ),
            (
                "- The TaskTrove path scan is exact for the listed scanned sources but "
                "skips large Parquets above the configured size cap. Absence of an exact "
                "ID is not proof that no semantically duplicate instruction exists."
            ),
            "- Container execution requires a running Docker daemon.",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rl-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--sft-sample-size", type=int, default=128)
    parser.add_argument("--sft-records-per-window", type=int, default=16)
    parser.add_argument("--sft-window-mb", type=int, default=8)
    parser.add_argument("--request-workers", type=int, default=1)
    parser.add_argument("--near-title-threshold", type=float, default=95.0)
    parser.add_argument("--tasktrove-revision", default="main")
    parser.add_argument("--tasktrove-max-source-mb", type=int, default=2)
    parser.add_argument("--package-parquet", action="store_true")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    LOGGER.info("Loading LiteCoder RL metadata")
    rl_tasks = load_rl_tasks()
    LOGGER.info("Auditing %s", args.rl_root)
    rl_audit = audit_rl_root(args.rl_root, rl_tasks)
    LOGGER.info("Sampling LiteCoder SFT")
    sft_tasks, sft_bytes, windows = fetch_sft_sample(
        repo_id=SFT_REPO,
        revision=SFT_REVISION,
        sample_size=args.sft_sample_size,
        records_per_window=args.sft_records_per_window,
        window_bytes=args.sft_window_mb * 1024 * 1024,
        workers=args.request_workers,
    )
    matches = match_sft_to_rl(sft_tasks, rl_tasks, args.near_title_threshold)

    api = HfApi()
    sft_info = api.dataset_info(SFT_REPO, revision=SFT_REVISION)
    sft_rows = 11_255
    if sft_info.card_data and sft_info.card_data.get("dataset_info"):
        dataset_info = sft_info.card_data["dataset_info"]
        if isinstance(dataset_info, dict):
            sft_rows = int(
                dataset_info.get("splits", {})
                .get("train", {})
                .get("num_examples", sft_rows)
            )

    LOGGER.info("Resolving current TaskTrove catalog")
    tasktrove, source_sizes = tasktrove_catalog(
        repo_id=TASKTROVE_REPO, revision=args.tasktrove_revision
    )
    LOGGER.info("Scanning bounded TaskTrove path columns")
    tasktrove["path_scan"] = scan_tasktrove_paths(
        repo_id=TASKTROVE_REPO,
        revision=tasktrove["revision"],
        source_sizes=source_sizes,
        rl_task_ids={task.task_id for task in rl_tasks},
        max_file_bytes=args.tasktrove_max_source_mb * 1024 * 1024,
    )
    tasktrove["source_sizes"] = source_sizes

    package_path = args.output_dir / PACKAGE_RELATIVE_PATH
    if args.package_parquet:
        package_path.parent.mkdir(parents=True, exist_ok=True)
        task_dirs = find_tasks(args.rl_root, recursive=True)
        to_parquet(args.rl_root, package_path, task_dirs, compression="gz")

    payload = {
        "sources": {
            "rl_repo": RL_REPO,
            "rl_revision": RL_REVISION,
            "sft_repo": SFT_REPO,
            "sft_revision": SFT_REVISION,
        },
        "rl_static_audit": rl_audit,
        "sft_rl_overlap": {
            "sampling_method": "evenly spaced HTTP byte ranges",
            "source_bytes": sft_bytes,
            "total_rows": sft_rows,
            "sample_size": len(sft_tasks),
            "windows": windows,
            "near_title_threshold": args.near_title_threshold,
            "matches": [asdict(match) for match in matches],
        },
        "tasktrove": tasktrove,
    }
    if package_path.is_file():
        package_audit = audit_tasktrove_package(package_path)
        package_audit["path"] = PACKAGE_RELATIVE_PATH.as_posix()
        payload["tasktrove_package"] = package_audit
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "report.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    (args.output_dir / "README.md").write_text(
        report_markdown(payload), encoding="utf-8"
    )


if __name__ == "__main__":
    main()
