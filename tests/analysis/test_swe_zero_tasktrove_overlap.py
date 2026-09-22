from __future__ import annotations

import io
import tarfile

from scripts.analysis.report_swe_zero_tasktrove_overlap import (
    SampleTask,
    TaskTroveTask,
    archive_instruction,
    instruction_sha256,
    match_tasks,
    normalized_instruction,
    sample_windows,
    swe_zero_instruction,
    tasktrove_problem_statement,
)


def test_instruction_extractors_produce_same_problem_hash() -> None:
    problem = "Fix the parser.\n\nIt should accept `--dry-run`."
    swe_zero_row = {
        "trajectory": [
            {
                "role": "user",
                "content": (
                    "<uploaded_files>/workspace/repo</uploaded_files>\n"
                    f"<issue_description>\n{problem}\n</issue_description>"
                ),
            }
        ]
    }
    tasktrove_instruction = (
        "# Bug Fix Task\n\n## Environment Setup\nclone repo\n\n"
        f"## Problem Statement\n{problem}\n\n"
        "## Tests that must keep passing\n- test_old\n"
    )

    swe_problem = normalized_instruction(swe_zero_instruction(swe_zero_row))
    tasktrove_problem = normalized_instruction(
        tasktrove_problem_statement(tasktrove_instruction)
    )

    assert swe_problem == tasktrove_problem
    assert instruction_sha256(swe_problem) == instruction_sha256(tasktrove_problem)


def test_archive_instruction_reads_nested_instruction_without_extracting() -> None:
    buffer = io.BytesIO()
    payload = b"Do the thing.\n"
    with tarfile.open(fileobj=buffer, mode="w:gz") as archive:
        info = tarfile.TarInfo("task/instruction.md")
        info.size = len(payload)
        archive.addfile(info, io.BytesIO(payload))

    assert archive_instruction(buffer.getvalue()) == "Do the thing.\n"


def test_sample_windows_are_bounded_and_evenly_spaced() -> None:
    assert sample_windows(100, 10, 3) == [(0, 4), (48, 3), (97, 3)]
    assert sample_windows(3, 10, 4) == [(0, 1), (1, 1), (2, 1)]


def test_match_tasks_prefers_id_then_hash_then_near_instruction() -> None:
    exact_text = normalized_instruction("Fix exact behavior")
    hash_text = normalized_instruction("Add deterministic output")
    near_text = normalized_instruction("Handle malformed configuration files safely")
    samples = [
        _sample("owner__repo-1", exact_text),
        _sample("owner__repo-2", hash_text),
        _sample(
            "owner__repo-3",
            normalized_instruction("Handle malformed configuration file safely"),
        ),
        _sample("owner__repo-4", normalized_instruction("Unrelated request")),
    ]
    candidates = [
        TaskTroveTask(
            source="swe",
            path="owner__repo-1",
            instance_id="owner__repo-1",
            normalized_instruction=exact_text,
            instruction_sha256=instruction_sha256(exact_text),
        ),
        TaskTroveTask(
            source="swegym",
            path="swegym-1",
            normalized_instruction=hash_text,
            instruction_sha256=instruction_sha256(hash_text),
        ),
        TaskTroveTask(
            source="swegym",
            path="swegym-2",
            normalized_instruction=near_text,
            instruction_sha256=instruction_sha256(near_text),
        ),
    ]

    results = match_tasks(samples, candidates, near_threshold=90.0)

    assert [result.match_type for result in results] == [
        "id+instruction_hash",
        "instruction_hash",
        "near_instruction",
        "none",
    ]


def _sample(instance_id: str, instruction: str) -> SampleTask:
    return SampleTask(
        row_index=0,
        instance_id=instance_id,
        source_dataset="source",
        repo="owner/repo",
        license="MIT",
        instruction=instruction,
        normalized_instruction=instruction,
        instruction_sha256=instruction_sha256(instruction),
    )
