from __future__ import annotations

from scripts.analysis.report_litecoder_terminal_overlap import (
    RlTask,
    SftTask,
    byte_windows,
    extract_sft_instruction,
    match_sft_to_rl,
    normalized_text,
    text_hash,
)


def test_extract_sft_instruction_removes_agent_wrapper_and_screen() -> None:
    prompt = """Agent protocol text.

Task Description:
## Task Title: Build a tiny CLI

Create `hello.txt`.

Current terminal state:
Current Terminal Screen:
root@host:/app#
"""

    assert extract_sft_instruction(prompt) == (
        "## Task Title: Build a tiny CLI\n\nCreate `hello.txt`."
    )

    bare_prompt = "## Task Title: Build a tiny CLI\n\nCreate `hello.txt`.\n"
    assert extract_sft_instruction(bare_prompt) == bare_prompt.strip()


def test_byte_windows_cover_beginning_and_end() -> None:
    windows = byte_windows(
        total_bytes=1_000, sample_size=8, records_per_window=2, window_bytes=100
    )

    assert windows == [(0, 100), (300, 100), (600, 100), (900, 100)]


def test_overlap_prefers_exact_instruction_then_exact_title() -> None:
    rl_tasks = [
        RlTask(
            task_id="hello",
            name="LiteCoder/hello",
            title="Build a tiny CLI",
            instruction="## Build a tiny CLI\n\nCreate `hello.txt`.",
            instruction_hash=text_hash("## Build a tiny CLI\n\nCreate `hello.txt`."),
            normalized_title=normalized_text("Build a tiny CLI"),
        ),
        RlTask(
            task_id="other",
            name="LiteCoder/other",
            title="Analyze a database",
            instruction="Inspect a SQLite database.",
            instruction_hash=text_hash("Inspect a SQLite database."),
            normalized_title=normalized_text("Analyze a database"),
        ),
    ]
    sft_tasks = [
        SftTask(
            row_id=1,
            instruction="## Build a tiny CLI\n\nCreate `hello.txt`.",
            instruction_hash=text_hash("## Build a tiny CLI\n\nCreate `hello.txt`."),
            title="Build a tiny CLI",
            normalized_title=normalized_text("Build a tiny CLI"),
            byte_offset=10,
        ),
        SftTask(
            row_id=2,
            instruction="A differently worded database task.",
            instruction_hash=text_hash("A differently worded database task."),
            title="Analyze a database",
            normalized_title=normalized_text("Analyze a database"),
            byte_offset=20,
        ),
    ]

    matches = match_sft_to_rl(sft_tasks, rl_tasks, near_title_threshold=95.0)

    assert matches[0].match_type == "exact_instruction"
    assert matches[0].rl_task_id == "hello"
    assert matches[0].instruction_similarity == 100.0
    assert matches[1].match_type == "exact_title"
    assert matches[1].rl_task_id == "other"
