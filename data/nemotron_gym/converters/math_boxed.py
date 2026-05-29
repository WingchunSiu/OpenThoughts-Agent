"""Convert math datasets (OpenMathReasoning, advanced_calculations, stack_overflow)."""

from __future__ import annotations

from ..adapter import (
    HarborTask,
    STANDARD_TEST_SH,
    render_dockerfile,
    render_metadata,
    sanitize_text,
    task_id_for,
)
from ..verifiers import MATH_BOXED_VERIFIER_PY, NUMERIC_COMPARE_VERIFIER_PY
from . import register
from ._common import extract_prompt


_BASE_IMAGE = "python:3.11-slim-bookworm"
_INSTRUCTION_HEADER = (
    "You are a careful mathematical reasoner. Read the problem below and "
    "show your work, then write your final answer at the path `/app/answer.txt`.\n\n"
    "Format requirement: the answer file should contain `\\boxed{<answer>}` on its\n"
    "last meaningful line. The verifier extracts the last `\\boxed{...}` it finds.\n\n"
    "---\n\n"
)


def _convert(row: dict, source_dataset: str, *, row_idx: int) -> HarborTask | None:
    prompt = extract_prompt(row)
    expected = row.get("expected_answer") or row.get("answer")
    if not isinstance(expected, str) or not expected.strip():
        return None
    expected = sanitize_text(expected, field_name="expected_answer", max_len=8 * 1024)
    instruction_md = _INSTRUCTION_HEADER + prompt
    dockerfile = render_dockerfile(
        base=_BASE_IMAGE,
        pip_packages=("sympy==1.13.3", "antlr4-python3-runtime==4.11"),
    )
    task_id = task_id_for(source_dataset.split("/")[-1], prompt + "||" + expected)
    return HarborTask(
        task_id=task_id,
        instruction_md=instruction_md,
        dockerfile=dockerfile,
        test_sh=STANDARD_TEST_SH,
        verifier_py=MATH_BOXED_VERIFIER_PY,
        verifier_data={"expected_answer": expected},
        metadata=render_metadata(
            source_dataset=source_dataset,
            source_uuid=row.get("uuid") if isinstance(row.get("uuid"), str) else None,
            extra={"row_index": row_idx, "family": "math_boxed"},
        ),
    )


@register("nvidia/Nemotron-RL-math-OpenMathReasoning")
def convert_openmathreasoning(row: dict, row_idx: int) -> HarborTask | None:
    return _convert(row, "nvidia/Nemotron-RL-math-OpenMathReasoning", row_idx=row_idx)


_ADVANCED_INSTRUCTION_HEADER = (
    "You are solving a calculation task. Read the problem below and write the "
    "final numeric answer (a single number) to `/app/answer.txt`.\n\n"
    "IMPORTANT: If the problem asks for **multiple** values (e.g. "
    "\"Get me the values for X, Y, and Z\"), the verifier only grades the "
    "LAST expression. Compute every expression if you wish, but write ONLY "
    "the value of the LAST one to `/app/answer.txt`. The verifier extracts "
    "the last numeric token from your answer file and compares it to the "
    "reference value within tolerance.\n\n"
    "---\n\n"
)


@register("nvidia/Nemotron-RL-math-advanced_calculations")
def convert_advanced_calculations(row: dict, row_idx: int) -> HarborTask | None:
    """Different schema from OpenMathReasoning — uses `simplified_values` (list[float]).

    For multi-value prompts (e.g. "Get me the values for X, Y, and Z"), we
    grade only the LAST value. This matches what the agent naturally does:
    the verifier extracts the *last* numeric token from /app/answer.txt, so
    the last `simplified_values` entry is the correct reference. (The
    original v1 of this converter used `sv[0]`, which mismatched the agent's
    answer on ~60% of rows and caused a 2.5% solve rate.)
    """
    prompt = extract_prompt(row)
    sv = row.get("simplified_values")
    if not isinstance(sv, list) or not sv:
        return None
    val = sv[-1]
    if not isinstance(val, (int, float)):
        return None
    val = float(val)
    dockerfile = render_dockerfile(base=_BASE_IMAGE)
    gt = row.get("ground_truth") if isinstance(row.get("ground_truth"), str) else None
    task_id = task_id_for(
        "math-advcalc",
        prompt[:128] + "||" + repr(val) + "|" + str(row_idx),
    )
    return HarborTask(
        task_id=task_id,
        instruction_md=_ADVANCED_INSTRUCTION_HEADER + prompt,
        dockerfile=dockerfile,
        test_sh=STANDARD_TEST_SH,
        verifier_py=NUMERIC_COMPARE_VERIFIER_PY,
        verifier_data={
            "expected_value": val,
            "tolerance_abs": 1e-4,
            "tolerance_rel": 1e-4,
        },
        metadata=render_metadata(
            source_dataset="nvidia/Nemotron-RL-math-advanced_calculations",
            source_uuid=None,
            extra={
                "row_index": row_idx,
                "family": "numeric_compare",
                "ground_truth_expr": gt,
                "breadth": row.get("breadth"),
                "max_depth": row.get("max_depth"),
            },
        ),
    )


@register("nvidia/Nemotron-RL-math-stack_overflow")
def convert_math_stack_overflow(row: dict, row_idx: int) -> HarborTask | None:
    return _convert(row, "nvidia/Nemotron-RL-math-stack_overflow", row_idx=row_idx)
