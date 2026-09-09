from __future__ import annotations

import subprocess

import pytest

from data.tasktrove.build_stack_pytest_v2_repair import (
    PYTEST_TEST_SH,
    PYTEST_TEST_SH_IMAGE,
    drop_reasons,
    extract_requirements,
    patch_task,
    pytest_test_sh,
    score_pytest_run,
    transform_task,
)

TRAP_VERIFIER = """#!/bin/bash
set -e

cleanup() {
    if [ $? -eq 0 ]; then
        echo "1" > /logs/verifier/reward.txt
    else
        echo "0" > /logs/verifier/reward.txt
    fi
}
trap cleanup EXIT

python3 -m venv /app/.venv
source /app/.venv/bin/activate
pip install --quiet pytest
pip install --quiet sklearn yaml PIL __future__ numpy 2>/dev/null || true
pytest /tests/test_solution.py -v --tb=short 2>&1 | tee /logs/verifier/pytest_output.txt
"""

VALID_TEST = """\
def test_add():
    from widget import add

    assert add(1, 2) == 3
"""

SYNTAX_TEST = """\
from {{cookiecutter.package_name}}.app import RedisClient

def test_client():
    assert RedisClient()
"""

COMMENTED_OUT_TEST = """\
import pytest

# def test_greet():
#     assert False
"""

SOLUTION_TRAP = """#!/bin/bash
set -e
mkdir -p /logs/verifier
cleanup() {
    if [ $? -eq 0 ]; then
        echo "1" > /logs/verifier/reward.txt
    else
        echo "0" > /logs/verifier/reward.txt
    fi
}
trap cleanup EXIT
cd /app
pip3 install --quiet pytest 2>/dev/null || true
if [ ! -f /app/solution.py ]; then
    for f in /app/*.py; do
        [ -f "$f" ] && cp "$f" /app/solution.py && break
    done
fi
pytest /tests/test_solution.py -v --tb=short
"""

SOLUTION_TEST = """\
import sys
sys.path.insert(0, '/app')
from solution import *
import unittest
import numpy as np

class TestAdd(unittest.TestCase):
    def test_add(self):
        self.assertEqual(add(1, 2), 3)
"""


def _task(**overrides: bytes) -> dict[str, bytes]:
    files = {
        "environment/Dockerfile": b"FROM ubuntu:24.04\n",
        "instruction.md": b"Implement widget.add at /app/widget.py.\n",
        "task.toml": b'version = "1.0"\n',
        "tests/test.sh": TRAP_VERIFIER.encode(),
        "tests/test_solution.py": VALID_TEST.encode(),
    }
    files.update(overrides)
    return files


@pytest.mark.parametrize(
    ("pytest_exit", "total", "skipped", "summary_ok", "expected"),
    [
        (0, 3, 0, True, 1),
        (1, 3, 0, True, 0),
        (0, 3, 3, True, 0),
        (0, 0, 0, True, 0),
        (2, 0, 0, False, 0),
        (5, 0, 0, False, 0),
        (3, 0, 0, False, None),
        (4, 1, 0, True, None),
        (2, 1, 0, True, 0),
        (5, 0, 0, True, 0),
    ],
)
def test_score_pytest_run_matches_v413_contract(
    pytest_exit: int,
    total: int,
    skipped: int,
    summary_ok: bool,
    expected: int | None,
) -> None:
    assert (
        score_pytest_run(
            pytest_exit=pytest_exit,
            total=total,
            skipped=skipped,
            summary_ok=summary_ok,
        )
        == expected
    )


def test_extract_requirements_remaps_and_drops_stdlib() -> None:
    assert extract_requirements(TRAP_VERIFIER) == [
        "scikit-learn",
        "PyYAML",
        "Pillow",
        "numpy",
    ]


def test_extract_requirements_adds_known_test_imports() -> None:
    assert extract_requirements(SOLUTION_TRAP, SOLUTION_TEST) == ["numpy"]


def test_drop_reasons_reject_unparseable_and_empty_suites() -> None:
    assert drop_reasons(_task(**{"tests/test_solution.py": SYNTAX_TEST.encode()})) == [
        "syntax_error"
    ]
    assert drop_reasons(
        _task(**{"tests/test_solution.py": COMMENTED_OUT_TEST.encode()})
    ) == ["no_tests"]
    assert drop_reasons(_task()) == []


def test_patch_task_replaces_trap_and_keeps_image_and_tests() -> None:
    original = _task()
    transformed = patch_task(original, install_pytest=True)

    assert transformed["tests/test.sh"] == PYTEST_TEST_SH.encode()
    assert transformed["environment/Dockerfile"] == original["environment/Dockerfile"]
    assert transformed["tests/test_solution.py"] == original["tests/test_solution.py"]
    assert transformed["instruction.md"] == original["instruction.md"]
    assert transformed["task.toml"] == original["task.toml"]
    assert b"trap cleanup EXIT" not in transformed["tests/test.sh"]
    assert b"|| true" not in transformed["tests/test.sh"]
    assert transformed["tests/requirements.txt"] == (
        b"scikit-learn\nPyYAML\nPillow\nnumpy\n"
    )


def test_patch_task_rejects_unknown_verifier_contract() -> None:
    with pytest.raises(ValueError, match="trap-always-reward"):
        patch_task(_task(**{"tests/test.sh": b"#!/bin/bash\npytest\n"}))


def test_wrapper_is_valid_shell_and_does_not_prewrite_reward() -> None:
    subprocess.run(["bash", "-n"], input=PYTEST_TEST_SH.encode(), check=True)
    subprocess.run(["bash", "-n"], input=PYTEST_TEST_SH_IMAGE.encode(), check=True)
    assert "trap cleanup EXIT" not in PYTEST_TEST_SH
    assert PYTEST_TEST_SH.index('rm -f "$REWARD"') < PYTEST_TEST_SH.index("pytest")


def test_pytest_only_source_omits_empty_requirements() -> None:
    script = TRAP_VERIFIER.replace(
        "pip install --quiet sklearn yaml PIL __future__ numpy 2>/dev/null || true\n",
        "",
    )
    transformed = patch_task(_task(**{"tests/test.sh": script.encode()}))
    assert extract_requirements(script) == []
    assert "tests/requirements.txt" not in transformed


def test_transform_task_drops_invalid_rows() -> None:
    patched, reasons = transform_task(_task(), install_pytest=True)
    assert reasons == []
    assert patched is not None
    assert patched["tests/test.sh"] == PYTEST_TEST_SH.encode()

    dropped, reasons = transform_task(
        _task(**{"tests/test_solution.py": SYNTAX_TEST.encode()})
    )
    assert dropped is None
    assert reasons == ["syntax_error"]


def test_solution_source_keeps_oracle_and_does_not_copy_filenames() -> None:
    dockerfile = b"FROM python:3.10-slim\nWORKDIR /app\nRUN pip install pytest\n"
    original = _task(
        **{
            "environment/Dockerfile": dockerfile,
            "tests/test.sh": SOLUTION_TRAP.encode(),
            "tests/test_solution.py": SOLUTION_TEST.encode(),
            "solution/solution.py": b"def add(a, b):\n    return a + b\n",
        }
    )
    transformed = patch_task(original, install_pytest=False)

    assert transformed["tests/test.sh"] == PYTEST_TEST_SH_IMAGE.encode()
    assert transformed["tests/test.sh"] == pytest_test_sh(install_pytest=False).encode()
    assert b'cp "$f" /app/solution.py' not in transformed["tests/test.sh"]
    assert b"|| true" not in transformed["tests/test.sh"]
    assert transformed["environment/Dockerfile"] == dockerfile
    assert transformed["solution/solution.py"] == original["solution/solution.py"]
    assert transformed["tests/test_solution.py"] == original["tests/test_solution.py"]
    assert transformed["tests/requirements.txt"] == b"numpy\n"
