"""
Utility helpers shared across HPC launch entry points.
"""

from __future__ import annotations

import os
import re
import socket
import shutil
from collections import defaultdict
from typing import Any, Mapping, Optional

from hpc.hpc import detect_hpc

from .job_name_ignore_list import JOB_NAME_IGNORE_KEYS
from .arguments import JobType
from .datagen_launch_utils import derive_datagen_job_name
from .sft_launch_utils import build_accelerate_config_block


def sanitize_repo_for_job(repo_id: str) -> str:
    """Return a filesystem-safe representation of a repo identifier."""

    safe = re.sub(r"[^A-Za-z0-9._\-]+", "-", repo_id.strip())
    safe = safe.strip("-_")
    return safe or "consolidate"


def sanitize_repo_component(value: Optional[str]) -> Optional[str]:
    """Extract the meaningful suffix from trace repositories (traces-<slug>)."""

    if not value:
        return None
    match = re.search(r"traces-([A-Za-z0-9._\-]+)", value)
    return match.group(1) if match else None


def get_job_name(cli_args: Mapping[str, Any]) -> str:
    """Derive a stable job name from user-provided CLI arguments."""

    job_type = str(cli_args.get("job_type", JobType.default_value()) or JobType.default_value()).lower()
    if job_type == JobType.CONSOLIDATE.value:
        repo_id = (
            cli_args.get("consolidate_repo_id")
            or cli_args.get("consolidate_base_repo")
            or "consolidate"
        )
        job_name = f"{sanitize_repo_for_job(str(repo_id))}_consolidate"
        if len(job_name) > 96:
            job_name = job_name[:96]
        return job_name
    if job_type == JobType.DATAGEN.value:
        return derive_datagen_job_name(cli_args)

    job_name_components: list[str] = []
    job_name_suffix: Optional[str] = None

    for key, value in cli_args.items():
        if not isinstance(value, (str, int, float)):
            continue
        if value == "None" or key in JOB_NAME_IGNORE_KEYS:
            continue

        if key == "seed":
            try:
                if float(value) == 42:
                    continue
            except (TypeError, ValueError):
                pass

        if key not in {"dataset", "model_name_or_path"}:
            job_name_components.append(str(key).replace("_", "-"))

        value_str = str(value)
        if value_str == "Qwen/Qwen2.5-32B-Instruct":
            job_name_suffix = "_32B"
        elif value_str == "Qwen/Qwen2.5-14B-Instruct":
            job_name_suffix = "_14B"
        elif value_str == "Qwen/Qwen2.5-3B-Instruct":
            job_name_suffix = "_3B"
        elif value_str == "Qwen/Qwen2.5-1.5B-Instruct":
            job_name_suffix = "_1.5B"
        else:
            job_name_components.append(value_str.split("/")[-1])

    job_name = "_".join(job_name_components)
    job_name = (
        job_name.replace("/", "_")
        .replace("?", "")
        .replace("*", "")
        .replace("{", "")
        .replace("}", "")
        .replace(":", "")
        .replace('"', "")
        .replace(" ", "_")
    )
    if job_name_suffix:
        job_name += job_name_suffix

    if len(job_name) > 96:
        print("Truncating job name to less than HF limit of 96 characters...")
        job_name = "_".join(
            "-".join(segment[:4] for segment in chunk.split("-"))
            for chunk in job_name.split("_")
        )
        if len(job_name) > 96:
            raise ValueError(
                f"Job name {job_name} is still too long (96 characters) after truncation. "
                "Try renaming the dataset or providing a shorter YAML config."
            )

    return job_name


def check_exists(local_path: str | os.PathLike[str]) -> bool:
    """Return True when ``local_path`` exists."""

    return os.path.exists(local_path)


def extract_template_keys(file_path: str) -> list[str]:
    with open(file_path, "r") as f:
        file = f.read()
    return re.findall(r"(?<!\$)\{([^{}]*)\}", file)


def fill_template(file_path: str, exp_args: dict, new_file_path: str) -> None:
    with open(file_path, "r") as f:
        file = f.read()

    file = re.sub(r"(?<!\$)\{([^{}]*)\}", lambda m: exp_args[m.group(1)], file)

    with open(new_file_path, "w") as f:
        f.write(file)


def _escape_bash_variables(text: str) -> str:
    result: list[str] = []
    i = 0
    length = len(text)
    while i < length:
        if text[i] == "$" and i + 1 < length and text[i + 1] == "{":
            start = i
            depth = 1
            j = i + 2
            while j < length and depth > 0:
                if text[j] == "{":
                    depth += 1
                elif text[j] == "}":
                    depth -= 1
                j += 1
            inner = text[i + 2 : j - 1]
            escaped_inner = _escape_bash_variables(inner)
            result.append("${{" + escaped_inner + "}}")
            i = j
        else:
            result.append(text[i])
            i += 1
    return "".join(result)


def construct_sbatch_script(exp_args: dict) -> str:
    base_script_path = exp_args["train_sbatch_path"]
    with open(base_script_path, "r") as f:
        base_script = f.read()

    kwargs = defaultdict(str, **exp_args)
    kwargs["accelerate_config_block"] = build_accelerate_config_block(exp_args)

    json_files_cat = re.findall(r"cat.*?<<EOT >.*?EOT", base_script, re.DOTALL)
    json_filenames = []
    for json_file in json_files_cat:
        json_file_name = re.match(
            r"cat.*?<<EOT >.*?(\S+).*?EOT", json_file, re.DOTALL
        ).group(1)
        json_filenames.append(json_file_name)

        base_script = re.sub(
            r"cat.*?<<EOT >.*?" + json_file_name.replace("$", "\\$") + r".*?EOT",
            f"cat {json_file_name}",
            base_script,
            count=1,
            flags=re.DOTALL,
        )

    base_script = _escape_bash_variables(base_script)

    time_limit = kwargs.get("time_limit")
    if time_limit is None:
        time_limit = "01:00:00"
        kwargs["time_limit"] = time_limit

    hpc = detect_hpc()
    hpc_name = hpc.name
    if hpc_name == "jureca" or hpc_name == "juwels":
        login_node = socket.gethostname().split(".")[0] + "i"
        if "{login_node}" in base_script:
            if kwargs.get("internet_node", False):
                if not shutil.which("proxychains4"):
                    raise RuntimeError("proxychains4 not found, please install it to use internet_node")
            base_script = base_script.replace("{login_node}", login_node)

    sbatch_script = base_script.format(**kwargs)
    sbatch_script = _ensure_dependency_directive(sbatch_script, exp_args.get("dependency"))

    env_block = {
        "DISABLE_VERSION_CHECK": "1",
    }
    stage_value = str(exp_args.get("stage") or "").lower()
    if exp_args.get("use_mca") and stage_value == "sft":
        env_block["USE_MCA"] = "1"
        os.environ.setdefault("USE_MCA", "1")

    sbatch_script = _inject_env_block(sbatch_script, env_block)

    for json_file, json_file_name in zip(json_files_cat, json_filenames):
        sbatch_script = sbatch_script.replace(f"cat {json_file_name}", json_file)

    sbatch_dir = os.path.join(kwargs["experiments_dir"], "sbatch_scripts")
    os.makedirs(sbatch_dir, exist_ok=True)
    sbatch_script_path = os.path.join(sbatch_dir, f"{kwargs['job_name']}.sbatch")
    with open(sbatch_script_path, "w") as f:
        f.write(sbatch_script)
        print(f"Wrote sbatch script to {sbatch_script_path}")

    return sbatch_script_path


__all__ = [
    "check_exists",
    "construct_sbatch_script",
    "extract_template_keys",
    "fill_template",
    "get_job_name",
    "sanitize_repo_for_job",
    "sanitize_repo_component",
]
