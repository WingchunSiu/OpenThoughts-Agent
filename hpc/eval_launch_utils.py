"""Utilities for launching Harbor eval jobs via the HPC launcher."""

from __future__ import annotations

import json
import os
import shlex
import re
from pathlib import Path
from typing import Any, Dict, Optional

from data.generation import BaseDataGenerator

from hpc.core_launch_utils import cleanup_endpoint_file, validate_trace_backend
from hpc.launch_utils import launch_sbatch, _parse_optional_int, _inject_env_block
from hpc.datagen_launch_utils import default_vllm_endpoint_path, _build_vllm_env_vars
from scripts.harbor.job_config_utils import load_job_config

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_REGISTRY_HINTS = [
    Path(os.environ.get("HARBOR_REGISTRY_PATH", "")).expanduser()
    if os.environ.get("HARBOR_REGISTRY_PATH")
    else None,
    PROJECT_ROOT.parent / "harbor" / "registry.json",
]


def _resolve_repo_path(path_like: str) -> Path:
    path = Path(path_like).expanduser()
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    return path.resolve()


def _resolve_workspace_path(path_like: str) -> Path:
    path = Path(path_like).expanduser()
    if path.is_absolute():
        return path
    return (PROJECT_ROOT / path).resolve()


def _load_harbor_registry() -> dict | None:
    for candidate in DEFAULT_REGISTRY_HINTS:
        if candidate and candidate.exists():
            try:
                return json.loads(candidate.read_text())
            except Exception:
                return None
    return None


def _build_dataset_slug_set(registry: dict | None) -> set[str]:
    if not registry:
        return set()
    entries: set[str] = set()
    for item in registry:
        name = item.get("name")
        version = item.get("version")
        if not name:
            continue
        if version:
            entries.add(f"{name}@{version}")
        entries.add(name)
    return entries


def _validate_harbor_dataset_slug(slug: str) -> None:
    registry = _load_harbor_registry()
    if not registry:
        return
    valid = _build_dataset_slug_set(registry)
    if slug not in valid:
        raise ValueError(
            f"Dataset '{slug}' is not in the local Harbor registry "
            f"(known datasets: {sorted(list(valid))[:8]} ...). "
            "Specify --eval-dataset-path instead or update the registry hint."
        )


def _coerce_agent_kwargs(value: Any) -> Dict[str, Any]:
    if value in (None, "", {}):
        return {}
    if isinstance(value, dict):
        return value
    if isinstance(value, str):
        try:
            parsed = json.loads(value)
        except json.JSONDecodeError as exc:
            raise ValueError(f"Failed to parse eval agent kwargs JSON: {exc}") from exc
        if not isinstance(parsed, dict):
            raise ValueError("Eval agent kwargs must decode to an object/dict.")
        return parsed
    raise ValueError("Eval agent kwargs must be provided as JSON string or dict.")


def prepare_eval_configuration(exp_args: dict) -> dict:
    """Normalize eval config inputs prior to sbatch generation."""

    if exp_args.get("_datagen_config_obj") is None:
        raise ValueError("Eval jobs reuse the datagen engine. Provide --datagen-config.")

    harbor_cfg = exp_args.get("trace_harbor_config")
    if not harbor_cfg:
        raise ValueError("Eval jobs require --trace-harbor-config pointing at an eval YAML.")
    resolved_cfg = _resolve_repo_path(harbor_cfg)
    if "_eval_" not in resolved_cfg.name:
        raise ValueError(
            f"Eval Harbor YAML '{resolved_cfg.name}' must include '_eval_' in the filename."
        )
    harbor_job = load_job_config(resolved_cfg)
    if not harbor_job.agents:
        raise ValueError(f"Eval Harbor YAML '{resolved_cfg.name}' must define at least one agent.")

    exp_args["_eval_harbor_config_resolved"] = str(resolved_cfg)
    exp_args["_eval_harbor_config"] = harbor_job

    dataset_path = exp_args.get("trace_input_path")
    harbor_dataset = exp_args.get("harbor_dataset")
    if dataset_path and harbor_dataset:
        raise ValueError(
            "Eval jobs accept either --trace-input-path or --harbor-dataset, but not both."
        )
    if dataset_path:
        resolved_dataset = _resolve_repo_path(dataset_path)
        exp_args["_eval_dataset_path_resolved"] = str(resolved_dataset)
        exp_args["trace_input_path"] = str(resolved_dataset)
    if harbor_dataset:
        slug = harbor_dataset.strip()
        if not slug:
            raise ValueError("--harbor-dataset cannot be empty.")
        _validate_harbor_dataset_slug(slug)
        exp_args["harbor_dataset"] = slug

    if not (exp_args.get("harbor_dataset") or exp_args.get("_eval_dataset_path_resolved")):
        raise ValueError(
            "Eval jobs require either --harbor-dataset or --trace-input-path to specify tasks."
        )

    benchmark_repo = exp_args.get("eval_benchmark_repo")
    if not benchmark_repo:
        raise ValueError(
            "Eval jobs require --eval-benchmark-repo so Supabase rows can be created."
        )

    model_name = exp_args.get("trace_model")
    if not model_name:
        vllm_cfg = exp_args.get("_datagen_vllm_server_config")
        if vllm_cfg and getattr(vllm_cfg, "model_path", None):
            model_name = vllm_cfg.model_path
    if not model_name:
        model_name = exp_args.get("datagen_model")
    if not model_name and harbor_job.agents:
        model_name = harbor_job.agents[0].model_name
    if not model_name:
        raise ValueError("Eval jobs require --trace-model (or --datagen-model).")
    exp_args["_eval_model_name"] = model_name
    exp_args["trace_model"] = model_name

    agent_cfg = harbor_job.agents[0]
    agent_name = (
        exp_args.get("trace_agent_name")
        or agent_cfg.name
        or (agent_cfg.import_path or "terminus-2")
    )
    exp_args["_eval_agent_name"] = agent_name
    if "trace_agent_name" not in exp_args:
        exp_args["trace_agent_name"] = agent_name

    base_agent_kwargs = dict(agent_cfg.kwargs or {})
    datagen_agent_defaults = dict(exp_args.get("_datagen_extra_agent_kwargs") or {})
    base_agent_kwargs.update(datagen_agent_defaults)
    cli_agent_kwargs = _coerce_agent_kwargs(exp_args.get("trace_agent_kwargs"))
    agent_kwargs: Dict[str, Any] = dict(base_agent_kwargs)
    agent_kwargs.update(cli_agent_kwargs)
    exp_args["_eval_agent_kwargs"] = agent_kwargs

    if exp_args.get("trace_env"):
        eval_env = exp_args["trace_env"]
    else:
        env_cfg = getattr(harbor_job.environment, "type", None)
        if hasattr(env_cfg, "value"):
            eval_env = env_cfg.value
        elif env_cfg:
            eval_env = str(env_cfg)
        else:
            eval_env = "daytona"
    exp_args["_eval_env"] = str(eval_env)

    trace_backend_value = exp_args.get("trace_backend") or exp_args.get("datagen_backend")
    trace_backend = validate_trace_backend(
        trace_backend_value,
        allow_vllm=True,
        job_type="eval",
    )
    exp_args["trace_backend"] = trace_backend

    default_n_concurrent = harbor_job.orchestrator.n_concurrent_trials
    if default_n_concurrent is None or default_n_concurrent < 1:
        default_n_concurrent = 64
    n_concurrent_override = _parse_optional_int(
        exp_args.get("trace_n_concurrent"),
        "--trace_n_concurrent",
    )
    n_concurrent_int = n_concurrent_override or default_n_concurrent
    exp_args["_eval_n_concurrent"] = max(1, int(n_concurrent_int))

    default_n_attempts = harbor_job.n_attempts or 3
    n_attempts_override = _parse_optional_int(
        exp_args.get("trace_n_attempts"),
        "--trace_n_attempts",
    )
    n_attempts_int = n_attempts_override or default_n_attempts
    exp_args["_eval_n_attempts"] = max(1, int(n_attempts_int))

    expected_override = _parse_optional_int(
        exp_args.get("trace_expected_trials"),
        "--trace_expected_trials",
    )
    expected_trials_int = expected_override or exp_args["_eval_n_concurrent"]
    exp_args["_eval_expected_trials"] = max(1, int(expected_trials_int))

    return exp_args


def _shell_quote(value: Optional[str]) -> str:
    if value in (None, "", "None"):
        return "''"
    return shlex.quote(str(value))


def _write_agent_kwargs_file(
    agent_kwargs: Dict[str, Any],
    configs_dir: Path,
    job_name: str,
) -> Optional[Path]:
    if not agent_kwargs:
        return None
    target = configs_dir / f"{job_name}_eval_agent_kwargs.json"
    target.write_text(json.dumps(agent_kwargs, indent=2), encoding="utf-8")
    return target


def launch_eval_job(exp_args: dict, hpc) -> None:
    """Construct and submit the sbatch script for eval workloads."""

    print("\n=== EVAL MODE ===")
    template_path = Path(__file__).parent / "sbatch_eval" / f"{hpc.name}_eval_harbor.sbatch"
    if not template_path.exists():
        raise FileNotFoundError(
            f"No eval sbatch template found for cluster '{hpc.name}'. "
            f"Expected {template_path}."
        )

    experiments_subdir = exp_args.get("experiments_dir") or "experiments"
    experiments_abs = _resolve_workspace_path(experiments_subdir)
    sbatch_dir = experiments_abs / "sbatch"
    sbatch_dir.mkdir(parents=True, exist_ok=True)
    configs_dir = experiments_abs / "configs"
    configs_dir.mkdir(parents=True, exist_ok=True)

    job_name = exp_args.get("job_name")
    if not job_name:
        raise ValueError("Eval jobs require a --job_name.")

    harbor_cfg = exp_args.get("_eval_harbor_config_resolved")
    dataset_path = exp_args.get("_eval_dataset_path_resolved")
    dataset_slug = exp_args.get("harbor_dataset")

    model_name = exp_args.get("_eval_model_name") or exp_args.get("trace_model") or exp_args.get("datagen_model")
    if not model_name:
        raise ValueError("Unable to determine eval model; provide --trace-model or set it in the datagen config.")

    agent_name = exp_args.get("_eval_agent_name") or exp_args.get("trace_agent_name")
    if not agent_name:
        raise ValueError("Eval jobs require an agent name (set one in the Harbor YAML or pass --trace-agent-name).")

    agent_kwargs = exp_args.get("_eval_agent_kwargs") or {}
    agent_kwargs_path = _write_agent_kwargs_file(agent_kwargs, configs_dir, job_name)

    eval_env = exp_args.get("_eval_env") or "daytona"
    n_concurrent = int(exp_args.get("_eval_n_concurrent", 64))
    n_attempts = int(exp_args.get("_eval_n_attempts", 3))
    expected_trials = int(exp_args.get("_eval_expected_trials", n_concurrent))

    upload_username = exp_args.get("job_creator") or os.environ.get("USER", "unknown")
    hf_repo_prefix = "DCAgent2"
    upload_mode = "skip_on_error"
    benchmark_repo = exp_args.get("eval_benchmark_repo")

    dataset_label = dataset_slug or (dataset_path or "").strip()
    if not dataset_label:
        dataset_label = "dataset"

    trace_engine_value = exp_args.get("trace_engine") or exp_args.get("datagen_engine") or ""
    trace_engine = str(trace_engine_value or "").lower()
    trace_backend_value = exp_args.get("trace_backend") or exp_args.get("datagen_backend") or "vllm"
    trace_backend = str(trace_backend_value or "vllm").lower()
    vllm_cfg = exp_args.get("_datagen_vllm_server_config")
    requires_vllm_endpoint = bool(
        vllm_cfg and trace_engine == "vllm_local" and trace_backend in {"vllm", "ray"}
    )
    experiments_dir_value = exp_args.get("experiments_dir") or experiments_subdir
    trace_endpoint_json = exp_args.get("trace_endpoint_json") or exp_args.get("vllm_endpoint_json_path") or ""
    if requires_vllm_endpoint and not trace_endpoint_json:
        if not experiments_dir_value:
            raise ValueError("experiments_dir is required to derive the vLLM endpoint JSON path.")
        trace_endpoint_json = default_vllm_endpoint_path(experiments_dir_value)
        exp_args["vllm_endpoint_json_path"] = trace_endpoint_json
    if requires_vllm_endpoint:
        cleanup_endpoint_file(trace_endpoint_json, descriptor="stale eval endpoint file")

    default_health_attempts = getattr(BaseDataGenerator, "HEALTHCHECK_MAX_ATTEMPTS", 20)
    trace_health_max_attempts = _parse_optional_int(
        exp_args.get("trace_health_max_attempts"),
        "--trace_health_max_attempts",
    ) or default_health_attempts

    default_health_delay = getattr(BaseDataGenerator, "HEALTHCHECK_RETRY_DELAY", 30)
    trace_health_retry_delay = _parse_optional_int(
        exp_args.get("trace_health_retry_delay"),
        "--trace_health_retry_delay",
    ) or default_health_delay

    should_wait_for_endpoint = bool(trace_endpoint_json) and requires_vllm_endpoint
    trace_require_endpoint = "1" if requires_vllm_endpoint else "0"

    gpus_per_node_raw = exp_args.get("gpus_per_node")
    if isinstance(gpus_per_node_raw, str) and gpus_per_node_raw.strip() == "":
        gpus_per_node_raw = None
    default_gpu_count = getattr(hpc, "gpus_per_node", None)
    if isinstance(default_gpu_count, str) and not default_gpu_count.strip():
        default_gpu_count = None
    gpus_per_node = int(gpus_per_node_raw or default_gpu_count or 1)

    trace_ray_port = str(exp_args.get("datagen_ray_port") or exp_args.get("trace_ray_port") or 6379)
    trace_api_port = str(exp_args.get("datagen_api_port") or exp_args.get("trace_api_port") or 8000)

    tensor_parallel_size = getattr(vllm_cfg, "tensor_parallel_size", None) or 1
    pipeline_parallel_size = getattr(vllm_cfg, "pipeline_parallel_size", None) or 1
    data_parallel_size = getattr(vllm_cfg, "data_parallel_size", None) or 1

    vllm_env_vars: Dict[str, str] = {}
    if requires_vllm_endpoint:
        vllm_env_vars, exp_args = _build_vllm_env_vars(exp_args, include_pinggy=True)
        vllm_env_vars.setdefault("VLLM_BACKEND", trace_backend)
        vllm_env_vars.setdefault("VLLM_RAY_PORT", trace_ray_port)
        vllm_env_vars.setdefault("VLLM_API_PORT", trace_api_port)
        if gpus_per_node:
            vllm_env_vars.setdefault("VLLM_GPUS_PER_NODE", str(gpus_per_node))

    sbatch_output = sbatch_dir / f"{job_name}_eval.sbatch"
    substitutions = {
        "partition": hpc.partition,
        "time_limit": exp_args.get("time_limit") or os.environ.get("DEFAULT_TIME_LIMIT", "24:00:00"),
        "num_nodes": exp_args.get("num_nodes") or 1,
        "cpus_per_node": exp_args.get("cpus_per_node") or hpc.cpus_per_node,
        "gpus_per_node": str(gpus_per_node),
        "account": exp_args.get("account") or hpc.account,
        "experiments_dir": experiments_subdir,
        "job_name": job_name,
        "eval_dataset_path": _shell_quote(dataset_path),
        "harbor_dataset": _shell_quote(dataset_slug),
        "eval_harbor_config": _shell_quote(harbor_cfg),
        "eval_agent_name": _shell_quote(agent_name),
        "eval_model": _shell_quote(model_name),
        "eval_env": _shell_quote(eval_env),
        "eval_n_concurrent": str(n_concurrent),
        "eval_n_attempts": str(n_attempts),
        "eval_expected_trials": str(expected_trials),
        "eval_benchmark_repo": _shell_quote(benchmark_repo),
        "eval_upload_username": _shell_quote(upload_username),
        "eval_upload_mode": _shell_quote(upload_mode),
        "eval_hf_repo_prefix": _shell_quote(hf_repo_prefix),
        "agent_kwargs_path": _shell_quote(str(agent_kwargs_path) if agent_kwargs_path else ""),
        "dataset_label": _shell_quote(dataset_label),
        "trace_endpoint_json": _shell_quote(trace_endpoint_json or ""),
        "trace_wait_for_endpoint": "1" if should_wait_for_endpoint else "0",
        "trace_require_endpoint": trace_require_endpoint,
        "trace_health_max_attempts": str(trace_health_max_attempts),
        "trace_health_retry_delay": str(trace_health_retry_delay),
        "trace_backend": _shell_quote(trace_backend),
        "trace_engine": _shell_quote(trace_engine),
        "needs_vllm_server": "1" if requires_vllm_endpoint else "0",
        "trace_ray_port": _shell_quote(trace_ray_port),
        "trace_api_port": _shell_quote(trace_api_port),
        "trace_tensor_parallel_size": str(tensor_parallel_size),
        "trace_pipeline_parallel_size": str(pipeline_parallel_size),
        "trace_data_parallel_size": str(data_parallel_size),
        "trace_gpus_per_node": str(gpus_per_node),
    }

    template_text = template_path.read_text()

    placeholder_pattern = re.compile(r"{([A-Za-z0-9_]+)}")

    def _substitute(match: re.Match[str]) -> str:
        key = match.group(1)
        if key in substitutions:
            return str(substitutions[key])
        return match.group(0)

    sbatch_text = placeholder_pattern.sub(_substitute, template_text)
    if vllm_env_vars:
        sbatch_text = _inject_env_block(sbatch_text, vllm_env_vars)
    sbatch_output.write_text(sbatch_text, encoding="utf-8")
    os.chmod(sbatch_output, 0o750)

    if exp_args.get("dry_run"):
        print(f"DRY RUN: Eval sbatch script written to {sbatch_output}")
        print("--------")
        print(sbatch_text)
        print("--------")
        return

    job_id = launch_sbatch(str(sbatch_output))
    print(f"\nEval job submitted via {sbatch_output}")
    print(f"SLURM Job ID: {job_id}")
