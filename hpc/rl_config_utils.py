"""RL Training configuration parsing utilities for SkyRL.

This module provides YAML-based configuration for SkyRL RL training jobs,
replacing 50+ Hydra CLI arguments with a single --rl_config YAML file.

Usage:
    from hpc.rl_config_utils import parse_rl_config, build_skyrl_hydra_args

    parsed = parse_rl_config("terminal_bench.yaml")
    hydra_args = build_skyrl_hydra_args(parsed, exp_args, hpc)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml

# Directory containing built-in SkyRL config YAML files
SKYRL_CONFIG_DIR = Path(__file__).parent / "skyrl_yaml"


@dataclass
class ParsedRLConfig:
    """Result of parsing an RL configuration YAML file.

    Attributes:
        config_path: Resolved absolute path to the config file.
        raw: Raw dictionary from YAML parsing.
        entrypoint: SkyRL entrypoint module (e.g., examples.terminal_bench.entrypoints.main_tbench).
        config_groups: Hydra config groups to apply (e.g., {"terminal_bench_config": "terminal_bench"}).
        trainer: Trainer configuration dictionary.
        generator: Generator (vLLM) configuration dictionary.
        data: Data paths configuration dictionary.
        terminal_bench: Terminal bench specific settings (optional).
        tensor_parallel_size: Tensor parallel size extracted from generator config.
    """

    config_path: Path
    raw: Dict[str, Any]
    entrypoint: str
    config_groups: Dict[str, str] = field(default_factory=dict)
    trainer: Dict[str, Any] = field(default_factory=dict)
    generator: Dict[str, Any] = field(default_factory=dict)
    data: Dict[str, Any] = field(default_factory=dict)
    terminal_bench: Optional[Dict[str, Any]] = None
    tensor_parallel_size: int = 1


def resolve_rl_config_path(raw_path: str) -> Path:
    """Resolve RL config path, checking SKYRL_CONFIG_DIR fallback.

    Resolution order:
    1. If raw_path exists as-is, use it
    2. Check SKYRL_CONFIG_DIR / raw_path
    3. Check SKYRL_CONFIG_DIR / raw_path.yaml

    Args:
        raw_path: User-provided config path (can be relative or just a name).

    Returns:
        Resolved absolute path to the config file.

    Raises:
        FileNotFoundError: If config file cannot be found in any location.
    """
    path = Path(raw_path).expanduser()
    if path.exists():
        return path.resolve()

    # Check built-in configs directory
    fallback = SKYRL_CONFIG_DIR / raw_path
    if fallback.exists():
        return fallback.resolve()

    # Try with .yaml extension
    fallback_yaml = SKYRL_CONFIG_DIR / f"{raw_path}.yaml"
    if fallback_yaml.exists():
        return fallback_yaml.resolve()

    raise FileNotFoundError(
        f"RL config not found: {raw_path}\n"
        f"Searched: {path}, {SKYRL_CONFIG_DIR / raw_path}, {fallback_yaml}"
    )


def parse_rl_config(
    config_path: str,
    model_override: Optional[str] = None,
) -> ParsedRLConfig:
    """Parse RL config YAML and extract all settings.

    Args:
        config_path: Path to YAML config file (or name of built-in config).
        model_override: Optional model path to override config's model setting.

    Returns:
        ParsedRLConfig dataclass with all parsed settings.

    Raises:
        FileNotFoundError: If config file cannot be found.
        yaml.YAMLError: If config file is not valid YAML.
    """
    path = resolve_rl_config_path(config_path)

    with open(path) as f:
        raw = yaml.safe_load(f) or {}

    entrypoint = raw.get("entrypoint", "skyrl_train.entrypoints.main_base")
    config_groups = raw.get("config_groups", {})
    trainer = raw.get("trainer", {})
    generator = raw.get("generator", {})
    data = raw.get("data", {})
    terminal_bench = raw.get("terminal_bench")

    # Apply model override if provided
    if model_override:
        trainer.setdefault("policy", {}).setdefault("model", {})["path"] = model_override

    # Extract tensor parallel size from generator config
    tensor_parallel_size = generator.get("inference_engine_tensor_parallel_size", 1)

    return ParsedRLConfig(
        config_path=path,
        raw=raw,
        entrypoint=entrypoint,
        config_groups=config_groups,
        trainer=trainer,
        generator=generator,
        data=data,
        terminal_bench=terminal_bench,
        tensor_parallel_size=tensor_parallel_size,
    )


def _flatten_dict(d: Dict[str, Any], prefix: str = "") -> Dict[str, Any]:
    """Flatten a nested dictionary to dotted keys.

    Example:
        {"trainer": {"policy": {"lr": 1e-6}}}
        -> {"trainer.policy.lr": 1e-6}

    Args:
        d: Dictionary to flatten.
        prefix: Key prefix for recursion.

    Returns:
        Flattened dictionary with dotted keys.
    """
    items = {}
    for k, v in d.items():
        key = f"{prefix}.{k}" if prefix else k
        if isinstance(v, dict):
            items.update(_flatten_dict(v, key))
        elif v is not None:
            items[key] = v
    return items


def _format_hydra_arg(key: str, value: Any, prefix: str = "") -> str:
    """Format a single Hydra CLI argument.

    Handles special formatting for different types:
    - bool: lowercase true/false
    - list: YAML list notation (no outer quotes so Hydra parses as list, not string)
    - str/int/float: direct value

    Args:
        key: Dotted key name (e.g., "trainer.epochs").
        value: Value to format.
        prefix: Hydra prefix to use:
            - "" (empty): override existing key
            - "+": add new key (fails if exists)
            - "++": add or override (works either way)

    Returns:
        Formatted Hydra argument string (e.g., "trainer.epochs=10" or "++key=val").
    """
    if isinstance(value, bool):
        return f"{prefix}{key}={str(value).lower()}"
    elif isinstance(value, (list, tuple)):
        # Format as YAML list WITHOUT outer quotes so Hydra parses it as a list
        # (with outer quotes like "['a']", Hydra treats it as a string literal)
        # Use double quotes around string items to handle paths with special chars
        items = ",".join(
            f'"{v}"' if isinstance(v, str) else str(v)
            for v in value
        )
        return f"{prefix}{key}=[{items}]"
    else:
        return f"{prefix}{key}={value}"


def build_skyrl_hydra_args(
    parsed: ParsedRLConfig,
    exp_args: Dict[str, Any],
    hpc: Any,
) -> List[str]:
    """Convert parsed config + exp_args to Hydra CLI arguments.

    This function:
    1. Adds config groups with + prefix
    2. Derives paths from experiments_dir/job_name if not set
    3. Computes num_inference_engines from cluster config
    4. Flattens nested dicts to dotted Hydra keys
    5. Applies data paths from CLI

    Args:
        parsed: ParsedRLConfig from parse_rl_config().
        exp_args: Experiment arguments dictionary from CLI.
        hpc: HPC configuration object with cluster settings.

    Returns:
        List of Hydra CLI argument strings.
    """
    args = []

    # Config groups (+ prefix for Hydra)
    for group_name, config_name in parsed.config_groups.items():
        args.append(f"+{group_name}={config_name}")

    # Make copies to avoid mutating parsed config
    trainer = dict(parsed.trainer)
    generator = dict(parsed.generator)
    data = dict(parsed.data)

    # Derive paths if null
    experiments_dir = exp_args.get("experiments_dir", "")
    job_name = exp_args.get("job_name", "")

    if not trainer.get("run_name") and job_name:
        trainer["run_name"] = job_name
    if not trainer.get("export_path") and experiments_dir and job_name:
        trainer["export_path"] = f"{experiments_dir}/{job_name}/exports"
    if not trainer.get("ckpt_path") and experiments_dir and job_name:
        trainer["ckpt_path"] = f"{experiments_dir}/{job_name}/exports"

    # Derive placement from num_nodes
    num_nodes = int(exp_args.get("num_nodes", 1))
    gpus_per_node = int(exp_args.get("gpus_per_node", getattr(hpc, "gpus_per_node", 4)))
    placement = dict(trainer.get("placement", {}))

    policy_num_nodes = exp_args.get("policy_num_nodes")
    if placement.get("policy_num_nodes") is None:
        placement["policy_num_nodes"] = policy_num_nodes if policy_num_nodes is not None else num_nodes
    if placement.get("ref_num_nodes") is None:
        placement["ref_num_nodes"] = policy_num_nodes if policy_num_nodes is not None else num_nodes
    # Derive gpus_per_node from CLI (cluster-specific, not hardcoded in YAML)
    if placement.get("policy_num_gpus_per_node") is None or exp_args.get("gpus_per_node"):
        placement["policy_num_gpus_per_node"] = gpus_per_node
    if placement.get("ref_num_gpus_per_node") is None or exp_args.get("gpus_per_node"):
        placement["ref_num_gpus_per_node"] = gpus_per_node
    trainer["placement"] = placement

    # Compute num_inference_engines
    tp_size = parsed.tensor_parallel_size
    if generator.get("num_inference_engines") is None:
        generator["num_inference_engines"] = (num_nodes * gpus_per_node) // tp_size

    # Data paths from CLI
    if exp_args.get("train_data"):
        train_data = exp_args["train_data"]
        # Handle string that looks like a list
        if isinstance(train_data, str) and train_data.startswith("["):
            # Will be formatted properly by _format_hydra_arg
            import ast
            try:
                train_data = ast.literal_eval(train_data)
            except (ValueError, SyntaxError):
                pass
        data["train_data"] = train_data

    if exp_args.get("val_data"):
        val_data = exp_args["val_data"]
        if isinstance(val_data, str) and val_data.startswith("["):
            import ast
            try:
                val_data = ast.literal_eval(val_data)
            except (ValueError, SyntaxError):
                pass
        data["val_data"] = val_data

    # Model path
    if exp_args.get("model_path"):
        trainer.setdefault("policy", {}).setdefault("model", {})["path"] = exp_args["model_path"]

    # Build args for each section
    # Keys under engine_init_kwargs need ++ prefix (add or override) since some keys
    # may already exist in SkyRL's base config while others are new
    for section, values in [("trainer", trainer), ("generator", generator), ("data", data)]:
        for key, val in _flatten_dict(values, section).items():
            # engine_init_kwargs keys need ++ prefix to add-or-override
            # (+ would fail if key already exists in base config like enforce_eager)
            prefix = "++" if ".engine_init_kwargs." in key else ""
            args.append(_format_hydra_arg(key, val, prefix=prefix))

    # Terminal bench with + prefix (these are new keys added by the config group)
    if parsed.terminal_bench:
        for key, val in _flatten_dict(parsed.terminal_bench).items():
            args.append(_format_hydra_arg(f"terminal_bench_config.{key}", val, prefix="+"))

    return args


def get_skyrl_command_preview(
    entrypoint: str,
    hydra_args: List[str],
    max_args_shown: int = 10,
) -> str:
    """Generate a preview of the SkyRL command for dry-run output.

    Args:
        entrypoint: SkyRL entrypoint module.
        hydra_args: List of Hydra CLI arguments.
        max_args_shown: Maximum number of args to show before truncating.

    Returns:
        Formatted command string for display.
    """
    lines = [f"python -m {entrypoint} \\"]

    for i, arg in enumerate(hydra_args):
        if i < max_args_shown:
            lines.append(f"  {arg} \\")
        elif i == max_args_shown:
            lines.append(f"  ... ({len(hydra_args) - max_args_shown} more arguments)")
            break

    # Remove trailing backslash from last shown arg
    if lines and lines[-1].endswith(" \\"):
        lines[-1] = lines[-1][:-2]

    return "\n".join(lines)


__all__ = [
    "ParsedRLConfig",
    "SKYRL_CONFIG_DIR",
    "resolve_rl_config_path",
    "parse_rl_config",
    "build_skyrl_hydra_args",
    "get_skyrl_command_preview",
]
