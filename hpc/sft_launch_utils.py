from __future__ import annotations

import json
import os
import subprocess
import sys
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Callable, Optional

import yaml

from hpc.arguments import LlamaFactoryArgs
from hpc.data_argument_keys import DATA_ARGUMENT_KEYS
from hpc.launch_utils import (
    setup_experiments_dir,
    substitute_template,
    build_sbatch_directives,
)


def apply_mca_training_template(
    exp_args: dict,
    hpc,
    *,
    update_exp_args_fn: Callable[[dict, dict], dict],
) -> dict:
    """Point training jobs at the MCA-specific sbatch template when requested."""

    mca_template = Path(__file__).parent / "sbatch" / f"{hpc.name.lower()}_train_mca.sbatch"
    if mca_template.exists():
        return update_exp_args_fn(
            exp_args,
            {
                "train_sbatch_filename": mca_template.name,
                "train_sbatch_path": str(mca_template),
            },
        )

    print(
        f"Warning: MCA sbatch template {mca_template} not found for cluster {hpc.name}; using default template."
    )
    return exp_args


def build_training_parameters_link(hub_model_id: Optional[str]) -> Optional[str]:
    if not hub_model_id:
        return None
    hub_model_id = hub_model_id.strip("/")
    return f"https://huggingface.co/{hub_model_id}/blob/main/config.json"


def _normalize_deepspeed_path(ds_val: str) -> str:
    if not isinstance(ds_val, str):
        return ds_val
    mapping = {
        "dcft/train/zero3.json": "dcft/train/llamafactory/examples/deepspeed/ds_z3_config.json",
        "dcft/train/zero3_offload.json": "dcft/train/llamafactory/examples/deepspeed/ds_z3_offload_config.json",
        "dcft/train/zero2.json": "dcft/train/llamafactory/examples/deepspeed/ds_z2_config.json",
    }
    if ds_val in mapping:
        return mapping[ds_val]
    for old, new in mapping.items():
        if ds_val.endswith(old) or old in ds_val:
            return ds_val.replace(old, new)
    return ds_val


def ensure_deepspeed_config(base_config: dict, exp_args: dict) -> dict:
    """Ensure DeepSpeed settings exist and point at canonical config paths."""

    default_ds = LlamaFactoryArgs.__dataclass_fields__["deepspeed"].default
    if not base_config.get("deepspeed"):
        base_config["deepspeed"] = exp_args.get("deepspeed", default_ds) or default_ds

    if isinstance(base_config.get("deepspeed"), str):
        normalized = _normalize_deepspeed_path(base_config["deepspeed"])
        if normalized != base_config["deepspeed"]:
            print(f"Normalized deepspeed path: {base_config['deepspeed']} -> {normalized}")
            base_config["deepspeed"] = normalized
    return base_config


def maybe_compute_gradient_accumulation(base_config: dict, exp_args: dict) -> dict:
    num_nodes = int(exp_args.get("num_nodes"))
    num_gpus = int(exp_args.get("gpus_per_node"))
    raw_global_batch_size = exp_args.pop("global_batch_size", None)
    if raw_global_batch_size is None:
        raw_global_batch_size = base_config.pop("global_batch_size", None)
    else:
        base_config.pop("global_batch_size", None)

    if raw_global_batch_size is None:
        print("\nSkipping automatic gradient accumulation calculation because global_batch_size was not provided.")
        return base_config

    try:
        global_batch_size = int(raw_global_batch_size)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"Expected integer-like global_batch_size, got {raw_global_batch_size!r}"
        ) from exc

    total_gpu_count = num_nodes * num_gpus

    def _int_config_value(key: str, default: int = 1) -> int:
        raw_value = base_config.get(key, default)
        try:
            return int(raw_value)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"Expected integer-like value for {key}, got {raw_value!r}"
            ) from exc

    tensor_model_parallel_size = _int_config_value("tensor_model_parallel_size", 1)
    pipeline_model_parallel_size = _int_config_value("pipeline_model_parallel_size", 1)
    expert_model_parallel_size = _int_config_value("expert_model_parallel_size", 1)

    model_parallel_world_size = (
        tensor_model_parallel_size
        * pipeline_model_parallel_size
        * expert_model_parallel_size
    )
    if model_parallel_world_size <= 0:
        raise ValueError(
            f"Model parallel world size must be positive; got {model_parallel_world_size}"
        )

    if total_gpu_count % model_parallel_world_size != 0:
        print(
            f"Warning: total GPU count ({total_gpu_count}) is not divisible by model parallel size "
            f"({model_parallel_world_size}). Rounding down data parallel replicas."
        )
    data_parallel_replicas = max(total_gpu_count // model_parallel_world_size, 1)

    per_device_train_batch_size = base_config.get("per_device_train_batch_size", 1)
    try:
        per_device_train_batch_size = max(int(per_device_train_batch_size), 1)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"Expected integer-like per_device_train_batch_size, got {per_device_train_batch_size!r}"
        ) from exc

    effective_batch_denom = per_device_train_batch_size * data_parallel_replicas
    if effective_batch_denom == 0:
        raise ValueError("Effective batch denominator resolved to zero.")

    gradient_accumulation_steps = global_batch_size // effective_batch_denom
    if gradient_accumulation_steps == 0 or (
        gradient_accumulation_steps * effective_batch_denom != global_batch_size
    ):
        raise ValueError(
            "Global batch size is not divisible by per-device batch * data-parallel replicas. "
            f"global_batch_size={global_batch_size}, per_device_train_batch_size={per_device_train_batch_size}, "
            f"data_parallel_replicas={data_parallel_replicas}"
        )

    base_config["gradient_accumulation_steps"] = gradient_accumulation_steps
    base_config["per_device_train_batch_size"] = per_device_train_batch_size
    print(f"\nCalculated based on {num_nodes} nodes, {num_gpus} GPUs per node, and global batch size {global_batch_size}:")
    print(f"data_parallel_replicas: {data_parallel_replicas}")
    print(f"per_device_train_batch_size: {per_device_train_batch_size}")
    print(f"gradient_accumulation_steps: {gradient_accumulation_steps}")
    return base_config


def apply_data_argument_overrides(base_config: dict, exp_args: dict) -> None:
    for tag in DATA_ARGUMENT_KEYS:
        if tag in exp_args:
            tag_value = exp_args[tag]
            if tag_value is not None:
                base_config[tag] = tag_value


def maybe_apply_cluster_specific_env_overrides(exp_args: dict, hpc) -> dict:
    """Inject cluster-specific defaults into exp_args when the user hasn't set them."""

    if hpc is None:
        return exp_args

    hpc_name = str(getattr(hpc, "name", "") or "").lower()
    explicit_cli_keys = set(exp_args.get("_explicit_cli_keys", []) or [])

    def _set_default(key: str, value):
        if key in explicit_cli_keys:
            return
        if exp_args.get(key) is None:
            exp_args[key] = value

    if hpc_name == "capella":
        _set_default("data_shared_file_system", True)

    return exp_args


# =============================================================================
# Universal SFT Job Runner (Phase 2 refactoring)
# =============================================================================


@dataclass
class SFTJobConfig:
    """Configuration for an SFT training job (serialized to JSON for sbatch)."""

    job_name: str
    train_config_path: str
    experiments_dir: str
    cluster_name: str

    # Resource allocation
    num_nodes: int = 1
    gpus_per_node: int = 1
    cpus_per_node: int = 24

    # Training launcher: "torchrun" or "accelerate"
    launcher: str = "torchrun"

    # Accelerate config (if launcher == "accelerate")
    accelerate_config_path: Optional[str] = None

    # DeepSpeed config path
    deepspeed_config: Optional[str] = None

    # Networking
    master_port: int = 12802

    # SSH tunneling (JSC clusters)
    needs_ssh_tunnel: bool = False

    # CUDA path detection (Perlmutter)
    needs_cuda_detection: bool = False


class SFTJobRunner:
    """Runs SFT training jobs with proper distributed setup.

    This class encapsulates the SFT training logic that was previously
    spread across multiple cluster-specific sbatch scripts.

    Usage (from sbatch):
        python -m hpc.sft_launch_utils --config /path/to/config.json
    """

    def __init__(self, config: SFTJobConfig):
        self.config = config
        self._hpc = None

    def _get_hpc(self):
        """Lazy-load HPC configuration."""
        if self._hpc is None:
            from hpc.hpc import detect_hpc, clusters

            if self.config.cluster_name:
                # Find by name
                for c in clusters:
                    if c.name.lower() == self.config.cluster_name.lower():
                        self._hpc = c
                        break
                if self._hpc is None:
                    raise ValueError(f"Unknown cluster: {self.config.cluster_name}")
            else:
                self._hpc = detect_hpc()
        return self._hpc

    def run(self) -> int:
        """Execute the SFT training job.

        Returns:
            Exit code (0 for success)
        """
        print(f"=== SFTJobRunner: {self.config.job_name} ===")

        try:
            self._setup_environment()

            if self.config.launcher == "torchrun":
                exit_code = self._run_torchrun()
            else:
                exit_code = self._run_accelerate()

            if exit_code == 0:
                print(f"SFT job '{self.config.job_name}' completed successfully")
            else:
                print(f"SFT job '{self.config.job_name}' failed with code {exit_code}")

            return exit_code

        except Exception as e:
            print(f"SFT job failed with exception: {e}", file=sys.stderr)
            raise

    def _setup_environment(self):
        """Set up NCCL and other environment variables."""
        hpc = self._get_hpc()

        # Apply NCCL settings from HPC config
        for key, value in hpc.nccl_settings.items():
            os.environ[key] = value
            print(f"[env] {key}={value}")

        # Apply CUDA environment detection (Perlmutter, etc.)
        if self.config.needs_cuda_detection or hpc.needs_cuda_detection:
            from hpc.cuda_utils import setup_cuda_environment

            cuda_env = setup_cuda_environment()
            for key, value in cuda_env.items():
                os.environ[key] = value
                print(f"[cuda] {key}={value}")

    def _run_torchrun(self) -> int:
        """Launch training with torchrun."""
        # Get distributed training parameters from environment
        num_nodes = int(os.environ.get("NUM_NODES", self.config.num_nodes))
        gpus_per_node = int(os.environ.get("NUM_GPUS_PER_NODE", self.config.gpus_per_node))
        master_addr = os.environ.get("MASTER_ADDR", "localhost")
        master_port = os.environ.get("MASTER_PORT", str(self.config.master_port))
        slurm_job_id = os.environ.get("SLURM_JOB_ID", "0")

        cmd = [
            "torchrun",
            f"--nproc-per-node={gpus_per_node}",
            f"--nnodes={num_nodes}",
            f"--rdzv_id={slurm_job_id}",
            "--rdzv_backend=c10d",
            f"--rdzv_endpoint={master_addr}:{master_port}",
            "sft/llamafactory/src/train.py",
            self.config.train_config_path,
        ]

        print(f"Running torchrun command: {' '.join(cmd)}")
        sys.stdout.flush()

        return subprocess.call(cmd)

    def _run_accelerate(self) -> int:
        """Launch training with accelerate."""
        # Get distributed training parameters from environment
        num_nodes = int(os.environ.get("NUM_NODES", self.config.num_nodes))
        gpus_per_node = int(os.environ.get("NUM_GPUS_PER_NODE", self.config.gpus_per_node))
        master_addr = os.environ.get("MASTER_ADDR", "localhost")
        master_port = os.environ.get("MASTER_PORT", str(self.config.master_port))
        slurm_procid = os.environ.get("SLURM_PROCID", "0")

        # Build accelerate config if not provided
        accelerate_config = self.config.accelerate_config_path
        if not accelerate_config:
            accelerate_config = self._generate_accelerate_config(num_nodes, gpus_per_node)

        cmd = [
            "python", "-u", "-m", "accelerate.commands.launch",
            f"--rdzv_conf=rdzv_backend=c10d,rdzv_endpoint={master_addr}:{master_port}",
            f"--config_file={accelerate_config}",
            f"--main_process_ip={master_addr}",
            f"--main_process_port={master_port}",
            f"--machine_rank={slurm_procid}",
            "--tee=3",
            "sft/llamafactory/src/train.py",
            self.config.train_config_path,
        ]

        print(f"Running accelerate command: {' '.join(cmd)}")
        sys.stdout.flush()

        return subprocess.call(cmd)

    def _generate_accelerate_config(self, num_nodes: int, gpus_per_node: int) -> str:
        """Generate an accelerate config file for distributed training."""
        config_dir = Path(self.config.experiments_dir) / "accelerate_configs"
        config_dir.mkdir(parents=True, exist_ok=True)

        config_path = config_dir / f"{self.config.job_name}_accelerate.yaml"

        # Basic accelerate config for multi-node training
        config = {
            "compute_environment": "LOCAL_MACHINE",
            "distributed_type": "FSDP" if self.config.deepspeed_config is None else "DEEPSPEED",
            "downcast_bf16": "no",
            "enable_cpu_affinity": False,
            "machine_rank": 0,
            "main_training_function": "main",
            "mixed_precision": "bf16",
            "num_machines": num_nodes,
            "num_processes": num_nodes * gpus_per_node,
            "rdzv_backend": "c10d",
            "same_network": True,
            "tpu_env": [],
            "tpu_use_cluster": False,
            "tpu_use_sudo": False,
            "use_cpu": False,
        }

        if self.config.deepspeed_config:
            config["deepspeed_config"] = {
                "deepspeed_config_file": self.config.deepspeed_config,
                "zero3_init_flag": True,
            }
        else:
            # FSDP config
            config["fsdp_config"] = {
                "fsdp_auto_wrap_policy": "TRANSFORMER_BASED_WRAP",
                "fsdp_backward_prefetch": "BACKWARD_PRE",
                "fsdp_cpu_ram_efficient_loading": True,
                "fsdp_forward_prefetch": False,
                "fsdp_offload_params": False,
                "fsdp_sharding_strategy": "FULL_SHARD",
                "fsdp_state_dict_type": "SHARDED_STATE_DICT",
                "fsdp_sync_module_states": True,
                "fsdp_use_orig_params": True,
            }

        with open(config_path, "w") as f:
            yaml.dump(config, f, default_flow_style=False)

        print(f"Generated accelerate config: {config_path}")
        return str(config_path)


def construct_sft_sbatch_script(exp_args: dict, hpc) -> str:
    """Construct SFT sbatch script using the new universal template system.

    This is a drop-in replacement for construct_sbatch_script() for SFT jobs.
    It creates the sbatch script and returns the path, letting the caller
    handle job submission (including dependencies and max_restarts).

    Args:
        exp_args: Experiment arguments dictionary
        hpc: HPC cluster configuration

    Returns:
        Path to the generated sbatch script
    """
    print("\n=== SFT MODE (Universal Launcher) ===")

    # Resolve paths
    exp_paths = setup_experiments_dir(exp_args)
    experiments_subdir = str(exp_paths.root)

    job_name = exp_args.get("job_name")
    if not job_name:
        raise ValueError("SFT jobs require a --job_name.")

    # Extract config values
    train_config_path = exp_args.get("train_config_path_out")
    if not train_config_path:
        raise ValueError("SFT jobs require a train config path.")

    num_nodes = int(exp_args.get("num_nodes") or 1)
    gpus_per_node = int(exp_args.get("gpus_per_node") or hpc.gpus_per_node)
    cpus_per_node = int(exp_args.get("cpus_per_node") or hpc.cpus_per_node)

    # Build SFTJobConfig
    job_config = SFTJobConfig(
        job_name=job_name,
        train_config_path=train_config_path,
        experiments_dir=experiments_subdir,
        cluster_name=hpc.name,
        num_nodes=num_nodes,
        gpus_per_node=gpus_per_node,
        cpus_per_node=cpus_per_node,
        launcher=hpc.training_launcher,
        deepspeed_config=exp_args.get("deepspeed"),
        needs_ssh_tunnel=hpc.needs_ssh_tunnel,
        needs_cuda_detection=hpc.needs_cuda_detection,
        master_port=int(exp_args.get("master_port") or 12802),
    )

    # Write config JSON
    config_dir = exp_paths.configs if hasattr(exp_paths, "configs") else exp_paths.root / "configs"
    config_dir.mkdir(parents=True, exist_ok=True)
    config_path = config_dir / f"{job_name}_sft_config.json"
    config_path.write_text(json.dumps(asdict(job_config), indent=2))
    print(f"Wrote SFT job config to {config_path}")

    # Load and populate universal template
    template_path = Path(__file__).parent / "sbatch_sft" / "universal_sft.sbatch"
    template_text = template_path.read_text()

    # Build cluster-specific SBATCH directives
    sbatch_directives = build_sbatch_directives(hpc, exp_args)

    # Generate CUDA setup code
    cuda_setup = ""
    if hpc.needs_cuda_detection:
        cuda_setup = """# CUDA path detection (handled by Python runner)
# Additional CUDA setup can be done in SFTJobRunner._setup_environment()"""

    # Generate srun command based on launcher
    if hpc.needs_ssh_tunnel:
        # JSC clusters use proxychains4 for internet access
        srun_command = f'srun $PROXY_CMD python -m hpc.sft_launch_utils --config "{config_path}"'
    else:
        srun_command = f'srun python -m hpc.sft_launch_utils --config "{config_path}"'

    substitutions = {
        "time_limit": exp_args.get("time_limit") or "24:00:00",
        "num_nodes": str(num_nodes),
        "cpus_per_node": str(cpus_per_node),
        "experiments_dir": experiments_subdir,
        "job_name": job_name,
        "sbatch_extra_directives": "\n".join(sbatch_directives),
        "module_commands": hpc.get_module_commands(),
        "conda_activate": hpc.conda_activate or "# No conda activation configured",
        "cluster_env_file": hpc.dotenv_filename,
        "cuda_setup": cuda_setup,
        "nccl_exports": hpc.get_nccl_exports(),
        "ssh_tunnel_setup": hpc.get_ssh_tunnel_setup(),
        "master_port": str(job_config.master_port),
        "gpus_per_node": str(gpus_per_node),
        "config_path": str(config_path),
        "srun_command": srun_command,
    }

    sbatch_text = substitute_template(template_text, substitutions)

    # Write sbatch script
    sbatch_dir = exp_paths.sbatch if hasattr(exp_paths, "sbatch") else exp_paths.root / "sbatch_scripts"
    sbatch_dir.mkdir(parents=True, exist_ok=True)
    sbatch_output = sbatch_dir / f"{job_name}_sft.sbatch"
    sbatch_output.write_text(sbatch_text)
    os.chmod(sbatch_output, 0o750)
    print(f"Wrote SFT sbatch script to {sbatch_output}")

    return str(sbatch_output)


def run_sft_job_main():
    """Entry point for running SFT jobs from sbatch."""
    import argparse

    parser = argparse.ArgumentParser(description="Run SFT training job")
    parser.add_argument("--config", required=True, help="Path to JSON config file")
    args = parser.parse_args()

    with open(args.config) as f:
        config_dict = json.load(f)

    config = SFTJobConfig(**config_dict)
    runner = SFTJobRunner(config)
    sys.exit(runner.run())


if __name__ == "__main__":
    run_sft_job_main()
