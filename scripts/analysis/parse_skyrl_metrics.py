#!/usr/bin/env python3
"""
Parse SkyRL training metrics from console logs.

Scans log files for metric dictionary blocks and vLLM inference engine stats,
extracting them into:
- A CSV table with all metrics per step
- A CSV table with vLLM engine metrics (aggregated across engines)
- A markdown report with summary statistics

Usage:
    python parse_skyrl_metrics.py <log_folder> <output_folder>
    python parse_skyrl_metrics.py /path/to/logs /path/to/results
"""

import argparse
import json
import os
import re
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd


def strip_ansi(text: str) -> str:
    """Remove ANSI escape codes from text."""
    ansi_pattern = re.compile(r'\x1b\[[0-9;]*m')
    return ansi_pattern.sub('', text)


def extract_metrics_blocks(log_content: str) -> list[dict[str, Any]]:
    """
    Extract metric dictionary blocks from log content.

    Looks for blocks that start with {'async/staleness_max': and end with
    'trainer/global_step': N}
    """
    # Strip ANSI codes first
    content = strip_ansi(log_content)

    # Remove the Ray actor prefix from each line
    # Pattern: (skyrl_entrypoint pid=XXXXX) or similar
    lines = content.split('\n')
    cleaned_lines = []
    for line in lines:
        # Remove Ray actor prefix
        match = re.match(r'\([^)]+\)\s*(.*)', line)
        if match:
            cleaned_lines.append(match.group(1))
        else:
            cleaned_lines.append(line)

    content = '\n'.join(cleaned_lines)

    # Find all metric blocks
    # They start with {'async/staleness_max': and end with 'trainer/global_step': N}
    pattern = r"\{'async/staleness_max':[^}]+?'trainer/global_step':\s*\d+\}"

    metrics_list = []

    for match in re.finditer(pattern, content, re.DOTALL):
        block = match.group(0)

        # Parse the dictionary-like string
        metrics = parse_metrics_block(block)
        if metrics:
            metrics_list.append(metrics)

    return metrics_list


def parse_metrics_block(block: str) -> dict[str, Any] | None:
    """
    Parse a metrics block string into a dictionary.

    The block looks like:
    {'async/staleness_max': 0,
     'async/staleness_mean': '0.0000',
     ...
     'trainer/global_step': 1}
    """
    try:
        # Clean up the block for parsing
        # Replace single quotes with double quotes for JSON
        block = block.replace("'", '"')

        # Handle trailing commas (not valid JSON)
        block = re.sub(r',\s*}', '}', block)

        metrics = json.loads(block)

        # Convert string numbers to floats
        for key, value in metrics.items():
            if isinstance(value, str):
                try:
                    metrics[key] = float(value)
                except ValueError:
                    pass

        return metrics
    except json.JSONDecodeError as e:
        # Try alternative parsing
        try:
            # Use ast.literal_eval for Python dict syntax
            import ast
            metrics = ast.literal_eval(block.replace('"', "'"))

            # Convert string numbers to floats
            for key, value in metrics.items():
                if isinstance(value, str):
                    try:
                        metrics[key] = float(value)
                    except ValueError:
                        pass

            return metrics
        except Exception:
            print(f"Warning: Could not parse metrics block: {e}")
            return None


def extract_vllm_metrics(log_content: str) -> list[dict[str, Any]]:
    """
    Extract vLLM stat logger metrics from log content.

    Looks for lines like:
    (AsyncVLLMInferenceEngine pid=287294, ip=10.128.26.194) INFO 02-08 00:56:50 [loggers.py:248]
    Engine 000: Avg prompt throughput: 23.1 tokens/s, Avg generation throughput: 0.0 tokens/s,
    Running: 1 reqs, Waiting: 0 reqs, GPU KV cache usage: 0.2%, Prefix cache hit rate: 0.0%
    """
    # Strip ANSI codes first
    content = strip_ansi(log_content)

    # Pattern to match vLLM stat logger output
    # Captures: pid, ip, date, time, prompt_throughput, gen_throughput, running, waiting, kv_cache, prefix_cache
    pattern = re.compile(
        r'\(AsyncVLLMInferenceEngine pid=(\d+), ip=([^\)]+)\).*?'
        r'INFO (\d{2}-\d{2}) (\d{2}:\d{2}:\d{2}).*?'
        r'Engine \d+: '
        r'Avg prompt throughput: ([\d.]+) tokens/s, '
        r'Avg generation throughput: ([\d.]+) tokens/s, '
        r'Running: (\d+) reqs, '
        r'Waiting: (\d+) reqs, '
        r'GPU KV cache usage: ([\d.]+)%, '
        r'Prefix cache hit rate: ([\d.]+)%',
        re.MULTILINE
    )

    metrics_list = []
    for match in pattern.finditer(content):
        pid, ip, date, time_str, prompt_tp, gen_tp, running, waiting, kv_cache, prefix_cache = match.groups()

        metrics_list.append({
            'pid': int(pid),
            'ip': ip,
            'date': date,
            'time': time_str,
            'datetime_str': f"{date} {time_str}",
            'prompt_throughput_tokens_per_sec': float(prompt_tp),
            'generation_throughput_tokens_per_sec': float(gen_tp),
            'running_requests': int(running),
            'waiting_requests': int(waiting),
            'gpu_kv_cache_usage_pct': float(kv_cache),
            'prefix_cache_hit_rate_pct': float(prefix_cache),
        })

    return metrics_list


def aggregate_vllm_metrics(metrics: list[dict[str, Any]], window_seconds: int = 5) -> list[dict[str, Any]]:
    """
    Aggregate vLLM metrics across engines by time window.

    Each inference engine reports independently. This function groups metrics
    by timestamp and aggregates them.
    """
    if not metrics:
        return []

    # Group by datetime_str (already 1-second resolution)
    by_time = defaultdict(list)
    for m in metrics:
        by_time[m['datetime_str']].append(m)

    aggregated = []
    for time_str, engine_metrics in sorted(by_time.items()):
        n_engines = len(engine_metrics)

        # Aggregate metrics
        agg = {
            'datetime_str': time_str,
            'n_engines_reporting': n_engines,
            'unique_ips': len(set(m['ip'] for m in engine_metrics)),
            # Sum across engines
            'total_prompt_throughput_tokens_per_sec': sum(m['prompt_throughput_tokens_per_sec'] for m in engine_metrics),
            'total_generation_throughput_tokens_per_sec': sum(m['generation_throughput_tokens_per_sec'] for m in engine_metrics),
            'total_running_requests': sum(m['running_requests'] for m in engine_metrics),
            'total_waiting_requests': sum(m['waiting_requests'] for m in engine_metrics),
            # Average across engines
            'avg_prompt_throughput_per_engine': sum(m['prompt_throughput_tokens_per_sec'] for m in engine_metrics) / n_engines,
            'avg_generation_throughput_per_engine': sum(m['generation_throughput_tokens_per_sec'] for m in engine_metrics) / n_engines,
            'avg_running_requests_per_engine': sum(m['running_requests'] for m in engine_metrics) / n_engines,
            'avg_waiting_requests_per_engine': sum(m['waiting_requests'] for m in engine_metrics) / n_engines,
            'avg_gpu_kv_cache_usage_pct': sum(m['gpu_kv_cache_usage_pct'] for m in engine_metrics) / n_engines,
            'avg_prefix_cache_hit_rate_pct': sum(m['prefix_cache_hit_rate_pct'] for m in engine_metrics) / n_engines,
            # Min/Max for understanding variance
            'min_running_requests': min(m['running_requests'] for m in engine_metrics),
            'max_running_requests': max(m['running_requests'] for m in engine_metrics),
            'min_generation_throughput': min(m['generation_throughput_tokens_per_sec'] for m in engine_metrics),
            'max_generation_throughput': max(m['generation_throughput_tokens_per_sec'] for m in engine_metrics),
        }
        aggregated.append(agg)

    return aggregated


def generate_vllm_summary(vllm_metrics: list[dict[str, Any]], aggregated: list[dict[str, Any]]) -> dict[str, Any]:
    """Generate summary statistics for vLLM metrics."""
    if not aggregated:
        return {}

    summary = {
        'total_samples': len(vllm_metrics),
        'aggregated_time_points': len(aggregated),
        'avg_engines_reporting': sum(a['n_engines_reporting'] for a in aggregated) / len(aggregated),
        # Cluster-wide throughput
        'avg_total_prompt_throughput': sum(a['total_prompt_throughput_tokens_per_sec'] for a in aggregated) / len(aggregated),
        'avg_total_generation_throughput': sum(a['total_generation_throughput_tokens_per_sec'] for a in aggregated) / len(aggregated),
        'max_total_generation_throughput': max(a['total_generation_throughput_tokens_per_sec'] for a in aggregated),
        # Utilization indicators
        'avg_total_running_requests': sum(a['total_running_requests'] for a in aggregated) / len(aggregated),
        'avg_total_waiting_requests': sum(a['total_waiting_requests'] for a in aggregated) / len(aggregated),
        'max_total_running_requests': max(a['total_running_requests'] for a in aggregated),
        'max_total_waiting_requests': max(a['total_waiting_requests'] for a in aggregated),
        # Cache stats
        'avg_kv_cache_usage_pct': sum(a['avg_gpu_kv_cache_usage_pct'] for a in aggregated) / len(aggregated),
        'avg_prefix_cache_hit_rate_pct': sum(a['avg_prefix_cache_hit_rate_pct'] for a in aggregated) / len(aggregated),
        # Per-engine stats
        'avg_running_per_engine': sum(a['avg_running_requests_per_engine'] for a in aggregated) / len(aggregated),
        'avg_generation_throughput_per_engine': sum(a['avg_generation_throughput_per_engine'] for a in aggregated) / len(aggregated),
    }

    return summary


def process_log_file(log_path: Path) -> tuple[str, list[dict[str, Any]], list[dict[str, Any]]]:
    """Process a single log file and return its name, training metrics, and vLLM metrics."""
    with open(log_path, 'r', errors='replace') as f:
        content = f.read()

    metrics = extract_metrics_blocks(content)
    vllm_metrics = extract_vllm_metrics(content)

    # Extract a short name from the filename
    # e.g., "rl_rl-conf_qwen_8b_16GP_thin_bs64_grou_asyn_rloo_n_noct_stri_micr_auto_cons_v3_bala-yaml_mode-path__216747.out"
    # -> "v3_bala_216747"
    name = log_path.stem

    # Try to extract version and job ID
    version_match = re.search(r'_(v\d+_[a-z]+)', name)
    job_id_match = re.search(r'_(\d{6})\.', str(log_path))

    if version_match and job_id_match:
        short_name = f"{version_match.group(1)}_{job_id_match.group(1)}"
    elif job_id_match:
        short_name = f"job_{job_id_match.group(1)}"
    else:
        # Fallback: use last 30 chars
        short_name = name[-30:] if len(name) > 30 else name

    return short_name, metrics, vllm_metrics


def create_summary_statistics(df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Create summary statistics for each metric category."""
    summaries = {}

    # Group columns by category
    categories = defaultdict(list)
    for col in df.columns:
        if col in ['log_file', 'global_step']:
            continue
        if '/' in col:
            category = col.split('/')[0]
            categories[category].append(col)
        else:
            categories['other'].append(col)

    # Create summary for each category
    for category, columns in categories.items():
        if not columns:
            continue

        # Select only numeric columns
        numeric_cols = [c for c in columns if df[c].dtype in ['float64', 'int64']]
        if not numeric_cols:
            continue

        summary = df[numeric_cols].agg(['mean', 'std', 'min', 'max', 'count']).T
        summary.columns = ['Mean', 'Std', 'Min', 'Max', 'Count']
        summaries[category] = summary

    return summaries


def generate_markdown_report(
    all_data: dict[str, list[dict[str, Any]]],
    output_path: Path,
    df: pd.DataFrame,
    vllm_data: dict[str, dict[str, Any]] | None = None
) -> None:
    """Generate a markdown report with summary statistics."""

    with open(output_path, 'w') as f:
        f.write("# SkyRL Training Metrics Analysis\n\n")
        f.write(f"Generated from {len(all_data)} log files\n\n")

        # Overall summary
        f.write("## Overview\n\n")
        f.write("| Log File | Steps | Final Reward (mean) | Final Reward (max) | Total Time (s) |\n")
        f.write("|----------|-------|---------------------|-------------------|----------------|\n")

        for log_name, metrics in all_data.items():
            if not metrics:
                continue

            steps = len(metrics)
            rewards = [m.get('reward/avg_raw_reward', 0) for m in metrics]
            mean_reward = sum(rewards) / len(rewards) if rewards else 0
            max_reward = max(rewards) if rewards else 0
            total_time = sum(m.get('timing/step', 0) for m in metrics)

            f.write(f"| {log_name} | {steps} | {mean_reward:.4f} | {max_reward:.4f} | {total_time:.1f} |\n")

        f.write("\n")

        # Detailed statistics by category
        summaries = create_summary_statistics(df)

        for category, summary in summaries.items():
            f.write(f"## {category.title()} Metrics\n\n")
            f.write(summary.to_markdown())
            f.write("\n\n")

        # Per-log progression
        f.write("## Training Progression by Log\n\n")

        for log_name, metrics in all_data.items():
            if not metrics:
                continue

            f.write(f"### {log_name}\n\n")

            # Key metrics over time
            f.write("| Step | Reward | Pass@8 | KL | Loss | Step Time (s) | Gen Wait (s) |\n")
            f.write("|------|--------|--------|-----|------|---------------|-------------|\n")

            for m in metrics:
                step = m.get('trainer/global_step', 0)
                reward = m.get('reward/avg_raw_reward', 0)
                pass_at_8 = m.get('reward/avg_pass_at_8', 0)
                kl = m.get('policy/policy_kl', 0)
                loss = m.get('policy/final_loss', 0)
                step_time = m.get('timing/step', 0)
                gen_wait = m.get('timing/wait_for_generation_buffer', 0)

                f.write(f"| {step} | {reward:.4f} | {pass_at_8:.4f} | {kl:.6f} | {loss:.4f} | {step_time:.1f} | {gen_wait:.1f} |\n")

            f.write("\n")

        # Timing breakdown
        f.write("## Timing Analysis\n\n")

        timing_cols = [c for c in df.columns if c.startswith('timing/')]
        if timing_cols:
            timing_df = df[['log_file'] + timing_cols].copy()

            # Calculate percentages of step time
            if 'timing/step' in timing_df.columns:
                f.write("### Average Time Breakdown (% of step time)\n\n")

                breakdown = {}
                for col in timing_cols:
                    if col != 'timing/step':
                        avg_pct = (df[col] / df['timing/step'] * 100).mean()
                        breakdown[col.replace('timing/', '')] = avg_pct

                # Sort by percentage
                breakdown = dict(sorted(breakdown.items(), key=lambda x: x[1], reverse=True))

                f.write("| Component | Avg % of Step Time |\n")
                f.write("|-----------|-------------------|\n")
                for component, pct in breakdown.items():
                    f.write(f"| {component} | {pct:.1f}% |\n")

                f.write("\n")

        # Comparison across logs
        if len(all_data) > 1:
            f.write("## Cross-Log Comparison\n\n")

            comparison_metrics = [
                ('reward/avg_raw_reward', 'Avg Reward'),
                ('reward/avg_pass_at_8', 'Pass@8'),
                ('timing/step', 'Step Time (s)'),
                ('timing/wait_for_generation_buffer', 'Gen Wait Time (s)'),
                ('generate/avg_num_tokens', 'Avg Tokens'),
                ('async/staleness_mean', 'Staleness'),
            ]

            f.write("| Log | " + " | ".join(name for _, name in comparison_metrics) + " |\n")
            f.write("|-----|" + "|".join(["------" for _ in comparison_metrics]) + "|\n")

            for log_name, metrics in all_data.items():
                if not metrics:
                    continue

                row = [log_name]
                for metric_key, _ in comparison_metrics:
                    values = [m.get(metric_key, 0) for m in metrics]
                    mean_val = sum(values) / len(values) if values else 0
                    row.append(f"{mean_val:.4f}")

                f.write("| " + " | ".join(row) + " |\n")

            f.write("\n")

        # vLLM Inference Engine Analysis
        if vllm_data:
            f.write("## vLLM Inference Engine Analysis\n\n")
            f.write("Metrics from vLLM stat loggers (V1LoggingStatLoggerFixed).\n\n")
            f.write("> **Note**: Ray deduplicates similar log messages with `[repeated Nx across cluster]`,\n")
            f.write("> so we typically capture stats from one engine per timestamp. The stats shown are\n")
            f.write("> **per-engine** values. Multiply by num_inference_engines for cluster-wide estimates.\n\n")

            f.write("### Summary by Log (Per-Engine Stats)\n\n")
            f.write("| Log | Avg Running/Engine | Avg Waiting/Engine | Avg Gen Throughput/Engine | Avg KV Cache % | Avg Prefix Hit % |\n")
            f.write("|-----|-------------------|-------------------|--------------------------|----------------|------------------|\n")

            for log_name, data in vllm_data.items():
                summary = data.get('summary', {})
                if not summary:
                    continue

                f.write(f"| {log_name} ")
                f.write(f"| {summary.get('avg_running_per_engine', 0):.1f} ")
                f.write(f"| {summary.get('avg_total_waiting_requests', 0):.1f} ")
                f.write(f"| {summary.get('avg_generation_throughput_per_engine', 0):.1f} tok/s ")
                f.write(f"| {summary.get('avg_kv_cache_usage_pct', 0):.1f}% ")
                f.write(f"| {summary.get('avg_prefix_cache_hit_rate_pct', 0):.1f}% |\n")

            f.write("\n")

            # Utilization analysis
            f.write("### Utilization Analysis (Per-Engine)\n\n")
            f.write("Key indicators of inference engine utilization:\n\n")
            f.write("- **Running requests/engine**: Concurrent requests being processed by each engine\n")
            f.write("- **Waiting requests**: Requests queued (0 = engine not saturated, has spare capacity)\n")
            f.write("- **Generation throughput**: Decode tokens/sec per engine\n")
            f.write("  - 8B model on H100 can do **1000+ tok/s** when saturated\n")
            f.write("  - If seeing <300 tok/s with 0 waiting, engine is **starved for requests**\n\n")

            for log_name, data in vllm_data.items():
                summary = data.get('summary', {})
                if not summary:
                    continue

                f.write(f"#### {log_name}\n\n")

                avg_running = summary.get('avg_running_per_engine', 0)
                max_running = summary.get('max_total_running_requests', 0)
                avg_waiting = summary.get('avg_total_waiting_requests', 0)
                max_waiting = summary.get('max_total_waiting_requests', 0)
                avg_gen_tp = summary.get('avg_generation_throughput_per_engine', 0)
                max_gen_tp = summary.get('max_total_generation_throughput', 0)

                f.write(f"- **Running requests/engine**: avg={avg_running:.1f}, max={max_running}\n")
                f.write(f"- **Waiting requests**: avg={avg_waiting:.1f}, max={max_waiting}\n")
                f.write(f"- **Generation throughput/engine**: avg={avg_gen_tp:.1f} tok/s, max={max_gen_tp:.1f} tok/s\n")
                f.write(f"- **KV cache usage**: avg={summary.get('avg_kv_cache_usage_pct', 0):.1f}%\n")
                f.write(f"- **Prefix cache hit rate**: avg={summary.get('avg_prefix_cache_hit_rate_pct', 0):.1f}%\n")

                # Utilization assessment
                if avg_waiting == 0 and avg_running < 5:
                    f.write(f"- ⚠️ **Underutilized**: Engines starved for requests (0 waiting, avg {avg_running:.1f} running)\n")
                    f.write(f"  - Bottleneck is likely upstream (environment execution, not inference)\n")
                elif avg_waiting > 0:
                    f.write(f"- ✅ **Well-utilized**: Engines saturated (waiting > 0)\n")
                elif avg_gen_tp < 300:
                    f.write(f"- ⚠️ **Low throughput**: {avg_gen_tp:.0f} tok/s << expected 1000+ tok/s for saturated 8B model\n")
                else:
                    f.write(f"- ℹ️ **Moderate utilization**\n")

                f.write("\n")


def main():
    parser = argparse.ArgumentParser(
        description="Parse SkyRL training metrics from console logs"
    )
    parser.add_argument(
        "log_folder",
        type=str,
        help="Path to folder containing log files"
    )
    parser.add_argument(
        "output_folder",
        type=str,
        help="Path to output folder for results"
    )
    parser.add_argument(
        "--pattern",
        type=str,
        default="*.out",
        help="Glob pattern for log files (default: *.out)"
    )

    args = parser.parse_args()

    log_folder = Path(args.log_folder)
    output_folder = Path(args.output_folder)

    if not log_folder.exists():
        print(f"Error: Log folder does not exist: {log_folder}")
        sys.exit(1)

    # Create output folder
    output_folder.mkdir(parents=True, exist_ok=True)

    # Find all log files
    log_files = list(log_folder.glob(args.pattern))

    if not log_files:
        print(f"No log files matching '{args.pattern}' found in {log_folder}")
        sys.exit(1)

    print(f"Found {len(log_files)} log files")

    # Process each log file
    all_data = {}
    all_rows = []
    all_vllm_data = {}
    all_vllm_rows = []

    for log_path in sorted(log_files):
        print(f"Processing: {log_path.name}")
        log_name, metrics, vllm_metrics = process_log_file(log_path)

        if not metrics and not vllm_metrics:
            print(f"  Warning: No metrics found in {log_path.name}")
            continue

        if metrics:
            print(f"  Found {len(metrics)} training metric blocks")
            all_data[log_name] = metrics

            # Add to combined rows
            for m in metrics:
                row = {'log_file': log_name}
                row.update(m)
                all_rows.append(row)

        if vllm_metrics:
            print(f"  Found {len(vllm_metrics)} vLLM stat logger entries")
            aggregated = aggregate_vllm_metrics(vllm_metrics)
            summary = generate_vllm_summary(vllm_metrics, aggregated)

            all_vllm_data[log_name] = {
                'raw': vllm_metrics,
                'aggregated': aggregated,
                'summary': summary,
            }

            # Add aggregated to combined rows
            for a in aggregated:
                row = {'log_file': log_name}
                row.update(a)
                all_vllm_rows.append(row)

    if not all_rows and not all_vllm_rows:
        print("Error: No metrics found in any log files")
        sys.exit(1)

    # Create DataFrame for training metrics
    df = pd.DataFrame(all_rows) if all_rows else pd.DataFrame()

    # Rename trainer/global_step for easier access
    if not df.empty and 'trainer/global_step' in df.columns:
        df['global_step'] = df['trainer/global_step']

    # Save training metrics CSV
    if not df.empty:
        csv_path = output_folder / "metrics_table.csv"
        df.to_csv(csv_path, index=False)
        print(f"\nSaved training metrics table to: {csv_path}")

        # Save per-log CSVs
        for log_name, metrics in all_data.items():
            if metrics:
                log_df = pd.DataFrame(metrics)
                log_csv_path = output_folder / f"metrics_{log_name}.csv"
                log_df.to_csv(log_csv_path, index=False)
                print(f"Saved per-log training metrics to: {log_csv_path}")

    # Create DataFrame for vLLM metrics
    vllm_df = pd.DataFrame(all_vllm_rows) if all_vllm_rows else pd.DataFrame()

    # Save vLLM metrics CSV
    if not vllm_df.empty:
        vllm_csv_path = output_folder / "vllm_metrics_table.csv"
        vllm_df.to_csv(vllm_csv_path, index=False)
        print(f"\nSaved vLLM metrics table to: {vllm_csv_path}")

        # Save per-log vLLM CSVs
        for log_name, data in all_vllm_data.items():
            aggregated = data.get('aggregated', [])
            if aggregated:
                log_vllm_df = pd.DataFrame(aggregated)
                log_vllm_csv_path = output_folder / f"vllm_metrics_{log_name}.csv"
                log_vllm_df.to_csv(log_vllm_csv_path, index=False)
                print(f"Saved per-log vLLM metrics to: {log_vllm_csv_path}")

    # Generate markdown report
    md_path = output_folder / "metrics_report.md"
    generate_markdown_report(all_data, md_path, df, vllm_data=all_vllm_data if all_vllm_data else None)
    print(f"Saved markdown report to: {md_path}")

    # Print quick summary
    print("\n" + "=" * 60)
    print("QUICK SUMMARY")
    print("=" * 60)

    for log_name, metrics in all_data.items():
        if not metrics:
            continue

        steps = len(metrics)
        rewards = [m.get('reward/avg_raw_reward', 0) for m in metrics]
        final_reward = rewards[-1] if rewards else 0
        max_reward = max(rewards) if rewards else 0
        avg_step_time = sum(m.get('timing/step', 0) for m in metrics) / steps if steps else 0

        print(f"\n{log_name}:")
        print(f"  Steps: {steps}")
        print(f"  Final Reward: {final_reward:.4f}")
        print(f"  Max Reward: {max_reward:.4f}")
        print(f"  Avg Step Time: {avg_step_time:.1f}s")

        # Add vLLM summary if available
        if log_name in all_vllm_data:
            summary = all_vllm_data[log_name].get('summary', {})
            if summary:
                print(f"  vLLM (per-engine):")
                print(f"    Avg Running Reqs: {summary.get('avg_running_per_engine', 0):.1f}")
                print(f"    Avg Waiting Reqs: {summary.get('avg_total_waiting_requests', 0):.1f}")
                print(f"    Avg Gen Throughput: {summary.get('avg_generation_throughput_per_engine', 0):.1f} tok/s")
                print(f"    Avg Prefix Cache Hit: {summary.get('avg_prefix_cache_hit_rate_pct', 0):.1f}%")

    # Print vLLM-only summaries for logs that only have vLLM metrics
    for log_name, data in all_vllm_data.items():
        if log_name in all_data:
            continue  # Already printed above

        summary = data.get('summary', {})
        if summary:
            print(f"\n{log_name} (vLLM metrics only):")
            print(f"  vLLM (per-engine):")
            print(f"    Avg Running Reqs: {summary.get('avg_running_per_engine', 0):.1f}")
            print(f"    Avg Waiting Reqs: {summary.get('avg_total_waiting_requests', 0):.1f}")
            print(f"    Avg Gen Throughput: {summary.get('avg_generation_throughput_per_engine', 0):.1f} tok/s")
            print(f"    Avg Prefix Cache Hit: {summary.get('avg_prefix_cache_hit_rate_pct', 0):.1f}%")


if __name__ == "__main__":
    main()
