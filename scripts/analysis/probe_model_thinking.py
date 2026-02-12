#!/usr/bin/env python3
"""Probe a HuggingFace model with real environment prompts to inspect thinking output.

Loads a model via the HF transformers pipeline, prompts it with the first
user message from each of the first N rows of a trace dataset, saves the
prompts and raw model responses to a JSON file.

Usage:
    python -m scripts.analysis.probe_model_thinking \
        --model laion/GLM-4_7-swesmith-sandboxes-with_tests-oracle_verified_120s-maxeps-131k \
        --output probe_results.json

    # Custom dataset / number of prompts
    python -m scripts.analysis.probe_model_thinking \
        --model laion/GLM-4_7-swesmith-sandboxes-with_tests-oracle_verified_120s-maxeps-131k \
        --dataset DCAgent2/some-other-dataset \
        --num-prompts 10 \
        --max-new-tokens 2048 \
        --output results.json
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

DEFAULT_DATASET = (
    "DCAgent2/DCAgent_dev_set_v2_laion_exp_tas_timeout_multiplier_4_0_traces_20260211_064438"
)
DEFAULT_MODEL = (
    "laion/GLM-4_7-swesmith-sandboxes-with_tests-oracle_verified_120s-maxeps-131k"
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Probe a HF model with real environment prompts",
    )
    p.add_argument(
        "--model",
        default=DEFAULT_MODEL,
        help=f"HuggingFace model ID (default: {DEFAULT_MODEL})",
    )
    p.add_argument(
        "--dataset",
        default=DEFAULT_DATASET,
        help=f"HuggingFace dataset ID (default: {DEFAULT_DATASET})",
    )
    p.add_argument(
        "--num-prompts",
        type=int,
        default=5,
        help="Number of prompts to run (default: 5)",
    )
    p.add_argument(
        "--max-new-tokens",
        type=int,
        default=4096,
        help="Max new tokens to generate per prompt (default: 4096)",
    )
    p.add_argument(
        "--output",
        default="probe_results.json",
        help="Output JSON file path (default: probe_results.json)",
    )
    p.add_argument(
        "--conversations-column",
        default="conversations",
        help="Column name for conversations (default: conversations)",
    )
    p.add_argument(
        "--role-tag",
        default="role",
        help="Key for role in message dicts (default: role)",
    )
    p.add_argument(
        "--content-tag",
        default="content",
        help="Key for content in message dicts (default: content)",
    )
    p.add_argument(
        "--dtype",
        default="bfloat16",
        choices=["float16", "bfloat16", "float32", "auto"],
        help="Model dtype (default: bfloat16)",
    )
    return p.parse_args()


def extract_initial_prompts(
    dataset_name: str,
    num_prompts: int,
    conversations_col: str,
    role_tag: str,
    content_tag: str,
) -> list[dict]:
    """Extract the first user message from N evenly-spaced rows."""
    from datasets import load_dataset

    print(f"[probe] Loading dataset: {dataset_name}")
    ds = load_dataset(dataset_name, split="train")

    # Pick rows evenly spaced across the dataset (stride = len // num_prompts,
    # min 1) so we sample diverse tasks rather than just the first few.
    n = len(ds)
    stride = max(n // num_prompts, 1)
    indices = [i * stride for i in range(num_prompts) if i * stride < n]
    print(f"[probe] Dataset has {n} rows, sampling indices: {indices}")

    prompts = []
    for i in indices:
        row = ds[i]
        convs = row[conversations_col]
        # Find the first user message (the environment/system prompt)
        first_user_content = None
        for msg in convs:
            if msg[role_tag] == "user":
                first_user_content = msg[content_tag]
                break

        if first_user_content is None:
            print(f"  Row {i}: no user message found, skipping")
            continue

        task_id = row.get("task", row.get("trial_name", f"row_{i}"))
        prompts.append({
            "row_index": i,
            "task": task_id,
            "messages": [{"role": "user", "content": first_user_content}],
        })
        print(f"  Row {i} ({task_id}): prompt length {len(first_user_content)} chars")

    return prompts


def _resolve_device_and_dtype(dtype: str):
    """Pick the best available device and a compatible torch dtype."""
    import torch

    dtype_map = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
        "auto": "auto",
    }
    torch_dtype = dtype_map[dtype]

    if torch.cuda.is_available():
        return "auto", torch_dtype, "cuda"

    if torch.backends.mps.is_available():
        # MPS (Apple Metal) does not support bfloat16; fall back to float16.
        if torch_dtype == torch.bfloat16:
            print("[probe] MPS does not support bfloat16, using float16 instead")
            torch_dtype = torch.float16
        return "mps", torch_dtype, "mps"

    if torch_dtype == "auto":
        torch_dtype = torch.float32
    return "cpu", torch_dtype, "cpu"


def run_inference(
    model_name: str,
    prompts: list[dict],
    max_new_tokens: int,
    dtype: str,
) -> list[dict]:
    """Run inference on each prompt and return results."""
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM

    device_map, torch_dtype, device_label = _resolve_device_and_dtype(dtype)

    print(f"\n[probe] Loading model: {model_name} (dtype={dtype}, device={device_label})")
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch_dtype,
        device_map=device_map,
        trust_remote_code=True,
    )
    print(f"[probe] Model loaded on {model.device}")

    results = []
    for i, prompt_info in enumerate(prompts):
        messages = prompt_info["messages"]
        task = prompt_info["task"]

        print(f"\n[probe] Prompt {i + 1}/{len(prompts)} ({task})...")

        inputs = tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
        ).to(model.device)

        input_len = inputs["input_ids"].shape[-1]
        print(f"  Input tokens: {input_len}")

        t0 = time.time()
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
            )
        elapsed = time.time() - t0

        generated_ids = outputs[0][input_len:]
        response_text = tokenizer.decode(generated_ids, skip_special_tokens=False)
        output_len = len(generated_ids)

        print(f"  Output tokens: {output_len}, time: {elapsed:.1f}s")
        print(f"  Response preview: {repr(response_text[:200])}")

        results.append({
            "row_index": prompt_info["row_index"],
            "task": task,
            "input_tokens": input_len,
            "output_tokens": output_len,
            "generation_time_sec": round(elapsed, 2),
            "prompt_content": messages[0]["content"][:500] + "..."
                if len(messages[0]["content"]) > 500 else messages[0]["content"],
            "response": response_text,
        })

    return results


def main() -> None:
    args = parse_args()

    prompts = extract_initial_prompts(
        args.dataset,
        args.num_prompts,
        args.conversations_column,
        args.role_tag,
        args.content_tag,
    )

    if not prompts:
        print("[probe] No prompts extracted, exiting.")
        sys.exit(1)

    results = run_inference(
        args.model,
        prompts,
        args.max_new_tokens,
        args.dtype,
    )

    output = {
        "model": args.model,
        "dataset": args.dataset,
        "num_prompts": len(results),
        "max_new_tokens": args.max_new_tokens,
        "dtype": args.dtype,
        "results": results,
    }

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    print(f"\n[probe] Saved {len(results)} results to {output_path}")


if __name__ == "__main__":
    main()
