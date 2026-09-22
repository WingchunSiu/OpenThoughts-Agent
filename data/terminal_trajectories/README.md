# Terminal trajectories to Harbor tasks

This pipeline synthesizes new Harbor tasks from independently sourced human
terminal recordings or agent trajectories. It adapts the four-stage method from
[TerminalWorld](https://github.com/EuniAI/TerminalWorld): normalize and audit a
trajectory, distill a reference workflow, write an outcome-oriented instruction,
and generate a state-based verifier.

It does not download or copy TerminalWorld benchmark tasks or source recordings.
Do not use TerminalWorld's published tasks, canaries, or underlying recording IDs
as training seeds when TerminalWorld is an evaluation target.

## Input

Two local input formats are supported:

1. A directory containing one subdirectory per recording, with
   `recording.txt` and an optional `info.json`.
2. JSONL with a required `trajectory_id` (or `id`) and one of:
   `transcript`, Harbor-style `steps`, or chat-style `messages`/`conversations`.

Every source must declare a license in `source_license`/`license`, or receive an
explicit `--source-license` fallback. Useful provenance fields are `source_url`,
`source_type`, and `metadata`.

Example JSONL row:

```json
{"trajectory_id":"rollout-001","source_type":"agent_trajectory","source_license":"Apache-2.0","steps":[{"source":"agent","message":"$ printf hello > /app/result.txt"},{"source":"environment","observation":{"exit_code":0}}]}
```

## Run

```bash
python -m data.terminal_trajectories.generate \
  --input /path/to/trajectories.jsonl \
  --source-type agent_trajectory \
  --source-license Apache-2.0 \
  --engine-config /path/to/datagen.yaml \
  --output-dir /path/to/output \
  --filter-mode annotate
```

`annotate` is the default: PII, credential, destructive-command, TUI, and
short-input findings are written to `manifest.jsonl` but do not change selection.
Use `--filter-mode enforce` only for an explicit filtering experiment.
Sensitive spans are replaced with typed placeholders before a retained record is
sent to the teacher; the original transcript is never written to the output.

The output contains:

```text
output/
├── manifest.jsonl
├── tasks.parquet
└── tasks/
    └── trajectory-.../
        ├── task.toml
        ├── instruction.md
        ├── environment/Dockerfile
        ├── solution/solve.sh
        └── tests/{test.sh,test_state.py}
```

The environment planner selects one of three fixed Dockerfiles (`base`, `node`,
or `system`), bounding a generated dataset to at most three Daytona snapshots.
Verifier dependencies are installed at image-build time. Runtime verifier setup
does not access the network, and pytest collection/infrastructure errors do not
silently become reward zero.

## Validation boundary

Generation performs current-Harbor static loading and packages TaskTrove's
`path`/`task_binary` Parquet schema. This is only the first gate. Before publishing
a dataset, run fresh-container oracle, no-op, and partial-solution trials. Keep
build/verifier infrastructure errors separate from model failures, then use model
pass rates for difficulty calibration.

## Source status and validation order

The current validation order is:

1. Run dynamic oracle, no-op, partial-solution, and model rollout checks on
   LiteCoder-Terminal-RL-preview. Its 602 tasks already have Harbor structure;
   the remaining questions concern reproducibility, verifier correctness, and
   model difficulty.
2. Sample LFM2-Terminal trajectories and measure how often their terminal
   observations contain enough file evidence for deterministic reconstruction.
3. Use SWE-Hero or Open-SWE-Traces only after linking each trajectory back to
   its upstream repository task and checking TaskTrove overlap. Their public
   trajectory rows are useful SFT material but are not standalone Harbor tasks.
4. Extend this prototype toward Terminal-Universe-style reconstruction after
   the source-specific evidence audit passes.

[Terminal-Universe](https://arxiv.org/abs/2609.04148) is a method reference, not
a directly importable source. Its published paper reports reconstructed
environments and generated tasks but does not currently release the pipeline,
reconstructed workspaces, or Harbor-style task packages. Reproduction therefore
requires source adapters, deterministic replay of file evidence, model-based
environment completion, a sufficiency judge, task and verifier synthesis, and
sandbox validation. Some trajectories cannot recover files that were never
observed. Terminal-Universe also uses LiteCoder-Terminal among its inputs, so
derived tasks require provenance tracking and overlap checks before inclusion.
