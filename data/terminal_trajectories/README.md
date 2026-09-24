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

1. LiteCoder-Terminal-RL-preview: dynamic oracle + no-op checks ran on a
   stratified 25-task sample (2026-09-23, Modal sandboxes). Only 9/25 tasks pass
   both gates; all failures are task-side defects (missing image dependencies,
   dead external fetches, environment/solution desync), none are always-reward
   verifiers. The dataset is not RL-ready as published. Next: oracle-filter all
   602 tasks on CPU sandboxes, then run partial-solution and model rollout
   checks on the healthy subset only. See
   `outputs/litecoder-terminal-audit/dynamic-pilot-2026-09-23.md`.
2. Sample LFM2-Terminal trajectories and measure how often their terminal
   observations contain enough file evidence for deterministic reconstruction.
   LFM2 is the dominant Terminal-Universe input (139,841 trajectories -> 46,037
   environments, 95.2% post-completion sufficiency), but its Terminus-2
   shell-batch format has no structured Read/Write/Edit calls, so expect replay
   alone to be insufficient and budget for model-based completion.
3. Use SWE-Hero or Open-SWE-Traces only after linking each trajectory back to
   its upstream repository task and checking TaskTrove overlap. Their public
   trajectory rows are useful SFT material but are not standalone Harbor tasks.
4. Extend this prototype toward Terminal-Universe-style reconstruction after
   the source-specific evidence audit passes.

[Terminal-Universe](https://arxiv.org/abs/2609.04148) is a method reference, not
a directly importable source. As of 2026-09-24 the paper still releases no
pipeline code, reconstructed environments, generated tasks, corpus, or weights,
and reports no compute cost. Its replay recovers only read/write/edit evidence:
shell side effects (`echo > f`, `sed -i`, compiler outputs) are invisible to it,
and replay-only workspace sufficiency is ~40% (terminal) / ~20% (SWE), so
model-based environment completion is mandatory rather than optional. Its
training loop is teacher re-solving in reconstructed environments with verifier
filtering, then SFT — notably, its own ablation shows SFT on the raw source
trajectories scores *below* the base model (36.7 vs 47.0 on Terminal-Bench 2.1).
Terminal-Universe also uses LiteCoder-Terminal trajectories among its inputs, so
derived tasks require provenance tracking and overlap checks before inclusion.
See SOURCE_RESEARCH.md for the full analysis.
