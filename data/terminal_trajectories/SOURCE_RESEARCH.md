# Source research notes: LiteCoder-Terminal and Terminal-Universe

Findings from a close read of the two papers behind our terminal-task sources,
2026-09-23/24. These notes correct earlier secondhand summaries; where a number
here conflicts with an earlier conversation or note, trust this file.

## LiteCoder-Terminal ([arXiv 2605.29559](https://arxiv.org/abs/2605.29559), [release blog](https://huggingface.co/blog/Lite-Coder/releasing-litecoder-terminal))

- The 602 RL environments are fully synthetic, produced by the
  LiteCoder-Terminal-Gen pipeline (Claude Agent SDK): instruction refinement ->
  Dockerfile/artifacts -> reference solution -> verifier -> `task.toml`.
- **All headline results are SFT-only** (11,255 teacher trajectories, mostly
  MiniMax teachers, Terminus-2/OpenHands/Claude Code scaffolds). The 32B SFT
  model reaches 29.06 / 18.54 / 34.00 pass@1 on Terminal Bench 1.0 / 2.0 / Pro.
- The RL environments were used for exactly one experiment: **DMPO (offline
  preference optimization) on the 4B SFT model only**. Per environment they
  sampled two rollouts, scored them by verifier pass ratio, and kept only
  environments whose two rollouts diverged. Result: TB 2.0 4.78% -> 6.10%,
  TB Pro 21.50% -> 23.00% pass@1. No online RL (PPO/GRPO) was run, and no
  DMPO at 30B scale.
- **Implication for us:** the divergent-pair filter silently discards broken
  environments (two zero-score rollouts produce no pair), so "DMPO helped" is
  not evidence that the 602 environments are individually healthy. Our dynamic
  pilot (`outputs/litecoder-terminal-audit/dynamic-pilot-2026-09-23.md`)
  confirms they are not: 64% of a stratified 25-task sample fails oracle.
- Provenance: LiteCoder tasks carry the terminal-bench canary header and are
  derived from terminal-bench-style task synthesis. Keep them out of any
  Terminal Bench / TerminalWorld evaluation seeds.
- Their teacher pass@1 -> pass@4 gaps (e.g. 16.56% -> 28.75% on TB 1.0 for
  Qwen3-30B-A3B-Instruct) support using pass@k growth as an RL-suitability
  proxy in our own pilot. Note Qwen3-30B-A3B-Instruct (their baseline) is not
  Qwen3-Coder-30B-A3B-Instruct (our proposed pilot model).

## Terminal-Universe ([arXiv 2609.04148](https://arxiv.org/abs/2609.04148), Qwen team + Tsinghua, 2026-09-03)

### Release status

Still a **release blocker** as of 2026-09-24: the paper publishes method and
aggregate results only. No reconstruction code, no reconstructed environments,
no generated tasks, no corpus, no weights, and no future-release statement in
the text. The `QwenLM/Qwen-AgentWorld` GitHub repo sometimes cited alongside it
is a different project (language world models). No compute or API cost figures
are reported anywhere in the paper.

### The training-data loop (corrected understanding)

The SFT corpus is **teacher re-solutions inside reconstructed environments**,
not the source trajectories:

1. Reconstruct an environment from a trajectory (replay + completion + judge).
2. Synthesize a new task on it (Intent Recovery / Single-WS / Cross-WS /
   Multi-Round variants).
3. An agent authors a pytest verifier **inside the target container** and
   accepts it only if, on the initial unmodified workspace, at least one
   new-capability test fails, preservation tests pass, and there are zero
   collection/infra errors. No oracle solution is run (contrast: our dynamic
   gate runs the oracle and the nop; theirs checks the initial state only).
4. The teacher rolls out a solution (Claude Code scaffold, <=500 turns, 4 h).
5. Verifier filters; survivor trajectories become the SFT corpus
   (31,977 records, ~1.42B tokens, student Qwen3.5-27B).

A single footnote assigns **Qwen3.7-Max** to every model-driven component.

### Why not SFT on the raw trajectories (their central empirical argument)

Same 35.8k tasks, same chat template (their Table 4):

| Data | TB 2.1 avg |
|---|---|
| SFT on raw source trajectories | **36.7** (below the 47.0 base model) |
| Intent-Recovery re-solving in reconstructed envs | 52.1 |
| Base model | 47.0 |

Raw-trajectory SFT actively hurts because it imitates weaker, inconsistent,
unverifiable policies. **Correction:** the widely-quoted "52.9 vs 48.7" pair is
the *completion ablation* (replay+completion vs replay-only at matched volume),
not re-solving vs replay. Replay-only still beats base (48.7 vs 46.2) because
teachers often repair the workspace before solving.

Other load-bearing ablations: doubling the corpus by adding **environments**
helps (53.2 -> 56.0); doubling queries-per-env or solutions-per-query does not
(Table 10) — environments are the scarce resource. Verifier filtering is
roughly neutral on easy single-workspace tasks but matters on harder slices
(Cross-WS 53.2 -> 55.4 at half the data; Multi-Round round-level filtering
+2.2 MT@4).

### Reconstruction mechanics (what the paper does and does not specify)

- Replay parses only **read / write / edit** operations into an ordered event
  stream; commands are not replayed. Each pre-existing file is restored to its
  **earliest observed content**; agent-created files are excluded (workspace
  starts unsolved); truncated observations stay truncated and are left to the
  completion agent.
- **Gap the paper never discusses:** shell side effects (`echo > f`, `sed -i`,
  compilers writing binaries) are invisible to this mechanism unless a later
  read reveals the path. Combined with LFM2's trajectory format (Terminus-2
  episodes of JSON shell-command batches, no structured Read/Write/Edit calls),
  replay alone is weak: replay-only sufficiency is 40.2% (terminal) / 20.1%
  (SWE), rising to 93.5% / 77.1% only after agentic completion. **Completion is
  mandatory, not optional** — and its cost/quality is the pipeline's crux.
- "Agent edits recorded separately for later verification" is stated but no
  downstream consumer is shown; verifiers are behavioral pytest suites.
- The sufficiency judge is agentic (read-only shell/file tools, one JSON
  verdict), not a single LLM call.
- Batch/orchestration mechanics for the 68k-env reconstruction are entirely
  unspecified (one acknowledgement of a "CPU-based replay framework").

### Source trajectory corpora (their Table 12) — relevant to our audit order

359,593 trajectories -> 68,263 reconstructed envs -> ~40k judged -> 37,273
task-sufficient:

| Source | Trajectories | Envs | Post-completion sufficiency |
|---|---|---|---|
| LFM2-Terminal-SFT-Processed (`gyung`, HF, CC-BY-4.0) | 139,841 | 46,037 | 95.2% |
| SWE-smith | 95,851 | 8,476 | 74.6% |
| SWE-rebench | 67,074 | 6,118 | 77.9% |
| CoderForge | 32,964 | 3,978 | 57.4% |
| LiteCoder-Terminal (trajectories) | 19,711 | 2,725 | 64.2% |
| SWE-Gym | 4,152 | 929 | 40.0% |

Notes: LFM2-Terminal supplies two thirds of all environments with the highest
sufficiency — validating it as our next audit target. Their LiteCoder input is
the **trajectory** dataset, and their count (19,711) does not match the public
11,255-row SFT release — an unreconciled version gap to watch when we audit.
Because Terminal-Universe consumes LiteCoder trajectories, any future
Terminal-Universe-style reproduction on our side needs provenance and overlap
dedup against LiteCoder-derived tasks.

### What this means for our roadmap

1. LiteCoder: oracle-filter all 602 (cheap CPU sandboxes), keep the healthy
   subset; repair only if cheaper than synthesis.
2. LFM2-Terminal audit: measure how much pre-edit file content its Terminus-2
   shell-batch trajectories expose; expect replay to be insufficient on its own
   and budget for model-based completion if we go this route.
3. Our `data/terminal_trajectories` prototype already encodes two lessons from
   both papers: build-time verifier deps, fail-closed infra errors, and bounded
   environment profiles. Terminal-Universe's verifier red-check (new-capability
   test must fail on the initial workspace) is a cheap addition worth adopting
   alongside our oracle/nop gates.
