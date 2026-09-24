# LiteCoder-Terminal dynamic pilot (2026-09-23)

First dynamic gate for `Lite-Coder/LiteCoder-Terminal-RL-preview`, following the
static audit in this directory. The static report established that all 602 tasks
load through Harbor and package into TaskTrove Parquet; this pilot asks whether
the environments and verifiers actually work in fresh containers.

## Setup

- Host: Jupiter login node `jpbl-s01-01`, user `siu2`. No local Docker daemon;
  Apptainer unavailable (siu2 is not in the `container` group); no Daytona key.
  Trials ran on **Modal** sandboxes through Harbor's modal environment.
- Runtime: dedicated venv `/e/scratch/jureap59/siu2/venv/harbor-modal` with
  `harbor` 0.1.45 (installed from `/e/scratch/jureap59/feuer1/harbor`) and
  **modal 1.5.5**. Harbor's modal backend requires `modal>=1.4`
  (`Sandbox.filesystem`); the `otagent` env pins modal 1.3.2 and fails with
  `AttributeError: 'Sandbox' object has no attribute 'filesystem'`, which
  surfaces as `AddTestsDirError`. Do not use the otagent env for modal trials.
- Dataset: revision `6fe7e994ff12d678de9b803da5c9907c8394a89c`, snapshot at
  `/e/scratch/jureap59/siu2/hf_hub/datasets--Lite-Coder--LiteCoder-Terminal-RL-preview/snapshots/6fe7e994.../`.
- Run manifest, configs, per-trial logs:
  `/e/scratch/jureap59/siu2/litecoder_pilot/` (`run_manifest.json`, `jobs/`).

## Sample

25 of 602 tasks (`/e/scratch/jureap59/siu2/litecoder_pilot_sample.json`):
stratified by `task.toml` metadata (9 categories x easy/medium/hard), seeded by
sha256 with seed `litecoder-pilot-v1`, tasks needing >2 CPUs or >4 GB excluded
from the stratified picks. Forced inclusions: `dns-server-flask-dashboard` and
`audit-users-harden-ssh` (SFT/RL exact-title overlap candidates),
`rsa-key-reconstruct-decrypt__sample_0090_4784c0d7` (a `__sample_`
near-duplicate variant), `custom-kernel-boot-logo` (heavy-resource probe).

## Method

Per task, one fresh container from the declared `paraliine/<task>:latest` image:

```
harbor jobs start -p <tasks25 dir> -a oracle -e modal -n 8 -y --job-name litecoder-pilot25-oracle -o <jobs dir>
harbor jobs start -p <tasks25 dir> -a nop    -e modal -n 8 -y --job-name litecoder-pilot25-nop    -o <jobs dir>
```

- **oracle**: runs `solution/solve.sh`; a healthy task must score reward 1.
- **nop**: takes no action; a healthy verifier must score reward 0 (catches
  always-reward verifiers).
- `n_attempts=1`. Partial-solution checks were not run in this round.

## Results

| Verdict | Count |
|---|---|
| PASS (oracle = 1.0, nop = 0.0) | **9 / 25** |
| Task defect (oracle fails) | **16 / 25** |
| Verifier suspect (nop > 0) | **0** |
| Infrastructure failure (sandbox/build/upload) | **0** |

PASS: `audit-users-harden-ssh`, `clean-git-history-surgery`,
`cmake-calculator-build`, `crack-vault-encryption-flaw`,
`fix-recursive-make-build`, `flask-docker-nginx-ssl`,
`optimize-static-lib-linking`, `qt5-multi-version-build-matrix`,
`sales-analytics-dashboard`.

Oracle failure root causes (all read from per-trial agent/verifier logs):

| Cause | Tasks | Examples |
|---|---|---|
| Image missing Python deps the solution assumes | 4 | `numpy`, `pandas`, `pyzipper`, `Crypto` `ModuleNotFoundError` |
| PEP 668 blocks solution's `pip install` | 2 | `reddit-basket-analysis`, `network-topology-scanner` |
| External fetch failure | 2 | data URL returns HTTP 404 (`csv-stats-outlier-detector`); `git clone` needs GitHub auth (`git-bisect-bug-hunt`) |
| Missing binary / wrong arch in image | 4 | `file: command not found`; `Exec format error: /app/enigma` |
| Environment/solution state desync | 4 | `solve.sh` expects a `main` branch the image lacks; verifier references `../tests/test_outputs.py` from `/app/repo` (pytest exit 4); solutions that run but score 0 |

## Interpretation

- **The dataset is not RL-ready as published.** A stratified sample shows a 64%
  oracle failure rate. The failure modes match the static audit's flagged risks
  (mutable `:latest` images, verify-time dependency downloads, internet-reliant
  solutions), i.e. post-publication drift plus weak release-time validation.
- Failures are **task-side, not backend-side**: every sandbox started, every
  tests dir uploaded, every verifier executed. Daytona trials would face the
  same images from Docker Hub.
- The sample is biased toward light tasks; heavy ML-training tasks are
  underrepresented, so the full-corpus pass rate may be lower.
- No always-reward verifier was observed (all 25 nop trials scored 0.0).
- Consistent with the upstream paper ([arXiv 2605.29559](https://arxiv.org/abs/2605.29559)):
  its headline results come from SFT, and the RL environments were only used for
  a 4B DMPO experiment whose divergent-score pair filter **silently drops broken
  environments** (both rollouts score 0 -> no preference pair), so reported DMPO
  gains are not evidence of per-environment health.

## Next steps

1. Run the same oracle filter over all 602 tasks (CPU sandboxes only) to carve
   out the healthy subset as TaskTrove candidates; re-run nop on oracle-passers.
2. Only then a bounded GPU eval pilot (pass@1/pass@k) on the healthy subset.
3. Defect classes above are repairable (pin images, vendor inputs, fix
   solutions), but repair cost should be weighed against the
   terminal-trajectory synthesis route.
