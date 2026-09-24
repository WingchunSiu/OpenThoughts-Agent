# LiteCoder Terminal audit

## Result

- RL source: `Lite-Coder/LiteCoder-Terminal-RL-preview@6fe7e994ff12d678de9b803da5c9907c8394a89c`
- SFT source: `Lite-Coder/LiteCoder-Terminal-SFT@6acdbbdb29979e4b8ea717b12accc8214606d087`
- RL tasks: 602 local / 602 manifest
- Harbor static loads: 602/602
- Missing required files: 0
- Unresolved Git LFS pointers: 0
- Duplicate RL instruction rows: 0
- SFT sample: 128 rows from 11255 total rows
- SFT/RL exact instruction matches: 0
- SFT/RL exact title matches: 3
- SFT/RL near-title candidates: 3
- TaskTrove: `open-thoughts/TaskTrove@946884702046be1a7dcea2638186ad6b0d2ea103`
- TaskTrove directly names LiteCoder: False
- TaskTrove exact task-ID/path matches in bounded scan: 0
- Packaged TaskTrove rows: 602 (56121286 bytes)
- Packaged archive errors: 0
- Packaged Parquet SHA-256: `0874b36325af5b247dfdda45c079e6bf2096cfb3dc4bc18c7de29ff56db6ef38`

## Import readiness

The source is already organized as Harbor tasks and can be packaged into the TaskTrove `path` + `task_binary` Parquet schema without task reconstruction. Static compatibility is necessary but does not establish verifier correctness.

All 602 task configs allow internet. 602 verifier scripts install or download dependencies at verification time. These are reproducibility and reward-availability risks to audit before publication.

602 task configs reference a mutable `:latest` Docker image tag. Pin or remove those image references before a stable TaskTrove release.

Dynamic oracle/no-op/partial-solution gates were not run by this static report. A first dynamic pilot followed on 2026-09-23 (`dynamic-pilot-2026-09-23.md`): 25 stratified tasks in fresh Modal containers, oracle + nop. Result: 9/25 PASS, 16/25 oracle failures (all task-side defects: missing image deps, dead external URLs, environment/solution desync), 0 always-reward verifiers, 0 infrastructure failures. The dataset is not RL-ready as published; run the full-602 oracle filter before any GPU pilot.

The packaged Parquet is a local generated artifact. Recreate it with `--package-parquet`; this report records its checksum.

## SFT/RL overlap candidates

| SFT id | Match | SFT title | RL task | Title score | Instruction score |
|---:|---|---|---|---:|---:|
| 3608 | near_title | Automated Backup System with Rotation and Monitoring | shell-backup-rotation-system | 95.00 | 32.61 |
| 5427 | exact_title | Custom DNS Server Setup with Web Dashboard | dns-server-flask-dashboard | 100.00 | 35.65 |
| 9225 | exact_title | User Audit and Expiration | audit-users-harden-ssh | 100.00 | 31.97 |
| 9953 | exact_title | End-to-End News Article Classification with BERT Fine-Tuning | bert-news-classification-pipeline | 100.00 | 39.02 |
| 11183 | near_title | OpenSSL Cross-Compilation for ARM Cortex-A53 | cross-compile-openssl-arm64 | 95.00 | 41.64 |
| 11186 | near_title | CPU-Optimized Image Classifier with Feature Visualization | cifar10-cnn-visualizer | 95.00 | 41.51 |

## Limits

- The SFT result is a deterministic byte-stratified sample, not a full 11,255-row scan.
- The TaskTrove path scan is exact for the listed scanned sources but skips large Parquets above the configured size cap. Absence of an exact ID is not proof that no semantically duplicate instruction exists.
- Container execution requires a running Docker daemon.
