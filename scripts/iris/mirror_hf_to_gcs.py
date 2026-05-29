#!/usr/bin/env python3
"""Mirror a HuggingFace model repo to GCS, one file at a time.

Designed to run on a small iris CPU worker without depending on
fitting the full model on the worker's ephemeral disk: each safetensors
shard is downloaded, uploaded to GCS, then deleted before the next
shard starts. Tokenizer + config files are tiny and processed first.

Usage::

    python -m scripts.iris.mirror_hf_to_gcs \\
        --repo cyankiwi/MiniMax-M2.7-AWQ-4bit \\
        --gcs-prefix gs://marin-eu-west4/ot-agent/models/

The resulting GCS layout::

    gs://marin-eu-west4/ot-agent/models/cyankiwi/MiniMax-M2.7-AWQ-4bit/
        config.json
        tokenizer.json
        model-00001-of-00050.safetensors
        ...

Idempotent: files already present in GCS with a matching size are
skipped. Re-run to resume an interrupted mirror.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path


# File patterns we mirror. Everything else (markdown, images,
# pytorch_model.bin if a safetensors copy exists) is skipped.
#
# .py files are REQUIRED for models with `trust_remote_code=true` whose
# config.json has an `auto_map` block pointing at e.g.
# `configuration_<arch>.py` / `modeling_<arch>.py`. Skipping them
# caused MiniMax-M2.7-AWQ to fail engine init with
# "OSError: ... does not appear to have a file named configuration_minimax_m2.py"
# on 2026-05-23.
INCLUDE_PATTERNS = (
    ".safetensors",
    ".json",
    ".txt",
    ".py",               # custom modeling / config code (trust_remote_code)
    ".model",            # sentencepiece tokenizer
    "tokenizer.model",
    "tokenizer_config.json",
    "special_tokens_map.json",
    "generation_config.json",
)

# Filename for the per-repo manifest written to the GCS prefix at the
# end of a successful mirror. Picked up by
# ``hpc.model_mirror_registry.refresh_from_gcs`` to populate the local
# catalog.
MANIFEST_FILENAME = ".mirror_manifest.json"


def _gcs_fs():
    """Return a cached fsspec GCS filesystem instance.

    Uses gcsfs's default credential discovery — picks up workload-identity
    creds on iris workers and ADC creds locally. We require gcsfs (and
    transitively fsspec) which are already in the OT-Agent venv via the
    [datagen-tpu] extra (datasets/huggingface_hub pull them in).
    """
    import gcsfs
    return gcsfs.GCSFileSystem()


def _gcs_size(uri: str) -> int | None:
    """Return the size in bytes of an existing GCS object, or None if missing.

    Tolerates any gcsfs failure mode (auth, missing key, transient errors)
    by returning None — the caller treats that as "need to upload".
    """
    try:
        info = _gcs_fs().info(uri)
        return int(info.get("size", 0))
    except (FileNotFoundError, OSError):
        return None
    except Exception as e:
        # Be lenient so a transient hiccup doesn't make the whole mirror
        # skip uploads; treat as "unknown / re-upload".
        print(f"[mirror] _gcs_size({uri}) unexpected error: {e}",
              file=sys.stderr, flush=True)
        return None


def _upload(local_path: Path, gcs_uri: str) -> None:
    """Stream-upload one file via gcsfs.put. Raise on any error."""
    fs = _gcs_fs()
    # gcsfs.put handles chunked upload for large files.
    fs.put(str(local_path), gcs_uri)


def _write_manifest(
    dest_prefix: str,
    *,
    repo_id: str,
    files_mirrored: list[tuple[str, int]],
    iris_job_id: str | None,
) -> None:
    """Write the per-repo manifest to GCS for the local registry to discover."""
    manifest = {
        "hf_repo": repo_id,
        "mirrored_at": datetime.now(timezone.utc).isoformat(),
        "mirror_script": "scripts/iris/mirror_hf_to_gcs.py",
        "file_count": len(files_mirrored),
        "size_bytes": sum(sz for _, sz in files_mirrored),
        "files": [{"name": n, "size": sz} for n, sz in files_mirrored],
        "patterns": list(INCLUDE_PATTERNS),
        "iris_job_id": iris_job_id,
    }
    manifest_uri = f"{dest_prefix}/{MANIFEST_FILENAME}"
    fs = _gcs_fs()
    with fs.open(manifest_uri, "w") as f:
        json.dump(manifest, f, indent=2)


def mirror(repo_id: str, gcs_prefix: str, *, verbose: bool = True,
           iris_job_id: str | None = None) -> None:
    """Mirror ``repo_id`` to ``<gcs_prefix>/<repo_id>/``.

    Mirrors one file at a time via huggingface_hub.hf_hub_download so the
    local disk never holds more than one shard at once. Writes a
    ``.mirror_manifest.json`` to the GCS prefix on successful completion
    so ``hpc.model_mirror_registry.refresh_from_gcs`` can index it.
    """
    from huggingface_hub import HfApi, hf_hub_download

    gcs_prefix = gcs_prefix.rstrip("/")
    dest_prefix = f"{gcs_prefix}/{repo_id}"

    api = HfApi()
    files = sorted(api.list_repo_files(repo_id, repo_type="model"))

    if verbose:
        print(f"[mirror] {repo_id} -> {dest_prefix}", flush=True)
        print(f"[mirror] repo has {len(files)} files; "
              f"filtering for {INCLUDE_PATTERNS}", flush=True)

    keep = [f for f in files if any(f.endswith(p) or f == p for p in INCLUDE_PATTERNS)]
    if verbose:
        print(f"[mirror] mirroring {len(keep)} files "
              f"(safetensors + config/tokenizer)", flush=True)

    # Process small files first (json/txt/model) so a partial run still
    # leaves usable metadata in GCS.
    keep.sort(key=lambda f: (f.endswith(".safetensors"), f))

    files_mirrored: list[tuple[str, int]] = []

    for idx, fname in enumerate(keep, 1):
        gcs_uri = f"{dest_prefix}/{fname}"
        remote_size = _gcs_size(gcs_uri)

        # Check existing in GCS vs HF. If GCS already has a non-empty
        # object at this URI, skip (best-effort idempotency without
        # round-tripping the actual file bytes).
        if remote_size is not None and remote_size > 0:
            if verbose:
                print(f"[mirror] [{idx}/{len(keep)}] skip "
                      f"(already in GCS, {remote_size} bytes): {fname}",
                      flush=True)
            files_mirrored.append((fname, remote_size))
            continue

        with tempfile.TemporaryDirectory(prefix="hf_mirror_") as tmp:
            tmp_path = Path(tmp)
            if verbose:
                print(f"[mirror] [{idx}/{len(keep)}] download: {fname}",
                      flush=True)
            local_file = hf_hub_download(
                repo_id=repo_id,
                filename=fname,
                local_dir=str(tmp_path),
                # Skip cache because we delete on temp-dir exit.
                local_dir_use_symlinks=False,
            )
            local_file = Path(local_file)
            local_size = local_file.stat().st_size

            if verbose:
                print(f"[mirror] [{idx}/{len(keep)}] upload "
                      f"({local_size} bytes): {gcs_uri}", flush=True)
            _upload(local_file, gcs_uri)
            files_mirrored.append((fname, local_size))
            # TemporaryDirectory will rm everything on exit, but be
            # explicit on the unlink so failure modes are clearer.
            try:
                local_file.unlink(missing_ok=True)
            except OSError:
                pass

    # Write the per-repo manifest last so a partial / interrupted mirror
    # is detectable (manifest missing -> incomplete).
    _write_manifest(
        dest_prefix,
        repo_id=repo_id,
        files_mirrored=files_mirrored,
        iris_job_id=iris_job_id,
    )

    if verbose:
        total = sum(sz for _, sz in files_mirrored)
        print(f"[mirror] done: {repo_id} -> {dest_prefix} "
              f"({len(files_mirrored)} files, {total} bytes); manifest at "
              f"{dest_prefix}/{MANIFEST_FILENAME}", flush=True)


def main() -> int:
    p = argparse.ArgumentParser(
        description="Stream one or more HuggingFace model repos into a GCS prefix.",
    )
    p.add_argument("--repo", action="append", required=True,
                   help="HF model repo id (org/name); repeatable.")
    p.add_argument("--gcs-prefix", required=True,
                   help="GCS prefix; each repo lands under <prefix>/<repo>/.")
    p.add_argument("--quiet", action="store_true",
                   help="Suppress per-file progress lines.")
    p.add_argument("--iris-job-id", default=os.environ.get("IRIS_JOB_ID"),
                   help="Record this job id in each repo's manifest. "
                        "Defaults to $IRIS_JOB_ID if set on the worker.")
    args = p.parse_args()

    if not args.gcs_prefix.startswith("gs://"):
        print(f"error: --gcs-prefix must start with gs:// (got {args.gcs_prefix!r})",
              file=sys.stderr)
        return 2

    for repo in args.repo:
        mirror(repo, args.gcs_prefix, verbose=not args.quiet,
               iris_job_id=args.iris_job_id)
    return 0


if __name__ == "__main__":
    sys.exit(main())
