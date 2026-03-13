"""
HF Storage Bucket Offload Pipeline

Migrates model artifacts from HF repos (hot storage) to HF Storage Buckets (cold storage),
with optional Supabase DB updates to track where files have moved.

Usage:
    python offload.py --config config.yaml                    # dry run by default
    python offload.py --config config.yaml --execute          # actually run
    python offload.py --config config.yaml --execute --ingress repo_id  # pull back from cold storage

Requirements:
    pip install huggingface_hub>=1.5.0 pyyaml supabase python-dotenv
"""

import argparse
import fnmatch
import json
import logging
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional

import yaml

try:
    from huggingface_hub import (
        HfApi,
        create_bucket,
        list_bucket_tree,
        batch_bucket_files,
        list_repo_tree,
    )
except ImportError:
    print("ERROR: huggingface_hub >= 1.5.0 required. Run: pip install huggingface_hub>=1.5.0")
    sys.exit(1)

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("offload")


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

@dataclass
class Config:
    source_org: str
    bucket_org: str
    bucket_name: str
    source_repos: list[str]
    include_patterns: list[str] = field(default_factory=lambda: ["*.safetensors", "*.bin", "*.pt"])
    exclude_patterns: list[str] = field(default_factory=lambda: ["config.json", "tokenizer*", "README.md"])
    keep_recent: int = 1
    skip_delete: bool = True
    dry_run: bool = True
    verbose: bool = True
    supabase_enabled: bool = False
    supabase_url: str = ""
    supabase_table: str = ""
    supabase_path_column: str = ""

    @property
    def bucket_id(self) -> str:
        return f"{self.bucket_org}/{self.bucket_name}"

    @property
    def bucket_handle(self) -> str:
        return f"hf://buckets/{self.bucket_id}"

    @classmethod
    def from_yaml(cls, path: str) -> "Config":
        with open(path) as f:
            data = yaml.safe_load(f)

        source_org = data.get("source_org") or data.get("org")
        bucket_org = data.get("bucket_org") or data.get("org")
        if not source_org or not bucket_org:
            raise ValueError("Config must include source_org and bucket_org (or legacy org)")

        sb = data.get("supabase", {})
        return cls(
            source_org=source_org,
            bucket_org=bucket_org,
            bucket_name=data["bucket_name"],
            source_repos=data["source_repos"],
            include_patterns=data.get("include_patterns", ["*.safetensors", "*.bin", "*.pt"]),
            exclude_patterns=data.get("exclude_patterns", ["config.json", "tokenizer*", "README.md"]),
            keep_recent=data.get("keep_recent", 1),
            skip_delete=data.get("skip_delete", True),
            dry_run=data.get("dry_run", True),
            verbose=data.get("verbose", True),
            supabase_enabled=sb.get("enabled", False),
            supabase_url=sb.get("url", ""),
            supabase_table=sb.get("table", ""),
            supabase_path_column=sb.get("path_column", ""),
        )


# ---------------------------------------------------------------------------
# Supabase helper 
# ---------------------------------------------------------------------------

class SupabaseClient:
    """Thin wrapper for updating model path references in Supabase."""

    def __init__(self, config: Config):
        self.enabled = config.supabase_enabled
        if not self.enabled:
            return

        if not config.supabase_url or not config.supabase_table or not config.supabase_path_column:
            logger.warning("Supabase config incomplete (url/table/path_column) — DB updates will be skipped")
            self.enabled = False
            return

        try:
            from supabase import create_client
            from dotenv import load_dotenv
            load_dotenv()
            key = os.environ.get("SUPABASE_KEY", "")
            if not key:
                logger.warning("SUPABASE_KEY not set — DB updates will be skipped")
                self.enabled = False
                return
            self.client = create_client(config.supabase_url, key)
            self.table = config.supabase_table
            self.path_col = config.supabase_path_column
        except ImportError:
            logger.warning("supabase-py not installed — DB updates will be skipped")
            self.enabled = False

    def update_path(self, old_path: str, new_path: str, dry_run: bool = True):
        """Update a model path reference from HF repo to bucket location."""
        if not self.enabled:
            return
        if dry_run:
            logger.info(f"  [DB DRY RUN] Would update: {old_path} -> {new_path}")
            return

        # TODO: Richard — adjust this to your actual metadata schema.
        # Placeholder behavior currently does a simple path replacement:
        #   old_path (repo_id/file) -> new_path (hf://buckets/...)
        # If your schema needs it, add extra logic here to archive/delete old metadata rows.
        try:
            self.client.table(self.table).update(
                {self.path_col: new_path}
            ).eq(self.path_col, old_path).execute()
            logger.info(f"  [DB] Updated: {old_path} -> {new_path}")
        except Exception as e:
            logger.error(f"  [DB ERROR] Failed to update {old_path}: {e}")


# ---------------------------------------------------------------------------
# Core logic
# ---------------------------------------------------------------------------

class OffloadPipeline:
    def __init__(self, config: Config):
        self.config = config
        self.api = HfApi()
        self.db = SupabaseClient(config)
        self.report: list[dict] = []

    def _matches(self, filename: str, patterns: list[str]) -> bool:
        """Check if a filename matches any of the glob patterns."""
        base = Path(filename).name
        return any(fnmatch.fnmatch(filename, p) or fnmatch.fnmatch(base, p) for p in patterns)

    def _resolve_repo_id(self, repo_name: str) -> str:
        """Accept either bare repo names or fully-qualified repo IDs."""
        return repo_name if "/" in repo_name else f"{self.config.source_org}/{repo_name}"

    def _cleanup_local_file(self, local_path: Optional[str], staging_root: Path):
        """Delete a staged file and any empty parent directories."""
        if not local_path:
            return

        file_path = Path(local_path)
        if file_path.exists():
            file_path.unlink()

        current = file_path.parent
        while current != staging_root and current.exists():
            try:
                current.rmdir()
            except OSError:
                break
            current = current.parent

    def _should_offload(self, filename: str) -> bool:
        """Determine if a file should be offloaded based on include/exclude patterns."""
        if self._matches(filename, self.config.exclude_patterns):
            return False
        if self._matches(filename, self.config.include_patterns):
            return True
        return False

    def ensure_bucket(self):
        """Create the target bucket if it doesn't exist."""
        bucket_id = self.config.bucket_id
        if self.config.dry_run:
            logger.info(f"[DRY RUN] Would ensure bucket exists: {bucket_id}")
            return
        try:
            create_bucket(bucket_id, private=True, exist_ok=True)
            logger.info(f"Bucket ready: {bucket_id}")
        except Exception as e:
            logger.error(f"Failed to create bucket {bucket_id}: {e}")
            raise

    def discover_files(self, repo_id: str) -> list[dict]:
        """List all files in a repo and classify them for offloading."""
        logger.info(f"Scanning repo: {repo_id}")
        try:
            tree = list(list_repo_tree(repo_id, recursive=True))
        except Exception as e:
            logger.error(f"  Failed to list repo {repo_id}: {e}")
            return []

        files = []
        for item in tree:
            if hasattr(item, "rfilename"):
                fname = item.rfilename
            elif hasattr(item, "path"):
                fname = item.path
            else:
                continue

            # Skip directories
            if hasattr(item, "type") and item.type == "directory":
                continue

            size = getattr(item, "size", 0) or 0
            last_modified = getattr(item, "last_commit", None)
            if last_modified and hasattr(last_modified, "date"):
                last_modified = last_modified.date
            else:
                last_modified = None

            files.append({
                "filename": fname,
                "size": size,
                "last_modified": last_modified,
                "should_offload": self._should_offload(fname),
            })

        offloadable = [f for f in files if f["should_offload"]]
        keep = [f for f in files if not f["should_offload"]]

        logger.info(f"  Total files: {len(files)}")
        logger.info(f"  Offloadable: {len(offloadable)}")
        logger.info(f"  Always keep: {len(keep)}")

        # Apply keep_recent: sort by last_modified descending, keep N most recent
        if self.config.keep_recent > 0 and len(offloadable) > self.config.keep_recent:
            offloadable.sort(
                key=lambda f: f["last_modified"] or datetime.min.replace(tzinfo=timezone.utc),
                reverse=True,
            )
            kept = offloadable[: self.config.keep_recent]
            to_offload = offloadable[self.config.keep_recent :]
            for f in kept:
                f["should_offload"] = False
                logger.info(f"  Keeping (recent): {f['filename']}")
        else:
            to_offload = offloadable

        logger.info(f"  Will offload: {len(to_offload)}")
        return to_offload

    def egress(self, repo_id: str, files_to_offload: list[dict]):
        """
        Egress: Move files from HF repo -> HF Storage Bucket.

        Steps:
        1. Download file from repo to temp dir
        2. Upload to bucket under {repo_name}/{filename}
        3. Verify it exists in bucket
        4. Delete from source repo
        5. Update Supabase DB
        """
        if not files_to_offload:
            logger.info(f"  Nothing to offload from {repo_id}")
            return

        repo_name = repo_id.split("/")[-1] if "/" in repo_id else repo_id
        bucket_prefix = f"{repo_name}"
        total_size = sum(f["size"] for f in files_to_offload)

        logger.info(f"  Offloading {len(files_to_offload)} files ({_human_size(total_size)}) from {repo_id}")
        staging_root = Path("/tmp/offload_staging")

        for f in files_to_offload:
            fname = f["filename"]
            bucket_path = f"{bucket_prefix}/{fname}"
            old_hf_path = f"{repo_id}/{fname}"
            new_bucket_path = f"{self.config.bucket_handle}/{bucket_path}"
            local_path = None

            if self.config.dry_run:
                logger.info(f"  [DRY RUN] Would offload: {fname} ({_human_size(f['size'])})")
                logger.info(f"    From: {old_hf_path}")
                logger.info(f"    To:   {new_bucket_path}")
                self.db.update_path(old_hf_path, new_bucket_path, dry_run=True)
                self.report.append({
                    "action": "egress",
                    "file": fname,
                    "repo": repo_id,
                    "bucket_path": new_bucket_path,
                    "size": f["size"],
                    "status": "dry_run",
                })
                continue

            try:
                # Step 1: Download from repo
                logger.info(f"  Downloading: {fname}")
                local_path = self.api.hf_hub_download(
                    repo_id=repo_id,
                    filename=fname,
                    local_dir=str(staging_root),
                )

                # Step 2: Upload to bucket
                logger.info(f"  Uploading to bucket: {bucket_path}")
                batch_bucket_files(
                    self.config.bucket_id,
                    add=[(local_path, bucket_path)],
                )

                # Step 3: Verify
                logger.info(f"  Verifying in bucket...")
                found = False
                for item in list_bucket_tree(self.config.bucket_id, prefix=bucket_path, recursive=True):
                    if hasattr(item, "path") and item.path == bucket_path:
                        found = True
                        break
                if not found:
                    logger.error(f"  VERIFICATION FAILED: {bucket_path} not found in bucket!")
                    self.report.append({
                        "action": "egress",
                        "file": fname,
                        "repo": repo_id,
                        "status": "verify_failed",
                    })
                    continue

                # Step 4: Delete from source repo
                if self.config.skip_delete:
                    logger.info(f"  Skipping source delete (skip_delete=true): {fname}")
                else:
                    logger.info(f"  Deleting from repo: {fname}")
                    self.api.delete_file(
                        path_in_repo=fname,
                        repo_id=repo_id,
                        commit_message=f"[offload] Moved {fname} to cold storage bucket",
                    )

                # Step 5: Update DB
                self.db.update_path(old_hf_path, new_bucket_path, dry_run=False)

                self.report.append({
                    "action": "egress",
                    "file": fname,
                    "repo": repo_id,
                    "bucket_path": new_bucket_path,
                    "size": f["size"],
                    "deleted_from_repo": not self.config.skip_delete,
                    "status": "success",
                })
                logger.info(f"  ✓ Offloaded: {fname}")

            except Exception as e:
                logger.error(f"  ✗ Failed to offload {fname}: {e}")
                self.report.append({
                    "action": "egress",
                    "file": fname,
                    "repo": repo_id,
                    "status": "error",
                    "error": str(e),
                })
            finally:
                self._cleanup_local_file(local_path, staging_root)

        # Cleanup staging dir
        import shutil
        staging = staging_root
        if staging.exists():
            shutil.rmtree(staging)

    def ingress(self, repo_id: str, files: Optional[list[str]] = None):
        """
        Ingress: Pull files back from bucket -> HF repo.

        If files is None, pulls back ALL files for that repo from the bucket.
        """
        repo_name = repo_id.split("/")[-1] if "/" in repo_id else repo_id
        bucket_prefix = f"{repo_name}"

        # List what's in the bucket for this repo
        try:
            bucket_files = [
                item for item in list_bucket_tree(self.config.bucket_id, prefix=bucket_prefix, recursive=True)
                if hasattr(item, "type") and item.type == "file"
            ]
        except Exception as e:
            logger.error(f"Failed to list bucket files for {repo_name}: {e}")
            return

        if files:
            bucket_files = [f for f in bucket_files if any(
                fnmatch.fnmatch(f.path.removeprefix(f"{bucket_prefix}/"), pat) for pat in files
            )]

        if not bucket_files:
            logger.info(f"No files to restore for {repo_id}")
            return

        logger.info(f"Restoring {len(bucket_files)} files to {repo_id}")
        staging_root = Path("/tmp/ingress_staging")

        for bf in bucket_files:
            bucket_path = bf.path
            repo_filename = bucket_path.removeprefix(f"{bucket_prefix}/")
            bucket_full = f"{self.config.bucket_handle}/{bucket_path}"
            local_dest = staging_root / repo_filename

            if self.config.dry_run:
                logger.info(f"  [DRY RUN] Would restore: {repo_filename}")
                logger.info(f"    From: {bucket_full}")
                logger.info(f"    To:   {repo_id}/{repo_filename}")
                self.report.append({
                    "action": "ingress",
                    "file": repo_filename,
                    "repo": repo_id,
                    "status": "dry_run",
                })
                continue

            try:
                # Download from bucket to local
                logger.info(f"  Downloading from bucket: {bucket_path}")
                from huggingface_hub import download_bucket_files
                os.makedirs(local_dest.parent, exist_ok=True)
                download_bucket_files(
                    self.config.bucket_id,
                    files=[(bucket_path, str(local_dest))],
                )

                # Upload to repo
                logger.info(f"  Uploading to repo: {repo_filename}")
                self.api.upload_file(
                    path_or_fileobj=str(local_dest),
                    path_in_repo=repo_filename,
                    repo_id=repo_id,
                    commit_message=f"[ingress] Restored {repo_filename} from cold storage",
                )

                # Update DB
                old_bucket_path = bucket_full
                new_repo_path = f"{repo_id}/{repo_filename}"
                self.db.update_path(old_bucket_path, new_repo_path, dry_run=False)

                # Optionally delete from bucket after restore
                # batch_bucket_files(self.config.bucket_id, delete=[bucket_path])

                logger.info(f"  ✓ Restored: {repo_filename}")
                self.report.append({
                    "action": "ingress",
                    "file": repo_filename,
                    "repo": repo_id,
                    "status": "success",
                })

            except Exception as e:
                logger.error(f"  ✗ Failed to restore {repo_filename}: {e}")
                self.report.append({
                    "action": "ingress",
                    "file": repo_filename,
                    "repo": repo_id,
                    "status": "error",
                    "error": str(e),
                })
            finally:
                self._cleanup_local_file(str(local_dest), staging_root)

        # Cleanup
        import shutil
        staging = staging_root
        if staging.exists():
            shutil.rmtree(staging)

    def run_egress(self):
        """Run the full egress pipeline for all configured repos."""
        logger.info("=" * 60)
        logger.info("OFFLOAD PIPELINE — EGRESS (repo -> bucket)")
        logger.info(f"Source org: {self.config.source_org}")
        logger.info(f"Bucket org: {self.config.bucket_org}")
        logger.info(f"Bucket: {self.config.bucket_id}")
        logger.info(f"Skip delete: {self.config.skip_delete}")
        logger.info(f"Dry run: {self.config.dry_run}")
        logger.info("=" * 60)

        self.ensure_bucket()

        for repo_name in self.config.source_repos:
            repo_id = self._resolve_repo_id(repo_name)
            files_to_offload = self.discover_files(repo_id)
            self.egress(repo_id, files_to_offload)

        self.print_report()

    def run_ingress(self, repo_id: str, files: Optional[list[str]] = None):
        """Run ingress for a specific repo."""
        logger.info("=" * 60)
        logger.info("OFFLOAD PIPELINE — INGRESS (bucket -> repo)")
        logger.info(f"Target repo: {repo_id}")
        logger.info(f"Dry run: {self.config.dry_run}")
        logger.info("=" * 60)

        self.ingress(repo_id, files)
        self.print_report()

    def print_report(self):
        """Print a summary of all operations."""
        logger.info("")
        logger.info("=" * 60)
        logger.info("REPORT")
        logger.info("=" * 60)

        if not self.report:
            logger.info("No operations performed.")
            return

        success = [r for r in self.report if r["status"] == "success"]
        dry_run = [r for r in self.report if r["status"] == "dry_run"]
        errors = [r for r in self.report if r["status"] == "error"]
        verify_failed = [r for r in self.report if r["status"] == "verify_failed"]

        if dry_run:
            total_size = sum(r.get("size", 0) for r in dry_run)
            logger.info(f"[DRY RUN] Would process {len(dry_run)} files ({_human_size(total_size)})")
        if success:
            total_size = sum(r.get("size", 0) for r in success)
            logger.info(f"Success: {len(success)} files ({_human_size(total_size)})")
        if errors:
            logger.info(f"Errors: {len(errors)} files")
            for r in errors:
                logger.info(f"  - {r['file']}: {r.get('error', 'unknown')}")
        if verify_failed:
            logger.info(f"Verify failed: {len(verify_failed)} files")

        # Save report as JSON
        report_path = f"offload_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(report_path, "w") as f:
            json.dump(self.report, f, indent=2, default=str)
        logger.info(f"Report saved to: {report_path}")


def _human_size(size_bytes: int) -> str:
    """Convert bytes to human-readable size."""
    for unit in ["B", "KB", "MB", "GB", "TB"]:
        if abs(size_bytes) < 1024:
            return f"{size_bytes:.1f} {unit}"
        size_bytes /= 1024
    return f"{size_bytes:.1f} PB"


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="HF Storage Bucket Offload Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Dry run (see what would happen)
  python offload.py --config config.yaml

  # Actually offload files to cold storage
  python offload.py --config config.yaml --execute

  # Restore files from cold storage back to a repo
  python offload.py --config config.yaml --execute --ingress laion/Qwen3-32B-SweSmith-20step

  # Restore specific files
  python offload.py --config config.yaml --execute --ingress laion/Qwen3-32B-SweSmith-20step --files "*.safetensors"
        """,
    )
    parser.add_argument("--config", "-c", required=True, help="Path to config YAML")
    parser.add_argument("--execute", action="store_true",
                        help="Actually execute (without this flag, runs in dry-run mode)")
    parser.add_argument("--ingress", metavar="REPO_ID",
                        help="Restore files FROM bucket TO this repo (e.g. laion/model-name)")
    parser.add_argument("--files", nargs="*",
                        help="Specific file patterns to restore (for --ingress only)")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")

    args = parser.parse_args()

    config = Config.from_yaml(args.config)

    # --execute overrides the config's dry_run setting
    if args.execute:
        config.dry_run = False
    else:
        config.dry_run = True
        logger.info("*** DRY RUN MODE — pass --execute to actually perform operations ***\n")

    if args.verbose:
        config.verbose = True
        logging.getLogger().setLevel(logging.DEBUG)

    pipeline = OffloadPipeline(config)

    if args.ingress:
        pipeline.run_ingress(args.ingress, args.files)
    else:
        pipeline.run_egress()


if __name__ == "__main__":
    main()
