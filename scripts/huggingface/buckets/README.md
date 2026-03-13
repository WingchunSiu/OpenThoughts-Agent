# HF Storage Bucket Offload Pipeline

Migrates model checkpoints and artifacts from HF repos (hot storage) to HF Storage Buckets (cold storage), with optional Supabase DB updates.

## Setup

```bash
pip install huggingface_hub>=1.6.0 pyyaml supabase python-dotenv
huggingface-cli login  # authenticate with your HF token
```

## Bucket Management (HF CLI)

The `huggingface_hub` library provides native CLI commands for bucket management. Use these directly:

```bash
# Create a bucket
hf buckets create my-bucket
hf buckets create my-bucket --private
hf buckets create my-org/shared-bucket

# List buckets
hf buckets list
hf buckets list --namespace my-org
hf buckets list --format json

# Get bucket info
hf buckets info username/my-bucket
hf buckets info username/my-bucket --format json

# Delete a bucket
hf buckets delete username/my-bucket
hf buckets delete username/my-bucket --yes

# Move/rename a bucket
hf buckets move username/old-name username/new-name
hf buckets move username/my-bucket my-org/my-bucket
```

## File Operations (HF CLI)

```bash
# List files in a bucket
hf buckets list username/my-bucket
hf buckets list username/my-bucket -R  # recursive
hf buckets list username/my-bucket --tree -h -R  # tree format with human-readable sizes

# Upload a file
hf buckets cp ./config.json hf://buckets/username/my-bucket
hf buckets cp ./model.safetensors hf://buckets/username/my-bucket/models/model.safetensors

# Download a file
hf buckets cp hf://buckets/username/my-bucket/config.json ./config.json
hf buckets cp hf://buckets/username/my-bucket/config.json - | jq .  # to stdout

# Remove files
hf buckets rm username/my-bucket/old-model.bin
hf buckets rm username/my-bucket/logs/ --recursive
hf buckets rm username/my-bucket --recursive --include "*.tmp"

# Sync directory with bucket
hf buckets sync ./data hf://buckets/username/my-bucket  # upload
hf buckets sync hf://buckets/username/my-bucket ./data  # download
hf buckets sync ./data hf://buckets/username/my-bucket --delete  # with cleanup
hf buckets sync ./data hf://buckets/username/my-bucket --dry-run | jq '.action'  # preview
```

## Offload Pipeline Usage

The `offload.py` script provides automated migration workflows with filtering, retention policies, and DB updates.

### Egress (repo → bucket)

```bash
# Dry run — see what would be offloaded, nothing is actually moved
python offload.py --config config.yaml

# Execute for real
python offload.py --config config.yaml --execute
```

### Ingress (bucket → repo)

```bash
# Restore all cold-stored files back to a repo
python offload.py --config config.yaml --execute --ingress laion/Qwen3-32B-SweSmith-20step

# Restore only specific files
python offload.py --config config.yaml --execute --ingress laion/Qwen3-32B-SweSmith-20step --files "*.safetensors"
```

## How it works

**Egress flow:**
1. Scans each source repo for files matching `include_patterns`
2. Filters out files matching `exclude_patterns` (config.json, tokenizer, etc.)
3. Keeps the N most recent files (`keep_recent`) in hot storage
4. Downloads remaining files → uploads to bucket under `{repo_name}/{filename}`
5. Verifies the file exists in the bucket
6. Deletes from source repo with a commit message (unless `skip_delete: true`)
7. Updates Supabase DB path reference (if enabled)

**Ingress flow:**
1. Lists files in the bucket under the repo's prefix
2. Downloads from bucket → uploads back to the HF repo
3. Updates Supabase DB path reference (if enabled)

## Config

See `config.yaml` for all options. Key settings:

- `source_org` — HF org/user to read repos from (e.g. `laion`)
- `bucket_org` — HF org/user that owns the bucket (can be personal for testing)
- `bucket_name` — name for the cold storage bucket
- `source_repos` — list of repo names to offload from
- `include/exclude_patterns` — glob patterns for file selection
- `keep_recent` — number of most recent checkpoint files to keep in hot storage
- `skip_delete` — if true, upload only (no source repo deletion)
- `supabase` — DB integration config

## Testing on your own account

1. Create a test repo under your personal HF account
2. Push some dummy `.safetensors` files to it
3. Set `source_org` to your source repo owner and `bucket_org` to your username
4. Keep `skip_delete: true` for first live test
5. Run dry run, verify output looks right
6. Run with `--execute`, confirm files moved correctly
7. Test ingress to pull them back

## Supabase Integration

The DB update logic in `offload.py` is schema-dependent and should be customized for your deployment:
- The table name and schema
- Which column stores the HF path
- Any additional columns that need updating (status, timestamp, etc.)

Suggested production DB flow (metadata only):
- On successful offload: write/update mapping from `repo_id/path` -> `hf://buckets/.../path`
- If your schema uses separate metadata rows: optionally delete or archive the old hot-storage reference row
- On ingress restore: reverse mapping from bucket path back to repo path

Set `SUPABASE_KEY` as an environment variable (or in a `.env` file).

## Notes

- Always start with `--dry-run` (the default) before executing
- The bucket uses Xet dedup, so similar checkpoints share storage efficiently
- Reports are saved as JSON files after each run
- Files are offloaded one at a time with verification — if something fails mid-run, already-completed files are safe
