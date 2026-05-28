# affine-opeg

Off-policy rollout **generator** + **publisher** for Affine's
[`distill-v2`](https://github.com/AffineFoundation/affinetes/tree/main/environments/distill-v2)
evaluation pipeline.

Python import name: `affine_opeg`.

## What it does

1. **Generator** schedules `(env_name, task_id, teacher_model)` cells from a
   sampling pool (CSPRNG-backed so miners can't predict the next task).
2. For each cell it runs N agentic rollouts inside per-task Docker sandboxes
   (swe-rebench style), driven by [`affent`](https://github.com/AffineFoundation/affent).
3. Each rollout's trajectory + reward goes to Postgres; the same row holds a
   zstd-compressed full trace for downstream evaluators.
4. **Publisher** rolls each finished cell up as one immutable parquet shard
   in R2 (or any S3-compatible store), maintains a small `manifest.jsonl`
   index, and writes a thin `metadata.json` pointer for cheap consumer
   discovery.
5. A separate **promoter** copies mature shards from the private bucket to
   a public bucket under a configurable per-day rate cap (no per-cell
   maturation delay) — matches the SWE-Infinite consumer convention.

## Quick start

```bash
pip install -e .

# Generator worker
export AFR_DB__HOST=...   # postgres connection
export AFR_BLOB__BUCKET=...
export AFR_BLOB__PUBLIC_BUCKET=...
export AFR_BLOB__ENDPOINT=...
export AFR_BLOB__ACCESS_KEY=...
export AFR_BLOB__SECRET_KEY=...
export AFR_SAMPLING_LIST=smoke-01
export CHUTES_API_KEY=...

affine-opeg-generator
affine-opeg-publisher
```

## Layout

```
src/affine_opeg/
├── application/          # producer loop, cell scheduler, generate_rollout use-case
├── adapters/             # docker sandbox, affent CLI loop, openai-compat teacher, …
├── domain/               # cell / rollout / pair models + ports (Protocol interfaces)
├── infrastructure/       # config (AFR_* env), DB session, structlog, metrics
├── publishing/           # publisher (write R2) + promoter (private → public)
└── workers/              # CLI entry points the two ``affine-opeg-*`` scripts wrap

migrations/               # alembic versions
```

## R2 layout written by publisher / promoter

```
{base_url}/manifest.jsonl                # append-only per-cell index
{base_url}/metadata.json                 # {"version":1, tasks: {total | staged_up_to + completed_up_to}}
{base_url}/tasks/{task_idx:08d}.parquet  # one immutable shard = one cell
```

The downstream evaluator (`affinetes/environments/distill-v2`) consumes the
public bucket via the R2 dev subdomain or any custom domain.
