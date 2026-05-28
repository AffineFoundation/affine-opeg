"""R2 publisher for the rollout corpus.

Periodically dumps successful rollouts (PG ``rollouts`` table, plus the
inline zstd-compressed trajectory blob) into a parquet file and uploads
it to an R2 (or any S3-compatible) bucket. Evaluators (the affinetes
``distill-v2`` environment) pull this parquet via a presigned/public
URL — no DB access required on the eval side.
"""
