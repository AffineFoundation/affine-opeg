"""Blob storage port.

Used for async archival of normalized trajectories to S3/MinIO. The hot path
serves trajectories from ``rollouts.extra_compressed`` (Postgres TOAST); the
blob is a long-term replica plus external-access entry point.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable


@runtime_checkable
class BlobStore(Protocol):
    """Object storage with presigned URL support."""

    async def put(self, key: str, data: bytes, *, content_type: str = "application/octet-stream") -> str:
        """Upload bytes; returns a stable URI (e.g. ``s3://bucket/key``)."""

    async def get(self, uri: str) -> bytes: ...

    async def exists(self, uri: str) -> bool: ...

    async def presigned_get_url(self, uri: str, *, expires_in: int = 300) -> str: ...
