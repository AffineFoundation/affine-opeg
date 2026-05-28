"""S3 / MinIO blob store implementation."""

from __future__ import annotations

from urllib.parse import urlparse

import aioboto3

from affine_opeg.domain.errors import StorageError
from affine_opeg.infrastructure.config import BlobConfig


def _split_uri(uri: str) -> tuple[str, str]:
    parsed = urlparse(uri)
    if parsed.scheme != "s3":
        raise StorageError(f"unsupported scheme: {uri}")
    bucket = parsed.netloc
    key = parsed.path.lstrip("/")
    if not bucket or not key:
        raise StorageError(f"invalid s3 uri: {uri}")
    return bucket, key


class S3BlobStore:
    """Async S3 access.

    Single-process safe: ``aioboto3.Session`` is cheap; the underlying
    botocore client is recreated per call but pooled internally by aiohttp.
    """

    def __init__(self, cfg: BlobConfig) -> None:
        self._cfg = cfg
        self._session = aioboto3.Session(
            aws_access_key_id=cfg.access_key.get_secret_value() if cfg.access_key else None,
            aws_secret_access_key=cfg.secret_key.get_secret_value() if cfg.secret_key else None,
            region_name=cfg.region,
        )

    def _client_kwargs(self) -> dict:
        kw = {"region_name": self._cfg.region}
        if self._cfg.endpoint:
            kw["endpoint_url"] = self._cfg.endpoint
        return kw

    async def put(self, key: str, data: bytes, *, content_type: str = "application/octet-stream") -> str:
        async with self._session.client("s3", **self._client_kwargs()) as s3:
            await s3.put_object(Bucket=self._cfg.bucket, Key=key, Body=data, ContentType=content_type)
        return f"s3://{self._cfg.bucket}/{key}"

    async def get(self, uri: str) -> bytes:
        bucket, key = _split_uri(uri)
        async with self._session.client("s3", **self._client_kwargs()) as s3:
            obj = await s3.get_object(Bucket=bucket, Key=key)
            async with obj["Body"] as stream:
                return await stream.read()

    async def exists(self, uri: str) -> bool:
        bucket, key = _split_uri(uri)
        async with self._session.client("s3", **self._client_kwargs()) as s3:
            try:
                await s3.head_object(Bucket=bucket, Key=key)
                return True
            except s3.exceptions.ClientError:
                return False

    async def presigned_get_url(self, uri: str, *, expires_in: int = 300) -> str:
        bucket, key = _split_uri(uri)
        async with self._session.client("s3", **self._client_kwargs()) as s3:
            return await s3.generate_presigned_url(
                "get_object", Params={"Bucket": bucket, "Key": key}, ExpiresIn=expires_in
            )
