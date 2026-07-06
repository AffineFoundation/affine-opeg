"""Docker data-root janitor — evict unused swe-rebench eval images (LRU).

Runs as a compose sidecar (``distill-v2-janitor``) so disk housekeeping
ships with the stack instead of a host cron. Every ``JANITOR_INTERVAL_S``
it checks the docker data-root filesystem; once usage crosses
``JANITOR_THRESH`` percent it deletes the oldest unused ``swerebench/*``
images until usage falls back to ``JANITOR_TARGET`` percent.

Requirements (wire in docker-compose):
  * ``/var/run/docker.sock`` mounted — run docker CLI against the host daemon.
  * the data-root filesystem mounted read-only at ``JANITOR_DATA_PATH``
    (default ``/mnt/docker``) so ``df`` measures real disk pressure.

Safety: only ``swerebench/*`` images are eligible; ``docker rmi`` refuses
images held by a running container, so active ``afr-rb-*`` sandboxes are
skipped automatically and ``distill-producer`` is never matched.
"""

from __future__ import annotations

import os
import subprocess
import time

try:  # best-effort structured logging; fall back to print if config is absent
    from affine_opeg.infrastructure.config import load_config
    from affine_opeg.infrastructure.logging import configure_logging, get_logger
    configure_logging(load_config(), service="janitor")
    _log = get_logger("ops.janitor")

    def log(event: str, **kw: object) -> None:
        _log.info(event, **kw)
except Exception:  # noqa: BLE001
    def log(event: str, **kw: object) -> None:
        print(event, kw, flush=True)


def _pct_used(path: str) -> int | None:
    """Percent used of the filesystem backing ``path`` (``df --output=pcent``)."""
    try:
        out = subprocess.run(
            ["df", "--output=pcent", path],
            capture_output=True, text=True, timeout=30, check=True,
        ).stdout.strip().splitlines()
        return int("".join(c for c in out[-1] if c.isdigit()))
    except Exception as exc:  # noqa: BLE001
        log("janitor.df_failed", path=path, error=str(exc)[:200])
        return None


def _swerebench_images_oldest_first() -> list[str]:
    """Image IDs for ``swerebench/*``, oldest CreatedAt first."""
    try:
        out = subprocess.run(
            ["docker", "images", "swerebench/*", "--format", "{{.CreatedAt}}\t{{.ID}}"],
            capture_output=True, text=True, timeout=60, check=True,
        ).stdout.strip()
    except Exception as exc:  # noqa: BLE001
        log("janitor.list_failed", error=str(exc)[:200])
        return []
    rows = []
    for line in out.splitlines():
        if "\t" in line:
            created, image_id = line.split("\t", 1)
            rows.append((created, image_id))
    # sort by CreatedAt string (docker's format is lexically sortable enough
    # for LRU; ties broken by id). Dedup ids preserving order.
    rows.sort(key=lambda r: r[0])
    seen: set[str] = set()
    ids: list[str] = []
    for _created, image_id in rows:
        if image_id not in seen:
            seen.add(image_id)
            ids.append(image_id)
    return ids


def _rmi(image_id: str) -> bool:
    """Delete an image; returns False if in use (docker rmi refuses) or errors."""
    res = subprocess.run(
        ["docker", "rmi", image_id],
        capture_output=True, text=True, timeout=120,
    )
    return res.returncode == 0


def run_once(*, data_path: str, thresh: int, target: int) -> None:
    cur = _pct_used(data_path)
    if cur is None:
        return
    if cur < thresh:
        log("janitor.ok", pct=cur, thresh=thresh)
        return
    log("janitor.cleanup_start", pct=cur, thresh=thresh, target=target)
    removed = 0
    for image_id in _swerebench_images_oldest_first():
        now = _pct_used(data_path)
        if now is None or now <= target:
            break
        if _rmi(image_id):
            removed += 1
    log("janitor.cleanup_end", pct=_pct_used(data_path), removed=removed)


def main() -> None:
    data_path = os.environ.get("JANITOR_DATA_PATH", "/mnt/docker")
    thresh = int(os.environ.get("JANITOR_THRESH", "85"))
    target = int(os.environ.get("JANITOR_TARGET", "70"))
    interval = float(os.environ.get("JANITOR_INTERVAL_S", "1800"))
    enabled = os.environ.get("JANITOR_ENABLED", "1").strip() not in ("0", "false", "no", "")
    log("janitor.start", data_path=data_path, thresh=thresh, target=target,
        interval_s=interval, enabled=enabled)
    if not enabled:
        # Stay alive but idle so `restart: always` doesn't loop-crash the service.
        while True:
            time.sleep(3600)
    while True:
        try:
            run_once(data_path=data_path, thresh=thresh, target=target)
        except Exception as exc:  # noqa: BLE001
            log("janitor.cycle_failed", error=str(exc)[:300])
        time.sleep(interval)


if __name__ == "__main__":
    main()
