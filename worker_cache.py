"""Resolution of Runpod's cached Hugging Face snapshots.

Kept apart from handler.py so it can be imported and tested without the
import-time model load that the Runpod worker depends on.
"""

from __future__ import annotations

from pathlib import Path

DEFAULT_CACHE_ROOT = "/runpod-volume/huggingface-cache/hub"


def resolve_cached_snapshot(model_id: str, revision: str, cache_root: str = DEFAULT_CACHE_ROOT) -> str:
    """Locate the cached copy of one pinned commit.

    Runpod caches a model under models--<org>--<name>/snapshots/<commit>, so
    asking for the pinned commit by name verifies the pin instead of trusting
    whatever `refs/main` currently points at. A miss is fatal on purpose: the
    alternative is a silent multi-gigabyte download on every cold start, which
    is the thing caching exists to avoid.
    """
    org, _, name = model_id.partition("/")
    if not org or not name:
        raise RuntimeError(f"OPENJEV_MODEL must be <org>/<name>, got {model_id!r}")

    root = Path(cache_root) / f"models--{org}--{name}"
    snapshot = root / "snapshots" / revision
    if snapshot.is_dir():
        return str(snapshot)

    refs_main = root / "refs" / "main"
    if refs_main.is_file():
        cached = refs_main.read_text().strip()
        raise RuntimeError(
            f"{model_id}: the cache holds commit {cached}, but the manifest pins "
            f"{revision}. Re-cache the pinned commit or update manifests/models.json."
        )
    raise RuntimeError(
        f"No cached snapshot for {model_id} under {root}. Set the endpoint's Model "
        f"field to {model_id} so Runpod caches it before workers start."
    )
