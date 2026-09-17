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


def describe_cache(cache_root: str = DEFAULT_CACHE_ROOT, max_entries: int = 40) -> dict:
    """Report what is actually on disk around the cache root.

    Worker container logs are not always retrievable, so a startup failure has
    to be able to explain itself through the job response instead. This walks
    the mount point and the cache tree and reports what exists, including
    whether snapshot entries are symlinks into a blobs directory.
    """
    report: dict = {"cache_root": cache_root}
    root = Path(cache_root)

    probes = ["/runpod-volume", "/runpod-volume/huggingface-cache", cache_root]
    report["exists"] = {probe: Path(probe).is_dir() for probe in probes}

    def listing(path: Path) -> list[str]:
        try:
            return sorted(entry.name for entry in path.iterdir())[:max_entries]
        except OSError as error:
            return [f"<unreadable: {error}>"]

    for probe in probes:
        candidate = Path(probe)
        if candidate.is_dir():
            report.setdefault("listings", {})[probe] = listing(candidate)

    if root.is_dir():
        models = {}
        for model_dir in sorted(root.iterdir())[:max_entries]:
            if not model_dir.is_dir():
                continue
            entry: dict = {"contents": listing(model_dir)}
            snapshots = model_dir / "snapshots"
            if snapshots.is_dir():
                entry["snapshots"] = listing(snapshots)
                for snap in sorted(snapshots.iterdir())[:2]:
                    if snap.is_dir():
                        files = sorted(snap.iterdir())[:max_entries]
                        entry["snapshot_sample"] = {
                            "name": snap.name,
                            "files": [f.name for f in files],
                            "symlinks": {
                                f.name: (str(f.resolve()), f.resolve().exists())
                                for f in files
                                if f.is_symlink()
                            },
                        }
                        break
            refs_main = model_dir / "refs" / "main"
            if refs_main.is_file():
                try:
                    entry["refs_main"] = refs_main.read_text().strip()
                except OSError as error:
                    entry["refs_main"] = f"<unreadable: {error}>"
            models[model_dir.name] = entry
        report["models"] = models

    return report
