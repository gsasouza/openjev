import pytest

from worker_cache import resolve_cached_snapshot

MODEL = "Qwen/Qwen3.5-4B"
PINNED = "851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a"
OTHER = "22e683669bc0f0bd69640a1354a6d0aebcfeede5"


def build_cache(root, revision, refs_main=None):
    model_root = root / f"models--{MODEL.replace('/', '--')}"
    (model_root / "snapshots" / revision).mkdir(parents=True)
    if refs_main is not None:
        (model_root / "refs").mkdir(parents=True)
        (model_root / "refs" / "main").write_text(refs_main + "\n")
    return model_root


def test_pinned_snapshot_resolves(tmp_path):
    model_root = build_cache(tmp_path, PINNED, refs_main=PINNED)
    resolved = resolve_cached_snapshot(MODEL, PINNED, str(tmp_path))
    assert resolved == str(model_root / "snapshots" / PINNED)


def test_pinned_snapshot_wins_even_if_refs_main_moved(tmp_path):
    # The pin is the authority: a newer main must not silently be substituted.
    model_root = build_cache(tmp_path, PINNED, refs_main=OTHER)
    assert resolve_cached_snapshot(MODEL, PINNED, str(tmp_path)) == str(
        model_root / "snapshots" / PINNED
    )


def test_cache_holding_a_different_commit_is_rejected(tmp_path):
    build_cache(tmp_path, OTHER, refs_main=OTHER)
    with pytest.raises(RuntimeError) as error:
        resolve_cached_snapshot(MODEL, PINNED, str(tmp_path))
    assert OTHER in str(error.value) and PINNED in str(error.value)


def test_empty_cache_is_rejected(tmp_path):
    with pytest.raises(RuntimeError, match="No cached snapshot"):
        resolve_cached_snapshot(MODEL, PINNED, str(tmp_path))


def test_model_id_must_be_org_slash_name(tmp_path):
    with pytest.raises(RuntimeError, match="<org>/<name>"):
        resolve_cached_snapshot("Qwen3.5-4B", PINNED, str(tmp_path))


def test_describe_cache_reports_what_exists(tmp_path):
    from worker_cache import describe_cache

    build_cache(tmp_path, OTHER, refs_main=OTHER)
    report = describe_cache(str(tmp_path))

    assert report["cache_root"] == str(tmp_path)
    assert report["exists"][str(tmp_path)] is True
    model = report["models"]["models--Qwen--Qwen3.5-4B"]
    assert model["snapshots"] == [OTHER]
    assert model["refs_main"] == OTHER


def test_describe_cache_flags_broken_symlinks(tmp_path):
    from worker_cache import describe_cache

    model_root = build_cache(tmp_path, PINNED, refs_main=PINNED)
    snapshot = model_root / "snapshots" / PINNED
    (snapshot / "config.json").symlink_to(tmp_path / "blobs" / "missing")
    report = describe_cache(str(tmp_path))

    sample = report["models"]["models--Qwen--Qwen3.5-4B"]["snapshot_sample"]
    assert sample["symlinks"]["config.json"][1] is False


def test_describe_cache_survives_a_missing_root(tmp_path):
    from worker_cache import describe_cache

    report = describe_cache(str(tmp_path / "nope"))
    assert report["exists"][str(tmp_path / "nope")] is False
    assert "models" not in report
