"""The image must contain every local module the handler imports.

worker_cache.py was split out of handler.py but never added to the
Dockerfile's COPY, so handler.py died at import with ModuleNotFoundError on
every worker. Unit tests could not catch it: the file is present in the repo
and on CI's sys.path, and absent only inside the image.
"""

from __future__ import annotations

import ast
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
ENTRYPOINT = "handler.py"


def local_modules(root: Path = ROOT) -> set[str]:
    """Top-level names importable only because a sibling .py file exists."""
    return {path.stem for path in root.glob("*.py")}


def imported_by(filename: str, root: Path = ROOT) -> set[str]:
    tree = ast.parse((root / filename).read_text())
    names = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
            names.add(node.module.split(".")[0])
        elif isinstance(node, ast.Import):
            for alias in node.names:
                names.add(alias.name.split(".")[0])
    return names


def copied_into_image(root: Path = ROOT) -> set[str]:
    dockerfile = (root / "Dockerfile").read_text()
    copied = set()
    for line in re.findall(r"^COPY\s+(.+)$", dockerfile, re.M):
        # Drop the destination; every remaining token is a source path.
        for token in line.split()[:-1]:
            copied.add(Path(token).name)
    return copied


def test_handler_local_imports_are_copied_into_the_image():
    required = imported_by(ENTRYPOINT) & local_modules()
    missing = sorted(f"{name}.py" for name in required if f"{name}.py" not in copied_into_image())
    assert not missing, (
        f"{ENTRYPOINT} imports {missing} but the Dockerfile never COPYs them, "
        "so the container dies at import with ModuleNotFoundError."
    )


def test_the_entrypoint_itself_is_copied():
    assert ENTRYPOINT in copied_into_image()


def test_the_guard_actually_detects_a_missing_copy(tmp_path):
    # Guard against the guard silently passing: a Dockerfile without the
    # module must fail the same check.
    dockerfile = (ROOT / "Dockerfile").read_text()
    stripped = dockerfile.replace(" worker_cache.py", "")
    assert stripped != dockerfile, "fixture is stale: adjust to the current COPY line"

    (tmp_path / "Dockerfile").write_text(stripped)
    for name in ("handler.py", "worker_cache.py"):
        (tmp_path / name).write_text((ROOT / name).read_text())

    required = imported_by(ENTRYPOINT, tmp_path) & local_modules(tmp_path)
    missing = [f"{n}.py" for n in required if f"{n}.py" not in copied_into_image(tmp_path)]
    assert missing == ["worker_cache.py"]
