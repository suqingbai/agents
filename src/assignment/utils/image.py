"""Describe a task's testbed image, built from a local checkout.

The repository under test is copied in from `task.source` — normally the
`chess_app/` submodule — rather than cloned inside the build. That keeps the
build offline, works with a private repository without any credential, and
avoids a network round trip on every rebuild.

The recipe is backend-neutral (`ImageSpec`); each sandbox backend builds it its
own way, Modal on Modal and Docker on this machine.

The cost is that the image reflects whatever is on disk, so `verify_source`
checks the `chess_app` source before every build to confirm it matches the
source the assignment intends people to fix. A testbed silently built from an
already-fixed working tree would make a broken agent look like it passed.
"""

from __future__ import annotations

import logging
import subprocess
from collections.abc import Mapping
from dataclasses import dataclass, replace
from pathlib import Path

from assignment.task import Task

logger = logging.getLogger(__name__)

# Local caches, credentials, and git metadata never belong in the testbed. The
# .git directory in particular is a gitlink file in a submodule checkout and
# would be meaningless inside the container; the build makes a fresh repository
# instead.
IGNORED_NAMES = {".git", ".venv", ".pytest_cache", "__pycache__", ".env", ".DS_Store"}

class SourceMismatch(Exception):
    """The local checkout is not the commit the task is defined against."""

def _git(source: Path, *args: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        ["git", "-C", str(source), *args], capture_output=True, text=True, check=False
    )

def is_ignored(path: Path) -> bool:
    """Whether a file should be kept out of the build context."""
    return any(part in IGNORED_NAMES for part in path.parts) or path.suffix == ".pyc"

def verify_source(task: Task, strict: bool = True) -> None:
    """Check that the local checkout matches the task's base commit.

    Args:
        task: The task whose `source` to check.
        strict: Raise on a mismatch. When False, mismatches are logged as
            warnings and the build proceeds — useful when deliberately testing
            a modified checkout.

    Raises:
        SourceMismatch: The checkout is missing, on the wrong commit, or dirty.
    """

    def complain(message: str) -> None:
        if strict:
            raise SourceMismatch(message)
        logger.warning("%s (continuing: strict=False)", message)

    if not task.source.is_dir():
        raise SourceMismatch(
            f"{task.source} does not exist. If it is a submodule, run `git submodule update --init`."
        )

    head = _git(task.source, "rev-parse", "HEAD")
    if head.returncode != 0:
        complain(f"{task.source} is not a git checkout: {head.stderr.strip()}")
        return

    if head.stdout.strip() != task.base_commit:
        complain(
            f"{task.source} is at {head.stdout.strip()[:12]}, but {task.id} is defined against "
            f"{task.base_commit[:12]}. The testbed would not be the task's base commit."
        )

    dirty = _git(task.source, "status", "--porcelain")
    if dirty.stdout.strip():
        # Each line is a status code then the path. The code's width varies, so
        # split on whitespace rather than slicing at a fixed offset.
        paths = [line.strip().split(None, 1)[-1] for line in dirty.stdout.strip().splitlines()[:5]]
        changed = ", ".join(paths)
        complain(
            f"{task.source} has uncommitted changes ({changed}). The testbed would contain them, "
            "which silently invalidates the evaluation if one of them is the fix."
        )

@dataclass(frozen=True)
class ImageSpec:
    """What a sandbox image contains, independent of where it is built.

    Either `base` names a published image, or `dockerfile` is built from
    `context` with `is_ignored` files left out. `pins` are installed on top,
    last, and `files` are copied into each sandbox when it starts.
    """

    base: str | None = None
    dockerfile: Path | None = None
    context: Path | None = None
    pins: tuple[str, ...] = ()
    # (local file, absolute path in the sandbox)
    files: tuple[tuple[Path, str], ...] = ()
    force_build: bool = False

    def __post_init__(self):
        if (self.base is None) == (self.dockerfile is None):
            raise ValueError("An ImageSpec needs exactly one of `base` or `dockerfile`.")
        if self.dockerfile is not None and self.context is None:
            raise ValueError("An ImageSpec built from a Dockerfile needs a `context`.")

    def with_files(self, files: Mapping[Path, str]) -> ImageSpec:
        """A copy that also places `files` (local path -> sandbox path) in the sandbox."""
        return replace(self, files=self.files + tuple(files.items()))


def build_testbed_image(task: Task, strict: bool = True, force_build: bool = False) -> ImageSpec:
    """Describe the image holding the repository under test at its base commit.

    Args:
        task: The task whose Dockerfile and source checkout to build from.
        strict: Refuse when the checkout does not match the task's base commit.
            See `verify_source`.
        force_build: Skip the backend's build cache.

    Returns:
        The image recipe, with the repository installed at /testbed once built.
        Pass it to `Environment(image=...)`; the sandbox backend builds it.
    """
    verify_source(task, strict=strict)
    return ImageSpec(
        dockerfile=task.dockerfile,
        context=task.source,
        pins=task.pins,
        force_build=force_build,
    )
