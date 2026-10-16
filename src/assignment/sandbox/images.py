"""Docker images for sandboxes, built on this machine.

A sandbox container is assembled from two images:

1. The *testbed* image: a published image as is (SWE-bench), or the task
   Dockerfile verbatim plus the pins layer, built from the filtered source
   checkout.
2. The *server* image: SWE-ReX's server on its own Python, mounted read-only at
   /opt/swerex in every container. It is built once per SWE-ReX version and
   never touches the testbed's Python or PATH.

Built images are tagged by a hash of everything that goes into them, so an
unchanged recipe is never rebuilt, and an edit makes exactly one new build.
"""

from __future__ import annotations

import hashlib
import logging
import shlex
import shutil
import tempfile
import time
from dataclasses import dataclass
from importlib.metadata import version
from pathlib import Path

from assignment.sandbox.config import DOCKER_BUILD_CONTEXT_VAR
from assignment.sandbox.dockercli import Daemon, DockerError, docker
from assignment.utils.image import ImageSpec, is_ignored

logger = logging.getLogger(__name__)

TESTBED_REPO = "assignment-testbed"
SERVER_REPO = "assignment-swerex"
# Every image and container the backend creates carries this label.
LABEL = "assignment.sandbox"
SERVER_ROOT = "/opt/swerex"
SERVER_PYTHON = "3.12"
UV_IMAGE = "ghcr.io/astral-sh/uv:0.12.19"
# Part of every tag: bump it when a recipe changes in a way the hashed inputs
# below do not capture.
RECIPE_VERSION = "1"


@dataclass(frozen=True)
class PreparedImages:
    """Image references ready on the local daemon."""

    testbed: str
    server: str


def server_dockerfile(swerex_version: str) -> str:
    """The SWE-ReX server, on a uv-managed CPython of its own under /opt/swerex."""
    return f"""\
FROM python:{SERVER_PYTHON}-slim AS build
COPY --from={UV_IMAGE} /uv /usr/local/bin/uv
# only-managed: never reuse a python{SERVER_PYTHON} that the base image happens to have.
RUN UV_PYTHON_PREFERENCE=only-managed UV_PYTHON_INSTALL_DIR={SERVER_ROOT}/python \\
        uv venv --quiet --python {SERVER_PYTHON} {SERVER_ROOT}/venv \\
    && uv pip install --quiet --python {SERVER_ROOT}/venv/bin/python swe-rex=={swerex_version}
FROM scratch
COPY --from=build {SERVER_ROOT}/python {SERVER_ROOT}/python
COPY --from=build {SERVER_ROOT}/venv {SERVER_ROOT}/venv
"""


def testbed_dockerfile(spec: ImageSpec) -> str:
    """The task Dockerfile verbatim (or the published image), then the pins."""
    if spec.dockerfile is not None:
        lines = [spec.dockerfile.read_text().rstrip("\n")]
    else:
        lines = [f"FROM {spec.base}"]
    if spec.pins:
        # Last, as on Modal, so the task's versions are the ones that survive.
        lines.append(
            "RUN python -m pip install --no-cache-dir "
            + " ".join(shlex.quote(pin) for pin in spec.pins)
        )
    return "\n".join(lines) + "\n"


def copy_context(source: Path, destination: Path) -> Path:
    """Copy `source` to `destination`, leaving out what `is_ignored` rejects.

    The build gets this copy, never the checkout itself, and the generated
    Dockerfile stays outside it: a task's `COPY . /testbed` must see exactly the
    repository and nothing of ours.
    """

    def ignore(directory: str, names: list[str]) -> set[str]:
        relative = Path(directory).relative_to(source)
        return {name for name in names if is_ignored(relative / name)}

    shutil.copytree(source, destination, symlinks=True, ignore=ignore)
    return destination


def recipe_tag(repo: str, dockerfile: str, context: Path | None, platform: str) -> str:
    """A tag naming exactly these build inputs.

    Hashes the Dockerfile text, the platform, and every context entry's path,
    type, executable bit, and content. Timestamps are left out, so touching a
    file does not cause a rebuild.
    """
    digest = hashlib.sha256()

    def feed(*parts: str | bytes) -> None:
        for part in parts:
            digest.update(part if isinstance(part, bytes) else part.encode())
            digest.update(b"\0")

    feed(RECIPE_VERSION, platform, dockerfile)
    if context is not None:
        for path in sorted(context.rglob("*")):
            relative = path.relative_to(context).as_posix()
            if path.is_symlink():
                feed("link", relative, str(path.readlink()))
            elif path.is_dir():
                feed("dir", relative)
            else:
                executable = "x" if path.stat().st_mode & 0o111 else "-"
                feed(
                    "file",
                    relative,
                    executable,
                    hashlib.sha256(path.read_bytes()).digest(),
                )
    return f"{repo}:{digest.hexdigest()[:16]}"


def has_image(reference: str) -> bool:
    try:
        docker("image", "inspect", "--format", "{{.Id}}", reference)
    except DockerError:
        return False
    return True


def build(
    tag: str,
    dockerfile: str,
    context: Path,
    *,
    daemon: Daemon,
    builder: str | None,
    no_cache: bool = False,
) -> None:
    """Build `tag` and load it into the local daemon.

    Raises:
        DockerError: The build failed. There is no fallback to building
            elsewhere; the message says how to pick another builder.
    """
    dockerfile_path = context.parent / f"{context.name}.Dockerfile"
    dockerfile_path.write_text(dockerfile)
    args = [
        "buildx", "build", "--load",
        "--platform", daemon.platform,
        "--tag", tag,
        "--label", f"{LABEL}=1",
        "--file", str(dockerfile_path),
    ]  # fmt: skip
    if builder:
        args += ["--builder", builder]
    if no_cache:
        args.append("--no-cache")

    started = time.monotonic()
    logger.info(
        "Building Docker image %s (the first build can take a few minutes)", tag
    )
    try:
        docker(*args, str(context), timeout=None)
    except DockerError as exc:
        details = "\n".join(str(exc).splitlines()[-40:])
        where = f"buildx builder {builder!r}" if builder else "this machine's Docker"
        raise DockerError(
            f"Could not build the sandbox image {tag} with {where}:\n{details}\n"
            "Nothing falls back to building elsewhere. To build with another buildx builder "
            f"whose result loads into the local daemon, set {DOCKER_BUILD_CONTEXT_VAR}=<builder> "
            "(list builders with `docker buildx ls`), or use SANDBOX=modal."
        ) from exc
    if not has_image(tag):
        raise DockerError(
            f"buildx builder {builder!r} built {tag} but did not load it into the local daemon. "
            f"{DOCKER_BUILD_CONTEXT_VAR} must name a builder whose `--load` output reaches this "
            "machine's Docker, such as a docker-container or remote BuildKit builder."
        )
    logger.info("Built %s in %.1fs", tag, time.monotonic() - started)


def pull(reference: str) -> None:
    """Pull a published image unless the local daemon already has it."""
    if has_image(reference):
        return
    started = time.monotonic()
    logger.info("Pulling %s (large images take several minutes)", reference)
    try:
        docker("pull", "--quiet", reference, timeout=None)
    except DockerError as exc:
        raise DockerError(
            f"Could not pull the sandbox image {reference}: {exc}"
        ) from exc
    logger.info("Pulled %s in %.1fs", reference, time.monotonic() - started)


def prepare_images(
    spec: ImageSpec, *, daemon: Daemon, builder: str | None
) -> PreparedImages:
    """Make the testbed and server images for `spec` available on the local daemon."""
    with tempfile.TemporaryDirectory(prefix="assignment-sandbox-build-") as scratch:
        scratch_dir = Path(scratch)

        server_recipe = server_dockerfile(version("swe-rex"))
        server_context = scratch_dir / "server"
        server_context.mkdir()
        server = recipe_tag(SERVER_REPO, server_recipe, None, daemon.platform)
        if not has_image(server):
            build(server, server_recipe, server_context, daemon=daemon, builder=builder)

        if spec.dockerfile is None and not spec.pins:
            # A published image runs as is; nothing is built on top of it.
            pull(spec.base)
            return PreparedImages(testbed=spec.base, server=server)

        testbed_recipe = testbed_dockerfile(spec)
        if spec.dockerfile is not None:
            testbed_context = copy_context(spec.context, scratch_dir / "testbed")
        else:
            testbed_context = scratch_dir / "testbed"
            testbed_context.mkdir()
        testbed = recipe_tag(
            TESTBED_REPO, testbed_recipe, testbed_context, daemon.platform
        )
        if spec.force_build or not has_image(testbed):
            build(
                testbed,
                testbed_recipe,
                testbed_context,
                daemon=daemon,
                builder=builder,
                no_cache=spec.force_build,
            )
        return PreparedImages(testbed=testbed, server=server)
