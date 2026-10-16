"""The Docker backend without a Docker daemon: every docker call is faked."""

from __future__ import annotations

import asyncio
import os
from collections.abc import Callable
from importlib.metadata import version
from pathlib import Path

import modal
import pytest
from support import CHESS_TASK

from assignment.sandbox import DockerSandboxConfig, dockercli, images
from assignment.sandbox import docker as docker_backend
from assignment.sandbox.docker import (
    SWEREX_PORT,
    AssignmentDockerDeployment,
    DockerBackend,
    as_image_spec,
)
from assignment.sandbox.dockercli import (
    Daemon,
    DockerError,
    engine_major,
    is_local_endpoint,
    local_daemon,
)
from assignment.task import Task
from assignment.utils.image import ImageSpec

DAEMON = Daemon(
    endpoint="unix:///var/run/docker.sock", version="29.8.1", os="linux", arch="amd64"
)


class FakeDocker:
    """Stands in for `docker()`: records each call and answers from a handler."""

    def __init__(self, handler: Callable[[tuple[str, ...]], str] | None = None):
        self.calls: list[tuple[str, ...]] = []
        self.inputs: list[bytes | None] = []
        self.handler = handler or (lambda args: "")

    def __call__(
        self,
        *args: str,
        input=None,
        timeout=None,
        check=True,
        merge_stderr=False,
        detached=False,
    ):
        self.calls.append(args)
        self.inputs.append(input)
        return self.handler(args)

    def called(self, *prefix: str) -> list[tuple[str, ...]]:
        return [call for call in self.calls if call[: len(prefix)] == prefix]


@pytest.fixture
def fake_docker(monkeypatch):
    def install(handler=None) -> FakeDocker:
        fake = FakeDocker(handler)
        for module in (dockercli, images, docker_backend):
            monkeypatch.setattr(module, "docker", fake)
        return fake

    return install


# -- Which daemon -------------------------------------------------------------


@pytest.mark.parametrize(
    ("endpoint", "local"),
    [
        ("unix:///var/run/docker.sock", True),
        ("unix:///home/me/.colima/default/docker.sock", True),
        ("npipe:////./pipe/docker_engine", True),
        ("tcp://127.0.0.1:2375", True),
        ("tcp://localhost:2376", True),
        ("tcp://[::1]:2375", True),
        ("tcp://10.0.0.5:2376", False),
        ("tcp://gpu-box:2376", False),
        ("ssh://me@gpu-box", False),
        ("ssh://localhost", False),
        ("", False),
    ],
)
def test_only_daemons_on_this_machine_count_as_local(endpoint, local):
    assert is_local_endpoint(endpoint) is local


@pytest.mark.parametrize(
    ("text", "major"),
    [("29.8.1", 29), ("28.0.0-rc.1", 28), ("v27.5.1", 27), ("dev", None)],
)
def test_engine_major(text, major):
    assert engine_major(text) == major


def daemon_answers(endpoint: str, server: str):
    def handler(args):
        if args[0] == "context":
            return endpoint
        if args[0] == "version":
            return server
        raise AssertionError(args)

    return handler


def test_local_daemon_reports_what_it_found(fake_docker):
    fake_docker(daemon_answers("unix:///var/run/docker.sock", "29.8.1|linux|amd64"))
    assert local_daemon() == DAEMON
    assert DAEMON.platform == "linux/amd64"


def test_a_remote_daemon_is_refused_before_contacting_it(fake_docker):
    fake = fake_docker(daemon_answers("ssh://me@gpu-box", "29.8.1|linux|amd64"))
    with pytest.raises(
        DockerError, match=r"this machine's Docker daemon only.*ssh://me@gpu-box"
    ):
        local_daemon()
    assert not fake.called("version")


def test_an_engine_older_than_28_is_refused_with_the_reason(fake_docker):
    fake_docker(daemon_answers("unix:///var/run/docker.sock", "27.5.1|linux|amd64"))
    with pytest.raises(
        DockerError, match=r"Engine 27.5.1 .* too old.*Engine 28 or newer.*image mounts"
    ):
        local_daemon()


def test_windows_containers_are_refused(fake_docker):
    fake_docker(
        daemon_answers("npipe:////./pipe/docker_engine", "29.1.0|windows|amd64")
    )
    with pytest.raises(DockerError, match="Linux containers"):
        local_daemon()


def test_an_unreachable_daemon_says_so(fake_docker):
    def handler(args):
        if args[0] == "context":
            return "unix:///var/run/docker.sock"
        raise DockerError("Cannot connect to the Docker daemon")

    fake_docker(handler)
    with pytest.raises(DockerError, match="not reachable.*Cannot connect"):
        local_daemon()


def test_a_missing_docker_cli_is_a_docker_error(monkeypatch):
    monkeypatch.setenv("PATH", "/nonexistent")
    with pytest.raises(DockerError, match="not installed"):
        dockercli.docker("version")


# -- Images -------------------------------------------------------------------


def test_image_arguments_become_recipes(tmp_path):
    spec = ImageSpec(base="python:3.12")
    assert as_image_spec(spec) is spec
    assert as_image_spec("python:3.12") == ImageSpec(base="python:3.12")

    dockerfile = tmp_path / "Dockerfile"
    dockerfile.write_text("FROM python:3.12\n")
    for value in (dockerfile, str(dockerfile)):
        assert as_image_spec(value) == ImageSpec(
            dockerfile=dockerfile, context=tmp_path
        )


def test_a_modal_image_is_refused_by_the_docker_backend():
    with pytest.raises(TypeError, match="modal.Image only works with SANDBOX=modal"):
        as_image_spec(modal.Image.debian_slim())


def test_modal_sandbox_kwargs_are_refused_by_the_docker_backend():
    with pytest.raises(ValueError, match="exposed_ports"):
        DockerBackend(
            DockerSandboxConfig(),
            image="python:3.12",
            exposed_ports=(),
            startup_timeout=1,
            runtime_timeout=1,
            deployment_timeout=1,
            install_pipx=True,
            modal_sandbox_kwargs={"encrypted_ports": [8000]},
        )


def test_testbed_dockerfile_is_the_task_dockerfile_then_the_pins():
    task = Task.load(CHESS_TASK)
    spec = ImageSpec(dockerfile=task.dockerfile, context=task.source, pins=task.pins)

    text = images.testbed_dockerfile(spec)

    original = task.dockerfile.read_text().rstrip("\n")
    assert text.startswith(original + "\n")
    rest = text[len(original) + 1 :].splitlines()
    assert rest == ["RUN python -m pip install --no-cache-dir " + " ".join(task.pins)]


def test_a_published_image_with_pins_builds_from_it():
    spec = ImageSpec(base="python:3.12-slim", pins=("rich==13.0.0",))
    assert images.testbed_dockerfile(spec) == (
        "FROM python:3.12-slim\nRUN python -m pip install --no-cache-dir rich==13.0.0\n"
    )


def test_server_image_has_its_own_python_and_the_client_swerex_version():
    text = images.server_dockerfile(version("swe-rex"))
    assert "UV_PYTHON_PREFERENCE=only-managed" in text
    assert f"swe-rex=={version('swe-rex')}" in text
    # Only the server's own files end up in the image.
    final_stage = text.split("FROM scratch", 1)[1]
    assert final_stage.split() == [
        "COPY", "--from=build", "/opt/swerex/python", "/opt/swerex/python",
        "COPY", "--from=build", "/opt/swerex/venv", "/opt/swerex/venv",
    ]  # fmt: skip


def make_source(root: Path) -> Path:
    for name, content in {
        "pyproject.toml": "[project]\n",
        "src/app.py": "print('hi')\n",
        "README.md": "# app\n",
        ".git": "gitdir: ../.git/modules/app\n",
        ".env": "SECRET=1\n",
        "src/__pycache__/app.cpython-312.pyc": "x",
        "src/stale.pyc": "x",
        ".venv/bin/python": "x",
        ".pytest_cache/README.md": "x",
    }.items():
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content)
    return root


def test_build_context_leaves_out_ignored_files(tmp_path):
    source = make_source(tmp_path / "source")
    context = images.copy_context(source, tmp_path / "context")
    files = sorted(
        p.relative_to(context).as_posix() for p in context.rglob("*") if p.is_file()
    )
    assert files == ["README.md", "pyproject.toml", "src/app.py"]


def test_testbed_tag_tracks_content_not_timestamps(tmp_path):
    source = make_source(tmp_path / "source")

    def tag(dockerfile="FROM python:3.12\nCOPY . /testbed\n", platform="linux/amd64"):
        context = tmp_path / f"context-{len(list(tmp_path.iterdir()))}"
        images.copy_context(source, context)
        return images.recipe_tag("assignment-testbed", dockerfile, context, platform)

    first = tag()
    assert first.startswith("assignment-testbed:") and len(first.split(":")[1]) == 16

    os.utime(source / "src" / "app.py", (1, 1))
    (source / "src" / "__pycache__" / "app.cpython-312.pyc").write_text("changed")
    (source / ".env").write_text("SECRET=2\n")
    assert tag() == first, "timestamps and ignored files must not change the tag"

    assert tag(dockerfile="FROM python:3.13\nCOPY . /testbed\n") != first
    assert tag(platform="linux/arm64") != first
    (source / "src" / "app.py").chmod(0o755)
    executable = tag()
    assert executable != first
    (source / "src" / "app.py").write_text("print('bye')\n")
    assert tag() not in (first, executable)


def test_a_failed_build_names_the_escape_hatch_and_does_not_fall_back(
    fake_docker, tmp_path
):
    def handler(args):
        if args[:2] == ("buildx", "build"):
            raise DockerError(
                "`docker buildx build` failed (exit 1): exec format error"
            )
        raise AssertionError(args)

    fake = fake_docker(handler)
    context = tmp_path / "ctx"
    context.mkdir()
    with pytest.raises(
        DockerError, match=r"exec format error(.|\n)*SANDBOX_DOCKER_BUILD_CONTEXT"
    ):
        images.build(
            "assignment-testbed:abc",
            "FROM scratch\n",
            context,
            daemon=DAEMON,
            builder=None,
        )
    assert len(fake.called("buildx", "build")) == 1


def test_a_build_keeps_its_dockerfile_outside_the_context(fake_docker, tmp_path):
    fake = fake_docker(lambda args: "sha256:1")
    context = tmp_path / "ctx"
    context.mkdir()

    images.build(
        "assignment-testbed:abc",
        "FROM scratch\n",
        context,
        daemon=DAEMON,
        builder="bk",
        no_cache=True,
    )

    (build,) = fake.called("buildx", "build")
    dockerfile = Path(build[build.index("--file") + 1])
    assert context not in dockerfile.parents
    assert build[-1] == str(context)
    assert {"--load", "--no-cache"} <= set(build)
    assert build[build.index("--builder") + 1] == "bk"
    assert build[build.index("--platform") + 1] == "linux/amd64"
    assert build[build.index("--label") + 1] == "assignment.sandbox=1"


def test_a_builder_that_does_not_load_locally_is_reported(fake_docker, tmp_path):
    def handler(args):
        if args[:2] == ("image", "inspect"):
            raise DockerError("No such image")
        return ""

    fake_docker(handler)
    context = tmp_path / "ctx"
    context.mkdir()
    with pytest.raises(DockerError, match="did not load it into the local daemon"):
        images.build(
            "assignment-testbed:abc",
            "FROM scratch\n",
            context,
            daemon=DAEMON,
            builder="far",
        )


def test_a_published_image_is_pulled_not_built(fake_docker):
    present = {"assignment-swerex"}

    def handler(args):
        if args[:2] == ("image", "inspect"):
            if args[-1].split(":")[0] in present:
                return "sha256:1"
            raise DockerError("No such image")
        return ""

    fake = fake_docker(handler)
    prepared = images.prepare_images(
        ImageSpec(base="swebench/sweb.eval.x86_64.demo:latest"),
        daemon=DAEMON,
        builder=None,
    )

    assert prepared.testbed == "swebench/sweb.eval.x86_64.demo:latest"
    assert prepared.server.startswith("assignment-swerex:")
    assert not fake.called("buildx")
    assert fake.called("pull") == [
        ("pull", "--quiet", "swebench/sweb.eval.x86_64.demo:latest")
    ]


def test_present_images_are_not_rebuilt_unless_forced(fake_docker):
    fake = fake_docker(lambda args: "sha256:1")
    task = Task.load(CHESS_TASK)
    spec = ImageSpec(dockerfile=task.dockerfile, context=task.source, pins=task.pins)

    first = images.prepare_images(spec, daemon=DAEMON, builder=None)
    assert not fake.called("buildx")

    again = images.prepare_images(
        ImageSpec(
            dockerfile=task.dockerfile,
            context=task.source,
            pins=task.pins,
            force_build=True,
        ),
        daemon=DAEMON,
        builder=None,
    )
    assert again == first
    (build,) = fake.called("buildx", "build")
    assert "--no-cache" in build and build[build.index("--tag") + 1] == first.testbed


# -- Container lifecycle -------------------------------------------------------


def test_container_flags_and_cleanup_when_start_fails(
    fake_docker, monkeypatch, tmp_path
):
    monkeypatch.setattr(docker_backend, "local_daemon", lambda: DAEMON)
    monkeypatch.setattr(
        docker_backend,
        "prepare_images",
        lambda spec, daemon, builder: images.PreparedImages(
            testbed="assignment-testbed:t", server="assignment-swerex:s"
        ),
    )

    def handler(args):
        if args[0] == "start":
            raise DockerError(
                "`docker start` failed (exit 1): exec: timeout: not found"
            )
        if args[:2] == ("container", "inspect"):
            raise DockerError("No such container")
        return ""

    fake = fake_docker(handler)
    payload = tmp_path / "chess_server.py"
    payload.write_text("print('serve')\n")
    deployment = AssignmentDockerDeployment(
        image=ImageSpec(base="python:3.12").with_files(
            {payload: "/opt/assignment/chess_server.py"}
        ),
        exposed_ports=[8000],
        deployment_timeout=90.5,
        run_args=["--cpus=2"],
    )

    with pytest.raises(DockerError, match="timeout: not found"):
        asyncio.run(deployment.start())

    (create,) = fake.called("create")
    name = create[create.index("--name") + 1]
    assert name.startswith("assignment-sandbox-")
    assert create[create.index("--label") + 1] == "assignment.sandbox=1"
    assert {"--rm", "--init"} <= set(create)
    assert create[create.index("--entrypoint") + 1] == ""
    assert create[create.index("--security-opt") + 1] == "no-new-privileges"
    assert "--privileged" not in create
    assert "--cpus=2" in create
    # Published on loopback only, the control port first.
    publishes = [create[i + 1] for i, arg in enumerate(create) if arg == "-p"]
    assert publishes == [f"127.0.0.1::{SWEREX_PORT}", "127.0.0.1::8000"]
    mount = create[create.index("--mount") + 1]
    assert mount == (
        "type=image,source=assignment-swerex:s,target=/opt/swerex,"
        "image-subpath=opt/swerex,readonly=true"
    )
    command = create[create.index("assignment-testbed:t") + 1 :]
    assert command[:4] == ("timeout", "-s", "KILL", "90")
    assert command[4] == "/opt/swerex/venv/bin/swerex-remote"
    assert command[command.index("--port") + 1] == str(SWEREX_PORT)

    # The per-run files were copied in before the start was attempted ...
    copy_index = fake.calls.index(("cp", "-", f"{name}:/"))
    assert fake.inputs[copy_index]
    # ... and the failed start removed the container, waiting until it was gone.
    removal = [
        ("rm", "--force", name),
        ("container", "inspect", "--format", "{{.Id}}", name),
    ]
    assert fake.calls[-2:] == removal
    assert deployment.container is None
    asyncio.run(deployment.stop())
    assert fake.calls[-2:] == removal, "a second stop() must be a no-op"


def test_a_create_that_times_out_is_still_removed(fake_docker, monkeypatch):
    monkeypatch.setattr(docker_backend, "local_daemon", lambda: DAEMON)
    monkeypatch.setattr(
        docker_backend,
        "prepare_images",
        lambda spec, daemon, builder: images.PreparedImages(testbed="t", server="s"),
    )

    def handler(args):
        if args[0] == "create":
            raise DockerError("`docker create --name` did not finish within 900s.")
        if args[:2] == ("container", "inspect"):
            raise DockerError("No such container")
        return ""

    fake = fake_docker(handler)
    deployment = AssignmentDockerDeployment(image=ImageSpec(base="python:3.12"))
    with pytest.raises(DockerError, match="did not finish"):
        asyncio.run(deployment.start())

    (create,) = fake.called("create")
    assert ("rm", "--force", create[create.index("--name") + 1]) in fake.calls


def test_port_url_reads_the_loopback_publish(fake_docker):
    fake_docker(lambda args: "8000/tcp -> 127.0.0.1:49153\n8880/tcp -> 127.0.0.1:49154")
    deployment = AssignmentDockerDeployment(image=ImageSpec(base="python:3.12"))
    deployment.container = "assignment-sandbox-x"

    assert deployment.port_url(8000) == "http://127.0.0.1:49153"
    with pytest.raises(
        ValueError,
        match="Port 9000 was not forwarded. Available forwarded ports: 8000.",
    ):
        deployment.port_url(9000)
    deployment.container = None
