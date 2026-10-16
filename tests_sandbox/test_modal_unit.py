"""The Modal backend builds and starts sandboxes exactly as before the split."""

from __future__ import annotations

from pathlib import Path

import pytest

from assignment import env
from assignment.sandbox import ModalSandboxConfig, create_backend
from assignment.sandbox import modal as modal_backend
from assignment.utils.image import ImageSpec, is_ignored


class FakeImage:
    """Records the chain of Modal image calls instead of building anything."""

    def __init__(self, steps):
        self.steps = steps

    def pip_install(self, *packages):
        return FakeImage([*self.steps, ("pip_install", packages)])

    def add_local_file(self, source, target, copy=False):
        return FakeImage([*self.steps, ("add_local_file", source, target, copy)])


def test_old_import_paths_still_work():
    assert env._tls_port_configuration is modal_backend._tls_port_configuration
    assert env.AssignmentModalDeployment is modal_backend.AssignmentModalDeployment


def test_a_testbed_recipe_builds_the_same_modal_image_chain_as_before(monkeypatch):
    calls = []

    def from_dockerfile(path, **kwargs):
        calls.append((path, kwargs))
        return FakeImage([("from_dockerfile", path)])

    monkeypatch.setattr(modal_backend.modal.Image, "from_dockerfile", from_dockerfile)
    files = (Path("/src/chess_server.py"), Path("/src/sandbox_python.py"))
    spec = ImageSpec(
        dockerfile=Path("/task/Dockerfile"),
        context=Path("/chess_app"),
        pins=("fastapi==0.141.1", "starlette==1.6.0"),
        force_build=True,
    ).with_files({path: f"/opt/assignment/{path.name}" for path in files})

    image = modal_backend.to_modal_image(spec)

    assert calls == [
        (
            "/task/Dockerfile",
            {"context_dir": "/chess_app", "force_build": True, "ignore": is_ignored},
        )
    ]
    # Dockerfile, then the pins, then the files: the order the old code used.
    assert image.steps == [
        ("from_dockerfile", "/task/Dockerfile"),
        ("pip_install", ("fastapi==0.141.1", "starlette==1.6.0")),
        (
            "add_local_file",
            "/src/chess_server.py",
            "/opt/assignment/chess_server.py",
            True,
        ),
        (
            "add_local_file",
            "/src/sandbox_python.py",
            "/opt/assignment/sandbox_python.py",
            True,
        ),
    ]


class RecordingDeployment:
    def __init__(self, **kwargs):
        self.kwargs = kwargs


@pytest.fixture
def recorded(monkeypatch):
    monkeypatch.setattr(modal_backend, "AssignmentModalDeployment", RecordingDeployment)

    def make(**overrides):
        options = {
            "image": "python:3.12",
            "exposed_ports": (),
            "startup_timeout": 1,
            "runtime_timeout": 2,
            "deployment_timeout": 3,
            "install_pipx": True,
            "modal_sandbox_kwargs": None,
        } | overrides
        return create_backend(ModalSandboxConfig(), **options).deployment.kwargs

    return make


def test_exposed_ports_become_encrypted_ports(recorded):
    kwargs = recorded(
        exposed_ports=[8000], modal_sandbox_kwargs={"encrypted_ports": [9000], "cpu": 2}
    )
    assert kwargs["modal_sandbox_kwargs"] == {"encrypted_ports": [9000, 8000], "cpu": 2}


def test_other_images_and_options_pass_through_unchanged(recorded):
    kwargs = recorded(modal_sandbox_kwargs={"cpu": 2})
    assert kwargs == {
        "image": "python:3.12",
        "startup_timeout": 1,
        "runtime_timeout": 2,
        "deployment_timeout": 3,
        "install_pipx": True,
        "modal_sandbox_kwargs": {"cpu": 2},
    }
