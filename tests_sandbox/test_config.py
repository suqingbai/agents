"""Choosing the sandbox backend: `backend=`, then `SANDBOX`, then Modal."""

from __future__ import annotations

import dotenv
import pytest

from assignment.sandbox import config
from assignment.sandbox.config import (
    DockerSandboxConfig,
    ModalSandboxConfig,
    resolve_sandbox_config,
)


@pytest.fixture(autouse=True)
def clean_environment(monkeypatch):
    for name in ("SANDBOX", "SANDBOX_DOCKER_RUN_ARGS", "SANDBOX_DOCKER_BUILD_CONTEXT"):
        monkeypatch.delenv(name, raising=False)
    # The repository's own .env must not leak into these tests.
    monkeypatch.setattr(config, "load_dotenv", lambda: False)


def test_default_is_modal():
    assert resolve_sandbox_config() == ModalSandboxConfig()


def test_environment_variable_selects_docker(monkeypatch):
    monkeypatch.setenv("SANDBOX", " Docker ")
    monkeypatch.setenv("SANDBOX_DOCKER_RUN_ARGS", "--cpus=2 --memory '4g'")
    monkeypatch.setenv("SANDBOX_DOCKER_BUILD_CONTEXT", "remote-builder")

    assert resolve_sandbox_config() == DockerSandboxConfig(
        run_args=["--cpus=2", "--memory", "4g"], build_context="remote-builder"
    )


def test_explicit_backend_wins_over_the_environment(monkeypatch):
    monkeypatch.setenv("SANDBOX", "docker")
    assert resolve_sandbox_config("modal") == ModalSandboxConfig()

    monkeypatch.setenv("SANDBOX", "modal")
    assert resolve_sandbox_config("docker").type == "docker"


def test_a_config_object_is_used_as_is(monkeypatch):
    monkeypatch.setenv("SANDBOX", "modal")
    ready = DockerSandboxConfig(run_args=["--cpus=1"])
    assert resolve_sandbox_config(ready) is ready


def test_dotenv_is_read_but_the_environment_wins(monkeypatch, tmp_path):
    env_file = tmp_path / ".env"
    env_file.write_text("SANDBOX=docker\n")
    monkeypatch.setattr(config, "load_dotenv", lambda: dotenv.load_dotenv(env_file))

    assert resolve_sandbox_config().type == "docker"

    monkeypatch.setenv("SANDBOX", "modal")
    assert resolve_sandbox_config().type == "modal"


def test_a_foreign_sandbox_value_is_rejected_loudly(monkeypatch):
    # Gemini CLI sets SANDBOX inside its own sandboxes.
    monkeypatch.setenv("SANDBOX", "sandbox-exec")
    with pytest.raises(ValueError, match=r"SANDBOX='sandbox-exec'.*Gemini"):
        resolve_sandbox_config()


def test_an_unknown_backend_argument_names_the_argument():
    with pytest.raises(ValueError, match=r"backend='podman'"):
        resolve_sandbox_config("podman")
