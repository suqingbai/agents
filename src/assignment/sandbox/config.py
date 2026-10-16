"""Which sandbox backend to use, and its settings.

The backend is chosen once, in this order: an explicit `backend=` argument, then
the `SANDBOX` environment variable (also read from `.env`), then Modal.
"""

from __future__ import annotations

import os
import shlex
from typing import Annotated, Literal

from dotenv import load_dotenv
from pydantic import BaseModel, ConfigDict, Field

ENV_VAR = "SANDBOX"
DOCKER_RUN_ARGS_VAR = "SANDBOX_DOCKER_RUN_ARGS"
DOCKER_BUILD_CONTEXT_VAR = "SANDBOX_DOCKER_BUILD_CONTEXT"


class ModalSandboxConfig(BaseModel):
    """Run sandboxes on Modal, the default."""

    model_config = ConfigDict(extra="forbid")
    type: Literal["modal"] = "modal"


class DockerSandboxConfig(BaseModel):
    """Run sandboxes as containers on this machine's Docker daemon."""

    model_config = ConfigDict(extra="forbid")
    type: Literal["docker"] = "docker"
    # Extra `docker create` flags, e.g. ["--cpus=2", "--memory=4g", "--runtime=runsc"].
    run_args: list[str] = []
    # The buildx builder that builds images, e.g. a docker-container or remote
    # BuildKit builder. Its result is loaded into the local daemon. None is the
    # local daemon's own builder.
    build_context: str | None = None


SandboxConfig = Annotated[
    ModalSandboxConfig | DockerSandboxConfig, Field(discriminator="type")
]


def resolve_sandbox_config(
    explicit: str | ModalSandboxConfig | DockerSandboxConfig | None = None,
) -> ModalSandboxConfig | DockerSandboxConfig:
    """Pick the sandbox backend.

    Args:
        explicit: A backend name (`"modal"` or `"docker"`) or a ready config.
            None defers to the `SANDBOX` variable, then to Modal.

    Raises:
        ValueError: The name is not a backend. Other tools also set `SANDBOX`
            (Gemini CLI does inside its sandboxes), so an unknown value is an
            error rather than a silent fallback to Modal.
    """
    if isinstance(explicit, (ModalSandboxConfig, DockerSandboxConfig)):
        return explicit

    # The evaluation scripts never import assignment.agent, which is where .env
    # is otherwise loaded, so the agent and the grader could disagree on the
    # backend without this.
    load_dotenv()
    source = "backend" if explicit else ENV_VAR
    name = (explicit or os.environ.get(ENV_VAR) or "modal").strip().lower()
    if name == "modal":
        return ModalSandboxConfig()
    if name == "docker":
        return DockerSandboxConfig(
            run_args=shlex.split(os.environ.get(DOCKER_RUN_ARGS_VAR, "")),
            build_context=os.environ.get(DOCKER_BUILD_CONTEXT_VAR, "").strip() or None,
        )
    raise ValueError(
        f"{source}={name!r} is not a sandbox backend; use 'modal' or 'docker'. "
        f"If you did not set {ENV_VAR} yourself, another tool did (Gemini CLI sets it "
        f"inside its sandboxes): set {ENV_VAR}=modal or {ENV_VAR}=docker explicitly."
    )
