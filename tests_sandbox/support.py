"""Constants and helpers shared by the sandbox backend tests."""

from __future__ import annotations

from pathlib import Path

from assignment.sandbox.dockercli import docker
from assignment.sandbox.images import LABEL

ROOT = Path(__file__).resolve().parents[1]
CHESS_TASK = ROOT / "tasks" / "chess-terminal-move"
# Small, and the same base the chess testbed uses, so it is pulled once.
IMAGE = "python:3.12-slim"


def sandbox_containers() -> set[str]:
    """Names of every sandbox container on the daemon, running or not."""
    names = docker(
        "ps", "--all", "--filter", f"label={LABEL}=1", "--format", "{{.Names}}"
    )
    return set(names.split())
