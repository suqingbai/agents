"""The chess sandbox and the grader on Docker, mirroring the Modal tests."""

from __future__ import annotations

import base64
import difflib
import subprocess

import pytest
from support import CHESS_TASK, ROOT

from assignment.eval import EvaluationSpec, TestStatus, evaluate
from assignment.task import Task

pytestmark = [pytest.mark.docker, pytest.mark.usefixtures("no_leaked_containers")]


@pytest.fixture(scope="module")
def chess_sandbox_module():
    # chess_sandbox imports the agent package, which is the student's work in
    # progress; skip rather than error while it does not import.
    return pytest.importorskip(
        "assignment.chess_sandbox",
        reason="assignment.agent does not import yet",
        exc_type=ImportError,
    )


def test_chess_app_is_playable_through_the_docker_sandbox(chess_sandbox_module):
    task = Task.load(CHESS_TASK)
    with chess_sandbox_module.ChessSandbox(backend="docker") as sandbox:
        assert sandbox.server_url.startswith("http://127.0.0.1:")

        current = sandbox.state()
        assert current["turn"] == "white"
        assert current["history"] == []
        with pytest.raises(chess_sandbox_module.IllegalMove):
            sandbox.play("e2e5")
        assert sandbox.reset()["history"] == []

        # A model-written snippet runs inside, against the local server.
        snippet = base64.b64encode(b"print('snippet ran')").decode()
        ran = sandbox.execute(
            f"python /opt/assignment/sandbox_python.py {sandbox.port} {snippet}"
        )
        assert '"stdout": "snippet ran\\n"' in ran["output"]

        # /testbed is exactly the task's source at its base commit, pins on top.
        tracked = subprocess.run(
            ["git", "-C", str(task.source), "ls-files"],
            capture_output=True,
            text=True,
            check=True,
        ).stdout.split()
        inside = sandbox.execute("git ls-files", cwd="/testbed")["output"].split()
        assert sorted(inside) == sorted(tracked)
        assert sandbox.execute("git status --porcelain", cwd="/testbed")["output"] == ""
        frozen = sandbox.execute("python -m pip freeze")["output"].lower().split()
        assert {pin.lower() for pin in task.pins} <= set(frozen)


def noop_patch() -> str:
    """A patch that applies cleanly and changes no behaviour."""
    original = (ROOT / "chess_app" / "README.md").read_text().splitlines(keepends=True)
    changed = [*original, "<!-- applied by the sandbox tests -->\n"]
    diff = difflib.unified_diff(original, changed, "a/README.md", "b/README.md")
    return "diff --git a/README.md b/README.md\n" + "".join(diff)


def test_grading_the_public_chess_spec_on_docker(docker_daemon):
    spec = EvaluationSpec.load(CHESS_TASK / "public_tests")

    report = evaluate(
        spec, patch=noop_patch(), task=Task.load(CHESS_TASK), backend="docker"
    )

    assert not report.error, report.error
    assert report.patch_applied
    # The unfixed engine fails the graded test and keeps every other one.
    assert set(report.fail_to_pass.values()) != {TestStatus.PASSED}
    assert set(report.pass_to_pass.values()) == {TestStatus.PASSED}
    assert len(report.pass_to_pass) == len(spec.pass_to_pass)
    assert not report.resolved
