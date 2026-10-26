"""Tests for the old-prefix text sent to the context-compaction model call.

`build_compaction_old_prefix` turns the old prefix of `agent_state` (assistant
messages and their linked `tool` messages) into the plain text that
`COMPACTION_USER_PROMPT_TEMPLATE` places between its `<old_prefix>` tags.
"""

import json

import pytest

from assignment.agent.base import (
    COMPACTION_USER_PROMPT_TEMPLATE,
    build_compaction_old_prefix,
)

TASK = "print hello, world to the terminal"


def make_call(call_id: str, name: str, arguments: dict) -> dict:
    """One tool call, shaped as the agent stores it on an assistant message."""
    return {
        "id": call_id,
        "type": "function",
        "function": {"name": name, "arguments": json.dumps(arguments)},
    }


def make_action(content: str | None, *calls: dict) -> dict:
    """An assistant message, with tool calls only when some are given."""
    message = {"role": "assistant", "content": content}
    if calls:
        message["tool_calls"] = list(calls)
    return message


def make_observation(call_id: str, content: str) -> dict:
    """A tool message answering the call with `call_id`."""
    return {"role": "tool", "tool_call_id": call_id, "content": content}


LS_CALL = make_call("call_ls", "execute", {"command": "ls -la"})
LS_OUTPUT = "<output>hello.py\nREADME.md\n</output>\n<returncode>0</returncode>"
LS_PREFIX = [
    make_action("List the repository to find the script.", LS_CALL),
    make_observation("call_ls", LS_OUTPUT),
]


def test_exact_old_prefix_with_previous_summary():
    """A representative later compaction, checked character for character."""
    old_prefix = build_compaction_old_prefix(
        LS_PREFIX, previous_summary="## Objective\nPrint hello, world."
    )

    assert old_prefix == (
        "[previous summary]\n"
        "## Objective\nPrint hello, world.\n"
        "[assistant action]\n"
        "List the repository to find the script.\n"
        '[tool call] execute {"command": "ls -la"}\n'
        "[tool observation]\n"
        "<output>hello.py\nREADME.md\n</output>\n<returncode>0</returncode>"
    )


def test_exact_old_prefix_for_first_compaction():
    """No previous summary: its line is left out entirely."""
    old_prefix = build_compaction_old_prefix(LS_PREFIX)

    assert old_prefix == (
        "[assistant action]\n"
        "List the repository to find the script.\n"
        '[tool call] execute {"command": "ls -la"}\n'
        "[tool observation]\n"
        "<output>hello.py\nREADME.md\n</output>\n<returncode>0</returncode>"
    )


def test_old_prefix_fills_the_user_prompt_template():
    """The function's output and the template fit into the full user message."""
    message = COMPACTION_USER_PROMPT_TEMPLATE.format(
        original_task=TASK,
        old_prefix=build_compaction_old_prefix(
            LS_PREFIX, previous_summary="## Objective\nPrint hello, world."
        ),
    )

    assert message == (
        "<task_context>\n"
        "print hello, world to the terminal\n"
        "</task_context>\n"
        "\n"
        "<old_prefix>\n"
        "[previous summary]\n"
        "## Objective\nPrint hello, world.\n"
        "[assistant action]\n"
        "List the repository to find the script.\n"
        '[tool call] execute {"command": "ls -la"}\n'
        "[tool observation]\n"
        "<output>hello.py\nREADME.md\n</output>\n<returncode>0</returncode>\n"
        "</old_prefix>\n"
        "\n"
        "Write the working-memory summary of the old prefix above, "
        "using the required sections."
    )


@pytest.mark.parametrize("missing", [None, ""])
def test_first_compaction_has_no_previous_summary_line(missing):
    old_prefix = build_compaction_old_prefix(LS_PREFIX, previous_summary=missing)

    assert "[previous summary]" not in old_prefix
    assert old_prefix.startswith("[assistant action]\n")


def test_previous_summary_comes_first_as_bare_text():
    """The summary opens the old prefix, without wrapper sentence or tags."""
    summary = "## Objective\nPrint hello, world.\n## Blockers\nnone"

    old_prefix = build_compaction_old_prefix(LS_PREFIX, previous_summary=summary)

    assert old_prefix.startswith(f"[previous summary]\n{summary}\n[assistant action]\n")
    assert "<summary>" not in old_prefix
    assert old_prefix.count(summary) == 1


def test_single_action_with_one_tool_call():
    old_prefix = build_compaction_old_prefix(
        [
            make_action(
                "Run the script.",
                make_call("call_run", "execute", {"command": "python hello.py"}),
            ),
            make_observation(
                "call_run", "<output>hello, world\n</output>\n<returncode>0</returncode>"
            ),
        ]
    )

    assert old_prefix == (
        "[assistant action]\n"
        "Run the script.\n"
        '[tool call] execute {"command": "python hello.py"}\n'
        "[tool observation]\n"
        "<output>hello, world\n</output>\n<returncode>0</returncode>"
    )


def test_multiple_tool_calls_are_each_followed_by_their_own_observation():
    """Observations are paired by tool_call_id, in the order of the calls."""
    old_prefix = build_compaction_old_prefix(
        [
            make_action(
                "Read both files.",
                make_call("call_a", "execute", {"command": "cat a.txt"}),
                make_call("call_b", "execute", {"command": "cat b.txt"}),
                make_call("call_c", "invoke_skill", {"name": "testing"}),
            ),
            # Stored out of call order, to show pairing goes by tool_call_id.
            make_observation("call_c", "SKILL BODY"),
            make_observation("call_a", "A CONTENT"),
            make_observation("call_b", "B CONTENT"),
        ]
    )

    assert old_prefix == (
        "[assistant action]\n"
        "Read both files.\n"
        '[tool call] execute {"command": "cat a.txt"}\n'
        "[tool observation]\n"
        "A CONTENT\n"
        '[tool call] execute {"command": "cat b.txt"}\n'
        "[tool observation]\n"
        "B CONTENT\n"
        '[tool call] invoke_skill {"name": "testing"}\n'
        "[tool observation]\n"
        "SKILL BODY"
    )


def test_several_actions_keep_their_original_order():
    old_prefix = build_compaction_old_prefix(
        [
            make_action("First look.", LS_CALL),
            make_observation("call_ls", LS_OUTPUT),
            make_action(
                "Then run.",
                make_call("call_run", "execute", {"command": "python hello.py"}),
            ),
            make_observation("call_run", "hello, world"),
        ]
    )

    assert old_prefix == (
        "[assistant action]\n"
        "First look.\n"
        '[tool call] execute {"command": "ls -la"}\n'
        "[tool observation]\n"
        f"{LS_OUTPUT}\n"
        "[assistant action]\n"
        "Then run.\n"
        '[tool call] execute {"command": "python hello.py"}\n'
        "[tool observation]\n"
        "hello, world"
    )


def test_assistant_action_without_tool_calls():
    """A reply with no tool call (the loop keeps it) is listed as text alone."""
    old_prefix = build_compaction_old_prefix(
        [
            make_action("I think the script already exists."),
            make_action("Check it.", LS_CALL),
            make_observation("call_ls", LS_OUTPUT),
        ]
    )

    assert old_prefix == (
        "[assistant action]\n"
        "I think the script already exists.\n"
        "[assistant action]\n"
        "Check it.\n"
        '[tool call] execute {"command": "ls -la"}\n'
        "[tool observation]\n"
        f"{LS_OUTPUT}"
    )


@pytest.mark.parametrize("content", [None, ""])
def test_assistant_action_without_text_keeps_label_and_skips_text_line(content):
    """A tool-calling reply with no text has no empty line after its label."""
    old_prefix = build_compaction_old_prefix(
        [make_action(content, LS_CALL), make_observation("call_ls", LS_OUTPUT)]
    )

    assert old_prefix == (
        "[assistant action]\n"
        '[tool call] execute {"command": "ls -la"}\n'
        "[tool observation]\n"
        f"{LS_OUTPUT}"
    )


def test_tool_output_is_used_as_is():
    """Observations are not truncated or reformatted again."""
    long_output = "x" * 9_000 + "\n[1000 characters elided; read a narrower range]\n" + "y" * 900

    old_prefix = build_compaction_old_prefix(
        [make_action("Read it.", LS_CALL), make_observation("call_ls", long_output)]
    )

    assert old_prefix.endswith(f"[tool observation]\n{long_output}")


def test_old_prefix_is_plain_text_not_role_messages():
    old_prefix = build_compaction_old_prefix(LS_PREFIX)

    assert isinstance(old_prefix, str)
    assert '"role"' not in old_prefix
    assert "call_ls" not in old_prefix, "tool_call_id is used for pairing, not shown"
