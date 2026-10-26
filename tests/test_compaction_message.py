"""Tests for the user message sent to the context-compaction model call.

`build_compaction_user_message` turns the old prefix of `agent_state` (assistant
messages and their linked `tool` messages) into the plain-text user message
described by `COMPACTION_USER_PROMPT_TEMPLATE`.
"""

import json

from assignment.agent import base

TASK = "print hello, world to the terminal"
CLOSING = (
    "Write the working-memory summary of the old prefix above, "
    "using the required sections."
)


def build_compaction_user_message(*args, **kwargs):
    # Looked up at call time, so a missing function fails each test on its own
    # rather than aborting collection of the whole suite.
    return base.build_compaction_user_message(*args, **kwargs)


def make_call(call_id: str, name: str, arguments: dict) -> dict:
    """One tool call, shaped as the agent stores it on an assistant message."""
    return {
        "id": call_id,
        "type": "function",
        "function": {"name": name, "arguments": json.dumps(arguments)},
    }


def make_action(content: str, *calls: dict) -> dict:
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


def test_exact_message_with_task_and_previous_summary():
    """A representative later compaction, checked character for character."""
    old_prefix = [
        make_action("List the repository to find the script.", LS_CALL),
        make_observation("call_ls", LS_OUTPUT),
    ]

    message = build_compaction_user_message(
        old_prefix,
        previous_summary="## Objective\nPrint hello, world.",
        original_task=TASK,
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
        f"{CLOSING}"
    )


def test_exact_message_for_first_compaction_without_task():
    """No task and no previous summary: both are left out entirely."""
    old_prefix = [
        make_action("List the repository to find the script.", LS_CALL),
        make_observation("call_ls", LS_OUTPUT),
    ]

    message = build_compaction_user_message(old_prefix)

    assert message == (
        "<old_prefix>\n"
        "[assistant action]\n"
        "List the repository to find the script.\n"
        '[tool call] execute {"command": "ls -la"}\n'
        "[tool observation]\n"
        "<output>hello.py\nREADME.md\n</output>\n<returncode>0</returncode>\n"
        "</old_prefix>\n"
        "\n"
        f"{CLOSING}"
    )


def test_task_context_holds_the_original_task():
    old_prefix = [make_action("Look around.", LS_CALL), make_observation("call_ls", LS_OUTPUT)]

    message = build_compaction_user_message(old_prefix, original_task=TASK)

    assert message.startswith(f"<task_context>\n{TASK}\n</task_context>\n\n<old_prefix>\n")


def test_task_context_block_is_omitted_without_a_task():
    """No task means no block at all, never an empty one."""
    old_prefix = [make_action("Look around.", LS_CALL), make_observation("call_ls", LS_OUTPUT)]

    for missing in (None, ""):
        message = build_compaction_user_message(old_prefix, original_task=missing)

        assert "task_context" not in message
        assert message.startswith("<old_prefix>\n")


def test_first_compaction_has_no_previous_summary_line():
    old_prefix = [make_action("Look around.", LS_CALL), make_observation("call_ls", LS_OUTPUT)]

    for missing in (None, ""):
        message = build_compaction_user_message(old_prefix, previous_summary=missing)

        assert "[previous summary]" not in message
        assert "<old_prefix>\n[assistant action]\n" in message


def test_previous_summary_comes_first_as_bare_text():
    """The summary opens the old prefix, without wrapper sentence or tags."""
    summary = "## Objective\nPrint hello, world.\n## Blockers\nnone"
    old_prefix = [make_action("Look around.", LS_CALL), make_observation("call_ls", LS_OUTPUT)]

    message = build_compaction_user_message(old_prefix, previous_summary=summary)

    assert f"<old_prefix>\n[previous summary]\n{summary}\n[assistant action]\n" in message
    assert "<summary>" not in message
    assert message.count(summary) == 1


def test_single_action_with_one_tool_call():
    old_prefix = [
        make_action("Run the script.", make_call("call_run", "execute", {"command": "python hello.py"})),
        make_observation("call_run", "<output>hello, world\n</output>\n<returncode>0</returncode>"),
    ]

    message = build_compaction_user_message(old_prefix)

    assert (
        "<old_prefix>\n"
        "[assistant action]\n"
        "Run the script.\n"
        '[tool call] execute {"command": "python hello.py"}\n'
        "[tool observation]\n"
        "<output>hello, world\n</output>\n<returncode>0</returncode>\n"
        "</old_prefix>\n"
    ) in message


def test_multiple_tool_calls_are_each_followed_by_their_own_observation():
    """Observations are paired by tool_call_id, in the order of the calls."""
    old_prefix = [
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

    message = build_compaction_user_message(old_prefix)

    assert (
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
        "SKILL BODY\n"
        "</old_prefix>\n"
    ) in message


def test_several_actions_keep_their_original_order():
    old_prefix = [
        make_action("First look.", LS_CALL),
        make_observation("call_ls", LS_OUTPUT),
        make_action("Then run.", make_call("call_run", "execute", {"command": "python hello.py"})),
        make_observation("call_run", "hello, world"),
    ]

    message = build_compaction_user_message(old_prefix)

    assert (
        "[assistant action]\n"
        "First look.\n"
        '[tool call] execute {"command": "ls -la"}\n'
        "[tool observation]\n"
        f"{LS_OUTPUT}\n"
        "[assistant action]\n"
        "Then run.\n"
        '[tool call] execute {"command": "python hello.py"}\n'
        "[tool observation]\n"
        "hello, world\n"
        "</old_prefix>\n"
    ) in message


def test_assistant_action_without_tool_calls():
    """A reply with no tool call (the loop keeps it) is listed as text alone."""
    old_prefix = [
        make_action("I think the script already exists."),
        make_action("Check it.", LS_CALL),
        make_observation("call_ls", LS_OUTPUT),
    ]

    message = build_compaction_user_message(old_prefix)

    assert (
        "<old_prefix>\n"
        "[assistant action]\n"
        "I think the script already exists.\n"
        "[assistant action]\n"
        "Check it.\n"
        '[tool call] execute {"command": "ls -la"}\n'
    ) in message


def test_tool_output_is_used_as_is():
    """Observations are not truncated or reformatted again."""
    long_output = "x" * 9_000 + "\n[1000 characters elided; read a narrower range]\n" + "y" * 900
    old_prefix = [make_action("Read it.", LS_CALL), make_observation("call_ls", long_output)]

    message = build_compaction_user_message(old_prefix)

    assert f"[tool observation]\n{long_output}\n</old_prefix>" in message


def test_old_prefix_is_plain_text_not_role_messages():
    old_prefix = [make_action("Look around.", LS_CALL), make_observation("call_ls", LS_OUTPUT)]

    message = build_compaction_user_message(old_prefix)

    assert isinstance(message, str)
    assert '"role"' not in message
    assert "call_ls" not in message, "tool_call_id is used for pairing, not shown"


def test_system_and_task_messages_are_not_included():
    """Only assistant actions and their observations make up the old prefix."""
    old_prefix = [
        {"role": "system", "content": "SYSTEM PROMPT TEXT"},
        {"role": "user", "content": "TASK PROMPT TEXT"},
        make_action("Look around.", LS_CALL),
        make_observation("call_ls", LS_OUTPUT),
    ]

    message = build_compaction_user_message(old_prefix, original_task=TASK)

    assert "SYSTEM PROMPT TEXT" not in message
    assert "TASK PROMPT TEXT" not in message
    assert "<old_prefix>\n[assistant action]\nLook around.\n" in message
