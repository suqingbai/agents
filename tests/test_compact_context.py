"""Tests for the request `Agent.compact_context` sends to the model.

The API client is replaced by a recorder that answers with a canned summary, so
these run offline and only inspect the compaction request itself.
"""

import json
from types import SimpleNamespace

import pytest

from openai.types.chat import ChatCompletion

from assignment.agent.base import (
    COMPACTION_SYSTEM_PROMPT,
    COMPACTION_USER_PROMPT_TEMPLATE,
    Agent,
    build_compaction_old_prefix,
)

SYSTEM_PROMPT = "You are an autonomous software engineer."
TASK_PROMPT = "Resolve the issue below.\n\n<issue>\nprint hello, world\n</issue>"
SUMMARY = "## Objective\nPrint hello, world."


def make_call(call_id: str, command: str) -> dict:
    """One `execute` tool call, shaped as the agent stores it."""
    return {
        "id": call_id,
        "type": "function",
        "function": {"name": "execute", "arguments": json.dumps({"command": command})},
    }


def make_step(content: str, *calls: tuple[str, str, str]) -> list[dict]:
    """One assistant step followed by the tool observation of each of its calls.

    Each call is given as (call_id, command, observation).
    """
    action = {
        "role": "assistant",
        "content": content,
        "tool_calls": [make_call(call_id, command) for call_id, command, _ in calls],
    }
    observations = [
        {"role": "tool", "tool_call_id": call_id, "content": observation}
        for call_id, _, observation in calls
    ]
    return [action, *observations]


STEP_LS = make_step("Look around.", ("call_ls", "ls -la", "LS OUTPUT"))
STEP_CAT = make_step(
    "Read both files.",
    ("call_cat_a", "cat a.txt", "A OUTPUT"),
    ("call_cat_b", "cat b.txt", "B OUTPUT"),
)
STEP_RUN = make_step("Run the script.", ("call_run", "python hello.py", "RUN OUTPUT"))
STEP_TEST = make_step(
    "Run the tests and diff.",
    ("call_test", "pytest -q", "TEST OUTPUT"),
    ("call_diff", "git diff", "DIFF OUTPUT"),
)
STEPS = [STEP_LS, STEP_CAT, STEP_RUN, STEP_TEST]


def make_completion(content: str) -> ChatCompletion:
    """A plain text chat completion, as the API would return it."""
    return ChatCompletion.model_validate(
        {
            "id": "chatcmpl-test",
            "object": "chat.completion",
            "created": 0,
            "model": "gpt-4.1",
            "choices": [
                {
                    "index": 0,
                    "finish_reason": "stop",
                    "message": {"role": "assistant", "content": content},
                }
            ],
        }
    )


@pytest.fixture
def make_agent(monkeypatch):
    """Build an Agent holding `steps` in its state, with a recording API client.

    Returns the agent and the list of calls made to the API, each entry being
    the keyword arguments of one request.
    """
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.setenv("OPENAI_BASE_URL", "http://localhost:0/v1")

    def _make(steps: list[list[dict]], keep_recent_steps: int):
        agent = Agent(
            SimpleNamespace(),
            model="gpt-4.1",
            compaction_keep_recent_steps=keep_recent_steps,
        )
        agent.system_prompt = SYSTEM_PROMPT
        agent.task_prompt = TASK_PROMPT
        agent.agent_state = [message for step in steps for message in step]

        calls = []

        def create(**kwargs):
            calls.append(kwargs)
            return make_completion(SUMMARY)

        agent.client.chat.completions.create = create
        return agent, calls

    return _make


def expected_request(old_steps: list[list[dict]]) -> list[dict]:
    """The compaction request for a first compaction of `old_steps`."""
    old_prefix = [message for step in old_steps for message in step]
    return [
        {"role": "system", "content": COMPACTION_SYSTEM_PROMPT},
        {
            "role": "user",
            "content": COMPACTION_USER_PROMPT_TEMPLATE.format(
                original_task=TASK_PROMPT,
                old_prefix=build_compaction_old_prefix(old_prefix),
            ),
        },
    ]


def test_compaction_request_uses_compaction_system_prompt(make_agent):
    agent, calls = make_agent(STEPS, keep_recent_steps=1)

    agent.compact_context()

    assert len(calls) == 1
    assert calls[0]["messages"][0] == {"role": "system", "content": COMPACTION_SYSTEM_PROMPT}


def test_compaction_user_message_fills_the_template(make_agent):
    """The user message carries the original task and the serialized old prefix."""
    agent, calls = make_agent(STEPS, keep_recent_steps=1)

    agent.compact_context()

    assert calls[0]["messages"][1] == expected_request(STEPS[:-1])[1]


@pytest.mark.parametrize("keep_recent_steps", [1, 2, 3])
def test_recent_steps_are_left_out_of_the_old_prefix(make_agent, keep_recent_steps):
    """The last N steps, each with all its observations, are not compacted."""
    agent, calls = make_agent(STEPS, keep_recent_steps=keep_recent_steps)

    agent.compact_context()

    assert calls[0]["messages"] == expected_request(STEPS[:-keep_recent_steps])

    user_message = calls[0]["messages"][1]["content"]
    for step in STEPS[-keep_recent_steps:]:
        for message in step:
            assert message["content"] not in user_message


def test_kept_step_takes_all_its_tool_observations_with_it(make_agent):
    """A kept step with several tool calls keeps every one of its observations."""
    agent, calls = make_agent(STEPS, keep_recent_steps=1)

    agent.compact_context()

    user_message = calls[0]["messages"][1]["content"]
    assert "pytest -q" not in user_message
    assert "TEST OUTPUT" not in user_message
    assert "git diff" not in user_message
    assert "DIFF OUTPUT" not in user_message
    assert "RUN OUTPUT" in user_message
