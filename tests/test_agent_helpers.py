"""Generic agent helpers, tested with non-database tools to show they are reusable beyond databases."""

import pytest
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from agent_fakes import ScriptedChatModel, add, echo, text, tool_call
from dllmforge.agent_core import SimpleAgent
from dllmforge.agent_messages import extract_answer_and_steps, parse_json_object
from dllmforge.agent_tools import think, to_langchain_tools
from dllmforge.llm_factory import create_chat_model, uses_text_routing

# ── agent_tools ───────────────────────────────────────────────────────────────


def test_to_langchain_tools_keeps_names_docs_and_schema():
    tools = to_langchain_tools([add, echo, think])
    assert [t.name for t in tools] == ["add", "echo", "think"]
    assert tools[0].invoke({"a": 2, "b": 3}) == 5
    assert set(tools[0].args) == {"a", "b"}
    assert "Add two integers" in tools[0].description
    assert to_langchain_tools(tools) == tools  # existing tools pass through


def test_think_accepts_plan_or_thought():
    assert think(plan="look up") == "Plan recorded: look up"
    assert think(thought="x") == "Plan recorded: x"
    assert think() == "Plan recorded."


@pytest.mark.parametrize("routing", [False, True])
def test_converted_tools_run_in_simple_agent(routing):
    if routing:
        llm = ScriptedChatModel(
            responses=[text('{"tool": "add", "args": {"a": 2, "b": 3}}'),
                       text('{"final_answer": "5"}')])
    else:
        llm = ScriptedChatModel(responses=[tool_call("add", {"a": 2, "b": 3}), text("The sum is 5.")])
    agent = SimpleAgent(llm=llm, enable_text_tool_routing=routing)
    for tool in to_langchain_tools([add, echo]):
        agent.add_tool(tool)
    agent.compile()
    answer, steps = extract_answer_and_steps(agent.app.invoke({"messages": [("user", "2+3?")]})["messages"], routing)
    assert answer in {"5", "The sum is 5."}
    assert [(s.tool, s.content) for s in steps] == [("add", "5")]


# ── agent_messages ────────────────────────────────────────────────────────────


def test_extract_native_shape_with_args_and_parser():
    messages = [
        HumanMessage("q"),
        tool_call("echo", {"text": "hi"}, "c1"),
        ToolMessage(content="Echo: hi", name="echo", tool_call_id="c1"),
        AIMessage("done"),
    ]
    answer, steps = extract_answer_and_steps(messages, False, {"echo": lambda c: {"upper": c.upper()}})
    assert answer == "done"
    assert steps[0].args == {"text": "hi"} and steps[0].details == {"upper": "ECHO: HI"}


def test_extract_text_routing_shape():
    messages = [
        HumanMessage("q"),
        HumanMessage("Tool 'echo' result: Echo: a\nmore"),
        AIMessage('{"final_answer": "line1\nline2"}'),
    ]
    answer, steps = extract_answer_and_steps(messages, True)
    assert answer == "line1\nline2"
    assert steps[0].tool == "echo" and steps[0].content == "Echo: a\nmore"


def test_extract_text_routing_plain_answer_and_skips_tool_json():
    assert extract_answer_and_steps([AIMessage("plain")], True)[0] == "plain"
    assert extract_answer_and_steps([AIMessage("answer"), AIMessage('{"tool": "x", "args": {}}')], True)[0] == "answer"
    assert extract_answer_and_steps([HumanMessage("q")], True) == ("", [])


def test_parse_json_object():
    assert parse_json_object('x {"a": 1} y') == {"a": 1}
    assert parse_json_object("no json") is None


# ── llm_factory ───────────────────────────────────────────────────────────────


def test_routing_providers():
    assert uses_text_routing("Deltares") and uses_text_routing("ollama")
    assert not uses_text_routing("azure-openai")


def test_unknown_provider():
    with pytest.raises(ValueError, match="Unknown provider"):
        create_chat_model("nope")


def test_deltares_base_url_from_env(monkeypatch):
    monkeypatch.setenv("DELTARES_LLM_BASE_URL", "https://example.invalid")
    llm = create_chat_model("deltares", "m1")
    assert llm.base_url == "https://example.invalid" and llm.model_name == "m1"
