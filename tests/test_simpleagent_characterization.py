"""Characterization tests for ``dllmforge.agent_core.SimpleAgent``.

These tests pin the *current* behaviour of SimpleAgent (which must not be modified) so that the
database layer can build on it safely. Some assertions document behaviour that looks unintended;
those are marked "documents current behaviour" and must only change together with SimpleAgent.
"""
import pytest
from langchain_core.messages import SystemMessage
from langchain_core.tools import StructuredTool

from agent_fakes import ScriptedChatModel, add, text, tool_call
from dllmforge.agent_core import SimpleAgent

ADD_TOOL = StructuredTool.from_function(add)


def _agent(llm, text_routing: bool, max_tool_iterations: int = 3) -> SimpleAgent:
    agent = SimpleAgent(system_message="SYSTEM-PROMPT",
                        llm=llm,
                        enable_text_tool_routing=text_routing,
                        max_tool_iterations=max_tool_iterations)
    agent.add_tool(ADD_TOOL)
    agent.compile()
    return agent


def _system_messages(messages):
    return [m for m in messages if m.type == "system"]


class TestNativeToolCalling:

    def test_tool_is_executed_and_final_answer_returned(self):
        llm = ScriptedChatModel(responses=[tool_call("add", {"a": 2, "b": 3}), text("The sum is 5.")])
        result = _agent(llm, text_routing=False).app.invoke({"messages": [("user", "2+3?")]})

        tool_messages = [m for m in result["messages"] if m.type == "tool"]
        assert [(m.name, m.content) for m in tool_messages] == [("add", "5")]
        assert result["messages"][-1].content == "The sum is 5."

    def test_system_message_is_added_on_every_model_call(self):
        """P12 — documents current behaviour: a new SystemMessage is added per LLM call."""
        llm = ScriptedChatModel(responses=[tool_call("add", {"a": 2, "b": 3}), text("5")])
        result = _agent(llm, text_routing=False).app.invoke({"messages": [("user", "2+3?")]})

        assert len(llm.calls) == 2
        assert len(_system_messages(result["messages"])) == 2
        # The second LLM call therefore sees a system message after the user turn.
        assert len(_system_messages(llm.calls[1])) == 2

    def test_system_message_supplied_in_input_is_not_duplicated(self):
        """Workaround used by DatabaseAgent: pass the system message as the first input message."""
        llm = ScriptedChatModel(responses=[tool_call("add", {"a": 2, "b": 3}), text("5")])
        agent = _agent(llm, text_routing=False)
        result = agent.app.invoke({"messages": [SystemMessage(content="SYSTEM-PROMPT"), ("user", "2+3?")]})

        assert len(_system_messages(result["messages"])) == 1
        assert all(len(_system_messages(call)) == 1 for call in llm.calls)


class TestTextRouting:

    def test_tool_result_is_appended_as_human_message(self):
        llm = ScriptedChatModel(
            responses=[text('{"tool": "add", "args": {"a": 2, "b": 3}}'),
                       text('{"final_answer": "5"}')])
        result = _agent(llm, text_routing=True).app.invoke({"messages": [("user", "2+3?")]})

        contents = [m.content for m in result["messages"]]
        assert "Tool 'add' result: 5" in contents
        assert result["messages"][-1].type == "ai"
        assert result["messages"][-1].content == '{"final_answer": "5"}'

    def test_routing_instructions_are_appended_to_system_message(self):
        llm = ScriptedChatModel(responses=[text('{"final_answer": "ok"}')])
        _agent(llm, text_routing=True).app.invoke({"messages": [("user", "hi")]})

        system = llm.calls[0][0]
        assert system.type == "system"
        assert system.content.startswith("SYSTEM-PROMPT")
        assert "You can call tools by responding ONLY with a JSON object." in system.content
        assert "- add" in system.content

    def test_iteration_limit_ends_without_final_answer(self):
        """Documents current behaviour: when max_tool_iterations is exhausted no AI message is returned."""
        llm = ScriptedChatModel(responses=[text('{"tool": "add", "args": {"a": 1, "b": 1}}')])
        result = _agent(llm, text_routing=True, max_tool_iterations=2).app.invoke({"messages": [("user", "loop")]})

        assert len(llm.calls) == 2
        assert result["messages"][-1].type == "human"
        assert result["messages"][-1].content == "Tool 'add' result: 2"
        assert not [m for m in result["messages"] if m.type == "ai"]

    def test_two_json_objects_in_one_response_are_treated_as_final_answer(self):
        """Documents current behaviour: multiple JSON objects cannot be parsed, so no tool runs."""
        content = '{"tool": "add", "args": {"a": 2, "b": 3}} {"final_answer": "5"}'
        llm = ScriptedChatModel(responses=[text(content)])
        result = _agent(llm, text_routing=True).app.invoke({"messages": [("user", "2+3?")]})

        assert len(llm.calls) == 1
        assert result["messages"][-1].content == content


def test_create_simple_workflow_then_compile_raises(monkeypatch):
    """P13 — documents current behaviour: compile() rebuilds the workflow, so calling
    create_simple_workflow() first (as create_basic_agent does) adds the 'agent' node twice."""
    agent = SimpleAgent(llm=ScriptedChatModel(responses=[text("hi")]))
    agent.create_simple_workflow()
    with pytest.raises(ValueError, match="already present"):
        agent.compile()


def test_create_basic_agent_raises(monkeypatch):
    """P13 via the public helper, with the provider-backed LLM replaced by a fake."""
    import dllmforge.langchain_api as langchain_api

    class _FakeAPI:

        def __init__(self, *args, **kwargs):
            self.llm = ScriptedChatModel(responses=[text("hi")])

    monkeypatch.setattr(langchain_api, "LangchainAPI", _FakeAPI)
    from dllmforge.agent_core import create_basic_agent

    with pytest.raises(ValueError, match="already present"):
        create_basic_agent("system")
