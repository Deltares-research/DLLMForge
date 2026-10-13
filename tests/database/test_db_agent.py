"""DatabaseAgent with a scripted fake LLM in both SimpleAgent modes (no network)."""
from langchain_core.messages import SystemMessage

from agent_fakes import ScriptedChatModel, text, tool_call
from dllmforge.database.agent import DatabaseAgent
from db_fixtures import add_custom_capabilities, make_project

SQL = "SELECT Name FROM Rivers WHERE RiverId = 1"


def test_native_mode_runs_sql_and_reports_steps(tmp_path):
    llm = ScriptedChatModel(responses=[
        tool_call("think", {"plan": "query Rivers"}, "c1"),
        tool_call("execute_rivers_sql", {"query": SQL}, "c2"),
        text("The river is Lek."),
    ])
    agent = DatabaseAgent(make_project(tmp_path), model_provider="azure-openai", llm=llm)
    answer, steps = agent.ask("Which river has id 1?")

    assert answer == "The river is Lek."
    assert [s.tool for s in steps] == ["think", "execute_rivers_sql"]
    assert steps[1].args == {"query": SQL}
    assert steps[1].details["sql"] == [SQL] and "Lek" in steps[1].details["results"]
    # Exactly one system message per model call (SimpleAgent P12 workaround), carrying the DB instructions.
    for call in llm.calls:
        assert sum(isinstance(m, SystemMessage) for m in call) == 1
        assert "execute_rivers_sql" in call[0].content


def test_text_routing_mode(tmp_path):
    llm = ScriptedChatModel(responses=[
        text('{"tool": "execute_rivers_sql", "args": {"query": "%s"}}' % SQL),
        text('{"final_answer": "Lek"}'),
    ])
    agent = DatabaseAgent(make_project(tmp_path), model_provider="ollama", llm=llm)
    assert agent.text_routing
    answer, steps = agent.ask("Which river has id 1?")
    assert answer == "Lek"
    assert steps[0].tool == "execute_rivers_sql" and steps[0].details["sql"] == [SQL]
    system = llm.calls[0][0]
    assert isinstance(system, SystemMessage) and '{"final_answer"' in system.content  # added by SimpleAgent


def test_unsafe_sql_is_refused_through_agent(tmp_path):
    llm = ScriptedChatModel(responses=[tool_call("execute_rivers_sql", {"query": "DELETE FROM Rivers"}), text("no")])
    agent = DatabaseAgent(make_project(tmp_path), model_provider="openai", llm=llm)
    _, steps = agent.ask("delete everything")
    assert steps[0].content.startswith("ERROR: Only SELECT queries are permitted.")


def test_history_and_env_provider(tmp_path, monkeypatch):
    monkeypatch.setenv("RIVERS_MODEL_PROVIDER", "deltares")
    llm = ScriptedChatModel(responses=[text('{"final_answer": "ok"}')])
    agent = DatabaseAgent(make_project(tmp_path), llm=llm)
    assert agent.model_provider == "deltares" and agent.text_routing
    agent.ask("second", history=[("first", "answer one")])
    contents = [m.content for m in llm.calls[0]]
    assert "first" in contents and "answer one" in contents
    assert agent.tool_names == ["get_schema", "list_tables", "describe_table", "execute_rivers_sql", "think"]


CUSTOM_SQL = "SELECT RiverId, Name, LengthKm FROM Rivers WHERE Name = 'Waal'"


def test_custom_capabilities_native_mode(tmp_path):
    llm = ScriptedChatModel(responses=[
        tool_call("river_by_name", {"name": "Waal"}, "c1"),
        tool_call("km_to_miles", {"km": 20.0}, "c2"),
        text("Waal is 20 km (12.4 miles)."),
    ])
    agent = DatabaseAgent(add_custom_capabilities(make_project(tmp_path)), model_provider="openai", llm=llm)
    assert agent.tool_names == [
        "get_schema", "list_tables", "describe_table", "execute_rivers_sql", "river_by_name", "km_to_miles", "think"
    ]
    _, steps = agent.ask("How long is the Waal in miles?")
    assert [s.tool for s in steps] == ["river_by_name", "km_to_miles"]
    assert steps[0].details["sql"] == [CUSTOM_SQL] and "Waal" in steps[0].details["results"]
    assert steps[1].content == "12.4 miles" and steps[1].details["sql"] == []


def test_custom_capabilities_text_routing_mode(tmp_path):
    llm = ScriptedChatModel(responses=[
        text('{"tool": "river_by_name", "args": {"name": "Waal"}}'),
        text('{"tool": "km_to_miles", "args": {"km": 20.0}}'),
        text('{"final_answer": "12.4 miles"}'),
    ])
    agent = DatabaseAgent(add_custom_capabilities(make_project(tmp_path)), model_provider="ollama", llm=llm)
    answer, steps = agent.ask("How long is the Waal in miles?")
    assert answer == "12.4 miles"
    assert [s.tool for s in steps] == ["river_by_name", "km_to_miles"]
    assert steps[0].details["sql"] == [CUSTOM_SQL]
    assert "12.4 miles" in steps[1].content
