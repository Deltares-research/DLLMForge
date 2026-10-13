"""Database MCP server, exercised in-process (no network, no LLM)."""
import anyio
import pytest

pytest.importorskip("mcp")

from mcp import Client  # noqa: E402

from dllmforge import agent_core  # noqa: E402
from dllmforge.database.agent import DatabaseAgent  # noqa: E402
from dllmforge.database.mcp_server import build_database_mcp_server  # noqa: E402
from agent_fakes import ScriptedChatModel, text  # noqa: E402
from db_fixtures import add_custom_capabilities, make_project  # noqa: E402

SQL = "SELECT Name FROM Rivers WHERE RiverId = 2"


def _text(result) -> str:
    return "".join(getattr(part, "text", "") for part in result.content)


@pytest.fixture
def no_simple_agent(monkeypatch):

    def fail(*args, **kwargs):
        raise AssertionError("The MCP server must not start SimpleAgent")

    monkeypatch.setattr(agent_core.SimpleAgent, "__init__", fail)


def test_tools_resources_prompt(tmp_path, no_simple_agent):
    server = build_database_mcp_server(make_project(tmp_path, notes="## Rivers\nkm\n"), require_ack=False)

    async def main():
        tools = {t.name: t for t in await server.list_tools()}
        resources = {str(r.uri) for r in await server.list_resources()}
        prompts = [p.name for p in await server.list_prompts()]
        result = await server.call_tool("execute_rivers_sql", {"query": SQL})
        refused = await server.call_tool("execute_rivers_sql", {"query": "DROP TABLE Rivers"})
        schema = await server.read_resource("rivers://schema")
        return tools, resources, prompts, result, refused, list(schema)

    tools, resources, prompts, result, refused, schema = anyio.run(main)
    assert list(tools) == [
        "get_instructions", "get_schema", "list_tables", "describe_table", "execute_rivers_sql", "think"
    ]
    assert tools["execute_rivers_sql"].annotations.read_only_hint is True
    assert resources == {"rivers://instructions", "rivers://schema", "rivers://docs/rivers"}
    assert prompts == ["rivers_assistant"]
    assert "Waal" in _text(result) and _text(result).endswith(f"Executed SQL:\n{SQL}")
    assert "Only SELECT queries are permitted" in _text(refused)
    assert "TABLE Rivers" in schema[0].content


def test_instruction_ack_through_client(tmp_path, no_simple_agent):
    server = build_database_mcp_server(make_project(tmp_path), require_ack=True)

    async def main():
        async with Client(server) as client:
            missing = await client.call_tool("execute_rivers_sql", {"query": SQL})
            payload = await client.call_tool("get_instructions", {})
            version = payload.structured_content["instructions_version"]
            ok = await client.call_tool("execute_rivers_sql", {"query": SQL, "instructions_version": version})
        return missing, payload, ok

    missing, payload, ok = anyio.run(main)
    assert missing.is_error and "get_instructions" in _text(missing)
    assert payload.structured_content["instructions_version"].startswith("rivers-instructions-")
    assert "Never extrapolate" in payload.structured_content["system_prompt"]
    assert not ok.is_error and "Waal" in _text(ok)


def test_capability_parity_with_native_agent(tmp_path):
    project = make_project(tmp_path)
    server = build_database_mcp_server(project, require_ack=False)
    mcp_names = [t.name for t in anyio.run(server.list_tools)]
    native = DatabaseAgent(project, model_provider="openai", llm=ScriptedChatModel(responses=[text("x")]))
    assert [n for n in mcp_names if n != "get_instructions"] == native.tool_names


def test_mcp_and_native_share_instructions(tmp_path):
    project = make_project(tmp_path)
    server = build_database_mcp_server(project, require_ack=False)

    async def main():
        async with Client(server) as client:
            return (await client.call_tool("get_instructions", {})).structured_content

    payload = anyio.run(main)
    native = DatabaseAgent(project, model_provider="openai", llm=ScriptedChatModel(responses=[text("x")]))
    assert payload["system_prompt"] == native.system_message


def test_custom_capabilities_through_mcp(tmp_path, no_simple_agent):
    server = build_database_mcp_server(add_custom_capabilities(make_project(tmp_path)), require_ack=False)

    async def main():
        async with Client(server) as client:
            tools = [t.name for t in (await client.list_tools()).tools]
            found = await client.call_tool("river_by_name", {"name": "Waal"})
            injected = await client.call_tool("river_by_name", {"name": "x'; DELETE FROM Rivers; --"})
            miles = await client.call_tool("km_to_miles", {"km": 16.09344})
        return tools, found, injected, miles

    tools, found, injected, miles = anyio.run(main)
    assert tools == [
        "get_instructions", "get_schema", "list_tables", "describe_table", "execute_rivers_sql", "river_by_name",
        "km_to_miles", "think"
    ]
    assert not found.is_error and "Waal" in _text(found)
    assert _text(found).endswith("Executed SQL:\nSELECT RiverId, Name, LengthKm FROM Rivers WHERE Name = 'Waal'")
    assert "ERROR: Only SELECT queries are permitted" in _text(injected)
    assert _text(miles) == "10.0 miles"


def test_custom_capability_parity_with_native_agent(tmp_path):
    project = add_custom_capabilities(make_project(tmp_path))
    mcp_names = [t.name for t in anyio.run(build_database_mcp_server(project, require_ack=False).list_tools)]
    native = DatabaseAgent(project, model_provider="openai", llm=ScriptedChatModel(responses=[text("x")]))
    assert [n for n in mcp_names if n != "get_instructions"] == native.tool_names
