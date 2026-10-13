"""
MCP integration: exposes a :class:`~dllmforge.database.project.DatabaseProject`'s capabilities to external
MCP clients (e.g. GitHub Copilot). Requires the ``mcp`` extra.

The server is LLM-free. Tool calls go straight to the same capability functions the native
``DatabaseAgent`` uses; no SimpleAgent or LangGraph workflow is started (this module must not import
``dllmforge.database.agent`` or ``dllmforge.agent_core``).
"""
from pathlib import Path
from typing import Optional

from mcp.server import MCPServer
from mcp.server.mcpserver.resources.types import FunctionResource

from ..agent_tools import think
from ..mcp_utils import (InstructionAck, InstructionsPayload, create_server, env_flag, register_functions, run,
                         with_instruction_ack)
from .capabilities import DatabaseCapabilities
from .context import build_database_instructions
from .project import DatabaseProject

__all__ = ["build_database_mcp_server", "run_database_mcp_server"]


def _add_resource(server: MCPServer, uri: str, name: str, read, mime_type: str, description: str) -> None:
    server.add_resource(
        FunctionResource.from_function(read, uri=uri, name=name, mime_type=mime_type, description=description))


def build_database_mcp_server(project: DatabaseProject, require_ack: Optional[bool] = None) -> MCPServer:
    """Build the MCP server for ``project``.

    Tools: ``get_instructions``, the database capabilities (``get_schema``, ``list_tables``,
    ``describe_table``, ``execute_<source>_sql``, project extras) and ``think``.
    Resources: ``<name>://instructions``, ``<name>://schema``, ``<name>://docs/<source>`` and project resources.
    Prompt: ``<name>_assistant``.

    Args:
        project: The database project.
        require_ack: Require ``instructions_version`` on SQL tools; defaults to ``<PREFIX>_REQUIRE_INSTRUCTIONS_ACK``.
    """
    prefix = project.env_prefix
    if require_ack is None:
        require_ack = env_flag(f"{prefix}_REQUIRE_INSTRUCTIONS_ACK")

    capabilities = DatabaseCapabilities(project)

    def instructions() -> str:
        return build_database_instructions(project, capabilities)

    ack = InstructionAck(instructions, f"{project.name}-instructions", required=require_ack)
    server = create_server(project.title,
                           prefix,
                           instructions=(f"{project.description.strip()}\n\nCall get_instructions (or read "
                                         f"{project.name}://instructions) before answering questions."))

    def get_instructions() -> InstructionsPayload:
        """Return the assistant instructions (rules, schema, examples) and the current instructions_version.

        Load these into your agent instructions before using the query tools.
        """
        return ack.payload()

    sql_tools = {source.tool_name for source in project.data_sources}
    functions = [
        with_instruction_ack(func, ack) if func.__name__ in sql_tools else func for func in capabilities.functions()
    ]
    register_functions(server, [get_instructions, *functions, think])

    _add_resource(server, f"{project.name}://instructions", f"{project.title} instructions", instructions,
                  "text/markdown", "Assistant instructions: rules, schema and examples.")
    _add_resource(server, f"{project.name}://schema", f"{project.title} schema", capabilities.schema_text, "text/plain",
                  "Live database schema with domain notes.")
    for source in project.data_sources:
        if source.domain_notes_path and Path(source.domain_notes_path).is_file():
            path = Path(source.domain_notes_path)
            _add_resource(server,
                          f"{project.name}://docs/{source.name}",
                          f"{source.name} domain notes",
                          lambda path=path: path.read_text(encoding="utf-8"),
                          "text/markdown",
                          f"Domain documentation for data source '{source.name}'.")
    for resource in project.resources:
        _add_resource(server, f"{project.name}://{resource.path}", resource.name, resource.read, resource.mime_type,
                      resource.description)

    @server.prompt(name=f"{project.name.replace('-', '_')}_assistant",
                   description=f"Load the {project.title} assistant instructions.")
    def assistant_prompt(user_question: Optional[str] = None) -> str:
        text = instructions()
        return f"{text}\n\nUser question:\n{user_question}" if user_question else text

    return server


def run_database_mcp_server(project: DatabaseProject) -> None:
    """Build and run the server with the transport configured by ``<PREFIX>_MCP_*`` env vars."""
    run(build_database_mcp_server(project), project.env_prefix)
