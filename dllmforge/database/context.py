"""
Database instructions shared by the native agent (system message) and MCP clients (prompt/resource).

The text is protocol-free: it never describes how to call tools (JSON formats, ``final_answer``),
because SimpleAgent adds its own routing instructions and MCP clients use their own tool protocol.
"""
from typing import Optional

from .capabilities import DatabaseCapabilities
from .project import DatabaseProject

__all__ = ["CORE_DATABASE_RULES", "build_database_instructions"]

CORE_DATABASE_RULES = """\
1. Answer only from what the tools return: database query results and, when a document search tool is
   available, passages from the project documents (cite these as [file p.N]). Never extrapolate, guess or
   fill gaps from general knowledge; if a query returns no rows, say so clearly.
2. The data is read-only: write only SELECT (or WITH ... SELECT) queries.
3. Show the executed SQL for every result you report, and cite the table(s) and column(s) you used.
4. State your confidence: High (exact match), Medium (approximate or aggregated), Low (indirect inference).
5. Every result row must be self-identifying: select and order by each column that is not fixed by the
   WHERE clause, so rows that would otherwise look identical can be told apart. Report multi-row results
   as a markdown table with one row per database row.
6. Keep queries efficient: filter early, aggregate where possible and add LIMIT to exploratory queries.
   If a result is truncated, refine the query instead of drawing conclusions from partial data.
7. If you are unsure which table or column to use, inspect the schema (get_schema, list_tables,
   describe_table) before querying."""


def _section(title: str, body: str) -> str:
    return f"## {title}\n\n{body.strip()}"


def build_database_instructions(project: DatabaseProject,
                                capabilities: Optional[DatabaseCapabilities] = None,
                                include_schema: bool = True) -> str:
    """Build the instructions for ``project``.

    Args:
        project: The database project.
        capabilities: Reuse an existing instance to share its schema cache.
        include_schema: Include the full schema text (introspects the databases on first use).
    """
    capabilities = capabilities or DatabaseCapabilities(project)
    sources = "\n".join(f"- **{s.name}** ({s.backend.dialect}), query with `{s.tool_name}`"
                        f"{': ' + s.description if s.description else ''}" for s in project.data_sources)
    sections = [
        f"# {project.title}\n\n{project.description.strip()}",
        _section("Data sources", sources),
        _section("Rules", CORE_DATABASE_RULES),
    ]
    if project.instructions.strip():
        sections.append(_section("Project instructions", project.instructions))
    if include_schema:
        sections.append(_section("Schema", capabilities.schema_text()))
    if project.examples.strip():
        sections.append(_section("Examples", project.examples))
    return "\n\n".join(sections) + "\n"
