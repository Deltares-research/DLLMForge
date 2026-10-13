"""
Database capabilities: plain, typed Python functions shared by the native agent and the MCP server.

This module deliberately imports neither LangChain/LangGraph nor MCP. Adapters wrap the functions
returned by :func:`build_capabilities` (``agent_tools.to_langchain_tools`` for SimpleAgent,
``mcp_utils.register_functions`` for MCP), so both execution paths run identical database code.
"""
import re
import threading
from dataclasses import replace
from typing import Any, Callable, Dict, List, Mapping, Optional

from .project import DatabaseProject, DataSource, validate_capability
from .safety import UnsafeQueryError, validate_read_only
from .schema import load_domain_notes, render_table, render_tables

__all__ = ["DatabaseCapabilities", "build_capabilities", "run_sql", "split_executed_sql", "EXECUTED_SQL_MARKER"]

EXECUTED_SQL_MARKER = "Executed SQL:"
_EXECUTED_SQL = re.compile(r"Executed SQL:\n(.*?)(?=\n\nStatement \d+:\n|\Z)", re.DOTALL)


def split_executed_sql(result: str) -> Dict[str, object]:
    """Split a SQL capability result into ``{"results": str, "sql": list[str]}``."""
    statements = [s.strip() for s in _EXECUTED_SQL.findall(result)]
    results = _EXECUTED_SQL.sub("", result).strip() if statements else result.strip()
    return {"results": results, "sql": statements}


def _named(func: Callable[..., str], name: str, doc: str) -> Callable[..., str]:
    func.__name__ = name
    func.__qualname__ = name
    func.__doc__ = doc
    return func


def _with_params(statement: str, params: Optional[Mapping[str, Any]]) -> str:
    """Statement as shown to the user: bound values are listed in a trailing SQL comment."""
    if not params:
        return statement
    values = ", ".join(f"{name}={value!r}" for name, value in params.items())
    return f"{statement}\n-- parameters: {values}"


def run_sql(source: DataSource, sql: str, params: Optional[Mapping[str, Any]] = None) -> str:
    """Validate and execute read-only SQL against ``source``; always returns LLM-readable text.

    This is the safe execution path for every SQL capability, including project extras: the statement is
    checked by :func:`validate_read_only`, executed on the source's read-only backend (row limit, timeout),
    and database errors are returned as text with the source's ``error_hint`` and the executed SQL.

    ``params`` are bound by the database driver to ``:name`` placeholders (never formatted into the SQL);
    they require a single statement.
    """
    backend = source.backend
    try:
        statements = validate_read_only(sql, backend.dialect, backend.allow_multiple_statements and params is None)
    except UnsafeQueryError as exc:
        return str(exc)
    sections = []
    for number, statement in enumerate(statements, 1):
        shown = _with_params(statement, params)
        try:
            if params is None:
                text = backend.execute(statement).to_text()
            else:
                text = replace(backend.execute(statement, params), sql=shown).to_text()
        except Exception as exc:  # database errors are reported to the LLM so it can correct the SQL
            message = str(exc).strip() or type(exc).__name__
            hint = source.error_hint(message, statement) if source.error_hint else None
            text = f"SQL ERROR: {message}" + (f"\n{hint}" if hint else "")
            text += f"\n\n{EXECUTED_SQL_MARKER}\n{shown}"
        sections.append(f"Statement {number}:\n{text}" if len(statements) > 1 else text)
    return "\n\n".join(sections)


class DatabaseCapabilities:
    """Builds and backs the capability functions of one :class:`DatabaseProject`.

    The schema text is introspected lazily on first use and cached per instance.
    """

    def __init__(self, project: DatabaseProject):
        self.project = project
        self._schema_cache: Dict[str, str] = {}
        self._notes_cache: Dict[str, Dict[str, str]] = {}
        self._lock = threading.Lock()

    # ── shared implementations ─────────────────────────────────────────────

    def notes(self, source: DataSource) -> Dict[str, str]:
        if source.name not in self._notes_cache:
            self._notes_cache[source.name] = load_domain_notes(source.domain_notes_path)
        return self._notes_cache[source.name]

    def source_schema(self, source: DataSource) -> str:
        with self._lock:
            if source.name not in self._schema_cache:
                tables = source.backend.describe_tables(source.tables)
                self._schema_cache[source.name] = render_tables(tables, self.notes(source))
            return self._schema_cache[source.name]

    def schema_text(self) -> str:
        """Schema of all data sources, with domain notes."""
        parts = []
        for source in self.project.data_sources:
            header = f"=== DATA SOURCE '{source.name}' ({source.backend.dialect}) — query with {source.tool_name} ==="
            description = f"{source.description}\n\n" if source.description else ""
            parts.append(f"{header}\n{description}{self.source_schema(source)}")
        return "\n\n".join(parts)

    def _resolve_source(self, source: Optional[str]) -> DataSource:
        if source:
            return self.project.source(source)
        if len(self.project.data_sources) > 1:
            raise KeyError("Several data sources exist; pass source= one of "
                           f"{[s.name for s in self.project.data_sources]}")
        return self.project.data_sources[0]

    def run_sql(self, source: DataSource, sql: str) -> str:
        """Validate and execute read-only SQL; see :func:`run_sql`."""
        return run_sql(source, sql)

    # ── capability functions ───────────────────────────────────────────────

    def functions(self) -> List[Callable[..., str]]:
        """All capability functions: generic database capabilities + project extras."""
        functions = [self._get_schema(), self._list_tables(), self._describe_table()]
        functions += [self._execute_sql(source) for source in self.project.data_sources]
        extras = list(self.project.extra_capabilities)
        for func in extras:
            validate_capability(func)
        functions += extras
        names = [f.__name__ for f in functions]
        duplicates = sorted({n for n in names if names.count(n) > 1})
        if duplicates:
            raise ValueError(f"Duplicate capability names: {duplicates}")
        return functions

    def _get_schema(self) -> Callable[..., str]:

        def get_schema() -> str:
            return self.schema_text()

        return _named(
            get_schema, "get_schema", f"Return the schema of the {self.project.title} database(s): tables, columns, "
            "types, keys, row counts and domain notes. Call this when unsure which tables or columns to use.")

    def _list_tables(self) -> Callable[..., str]:

        def list_tables(source: Optional[str] = None) -> str:
            try:
                sources = [self._resolve_source(source)
                           ] if source or len(self.project.data_sources) == 1 else self.project.data_sources
            except KeyError as exc:
                return f"ERROR: {exc.args[0]}"
            lines = []
            for item in sources:
                tables = item.backend.list_tables()
                if item.tables is not None:
                    allowed = {t.lower() for t in item.tables}
                    tables = [t for t in tables if t.lower() in allowed or t.split(".")[-1].lower() in allowed]
                lines.append(f"{item.name}: " + ", ".join(tables))
            return "\n".join(lines)

        sources = ", ".join(s.name for s in self.project.data_sources)
        return _named(
            list_tables, "list_tables", "List the table names that can be queried.\n\n"
            f"Args:\n    source: Optional data source name ({sources}); all sources when omitted.")

    def _describe_table(self) -> Callable[..., str]:

        def describe_table(table: str, source: Optional[str] = None) -> str:
            try:
                item = self._resolve_source(source)
            except KeyError as exc:
                return f"ERROR: {exc.args[0]}"
            found = item.backend.describe_tables([table])
            if not found:
                return f"ERROR: Table {table!r} not found in data source {item.name!r}. Use list_tables."
            return render_table(found[0], self.notes(item))

        sources = ", ".join(s.name for s in self.project.data_sources)
        return _named(
            describe_table, "describe_table",
            "Describe one table: columns, types, keys, row count and domain notes.\n\n"
            "Args:\n    table: Table name as returned by list_tables.\n"
            f"    source: Data source name ({sources}); required only when there are several.")

    def _execute_sql(self, source: DataSource) -> Callable[..., str]:

        def execute_sql(query: Optional[str] = None, sql: Optional[str] = None) -> str:
            text = query if query is not None else sql
            if text is None or not str(text).strip():
                return "ERROR: A non-empty 'query' argument with a SELECT statement is required."
            return self.run_sql(source, str(text))

        backend = source.backend
        multiple = ("Several SELECT statements may be separated by semicolons."
                    if backend.allow_multiple_statements else "Submit exactly one SELECT statement per call.")
        doc = (f"Execute a read-only SQL query ({backend.dialect}) against the '{source.name}' data source"
               f"{': ' + source.description if source.description else '.'}\n\n"
               f"Only SELECT (or WITH ... SELECT) is allowed. {multiple} At most {backend.max_rows} rows are "
               "returned; the result always ends with the executed SQL.")
        if source.query_guidance:
            doc += f"\n\n{source.query_guidance}"
        doc += "\n\nArgs:\n    query: The SQL SELECT statement.\n    sql: Alias of query for compatibility."
        return _named(execute_sql, source.tool_name, doc)


def build_capabilities(project: DatabaseProject) -> List[Callable[..., str]]:
    """Return the capability functions for ``project`` (generic database capabilities + project extras)."""
    return DatabaseCapabilities(project).functions()
