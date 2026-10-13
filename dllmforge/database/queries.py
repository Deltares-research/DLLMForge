"""
Saved SQL queries: reusable read-only ``.sql`` statements with named ``:parameter`` placeholders.

All saved queries of a project are exposed through **one** capability, ``run_saved_query(name, parameters)``,
on both execution paths. Safety reuses the existing mechanisms:

- each statement is checked with :func:`~dllmforge.database.safety.validate_read_only` when it is defined;
- placeholders must equal the declared parameters;
- values are converted to the declared type and bound by the database driver via
  :func:`~dllmforge.database.capabilities.run_sql` — they are never formatted into the SQL text;
- execution uses the data source's read-only backend (row limit, timeout) like ``execute_<source>_sql``.
"""
import re
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

from .capabilities import run_sql
from .project import DataSource
from .safety import UnsafeQueryError, validate_read_only

__all__ = ["QueryParameter", "SavedQuery", "check_saved_query", "placeholders", "saved_queries_capability"]

PARAMETER_TYPES = ("str", "int", "float", "bool")
# Same rule as sqlalchemy.text(): ':name', but not '::cast' or '\:escaped'.
_PLACEHOLDER = re.compile(r"(?<![:\w\\]):(\w+)(?!:)")


@dataclass(frozen=True)
class QueryParameter:
    name: str
    type: str = "str"
    description: str = ""


@dataclass(frozen=True)
class SavedQuery:
    """A reusable read-only statement run against the data source named ``source``."""
    name: str
    sql: str
    source: str
    description: str = ""
    parameters: Tuple[QueryParameter, ...] = field(default_factory=tuple)


def placeholders(sql: str) -> List[str]:
    """Names of the ``:name`` placeholders in ``sql``, in order of first use."""
    return list(dict.fromkeys(_PLACEHOLDER.findall(sql)))


def check_saved_query(sql: str, dialect: str, parameters: Sequence[str]) -> str:
    """Validate a saved statement; returns it without a trailing semicolon. Raises ``ValueError``."""
    try:
        (statement, ) = validate_read_only(sql, dialect, allow_multiple=False)
    except UnsafeQueryError as exc:
        raise ValueError(str(exc).removeprefix("ERROR: ")) from None
    used, declared = set(placeholders(statement)), set(parameters)
    if used != declared:
        problems = []
        if used - declared:
            problems.append(f"undeclared placeholders {sorted(used - declared)}")
        if declared - used:
            problems.append(f"declared parameters not used in the SQL {sorted(declared - used)}")
        raise ValueError("; ".join(problems))
    return statement


def _convert(value: Any, type_name: str) -> Any:
    if type_name == "bool":
        if isinstance(value, bool):
            return value
        if str(value).strip().lower() in ("true", "false"):
            return str(value).strip().lower() == "true"
        raise ValueError("expected true or false")
    if type_name == "int":
        if isinstance(value, bool) or (isinstance(value, float) and not value.is_integer()):
            raise ValueError("expected an integer")
        return int(value)
    if type_name == "float":
        if isinstance(value, bool):
            raise ValueError("expected a number")
        return float(value)
    if not isinstance(value, (str, int, float)) or isinstance(value, bool):
        raise ValueError("expected text")
    return str(value)


def _bind(query: SavedQuery, parameters: Dict[str, Any]) -> Dict[str, Any]:
    declared = {p.name: p for p in query.parameters}
    unknown = sorted(set(parameters) - set(declared))
    missing = sorted(set(declared) - set(parameters))
    if unknown or missing:
        detail = "; ".join(f"{label}: {names}" for label, names in (("unknown", unknown), ("missing", missing))
                           if names)
        raise ValueError(f"wrong parameters for '{query.name}' ({detail}); expected {sorted(declared)}")
    bound = {}
    for name, value in parameters.items():
        try:
            bound[name] = _convert(value, declared[name].type)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"parameter '{name}': {exc} (got {value!r})") from None
    return bound


def _describe(query: SavedQuery) -> str:
    params = ", ".join(f"{p.name} ({p.type}{': ' + p.description if p.description else ''})"
                       for p in query.parameters) or "none"
    return f"- {query.name}: {query.description or 'no description'} Parameters: {params}."


def saved_queries_capability(queries: Sequence[SavedQuery], sources: Sequence[DataSource]) -> Callable[..., str]:
    """Build the ``run_saved_query`` capability; add it to ``DatabaseProject.extra_capabilities``.

    Every query is validated here against the dialect of its data source (raises ``ValueError``).
    """
    by_source = {source.name: source for source in sources}
    registry: Dict[str, Tuple[SavedQuery, DataSource]] = {}
    for query in queries:
        if query.name in registry:
            raise ValueError(f"duplicate saved query name: {query.name}")
        if query.source not in by_source:
            raise ValueError(f"saved query '{query.name}': unknown data source '{query.source}'")
        for param in query.parameters:
            if param.type not in PARAMETER_TYPES:
                raise ValueError(f"saved query '{query.name}': parameter '{param.name}' has unsupported type "
                                 f"'{param.type}' (use one of {PARAMETER_TYPES})")
        source = by_source[query.source]
        try:
            statement = check_saved_query(query.sql, source.backend.dialect, [p.name for p in query.parameters])
        except ValueError as exc:
            raise ValueError(f"saved query '{query.name}': {exc}") from None
        registry[query.name] = (SavedQuery(query.name, statement, query.source, query.description,
                                           tuple(query.parameters)), source)

    def run_saved_query(name: str, parameters: Optional[Dict[str, Any]] = None) -> str:
        query_source = registry.get(name)
        if query_source is None:
            return f"ERROR: Unknown saved query '{name}'. Available: {sorted(registry)}."
        query, source = query_source
        try:
            bound = _bind(query, dict(parameters or {}))
        except ValueError as exc:
            return f"ERROR: {exc}"
        return run_sql(source, query.sql, bound)

    run_saved_query.__doc__ = (
        "Run a predefined, reviewed read-only SQL query by name. Prefer it over writing SQL when one of the "
        "queries below answers the question. Pass the values as a JSON object, e.g. "
        "parameters={\"name\": value}; they are bound safely, never inserted into the SQL. "
        "The result includes the executed SQL.\n\nAvailable queries:\n" +
        "\n".join(_describe(query) for query, _ in registry.values()))
    return run_saved_query
