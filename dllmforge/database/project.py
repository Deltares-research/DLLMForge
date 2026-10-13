"""
The contract between a database project (BOI, CHW, ...) and the reusable database layer.

A project is plain Python: it builds its backends from its own configuration and passes them in.
Nothing here reads environment variables or knows about a specific database.
"""
import inspect
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, List, Optional, Sequence

from .backends import QueryBackend

__all__ = ["DataSource", "ProjectResource", "DatabaseProject", "RESERVED_CAPABILITY_NAMES", "validate_capability"]

_SOURCE_NAME = re.compile(r"^[a-z][a-z0-9_]*$")
_PROJECT_NAME = re.compile(r"^[a-z][a-z0-9-]*$")
_CAPABILITY_NAME = re.compile(r"^[A-Za-z_][A-Za-z0-9_]{0,63}$")

# Tool names added by the adapters (native agent: ``think``; MCP server: ``think`` and ``get_instructions``).
RESERVED_CAPABILITY_NAMES = frozenset({"think", "get_instructions"})


def validate_capability(func: Callable[..., str]) -> None:
    """Raise ``ValueError`` unless ``func`` can be exposed as a tool on both the native and the MCP path.

    A capability is a synchronous named function (no lambda) with a non-empty docstring, whose name is a valid
    tool name and not reserved by an adapter.
    """
    name = getattr(func, "__name__", None)
    if not callable(func) or not isinstance(name, str):
        raise ValueError(f"Capability {func!r} must be a named function")
    if not _CAPABILITY_NAME.match(name):
        raise ValueError(f"Capability name {name!r} must match {_CAPABILITY_NAME.pattern}")
    if name in RESERVED_CAPABILITY_NAMES:
        raise ValueError(f"Capability name {name!r} is reserved; reserved names: {sorted(RESERVED_CAPABILITY_NAMES)}")
    if not (getattr(func, "__doc__", None) or "").strip():
        raise ValueError(f"Capability {name!r} needs a docstring; it becomes the tool description")
    if inspect.iscoroutinefunction(func):
        raise ValueError(f"Capability {name!r} must be synchronous")


@dataclass
class DataSource:
    """One queryable database, exposed as the capability ``execute_<name>_sql``.

    Args:
        name: Lowercase identifier, e.g. ``"chw"``.
        backend: A :class:`~dllmforge.database.backends.QueryBackend`.
        description: What the data source contains (shown in instructions and tool description).
        tables: Optional table allowlist and display order for the schema context. This scopes what the
            LLM is told about; access control itself belongs to the database role.
        domain_notes_path: Optional markdown file with one ``## <table>`` section per table.
        query_guidance: Extra guidance appended to the SQL tool description.
        error_hint: Optional ``(error_message, sql) -> hint`` callback to add actionable advice to SQL errors.
    """

    name: str
    backend: QueryBackend
    description: str = ""
    tables: Optional[Sequence[str]] = None
    domain_notes_path: Optional[str | Path] = None
    query_guidance: str = ""
    error_hint: Optional[Callable[[str, str], Optional[str]]] = None

    def __post_init__(self) -> None:
        if not _SOURCE_NAME.match(self.name):
            raise ValueError(f"DataSource name {self.name!r} must match {_SOURCE_NAME.pattern}")

    @property
    def tool_name(self) -> str:
        return f"execute_{self.name}_sql"


@dataclass
class ProjectResource:
    """Extra read-only content for MCP clients, served at ``<project.name>://<path>``."""

    path: str
    name: str
    read: Callable[[], str]
    mime_type: str = "text/plain"
    description: str = ""


@dataclass
class DatabaseProject:
    """Everything that makes one database application specific.

    Args:
        name: Lowercase slug (``[a-z][a-z0-9-]*``), used for the MCP server and resource URIs.
        title: Human-readable name.
        description: Who the assistant is and what the data is about (first part of the instructions).
        data_sources: One or more queryable databases.
        instructions: Project-specific rules and query workflow (markdown).
        examples: Example questions and SQL.
        extra_capabilities: Project-specific plain functions with type hints and a docstring, returning ``str``.
            They are exposed to both the native agent and the MCP server (see :func:`validate_capability`).
            They may use any library or service; to query a data source, call
            :func:`dllmforge.database.capabilities.run_sql` so the read-only checks and row limit apply.
        resources: Extra MCP resources.
        env_prefix: Prefix for runtime settings such as ``<PREFIX>_MCP_TRANSPORT``; defaults to the upper-case name.
    """

    name: str
    title: str
    description: str
    data_sources: List[DataSource]
    instructions: str = ""
    examples: str = ""
    extra_capabilities: List[Callable[..., str]] = field(default_factory=list)
    resources: List[ProjectResource] = field(default_factory=list)
    env_prefix: str = ""

    def __post_init__(self) -> None:
        if not _PROJECT_NAME.match(self.name):
            raise ValueError(f"DatabaseProject name {self.name!r} must match {_PROJECT_NAME.pattern}")
        if not self.data_sources:
            raise ValueError("A DatabaseProject needs at least one DataSource")
        names = [source.name for source in self.data_sources]
        if len(set(names)) != len(names):
            raise ValueError(f"Duplicate DataSource names: {names}")
        for func in self.extra_capabilities:
            validate_capability(func)
        if not self.env_prefix:
            self.env_prefix = self.name.upper().replace("-", "_")

    def source(self, name: str) -> DataSource:
        for source in self.data_sources:
            if source.name == name:
                return source
        raise KeyError(f"Unknown data source {name!r}; available: {[s.name for s in self.data_sources]}")
