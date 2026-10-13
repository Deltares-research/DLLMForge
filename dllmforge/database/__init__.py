"""
Database support for DLLMForge: safe read-only access, schema context and shared capabilities.

Importing this package loads no SQLAlchemy or database driver (both imported lazily), no MCP SDK and no
agent framework beyond what ``dllmforge`` itself imports. The integrations live in submodules:

- ``dllmforge.database.agent``: ``DatabaseAgent`` configures ``SimpleAgent`` (native path)
- ``dllmforge.database.mcp_server``: ``build_database_mcp_server`` (MCP path; needs the ``mcp`` extra)
"""
from .backends import ColumnInfo, QueryBackend, QueryResult, SQLAlchemyBackend, TableInfo
from .capabilities import DatabaseCapabilities, build_capabilities, run_sql, split_executed_sql
from .context import build_database_instructions
from .project import DatabaseProject, DataSource, ProjectResource
from .safety import UnsafeQueryError, validate_read_only

__all__ = [
    "ColumnInfo", "QueryBackend", "QueryResult", "SQLAlchemyBackend", "TableInfo", "DatabaseCapabilities",
    "build_capabilities", "run_sql", "split_executed_sql", "build_database_instructions", "DatabaseProject",
    "DataSource", "ProjectResource", "UnsafeQueryError", "validate_read_only"
]
