"""Layering rules between the generic modules and the database package.

Note: ``import dllmforge.<anything>`` runs ``dllmforge/__init__.py``, which imports ``agent_core`` (LangGraph).
Runtime checks therefore assert on ``dllmforge.database.agent`` and optional drivers, while the static
checks below enforce which modules each file imports directly.
"""
import ast
import subprocess
import sys
from pathlib import Path

import pytest

PACKAGE = Path(__file__).resolve().parents[1] / "dllmforge"
GENERIC = ["agent_core.py", "llm_factory.py", "agent_messages.py", "agent_tools.py", "mcp_utils.py"]
FRAMEWORK_NEUTRAL = ["backends.py", "safety.py", "schema.py", "project.py", "capabilities.py", "context.py"]


def _imports(path: Path) -> set:
    """Absolute module names imported by ``path`` (relative imports resolved inside dllmforge)."""
    package = ["dllmforge"] + list(path.relative_to(PACKAGE).parent.parts)
    found = set()
    for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
        if isinstance(node, ast.Import):
            found.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            if node.level:
                base = package[:len(package) - node.level + 1]
                module = ".".join(base + ([node.module] if node.module else []))
            else:
                module = node.module or ""
            found.add(module)
            found.update(f"{module}.{alias.name}" for alias in node.names)
    return found


def _hits(imports: set, forbidden: tuple) -> set:
    return {m for m in imports if any(m == f or m.startswith(f + ".") for f in forbidden)}


@pytest.mark.parametrize("name", GENERIC)
def test_generic_modules_do_not_import_database(name):
    assert not _hits(_imports(PACKAGE / name), ("dllmforge.database", ))


@pytest.mark.parametrize("name", FRAMEWORK_NEUTRAL)
def test_database_core_is_framework_neutral(name):
    # sqlalchemy/psycopg are allowed as lazy imports inside SQLAlchemyBackend; the runtime test checks they are
    # not loaded eagerly.
    forbidden = ("dllmforge.agent_core", "langgraph", "langchain_core", "langchain", "mcp", "dllmforge.mcp_utils",
                 "dllmforge.database.agent", "dllmforge.database.mcp_server")
    assert not _hits(_imports(PACKAGE / "database" / name), forbidden)


def test_mcp_server_cannot_reach_native_agent():
    forbidden = ("dllmforge.agent_core", "langgraph", "dllmforge.database.agent", "dllmforge.llm_factory")
    assert not _hits(_imports(PACKAGE / "database" / "mcp_server.py"), forbidden)


def _loaded_after(statement: str) -> set:
    code = f"import sys; {statement}; print('\\n'.join(sys.modules))"
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True,
                         cwd=PACKAGE.parent).stdout
    return set(out.split())


def test_import_database_loads_no_optional_drivers():
    loaded = _loaded_after("import dllmforge.database")
    assert not {"psycopg", "mcp", "dllmforge.database.agent", "dllmforge.database.mcp_server"} & loaded
    # sqlalchemy may already be loaded by dllmforge/__init__ (LangChain imports it); the backend must not add it.
    if "sqlalchemy" not in _loaded_after("import dllmforge"):
        assert "sqlalchemy" not in loaded


def test_import_mcp_server_does_not_load_native_agent():
    pytest.importorskip("mcp")
    loaded = _loaded_after("import dllmforge.database.mcp_server")
    assert "dllmforge.database.agent" not in loaded
    assert "dllmforge.llm_factory" not in loaded
