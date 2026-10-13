import inspect

import pytest

from dllmforge.database import (DatabaseCapabilities, DatabaseProject, DataSource, SQLAlchemyBackend,
                                build_capabilities, run_sql, split_executed_sql)
from db_fixtures import add_custom_capabilities, make_project, make_rivers_db


def _by_name(project):
    return {f.__name__: f for f in build_capabilities(project)}


def test_capability_names_and_docs(tmp_path):
    caps = _by_name(make_project(tmp_path))
    assert list(caps) == ["get_schema", "list_tables", "describe_table", "execute_rivers_sql"]
    for func in caps.values():
        assert func.__doc__ and inspect.signature(func).return_annotation is str
    assert "Dutch rivers test data" in caps["execute_rivers_sql"].__doc__


def test_execute_runs_multiple_selects_like_boi(tmp_path):
    execute = _by_name(make_project(tmp_path))["execute_rivers_sql"]
    result = execute(query="SELECT Name FROM Rivers WHERE Name = 'Lek'; SELECT Name FROM Rivers WHERE Name = 'Waal'")
    assert "Statement 1:" in result and "Statement 2:" in result
    assert "Lek" in result and "Waal" in result


def test_execute_rejects_write_in_batch_with_boi_message(tmp_path):
    execute = _by_name(make_project(tmp_path))["execute_rivers_sql"]
    result = execute(query="SELECT 1; DELETE FROM Rivers")
    assert result == ("Statement 2: ERROR: Only SELECT queries are permitted. Received 'DELETE'. "
                      "Rewrite as a SELECT statement.")


def test_execute_accepts_sql_alias_and_requires_query(tmp_path):
    execute = _by_name(make_project(tmp_path))["execute_rivers_sql"]
    assert "Lek" in execute(sql="SELECT Name FROM Rivers WHERE RiverId = 1")
    assert execute().startswith("ERROR:")


def test_sql_errors_are_returned_with_hint(tmp_path):
    project = make_project(tmp_path,
                           error_hint=lambda msg, sql: "Hint: check column names" if "column" in msg else None)
    result = _by_name(project)["execute_rivers_sql"](query="SELECT Nope FROM Rivers")
    assert result.startswith("SQL ERROR:")
    assert "Hint: check column names" in result
    assert result.endswith("Executed SQL:\nSELECT Nope FROM Rivers")


def test_schema_tools(tmp_path):
    caps = _by_name(make_project(tmp_path, notes="## Stations\nGauging stations.\n"))
    assert caps["list_tables"]() == "rivers: Rivers, Stations"
    assert "Gauging stations." in caps["describe_table"]("stations")
    assert caps["describe_table"]("missing").startswith("ERROR:")
    schema = caps["get_schema"]()
    assert "DATA SOURCE 'rivers'" in schema and "TABLE Stations" in schema


def test_table_allowlist_scopes_schema(tmp_path):
    caps = _by_name(make_project(tmp_path, tables=["Stations"]))
    assert caps["list_tables"]() == "rivers: Stations"
    assert "TABLE Rivers" not in caps["get_schema"]()


def test_multiple_sources_and_extra_capability(tmp_path):
    a = DataSource("hrd", SQLAlchemyBackend.from_sqlite_path(make_rivers_db(tmp_path / "a.sqlite")))
    b = DataSource("boi", SQLAlchemyBackend.from_sqlite_path(make_rivers_db(tmp_path / "b.sqlite")))

    def find_location(query: str) -> str:
        """Find a location by approximate name."""
        return f"found {query}"

    project = DatabaseProject(name="boi",
                              title="BOI",
                              description="d",
                              data_sources=[a, b],
                              extra_capabilities=[find_location])
    caps = _by_name(project)
    assert list(caps)[-3:] == ["execute_hrd_sql", "execute_boi_sql", "find_location"]
    assert caps["describe_table"]("Rivers").startswith("ERROR:")  # ambiguous without source
    assert "TABLE Rivers" in caps["describe_table"]("Rivers", source="boi")
    assert caps["list_tables"]().splitlines() == ["hrd: Rivers, Stations", "boi: Rivers, Stations"]


def test_duplicate_capability_names_rejected(tmp_path):
    project = make_project(tmp_path)

    def get_schema() -> str:
        """Clashes with the generic capability."""
        return ""

    project.extra_capabilities.append(get_schema)
    with pytest.raises(ValueError, match="Duplicate"):
        build_capabilities(project)


def test_split_executed_sql():
    single = split_executed_sql("Results (1 row):\nx\n-\n1\n\nExecuted SQL:\nSELECT 1")
    assert single == {"results": "Results (1 row):\nx\n-\n1", "sql": ["SELECT 1"]}
    multi = split_executed_sql(
        "Statement 1:\nr1\n\nExecuted SQL:\nSELECT 1\n\nStatement 2:\nr2\n\nExecuted SQL:\nSELECT 2")
    assert multi["sql"] == ["SELECT 1", "SELECT 2"]
    assert split_executed_sql("ERROR: x") == {"results": "ERROR: x", "sql": []}


# ── custom (project-specific) capabilities ─────────────────────────────────


def test_method_run_sql_delegates_to_module_function(tmp_path):
    project = make_project(tmp_path)
    source = project.source("rivers")
    caps = DatabaseCapabilities(project)
    sql = "SELECT Name FROM Rivers WHERE RiverId = 1"
    assert caps.run_sql(source, sql) == run_sql(source, sql)
    for unsafe in ["DELETE FROM Rivers", "", "SELECT 1; DROP TABLE Rivers"]:
        assert caps.run_sql(source, unsafe) == run_sql(source, unsafe)
        assert "ERROR:" in run_sql(source, unsafe) and "Executed SQL" not in run_sql(source, unsafe)


def test_run_sql_keeps_row_limit_and_error_handling(tmp_path):
    project = make_project(tmp_path, error_hint=lambda message, sql: "HINT: check the table name")
    source = project.source("rivers")
    source.backend.max_rows = 2
    limited = run_sql(source, "SELECT Name FROM Rivers ORDER BY RiverId")
    assert "Lek" in limited and "Waal" in limited and "IJssel" not in limited and "first 2 rows" in limited
    failed = run_sql(source, "SELECT * FROM NoSuchTable")
    assert failed.startswith("SQL ERROR:") and "HINT: check the table name" in failed
    assert failed.endswith("Executed SQL:\nSELECT * FROM NoSuchTable")


def test_custom_capabilities_are_registered_after_builtins(tmp_path):
    caps = _by_name(add_custom_capabilities(make_project(tmp_path)))
    assert list(caps) == [
        "get_schema", "list_tables", "describe_table", "execute_rivers_sql", "river_by_name", "km_to_miles"
    ]
    assert "Look up a river" in inspect.getdoc(caps["river_by_name"])


def test_custom_sql_capability_uses_safe_path(tmp_path):
    caps = _by_name(add_custom_capabilities(make_project(tmp_path)))
    found = split_executed_sql(caps["river_by_name"]("Waal"))
    assert "Waal" in found["results"] and found["sql"] == [
        "SELECT RiverId, Name, LengthKm FROM Rivers WHERE Name = 'Waal'"
    ]
    injected = caps["river_by_name"]("x'; DELETE FROM Rivers; --")
    assert "ERROR: Only SELECT queries are permitted" in injected and "Executed SQL" not in injected
    assert "Lek" in caps["execute_rivers_sql"]("SELECT Name FROM Rivers WHERE RiverId = 1")  # data untouched


def test_custom_non_sql_capability(tmp_path):
    caps = _by_name(add_custom_capabilities(make_project(tmp_path)))
    assert caps["km_to_miles"](16.09344) == "10.0 miles"


def _project_with(tmp_path, func):
    source = make_project(tmp_path).source("rivers")
    return DatabaseProject(name="p", title="t", description="d", data_sources=[source], extra_capabilities=[func])


def _async_tool():

    async def fetch() -> str:
        """Fetch something."""
        return ""

    return fetch


def _named_tool(name):

    def func() -> str:
        """Do something."""
        return ""

    func.__name__ = name
    return func


@pytest.mark.parametrize("func, message", [
    (lambda: "", "must match"),
    (_async_tool(), "synchronous"),
    (_named_tool("think"), "reserved"),
    (_named_tool("get_instructions"), "reserved"),
    (_named_tool("bad-name"), "must match"),
    (_named_tool("   "), "must match"),
    ("not callable", "named function"),
])
def test_invalid_custom_capabilities_rejected(tmp_path, func, message):
    with pytest.raises(ValueError, match=message):
        _project_with(tmp_path, func)


def test_blank_docstring_rejected(tmp_path):

    def blank() -> str:
        """   """
        return ""

    with pytest.raises(ValueError, match="docstring"):
        _project_with(tmp_path, blank)


def test_capabilities_added_after_construction_are_validated(tmp_path):
    project = make_project(tmp_path)
    project.extra_capabilities.append(_named_tool("think"))
    with pytest.raises(ValueError, match="reserved"):
        build_capabilities(project)


def test_custom_capability_cannot_shadow_builtin(tmp_path):
    with pytest.raises(ValueError, match="Duplicate"):
        build_capabilities(_project_with(tmp_path, _named_tool("execute_rivers_sql")))
