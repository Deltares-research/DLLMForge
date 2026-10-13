import sqlite3

import pytest

pytest.importorskip("sqlalchemy")

from dllmforge.database.backends import QueryResult, SQLAlchemyBackend, render_value  # noqa: E402
from db_fixtures import make_rivers_db  # noqa: E402


@pytest.fixture
def backend(tmp_path):
    return SQLAlchemyBackend.from_sqlite_path(make_rivers_db(tmp_path / "r.sqlite", rows=60))


def test_execute_formats_table_and_sql(backend):
    result = backend.execute("SELECT Name FROM Rivers WHERE RiverId <= 2 ORDER BY RiverId")
    text = result.to_text()
    assert "Lek" in text and "Waal" in text
    assert text.endswith("Executed SQL:\nSELECT Name FROM Rivers WHERE RiverId <= 2 ORDER BY RiverId")
    assert "(2 rows)" in text


def test_truncates_at_max_rows(backend):
    result = backend.execute("SELECT * FROM Rivers")
    assert result.truncated and len(result.rows) == 50
    assert "more rows exist" in result.to_text()


def test_zero_rows():
    text = QueryResult(sql="SELECT 1 WHERE 0", columns=["x"], rows=[], truncated=False, max_rows=50).to_text()
    assert text.startswith("Query returned 0 rows.")


def test_connection_is_read_only(backend):
    with pytest.raises(sqlite3.OperationalError):
        backend.execute("DELETE FROM Rivers")


def test_missing_file_is_not_created(tmp_path):
    missing = tmp_path / "missing.sqlite"
    with pytest.raises(Exception):
        SQLAlchemyBackend.from_sqlite_path(missing).execute("SELECT 1")
    assert not missing.exists()


def test_introspection(backend):
    assert backend.list_tables() == ["Rivers", "Stations"]
    (stations, ) = backend.describe_tables(["stations"])
    assert stations.name == "Stations"
    assert stations.row_count == 1 and not stations.row_count_is_estimate
    assert [c.name for c in stations.columns] == ["StationId", "RiverId", "Name", "Data"]
    assert stations.columns[0].primary_key
    assert any("Rivers" in fk for fk in stations.foreign_keys)


def test_describe_keeps_requested_order(backend):
    assert [t.name for t in backend.describe_tables(["Stations", "Rivers"])] == ["Stations", "Rivers"]


def test_render_values(backend):
    assert render_value(None) == "NULL"
    assert render_value(b"\x00\x01") == "<binary 2 bytes>"
    assert render_value("x" * 1000, max_chars=10).startswith("x" * 10)
    assert "<binary 4 bytes>" in backend.execute("SELECT Data FROM Stations").to_text()


def test_repr_has_no_secrets(tmp_path):
    assert "SQLAlchemyBackend" in repr(SQLAlchemyBackend.from_sqlite_path(tmp_path / "x.sqlite"))
    assert "secret" not in repr(SQLAlchemyBackend("postgresql://reader:secret@db.invalid/chw"))


def test_dialect_defaults():
    sqlite = SQLAlchemyBackend("sqlite:///x.sqlite")
    assert sqlite.dialect == "sqlite" and sqlite.allow_multiple_statements
    postgres = SQLAlchemyBackend("postgresql://u@h/db", schemas=["chw"])
    assert postgres.dialect == "postgresql" and not postgres.allow_multiple_statements
    assert postgres.url.drivername == "postgresql+psycopg"


def test_rejects_engines_without_read_only_support():
    with pytest.raises(ValueError):
        SQLAlchemyBackend("mysql://u@h/db")
    with pytest.raises(ValueError):
        SQLAlchemyBackend("postgresql+asyncpg://u@h/db")


def test_percent_sign_is_not_a_placeholder(backend):
    assert "Lek" in backend.execute("SELECT Name FROM Rivers WHERE Name LIKE 'Le%'").to_text()
