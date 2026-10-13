"""SQLAlchemyBackend against a real PostgreSQL/PostGIS server.

Gated: set ``TEST_POSTGRES_DSN`` to a *disposable* database where the test user may create a schema, e.g.
    docker run --rm -d -p 55432:5432 -e POSTGRES_PASSWORD=test postgis/postgis:16-3.4
    TEST_POSTGRES_DSN=postgresql://postgres:test@localhost:55432/postgres pytest tests/database
"""
import os
import uuid

import pytest

DSN = os.getenv("TEST_POSTGRES_DSN")
pytestmark = pytest.mark.skipif(not DSN, reason="TEST_POSTGRES_DSN not set")
psycopg = pytest.importorskip("psycopg")
pytest.importorskip("sqlalchemy")

from dllmforge.database import DatabaseProject, DataSource, SQLAlchemyBackend, build_capabilities  # noqa: E402


@pytest.fixture(scope="module")
def schema():
    name = f"dllmforge_test_{uuid.uuid4().hex[:8]}"
    with psycopg.connect(DSN, autocommit=True) as conn:
        conn.execute("CREATE EXTENSION IF NOT EXISTS postgis")
        conn.execute(f"""
            CREATE SCHEMA {name};
            CREATE TABLE {name}.hazard_class (code text PRIMARY KEY, label text NOT NULL);
            COMMENT ON TABLE {name}.hazard_class IS 'Hazard classification lookup';
            CREATE TABLE {name}.segment (
                id integer PRIMARY KEY,
                hazard text REFERENCES {name}.hazard_class(code),
                geom geometry(Point, 4326));
            COMMENT ON COLUMN {name}.segment.hazard IS 'Erosion hazard code';
            INSERT INTO {name}.hazard_class VALUES ('H', 'High'), ('L', 'Low');
            INSERT INTO {name}.segment
                SELECT g, CASE WHEN g % 2 = 0 THEN 'H' ELSE 'L' END, ST_SetSRID(ST_MakePoint(4 + g * 0.01, 52), 4326)
                FROM generate_series(1, 120) AS g;
            CREATE SEQUENCE {name}.counter;
            ANALYZE {name}.segment;
        """)
    yield name
    with psycopg.connect(DSN, autocommit=True) as conn:
        conn.execute(f"DROP SCHEMA {name} CASCADE")


@pytest.fixture
def backend(schema):
    return SQLAlchemyBackend(DSN, schemas=[schema], statement_timeout_ms=2000)


def test_introspection(backend, schema):
    assert backend.list_tables() == [f"{schema}.hazard_class", f"{schema}.segment"]
    (segment, ) = backend.describe_tables(["SEGMENT"])
    assert segment.name == f"{schema}.segment"
    types = {c.name: c.data_type for c in segment.columns}
    assert types["geom"] == "geometry(Point,4326)"
    assert segment.columns[0].primary_key
    assert segment.row_count == 120 and segment.row_count_is_estimate
    assert any("hazard_class" in fk for fk in segment.foreign_keys)
    assert {c.name: c.comment for c in segment.columns}["hazard"] == "Erosion hazard code"
    exact = SQLAlchemyBackend(DSN, schemas=[schema], exact_row_counts=True).describe_tables(["hazard_class"])[0]
    assert exact.row_count == 2 and not exact.row_count_is_estimate
    assert exact.comment == "Hazard classification lookup"


def test_geometry_rendering_and_truncation(backend, schema):
    result = backend.execute(f"SELECT id, geom, ST_AsText(geom) AS wkt FROM {schema}.segment ORDER BY id")
    assert result.truncated and len(result.rows) == 50
    assert result.rows[0][1].startswith("<geometry Point SRID=4326")
    assert result.rows[0][2] == "POINT(4.01 52)"


def test_session_is_read_only(backend, schema):
    # Called on the backend directly, bypassing the SQL validator: the server must still refuse writes.
    with pytest.raises(psycopg.errors.ReadOnlySqlTransaction):
        backend.execute(f"SELECT nextval('{schema}.counter')")
    with pytest.raises(psycopg.Error):
        backend.execute(f"WITH d AS (DELETE FROM {schema}.segment RETURNING 1) SELECT count(*) FROM d")
    assert backend.execute(f"SELECT count(*) FROM {schema}.segment").rows == [("120", )]


def test_statement_timeout(backend):
    with pytest.raises(psycopg.errors.QueryCanceled):
        backend.execute("SELECT count(*) FROM generate_series(1, 1000000000)")


def test_capabilities_on_postgres(backend, schema):
    project = DatabaseProject(name="pg",
                              title="PG",
                              description="d",
                              data_sources=[DataSource("chw", backend, tables=["segment", "hazard_class"])])
    caps = {f.__name__: f for f in build_capabilities(project)}
    out = caps["execute_chw_sql"](
        query=f"SELECT h.label, count(*) AS n FROM {schema}.segment s "
        f"JOIN {schema}.hazard_class h ON h.code = s.hazard GROUP BY h.label ORDER BY h.label")
    assert "High" in out and "60" in out and "Executed SQL:" in out
    assert "exactly one SELECT" in caps["execute_chw_sql"](query="SELECT 1; SELECT 2")
    assert "Only SELECT" in caps["execute_chw_sql"](query=f"DELETE FROM {schema}.segment")
    assert caps["execute_chw_sql"](query="SELECT nope FROM nowhere").startswith("SQL ERROR:")
    schema_text = caps["get_schema"]()
    assert schema_text.index(f"TABLE {schema}.segment") < schema_text.index(f"TABLE {schema}.hazard_class")
    assert "SQLAlchemyBackend(" in repr(backend) and f"schemas={[schema]!r}" in repr(backend)
    if "@" in DSN and ":" in DSN.split("@")[0].split("//")[-1]:
        assert DSN.split("@")[0].split(":")[-1] not in repr(backend)
