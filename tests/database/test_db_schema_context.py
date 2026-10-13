import pytest

from dllmforge.database import DatabaseProject, DataSource, SQLAlchemyBackend, build_database_instructions
from dllmforge.database.backends import ColumnInfo, TableInfo
from dllmforge.database.schema import load_domain_notes, render_table
from db_fixtures import make_project


def test_domain_notes_case_insensitive_and_merged(tmp_path):
    path = tmp_path / "notes.md"
    path.write_text(
        "# Title\nintro ignored\n## HRDLocations\nfirst\n<!-- hidden -->\n"
        "## Table: `locaties`\nboi\n## hrdlocations\nsecond\n",
        encoding="utf-8")
    notes = load_domain_notes(path)
    assert set(notes) == {"hrdlocations", "locaties"}
    assert "first" in notes["hrdlocations"] and "second" in notes["hrdlocations"]
    assert "hidden" not in notes["hrdlocations"]


def test_missing_notes_file_is_empty(tmp_path):
    assert load_domain_notes(tmp_path / "nope.md") == {}
    assert load_domain_notes(None) == {}


def test_multiline_comment_blocks_are_skipped(tmp_path):
    path = tmp_path / "notes.md"
    path.write_text("## real\nkept\n<!--\n## example\nhidden\n-->\nalso kept\n", encoding="utf-8")
    assert load_domain_notes(path) == {"real": "kept\nalso kept"}


def test_render_table_matches_qualified_name_and_estimate():
    table = TableInfo(name="coastal.segments",
                      columns=[
                          ColumnInfo("id", "integer", nullable=False, primary_key=True),
                          ColumnInfo("geom", "geometry(LineString,4326)")
                      ],
                      row_count=1200,
                      row_count_is_estimate=True)
    text = render_table(table, {"segments": "One row per coastal segment."})
    assert "TABLE coastal.segments  (~1,200 rows)" in text
    assert "id integer [PK, NOT NULL]" in text
    assert "geometry(LineString,4326)" in text
    assert "One row per coastal segment." in text


def test_project_validation(tmp_path):
    backend = SQLAlchemyBackend.from_sqlite_path(tmp_path / "x.sqlite")
    with pytest.raises(ValueError):
        DataSource(name="Bad Name", backend=backend)
    source = DataSource(name="a", backend=backend)
    with pytest.raises(ValueError):
        DatabaseProject(name="UPPER", title="t", description="d", data_sources=[source])
    with pytest.raises(ValueError):
        DatabaseProject(name="p", title="t", description="d", data_sources=[])
    with pytest.raises(ValueError):
        DatabaseProject(name="p", title="t", description="d", data_sources=[source, DataSource("a", backend)])

    def no_doc():
        return ""

    with pytest.raises(ValueError):
        DatabaseProject(name="p", title="t", description="d", data_sources=[source], extra_capabilities=[no_doc])
    assert DatabaseProject(name="my-db", title="t", description="d", data_sources=[source]).env_prefix == "MY_DB"


def test_instructions_are_protocol_free_and_complete(tmp_path):
    project = make_project(tmp_path, notes="## rivers\nLengths are in kilometres.\n")
    project.instructions = "Use RiverId to join."
    project.examples = "Q: longest river\nSELECT Name FROM Rivers ORDER BY LengthKm DESC LIMIT 1"
    text = build_database_instructions(project)
    for expected in [
            "# Rivers", "execute_rivers_sql", "Never extrapolate", "TABLE Rivers", "kilometres", "Use RiverId to join.",
            "ORDER BY LengthKm"
    ]:
        assert expected in text
    for forbidden in ["final_answer", '{"tool"', "ONE ACTION PER RESPONSE"]:
        assert forbidden not in text


def test_instructions_without_schema_do_not_touch_database(tmp_path):
    project = make_project(tmp_path)
    project.data_sources[0].backend = SQLAlchemyBackend.from_sqlite_path(tmp_path / "missing.sqlite")
    assert "## Schema" not in build_database_instructions(project, include_schema=False)
