"""Small SQLite fixtures shared by the database tests (no LLM, no network)."""
import sqlite3
from pathlib import Path

from dllmforge.database import DatabaseProject, DataSource, SQLAlchemyBackend, run_sql


def make_rivers_db(path: Path, rows: int = 3) -> Path:
    conn = sqlite3.connect(path)
    conn.executescript("""
        CREATE TABLE Rivers (RiverId INTEGER PRIMARY KEY, Name TEXT NOT NULL, LengthKm REAL);
        CREATE TABLE Stations (StationId INTEGER PRIMARY KEY, RiverId INTEGER REFERENCES Rivers(RiverId),
                               Name TEXT, Data BLOB);
    """)
    names = ["Lek", "Waal", "IJssel"] + [f"River{i}" for i in range(3, rows)]
    conn.executemany("INSERT INTO Rivers VALUES (?, ?, ?)", [(i + 1, n, 10.0 * (i + 1)) for i, n in enumerate(names)])
    conn.execute("INSERT INTO Stations VALUES (1, 1, 'Hagestein', x'00010203')")
    conn.commit()
    conn.close()
    return path


def make_project(tmp_path: Path, notes: str = "", **source_kwargs) -> DatabaseProject:
    db = make_rivers_db(tmp_path / "rivers.sqlite")
    notes_path = None
    if notes:
        notes_path = tmp_path / "notes.md"
        notes_path.write_text(notes, encoding="utf-8")
    source = DataSource(name="rivers",
                        backend=SQLAlchemyBackend.from_sqlite_path(db),
                        description="Dutch rivers test data",
                        domain_notes_path=notes_path,
                        **source_kwargs)
    return DatabaseProject(name="rivers",
                           title="Rivers",
                           description="You answer questions about rivers.",
                           data_sources=[source])


def add_custom_capabilities(project: DatabaseProject) -> DatabaseProject:
    """Register two project-specific capabilities: one using the safe SQL path, one using no database."""
    source = project.source("rivers")

    def river_by_name(name: str) -> str:
        """Look up a river by its exact name.

        Args:
            name: River name, e.g. "Lek".
        """
        return run_sql(source, f"SELECT RiverId, Name, LengthKm FROM Rivers WHERE Name = '{name}'")

    def km_to_miles(km: float) -> str:
        """Convert kilometres to miles.

        Args:
            km: Distance in kilometres.
        """
        return f"{km / 1.609344:.1f} miles"

    project.extra_capabilities.extend([river_by_name, km_to_miles])
    return project
