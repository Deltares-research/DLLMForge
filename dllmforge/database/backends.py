"""
Read-only query backend for SQLite and PostgreSQL/PostGIS, built on SQLAlchemy 2.

The backend executes single, already validated statements (see ``safety.py``) and introspects the schema.
Each call uses its own connection and transaction, which is always rolled back, so one backend can be
shared between threads (MCP server). SQLAlchemy and the database drivers are imported lazily: importing
this module needs neither (``pip install dllmforge[database]`` installs them).
"""
import struct
import threading
import time
import warnings
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterator, List, Mapping, Optional, Protocol, Sequence, Tuple, runtime_checkable
from urllib.parse import quote

__all__ = [
    "ColumnInfo",
    "TableInfo",
    "QueryResult",
    "QueryBackend",
    "SQLAlchemyBackend",
    "SUPPORTED_DIALECTS",
    "render_value",
]

SUPPORTED_DIALECTS = ("sqlite", "postgresql")
DEFAULT_MAX_ROWS = 50
DEFAULT_MAX_CELL_CHARS = 300


@dataclass
class ColumnInfo:
    name: str
    data_type: str
    nullable: bool = True
    primary_key: bool = False
    comment: Optional[str] = None


@dataclass
class TableInfo:
    name: str
    columns: List[ColumnInfo]
    row_count: Optional[int] = None
    row_count_is_estimate: bool = False
    kind: str = "table"
    comment: Optional[str] = None
    foreign_keys: List[str] = field(default_factory=list)


@dataclass
class QueryResult:
    """Result of one read-only statement, truncated to ``max_rows``."""

    sql: str
    columns: List[str]
    rows: List[Tuple[str, ...]]
    truncated: bool
    max_rows: int

    def to_text(self) -> str:
        """Render as an LLM-friendly text table that always ends with the executed SQL."""
        if not self.rows:
            return f"Query returned 0 rows.\n\nExecuted SQL:\n{self.sql}"
        widths = [max(len(col), *(len(row[i]) for row in self.rows)) for i, col in enumerate(self.columns)]
        header = " | ".join(col.ljust(widths[i]) for i, col in enumerate(self.columns))
        separator = "-+-".join("-" * width for width in widths)
        body = [" | ".join(cell.ljust(widths[i]) for i, cell in enumerate(row)) for row in self.rows]
        if self.truncated:
            note = f"(first {self.max_rows} rows shown — more rows exist; add filters, aggregation or LIMIT)"
        else:
            note = f"({len(self.rows)} row{'s' if len(self.rows) != 1 else ''})"
        table = "\n".join([header, separator] + body)
        return f"Results {note}:\n{table}\n\nExecuted SQL:\n{self.sql}"


@runtime_checkable
class QueryBackend(Protocol):
    """What the database capabilities need from a database engine."""

    dialect: str
    max_rows: int
    allow_multiple_statements: bool

    def execute(self, statement: str, params: Optional[Mapping[str, Any]] = None) -> QueryResult:
        """Execute one validated read-only statement; ``params`` are bound to ``:name`` placeholders."""
        ...

    def list_tables(self) -> List[str]:
        """Return table and view names as they must be written in SQL."""
        ...

    def describe_tables(self, tables: Optional[Sequence[str]] = None) -> List[TableInfo]:
        """Describe the given tables (all tables when ``None``), in the given order."""
        ...


def _ewkb_summary(value: str) -> Optional[str]:
    """Summarise a hex (E)WKB geometry as '<geometry Point SRID=4326 …>', or None if not WKB."""
    if len(value) < 18 or len(value) % 2 or value[:2] not in ("00", "01"):
        return None
    try:
        raw = bytes.fromhex(value[:18])
    except ValueError:
        return None
    endian = "<" if raw[0] == 1 else ">"
    (type_code, ) = struct.unpack(endian + "I", raw[1:5])
    has_srid = bool(type_code & 0x20000000)
    base = type_code & 0x0FFFFFFF
    base = base % 1000 if base < 4000 else base
    names = {
        1: "Point",
        2: "LineString",
        3: "Polygon",
        4: "MultiPoint",
        5: "MultiLineString",
        6: "MultiPolygon",
        7: "GeometryCollection"
    }
    if base not in names:
        return None
    srid = f" SRID={struct.unpack(endian + 'I', raw[5:9])[0]}" if has_srid else ""
    return f"<geometry {names[base]}{srid}; select ST_AsText(...) or ST_AsGeoJSON(...) for coordinates>"


def render_value(value: Any, max_chars: int = DEFAULT_MAX_CELL_CHARS, is_geometry: bool = False) -> str:
    """Render one cell for the LLM: NULL, compact binary/geometry, truncated long text."""
    if value is None:
        return "NULL"
    if isinstance(value, (bytes, bytearray, memoryview)):
        return f"<binary {len(bytes(value))} bytes>"
    text = str(value)
    if is_geometry:
        summary = _ewkb_summary(text)
        if summary:
            return summary
    text = text.replace("\n", " ")
    if len(text) > max_chars:
        return f"{text[:max_chars]}…(+{len(text) - max_chars} chars)"
    return text


def _ordered_subset(available: List[str], wanted: Optional[Sequence[str]]) -> List[str]:
    """Keep ``wanted`` order; match case-insensitively and also on the unqualified table name."""
    if wanted is None:
        return list(available)
    lookup: Dict[str, str] = {}
    for name in available:
        lookup.setdefault(name.lower(), name)
        lookup.setdefault(name.split(".")[-1].lower(), name)
    result = []
    for name in wanted:
        match = lookup.get(name.lower())
        if match and match not in result:
            result.append(match)
    return result


_PG_DRIVERS = ("psycopg", "psycopg2")
_PG_GEOMETRY_TYPES = "SELECT oid FROM pg_type WHERE typname IN ('geometry', 'geography')"
_PG_CAN_SELECT = "SELECT has_table_privilege(:relation, 'SELECT')"
_PG_ROW_ESTIMATE = "SELECT CAST(reltuples AS bigint) FROM pg_class WHERE oid = to_regclass(:relation)"
_PG_COLUMN_TYPES = """
    SELECT a.attname, format_type(a.atttypid, a.atttypmod)
    FROM pg_attribute a
    WHERE a.attrelid = to_regclass(:relation) AND a.attnum > 0 AND NOT a.attisdropped"""


def _sqlalchemy() -> Any:
    try:
        import sqlalchemy
    except ImportError as exc:
        raise ImportError("SQLAlchemyBackend requires SQLAlchemy: pip install 'dllmforge[database]'") from exc
    return sqlalchemy


def _sqlite_query_only(dbapi_connection: Any, _record: Any) -> None:
    dbapi_connection.execute("PRAGMA query_only = ON")


def _postgres_read_only(dbapi_connection: Any, _record: Any) -> None:
    # Every transaction on this connection starts READ ONLY (psycopg 3 / psycopg2 respectively).
    if hasattr(dbapi_connection, "read_only"):
        dbapi_connection.read_only = True
    else:
        dbapi_connection.set_session(readonly=True)


def _type_name(sa_type: Any) -> str:
    try:
        return str(sa_type)
    except Exception:  # e.g. NullType for types SQLAlchemy does not know
        return type(sa_type).__name__


def _format_foreign_key(fk: Dict[str, Any]) -> str:
    target = ".".join(part for part in (fk.get("referred_schema"), fk["referred_table"]) if part)
    columns, referred = fk["constrained_columns"], fk["referred_columns"]
    if len(columns) == 1 and len(referred) == 1:
        return f"{columns[0]} -> {target}.{referred[0]}"
    return f"({', '.join(columns)}) -> {target}({', '.join(referred)})"


class SQLAlchemyBackend:
    """Read-only backend for SQLite and PostgreSQL/PostGIS on SQLAlchemy 2 (``pip install dllmforge[database]``).

    SQLAlchemy provides connections, statement execution and schema reflection for both engines. What it
    does not abstract is handled per dialect: read-only enforcement, statement timeouts, server-side row
    limiting, row-count estimates and geometry rendering. Other engines are rejected because their
    read-only enforcement is not implemented.

    - **SQLite**: open files with :meth:`from_sqlite_path` (``mode=ro`` URI, never creates the file);
      ``PRAGMA query_only`` on every connection; Python-side timeout; no connection pooling.
    - **PostgreSQL**: every transaction is READ ONLY with ``SET LOCAL statement_timeout``; a server-side
      cursor transfers at most ``max_rows + 1`` rows; PostGIS geometry values are summarised; table names
      are schema-qualified for the listed ``schemas``. ``postgresql://`` URLs use psycopg 3. Use a read-only
      database role as well.

    Args:
        url: SQLAlchemy URL, e.g. ``postgresql://user:pw@host:5432/db``. An URL without host or database
            (``postgresql://``) uses the standard ``PG*`` environment variables.
        schemas: Schemas to expose; their tables are reported as ``schema.table``. ``None`` exposes the
            default schema with unqualified names.
        max_rows: Maximum rows returned per statement.
        statement_timeout_ms: Time limit per statement.
        connect_timeout_s: PostgreSQL connection timeout.
        allow_multiple_statements: Accept several statements per call; defaults to ``True`` for SQLite
            (BOI parity) and ``False`` otherwise.
        exact_row_counts: PostgreSQL only: ``COUNT(*)`` instead of planner estimates. SQLite always counts.
        max_cell_chars: Truncate rendered cells to this length.
        engine_kwargs: Extra keyword arguments for :func:`sqlalchemy.create_engine`.
    """

    def __init__(self,
                 url: Any,
                 schemas: Optional[Sequence[str]] = None,
                 max_rows: int = DEFAULT_MAX_ROWS,
                 statement_timeout_ms: int = 30_000,
                 connect_timeout_s: int = 10,
                 allow_multiple_statements: Optional[bool] = None,
                 exact_row_counts: bool = False,
                 max_cell_chars: int = DEFAULT_MAX_CELL_CHARS,
                 engine_kwargs: Optional[Dict[str, Any]] = None):
        parsed = _sqlalchemy().engine.make_url(url)
        if parsed.drivername == "postgresql":
            parsed = parsed.set(drivername="postgresql+psycopg")
        dialect = parsed.get_backend_name()
        if dialect not in SUPPORTED_DIALECTS:
            raise ValueError(f"Unsupported database {dialect!r}; read-only access is implemented for "
                             f"{', '.join(SUPPORTED_DIALECTS)}")
        if dialect == "postgresql" and parsed.get_driver_name() not in _PG_DRIVERS:
            raise ValueError(f"Unsupported PostgreSQL driver {parsed.get_driver_name()!r}; use psycopg or psycopg2")
        self.url = parsed
        self.dialect = dialect
        self.schemas: List[Optional[str]] = list(schemas) if schemas else [None]
        self.max_rows = max_rows
        self.statement_timeout_ms = int(statement_timeout_ms)
        self.connect_timeout_s = connect_timeout_s
        self.allow_multiple_statements = (dialect == "sqlite"
                                          if allow_multiple_statements is None else allow_multiple_statements)
        self.exact_row_counts = exact_row_counts
        self.max_cell_chars = max_cell_chars
        self.sqlite_path: Optional[Path] = None
        self._engine_kwargs = dict(engine_kwargs or {})
        self._engine: Any = None
        self._geometry_oids: Optional[set] = None
        self._lock = threading.Lock()

    @classmethod
    def from_sqlite_path(cls, path: str | Path, **kwargs: Any) -> "SQLAlchemyBackend":
        """Open a SQLite file read-only (``mode=ro`` URI). A missing file raises instead of being created."""
        path = Path(path)
        url = _sqlalchemy().engine.URL.create("sqlite",
                                              database=f"file:{quote(str(path.resolve()))}",
                                              query={
                                                  "mode": "ro",
                                                  "uri": "true"
                                              })
        backend = cls(url, **kwargs)
        backend.sqlite_path = path
        return backend

    def __repr__(self) -> str:
        # Never expose the password.
        schemas = [schema for schema in self.schemas if schema]
        extra = f", schemas={schemas!r}" if schemas else ""
        return f"SQLAlchemyBackend(url={self.url.render_as_string(hide_password=True)!r}{extra})"

    # ── connections ────────────────────────────────────────────────────────

    @property
    def engine(self) -> Any:
        """The SQLAlchemy engine, created on first use."""
        with self._lock:
            if self._engine is None:
                self._engine = self._create_engine()
            return self._engine

    def _create_engine(self) -> Any:
        sa = _sqlalchemy()
        kwargs = dict(self._engine_kwargs)
        if self.dialect == "sqlite":
            kwargs.setdefault("poolclass", sa.pool.NullPool)
        else:
            kwargs.setdefault("pool_pre_ping", True)
            connect_args = dict(kwargs.pop("connect_args", {}))
            connect_args.setdefault("connect_timeout", self.connect_timeout_s)
            kwargs["connect_args"] = connect_args
        engine = sa.create_engine(self.url, **kwargs)
        sa.event.listen(engine, "connect", _sqlite_query_only if self.dialect == "sqlite" else _postgres_read_only)
        return engine

    @contextmanager
    def _connect(self) -> Iterator[Any]:
        """A read-only connection with the statement timeout. Rolled back on exit; driver errors unwrapped."""
        if self.sqlite_path is not None and not self.sqlite_path.is_file():
            raise FileNotFoundError(f"SQLite database not found: {self.sqlite_path}")
        sa = _sqlalchemy()
        try:
            with self.engine.connect() as conn:
                if self.dialect == "postgresql":
                    conn.exec_driver_sql(f"SET LOCAL statement_timeout = {self.statement_timeout_ms}")
                    yield conn
                    return
                raw = conn.connection.driver_connection
                if self.statement_timeout_ms:
                    deadline = time.monotonic() + self.statement_timeout_ms / 1000
                    raw.set_progress_handler(lambda: int(time.monotonic() > deadline), 10_000)
                try:
                    yield conn
                finally:
                    raw.set_progress_handler(None, 0)
        except sa.exc.DBAPIError as exc:
            # Report the driver's own error (concise, and what callers and tests expect).
            if self.dialect == "sqlite" and str(exc.orig) == "interrupted":
                raise TimeoutError(f"Query exceeded the {self.statement_timeout_ms / 1000:g} s time limit. "
                                   "Add filters or LIMIT.") from None
            raise exc.orig from None

    # ── QueryBackend ───────────────────────────────────────────────────────

    def execute(self, statement: str, params: Optional[Mapping[str, Any]] = None) -> QueryResult:
        # Without params: no_parameters, so '%' in LLM-written SQL is not read as a driver placeholder.
        # With params: text() binds ':name' placeholders (and escapes '%'); values never enter the SQL string.
        options: Dict[str, Any] = {} if params is not None else {"no_parameters": True}
        sa = _sqlalchemy()
        with self._connect() as conn:
            geometry_oids: set = set()
            if self.dialect == "postgresql":
                geometry_oids = self._geometry_type_oids(conn)
                options.update(stream_results=True, max_row_buffer=self.max_rows + 1)
            if params is not None:
                result = conn.execute(sa.text(statement), dict(params), execution_options=options)
            else:
                result = conn.exec_driver_sql(statement, execution_options=options)
            description = (result.cursor.description or []) if result.returns_rows else []
            fetched = result.fetchmany(self.max_rows + 1) if result.returns_rows else []
            result.close()
        columns = [d[0] for d in description]
        is_geometry = [d[1] in geometry_oids for d in description]
        rows = [
            tuple(render_value(v, self.max_cell_chars, is_geometry[i]) for i, v in enumerate(row))
            for row in fetched[:self.max_rows]
        ]
        return QueryResult(statement, columns, rows, len(fetched) > self.max_rows, self.max_rows)

    def list_tables(self) -> List[str]:
        with self._connect() as conn:
            return list(self._relations(conn))

    def describe_tables(self, tables: Optional[Sequence[str]] = None) -> List[TableInfo]:
        sa = _sqlalchemy()
        with self._connect() as conn, warnings.catch_warnings():
            # Without GeoAlchemy2, PostGIS types reflect as NullType; the native type is read separately.
            warnings.filterwarnings("ignore", message="Did not recognize type")
            relations = self._relations(conn)
            inspector = sa.inspect(conn)
            return [
                self._describe(conn, inspector, name, *relations[name])
                for name in _ordered_subset(list(relations), tables)
            ]

    # ── helpers ────────────────────────────────────────────────────────────

    def _qualified(self, conn: Any, schema: Optional[str], name: str) -> str:
        preparer = conn.dialect.identifier_preparer
        return f"{preparer.quote_schema(schema)}.{preparer.quote(name)}" if schema else preparer.quote(name)

    def _relations(self, conn: Any) -> Dict[str, Tuple[Optional[str], str, str]]:
        """``{reported name: (schema, name, kind)}`` for the readable tables and views, sorted per schema."""
        sa = _sqlalchemy()
        inspector = sa.inspect(conn)
        found: Dict[str, Tuple[Optional[str], str, str]] = {}
        for schema in self.schemas:
            names = [(n, "table") for n in inspector.get_table_names(schema=schema)]
            names += [(n, "view") for n in inspector.get_view_names(schema=schema)]
            try:
                names += [(n, "materialized view") for n in inspector.get_materialized_view_names(schema=schema)]
            except NotImplementedError:
                pass
            for name, kind in sorted(names):
                found[f"{schema}.{name}" if schema else name] = (schema, name, kind)
        if self.dialect == "postgresql":
            query = sa.text(_PG_CAN_SELECT)
            found = {
                reported: item
                for reported, item in found.items() if conn.execute(query, {
                    "relation": self._qualified(conn, item[0], item[1])
                }).scalar()
            }
        return found

    def _describe(self, conn: Any, inspector: Any, reported: str, schema: Optional[str], name: str,
                  kind: str) -> TableInfo:
        sa = _sqlalchemy()
        qualified = self._qualified(conn, schema, name)
        native_types: Dict[str, str] = {}
        if self.dialect == "postgresql":
            native_types = dict(conn.execute(sa.text(_PG_COLUMN_TYPES), {"relation": qualified}).all())
        primary_key = set(inspector.get_pk_constraint(name, schema=schema).get("constrained_columns") or [])
        columns = [
            ColumnInfo(name=c["name"],
                       data_type=native_types.get(c["name"]) or _type_name(c["type"]),
                       nullable=bool(c.get("nullable", True)),
                       primary_key=c["name"] in primary_key,
                       comment=c.get("comment")) for c in inspector.get_columns(name, schema=schema)
        ]
        foreign_keys = [_format_foreign_key(fk) for fk in inspector.get_foreign_keys(name, schema=schema)]
        try:
            comment = inspector.get_table_comment(name, schema=schema).get("text")
        except NotImplementedError:
            comment = None
        if self.dialect == "postgresql" and not self.exact_row_counts:
            estimate = conn.execute(sa.text(_PG_ROW_ESTIMATE), {"relation": qualified}).scalar()
            row_count, is_estimate = (estimate if estimate is not None and estimate >= 0 else None), True
        else:
            row_count, is_estimate = conn.exec_driver_sql(f"SELECT COUNT(*) FROM {qualified}").scalar(), False
        return TableInfo(name=reported,
                         columns=columns,
                         row_count=row_count,
                         row_count_is_estimate=is_estimate,
                         kind=kind,
                         comment=comment,
                         foreign_keys=foreign_keys)

    def _geometry_type_oids(self, conn: Any) -> set:
        if self._geometry_oids is None:
            self._geometry_oids = set(conn.exec_driver_sql(_PG_GEOMETRY_TYPES).scalars().all())
        return self._geometry_oids
