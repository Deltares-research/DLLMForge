"""
YAML configuration for database projects, validated with Pydantic (same pattern as ``IE_agent_config``).

``project.yaml`` holds non-secret project settings; Markdown files hold instructions and domain knowledge;
environment variables (``.env``) hold connection URLs, model and MCP settings. :func:`load_project` validates
the file and builds the existing :class:`~dllmforge.database.project.DatabaseProject`. Pydantic is used only
here; the database, agent and MCP code receive plain dataclasses.

Example::

    from dotenv import load_dotenv
    from dllmforge.database.agent import DatabaseAgent
    from dllmforge.database.config import load_project

    load_dotenv("database_projects/chw/.env")
    loaded = load_project("database_projects/chw/project.yaml")
    agent = DatabaseAgent(loaded.project, **loaded.agent.model_dump())
"""
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Union

from pydantic import BaseModel, ConfigDict, Field, ValidationError, ValidationInfo, field_validator, model_validator

from ..document_search import DEFAULT_INCLUDE, DocumentIndex, document_search_capability
from .backends import SQLAlchemyBackend
from .project import DatabaseProject, DataSource
from .queries import QueryParameter, SavedQuery, check_saved_query, saved_queries_capability

__all__ = [
    "AgentSettings", "DataSourceConfig", "DatabaseProjectConfig", "DocumentsConfig", "LoadedProject", "build_project",
    "load_project"
]

_ENV_NAME = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
_URL_SCHEMES = {
    "postgresql": ("postgresql", "postgres"),
    "sqlite": ("sqlite", )
}


def _resolve_file(value: Optional[Path], info: ValidationInfo) -> Optional[Path]:
    """Resolve ``value`` against the YAML directory (validation context ``base_dir``) and require it to exist."""
    if value is None:
        return None
    base_dir = (info.context or {}).get("base_dir")
    path = value if value.is_absolute() or base_dir is None else Path(base_dir) / value
    if not path.is_file():
        raise ValueError(f"file not found: {path}")
    return path


class _Model(BaseModel):
    model_config = ConfigDict(extra="forbid")


class DataSourceConfig(_Model):
    """One queryable database (becomes a ``DataSource`` with a ``SQLAlchemyBackend``)."""
    name: str = Field(pattern=r"^[a-z][a-z0-9_]*$", description="Identifier; the SQL tool is execute_<name>_sql")
    backend: Literal["postgresql", "sqlite"] = Field(description="Database engine")
    url_env: Optional[str] = Field(default=None,
                                   description="Environment variable holding the SQLAlchemy URL (postgresql). "
                                   "The value 'postgresql://' uses the standard PG* variables.")
    sqlite_path: Optional[Path] = Field(default=None, description="SQLite file, opened read-only (sqlite)")
    description: str = Field(default="", description="What the data source contains")
    schemas: Optional[List[str]] = Field(default=None, description="Schemas to expose (PostgreSQL)")
    tables: Optional[List[str]] = Field(default=None, description="Optional table allowlist and display order")
    table_notes_file: Optional[Path] = Field(default=None, description="Markdown with one '## <table>' per table")
    query_guidance: str = Field(default="", description="Extra guidance appended to the SQL tool description")
    statement_timeout_ms: int = Field(default=30_000, gt=0)
    max_rows: int = Field(default=50, gt=0)
    allow_multiple_statements: Optional[bool] = Field(default=None,
                                                      description="Default: True for SQLite, False otherwise")

    @field_validator("table_notes_file")
    @classmethod
    def _notes_exist(cls, value: Optional[Path], info: ValidationInfo) -> Optional[Path]:
        return _resolve_file(value, info)

    @field_validator("sqlite_path")
    @classmethod
    def _sqlite_exists(cls, value: Optional[Path], info: ValidationInfo) -> Optional[Path]:
        return _resolve_file(value, info)

    @model_validator(mode="after")
    def _connection(self) -> "DataSourceConfig":
        if self.backend == "sqlite":
            if self.sqlite_path is None or self.url_env is not None:
                raise ValueError("backend 'sqlite' needs sqlite_path (and no url_env)")
            return self
        if self.url_env is None or self.sqlite_path is not None:
            raise ValueError("backend 'postgresql' needs url_env (and no sqlite_path)")
        if not _ENV_NAME.match(self.url_env):
            raise ValueError("url_env must be the NAME of an environment variable, not a URL; "
                             "put the URL in .env")
        url = os.getenv(self.url_env, "").strip()
        if not url:
            raise ValueError(f"environment variable {self.url_env} is not set; add it to .env "
                             "(use 'postgresql://' to rely on the PG* variables)")
        scheme = url.split(":", 1)[0].split("+", 1)[0].lower()
        if scheme not in _URL_SCHEMES[self.backend]:
            raise ValueError(f"{self.url_env} has scheme {scheme!r}, which does not match backend {self.backend!r}")
        return self

    def url(self) -> str:
        return os.environ[self.url_env].strip() if self.url_env else ""


class QueryParameterConfig(_Model):
    type: Literal["str", "int", "float", "bool"] = "str"
    description: str = ""


class SavedQueryConfig(_Model):
    """A reusable read-only ``.sql`` file, run through the ``run_saved_query`` capability."""
    name: str = Field(pattern=r"^[a-z][a-z0-9_]*$")
    file: Path = Field(description="SELECT statement with ':name' placeholders for the parameters")
    description: str = Field(default="", description="When to use the query (shown to the AI)")
    source: Optional[str] = Field(default=None, description="Data source name; default: the only data source")
    parameters: Dict[str, QueryParameterConfig] = Field(default_factory=dict)

    @field_validator("file")
    @classmethod
    def _file_exists(cls, value: Path, info: ValidationInfo) -> Path:
        return _resolve_file(value, info)


class DocumentsConfig(_Model):
    """Project documents for the ``search_documents`` capability (keyword/BM25 search, in memory)."""
    enabled: bool = False
    paths: List[Path] = Field(default_factory=list, description="Files or directories (searched recursively)")
    include: List[str] = Field(default_factory=lambda: list(DEFAULT_INCLUDE), description="File name patterns")
    exclude: List[str] = Field(default_factory=list, description="File name or relative path patterns")
    chunk_size: int = Field(default=1000, ge=100, description="Passage size in characters")
    chunk_overlap: int = Field(default=200, ge=0)
    top_k: int = Field(default=5, ge=1, le=20, description="Passages returned per search by default")

    @field_validator("paths")
    @classmethod
    def _resolve_paths(cls, value: List[Path], info: ValidationInfo) -> List[Path]:
        base_dir = (info.context or {}).get("base_dir")
        return [p if p.is_absolute() or base_dir is None else Path(base_dir) / p for p in value]

    @model_validator(mode="after")
    def _check_enabled(self) -> "DocumentsConfig":
        if not self.enabled:
            return self
        if not self.paths:
            raise ValueError("documents.enabled is true but no paths are given")
        missing = [str(p) for p in self.paths if not p.exists()]
        if missing:
            raise ValueError(f"document paths not found: {missing}")
        if self.chunk_overlap >= self.chunk_size:
            raise ValueError("chunk_overlap must be smaller than chunk_size")
        return self


class AgentSettings(_Model):
    """Native-agent settings; field names equal the ``DatabaseAgent`` keyword arguments.

    The model provider and name come from ``<PREFIX>_MODEL_PROVIDER`` / ``<PREFIX>_MODEL_NAME``.
    """
    temperature: float = Field(default=0.1, ge=0, le=2)
    max_tool_iterations: int = Field(default=8, ge=1)


class DatabaseProjectConfig(_Model):
    """Contents of ``project.yaml``."""
    name: str = Field(pattern=r"^[a-z][a-z0-9-]*$", description="Project slug (MCP name, resource URIs)")
    title: str = Field(min_length=1)
    description: str = Field(min_length=1, description="Who the assistant is and what the data is about")
    env_prefix: str = Field(default="", description="Prefix for environment settings; default: upper-case name")
    instructions_file: Optional[Path] = Field(default=None, description="Markdown: how the AI answers")
    domain_knowledge_file: Optional[Path] = Field(default=None, description="Markdown: terminology and rules")
    examples_file: Optional[Path] = Field(default=None, description="Markdown: example questions and SQL")
    data_sources: List[DataSourceConfig] = Field(min_length=1)
    queries: List[SavedQueryConfig] = Field(default_factory=list, description="Saved .sql queries")
    documents: DocumentsConfig = Field(default_factory=DocumentsConfig)
    agent: AgentSettings = Field(default_factory=AgentSettings)

    @field_validator("instructions_file", "domain_knowledge_file", "examples_file")
    @classmethod
    def _files_exist(cls, value: Optional[Path], info: ValidationInfo) -> Optional[Path]:
        return _resolve_file(value, info)

    @field_validator("data_sources")
    @classmethod
    def _unique_sources(cls, value: List[DataSourceConfig]) -> List[DataSourceConfig]:
        names = [source.name for source in value]
        if len(set(names)) != len(names):
            raise ValueError(f"duplicate data source names: {names}")
        return value

    @model_validator(mode="after")
    def _check_queries(self) -> "DatabaseProjectConfig":
        sources = {source.name: source for source in self.data_sources}
        names = [query.name for query in self.queries]
        if len(set(names)) != len(names):
            raise ValueError(f"duplicate saved query names: {names}")
        for query in self.queries:
            if query.source is None:
                if len(sources) > 1:
                    raise ValueError(f"query '{query.name}': set source (one of {sorted(sources)})")
                query.source = next(iter(sources))
            if query.source not in sources:
                raise ValueError(f"query '{query.name}': unknown source '{query.source}'")
            try:
                check_saved_query(query.file.read_text(encoding="utf-8"), sources[query.source].backend,
                                  list(query.parameters))
            except ValueError as exc:
                raise ValueError(f"query '{query.name}' ({query.file.name}): {exc}") from None
        return self

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any], base_dir: Union[str, Path, None] = None) -> "DatabaseProjectConfig":
        """Validate a configuration dictionary; relative paths are resolved against ``base_dir``."""
        return cls.model_validate(config_dict, context={"base_dir": base_dir})

    @classmethod
    def load_from_file(cls, config_path: Union[str, Path]) -> "DatabaseProjectConfig":
        """Load and validate a YAML file. Raises ``ValueError`` naming the file and each invalid field."""
        config_path = Path(config_path).resolve()
        if config_path.suffix not in (".yaml", ".yml"):
            raise ValueError(f"Unsupported config file format: {config_path.suffix} (use .yaml)")
        if not config_path.is_file():
            raise ValueError(f"Project configuration not found: {config_path}")
        import yaml
        try:
            config_dict = yaml.safe_load(config_path.read_text(encoding="utf-8"))
        except yaml.YAMLError as exc:
            raise ValueError(f"Invalid YAML in {config_path}: {exc}") from None
        if not isinstance(config_dict, dict):
            raise ValueError(f"{config_path} must contain a mapping of settings")
        try:
            return cls.from_dict(config_dict, base_dir=config_path.parent)
        except ValidationError as exc:
            errors = "\n".join(f"  - {'.'.join(str(p) for p in error['loc']) or '<root>'}: {error['msg']}"
                               for error in exc.errors())
            raise ValueError(f"Invalid project configuration {config_path}:\n{errors}") from None


@dataclass(frozen=True)
class LoadedProject:
    """Result of :func:`load_project`."""
    project: DatabaseProject
    agent: AgentSettings
    config: DatabaseProjectConfig


def _read(path: Optional[Path]) -> str:
    return path.read_text(encoding="utf-8").strip() if path else ""


def _data_source(config: DataSourceConfig) -> DataSource:
    options: Dict[str, Any] = dict(max_rows=config.max_rows,
                                   statement_timeout_ms=config.statement_timeout_ms,
                                   allow_multiple_statements=config.allow_multiple_statements)
    if config.backend == "sqlite":
        backend = SQLAlchemyBackend.from_sqlite_path(config.sqlite_path, **options)
    else:
        backend = SQLAlchemyBackend(config.url(), schemas=config.schemas, **options)
    return DataSource(name=config.name,
                      backend=backend,
                      description=config.description,
                      tables=config.tables,
                      domain_notes_path=config.table_notes_file,
                      query_guidance=config.query_guidance)


def build_project(config: DatabaseProjectConfig) -> DatabaseProject:
    """Build the :class:`DatabaseProject` described by a validated configuration.

    Opens no database connection; builds the in-memory document index when ``documents.enabled``.
    """
    data_sources = [_data_source(source) for source in config.data_sources]
    saved = [
        SavedQuery(query.name, query.file.read_text(encoding="utf-8"), query.source or data_sources[0].name,
                   query.description,
                   tuple(QueryParameter(name, p.type, p.description) for name, p in query.parameters.items()))
        for query in config.queries
    ]
    extras = [saved_queries_capability(saved, data_sources)] if saved else []
    documents = config.documents
    if documents.enabled:
        index = DocumentIndex.from_paths(documents.paths, documents.include, documents.exclude, documents.chunk_size,
                                         documents.chunk_overlap)
        extras.append(document_search_capability(index, documents.top_k))
    instructions = "\n\n".join(text for text in (_read(config.instructions_file), _read(config.domain_knowledge_file))
                               if text)
    return DatabaseProject(name=config.name,
                           title=config.title,
                           description=config.description,
                           data_sources=data_sources,
                           extra_capabilities=extras,
                           instructions=instructions,
                           examples=_read(config.examples_file),
                           env_prefix=config.env_prefix)


def load_project(config_path: Union[str, Path]) -> LoadedProject:
    """Validate ``project.yaml`` and build the project. Load ``.env`` before calling this."""
    config = DatabaseProjectConfig.load_from_file(config_path)
    return LoadedProject(project=build_project(config), agent=config.agent, config=config)
