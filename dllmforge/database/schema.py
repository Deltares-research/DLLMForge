"""
Schema rendering and markdown domain notes for database instructions.

Domain notes are markdown files with one level-2 heading per table, e.g. ``## HRDLocations`` or
``## Table: `locaties```. Lookup is case-insensitive and also matches unqualified table names, so
``## segments`` documents ``coastal.segments``.
"""
import logging
from pathlib import Path
from typing import Dict, List, Optional

from .backends import TableInfo

__all__ = ["load_domain_notes", "render_table", "render_tables"]

logger = logging.getLogger(__name__)


def _heading_key(heading: str) -> str:
    heading = heading.strip()
    if heading.lower().startswith("table:"):
        heading = heading[len("table:"):]
    return heading.strip().strip("`").strip().lower()


def load_domain_notes(path: Optional[str | Path]) -> Dict[str, str]:
    """Parse a markdown file into ``{lowercase heading: section text}``.

    HTML comment lines and blocks are skipped. Repeated headings are merged (not overwritten).
    A missing path yields an empty dict.
    """
    if not path:
        return {}
    file = Path(path)
    if not file.is_file():
        logger.warning("Domain notes file not found: %s", file)
        return {}

    sections: Dict[str, List[str]] = {}
    current: Optional[str] = None
    in_comment = False
    for line in file.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if in_comment:
            in_comment = "-->" not in stripped
            continue
        if stripped.startswith("<!--"):
            in_comment = "-->" not in stripped
            continue
        if line.startswith("## "):
            current = _heading_key(line[3:])
            if current in sections:
                logger.warning("Duplicate domain-notes heading %r in %s; sections are merged", current, file)
                sections[current].append("")
            else:
                sections[current] = []
            continue
        if current is not None:
            sections[current].append(line)
    return {key: "\n".join(lines).strip() for key, lines in sections.items()}


def _notes_for(table: str, notes: Dict[str, str]) -> str:
    key = table.lower()
    return notes.get(key) or notes.get(key.split(".")[-1], "")


def render_table(table: TableInfo, notes: Optional[Dict[str, str]] = None) -> str:
    """Render one table with columns, row count, foreign keys and domain notes."""
    if table.row_count is None:
        count = "row count unknown"
    else:
        count = f"{'~' if table.row_count_is_estimate else ''}{table.row_count:,} rows"
    kind = "" if table.kind == "table" else f", {table.kind}"
    lines = [f"TABLE {table.name}  ({count}{kind})"]
    if table.comment:
        lines.append(f"  DESCRIPTION: {table.comment}")
    lines.append("  COLUMNS:")
    for column in table.columns:
        markers = (["PK"] if column.primary_key else []) + ([] if column.nullable else ["NOT NULL"])
        marker = f" [{', '.join(markers)}]" if markers else ""
        comment = f"  -- {column.comment}" if column.comment else ""
        lines.append(f"    {column.name} {column.data_type}{marker}{comment}")
    if table.foreign_keys:
        lines.append("  FOREIGN KEYS:")
        lines.extend(f"    {fk}" for fk in table.foreign_keys)
    domain = _notes_for(table.name, notes or {})
    if domain:
        lines.append("  DOMAIN NOTES:")
        lines.extend(f"  {line}" for line in domain.splitlines())
    return "\n".join(lines)


def render_tables(tables: List[TableInfo], notes: Optional[Dict[str, str]] = None) -> str:
    """Render several tables separated by blank lines."""
    return "\n\n".join(render_table(table, notes) for table in tables)
