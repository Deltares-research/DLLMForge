"""
Read-only SQL validation for database capabilities.

This is defence in depth. The real guard is the backend connection: SQLite is opened read-only and
PostgreSQL sessions run in read-only transactions, ideally under a read-only database role.

Policy: every statement must start with SELECT or WITH, must not contain data-modifying keywords
(which also blocks data-modifying CTEs, SELECT ... INTO and SELECT ... FOR UPDATE), and must not call
functions with side effects or file access.
"""
import re
from typing import Iterator, List, Tuple

__all__ = ["UnsafeQueryError", "split_statements", "strip_literals_and_comments", "validate_read_only"]

_COMMON_BLOCKED = {
    "insert", "update", "delete", "drop", "create", "alter", "truncate", "merge", "grant", "revoke", "into"
}
_BLOCKED_KEYWORDS = {
    "sqlite": _COMMON_BLOCKED | {"replace", "attach", "detach", "pragma", "vacuum", "reindex"},
    "postgresql": _COMMON_BLOCKED,
}
_BLOCKED_FUNCTIONS = {
    "sqlite": {"load_extension", "readfile", "writefile"},
    "postgresql": {
        "pg_read_file", "pg_read_binary_file", "pg_ls_dir", "pg_stat_file", "lo_import", "lo_export", "lo_unlink",
        "dblink", "dblink_exec", "set_config", "pg_terminate_backend", "pg_cancel_backend", "pg_reload_conf",
        "pg_sleep", "pg_advisory_lock", "nextval", "setval"
    },
}
_ALLOWED_FIRST_KEYWORDS = {"select", "with"}
# Blocked keywords that are also harmless scalar functions, e.g. SQLite's replace(x, y, z).
_FUNCTION_LIKE_KEYWORDS = {"replace"}

_WORD = re.compile(r"[A-Za-z_][A-Za-z0-9_$]*")
_DOLLAR_TAG = re.compile(r"\$([A-Za-z_][A-Za-z0-9_]*)?\$")


class UnsafeQueryError(ValueError):
    """Raised when a submitted query is not an allowed read-only statement."""


def _scan(sql: str) -> Iterator[Tuple[str, str]]:
    """Yield ``(kind, text)`` chunks; kind is 'code', 'string', 'ident' (double-quoted) or 'comment'."""
    i = start = 0
    n = len(sql)
    while i < n:
        ch = sql[i]
        if sql.startswith("--", i) or sql.startswith("/*", i):
            if start < i:
                yield "code", sql[start:i]
            if ch == "-":
                end = sql.find("\n", i)
                end = n if end == -1 else end
            else:
                end = sql.find("*/", i + 2)
                end = n if end == -1 else end + 2
            yield "comment", sql[i:end]
            i = start = end
        elif ch in ("'", '"'):
            if start < i:
                yield "code", sql[start:i]
            j = i + 1
            while j < n:
                if sql[j] == ch:
                    if j + 1 < n and sql[j + 1] == ch:  # doubled quote = escaped quote
                        j += 2
                        continue
                    break
                j += 1
            yield ("string" if ch == "'" else "ident"), sql[i:j + 1]
            i = start = j + 1
        elif ch == "$" and (i == 0 or not (sql[i - 1].isalnum() or sql[i - 1] == "_")):
            match = _DOLLAR_TAG.match(sql, i)
            if not match:
                i += 1
                continue
            if start < i:
                yield "code", sql[start:i]
            tag = match.group(0)
            end = sql.find(tag, match.end())
            end = n if end == -1 else end + len(tag)
            yield "string", sql[i:end]
            i = start = end
        else:
            i += 1
    if start < n:
        yield "code", sql[start:n]


def strip_literals_and_comments(sql: str) -> str:
    """Return the SQL with string literals, quoted identifiers and comments replaced by a space."""
    return "".join(chunk if kind == "code" else " " for kind, chunk in _scan(sql))


def split_statements(sql: str) -> List[str]:
    """Split SQL on semicolons outside literals and comments; empty statements are dropped."""
    statements: List[str] = []
    current = ""
    for kind, chunk in _scan(sql):
        if kind != "code":
            current += chunk
            continue
        parts = chunk.split(";")
        for index, part in enumerate(parts):
            current += part
            if index < len(parts) - 1:
                statements.append(current.strip())
                current = ""
    statements.append(current.strip())
    return [s for s in statements if strip_literals_and_comments(s).strip()]


def _validate_statement(statement: str, dialect: str) -> None:
    code = strip_literals_and_comments(statement)
    words = _WORD.findall(code)
    if not words:
        raise UnsafeQueryError("ERROR: Empty query submitted.")

    first = words[0].lower()
    if first not in _ALLOWED_FIRST_KEYWORDS:
        raise UnsafeQueryError(f"ERROR: Only SELECT queries are permitted. Received '{first.upper()}'. "
                               "Rewrite as a SELECT statement.")

    for match in _WORD.finditer(code):
        word = match.group(0).lower()
        is_call = code[match.end():].lstrip().startswith("(")
        if word in _BLOCKED_KEYWORDS[dialect] and not (word in _FUNCTION_LIKE_KEYWORDS and is_call):
            raise UnsafeQueryError(f"ERROR: Forbidden keyword '{word.upper()}' detected. "
                                   "Only read-only SELECT queries are allowed.")
        if word in _BLOCKED_FUNCTIONS[dialect] and is_call:
            raise UnsafeQueryError(f"ERROR: Forbidden function '{word}' detected. "
                                   "Only read-only SELECT queries are allowed.")


def validate_read_only(sql: str, dialect: str, allow_multiple: bool = False) -> List[str]:
    """Validate that ``sql`` only contains read-only SELECT/WITH statements.

    Args:
        sql: SQL text submitted by a client or LLM.
        dialect: ``"sqlite"`` or ``"postgresql"``.
        allow_multiple: Accept several semicolon-separated statements.

    Returns:
        The individual statements without trailing semicolons.

    Raises:
        UnsafeQueryError: With an LLM-readable message. For multi-statement input the message is
            prefixed with ``"Statement <n>: "``.
    """
    if dialect not in _BLOCKED_KEYWORDS:
        raise ValueError(f"Unsupported SQL dialect: {dialect!r}")
    statements = split_statements(sql)
    if not statements:
        raise UnsafeQueryError("ERROR: Empty query submitted.")
    if len(statements) > 1 and not allow_multiple:
        raise UnsafeQueryError("ERROR: Submit exactly one SELECT statement per call.")
    for number, statement in enumerate(statements, 1):
        try:
            _validate_statement(statement, dialect)
        except UnsafeQueryError as exc:
            if len(statements) > 1:
                raise UnsafeQueryError(f"Statement {number}: {exc}") from None
            raise
    return statements
