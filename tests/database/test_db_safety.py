import pytest

from dllmforge.database.safety import UnsafeQueryError, split_statements, validate_read_only


@pytest.mark.parametrize("sql", [
    "SELECT 1",
    "select * from t where name = 'drop table x'",
    "WITH a AS (SELECT 1 AS x) SELECT x FROM a",
    "SELECT replace(name, 'a', 'b') FROM t",
    'SELECT "update" FROM t',
    "SELECT 1 -- delete everything\n",
    "SELECT 1 /* insert */",
    "SELECT 1;",
])
def test_allowed_sqlite(sql):
    assert validate_read_only(sql, "sqlite")


@pytest.mark.parametrize("sql, word", [
    ("DELETE FROM t", "DELETE"),
    ("UPDATE t SET a = 1", "UPDATE"),
    ("PRAGMA writable_schema = 1", "PRAGMA"),
    ("ATTACH DATABASE 'x' AS y", "ATTACH"),
])
def test_rejects_non_select_with_boi_message(sql, word):
    with pytest.raises(UnsafeQueryError) as err:
        validate_read_only(sql, "sqlite")
    assert str(err.value) == (f"ERROR: Only SELECT queries are permitted. Received '{word}'. "
                              "Rewrite as a SELECT statement.")


@pytest.mark.parametrize("sql", [
    "WITH d AS (DELETE FROM t RETURNING *) SELECT * FROM d",
    "SELECT * INTO new_table FROM t",
    "SELECT pg_read_file('/etc/passwd')",
    "SELECT pg_sleep(100)",
    "SELECT set_config('default_transaction_read_only', 'off', false)",
    "SELECT * FROM dblink('host=x', 'select 1') AS t(a int)",
    "SELECT nextval('seq')",
])
def test_rejects_postgres_attacks(sql):
    with pytest.raises(UnsafeQueryError):
        validate_read_only(sql, "postgresql")


def test_dollar_quoted_strings_are_literals():
    assert validate_read_only("SELECT $$ drop table x $$", "postgresql")
    assert validate_read_only("SELECT $tag$ delete $tag$", "postgresql")


def test_load_extension_blocked_in_sqlite():
    with pytest.raises(UnsafeQueryError):
        validate_read_only("SELECT load_extension('x')", "sqlite")


def test_multiple_statements_need_opt_in():
    with pytest.raises(UnsafeQueryError, match="exactly one SELECT"):
        validate_read_only("SELECT 1; SELECT 2", "postgresql")
    assert validate_read_only("SELECT 1; SELECT 2", "sqlite", allow_multiple=True) == ["SELECT 1", "SELECT 2"]


def test_write_in_batch_reports_statement_number():
    with pytest.raises(UnsafeQueryError) as err:
        validate_read_only("SELECT 1; DELETE FROM t", "sqlite", allow_multiple=True)
    assert str(err.value).startswith("Statement 2: ERROR: Only SELECT queries are permitted. Received 'DELETE'.")


def test_split_ignores_semicolons_in_literals_and_comments():
    assert split_statements("SELECT ';'; SELECT 2 -- ; x\n") == ["SELECT ';'", "SELECT 2 -- ; x"]


@pytest.mark.parametrize("sql", ["", "   ", ";", "-- only a comment"])
def test_empty_rejected(sql):
    with pytest.raises(UnsafeQueryError):
        validate_read_only(sql, "sqlite")
