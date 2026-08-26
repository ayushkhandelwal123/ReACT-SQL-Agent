"""
db_tools.py
-----------
Thin sqlite3 wrappers exposed as LangChain @tool functions.

These are the only three things the agent is allowed to do to the database:
  1. sql_db_list_tables  - see what tables exist
  2. sql_db_schema       - see a table's columns + a few sample rows
  3. sql_db_query        - run a SELECT and get results back

Deliberately minimal, per LangChain's own tutorial warning: this is NOT
hardened for production. In a real deployment, connect with a DB user that
only has SELECT permission, and add your own validation before executing
model-generated SQL.
"""

import sqlite3
from langchain.tools import tool

DB_PATH = "chinook.db"


@tool
def sql_db_list_tables() -> str:
    """Input is an empty string, output is a comma-separated list of tables in the database."""
    con = sqlite3.connect(DB_PATH)
    try:
        cursor = con.cursor()
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
        tables = [row[0] for row in cursor.fetchall() if not row[0].startswith("sqlite_")]
        return ", ".join(tables)
    finally:
        con.close()


@tool
def sql_db_schema(table_names: str) -> str:
    """Input is a comma-separated list of tables, output is the schema and sample rows for those tables.
    Be sure the tables actually exist by calling sql_db_list_tables first.
    Example input: table1, table2, table3
    """
    con = sqlite3.connect(DB_PATH)
    try:
        cursor = con.cursor()
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
        valid_tables = {row[0] for row in cursor.fetchall() if not row[0].startswith("sqlite_")}

        results = []
        for table in table_names.split(","):
            table = table.strip()
            if table not in valid_tables:
                results.append(f"Error: table {table!r} not found in database")
                continue

            cursor.execute(
                "SELECT sql FROM sqlite_master WHERE type='table' AND name=?;",
                (table,),
            )
            schema_row = cursor.fetchone()
            if schema_row:
                results.append(schema_row[0])
                try:
                    quoted_table = '"' + table.replace('"', '""') + '"'
                    cursor.execute(f"SELECT * FROM {quoted_table} LIMIT 3;")
                    rows = cursor.fetchall()
                    if rows:
                        col_names = [d[0] for d in cursor.description]
                        results.append(
                            f"/*\n3 rows from {table} table:\n"
                            + "\t".join(col_names)
                            + "\n"
                            + "\n".join("\t".join(str(x) for x in row) for row in rows)
                            + "\n*/"
                        )
                except Exception as e:
                    results.append(f"Error fetching sample rows: {e}")
        return "\n\n".join(results)
    finally:
        con.close()


@tool
def sql_db_query(query: str) -> str:
    """Input is a detailed and correct SQL query, output is a result from the database.
    If the query is not correct, an error message is returned — rewrite and retry.
    If you see 'Unknown column' errors, call sql_db_schema again to check field names.
    """
    con = sqlite3.connect(DB_PATH)
    try:
        cursor = con.cursor()
        cursor.execute(query)
        return str(cursor.fetchall())
    except Exception as e:
        return f"Error: {e}"
    finally:
        con.close()


TOOLS = [sql_db_list_tables, sql_db_schema, sql_db_query]


if __name__ == "__main__":
    # Quick sanity check you can run directly: python db_tools.py
    for t in TOOLS:
        print(f"{t.name}: {t.description}\n")
