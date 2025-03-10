import os

from dotenv import load_dotenv
from sqlalchemy.exc import NoSuchTableError
from typing import List

from sqlalchemy import create_engine, Engine, text, MetaData, Table


def get_database_engine(url=None, pool_size=None) -> Engine:
    if url is None:
        load_dotenv()  # take environment variables from .env.
        DB_USER = os.getenv("DB_USER")
        DB_PASSWORD = os.getenv("DB_PASSWORD")
        DB_HOST = os.getenv("DB_HOST")
        DB_PORT = os.getenv("DB_PORT")
        DB_NAME = os.getenv("DB_NAME")
        url = f"postgresql+psycopg2://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}"

    return create_engine(
        url,
        pool_recycle=3600,
        pool_size=pool_size if pool_size else 1,
        pool_pre_ping=True,
        pool_use_lifo=True,
    )
    # if pool_size is None or pool_size==1:
    #     return create_engine(url, poolclass=NullPool, pool_pre_ping=True)
    # else:
    #     return create_engine(
    #         url, pool_recycle=3600, pool_size=pool_size, pool_pre_ping=True
    #     )


def has_column(conn, table_name, *args):
    """
    Checks whether the given table has the specified column(s).

    Parameters:
    conn (sqlalchemy.engine.base.Connection or Engine): The connection or engine to the database.
    table_name (str): The name of the table to inspect.
    *args (str): One or more column names to check for existence in the table.

    Returns:
    bool: True if all specified columns exist in the table, False otherwise.

    Example:
    >>> has_column(conn, 'users', 'id', 'name', 'email')
    True
    >>> has_column(conn, 'users', 'id', 'name', 'nonexistent_column')
    False
    """

    metadata = MetaData()
    try:
        table = Table(table_name, metadata, autoload_with=conn)
        columns_in_table = set(table.columns.keys())
        return all(column in columns_in_table for column in args)
    except NoSuchTableError:
        # The table does not exist
        return False


def tables_with_prefix(conn, prefix):
    # Define the escape character (e.g., '!')
    escape_char = "!"

    # Escape the underscore in the pattern using the escape character
    pattern = f"{prefix}{escape_char}_%"

    # Craft the SQL query with parameter placeholders
    query = text(
        f"""
            SELECT table_name
            FROM information_schema.tables
            WHERE table_name LIKE :pattern escape '{escape_char}'
            and table_name not like '%{escape_char}_impute' escape '{escape_char}'
            and table_name not like '%{escape_char}_part{escape_char}_%' escape '{escape_char}'
            AND table_type = 'BASE TABLE'
            AND table_schema NOT IN ('pg_catalog', 'information_schema')
        """
    )

    # Execute the query with bound parameters
    with conn.connect() as c:
        result = c.execute(
            query,
            {
                "pattern": pattern,
            },
        )
        # Fetch and return the table names
        return [row[0] for row in result.fetchall()]


def columns_with_prefix(conn, table, prefix):
    """
    Retrieve column names from a table where the column names start with a given prefix followed by an underscore.

    Parameters:
    - conn: A SQLAlchemy engine object.
    - table (str): The name of the table.
    - prefix (str): The prefix to search for.

    Returns:
    - List[str]: A list of column names matching the pattern.
    """
    # Define the escape character (e.g., '!')
    escape_char = "!"

    # Escape the underscore in the pattern using the escape character
    pattern = f"{prefix}{escape_char}_%"

    # Craft the SQL query with parameter placeholders
    query = text(
        f"""
        SELECT column_name
        FROM information_schema.columns
        WHERE table_name = :table_name
          AND column_name LIKE :pattern ESCAPE '{escape_char}'
    """
    )

    # Execute the query with bound parameters
    with conn.connect() as c:
        result = c.execute(
            query,
            {
                "table_name": table,
                "pattern": pattern,
            },
        )
        # Fetch and return the column names
        return [row[0] for row in result.fetchall()]


def set_autovacuum(alchemyEngine: Engine, enabled: bool, tables: List):
    autovacuum_value = "true" if enabled else "false"

    with alchemyEngine.connect() as conn:
        for table_name in tables:
            # Check if the table is partitioned
            is_partitioned_sql = text(f"""
                SELECT EXISTS (
                    SELECT 1
                    FROM pg_partitioned_table
                    WHERE partrelid = '{table_name}'::regclass
                )
            """)
            is_partitioned = conn.execute(is_partitioned_sql).scalar()

            if is_partitioned:
                # The table is partitioned; get all leaf partitions
                get_partitions_sql = text(f"""
                    WITH RECURSIVE partitions AS (
                        SELECT
                            inhrelid::regclass AS partition
                        FROM
                            pg_inherits
                        WHERE
                            inhparent = '{table_name}'::regclass
                        UNION ALL
                        SELECT
                            pg_inherits.inhrelid::regclass
                        FROM
                            pg_inherits
                            JOIN partitions ON pg_inherits.inhparent = partitions.partition::regclass
                    )
                    SELECT partition FROM partitions
                """)
                partitions = [
                    row[0] for row in conn.execute(get_partitions_sql).fetchall()
                ]
                # Set autovacuum on each partition
                for partition in partitions:
                    alter_sql = text(f"ALTER TABLE {partition} SET (autovacuum_enabled = {autovacuum_value})")
                    conn.execute(alter_sql)
            else:
                # The table is not partitioned; apply directly
                alter_sql = text(f"ALTER TABLE {table_name} SET (autovacuum_enabled = {autovacuum_value})")
                conn.execute(alter_sql)
