import os

from dotenv import load_dotenv
from sqlalchemy.exc import NoSuchTableError

from sqlalchemy import (
    create_engine,
    Engine,
    text,
    MetaData, Table
)


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
