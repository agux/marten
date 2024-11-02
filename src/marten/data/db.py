import pandas as pd
from sqlalchemy import (
    text,
)
from sqlalchemy.dialects.postgresql import insert

def update_on_conflict(table_def, conn, df: pd.DataFrame, primary_keys):
    """
    Insert new records, update existing records without nullifying columns not included in the dataframe
    """
    # Load the table metadata
    # table = sqlalchemy.Table(table, sqlalchemy.MetaData(), autoload_with=conn)
    # Create an insert statement from the DataFrame records
    insert_stmt = insert(table_def).values(df.to_dict(orient="records"))

    if hasattr(table_def, "__table__"):
        table_columns = table_def.__table__.columns
    else:
        table_columns = table_def.columns

    # Build a dictionary of column values to be updated, excluding primary keys and non-existent columns
    update_dict = {
        c.name: insert_stmt.excluded[c.name]
        for c in table_columns
        if c.name in df.columns and c.name not in primary_keys
    }
    # Construct the on_conflict_do_update statement
    on_conflict_stmt = insert_stmt.on_conflict_do_update(
        index_elements=primary_keys, set_=update_dict
    )
    # Execute the on_conflict_do_update statement
    conn.execute(on_conflict_stmt)


def ignore_on_conflict(table_def, conn, df, primary_keys):
    """
    Insert new records, ignore existing records
    """
    # table = sqlalchemy.Table(table, sqlalchemy.MetaData(), autoload_with=conn)
    insert_stmt = insert(table_def).values(df.to_dict(orient="records"))
    on_conflict_stmt = insert_stmt.on_conflict_do_nothing(index_elements=primary_keys)
    conn.execute(on_conflict_stmt)


def get_max_for_column(conn, symbol, table, col_for_max="date", non_null_col=None):
    query = f"SELECT max({col_for_max}) FROM {table} where 1=1"
    if non_null_col is not None:
        query += f" and {non_null_col} is not null"
    if symbol is not None:
        query += " and symbol = :symbol"
        result = conn.execute(text(query), {"symbol": symbol})
    else:
        result = conn.execute(text(query))
    return result.fetchone()[0]
