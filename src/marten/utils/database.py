import os

from dotenv import load_dotenv

from sqlalchemy.pool import NullPool

from sqlalchemy import (
    create_engine,
    Engine,
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

    if pool_size is None or pool_size==1:
        return create_engine(
            url,
            poolclass=NullPool,
        )
    else:
        return create_engine(
            url,
            pool_recycle=3600,
            pool_size=pool_size,
        )
