import os

from dotenv import load_dotenv

from sqlalchemy.pool import NullPool

from sqlalchemy import (
    create_engine,
    Engine,
)

def get_database_engine() -> Engine:
    load_dotenv()  # take environment variables from .env.

    DB_USER = os.getenv("DB_USER")
    DB_PASSWORD = os.getenv("DB_PASSWORD")
    DB_HOST = os.getenv("DB_HOST")
    DB_PORT = os.getenv("DB_PORT")
    DB_NAME = os.getenv("DB_NAME")

    alchemyEngine = create_engine(
        f"postgresql+psycopg2://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}",
        pool_recycle=3600,
        # pool_size=1,
        poolclass=NullPool,
    )

    return alchemyEngine
