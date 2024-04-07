import os

from dask.distributed import Client, as_completed, WorkerPlugin, get_worker
from marten.utils.database import get_database_engine
from marten.utils.logger import get_logger
from dotenv import load_dotenv

from neuralprophet import set_log_level


class LocalWorkerPlugin(WorkerPlugin):
    def __init__(self, logger_name):
        self.logger_name = logger_name

    def setup(self, worker):
        set_log_level("ERROR")

        load_dotenv()  # take environment variables from .env.
        DB_USER = os.getenv("DB_USER")
        DB_PASSWORD = os.getenv("DB_PASSWORD")
        DB_HOST = os.getenv("DB_HOST")
        DB_PORT = os.getenv("DB_PORT")
        DB_NAME = os.getenv("DB_NAME")

        db_url = (
            f"postgresql+psycopg2://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}"
        )

        worker.alchemyEngine = get_database_engine(db_url)
        worker.logger = get_logger(self.logger_name)
