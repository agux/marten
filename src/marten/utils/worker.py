import os
import time
import multiprocessing
from dask.distributed import WorkerPlugin
from marten.utils.database import get_database_engine
from marten.utils.logger import get_logger
from dotenv import load_dotenv


class LocalWorkerPlugin(WorkerPlugin):
    def __init__(self, logger_name):
        self.logger_name = logger_name

    def setup(self, worker):
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


def num_undone(futures):
    undone = 0
    for f in futures:
        # iterate over `futures`. call `f.done()` to check if it's done.
        # if it's done (True), call `del f` and then remove it from the `futures` list.
        # if False, increment `undone`.
        if f.done():
            f.release()
            futures.remove(f)
        else:
            undone += 1
    return undone


def await_futures(futures, until_all_completed=True):
    num = num_undone(futures)
    if until_all_completed:
        while num > 0:
            time.sleep(2**num)
            num = num_undone(futures)
    elif num > multiprocessing.cpu_count():
        time.sleep(2 ** (num - multiprocessing.cpu_count()))