import os
import time
import random
import multiprocessing
from dask.distributed import WorkerPlugin, get_worker, LocalCluster, Client
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
        worker.logger = get_logger(self.logger_name, role='worker')


def init_client(name, worker=-1, threads=1):
    cluster = LocalCluster(
        n_workers=worker if worker > 0 else multiprocessing.cpu_count(),
        threads_per_worker=threads,
        processes=True,
        # memory_limit="2GB",
    )
    client = Client(cluster)
    client.register_plugin(LocalWorkerPlugin(name))
    client.forward_logging()
    get_logger(name).info("dask dashboard can be accessed at: %s", cluster.dashboard_link)

    return client


def get_result(future):
    try:
        r = future.result()
        return r
    except Exception as e:
        get_logger().exception(e)
        pass

def num_undone(futures):
    undone = 0
    for f in futures:
        if f.done():
            get_result(f)
            futures.remove(f)
        else:
            undone += 1
    return undone

def random_seconds(a, b, max):
    return min(float(max), round(random.uniform(float(a), float(b)), 3))

def await_futures(futures, until_all_completed=True):
    num = num_undone(futures)

    ##FIXME: this log is for debugging hanging-task issue only. remove them after fixed
    # try:
    #     worker = get_worker()
    #     worker.logger.info("#futures: %s #undone: %s", len(futures), num)
    # except ValueError as e:
    #     if "No worker found" in str(e):
    #         # possible that this is not called from a worker process. simply ignore
    #         pass
    #     else:
    #         raise e
    # ---------end of debug code----------------------

    if until_all_completed:
        while num > 0:
            time.sleep(random_seconds(2 ** (num - 1), 2**num, 256))
            num = num_undone(futures)
    elif num > multiprocessing.cpu_count():
        delta = num - multiprocessing.cpu_count()
        time.sleep(random_seconds(2 ** (delta - 1), 2**delta, 256))
