import os
import time
import random
import multiprocessing
import dask
import dask.config
from dask.distributed import WorkerPlugin, get_worker, LocalCluster, Client
from marten.utils.database import get_database_engine
from marten.utils.logger import get_logger
from dotenv import load_dotenv

from datetime import datetime, timedelta

from pprint import pformat


class LocalWorkerPlugin(WorkerPlugin):
    def __init__(self, logger_name, args):
        self.logger_name = logger_name
        self.args = args

    def setup(self, worker):
        load_dotenv()  # take environment variables from .env.
        DB_USER = os.getenv("DB_USER")
        DB_PASSWORD = os.getenv("DB_PASSWORD")
        DB_HOST = os.getenv("DB_HOST")
        DB_PORT = os.getenv("DB_PORT")
        DB_NAME = os.getenv("DB_NAME")

        db_url = f"postgresql+psycopg2://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}"

        worker.alchemyEngine = get_database_engine(db_url)
        worker.logger = get_logger(self.logger_name, role="worker")
        worker.args = self.args


def init_client(name, max_worker=-1, threads=1, dashboard_port=None, args=None):
    dask.config.set({"distributed.scheduler.worker-ttl": "20 minutes"})

    cluster = LocalCluster(
        host="0.0.0.0",
        scheduler_port=getattr(args, "scheduler_port", 0),
        n_workers=1,
        threads_per_worker=threads,
        processes=True,
        dashboard_address=":8787" if dashboard_port is None else f":{dashboard_port}",
        # memory_limit="2GB",
        memory_limit=0,  # no limit
    )
    cluster.adapt(minimum=1, maximum=multiprocessing.cpu_count() if max_worker <= 0 else max_worker)
    client = Client(cluster)
    client.register_plugin(LocalWorkerPlugin(name, args))
    client.forward_logging()
    get_logger(name).info(
        "dask dashboard can be accessed at: %s", cluster.dashboard_link
    )
    get_logger(name).info("dask scheduler address: %s", cluster.scheduler_address)

    return client


def get_result(future):
    try:
        r = future.result()
        return r
    except Exception as e:
        get_logger().exception(e)
        pass


def get_results(futures):
    for f in futures:
        get_logger().debug("getting result for %s", _get_future_details(f))
        get_result(f)


def num_undone(futures, shared_vars):
    undone = 0
    if isinstance(futures, list):
        for f in futures:
            if f.done():
                get_result(f)
                futures.remove(f)
            else:
                undone += 1
    elif isinstance(futures, dict):
        for k in list(futures.keys()):
            f = futures[k]
            if f.done():
                get_result(f)
                if k in shared_vars:
                    shared_vars[k].delete()
                    shared_vars.pop(k)
                futures.pop(k)
            else:
                undone += 1
    return undone


def random_seconds(a, b, max):
    return min(float(max), round(random.uniform(float(a), float(b)), 3))


def handle_task_timeout(futures, task_timeout, shared_vars):
    for symbol in list(shared_vars.keys()):  # Create a list of keys to iterate over
        try:
            var = shared_vars[symbol]  # Access the variable using the key
            st_dict = var.get("200ms")
            if not "start_time" in st_dict:
                # the task has not been started by the worker process yet
                continue
            if datetime.now() >= st_dict["start_time"] + timedelta(
                seconds=task_timeout
            ):
                ## the task has timed out. if the future is not finished yet, cancel it.
                if symbol in futures:
                    future = futures[symbol]
                    if not future.done():
                        future.cancel()
                        futures.pop(symbol)
                var.delete()
                shared_vars.pop(
                    symbol
                )  # Safe to pop because we're not iterating over shared_vars directly
        except TimeoutError as e:
            pass


def _get_future_details(future):
    details = {
        "Key": future.key,
        "Status": future.status,
        "Function": getattr(future, "function", "N/A"),
        "Arguments": getattr(future, "args", "N/A"),
        "Keywords": getattr(future, "kwargs", "N/A"),
    }
    if future.status == "error":
        details["Exception"] = future.exception()
        details["Traceback"] = future.traceback()
    return details


def log_futures(futures):
    for f in futures:
        get_logger().debug(pformat(_get_future_details(f)))


def await_futures(
    futures,
    until_all_completed=True,
    task_timeout=None,
    shared_vars=None,
    multiplier=1,
    hard_wait=False,
):
    num = num_undone(futures, shared_vars)
    get_logger().debug("undone futures: %s", num)
    if until_all_completed:
        if task_timeout is not None and shared_vars is not None:
            while num is not None and num > 0:
                time.sleep(random_seconds(2 ** (num - 1), 2**num, 128))
                num = handle_task_timeout(futures, task_timeout, shared_vars)
        else:
            get_logger().debug("waiting until all futures complete: %s", num)
            while num > 0:
                time.sleep(random_seconds(2 ** (num - 1), 2**num, 128))
                if hard_wait:
                    get_results(futures)
                num = num_undone(futures, shared_vars)
                get_logger().debug("undone futures: %s", num)
                log_futures(futures)
    elif num > multiprocessing.cpu_count() * multiplier:
        delta = num - multiprocessing.cpu_count() * multiplier
        time.sleep(random_seconds(2 ** (delta - 1), 2**delta, 128))

        if task_timeout is not None and shared_vars is not None:
            handle_task_timeout(futures, task_timeout, shared_vars)
