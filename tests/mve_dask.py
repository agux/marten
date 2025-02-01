import faulthandler

faulthandler.enable()

import os

os.environ["PYTHONUNBUFFERED"] = "1"

import uuid
import time
import random
import logging
import pandas as pd
import numpy as np
from datetime import datetime
from types import SimpleNamespace

import dask
from dask.distributed import (
    Client,
    LocalCluster,
    worker_client,
    get_worker,
    wait,
    WorkerPlugin,
    Semaphore,
)

from marten.utils.worker import init_client

n_workers = 64
num_tier1_tasks = 10
num_tier2_tasks = 2e4

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
cluster = None
client = None


class LocalWorkerPlugin(WorkerPlugin):
    def __init__(self):
        self.prop = None

    def setup(self, worker):
        worker.logger = logging.getLogger()


def scale():
    global cluster, n_workers
    # cluster.scale(1)
    # time.sleep(10)
    cluster.scale(n_workers)


def make_df():
    start_date = "2000-01-01"
    end_date = "2025-02-01"

    date_range = pd.date_range(start=start_date, end=end_date, freq="B")
    np.random.seed(0)

    val1 = np.random.randn(len(date_range))
    val2 = np.random.rand(len(date_range)) * 100

    return pd.DataFrame({"Date": date_range, "val1": val1, "val2": val2})


def tier1_task(i1, p, num_tier2_tasks, locks, df):
    futures = []
    start_time = datetime.now()
    i2 = 0
    while (datetime.now() - start_time).seconds < 60 * 50:
        with worker_client() as client:
            futures.append(
                client.submit(
                    tier2_task,
                    i1,
                    i2,
                    locks,
                    df,
                    priority=p,
                    key=f"tier2_task_{i1}-{uuid.uuid4().hex}",
                )
            )
            if len(futures) > 500:
                _, undone = wait(futures, return_when="FIRST_COMPLETED")
                futures = list(undone)
        i2 += 1
    print(
        f'{datetime.now().strftime("%Y-%m-%d %H:%M:%S")} worker#{get_worker().name} waiting ALL_COMPLETED on tier1: #{i1}'
    )
    with worker_client():
        wait(futures)


def tier2_task(i1, i2, locks, df):
    print(
        f'{datetime.now().strftime("%Y-%m-%d %H:%M:%S")} worker#{get_worker().name} on tier2 task #{i1}:{i2}'
    )

    duration = random.uniform(5, 10)
    time.sleep(duration)

    return None


def main():
    global client, cluster

    # logger.error(
    #     "distributed.admin.event-loop: %s",
    #     dask.config.get("distributed.admin.event-loop"),
    # )

    # dask.config.set(
    #     {
    #         "distributed.worker.memory.terminate": False,
    #         "distributed.worker.resources.POWER": 2,
    #         "distributed.worker.multiprocessing-method": "spawn",
    #         "distributed.deploy.lost-worker-timeout": "2 hours",
    #         "distributed.scheduler.locks.lease-timeout": "15 minutes",
    #         "distributed.comm.retry.count": 10,
    #         "distributed.admin.log-length": 0,
    #         "distributed.admin.low-level-log-length": 0,
    #         "distributed.admin.system-monitor.log-length": 0,
    #     }
    # )
    # cluster = LocalCluster(
    #     n_workers=n_workers,
    #     threads_per_worker=1,
    #     scheduler_port=1234,
    #     dashboard_address=":8787",
    #     processes=True,
    #     silence_logs=logging.INFO,
    #     memory_limit=0,
    # )
    # client = Client(cluster, direct_to_workers=True, connection_limit=4096)
    # client.register_plugin(LocalWorkerPlugin())
    # client.forward_logging()
    client = init_client(
        "mve",
        max_worker=n_workers,
        args=SimpleNamespace(
            model="tsmixerx",
            dask_log=True,
            scheduler_port=8999,
            min_worker=4,
            max_worker=32,
        ),
    )

    # dask.config.set({"distributed.scheduler.locks.lease-timeout": "500s"})
    dask.config.set({"distributed.scheduler.locks.lease-timeout": "20s"})
    locks = {
        "lock1": Semaphore(max_leases=2, name="dummy_semaphore1"),
        "lock2": Semaphore(max_leases=1, name="dummy_semaphore2"),
    }

    df = make_df()

    client.submit(tier2_task, 0, 0, locks, df).result()

    futures = []
    for i1 in range(num_tier1_tasks):
        p = num_tier1_tasks - i1
        futures.append(
            client.submit(
                tier1_task,
                i1,
                p,
                num_tier2_tasks,
                locks,
                df,
                priority=p,
                key=f"tier1_task_{i1}-{uuid.uuid4().hex}",
            )
        )
        if len(futures) > 1:
            _, undone = wait(futures, return_when="FIRST_COMPLETED")
            futures = list(undone)

    wait(futures)
    print("all tasks completed.")


if __name__ == "__main__":
    main()
