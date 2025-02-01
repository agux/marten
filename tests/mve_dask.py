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

n_workers = 8
num_tier1_tasks = 10


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


def tier1_task(i1, p, locks):
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
                    priority=p,
                    key=f"tier2_task_{i1}-{uuid.uuid4().hex}",
                )
            )
            if len(futures) > 100:
                _, undone = wait(futures, return_when="FIRST_COMPLETED")
                futures = list(undone)
        i2 += 1
    print(
        f'{datetime.now().strftime("%Y-%m-%d %H:%M:%S")} worker#{get_worker().name} waiting ALL_COMPLETED on tier1: #{i1}'
    )
    with worker_client():
        wait(futures)


def tier2_task(i1, i2, locks):
    print(
        f'{datetime.now().strftime("%Y-%m-%d %H:%M:%S")} worker#{get_worker().name} on tier2 task #{i1}:{i2}'
    )

    duration = random.uniform(3, 8)
    time.sleep(duration)

    return None


def main():
    cluster = LocalCluster(
        n_workers=n_workers,
        threads_per_worker=1,
        processes=True,
    )
    client = Client(cluster)

    locks = {
        "lock1": Semaphore(max_leases=10, name="dummy_semaphore1"),
        "lock2": Semaphore(max_leases=10, name="dummy_semaphore2"),
    }

    futures = []
    for i1 in range(num_tier1_tasks):
        p = num_tier1_tasks - i1
        futures.append(
            client.submit(
                tier1_task,
                i1,
                p,
                locks,
            )
        )
        if len(futures) > 1:
            _, undone = wait(futures, return_when="FIRST_COMPLETED")
            futures = list(undone)

    wait(futures)
    print("all tasks completed.")


if __name__ == "__main__":
    main()
