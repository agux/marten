import faulthandler

faulthandler.enable()

import random
import logging
import uuid
import time
from datetime import datetime
from dask.distributed import Client, LocalCluster, worker_client, get_worker, wait

n_workers = 16
num_tier1_tasks = 10
num_tier2_tasks = 2e5

logger = logging.getLogger(__name__)


def tier2_task(i1, i2):
    logger.error(
        f'{datetime.now().strftime("%Y-%m-%d %H:%M:%S")} worker#{get_worker().name} on tier2 task #{i1}:{i2}'
    )
    duration = random.uniform(5, 20)
    end_time = time.perf_counter() + duration
    result = 0
    while time.perf_counter() < end_time:
        result += random.uniform(-1, 1)
    return result


def tier1_task(i1, p):
    futures = []
    with worker_client() as client:
        for i2 in range(int(num_tier2_tasks)):
            futures.append(
                client.submit(
                    tier2_task,
                    i1,
                    i2,
                    priority=p,
                    key=f"tier2_task_{i1}-{uuid.uuid4().hex}",
                )
            )
            if len(futures) > n_workers:
                _, undone = wait(futures, return_when="FIRST_COMPLETED")
                futures = list(undone)
        logger.error(
            f'{datetime.now().strftime("%Y-%m-%d %H:%M:%S")} worker#{get_worker().name} waiting ALL_COMPLETED'
        )
        wait(futures)


def main():
    cluster = LocalCluster(
        n_workers=n_workers//2,
        threads_per_worker=1,
        processes=True,
        silence_logs=logging.INFO,
    )
    client = Client(cluster)
    futures = []
    scaled = False
    for i1 in range(num_tier1_tasks):
        p = num_tier1_tasks - i1
        futures.append(
            client.submit(
                tier1_task, i1, p, priority=p, key=f"tier1_task_{i1}-{uuid.uuid4().hex}"
            )
        )
        if len(futures) > 1:
            if not scaled:
                cluster.scale(n_workers)
                scaled = True
            _, undone = wait(futures, return_when="FIRST_COMPLETED")
            futures = list(undone)

    wait(futures)
    logger.info("all tasks completed.")


if __name__ == "__main__":
    main()
