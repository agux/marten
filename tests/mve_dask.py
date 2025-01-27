import faulthandler

faulthandler.enable()

import random
import logging
import time
from datetime import datetime
from dask.distributed import Client, LocalCluster, worker_client, get_worker, wait

total_workload = 2e5
n_workers = 16

logger = logging.getLogger(__name__)


def tier2_task(i):
    logger.error(
        f'{datetime.now().strftime("%Y-%m-%d %H:%M:%S")} worker#{get_worker().name} on tier2 task #{i}'
    )
    duration = random.randint(5, 10)
    end_time = time.perf_counter() + duration
    result = 0
    while time.perf_counter() < end_time:
        result += random.uniform(-1, 1)
    return result


def tier1_task(p):
    futures = []
    with worker_client() as client:
        for i in range(int(total_workload)):
            futures.append(client.submit(tier2_task, i, priority=p))
            if len(futures) > n_workers:
                _, undone = wait(futures, return_when="FIRST_COMPLETED")
                futures = list(undone)
        logger.error(
            f'{datetime.now().strftime("%Y-%m-%d %H:%M:%S")} worker#{get_worker().name} waiting ALL_COMPLETED'
        )
        wait(futures)


def main():
    cluster = LocalCluster(
        n_workers=n_workers,
        threads_per_worker=1,
        processes=True,
        silence_logs=logging.INFO,
    )

    client = Client(cluster)

    futures = []

    for i in range(10):
        p = 10 - i
        futures.append(client.submit(tier1_task, p, priority=p))
        if len(futures) > 1:
            _, undone = wait(futures, return_when="FIRST_COMPLETED")
            futures = list(undone)

    wait(futures)
    logger.info("all tasks completed.")


if __name__ == "__main__":
    main()
