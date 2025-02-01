import os
os.environ["PYTHONUNBUFFERED"] = "1"

import time
import random
from datetime import datetime

from dask.distributed import (
    Client,
    LocalCluster,
    get_worker,
    wait,
    Semaphore,
)

n_workers = 16

def dummy_task(i, locks):
    print(
        f'{datetime.now().strftime("%Y-%m-%d %H:%M:%S")} worker#{get_worker().name} on tier2 task #{i}'
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

    locks = {}
    for i in range (10):
        locks[f"lock{i}"] = Semaphore(max_leases=10, name=f"dummy_semaphore{i}")

    futures = []
    i = 0
    while True:
        futures.append(
            client.submit(
                dummy_task,
                i,
                locks,
            )
        )
        if len(futures) > n_workers * 2:
            _, undone = wait(futures, return_when="FIRST_COMPLETED")
            futures = list(undone)
        i += 1

if __name__ == "__main__":
    main()
