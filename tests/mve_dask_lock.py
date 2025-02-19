import os

os.environ["PYTHONUNBUFFERED"] = "1"

import time
import random
import threading
from datetime import datetime

from dask.distributed import (
    Client,
    LocalCluster,
    get_worker,
    wait,
    Lock,
)

# NOTE: adjust the number of workers as needed.
n_workers = 16

def release(lock):
    lock.release()

def dummy_task(i):
    lock = Lock("shared_lock")
    while not lock.acquire("3s"):
        time.sleep(1)
    threading.Timer(time.sleep(random.uniform(1, 3)), release, (lock)).start()
    time.sleep(random.uniform(2, 4))
    print(
        f'{datetime.now().strftime("%Y-%m-%d %H:%M:%S")} worker#{get_worker().name} acquired lock and completed task #{i}'
    )
    release(lock)
    return None


def main():
    cluster = LocalCluster(
        n_workers=n_workers,
        threads_per_worker=1,
        processes=True,
    )
    client = Client(cluster)

    futures = []
    i = 0
    while True:
        futures.append(
            client.submit(
                dummy_task,
                i,
            )
        )
        if len(futures) > n_workers * 2:
            _, undone = wait(futures, return_when="FIRST_COMPLETED")
            futures = list(undone)
        i += 1


if __name__ == "__main__":
    main()
