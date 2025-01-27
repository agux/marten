import random
import logging
import time
from dask.distributed import Client, LocalCluster, wait

total_workload = 2e4
n_workers = 16

def dummy_task(i):
    print(f"working on task #{i}")
    time.sleep(random.randint(1,5))
    return True

def main():
    cluster = LocalCluster(
        n_workers=n_workers,
        threads_per_worker=1,
        processes=True,
        silence_logs=logging.INFO,
    )

    client = Client(cluster)

    futures = []

    for i in range(int(total_workload)):
        futures.append(client.submit(dummy_task, i))
        if len(futures) > n_workers * 3:
            _, undone = wait(futures, return_when="FIRST_COMPLETED")
            futures = list(undone)

if __name__ == "__main__":
    main()
