import faulthandler
faulthandler.enable()

import random
import logging
import time
from datetime import datetime
from dask.distributed import Client, LocalCluster, wait

total_workload = 5e4
n_workers = 16

logger = logging.getLogger(__name__)

def gil_holding_task(i):
    logger.error(f"{datetime.now().strftime("%Y-%m-%d %H:%M:%S")} working on task #{i}")
    duration = random.randint(5, 10)
    end_time = time.perf_counter() + duration
    result = 0
    while time.perf_counter() < end_time:
        result += 1
    return result


def main():
    cluster = LocalCluster(
        # n_workers=1,
        n_workers=n_workers,
        threads_per_worker=1,
        processes=True,
        silence_logs=logging.INFO,
    )

    client = Client(cluster)

    futures = []
    # upscaled = False

    for i in range(int(total_workload)):
        futures.append(client.submit(gil_holding_task, i))
        if len(futures) > n_workers:
        # if len(futures) > 1:
            # if not upscaled:
            #     cluster.scale(n_workers)
            #     upscaled = True
            _, undone = wait(futures, return_when="FIRST_COMPLETED")
            futures = list(undone)


if __name__ == "__main__":
    main()
