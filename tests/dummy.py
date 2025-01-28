import random
import logging
import uuid
import time

from datetime import datetime

from dask.distributed import (
    Client,
    LocalCluster,
    worker_client,
    get_worker,
    wait,
    WorkerPlugin,
)

logger = logging.getLogger(__name__)

def tier2_task(i1, i2):
    worker = get_worker()
    
    logger.error(
        f'{datetime.now().strftime("%Y-%m-%d %H:%M:%S")} worker#{get_worker().name} on tier2 task #{i1}:{i2}'
    )
    duration = random.uniform(1, 3)
    end_time = time.perf_counter() + duration
    result = 0
    while time.perf_counter() < end_time:
        result += random.uniform(-1, 1)
    return result


def tier1_task(i1, p, n_workers, num_tier2_tasks):
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
            if len(futures) > 500:
                _, undone = wait(futures, return_when="FIRST_COMPLETED")
                futures = list(undone)
        logger.error(
            f'{datetime.now().strftime("%Y-%m-%d %H:%M:%S")} worker#{get_worker().name} waiting ALL_COMPLETED'
        )
        wait(futures)
