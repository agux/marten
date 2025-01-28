import faulthandler

faulthandler.enable()


import threading
import uuid
import time
import logging
from datetime import datetime

import dask
from dask.distributed import (
    Client,
    LocalCluster,
    worker_client,
    get_worker,
    wait,
    WorkerPlugin,
)

from dummy import tier1_task, tier2_task

n_workers = 64
num_tier1_tasks = 10
num_tier2_tasks = 2e4

logger = logging.getLogger(__name__)
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


def main():
    global client, cluster

    logger.error(
        "distributed.admin.event-loop: %s",
        dask.config.get("distributed.admin.event-loop"),
    )

    dask.config.set(
        {
            "distributed.worker.memory.terminate": False,
            "distributed.worker.resources.POWER": 2,
            "distributed.worker.multiprocessing-method": "spawn",
            "distributed.deploy.lost-worker-timeout": "2 hours",
            "distributed.scheduler.locks.lease-timeout": "15 minutes",
            "distributed.comm.retry.count": 10,
            "distributed.admin.log-length": 0,
            "distributed.admin.low-level-log-length": 0,
            "distributed.admin.system-monitor.log-length": 0,
        }
    )
    cluster = LocalCluster(
        n_workers=n_workers,
        threads_per_worker=1,
        scheduler_port=1234,
        dashboard_address=":8787",
        processes=True,
        silence_logs=logging.INFO,
        memory_limit=0,
    )
    client = Client(cluster, direct_to_workers=True, connection_limit=4096)
    client.register_plugin(LocalWorkerPlugin())
    client.forward_logging()

    client.submit(tier2_task, 0, 0).result()

    # scale()

    futures = []
    for i1 in range(num_tier1_tasks):
        p = num_tier1_tasks - i1
        futures.append(
            client.submit(
                tier1_task,
                i1,
                p,
                n_workers,
                num_tier2_tasks,
                priority=p,
                key=f"tier1_task_{i1}-{uuid.uuid4().hex}",
            )
        )
        if len(futures) > 1:
            _, undone = wait(futures, return_when="FIRST_COMPLETED")
            futures = list(undone)

    wait(futures)
    logger.info("all tasks completed.")


if __name__ == "__main__":
    main()
