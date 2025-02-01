import faulthandler

faulthandler.enable()

import random
import logging
import uuid
import time
import numpy as np
import pandas as pd

from datetime import datetime

# from dask.distributed import print

from dask.distributed import (
    Client,
    LocalCluster,
    worker_client,
    get_worker,
    wait,
    WorkerPlugin,
)

from marten.utils.trainer import (
    remove_singular_variables,
    get_accelerator_locks,
)

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def sim_model():
    worker = get_worker()
    model = worker.model
    config = model.baseline_params()
    config["h"] = 20
    config["max_steps"] = 10000
    config["accelerator"] = "auto"
    config["validate"] = True
    config["random_seed"] = 7
    config["locks"] = get_accelerator_locks(0, gpu_leases=2, timeout="20s")
    merged_df = pd.read_hdf("dummy_data.h5", "df")
    if not model.accept_missing_data():
        df_na = merged_df.iloc[:, 1:].isna()
        if df_na.any().any():
            merged_df, _ = remove_singular_variables(merged_df)
            merged_df, impute_df = model.impute(merged_df, **config)
            merged_df.dropna(axis=1, how="any", inplace=True)
            if impute_df is not None:
                impute_df.dropna(axis=1, how="all", inplace=True)

    metrics = model.train(merged_df, **config)

    return metrics


def tier2_task(i1, i2, sem, df):
    # worker = get_worker()
    if i2 % 10 == 0:
        with sem:
            print(f'using semaphore: {sem.max_leases}')

    print(
        f'{datetime.now().strftime("%Y-%m-%d %H:%M:%S")} worker#{get_worker().name} on tier2 task #{i1}:{i2}'
    )

    # start_time = time.perf_counter()
    # df = df.copy()
    # task_memory = random.randint(100 * 1024**2, 500 * 1024**2)

    # data = np.ones(task_memory, dtype=np.uint8)

    duration = random.uniform(5, 10)
    time.sleep(duration)
    # end_time = start_time + duration
    # result = 0
    # while time.perf_counter() < end_time:
    #     result += random.uniform(-1, 1)

    return None


def tier1_task(i1, p, num_tier2_tasks, sem, df):
    sem.max_leases
    futures = []
    start_time = datetime.now()
    i2 = 0
    # for i2 in range(int(num_tier2_tasks)):
    while (datetime.now() - start_time).seconds < 60 * 50:
        with worker_client() as client:
            futures.append(
                client.submit(
                    tier2_task,
                    i1,
                    i2,
                    sem,
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
