import random
import logging
import uuid
import time
import numpy as np
import pandas as pd

from datetime import datetime

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

def tier2_task(i1, i2):
    worker = get_worker()

    logger.error(
        f'{datetime.now().strftime("%Y-%m-%d %H:%M:%S")} worker#{get_worker().name} on tier2 task #{i1}:{i2}'
    )
    # duration = random.uniform(5, 15)
    # end_time = time.perf_counter() + duration
    # result = 0
    # while time.perf_counter() < end_time:
    #     result += random.uniform(-1, 1)
    # return result
    model = worker.model
    config = model.baseline_params()
    config["h"] = 20
    config["max_steps"] = 10000
    config["accelerator"] = "auto"
    config["validate"] = True
    config["random_seed"] = 7
    config["locks"] = get_accelerator_locks(0, gpu_leases=2, timeout="20s")
    merged_df = pd.read_hdf(
        "159985-bond_metrics_em_view-us_yield_5y_change_rate-bond-unimputed.h5", 'df'
    )
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


def tier1_task(i1, p, n_workers, num_tier2_tasks):
    futures = []
    for i2 in range(int(num_tier2_tasks)):
        with worker_client() as client:
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
    with worker_client():
        wait(futures)
