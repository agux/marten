import os
import sys
import time

import multiprocessing

from dotenv import load_dotenv

import yappi

# import pytz
# import pandas as pd
# from IPython.display import display, HTML

from marten.utils.logger import get_logger
from marten.utils.database import get_database_engine
from marten.utils.worker import await_futures, init_client, scale_cluster_and_wait
from marten.data.const import us_index_list, cn_index_types
from marten.data.ta import calc_ta
from marten.data.worker_func import (
    etf_spot,
    etf_perf,
    etf_list,
    bond_ir,
    update_etf_metrics,
    get_cn_index_list,
    cn_index_daily,
    hk_index_spot,
    hk_index_daily,
    get_us_indices,
    bond_spot,
    bond_daily_hs,
    stock_zh_spot,
    stock_zh_daily_hist,
    rmb_exchange_rates,
    sge_spot,
    sge_spot_daily_hist,
    cn_bond_index,
    get_stock_bond_ratio_index,
    get_fund_dividend_events,
    interbank_rate,
    option_qvix,
)

# module_path = os.getenv("LOCAL_AKSHARE_DEV_MODULE")
# if module_path is not None and module_path not in sys.path:
# sys.path.insert(0, module_path)
import akshare as ak  # noqa: E402

logger = get_logger(__name__)
alchemyEngine = None
client = None
prog_args = None
futures = []


def init():
    global alchemyEngine, client, logger, prog_args
    logger.info("Using akshare version: %s", ak.__version__)

    load_dotenv()  # take environment variables from .env.

    DB_USER = os.getenv("DB_USER")
    DB_PASSWORD = os.getenv("DB_PASSWORD")
    DB_HOST = os.getenv("DB_HOST")
    DB_PORT = os.getenv("DB_PORT")
    DB_NAME = os.getenv("DB_NAME")

    workers = int(
        prog_args.worker
        if os.getenv("ETL_WORKERS") is None
        else os.getenv("ETL_WORKERS")
    )
    threads = int(
        prog_args.threads
        if os.getenv("ETL_THREADS") is None
        else os.getenv("ETL_THREADS")
    )
    port = int(
        prog_args.dashboard_port
        if os.getenv("ETL_DASHBOARD_PORT") is None
        else os.getenv("ETL_DASHBOARD_PORT")
    )

    db_url = (
        f"postgresql+psycopg2://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}"
    )
    alchemyEngine = get_database_engine(db_url, pool_size=threads)

    client = init_client(
        __name__,
        workers,
        threads,
        port,
    )
    # client.cluster.scale(workers if workers > 0 else multiprocessing.cpu_count())

def runnable(task):
    name = task.__name__
    if (prog_args.include is None or name in prog_args.include) and (
        prog_args.exclude is None or name not in prog_args.exclude
    ):
        return True
    elif (
        prog_args.include is not None
        and prog_args.exclude is not None
        and name in prog_args.include
        and name in prog_args.exclude
    ):
        raise ValueError(
            f"Conflicting options: {name} is given in both --include and --exclude arguments."
        )
    else:
        return False


def run(task, *args, **kwargs):
    global client, futures
    if runnable(task):
        future = client.submit(task, *args, **kwargs)
        futures.append(future)
        return future
    else:
        return None


def main(_args):
    global client, logger, cn_index_types, us_index_list, prog_args, futures
    prog_args = _args
    t_start = time.time()

    init()

    ## collect and await all task futures

    run(get_fund_dividend_events, priority=1)

    future_etf_spot = run(etf_spot)
    run(etf_perf)

    future_etf_list = run(etf_list, future_etf_spot, priority=1)
    future_bond_ir = run(bond_ir, priority=1)

    run(update_etf_metrics, future_etf_list, future_bond_ir)
    run(get_stock_bond_ratio_index, priority=1)

    future_cn_index_list = run(get_cn_index_list, cn_index_types)
    run(cn_index_daily, future_cn_index_list)

    future_hk_index_list = run(hk_index_spot)
    run(hk_index_daily, future_hk_index_list)

    run(get_us_indices, us_index_list)

    future_bond_spot = run(bond_spot)
    run(bond_daily_hs, future_bond_spot, prog_args.threads)

    future_stock_zh_spot = run(stock_zh_spot)
    run(stock_zh_daily_hist, future_stock_zh_spot, prog_args.threads)

    run(rmb_exchange_rates)

    future_sge_spot = run(sge_spot)
    run(sge_spot_daily_hist, future_sge_spot)

    run(cn_bond_index)

    run(interbank_rate)

    run(option_qvix)

    await_futures(futures)
    logger.info("ETL Time taken: %s seconds", time.time() - t_start)

    if runnable(calc_ta):
        n_workers = len(client.scheduler_info()["workers"])
        scale_cluster_and_wait(client, max(1, n_workers / 2))
        # client.cluster.scale(max(1, n_workers / 2))
        run(calc_ta)
        await_futures(futures)

    logger.info("Total time taken: %s seconds", time.time() - t_start)

    if client is not None:
        # Remember to close the client if your program is done with all computations
        client.close()


def run_main_with_profiling(args):
    yappi.set_clock_type("wall")  # Use wall time (real time) instead of CPU time
    yappi.start()

    main(args)

    yappi.stop()

    func_stats = yappi.get_func_stats()
    # Assuming "ttot" is the correct parameter for sorting by total execution time including sub-functions
    func_stats.sort("ttot", "desc")
    with open("func_stats_sorted.txt", "w") as file:
        # Redirect stdout to the file
        sys.stdout = file
        # Print all function stats to the file
        func_stats.print_all()
        # Reset stdout to its original value
        sys.stdout = sys.__stdout__

    # thread_stats = yappi.get_thread_stats()
    # thread_stats.save("thread_stats.prof", type="pstat")


if __name__ == "__main__":
    try:
        # profiler = cProfile.Profile()
        # profiler.enable()
        from types import SimpleNamespace

        args = SimpleNamespace(
            worker=multiprocessing.cpu_count(),
            threads=3,
        )

        main(args)

        # profiler.disable()
        # stats = pstats.Stats(profiler).sort_stats("cumtime")
        # stats.print_stats()
    except Exception as e:
        logger.exception("main process terminated")
