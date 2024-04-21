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
from marten.utils.worker import await_futures, init_client
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
)

# module_path = os.getenv("LOCAL_AKSHARE_DEV_MODULE")
# if module_path is not None and module_path not in sys.path:
# sys.path.insert(0, module_path)
import akshare as ak  # noqa: E402

cn_index_types = [
    ("上证系列指数", "sh"),
    ("深证系列指数", "sz"),
    # ("指数成份", ""),
    ("中证系列指数", "csi"),
]

us_index_list = [".IXIC", ".DJI", ".INX", ".NDX"]

logger = get_logger(__name__)
alchemyEngine = None
client = None


def init(args):
    global alchemyEngine, client, logger
    logger.info("Using akshare version: %s", ak.__version__)

    load_dotenv()  # take environment variables from .env.

    DB_USER = os.getenv("DB_USER")
    DB_PASSWORD = os.getenv("DB_PASSWORD")
    DB_HOST = os.getenv("DB_HOST")
    DB_PORT = os.getenv("DB_PORT")
    DB_NAME = os.getenv("DB_NAME")

    db_url = (
        f"postgresql+psycopg2://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}"
    )
    alchemyEngine = get_database_engine(db_url, pool_size=1)

    client = init_client(__name__,args.worker,args.threads,args.dashboard_port)


def main(args):
    global client, logger, cn_index_types, us_index_list
    t_start = time.time()

    init(args)

    ## collect and await all task futures
    futures = []
    futures.append(client.submit(etf_spot))
    futures.append(client.submit(etf_perf))

    future_etf_list = client.submit(etf_list)
    future_bond_ir = client.submit(bond_ir)

    futures.append(client.submit(update_etf_metrics, future_etf_list, future_bond_ir))

    future_cn_index_list = client.submit(get_cn_index_list, cn_index_types)
    futures.append(client.submit(cn_index_daily, future_cn_index_list))

    future_hk_index_list = client.submit(hk_index_spot)
    futures.append(client.submit(hk_index_daily, future_hk_index_list))

    futures.append(client.submit(get_us_indices, us_index_list))

    future_bond_spot = client.submit(bond_spot)
    futures.append(client.submit(bond_daily_hs, future_bond_spot))

    future_stock_zh_spot = client.submit(stock_zh_spot)
    futures.append(client.submit(stock_zh_daily_hist, future_stock_zh_spot))

    futures.append(client.submit(rmb_exchange_rates))

    future_sge_spot = client.submit(sge_spot)
    futures.append(client.submit(sge_spot_daily_hist, future_sge_spot))

    futures.extend(
        [
            future_etf_list,
            future_bond_ir,
            future_cn_index_list,
            future_bond_spot,
            future_stock_zh_spot,
            future_sge_spot,
        ]
    )

    await_futures(futures)
    logger.info("Time taken: %s seconds", time.time() - t_start)

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
