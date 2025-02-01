import faulthandler
faulthandler.enable()

import os
# OPENBLAS_NUM_THREADS = 1
# os.environ["OPENBLAS_NUM_THREADS"] = "1"
# os.environ["MKL_NUM_THREADS"] = "1"
# os.environ["OMP_NUM_THREADS"] = "1"
os.environ["PYTHONUNBUFFERED"] = "1"

import time

# import ray

from marten.models.base_model import BaseModel
from marten.utils.logger import get_logger
from marten.utils.worker import init_client, init_ray
from marten.utils.database import get_database_engine
from marten.utils.trainer import select_device
from marten.models.worker_func import (
    predict_best,
    # predict_adhoc,
    covars_and_search,
    covars_and_search_dummy,
    ensemble_topk_prediction,
)

from types import SimpleNamespace

from dotenv import load_dotenv

logger = get_logger(__name__)
# client = None
alchemyEngine = None
model: BaseModel = None


def init(args):
    global client, alchemyEngine, model
    match args.asset_type:
        case "stock":
            args.symbol_table = "stock_zh_a_hist_em_view"
        case "index":
            args.symbol_table = "index_daily_em_view"
        case "fund":
            args.symbol_table = "fund_etf_daily_em_view"
        case _:
            args.symbol_table = "unspecified"
    
    load_dotenv()  # take environment variables from .env.
    print_sys_info()

    client = init_client(
        __name__,
        args.max_worker,
        threads=args.threads,
        dashboard_port=args.dashboard_port,
        args=args,
    )
    # init_ray()

    if alchemyEngine is None:
        DB_USER = os.getenv("DB_USER")
        DB_PASSWORD = os.getenv("DB_PASSWORD")
        DB_HOST = os.getenv("DB_HOST")
        DB_PORT = os.getenv("DB_PORT")
        DB_NAME = os.getenv("DB_NAME")

        db_url = f"postgresql+psycopg2://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}"
        alchemyEngine = get_database_engine(db_url)

    match args.model.lower():
        case "timemixer":
            from marten.models.time_mixer import TimeMixerModel

            model = TimeMixerModel()
        case "tsmixerx":
            from marten.models.nf_tstimerx import TSMixerxModel
            model = TSMixerxModel()
        case _:
            model = None

def print_sys_info():
    import torch
    logger.info(torch.__config__.show())

def main(args):
    global client, alchemyEngine, logger, model

    t_start = time.time()
    init(args)

    for symbol in args.symbols:
        if args.adhoc:
            # future = client.submit(predict_adhoc, symbol, args)
            hps_id, cutoff_date, ranked_features, df = covars_and_search_dummy(
                model, client, symbol, alchemyEngine, logger, args
            )
            logger.info("Starting adhoc prediction")
            t3_start = time.time()

            ensemble_topk_prediction(
                client,
                symbol,
                args.random_seed,
                args.future_steps,
                args.topk,
                hps_id,
                cutoff_date,
                ranked_features,
                df,
                alchemyEngine,
                logger,
                args,
            )
            logger.info(
                "%s prediction completed. Time taken: %s seconds",
                symbol,
                round(time.time() - t3_start, 3),
            )
        else:
            predict_best(
                symbol,
                args.early_stopping,
                args.timestep_limit,
                args.epochs,
                args.random_seed,
                args.future_steps,
                args.topk,
                select_device(
                    args.accelerator,
                    getattr(args, "gpu_util_threshold", None),
                    getattr(args, "gpu_ram_threshold", None),
                ),
            )

    if client is not None:
        client.close()

    logger.info("Time taken: %s seconds", time.time() - t_start)


if __name__ == "__main__":
    try:
        args = SimpleNamespace(
            adhoc=True,
            model="TimeMixer",
            min_worker=2,
            max_worker=4,
            early_stopping=True,
            scheduler_port=8999,
            timestep_limit=-1,
            mini_itr=3,
            max_itr=3,
            batch_size=50,
            domain_size=500000,
            random_seed=7,
            epochs=1000,
            future_steps=20,
            max_covars=500,
            # symbols=["511220", "513800", "930955"],
            symbols=["399673"],
        )

        main(args)

    except Exception as e:
        logger.exception("main process terminated")
