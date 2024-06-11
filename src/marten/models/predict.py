import time
import os
OPENBLAS_NUM_THREADS = 1
os.environ["OPENBLAS_NUM_THREADS"] = f"{OPENBLAS_NUM_THREADS}"


from marten.utils.logger import get_logger
from marten.utils.worker import init_client
from marten.utils.database import get_database_engine
from marten.utils.neuralprophet import select_device
from marten.models.worker_func import (
    predict_best,
    # predict_adhoc,
    covars_and_search,
    ensemble_topk_prediction,
)

from types import SimpleNamespace

from dotenv import load_dotenv

logger = get_logger(__name__)
client = None
alchemyEngine = None


def init(args):
    global client, alchemyEngine
    client = init_client(
        __name__,
        args.max_worker,
        threads=args.threads,
        dashboard_port=args.dashboard_port,
        args=args,
    )

    if alchemyEngine is None:
        load_dotenv()  # take environment variables from .env.

        DB_USER = os.getenv("DB_USER")
        DB_PASSWORD = os.getenv("DB_PASSWORD")
        DB_HOST = os.getenv("DB_HOST")
        DB_PORT = os.getenv("DB_PORT")
        DB_NAME = os.getenv("DB_NAME")

        db_url = f"postgresql+psycopg2://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}"
        alchemyEngine = get_database_engine(db_url)


def main(args):
    global client, alchemyEngine, logger

    t_start = time.time()
    init(args)

    for symbol in args.symbols:
        if args.adhoc:
            # future = client.submit(predict_adhoc, symbol, args)
            hps_id, cutoff_date, ranked_features_future, df, df_future = (
                covars_and_search(client, symbol, alchemyEngine, logger, args)
            )
            logger.info("Starting adhoc prediction")
            t3_start = time.time()
            ensemble_topk_prediction(
                client,
                symbol,
                args.timestep_limit,
                args.random_seed,
                args.future_steps,
                args.topk,
                hps_id,
                cutoff_date,
                ranked_features_future,
                df,
                df_future,
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
            worker=-1,
            early_stopping=True,
            timestep_limit=1200,
            random_seed=7,
            epochs=500,
            future_steps=60,
            # symbols=["511220", "513800", "930955"],
            symbols=["511220"],
        )

        main(args)

    except Exception as e:
        logger.exception("main process terminated")
