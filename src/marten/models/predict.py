import time
import multiprocessing

from marten.utils.logger import get_logger
from marten.utils.worker import await_futures, init_client
from marten.models.worker_func import predict_best, predict_adhoc

from types import SimpleNamespace

logger = get_logger(__name__)
client = None


def init(args):
    global client
    client = init_client(
        __name__,
        (
            min(len(args.symbols), int(multiprocessing.cpu_count() * 0.9))
            if args.worker < 1
            else args.worker
        ),
        dashboard_port=args.dashboard_port,
    )


def main(args):
    global client

    t_start = time.time()
    init(args)

    futures = []
    for symbol in args.symbols:
        if args.adhoc:
            future = client.submit(
                predict_adhoc,
                symbol,
                args
            )
        else:
            future = client.submit(
                predict_best,
                symbol,
                args.early_stopping,
                args.timestep_limit,
                args.epochs,
                args.random_seed,
                args.future_steps,
                args.topk,
                "gpu" if args.accelerator else None,
            )
        futures.append(future)

    await_futures(futures)

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
