from dask.distributed import Client, LocalCluster
import time
import math
import logging

from datetime import datetime


def configure_parser(parser):
    parser.add_argument(
        "--n_workers",
        type=int,
        default=96,
        help="Number of worker processes (default: 96).",
    )
    parser.add_argument(
        "--n_tasks",
        type=int,
        default=None,
        help="Number of tasks to submit (default: same as n_workers).",
    )
    parser.add_argument(
        "--n",
        type=int,
        default=100_000_000,
        help="Problem size for the CPU-intensive function (default: 100,000,000).",
    )

    parser.set_defaults(func=run_test)


def run_test(args):
    logging.getLogger("distributed.nanny").setLevel(logging.CRITICAL)
    logging.getLogger("distributed.scheduler").setLevel(logging.CRITICAL)
    logging.getLogger("tornado.application").setLevel(logging.CRITICAL)
    
    print("Running performance test...")
    
    n_workers = args.n_workers
    n_tasks = args.n_tasks if args.n_tasks is not None else n_workers
    n = args.n

    # Create a local Dask cluster with the specified number of workers
    cluster = LocalCluster(n_workers=n_workers, threads_per_worker=1)
    client = Client(cluster)

    start_time = datetime.now()
    # Submit tasks to the Dask cluster
    futures = [client.submit(cpu_heavy_function, n) for _ in range(n_tasks)]

    # Gather the results (computation times for each task)
    times_taken = client.gather(futures)

    time_taken = (datetime.now() - start_time).total_seconds()

    # Print the computation time for each task
    for idx, t in enumerate(times_taken):
        print(f"Task {idx + 1} took {t:.2f} seconds.")

    print(f"Total: {time_taken:.2f} seconds.")

    # Shutdown the Dask client and cluster
    try:
        client.shutdown()
    except Exception:
        pass


# Define a CPU-intensive function
def cpu_heavy_function(n):
    logging.getLogger("distributed.nanny").setLevel(logging.CRITICAL)

    start_time = datetime.now()
    result = 0.0
    for i in range(1, n):
        result += math.sqrt(i) * math.log(i + 1)
    end_time = datetime.now()
    time_taken = (end_time - start_time).total_seconds()
    return time_taken
