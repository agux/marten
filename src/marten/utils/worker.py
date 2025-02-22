import os
import time
import random
import logging
import psutil
import multiprocessing
import threading
import torch
import math

# import ray

import dask
import dask.config
from dask.distributed import (
    WorkerPlugin,
    LocalCluster,
    Client,
    get_client,
    get_worker,
    worker_client,
    Future,
    Lock,
    Semaphore,
    as_completed,
)

from typing import List

from marten.utils.database import get_database_engine
from marten.utils.logger import get_logger
from marten.utils.net import get_machine_ips
from marten.utils.trainer import is_cuda_error, cuda_memory_stats

# from marten.utils.local_cluster import LocalCluster

from dotenv import load_dotenv

from datetime import datetime, timedelta

from pprint import pformat

logging.getLogger("NP.plotly").setLevel(logging.CRITICAL)
logging.getLogger("prophet.plot").disabled = True

compute_power = None


class LocalWorkerPlugin(WorkerPlugin):
    def __init__(self, logger_name, args):
        self.logger_name = logger_name
        self.args = args

    def setup(self, worker):
        load_dotenv()  # take environment variables from .env.
        DB_USER = os.getenv("DB_USER")
        DB_PASSWORD = os.getenv("DB_PASSWORD")
        DB_HOST = os.getenv("DB_HOST")
        DB_PORT = os.getenv("DB_PORT")
        DB_NAME = os.getenv("DB_NAME")

        db_url = f"postgresql+psycopg2://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}"

        worker.alchemyEngine = get_database_engine(db_url)
        worker.logger = get_logger(self.logger_name, role="worker")
        worker.args = self.args

        # worker.logger.info(torch.__config__.parallel_info())

        if hasattr(self.args, "model"):
            torch.set_num_interop_threads(1)
            # torch.cuda.set_per_process_memory_fraction(1.0)
            match self.args.model.lower():
                case "timemixer":
                    from marten.models.time_mixer import TimeMixerModel

                    worker.model = TimeMixerModel()
                case "tsmixerx":
                    from marten.models.nf_tsmixerx import TSMixerxModel

                    worker.model = TSMixerxModel()
                case _:
                    worker.model = None

        # if hasattr(self.args, "max_worker"):
        #     # Sets the number of threads used for intraop parallelism on CPU.
        #     n_threads = int(
        #         os.getenv(
        #             "TORCH_CPU_THREADS",
        #             psutil.cpu_count() / self.args.max_worker,
        #         )
        #     )
        #     if n_threads > 0:
        #         torch.set_num_threads(n_threads)

        rank_zero_logger = logging.getLogger("pytorch_lightning.utilities.rank_zero")
        rank_zero_logger.addFilter(IgnorePLFilter())

        logging.getLogger("NP.plotly").setLevel(logging.CRITICAL)
        logging.getLogger("prophet.plot").disabled = True
        # configure logging at the root level of Lightning
        logging.getLogger("lightning.pytorch").setLevel(logging.ERROR)

        worker.logger.debug(
            (
                "MALLOC_TRIM_THRESHOLD_: %s\n"
                "OMP_NUM_THREADS: %s\n"
                "MKL_NUM_THREADS: %s\n"
                "OPENBLAS_NUM_THREADS: %s\n"
                "NUMEXPR_NUM_THREADS: %s\n"
                "VECLIB_MAXIMUM_THREADS: %s\n"
                "torch.__config__.parallel_info():\n%s\n"
            ),
            os.getenv("MALLOC_TRIM_THRESHOLD_"),
            os.getenv("OMP_NUM_THREADS"),
            os.getenv("MKL_NUM_THREADS"),
            os.getenv("OPENBLAS_NUM_THREADS"),
            os.getenv("NUMEXPR_NUM_THREADS"),
            os.getenv("VECLIB_MAXIMUM_THREADS"),
            torch.__config__.parallel_info(),
        )


class IgnorePLFilter(logging.Filter):
    def filter(self, record):
        msg = record.getMessage()
        return (
            "GPU available:" not in msg
            and "TPU available:" not in msg
            and "HPU available:" not in msg
        )


class TaskException(Exception):
    def __init__(self, message, **args):
        # Call the base class constructor with the message
        super().__init__(message)
        self.message = message
        self.task_info = args

    def __str__(self):
        # Return a custom string representation
        if self.task_info:
            return f"{self.args[0]} : {self.task_info}"
        else:
            return self.args[0]


def local_machine_power():
    global compute_power

    if compute_power is not None:
        return compute_power

    if not torch.cuda.is_available():
        compute_power = 1
        return compute_power

    device_count = torch.cuda.device_count()
    total_memory = 0
    multi_processor_count = 0
    for i in range(device_count):
        total_memory += torch.cuda.get_device_properties(i).total_memory / (
            1024**2
        )  # Convert to MB
        multi_processor_count += torch.cuda.get_device_properties(
            i
        ).multi_processor_count
    compute_power = 2 if total_memory >= 8192 and multi_processor_count >= 64 else 1
    return compute_power


def init_ray(args):
    max_worker = getattr(args, "max_worker", -1)
    ray.init(
        num_cpus=getattr(
            args,
            "min_worker",
            int(max_worker) if max_worker > 0 else multiprocessing.cpu_count(),
        )
    )


def init_client(name, max_worker=-1, threads=1, dashboard_port=None, args=None):
    # setting worker resources in environment variable for restarted workers
    # os.environ["DASK_DISTRIBUTED__WORKER__RESOURCES__POWER"] = str(local_machine_power())
    power = local_machine_power()
    dask_admin_logs = getattr(args, "dask_admin_logs", False)
    dask.config.set(scheduler="processes")
    dask.config.set(
        {
            # "distributed.client.heartbeat": "5s",
            "distributed.worker.memory.terminate": False,
            # NOTE restarting worker may cause distributed.lock to malfunction, setting None to its client.scheduler
            # "distributed.worker.lifetime.duration": "1 hour",
            # "distributed.worker.lifetime.stagger": "2 minutes",
            # "distributed.worker.lifetime.restart": True,
            # "distributed.worker.profile.enabled": False,
            "distributed.worker.resources.POWER": power,
            # "distributed.worker.connections.outgoing": 100,
            # "distributed.worker.connections.incoming": 100,
            "distributed.worker.multiprocessing-method": getattr(
                args, "dask_multiprocessing", "spawn"
            ),
            # "distributed.worker.memory.recent-to-old-time": "45 minutes",
            # "distributed.deploy.lost-worker-timeout": "2 hours",
            # "distributed.scheduler.work-stealing-interval": "5 seconds",
            # "distributed.scheduler.worker-ttl": "8 hours",
            # "distributed.scheduler.worker-saturation": 0.0001,
            "distributed.scheduler.locks.lease-timeout": "15 minutes",
            # "distributed.scheduler.validate": True,
            "distributed.comm.retry.count": 10,
            # "distributed.comm.timeouts.connect": "120s",
            # "distributed.comm.timeouts.tcp": "600s",
            # "distributed.nanny.pre-spawn-environ.MALLOC_TRIM_THRESHOLD_": 0,
            # "distributed.nanny.environ.MALLOC_TRIM_THRESHOLD_": 0,
            "distributed.admin.log-length": 10000 if dask_admin_logs else 0,
            "distributed.admin.low-level-log-length": 1000 if dask_admin_logs else 0,
            "distributed.admin.system-monitor.log-length": (
                7200 if dask_admin_logs else 0
            ),
            # "distributed.admin.event-loop": "asyncio",  # tornado, asyncio, or uvloop
        }
    )
    mem_limit = os.getenv("dask_worker_memory_limit")
    cluster = LocalCluster(
        host="0.0.0.0",  # NOTE: if not using 0.0.0.0, remote machines may not be able to join the cluster
        # host="localhost",
        scheduler_port=getattr(args, "scheduler_port", 0),
        # scheduler_kwargs={"external_address": "localhost"},
        # n_workers=getattr(
        #     args,
        #     "min_worker",
        #     int(max_worker) if max_worker > 0 else multiprocessing.cpu_count(),
        # ),
        n_workers=int(max_worker),
        threads_per_worker=threads,
        processes=True,
        dashboard_address=":8787" if dashboard_port is None else f":{dashboard_port}",
        worker_dashboard_address=":0",
        silence_logs=logging.INFO if getattr(args, "dask_log", False) else logging.WARN,
        # memory_limit="2GB",
        memory_limit=mem_limit if mem_limit else 0,  # 0=no limit
    )
    # unstable. worker got killed prematurely even there's job running
    # cluster.adapt(
    #     interval="15s",
    #     target_duration="5m",
    #     wait_count="2000",
    #     minimum=getattr(
    #         args,
    #         "min_worker",
    #         int(round(max_worker / 10.0, 0)) if max_worker > 0 else 4,
    #     ),
    #     maximum=max_worker if max_worker > 0 else multiprocessing.cpu_count(),
    # )
    client = Client(
        cluster,
        # direct_to_workers=True,
        # connection_limit=512,
        # security=False
    )
    client.register_plugin(LocalWorkerPlugin(name, args))
    client.forward_logging()
    get_logger(name).info(
        "dask dashboard can be accessed at: %s", cluster.dashboard_link
    )
    get_logger(name).info("dask scheduler address: %s", cluster.scheduler_address)
    get_logger(name).info(
        "dask config: %s",
        dask.config.config,
    )

    return client


def get_result(future: Future):
    try:
        r = future.result()
        return r
    except Exception as e:
        get_logger().exception(e)
        pass


def get_results(futures):
    for f in futures:
        get_logger().debug("getting result for %s", _get_future_details(f))
        get_result(f)


def num_undone(futures: List[Future], shared_vars):
    undone = 0
    len_before = len(futures)
    if isinstance(futures, list):
        for f in futures[
            :
        ]:  # Iterate over a copy of the list to avoid mutation-within-loop issue
            if f.done():
                get_result(f)
                futures.remove(f)
            else:
                undone += 1
    elif isinstance(futures, dict):
        keys_to_remove = []
        for k in futures.keys():
            f = futures[k]
            if f.done():
                get_result(f)
                if k in shared_vars:
                    shared_vars[k].delete()
                    shared_vars.pop(k)
                keys_to_remove.append(k)
            else:
                undone += 1
        for k in keys_to_remove:
            futures.pop(k)

    else:
        get_logger().warning(
            "unsupported futures collection type: %s, %s", type(futures), futures
        )

    len_after = len(futures)
    get_logger().debug("len(futures) before: %s, after: %s", len_before, len_after)
    return undone


def random_seconds(a, b, max):
    return min(float(max), round(random.uniform(float(a), float(b)), 3))


def handle_task_timeout(futures, task_timeout, shared_vars):
    for symbol in list(shared_vars.keys()):  # Create a list of keys to iterate over
        try:
            var = shared_vars[symbol]  # Access the variable using the key
            st_dict = var.get("200ms")
            if not "start_time" in st_dict:
                # the task has not been started by the worker process yet
                continue
            if datetime.now() >= st_dict["start_time"] + timedelta(
                seconds=task_timeout
            ):
                ## the task has timed out. if the future is not finished yet, cancel it.
                if symbol in futures:
                    future = futures[symbol]
                    if not future.done():
                        future.cancel()
                        futures.pop(symbol)
                var.delete()
                shared_vars.pop(
                    symbol
                )  # Safe to pop because we're not iterating over shared_vars directly
        except TimeoutError as e:
            pass


def _get_future_details(future):
    details = {
        "Key": future.key,
        "Status": future.status,
        "Function": getattr(future, "function", "N/A"),
        "Arguments": getattr(future, "args", "N/A"),
        "Keywords": getattr(future, "kwargs", "N/A"),
    }
    if future.status == "error":
        details["Exception"] = future.exception()
        details["Traceback"] = future.traceback()
    return details


def log_futures(futures):
    for f in futures:
        get_logger().debug(pformat(_get_future_details(f)))


def await_futures(
    futures: List[Future],
    until_all_completed=True,
    task_timeout=None,
    shared_vars=None,
    multiplier=1,
    hard_wait=False,
    max_delay=15,
):
    num = num_undone(futures, shared_vars)
    get_logger().debug("undone futures: %s", num)
    if until_all_completed:
        if task_timeout is not None and shared_vars is not None:
            while num is not None and num > 0:
                time.sleep(random_seconds(num >> 2, num >> 1, max_delay))
                num = handle_task_timeout(futures, task_timeout, shared_vars)
        else:
            get_logger().debug("waiting until all futures complete: %s", num)
            for batch in as_completed(futures[:], timeout=task_timeout).batches():
                try:
                    for future in batch:
                        get_result(future)
                        futures.remove(future)
                except Exception as e:
                    get_logger().error(e, exc_info=True)
            # while num > 0:
            #     time.sleep(random_seconds(num >> 5, num >> 4, 30))
            #     if hard_wait:
            #         get_results(futures)
            #     num = num_undone(futures, shared_vars)
            #     get_logger().debug("undone futures: %s", num)
            #     log_futures(futures)
    elif num > multiprocessing.cpu_count() * multiplier:
        delta = num - multiprocessing.cpu_count() * multiplier
        threading.Event().wait(random_seconds(math.pow(delta, 0.9), delta, max_delay))
        # time.sleep(random_seconds(math.pow(delta, 0.8), delta, max_delay))

        if task_timeout is not None and shared_vars is not None:
            handle_task_timeout(futures, task_timeout, shared_vars)


def num_workers(local=True):
    workers = get_client().scheduler_info()["workers"]
    if local:
        count = 0
        machine_ips = get_machine_ips()
        # Iterate over the workers dictionary
        for worker_key, worker_info in workers.items():
            # Check if the worker's key or name matches any of the machine IPs
            if any(
                ip in worker_key
                or (isinstance(worker_info["name"], str) and ip in worker_info["name"])
                or "127.0.0.1" in worker_key
                or "127.0.0.1" in worker_info["name"]
                for ip in machine_ips
            ):
                count += 1
        return count
    else:
        return len(workers)


def hps_task_callback(future: Future):
    get_logger().debug("done future: %s", future)
    exception = None
    result = None
    try:
        result = future.result()

        lock_key = "workload_info.finished"
        with Lock(lock_key):
            fin = future.client.get_metadata(["workload_info", "finished"])
            future.client.set_metadata(["workload_info", "finished"], fin + 1)
    except Exception as e:
        exception = e

    if not isinstance(result, Exception) and exception is None:
        return
    elif exception is None:
        exception = result
    #     #TODO how to enforce retry with GPU=False in case of CUDA error
    #     future.key
    #     future.done()
    #     future.retry()
    get_logger().warning("exception in future: %s", exception)
    if not isinstance(exception, TaskException):
        return

    if hasattr(e.task_info, "worker") and getattr(e.task_info, "restart_worker", False):
        worker = e.task_info.worker
        get_logger().warning(
            "Restarting worker %s due to %s",
            worker.address,
            e.message,
        )
        future.client.restart_workers(
            workers=[worker.address], timeout=600, raise_for_error=False
        )


#     future.client.restart_workers()


def restart_all_workers(client: Client):
    try:
        client.restart(timeout="30s")
    except Exception as e:
        get_logger().warning("client.restart() failed: %s", e, exc_info=True)
        workers = client.scheduler_info()["workers"]
        worker_names = [worker_info["name"] for worker_info in workers.values()]
        client.restart_workers(worker_names, timeout=45, raise_for_error=False)


def restart_worker(exception):
    worker = get_worker()
    if worker is None:
        return
    get_logger().warning(
        "Restarting worker %s due to %s\n%s",
        worker.address,
        str(exception),
        cuda_memory_stats(),
    )
    with worker_client() as client:
        task_key = worker.get_current_task()
        client.set_metadata([task_key, "CUDA error"], is_cuda_error(exception))
        client.restart_workers(workers=[worker.address], timeout=900)


def release_lock(lock: Lock, after=10):
    if (
        lock is None
        or lock.get_value() <= 0
        # or (isinstance(lock, Lock) and not lock.locked)
        # or (isinstance(lock, Semaphore) and lock.get_value() <= 0)
    ):
        return

    def _release(worker_name):
        nonlocal lock
        get_logger().debug("lock %s will be released in %s seconds", lock.name, after)
        # time.sleep(after)
        try:
            get_logger().info(
                "[worker#%s] releasing lock: %s", worker_name, lock.name
            )
            lock.release()
            get_logger().info("[worker#%s] lock %s released", worker_name, lock.name)
        except Exception as e:
            msg = str(e).lower()
            if "lock is not yet acquired" in msg or "released too often" in msg:
                return
            get_logger().warning("exception releasing lock %s: %s", lock.name, str(e))

    w = get_worker()
    if after <= 0:
        _release(w.name)
    else:
        threading.Timer(after, _release, (w.name,)).start()


def gpu_util():
    util = torch.cuda.utilization()
    mu = torch.cuda.memory_usage()
    return util, mu


def mps_util():
    dam = torch.mps.driver_allocated_memory()
    cam = torch.mps.current_allocated_memory()
    return dam, cam


def wait_gpu(util_threshold=80, vram_threshold=80, stop_at=None):
    if stop_at is not None and time.time() > stop_at:
        return False
    util, mu = gpu_util()
    keep_waiting = util > 0 and (util >= util_threshold or mu >= vram_threshold)
    get_logger().debug(
        "gpu: %s/%s, vram: %s/%s, keep_waiting: %s",
        util,
        util_threshold,
        mu,
        vram_threshold,
        keep_waiting,
    )
    return keep_waiting


def wait_mps(stop_at=None):
    if stop_at is not None and time.time() > stop_at:
        return False
    dam, cam = mps_util()
    keep_waiting = cam > 0
    get_logger().debug(
        "mps driver allocated memory: %s, current allocated memory: %s, keep_waiting: %s",
        dam,
        cam,
        keep_waiting,
    )
    return keep_waiting


def cpu_util(interval=0.1):
    cpu_util = psutil.cpu_percent(interval)
    mem_util = psutil.virtual_memory().percent
    return cpu_util, mem_util


def wait_cpu(util_threshold=80, mem_threshold=80, stop_at=None, interval=0.1):
    if stop_at is not None and time.time() > stop_at:
        return False
    util, mem = cpu_util(interval)
    keep_waiting = util >= util_threshold or mem >= mem_threshold
    get_logger().debug(
        "cpu: %s/%s, mem: %s/%s, keep_waiting: %s",
        util,
        util_threshold,
        mem,
        mem_threshold,
        keep_waiting,
    )
    return keep_waiting


def workload_stage():
    """
    starting / progressing / finishing / unknown
    """
    client = get_client()
    # with worker_client() as client:
    workload_info = client.get_metadata(["workload_info"], None)
    if workload_info is None:
        stage = "unknown"
    else:
        finished = workload_info["finished"]
        total = workload_info["total"]
        workers = client.scheduler_info()["workers"]
        if total - finished <= len(workers):
            stage = "finishing"
        elif finished <= len(workers):
            stage = "starting"
        else:
            stage = "progressing"
        get_logger().debug(
            "finished:%s total:%s workers:%s stage:%s",
            finished,
            total,
            len(workers),
            stage,
        )
    return stage


def scale_cluster_and_wait(client: Client, n_workers: int, timeout: int = 60):
    """
    Scale the Dask cluster to the specified number of workers and wait until all workers are ready.

    Parameters:
    - client: dask.distributed.Client
        The Dask client connected to your cluster.
    - n_workers: int
        The desired number of workers to scale to.
    - timeout: int, optional (default=60)
        The maximum time in seconds to wait for the cluster to scale.

    Raises:
    - TimeoutError: If the cluster does not scale to the desired number of workers within the timeout period.
    """
    # Initiate scaling
    client.cluster.scale(n_workers)

    # Start timing
    start_time = time.time()

    # Loop until the desired number of workers is reached or timeout occurs
    while True:
        # Get the current number of workers
        current_workers = len(client.scheduler_info()["workers"])

        # Check if scaling is complete
        if current_workers == n_workers:
            break

        # Check for timeout
        elapsed_time = time.time() - start_time
        if elapsed_time > timeout:
            raise TimeoutError(
                f"Timeout: Expected {n_workers} workers, but {current_workers} are alive after {timeout} seconds."
            )

        # Optional: Sleep before the next check to avoid overwhelming the scheduler with requests
        time.sleep(1)
