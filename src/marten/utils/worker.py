import os
import time
import random
import psutil
import multiprocessing
import threading
import torch
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
)

from marten.utils.database import get_database_engine
from marten.utils.logger import get_logger
from marten.utils.net import get_machine_ips
from marten.utils.trainer import is_cuda_error, cuda_memory_stats

from dotenv import load_dotenv

from datetime import datetime, timedelta

from pprint import pformat

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
            match self.args.model.lower():
                case "timemixer":
                    from marten.models.time_mixer import TimeMixerModel

                    worker.model = TimeMixerModel()
                case _:
                    worker.model = None


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

def init_client(name, max_worker=-1, threads=1, dashboard_port=None, args=None):
    # setting worker resources in environment variable for restarted workers
    # os.environ["DASK_DISTRIBUTED__WORKER__RESOURCES__POWER"] = str(local_machine_power())
    power = local_machine_power()
    dask.config.set(
        {
            "distributed.worker.memory.terminate": False,
            # NOTE restarting worker may cause distributed.lock to malfunction, setting None to its client.scheduler
            # "distributed.worker.lifetime.duration": "1 hour",
            # "distributed.worker.lifetime.stagger": "1 minutes",
            # "distributed.worker.lifetime.restart": True,
            "distributed.worker.resources.POWER": power,
            "distributed.scheduler.work-stealing-interval": "5 seconds",
            "distributed.scheduler.worker-ttl": "30 minutes",
            "distributed.scheduler.worker-saturation": 0.0001,
            "distributed.comm.retry.count": 10,
            "distributed.comm.timeouts.connect": 120,
            "distributed.nanny.pre-spawn-environ.MALLOC_TRIM_THRESHOLD_": 0,
            "distributed.admin.log-length": 0,
            "distributed.admin.low-level-log-length": 0,
        }
    )
    cluster = LocalCluster(
        host="0.0.0.0",
        scheduler_port=getattr(args, "scheduler_port", 0),
        n_workers=getattr(
            args,
            "min_worker",
            int(max_worker) if max_worker > 0 else multiprocessing.cpu_count(),
        ),
        threads_per_worker=threads,
        processes=True,
        dashboard_address=":8787" if dashboard_port is None else f":{dashboard_port}",
        # memory_limit="2GB",
        memory_limit=0,  # no limit
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
    client = Client(cluster)
    client.register_plugin(LocalWorkerPlugin(name, args))
    client.forward_logging()
    get_logger(name).info(
        "dask dashboard can be accessed at: %s", cluster.dashboard_link
    )
    get_logger(name).info("dask scheduler address: %s", cluster.scheduler_address)

    return client


def get_result(future):
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


def num_undone(futures, shared_vars):
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
    futures,
    until_all_completed=True,
    task_timeout=None,
    shared_vars=None,
    multiplier=1,
    hard_wait=False,
):
    num = num_undone(futures, shared_vars)
    get_logger().debug("undone futures: %s", num)
    if until_all_completed:
        if task_timeout is not None and shared_vars is not None:
            while num is not None and num > 0:
                time.sleep(random_seconds(2 ** (num - 1), 2**num, 128))
                num = handle_task_timeout(futures, task_timeout, shared_vars)
        else:
            get_logger().debug("waiting until all futures complete: %s", num)
            while num > 0:
                time.sleep(random_seconds(2 ** (num - 1), 2**num, 128))
                if hard_wait:
                    get_results(futures)
                num = num_undone(futures, shared_vars)
                get_logger().debug("undone futures: %s", num)
                log_futures(futures)
    elif num > multiprocessing.cpu_count() * multiplier:
        delta = num - multiprocessing.cpu_count() * multiplier
        time.sleep(random_seconds(2 ** (delta - 1), 2**delta, 128))

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
            future.client.set_metadata(["workload_info", "finished"], fin+1)
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


def release_lock(lock, after=10):
    if lock is None:
        return

    def _release():
        get_logger().debug("lock %s will be released in %s seconds", lock.name, after)
        time.sleep(after)
        try:
            lock.release()
            get_logger().debug("lock %s released", lock.name)
        except Exception as e:
            if "lock is not yet acquired" not in str(e).lower():
                get_logger().warning(
                    "exception releasing lock %s: %s", lock.name, str(e)
                )

    if after <= 0:
        _release()
    else:
        threading.Timer(after, _release).start()

def wait_gpu(util_threshold=80, vram_threshold=80, stop_at=None):
    util = torch.cuda.utilization()
    mu = torch.cuda.memory_usage()
    return (
        util > 0
        and (util >= util_threshold or mu >= vram_threshold)
        and (stop_at is None or time.time() <= stop_at)
    )


def wait_cpu(util_threshold=80, mem_threshold=80, stop_at=None):
    cpu_util = psutil.cpu_percent(1)
    mem_util = psutil.virtual_memory().percent
    return (cpu_util >= util_threshold or mem_util >= mem_threshold) and (
        stop_at is None or time.time() <= stop_at
    )


def workload_stage():
    """
    starting / progressing / finishing / unknown
    """
    with worker_client() as client:
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
