import torch
import psutil
import socket

import dask
from dask.distributed import get_worker, Semaphore

from marten.utils.logger import get_logger


def get_accelerator_locks(cpu_leases=1, gpu_leases=1, mps_leases=1, timeout="10s"):
    if cpu_leases > 0 or gpu_leases > 0 or mps_leases > 0:
        dask.config.set({"distributed.scheduler.locks.lease-timeout": timeout})
    locks = {}
    if cpu_leases > 0:
        locks["cpu"] = Semaphore(
            max_leases=cpu_leases, name=f"""{socket.gethostname()}::CPU"""
        )
    if gpu_leases > 0:
        locks["gpu"] = Semaphore(
            max_leases=gpu_leases, name=f"""{socket.gethostname()}::GPU-auto"""
        )
    if mps_leases > 0:
        locks["mps"] = Semaphore(
            max_leases=mps_leases, name=f"""{socket.gethostname()}::MPS"""
        )
    return locks


def is_cuda_error(exception):
    exmsg = str(exception)
    return not isinstance(exception, TimeoutError) and (
        isinstance(exception, torch.cuda.OutOfMemoryError)
        or "out of memory" in exmsg
        or "CUDA error" in exmsg
        or "cuDNN error" in exmsg
        or "unable to find an engine" in exmsg
    )


def cuda_memory_stats():
    return {
        "mem_allocated": f"{torch.cuda.memory_allocated() / 1024**2} MB",
        "mem_reserved": f"{torch.cuda.memory_reserved() / 1024**2} MB",
    }


def log_retry(retry_state):
    if retry_state.outcome.failed:
        exception = retry_state.outcome.exception()
        get_logger().warning(
            f"Retrying, attempt {retry_state.attempt_number} after exception: {exception}"
        )


def log_train_args(df, *args, **kwargs):
    worker = get_worker()
    logger = worker.logger
    logger.info(
        (
            "Model training arguments:\n"
            "Dataframe %s:\n%s\n%s\n"
            "Positional arguments:%s\n"
            "Keyword arguments:%s"
        ),
        df.shape,
        df.describe().to_string(),
        df,
        args,
        kwargs,
    )


# this simple triage algorithm is to be deprecated
def select_device(accelerator, util_threshold=80, vram_threshold=80):
    cpu_util = psutil.cpu_percent(0.1)
    mem_util = psutil.virtual_memory().percent
    gpu_util = torch.cuda.utilization()
    vram_util = torch.cuda.memory_usage()
    return (
        "gpu"
        if accelerator
        and gpu_util < min(util_threshold, cpu_util)
        and vram_util < min(vram_threshold, mem_util)
        else None
    )


def select_randk_covars(df, ranked_features, covar_dist, k):
    # get all column names not in ("ds", "y") from df
    # covar_cols = [col for col in df.columns if col not in ("ds", "y")]
    sorted_pairs = sorted(enumerate(covar_dist), key=lambda x: x[1], reverse=True)
    top_k_indices = [index for index, _ in sorted_pairs[:k]]
    top_k_features = [ranked_features[i] for i in top_k_indices]
    columns_to_keep = ["ds", "y"]
    for f in top_k_features:
        if "::ta_" in f:
            # Technical indicators involved
            ti_prefix, cov_table, table, symbol = f.split("::")
            columns_to_keep += [
                c
                for c in df.columns
                if c.startswith(ti_prefix + "_") and c.endswith(f"{table}::{symbol}")
            ]
        else:
            columns_to_keep.append(f)
    # get_logger().info(
    #     "ranked_features:\n%s\ntop_k_features:\n%s\ndf.columns:\n%s\ncolumns_to_keep:\n%s\n",
    #     ranked_features,
    #     top_k_features,
    #     df.columns,
    #     columns_to_keep,
    # )
    return df[columns_to_keep].copy()


def optimize_torch_on_gpu():
    torch.cuda.set_per_process_memory_fraction(1.0)


def optimize_torch_on_cpu(
    ratio=0.85,
    n_threads=None,
) -> int:
    if not n_threads:
        cpu_cap = (100.0 - psutil.cpu_percent(1)) / 100.0
        n_cores = float(psutil.cpu_count())
        # n_workers = max(1.0, float(num_workers()))
        # n_threads = min(int(n_cores * cpu_cap * 0.85), n_cores/n_workers)
        n_threads = max(1, int(n_cores * cpu_cap * ratio))
    torch.set_num_threads(
        n_threads
    )  # Sets the number of threads used for intraop parallelism on CPU.

    # comment out to avoid below error:
    # cannot set number of interop threads after parallel work has started or set_num_interop_threads called
    # torch.set_num_interop_threads(n_threads)

    get_logger().debug(
        "machine: %s, cpu_cap: %s, n_cores: %s optimizing torch CPU thread: %s",
        socket.gethostname(),
        round(cpu_cap, 3),
        int(n_cores),
        n_threads,
    )

    return n_threads

    # Enable cuDNN auto-tuner
    # torch.backends.cudnn.benchmark = True


def remove_singular_variables(df):
    # Identify variables with singular values
    singular_vars = [col for col in df.columns if df[col].nunique() == 1]
    if singular_vars:
        get_logger().debug("Removing variables with singular values: %s", singular_vars)
        # Drop singular variables
        df = df.drop(columns=singular_vars)
    return df, singular_vars


def huber_loss(y_true, y_pred, delta=1.0):
    """
    Calculate the Huber loss between y_true and y_pred.

    Parameters:
    y_true : array_like
        True values.
    y_pred : array_like
        Predicted values.
    delta : float, optional
        The threshold parameter that controls the point where the loss function changes from quadratic to linear.
        Default is 1.0.

    Returns:
    float
        The computed Huber loss.
    """
    # Compute the residuals
    r = y_true - y_pred

    # Compute the absolute residuals
    abs_r = np.abs(r)

    # Use np.minimum to get the quadratic term where abs_r <= delta
    quadratic_term = np.minimum(abs_r, delta)

    # Compute the linear term (the difference between abs_r and the quadratic term)
    linear_term = abs_r - quadratic_term

    # Compute the Huber loss without any explicit conditionals or branches
    loss = 0.5 * quadratic_term**2 + delta * linear_term

    # Sum over all the elements to get the total loss
    return np.mean(loss)
