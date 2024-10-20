import torch
import psutil
import socket
from dask.distributed import get_worker
from marten.utils.logger import get_logger


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
        "mem_cached": f"{torch.cuda.memory_cached() / 1024**2} MB",
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
    return (
        "gpu"
        if accelerator
        and torch.cuda.utilization() < util_threshold
        and torch.cuda.memory_usage() < vram_threshold
        else None
    )


def select_randk_covars(df, ranked_features, covar_dist, k):
    # get all column names not in ("ds", "y") from df
    # covar_cols = [col for col in df.columns if col not in ("ds", "y")]
    sorted_pairs = sorted(enumerate(covar_dist), key=lambda x: x[1], reverse=True)
    top_k_indices = [index for index, _ in sorted_pairs[:k]]
    top_k_features = [ranked_features[i] for i in top_k_indices]
    columns_to_keep = ["ds", "y"] + top_k_features
    return df[columns_to_keep]


def optimize_torch(ratio=0.85) -> int:
    torch.cuda.set_per_process_memory_fraction(1.0)

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
