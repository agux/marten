import torch
from dask.distributed import get_worker
from marten.utils.logger import get_logger


def should_retry(exception):
    exmsg = str(exception)
    return not isinstance(exception, TimeoutError) and (
        isinstance(exception, torch.cuda.OutOfMemoryError)
        or "out of memory" in exmsg
        or "CUDA error" in exmsg
        or "cuDNN error" in exmsg
        or "unable to find an engine" in exmsg
    )


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


def select_device(accelerator, util_threshold=80, vram_threshold=80):
    return (
        "gpu"
        if accelerator
        and torch.cuda.utilization() < util_threshold
        and torch.cuda.memory_usage() < vram_threshold
        else None
    )
