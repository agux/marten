import torch
from marten.utils.logger import get_logger

def should_retry(exception):
    return isinstance(exception, torch.cuda.OutOfMemoryError) or (
        "out of memory" in str(exception)
    )


def log_retry(retry_state):
    if retry_state.outcome.failed:
        exception = retry_state.outcome.exception()
        get_logger().warning(
            f"Retrying, attempt {retry_state.attempt_number} after exception: {exception}"
        )
