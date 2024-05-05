import requests
import time
import random

from marten.utils.logger import get_logger


def make_request(
    url,
    params=None,
    initial_timeout=45,
    max_timeout=None,
    max_attempts=5,
    max_delay=60,
    **kwargs,
):
    attempt = 0
    delay = random.uniform(1, 10)
    while attempt < max_attempts:
        # timeout = initial_timeout if attempt < max_attempts - 1 else None
        timeout = (
            min(
                max_timeout,
                initial_timeout
                + float(max_timeout - initial_timeout) / max_attempts * attempt,
            )
            if max_timeout is not None
            else initial_timeout if attempt < max_attempts - 1 else None
        )

        try:
            response = requests.get(url, params=params, timeout=timeout, **kwargs)
            # If the request was successful, return the response
            return response
        except requests.exceptions.Timeout:
            # If a timeout exception occurs, print an error and retry if attempts are left
            get_logger().warning(
                f"[{url}] attempt {attempt + 1} timed out. Retrying..."
                if attempt < max_attempts - 1
                else f"[{url}] final attempt timed out."
            )
            if attempt >= max_attempts - 1:
                raise e
        except Exception as e:
            raise e
        attempt += 1

        # wait before next attempt
        time.sleep(delay)

        delay = min(max_delay, (delay + random.uniform(1, 5)) * 2) + random.uniform(
            1, 10
        )

    # If the loop completes without returning, all attempts failed
    return None
