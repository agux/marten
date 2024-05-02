import requests

from marten.utils.logger import get_logger

def make_request(url, params=None, initial_timeout=45, total_attempts=5):
    attempt = 0
    while attempt < total_attempts:
        # Set timeout for the first two attempts, no timeout for the last attempt
        timeout = initial_timeout if attempt < total_attempts - 1 else None

        try:
            response = requests.get(url, params=params, timeout=timeout)
            # If the request was successful, return the response
            return response
        except requests.exceptions.Timeout:
            # If a timeout exception occurs, print an error and retry if attempts are left
            get_logger().warning(
                f"[{url}] attempt {attempt + 1} timed out. Retrying..."
                if attempt < total_attempts - 1
                else "[{url}] final attempt timed out."
            )
        except Exception as e:
            raise e
        attempt += 1
        
    # If the loop completes without returning, all attempts failed
    return None
