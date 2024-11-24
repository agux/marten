import logging
import os

from typing import Literal

from logging import Logger

from dotenv import load_dotenv

logger = None

class CustomFormatter(logging.Formatter):
    def format(self, record):
        # Check if 'worker' attribute exists, if not, set it to a default value
        if hasattr(record, "worker"):
            record.role = f"worker {record.worker}"
        else:
            record.role = "master"  # Default value for missing 'worker'
        # Call the superclass's format method to do the actual message formatting
        return super(CustomFormatter, self).format(record)


class WorkerMessageFilter(logging.Filter):
    def filter(self, record):
        # Define the specific part of the message you want to check for
        suppress_message_part = "Unmanaged memory use is high."
        # Return False if the specific message part is in the log record's message, suppressing it
        return suppress_message_part not in record.getMessage()


def get_logger(name=None, role: Literal["client", "worker"] = "client") -> Logger:
    # global logger

    # if logger is not None:
    #     return logger

    logger = logging.getLogger(name if name is not None else __name__)

    if not logger.handlers:  # Check if the logger already has handlers
        load_dotenv()

        level = os.getenv("LOG_LEVEL")
        level = level.upper() if level else None
        
        logger.setLevel(logging.INFO if level is None else level)

        formatter = None
        if role == 'client':
            formatter = CustomFormatter(
                "%(asctime)s - [%(role)s] - %(name)s - %(levelname)s - %(message)s",
                datefmt="%Y-%m-%d %H:%M:%S",
            )
            file_handler = logging.FileHandler(f"{logger.name}.log")
            file_handler.setLevel(logging.INFO if level is None else level)
            file_handler.setFormatter(formatter)
            logger.addHandler(file_handler)

            console_handler = logging.StreamHandler()
            console_handler.setLevel(logging.INFO if level is None else level)
            console_handler.setFormatter(formatter)
            logger.addHandler(console_handler)

            logging.getLogger("distributed.worker.memory").addFilter(
                WorkerMessageFilter()
            )
        elif role == 'worker':
            pass
            # formatter = logging.Formatter(
            #     "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
            #     datefmt="%Y-%m-%d %H:%M:%S",
            # )

            # Disable propagation to the root logger
            # logger.propagate = False


    return logger
