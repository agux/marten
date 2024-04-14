import logging
import os

from typing import Literal

from logging import Logger

from dotenv import load_dotenv


def get_logger(name, role: Literal["client", "worker"] = "client") -> Logger:
    load_dotenv()

    level = os.getenv("LOG_LEVEL")

    logger = logging.getLogger(name if name is not None else __name__)
    if not logger.handlers:  # Check if the logger already has handlers
        logger.setLevel(logging.INFO if level is None else level)

        formatter = None
        if role == 'client':
            formatter = logging.Formatter(
                "%(asctime)s - [worker %(worker)s] - %(name)s - %(levelname)s - %(message)s",
                datefmt="%Y-%m-%d %H:%M:%S",
            )
            file_handler = logging.FileHandler(f"{logger.name}.log")
            file_handler.setLevel(logging.INFO if level is None else level)
            file_handler.setFormatter(formatter)
            logger.addHandler(file_handler)
        elif role == 'worker':
            formatter = logging.Formatter(
                "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
                datefmt="%Y-%m-%d %H:%M:%S",
            )

        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO if level is None else level)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

    # Disable propagation to the root logger
    # logger.propagate = False

    return logger
