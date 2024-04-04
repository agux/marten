import logging
import os

from logging import Logger

from dotenv import load_dotenv

def get_logger(name) -> Logger:
    load_dotenv()

    level = os.getenv("LOG_LEVEL")

    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO if level is None else level)

    file_handler = logging.FileHandler("etl.log")
    console_handler = logging.StreamHandler()

    # Step 4: Create a formatter
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Step 5: Attach the formatter to the handlers
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)

    # Step 6: Add the handlers to the logger
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    return logger
