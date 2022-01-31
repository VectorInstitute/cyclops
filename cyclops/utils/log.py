"""Python logging function."""

import os
import logging
from typing import Optional


LOG_FORMAT = "%(asctime)-15s %(levelname)-5s %(name)-15s - %(message)s"
LOG_FILE_PATH = os.path.join(os.environ["PROJECT_ROOT"], "log.log")


def setup_logging(
    log_path: Optional[str] = None,
    log_level: Optional[str] = "DEBUG",
    print_level: Optional[str] = "INFO",
    logger: Optional[logging.Logger] = None,
    fmt: Optional[str] = LOG_FORMAT,
):
    """
    Create logger, and set it up.

    Parameters
    ----------
    log_path : str, optional
        Path to output log file.
    log_level : str, optional
        Log level for logging, defaults to DEBUG.
    print_level : str, optional
        Print level for logging, defaults to INFO.
    logger : logging.Logger, optional
        Pass logger if already exists, else a new logger object is created.
    fmt : str, optional
        Logging format, default format specified above.
    """
    logger = logger if logger else logging.getLogger()
    logger.setLevel(log_level)
    logger.handlers = []

    fmt = logging.Formatter(fmt=fmt)
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(fmt)
    stream_handler.setLevel(print_level)
    logger.addHandler(stream_handler)

    if log_path:
        file_handler = logging.FileHandler(log_path)
        file_handler.setFormatter(fmt)
        logger.addHandler(file_handler)
        logger.info("Log file is %s", log_path)
