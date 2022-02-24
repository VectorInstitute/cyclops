"""Useful functions for timing, profiling."""

import logging
import time
from typing import Callable

from cyclops.utils.log import setup_logging, LOG_FILE_PATH


# Logging.
LOGGER = logging.getLogger(__name__)
setup_logging(log_path=LOG_FILE_PATH, print_level="INFO", logger=LOGGER)


def time_function(func: Callable) -> Callable:
    """Timing decorator function.

    Parameters
    ----------
    func: function
        Function to apply decorator.

    Returns
    -------
    Callable
        Wrapper function to apply as decorator.
    """

    def wrapper_func(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        time_taken = time.time() - start_time
        LOGGER.info(f"Finished executing function {func.__name__} in {time_taken} s")
        return result

    return wrapper_func
