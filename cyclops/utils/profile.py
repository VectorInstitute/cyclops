"""Useful functions for timing, profiling."""

import logging
import time
from typing import Callable

from cyclops.utils.log import setup_logging

# Logging.
LOGGER = logging.getLogger(__name__)
setup_logging(print_level="INFO", logger=LOGGER)


def time_function(func: Callable) -> Callable:
    """Time decorator function.

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
        LOGGER.info("Finished executing function %s in %f s", func.__name__, time_taken)
        return result

    return wrapper_func
