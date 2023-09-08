"""Useful functions for timing, profiling."""

import logging
import time
from typing import Any, Callable, Dict, List

from cyclops.utils.log import setup_logging


# Logging.
LOGGER = logging.getLogger(__name__)
setup_logging(print_level="INFO", logger=LOGGER)


def time_function(func: Callable[..., Any]) -> Callable[..., Any]:
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

    def wrapper_func(*args: List[Any], **kwargs: Dict[str, Any]) -> Any:
        start_time = time.time()
        result = func(*args, **kwargs)
        time_taken = time.time() - start_time
        LOGGER.info("Finished executing function %s in %f s", func.__name__, time_taken)
        return result

    return wrapper_func
