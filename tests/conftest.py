"""Pytest configuration file."""

from typing import Any, Optional

import psutil


def pytest_xdist_auto_num_workers(config: Any) -> Optional[int]:
    """Configure the number of workers for xdist.

    This returns the number of workers to use for xdist when the `-n auto` or
    `-n logical` flags are passed to pytest. The number of workers is set to
    the number of CPUs divided by 4, with a minimum of 2 workers.

    Parameters
    ----------
    config : Any
        Configuration object.

    Returns
    -------
    Optional[int]
        Number of workers to use for xdist or `None` to fall back to default
        behavior.
    """
    num_cpus = None
    if config.option.numprocesses == "auto":
        num_cpus = psutil.cpu_count(logical=False)
    elif config.option.numprocesses == "logical":
        num_cpus = psutil.cpu_count(logical=True)

    if num_cpus is not None:
        return max(2, num_cpus // 4)

    return num_cpus
