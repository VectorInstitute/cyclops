"""pytest plugins and constants for tests/cyclops/evaluate/metrics/."""
import contextlib
import os
import socket
import sys
from functools import partial

import pytest
import torch.distributed
from torch.multiprocessing import Pool, set_sharing_strategy, set_start_method


with contextlib.suppress(RuntimeError):
    set_start_method("spawn")
    set_sharing_strategy("file_system")


NUM_PROCESSES = 2
BATCH_SIZE = 16 * NUM_PROCESSES
NUM_BATCHES = 8
NUM_CLASSES = 10
NUM_LABELS = 10
EXTRA_DIM = 4
THRESHOLD = 0.6


def get_open_port():
    """Get an open port.

    Reference
    ---------
    1. https://stackoverflow.com/questions/66348957/pytorch-ddp-get-stuck-in-getting-free-port

    """
    with contextlib.closing(socket.socket(socket.AF_INET, socket.SOCK_STREAM)) as s:
        s.bind(("", 0))
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        return s.getsockname()[1]


def setup_ddp(rank, world_size, port):
    """Initialize distributed environment."""
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = str(port)

    if torch.distributed.is_available() and sys.platform not in ("win32", "cygwin"):
        torch.distributed.init_process_group("gloo", rank=rank, world_size=world_size)


def pytest_configure():
    """Inject attributes into the pytest namespace."""
    pool = Pool(processes=NUM_PROCESSES)
    pool.starmap(
        partial(setup_ddp, port=get_open_port()),
        [(rank, NUM_PROCESSES) for rank in range(NUM_PROCESSES)],
    )
    pytest.pool = pool  # type: ignore


def pytest_sessionfinish():
    """Close the global multiprocessing pool after all tests are done."""
    pytest.pool.close()  # type: ignore
    pytest.pool.join()  # type: ignore
