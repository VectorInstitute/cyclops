"""pytest plugins and constants for tests/cyclops/evaluate/metrics/."""
import contextlib
import os
import sys

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


MAX_PORT = 8100
START_PORT = 8088
CURRENT_PORT = START_PORT


def setup_ddp(rank, world_size):
    """Initialize distributed environment."""
    global CURRENT_PORT  # noqa: PLW0603

    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = str(CURRENT_PORT)

    CURRENT_PORT += 1
    if CURRENT_PORT > MAX_PORT:
        CURRENT_PORT = START_PORT

    if torch.distributed.is_available() and sys.platform not in ("win32", "cygwin"):
        torch.distributed.init_process_group("gloo", rank=rank, world_size=world_size)


def pytest_configure():
    """Inject attributes into the pytest namespace."""
    pool = Pool(processes=NUM_PROCESSES)
    pool.starmap(setup_ddp, [(rank, NUM_PROCESSES) for rank in range(NUM_PROCESSES)])
    pytest.pool = pool  # type: ignore


def pytest_sessionfinish():
    """Close the global multiprocessing pool after all tests are done."""
    pytest.pool.close()  # type: ignore
    pytest.pool.join()  # type: ignore
