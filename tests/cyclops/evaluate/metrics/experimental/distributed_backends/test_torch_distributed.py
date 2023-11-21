"""Test the torch distributed backend."""
import sys
from functools import partial

import pytest

from cyclops.evaluate.metrics.experimental.distributed_backends.torch_distributed import (
    TorchDistributed,
)
from cyclops.utils.optional import import_optional_module

from ...conftest import NUM_PROCESSES
from ..testers import DummyListStateMetric, DummyMetric


torch = import_optional_module("torch", error="warn")


def _test_torch_distributed_class(rank: int, worldsize: int = NUM_PROCESSES):
    """Run test."""
    torch_dist = import_optional_module("torch.distributed", error="warn")
    if torch is None:
        with pytest.raises(
            ImportError,
            match="For availability of `TorchDistributed` please install .*",
        ):
            backend = TorchDistributed()

    if torch_dist is None:
        with pytest.raises(RuntimeError):
            backend = TorchDistributed()

    # skip if torch distributed is not available
    if torch_dist is None:
        pytest.skip("`torch.distributed` is not available")

    backend = TorchDistributed()

    assert backend.is_initialized == torch_dist.is_initialized()
    assert backend.rank == rank
    assert backend.world_size == worldsize

    # test all simple all gather (tensors of the same size)
    tensor = torch.ones(2)  # type: ignore
    result = backend._simple_all_gather(tensor)
    assert len(result) == worldsize
    for idx in range(worldsize):
        val = result[idx]
        assert (val == torch.ones_like(val)).all()  # type: ignore

    # test all gather uneven tensors
    tensor = torch.ones(rank)  # type: ignore
    result = backend.all_gather(tensor)
    assert len(result) == worldsize
    for idx in range(worldsize):
        val = result[idx]
        assert (val == torch.ones_like(val)).all()  # type: ignore

    # test all gather multidimensional uneven tensors
    tensor = torch.ones(rank + 1, 2 - rank)  # type: ignore
    result = backend.all_gather(tensor)
    assert len(result) == worldsize
    for idx in range(worldsize):
        val = result[idx]
        assert (val == torch.ones_like(val)).all()  # type: ignore


@pytest.mark.skipif(sys.platform == "win32", reason="DDP not available on windows")
def test_torch_distributed_backend_class():
    """Test `TorchDistributed` class."""
    pytest.torch_pool.map(_test_torch_distributed_class, range(NUM_PROCESSES))  # type: ignore


def _test_dist_sum(rank: int, worldsize: int = NUM_PROCESSES) -> None:
    dummy = DummyMetric(dist_backend="torch_distributed")
    dummy._reductions = {"foo": torch.sum}
    dummy.foo = torch.tensor(1)
    dummy.sync()

    assert dummy.foo == worldsize


def _test_dist_cat(rank: int, worldsize: int = NUM_PROCESSES) -> None:
    dummy = DummyMetric(dist_backend="torch_distributed")
    dummy._reductions = {"foo": torch.cat}
    dummy.foo = [torch.tensor([1])]
    dummy.sync()

    assert torch.all(torch.eq(dummy.foo, torch.tensor([1, 1])))


def _test_dist_sum_cat(rank: int, worldsize: int = NUM_PROCESSES) -> None:
    dummy = DummyMetric(dist_backend="torch_distributed")
    dummy._reductions = {"foo": torch.cat, "bar": torch.sum}
    dummy.foo = [torch.tensor([1])]
    dummy.bar = torch.tensor(1)
    dummy.sync()

    assert torch.all(torch.eq(dummy.foo, torch.tensor([1, 1])))
    assert dummy.bar == worldsize


def _test_dist_compositional_tensor(rank: int, worldsize: int = NUM_PROCESSES) -> None:
    dummy = DummyMetric(dist_backend="torch_distributed")
    dummy = dummy.clone() + dummy.clone()
    dummy.update(torch.tensor(1))
    val = dummy.compute()
    assert val == 2 * worldsize


@pytest.mark.skipif(sys.platform == "win32", reason="DDP not available on windows")
@pytest.mark.skipif(torch is None, reason="torch is not available")
@pytest.mark.parametrize(
    "process",
    [
        _test_dist_cat,
        _test_dist_sum,
        _test_dist_sum_cat,
        _test_dist_compositional_tensor,
    ],
)
def test_ddp(process):
    """Test ddp functions."""
    pytest.torch_pool.map(process, range(NUM_PROCESSES))  # type: ignore


def _test_sync_on_compute_tensor_state(rank):
    dummy = DummyMetric(dist_backend="torch_distributed")
    dummy.update(torch.tensor(rank + 1))
    val = dummy.compute()

    assert val == 3


def _test_sync_on_compute_list_state(rank):
    dummy = DummyListStateMetric(dist_backend="torch_distributed")
    dummy.update(torch.tensor(rank + 1))
    val = dummy.compute()
    assert val.sum() == 3
    assert torch.allclose(val, torch.tensor([1, 2])) or torch.allclose(
        val,
        torch.tensor([2, 1]),
    )


@pytest.mark.skipif(sys.platform == "win32", reason="DDP not available on windows")
@pytest.mark.parametrize(
    "test_func",
    [_test_sync_on_compute_list_state, _test_sync_on_compute_tensor_state],
)
def test_sync_on_compute(test_func):
    """Test that synchronization of states can be enabled and disabled for compute."""
    pytest.torch_pool.map(partial(test_func), range(NUM_PROCESSES))
