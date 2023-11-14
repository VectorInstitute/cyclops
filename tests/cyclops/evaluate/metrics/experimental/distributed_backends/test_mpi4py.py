"""Test mpi4py backend."""

from functools import partial

import numpy as np
import numpy.array_api as anp
import pytest

from cyclops.evaluate.metrics.experimental.distributed_backends.mpi4py import (
    MPI4Py,
)
from cyclops.utils.optional import import_optional_module

from ...conftest import NUM_PROCESSES
from ..testers import DummyListStateMetric, DummyMetric


MPI = import_optional_module("mpi4py.MPI", error="ignore")


def _test_mpi4py_class_init(rank: int, worldsize: int = 2):
    """Run test."""
    if MPI is None:
        with pytest.raises(
            ImportError,
            match="For availability of MPI4Py please install mpi4py first.",
        ):
            backend = MPI4Py()
            assert not backend.is_initialized
        pytest.skip("`mpi4py` is not installed.")

    backend = MPI4Py()
    assert backend.is_initialized
    assert backend.rank == rank
    assert backend.world_size == worldsize


@pytest.mark.integration_test()
def test_mpi4py_backend_class_init():
    """Test `TorchDistributed` class."""
    pytest.mpi_pool.starmap(_test_mpi4py_class_init, [(rank, 2) for rank in range(2)])  # type: ignore


def _test_all_gather_simple(rank: int, worldsize: int = 2):
    """Run test."""
    backend = MPI4Py()

    array = anp.ones(5)
    result = backend.all_gather(array)
    assert len(result) == worldsize
    for idx in range(worldsize):
        val = result[idx]
        assert anp.all(val == anp.ones_like(val))


def _test_all_gather_uneven_arrays(rank: int, worldsize: int = 2):
    """Run test."""
    backend = MPI4Py()

    array = anp.ones(rank)
    result = backend.all_gather(array)
    assert len(result) == worldsize
    for idx in range(worldsize):
        val = result[idx]
        assert anp.all(val == anp.ones_like(val))


def _test_all_gather_uneven_multidim_arrays(rank: int, worldsize: int = 2):
    """Run test."""
    backend = MPI4Py()

    array = anp.ones((rank + 1, 2 - rank, 2))
    result = backend.all_gather(array)
    assert len(result) == worldsize
    for idx in range(worldsize):
        val = result[idx]
        assert anp.all(val == anp.ones_like(val))


@pytest.mark.integration_test()
@pytest.mark.skipif(MPI is None, reason="`mpi4py` is not installed.")
@pytest.mark.parametrize(
    "case_fn",
    [
        _test_all_gather_simple,
        _test_all_gather_uneven_arrays,
        _test_all_gather_uneven_multidim_arrays,
    ],
)
def test_mpi4py_all_gather(case_fn):
    """Test `all_gather` method."""
    pytest.mpi_pool.starmap(case_fn, [(rank, 2) for rank in range(NUM_PROCESSES)])  # type: ignore


def _test_dist_sum(rank: int, worldsize: int = NUM_PROCESSES) -> None:
    dummy = DummyMetric(dist_backend="mpi4py")
    dummy._reductions = {"foo": anp.sum}
    dummy.foo = anp.asarray(1)
    dummy.sync()

    assert anp.all(dummy.foo == anp.asarray(worldsize))


def _test_dist_cat(rank: int, worldsize: int = NUM_PROCESSES) -> None:
    dummy = DummyMetric(dist_backend="mpi4py")
    dummy._reductions = {"foo": anp.concat}
    dummy.foo = [anp.asarray([1])]
    dummy.sync()

    assert anp.all(anp.equal(dummy.foo, anp.asarray([1, 1])))


def _test_dist_sum_cat(rank: int, worldsize: int = NUM_PROCESSES) -> None:
    dummy = DummyMetric(dist_backend="mpi4py")
    dummy._reductions = {"foo": anp.concat, "bar": anp.sum}
    dummy.foo = [anp.asarray([1])]
    dummy.bar = anp.asarray(1)
    dummy.sync()

    assert anp.all(anp.equal(dummy.foo, anp.asarray([1, 1])))
    assert dummy.bar == worldsize


def _test_dist_compositional_array(rank: int, worldsize: int = NUM_PROCESSES) -> None:
    dummy = DummyMetric(dist_backend="mpi4py")
    dummy = dummy.clone() + dummy.clone()
    dummy.update(anp.asarray(1, dtype=anp.float32))
    val = dummy.compute()
    print(val)
    assert val == 2 * worldsize


@pytest.mark.integration_test()
@pytest.mark.skipif(MPI is None, reason="`mpi4py` is not available")
@pytest.mark.parametrize(
    "process",
    [
        _test_dist_cat,
        _test_dist_sum,
        _test_dist_sum_cat,
        _test_dist_compositional_array,
    ],
)
def test_ddp(process):
    """Test ddp functions."""
    pytest.mpi_pool.map(process, range(NUM_PROCESSES))  # type: ignore


def _test_sync_on_compute_array_state(rank):
    dummy = DummyMetric(dist_backend="mpi4py")
    dummy.update(anp.asarray(rank + 1, dtype=anp.float32))
    val = dummy.compute()

    assert anp.all(val == 3)


def _test_sync_on_compute_list_state(rank):
    dummy = DummyListStateMetric(dist_backend="mpi4py")
    dummy.update(anp.asarray(rank + 1, dtype=anp.float32))
    val = dummy.compute()
    assert anp.all(anp.sum(val) == 3)
    assert np.allclose(val, anp.asarray([1, 2])) or np.allclose(
        val,
        anp.asarray([2, 1]),
    )


@pytest.mark.integration_test()
@pytest.mark.parametrize(
    "test_func",
    [_test_sync_on_compute_list_state, _test_sync_on_compute_array_state],
)
def test_sync_on_compute(test_func):
    """Test that synchronization of states can be enabled and disabled for compute."""
    pytest.mpi_pool.map(partial(test_func), range(NUM_PROCESSES))  # type: ignore
