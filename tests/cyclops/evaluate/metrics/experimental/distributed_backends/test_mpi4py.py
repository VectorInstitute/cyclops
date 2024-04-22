"""Test mpi4py backend."""

import numpy as np
import numpy.array_api as anp
import pytest

from cyclops.evaluate.metrics.experimental.distributed_backends.mpi4py import (
    MPI4Py,
)
from cyclops.utils.optional import import_optional_module

from ..testers import DummyListStateMetric, DummyMetric


MPI = import_optional_module("mpi4py.MPI", error="warn")


@pytest.mark.skipif(MPI is None, reason="`mpi4py` is not installed.")
@pytest.mark.mpi(min_size=2)
def test_mpi4py_class_init():
    """Test MPI4Py class initialization."""
    comm = MPI.COMM_WORLD
    backend = MPI4Py()
    assert backend.is_initialized
    assert backend.rank == comm.Get_rank()
    assert backend.world_size == comm.size


@pytest.mark.skipif(MPI is None, reason="`mpi4py` is not installed.")
@pytest.mark.mpi(min_size=2)
def test_all_gather_simple():
    """Test all_gather with equal arrays."""
    backend = MPI4Py()
    comm = MPI.COMM_WORLD

    array = anp.ones(5)
    result = backend.all_gather(array)
    assert len(result) == comm.size
    for idx in range(comm.size):
        val = result[idx]
        assert anp.all(val == anp.ones_like(val))


@pytest.mark.skipif(MPI is None, reason="`mpi4py` is not installed.")
@pytest.mark.mpi(min_size=2)
def test_all_gather_uneven_arrays():
    """Test all_gather with uneven arrays."""
    backend = MPI4Py()
    comm = MPI.COMM_WORLD

    array = anp.ones(comm.Get_rank())
    result = backend.all_gather(array)
    assert len(result) == comm.size
    for idx in range(comm.size):
        val = result[idx]
        assert anp.all(val == anp.ones_like(val))


@pytest.mark.skipif(MPI is None, reason="`mpi4py` is not installed.")
@pytest.mark.mpi(min_size=2)
def test_all_gather_uneven_multidim_arrays():
    """Test all_gather with uneven multidimensional arrays."""
    backend = MPI4Py()
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    worldsize = comm.size

    array = anp.ones((rank + 1, 2 - rank, 2))
    result = backend.all_gather(array)
    assert len(result) == worldsize
    for idx in range(worldsize):
        val = result[idx]
        assert anp.all(val == anp.ones_like(val))


@pytest.mark.skipif(MPI is None, reason="`mpi4py` is not installed.")
@pytest.mark.mpi(min_size=2)
def test_dist_sum() -> None:
    """Test sum reduction."""
    dummy = DummyMetric(dist_backend="mpi4py")
    dummy._reductions = {"foo": anp.sum}
    dummy.foo = anp.asarray(1)
    dummy.sync()

    assert anp.all(dummy.foo == anp.asarray(MPI.COMM_WORLD.size))


@pytest.mark.skipif(MPI is None, reason="`mpi4py` is not installed.")
@pytest.mark.mpi(min_size=2)
def test_dist_cat() -> None:
    """Test concat reduction."""
    dummy = DummyMetric(dist_backend="mpi4py")
    dummy._reductions = {"foo": anp.concat}
    dummy.foo = [anp.asarray([1])]
    dummy.sync()

    assert anp.all(anp.equal(dummy.foo, anp.asarray([1, 1])))


@pytest.mark.skipif(MPI is None, reason="`mpi4py` is not installed.")
@pytest.mark.mpi(min_size=2)
def test_dist_sum_cat() -> None:
    """Test sum and concat reductions."""
    dummy = DummyMetric(dist_backend="mpi4py")
    dummy._reductions = {"foo": anp.concat, "bar": anp.sum}
    dummy.foo = [anp.asarray([1])]
    dummy.bar = anp.asarray(1)
    dummy.sync()

    assert anp.all(anp.equal(dummy.foo, anp.asarray([1, 1])))
    assert dummy.bar == MPI.COMM_WORLD.size


@pytest.mark.skipif(MPI is None, reason="`mpi4py` is not installed.")
@pytest.mark.mpi(min_size=2)
def test_dist_compositional_array() -> None:
    """Test compositional metric with array state."""
    dummy = DummyMetric(dist_backend="mpi4py")
    dummy = dummy.clone() + dummy.clone()
    dummy.update(anp.asarray(1, dtype=anp.float32))
    val = dummy.compute()
    assert val == 2 * MPI.COMM_WORLD.size


@pytest.mark.skipif(MPI is None, reason="`mpi4py` is not installed.")
@pytest.mark.mpi(min_size=2)
def test_sync_on_compute_array_state():
    """Test sync on compute with array state."""
    rank = MPI.COMM_WORLD.Get_rank()
    dummy = DummyMetric(dist_backend="mpi4py")
    dummy.update(anp.asarray(rank + 1, dtype=anp.float32))
    val = dummy.compute()

    assert anp.all(val == 3)


@pytest.mark.skipif(MPI is None, reason="`mpi4py` is not installed.")
@pytest.mark.mpi(min_size=2)
def test_sync_on_compute_list_state():
    """Test sync on compute with list state."""
    rank = MPI.COMM_WORLD.Get_rank()
    dummy = DummyListStateMetric(dist_backend="mpi4py")
    dummy.update(anp.asarray(rank + 1, dtype=anp.float32))
    val = dummy.compute()
    assert anp.all(anp.sum(val) == 3)
    assert np.allclose(val, anp.asarray([1, 2])) or np.allclose(
        val,
        anp.asarray([2, 1]),
    )
