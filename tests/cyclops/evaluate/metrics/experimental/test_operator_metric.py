"""Test the OperatorMetric class."""
import numpy as np
import numpy.array_api as anp
import pytest

from cyclops.evaluate.metrics.experimental.metric import Metric, OperatorMetric
from cyclops.evaluate.metrics.experimental.utils.types import Array


class DummyMetric(Metric):
    """DummyMetric class for testing operator metrics."""

    name: str = "DummyMetric"

    def __init__(self, val_to_return: Array) -> None:
        super().__init__()
        self.add_state_default_factory(
            "_num_updates",
            lambda xp: xp.asarray(0, device=self._device),  # type: ignore
            dist_reduce_fn="sum",
        )
        self._val_to_return = val_to_return

    def _update_state(self, unused_arg: Array) -> None:
        """Compute state."""
        self._num_updates += 1  # type: ignore

    def _compute_metric(self):
        """Compute result."""
        return anp.asarray(self._val_to_return)


def test_metrics_abs():
    """Test that `abs` operator works and returns an operator metric."""
    metric = DummyMetric(anp.asarray(-2, dtype=anp.float32))
    abs_metric = abs(metric)
    assert isinstance(abs_metric, OperatorMetric)
    abs_metric.update(anp.asarray(0))  # dummy value to get array namespace
    assert np.allclose(anp.asarray(2, dtype=anp.float32), abs_metric.compute())


@pytest.mark.parametrize(
    ("second_operand", "expected_result"),
    [
        (2, anp.asarray(4)),
        (2.0, anp.asarray(4.0)),
        (DummyMetric(anp.asarray(2)), anp.asarray(4)),
        (anp.asarray(2), anp.asarray(4)),
    ],
)
def test_metrics_add(second_operand, expected_result):
    """Test that `add` operator works and returns an operator metric."""
    first_metric = DummyMetric(
        anp.asarray(
            2,
            dtype=anp.float32 if isinstance(second_operand, float) else None,
        ),
    )

    final_add = first_metric + second_operand
    assert isinstance(final_add, OperatorMetric)
    final_add.update(anp.asarray(0))  # dummy value to get array namespace
    assert np.allclose(expected_result, final_add.compute())

    if not isinstance(second_operand, DummyMetric):
        with pytest.raises(TypeError, match="unsupported operand type.*"):
            final_radd = second_operand + first_metric  # type: ignore
    else:
        final_radd = second_operand + first_metric
        assert isinstance(final_radd, OperatorMetric)
        final_radd.update(anp.asarray(0))  # dummy value to get array namespace
        assert np.allclose(expected_result, final_radd.compute())


@pytest.mark.parametrize(
    ("second_operand", "expected_result"),
    [
        (False, anp.asarray(False)),
        (2, anp.asarray(2)),
        (DummyMetric(anp.asarray(42)), anp.asarray(42)),
        (anp.asarray(2), anp.asarray(2)),
    ],
)
def test_metrics_and(second_operand, expected_result):
    """Test that `and` operator works and returns an operator metric."""
    first_metric = DummyMetric(
        anp.asarray(True) if isinstance(second_operand, bool) else anp.asarray(42),
    )

    final_and = first_metric & second_operand
    assert isinstance(final_and, OperatorMetric)
    final_and.update(anp.asarray(0))  # dummy value to get array namespace
    assert np.allclose(expected_result, final_and.compute())

    if not isinstance(second_operand, DummyMetric):
        with pytest.raises(TypeError, match="unsupported operand type.*"):
            final_rand = second_operand & first_metric  # type: ignore
    else:
        final_rand = second_operand & first_metric
        assert isinstance(final_rand, OperatorMetric)
        final_rand.update(anp.asarray(0))  # dummy value to get array namespace
        assert np.allclose(expected_result, final_rand.compute())


@pytest.mark.parametrize(
    ("second_operand", "expected_result"),
    [
        (DummyMetric(anp.asarray(2)), anp.asarray(True)),
        (2, anp.asarray(True)),
        (2.0, anp.asarray(True)),
        (anp.asarray(2), anp.asarray(True)),
    ],
)
def test_metrics_eq(second_operand, expected_result):
    """Test that `eq` operator works and returns an operator metric."""
    first_metric = DummyMetric(
        anp.asarray(
            2,
            dtype=anp.float32 if isinstance(second_operand, float) else None,
        ),
    )

    final_eq = first_metric == second_operand
    assert isinstance(final_eq, OperatorMetric)
    final_eq.update(anp.asarray(0))  # dummy value to get array namespace
    assert anp.all(expected_result == final_eq.compute())


@pytest.mark.parametrize(
    ("second_operand", "expected_result"),
    [
        (DummyMetric(anp.asarray(2)), anp.asarray(2)),
        (2, anp.asarray(2)),
        (2.0, anp.asarray(2.0)),
        (anp.asarray(2), anp.asarray(2)),
    ],
)
def test_metrics_floordiv(second_operand, expected_result):
    """Test that `floordiv` operator works and returns an operator metric."""
    first_metric = DummyMetric(
        anp.asarray(
            5,  # Python scalars can only be promoted with floating-point arrays
            dtype=anp.float32 if isinstance(second_operand, float) else None,
        ),
    )

    final_floordiv = first_metric // second_operand
    assert isinstance(final_floordiv, OperatorMetric)
    final_floordiv.update(anp.asarray(0))  # dummy value to get array namespace

    assert np.allclose(expected_result, final_floordiv.compute())


@pytest.mark.parametrize(
    ("second_operand", "expected_result"),
    [
        (DummyMetric(anp.asarray(2)), anp.asarray(True)),
        (2, anp.asarray(True)),
        (2.0, anp.asarray(True)),
        (anp.asarray(2), anp.asarray(True)),
    ],
)
def test_metrics_ge(second_operand, expected_result):
    """Test that `ge` operator works and returns an operator metric."""
    first_metric = DummyMetric(
        anp.asarray(
            5,  # Python scalars can only be promoted with floating-point arrays
            dtype=anp.float32 if isinstance(second_operand, float) else None,
        ),
    )

    final_ge = first_metric >= second_operand
    assert isinstance(final_ge, OperatorMetric)
    final_ge.update(anp.asarray(0))  # dummy value to get array namespace

    assert anp.all(expected_result == final_ge.compute())


@pytest.mark.parametrize(
    ("second_operand", "expected_result"),
    [
        (DummyMetric(2), anp.asarray(True)),
        (2, anp.asarray(True)),
        (2.0, anp.asarray(True)),
        (anp.asarray(2), anp.asarray(True)),
    ],
)
def test_metrics_gt(second_operand, expected_result):
    """Test that `gt` operator works and returns an operator metric."""
    first_metric = DummyMetric(
        anp.asarray(
            5,  # Python scalars can only be promoted with floating-point arrays
            dtype=anp.float32 if isinstance(second_operand, float) else None,
        ),
    )

    final_gt = first_metric > second_operand
    assert isinstance(final_gt, OperatorMetric)
    final_gt.update(anp.asarray(0))  # dummy value to get array namespace

    assert anp.all(expected_result == final_gt.compute())


def test_metrics_invert():
    """Test that `invert` operator works and returns an operator metric."""
    first_metric = DummyMetric(anp.asarray(1))

    final_inverse = ~first_metric
    assert isinstance(final_inverse, OperatorMetric)
    final_inverse.update(anp.asarray(0))  # dummy value to get array namespace
    assert np.allclose(anp.asarray(-2), final_inverse.compute())


@pytest.mark.parametrize(
    ("second_operand", "expected_result"),
    [
        (DummyMetric(2), anp.asarray(False)),
        (2, anp.asarray(False)),
        (2.0, anp.asarray(False)),
        (anp.asarray(2), anp.asarray(False)),
    ],
)
def test_metrics_le(second_operand, expected_result):
    """Test that `le` operator works and returns an operator metric."""
    first_metric = DummyMetric(
        anp.asarray(
            5,  # Python scalars can only be promoted with floating-point arrays
            dtype=anp.float32 if isinstance(second_operand, float) else None,
        ),
    )

    final_le = first_metric <= second_operand
    assert isinstance(final_le, OperatorMetric)
    final_le.update(anp.asarray(0))  # dummy value to get array namespace

    assert anp.all(expected_result == final_le.compute())


@pytest.mark.parametrize(
    ("second_operand", "expected_result"),
    [
        (DummyMetric(2), anp.asarray(False)),
        (2, anp.asarray(False)),
        (2.0, anp.asarray(False)),
        (anp.asarray(2), anp.asarray(False)),
    ],
)
def test_metrics_lt(second_operand, expected_result):
    """Test that `lt` operator works and returns an operator metric."""
    first_metric = DummyMetric(
        anp.asarray(
            5,  # Python scalars can only be promoted with floating-point arrays
            dtype=anp.float32 if isinstance(second_operand, float) else None,
        ),
    )

    final_lt = first_metric < second_operand
    assert isinstance(final_lt, OperatorMetric)
    final_lt.update(anp.asarray(0))  # dummy value to get array namespace

    assert anp.all(expected_result == final_lt.compute())


@pytest.mark.parametrize(
    ("second_operand", "expected_result"),
    [
        (DummyMetric(anp.asarray([2, 2, 2])), anp.asarray(12)),
        (anp.asarray([2, 2, 2]), anp.asarray(12)),
    ],
)
def test_metrics_matmul(second_operand, expected_result):
    """Test that `matmul` operator works and returns an operator metric."""
    first_metric = DummyMetric(anp.asarray([2, 2, 2]))

    final_matmul = first_metric @ second_operand
    assert isinstance(final_matmul, OperatorMetric)
    final_matmul.update(anp.asarray(0))  # dummy value to get array namespace

    assert np.allclose(expected_result, final_matmul.compute())


@pytest.mark.parametrize(
    ("second_operand", "expected_result"),
    [
        (DummyMetric(2), anp.asarray(1)),
        (2, anp.asarray(1)),
        (2.0, anp.asarray(1)),
        (anp.asarray(2), anp.asarray(1)),
    ],
)
def test_metrics_mod(second_operand, expected_result):
    """Test that `mod` operator works and returns an operator metric."""
    first_metric = DummyMetric(
        anp.asarray(
            5,  # Python scalars can only be promoted with floating-point arrays
            dtype=anp.float32 if isinstance(second_operand, float) else None,
        ),
    )

    final_mod = first_metric % second_operand
    assert isinstance(final_mod, OperatorMetric)
    final_mod.update(anp.asarray(0))  # dummy value to get array namespace

    assert np.allclose(expected_result, final_mod.compute())


@pytest.mark.parametrize(
    ("second_operand", "expected_result"),
    [
        (DummyMetric(2), anp.asarray(4)),
        (2, anp.asarray(4)),
        (2.0, anp.asarray(4.0)),
        pytest.param(anp.asarray(2), anp.asarray(4)),
    ],
)
def test_metrics_mul(second_operand, expected_result):
    """Test that `mul` operator works and returns an operator metric."""
    first_metric = DummyMetric(
        anp.asarray(
            2,  # Python scalars can only be promoted with floating-point arrays
            dtype=anp.float32 if isinstance(second_operand, float) else None,
        ),
    )

    final_mul = first_metric * second_operand
    assert isinstance(final_mul, OperatorMetric)
    final_mul.update(anp.asarray(0))  # dummy value to get array namespace
    assert np.allclose(expected_result, final_mul.compute())

    if not isinstance(second_operand, DummyMetric):
        with pytest.raises(TypeError, match="unsupported operand type.*"):
            final_rmul = second_operand * first_metric  # type: ignore
    else:
        final_rmul = second_operand * first_metric
        assert isinstance(final_rmul, OperatorMetric)
        final_rmul.update(anp.asarray(0))  # dummy value to get array namespace
        assert np.allclose(expected_result, final_rmul.compute())


@pytest.mark.parametrize(
    ("second_operand", "expected_result"),
    [
        (DummyMetric(anp.asarray(2)), anp.asarray(False)),
        (2, anp.asarray(False)),
        (2.0, anp.asarray(False)),
        (anp.asarray(2), anp.asarray(False)),
    ],
)
def test_metrics_ne(second_operand, expected_result):
    """Test that `!=` operator works and returns an operator metric."""
    first_metric = DummyMetric(
        anp.asarray(
            2,  # Python scalars can only be promoted with floating-point arrays
            dtype=anp.float32 if isinstance(second_operand, float) else None,
        ),
    )

    final_ne = first_metric != second_operand
    assert isinstance(final_ne, OperatorMetric)
    final_ne.update(anp.asarray(0))  # dummy value to get array namespace

    assert anp.all(expected_result == final_ne.compute())


def test_metrics_neg():
    """Test that `neg` operator works and returns an operator metric."""
    first_metric = DummyMetric(anp.asarray(1))

    final_neg = -first_metric
    assert isinstance(final_neg, OperatorMetric)
    final_neg.update(anp.asarray(0))  # dummy value to get array namespace
    assert np.allclose(anp.asarray(-1), final_neg.compute())


@pytest.mark.parametrize(
    ("second_operand", "expected_result"),
    [
        (DummyMetric(anp.asarray([1, 0, 3])), anp.asarray([-1, -2, 3])),
        (anp.asarray([1, 0, 3]), anp.asarray([-1, -2, 3])),
    ],
)
def test_metrics_or(second_operand, expected_result):
    """Test that `or` operator works and returns an operator metric."""
    first_metric = DummyMetric(anp.asarray([-1, -2, 3]))

    final_or = first_metric | second_operand
    assert isinstance(final_or, OperatorMetric)
    final_or.update(anp.asarray(0))  # dummy value to get array namespace
    assert np.allclose(expected_result, final_or.compute())

    if not isinstance(second_operand, DummyMetric):
        with pytest.raises(TypeError, match="unsupported operand type.*"):
            final_ror = second_operand | first_metric  # type: ignore
    else:
        final_ror = second_operand | first_metric
        assert isinstance(final_ror, OperatorMetric)
        final_ror.update(anp.asarray(0))  # dummy value to get array namespace
        assert np.allclose(expected_result, final_ror.compute())


def test_metrics_pos():
    """Test that `pos` operator works and returns an operator metric."""
    first_metric = DummyMetric(anp.asarray(-1))

    final_pos = +first_metric
    assert isinstance(final_pos, OperatorMetric)
    final_pos.update(np.asanyarray(0))  # dummy value to get array namespace
    assert np.allclose(anp.asarray(1), final_pos.compute())


@pytest.mark.parametrize(
    ("second_operand", "expected_result"),
    [
        (DummyMetric(2), anp.asarray(4)),
        (2, anp.asarray(4)),
        (2.0, anp.asarray(4.0)),
        (anp.asarray(2), anp.asarray(4)),
    ],
)
def test_metrics_pow(second_operand, expected_result):
    """Test that `pow` operator works and returns an operator metric."""
    first_metric = DummyMetric(
        anp.asarray(
            2,
            dtype=anp.float32 if isinstance(second_operand, float) else None,
        ),
    )

    final_pow = first_metric**second_operand

    assert isinstance(final_pow, OperatorMetric)

    final_pow.update(np.asarray(0))  # dummy value to get array namespace
    assert np.allclose(expected_result, final_pow.compute())


@pytest.mark.parametrize(
    ("second_operand", "expected_result"),
    [
        (DummyMetric(2), anp.asarray(1)),
        (2, anp.asarray(1)),
        (2.0, anp.asarray(1.0)),
        (anp.asarray(2), anp.asarray(1)),
    ],
)
def test_metrics_sub(second_operand, expected_result):
    """Test that `sub` operator works and returns an operator metric."""
    first_metric = DummyMetric(
        anp.asarray(
            3,
            dtype=anp.float32 if isinstance(second_operand, float) else None,
        ),
    )

    final_sub = first_metric - second_operand

    assert isinstance(final_sub, OperatorMetric)
    final_sub.update(anp.asarray(0))  # dummy value to get array namespace
    assert np.allclose(expected_result, final_sub.compute())


@pytest.mark.parametrize(
    ("second_operand", "expected_result"),
    [
        (DummyMetric(anp.asarray(3.0)), anp.asarray(2.0)),
        (3, anp.asarray(2.0)),
        (3.0, anp.asarray(2.0)),
        (anp.asarray(3.0), anp.asarray(2.0)),
    ],
)
def test_metrics_truediv(second_operand, expected_result):
    """Test that `truediv` operator works and returns an operator metric."""
    first_metric = DummyMetric(anp.asarray(6.0))  # only floating-point arrays

    final_truediv = first_metric / second_operand

    assert isinstance(final_truediv, OperatorMetric)
    final_truediv.update(anp.asarray(0))  # dummy value to get array namespace
    assert np.allclose(expected_result, final_truediv.compute())


@pytest.mark.parametrize(
    ("second_operand", "expected_result"),
    [
        (DummyMetric([1, 0, 3]), anp.asarray([-2, -2, 0])),
        (anp.asarray([1, 0, 3]), anp.asarray([-2, -2, 0])),
    ],
)
def test_metrics_xor(second_operand, expected_result):
    """Test that `xor` operator works and returns an operator metric."""
    first_metric = DummyMetric([-1, -2, 3])

    final_xor = first_metric ^ second_operand
    assert isinstance(final_xor, OperatorMetric)
    final_xor.update(anp.asarray(0))  # dummy value to get array namespace
    assert np.allclose(expected_result, final_xor.compute())

    if not isinstance(second_operand, DummyMetric):
        with pytest.raises(TypeError, match="unsupported operand type.*"):
            final_rxor = second_operand ^ first_metric  # type: ignore
    else:
        final_rxor = second_operand ^ first_metric
        assert isinstance(final_rxor, OperatorMetric)
        final_rxor.update(anp.asarray(0))  # dummy value to get array namespace
        assert np.allclose(expected_result, final_rxor.compute())


def test_operator_metrics_update():
    """Test update method for operator metrics."""
    compos = DummyMetric(anp.asarray(5)) + DummyMetric(anp.asarray(4))

    assert isinstance(compos, OperatorMetric)
    compos.update(anp.asarray(0))  # dummy value to get array namespace
    compos.update(anp.asarray(0))  # dummy value to get array namespace
    compos.update(anp.asarray(0))  # dummy value to get array namespace

    assert isinstance(compos.metric_a, DummyMetric)
    assert isinstance(compos.metric_b, DummyMetric)

    assert compos.metric_a._num_updates == 3  # type: ignore
    assert compos.metric_b._num_updates == 3  # type: ignore
