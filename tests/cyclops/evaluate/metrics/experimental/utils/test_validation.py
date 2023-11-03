"""Test utility functions for validating input arrays."""
import numpy as np
import numpy.array_api as anp

from cyclops.evaluate.metrics.experimental.utils.validation import (
    is_floating_point,
    is_numeric,
)
from cyclops.utils.optional import import_optional_module


def test_is_floating_point():
    """Test `is_floating_point`."""
    x = anp.asarray([1, 2, 3], dtype=anp.float32)
    assert is_floating_point(x)

    x = anp.asarray([1, 2, 3], dtype=anp.float64)
    assert is_floating_point(x)

    torch = import_optional_module("torch")
    if torch is not None:
        x = torch.tensor([1, 2, 3], dtype=torch.float16)
        assert is_floating_point(x)

        x = torch.tensor([1, 2, 3], dtype=torch.bfloat16)
        assert is_floating_point(x)

    x = anp.asarray([1, 2, 3], dtype=anp.int32)
    assert not is_floating_point(x)

    x = np.zeros((3, 3), dtype=np.bool_)
    assert not is_floating_point(x)


def test_is_numeric():
    """Test `is_numeric`."""
    numeric_dtypes = [
        anp.int8,
        anp.int16,
        anp.int32,
        anp.int64,
        anp.uint8,
        anp.uint16,
        anp.uint32,
        anp.uint64,
        anp.float32,
        anp.float64,
    ]

    for dtype in numeric_dtypes:
        x = anp.asarray([1, 2, 3], dtype=dtype)
        assert is_numeric(x)
