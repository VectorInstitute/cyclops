"""Test query util fns."""

import numpy as np

from cyclops.query.util import to_list


def test_to_list():
    """Test to_list fn."""
    assert to_list("kobe") == ["kobe"]
    assert to_list(np.array([1, 2])) == [1, 2]
