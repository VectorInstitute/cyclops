"""unit tests for Synthetic Applicator."""

import numpy as np
import pytest

from cyclops.monitor.synthetic_applicator import SyntheticShiftApplicator


@pytest.fixture(name="X")
def fixture_x():
    """Create a test input."""
    x = np.random.rand(100, 10)
    return x


def test_sa_gn_shift(X):
    """Test SyntheticShiftApplicator gaussian shift."""
    applicator = SyntheticShiftApplicator("gn_shift")
    X_s = applicator.apply_shift(X)
    assert X_s.shape[-1] == X.shape[-1]
