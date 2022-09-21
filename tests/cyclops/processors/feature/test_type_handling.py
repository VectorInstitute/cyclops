"""Test type handling fns."""

import pandas as pd
import pytest

from cyclops.processors.constants import BINARY, NUMERIC, ORDINAL, STRING
from cyclops.processors.feature.type_handling import get_unique, valid_feature_type


def test_get_unique():
    """Test get_unique fn."""
    assert (get_unique(pd.Series([1, 3, 5])) == [1, 3, 5]).all()
    assert (get_unique(pd.Series([1, 1, 5])) == [1, 5]).all()


def test_valid_feature_type():
    """Test valid_feature_type fn."""
    assert valid_feature_type(NUMERIC)
    assert valid_feature_type(BINARY)
    assert valid_feature_type(STRING)
    assert valid_feature_type(ORDINAL)
    with pytest.raises(ValueError):
        valid_feature_type("donkey")
    assert not valid_feature_type("donkey", raise_error=False)
