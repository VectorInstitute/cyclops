"""Test feature module."""

import unittest

import numpy as np
import pandas as pd
import pytest

from cyclops.processors.column_names import ENCOUNTER_ID
from cyclops.processors.feature.feature import Features


class TestFeatures(unittest.TestCase):
    """Test Features class."""

    def setUp(self):
        """Create test features to test."""
        self.test_data = pd.DataFrame(
            {"feat_A": [False, True], "feat_B": [1.2, 3], ENCOUNTER_ID: [101, 201]}
        )
        self.features = Features(
            data=self.test_data, features=["feat_A", "feat_B"], by=ENCOUNTER_ID
        )

    def test_slice(self):
        """Test slice method."""
        with pytest.raises(ValueError):
            self.features.slice("feat_donkey", [10])

        sliced_by_indices = self.features.slice("feat_B", 3, replace=True)
        assert np.array_equal(sliced_by_indices, np.array([201]))
        assert len(self.features.data) == 1