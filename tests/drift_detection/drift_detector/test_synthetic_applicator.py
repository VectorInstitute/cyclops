'''unit tests for Synthetic Applicator'''

import numpy as np
import pytest

from drift_detection.drift_detector.synthetic_applicator import SyntheticShiftApplicator

@pytest.fixture
def X():
    '''create a test input'''
    X = np.random.rand(100, 10)
    return X

# Test all shift types
for shift_type in SyntheticShiftApplicator.shift_types:
    def test_SyntheticShiftApplicator(X, shift_type=shift_type):
        '''test SyntheticShiftApplicator'''
        applicator = SyntheticShiftApplicator(shift_type)
        X_s, X_t = applicator.apply_shift(X)
        assert X_t.shape == X.shape

