'''unit tests for Clinical Applicator'''

import numpy as np
import pytest

from drift_detection.drift_detector.clinical_applicator import ClinicalShiftApplicator

@pytest.fixture
def X():
    '''create a test input'''
    X = np.random.rand(100, 10)
    return X
# Test all shift types
for shift_type in ClinicalShiftApplicator.shift_types:
    def test_ClinicalShiftApplicator(X, shift_type=shift_type):
        '''test ClinicalShiftApplicator'''
        applicator = ClinicalShiftApplicator(shift_type)
        X_s, X_t = applicator.apply_shift(X)
        assert X_t.shape == X.shape

