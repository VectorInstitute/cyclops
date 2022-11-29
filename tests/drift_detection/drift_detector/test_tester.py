'''unit tests for Tester module'''


import numpy as np
import pytest
from drift_detection.drift_detector.tester import DCTester, TSTester

@pytest.fixture
def source_target():
    '''create a test input'''
    X_source = np.random.rand(100, 10)
    X_target = np.random.rand(100, 10)
    return X_source, X_target

#list all TStester test methods

# iterate and test TSTester for all available methods
for method in TSTester.get_available_test_methods():
    def test_TSTester(source_target, method=method):
        '''test TSTester'''
        tester = TSTester(method)
        X_source, X_target = source_target
        tester.fit(X_source)
        p_val = tester.test_shift(X_target)[0]
        assert p_val >= 0 and p_val <= 1

# iterate and test DCTester for all available methods
for method in DCTester.get_available_test_methods():
    def test_DCTester(method=method):
        '''test DCTester'''
        tester = DCTester(method)
        X_source, X_target = source_target
        tester.fit(X_source)
        p_val = tester.test_shift(X_target)[0]
        assert p_val >= 0 and p_val <= 1


