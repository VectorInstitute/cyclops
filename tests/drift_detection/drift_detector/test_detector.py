'''integration tests for Reductor module'''

import numpy as np
import torch
import pytest


from drift_detection.drift_detector.reductor import Reductor
from drift_detection.drift_detector.tester import DCTester, TSTester
from drift_detection.drift_detector.detector import Detector

@pytest.fixture
def source_target():
    '''create a test input'''
    X_source = np.random.rand(100, 10)
    X_target = np.random.rand(100, 10)
    return X_source, X_target

# test detector with pca reductor and mmd tester

@pytest.mark.integtest
def test_detector_pca_mmd(source_target):
    '''test Detector'''
    reductor = Reductor("PCA", n_components=2)
    tester = TSTester("mmd")
    detector = Detector(reductor, tester)
    X_source, X_target = source_target
    detector.fit(X_source)
    p_val = detector.test_shift(X_target)["p_val"]
    assert p_val >= 0 and p_val <= 1


