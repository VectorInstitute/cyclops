"""Feature shift detector.

This module implements the feature shift detector (FSD) algorithm described in
the paper below.

References
----------
@inproceedings{kulinski2020feature,
author = {Kulinski, Sean and Bagchi, Saurabh and Inouye, David I.},
booktitle = {Neural Information Processing Systems (NeurIPS)},
title = {Feature Shift Detection: Localizing Which Features Have Shifted via
         Conditional Distribution Tests},
year = {2020}
}

"""

from .featureshiftdetector import FeatureShiftDetector

__all__ = ["FeatureShiftDetector"]
