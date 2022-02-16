"""Labs processor module."""

import pandas as pd

from cyclops.processors.base import Processor


class LabsProcessor(Processor):
    """Labs processor class."""

    def __init__(self, data: pd.DataFrame, must_have_features: list):
        """Instantiate."""
        super().__init__()
