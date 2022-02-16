"""Base processor module."""

import pandas as pd


class Processor:
    """Base processor.

    Attributes
    ----------
    data: pandas.DataFrame
        Dataframe with raw features.
    must_have_features: list
        List of column names of features that must be present in data.
    """

    def __init__(self, data: pd.DataFrame, must_have_features: list):
        """Instantiate.

        Parameters
        ----------
        data: pandas.DataFrame
            Dataframe with raw features.
        must_have_features: list
            List of column names of features that must be present in data.
        """
        assert type(data) is pd.DataFrame
        assert type(must_have_features) is list
        self.data = data
        self.must_have_features = must_have_features

        # Checks if feature columns are present in dataframe.
        self._check_must_have_features()

    def _check_must_have_features(self):
        """Check if data has minimum feature columns for processing."""
        column_names = list(self.data.columns)
        for column_name in self.must_have_features:
            assert column_name in column_names
