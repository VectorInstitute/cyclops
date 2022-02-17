"""Base processor module."""

import pandas as pd


class Processor:
    """Base processor.

    Attributes
    ----------
    data: pandas.DataFrame
        Dataframe with raw features.
    must_have_columns: list
        List of column names that must be present in data.
    """

    def __init__(self, data: pd.DataFrame, must_have_columns: list):
        """Instantiate.

        Parameters
        ----------
        data: pandas.DataFrame
            Dataframe with raw features.
        must_have_columns: list
            List of column names of features that must be present in data.
        """
        assert type(data) is pd.DataFrame
        assert type(must_have_columns) is list
        self.data = data.copy()
        self.must_have_columns = must_have_columns

        self._check_must_have_columns()
        self._gather_required_columns()

    def _check_must_have_columns(self):
        """Check if data has required columns for processing."""
        column_names = list(self.data.columns)
        for column_name in self.must_have_columns:
            assert column_name in column_names

    def _gather_required_columns(self):
        """Gather only required columns and discard rest."""
        self.data = self.data[self.must_have_columns].copy()
