"""Clinical Shift Applicator module."""

from typing import Union

import numpy as np
import pandas as pd

from .utils import get_args


class ClinicalShiftApplicator:
    """The ClinicalShiftApplicator class is used induce synthetic clinical shifts.

    Parameters
    ----------
    shift_type: str
        Method used to induce shift in data.
        Options include: "seasonal", "hospital_type".

    """

    def __init__(self, shift_type: str, source, target):

        self.shift_type = shift_type

        self.method_args = {"source": source, "target": target}

        self.shift_types = {
            "time": self.time,
            "month": self.month,
            "hospital_type": self.hospital_type,
        }

        if self.shift_type not in self.shift_types:
            raise ValueError(f"Shift type {self.shift_type} not supported. ")

    def apply_shift(self, X, metadata: pd.DataFrame, metadata_mapping: dict):
        """apply_shift.

        Returns
        -------
        X: pd.DataFrame
            Data to apply shift to.
        y:
            Outcome labels.
        metadata: pd.DataFrame
            Dataframe containing admin variables to filter on
            (e.g. "hospital_id", "admit_timestamp").

        """
        # list(X.index.get_level_values(0).unique())
        # X = X[np.in1d(X.index.get_level_values(0), metadata["encounter_id"])]

        X_s, X_t = self.shift_types[self.shift_type](
            X,
            metadata,
            metadata_mapping,
            **get_args(self.shift_types[self.shift_type], self.method_args),
        )

        # list(X.index.get_level_values(0).unique())

        return (X_s.values, X_t.values)

    def time(
        self,
        X: Union[np.ndarray, pd.DataFrame],
        metadata: pd.DataFrame,
        metadata_mapping: dict,
        source,
        target,
    ):
        """Shift in time.

        Parameters
        ----------
        X: pd.DataFrame
            Data to apply shift to.
        metadata: pd.DataFrame
            Dataframe containing admin variables to filter on
            (e.g. "hospital_id", "admit_timestamp").
        source: list[datetime.date]
            Start and end of source data.
        target: list[datetime.date]
            Start and end of target data.
        encounter_id: str
            Column name for encounter ids.
        admit_timestamp: str
            Column name for admission timestamps.

        """
        if isinstance(X, np.ndarray):
            X = pd.DataFrame(X)
        encounter_id = metadata_mapping["id"]
        admit_timestamp = metadata_mapping["timestamp"]

        ids_source = metadata.loc[
            (
                (metadata[admit_timestamp].dt.date > pd.to_datetime(source[0]).date())
                & (
                    metadata[admit_timestamp].dt.date < pd.to_datetime(source[1]).date()
                ),
            ),
            encounter_id,
        ]
        ids_target = metadata.loc[
            (
                (metadata[admit_timestamp].dt.date > pd.to_datetime(target[0]).date())
                & (
                    metadata[admit_timestamp].dt.date < pd.to_datetime(target[1]).date()
                ),
            ),
            encounter_id,
        ]
        X_source = X.loc[X.index.get_level_values(0).isin(ids_source)]
        X_target = X.loc[X.index.get_level_values(0).isin(ids_target)]
        return (X_source, X_target)

    def month(
        self,
        X: Union[np.ndarray, pd.DataFrame],
        metadata: pd.DataFrame,
        metadata_mapping: dict,
        source,
        target,
    ):
        """Shift for selection of months.

        Parameters
        ----------
        X: pd.DataFrame
            Data to apply shift to.
        metadata: pd.DataFrame
            Dataframe containing admin variables to filter on
            (e.g. "hospital_id", "admit_timestamp").
        encounter_id: str
            Column name for encounter ids.
        admit_timestamp: str
            Column name for admission timestamps.

        """
        if isinstance(X, np.ndarray):
            X = pd.DataFrame(X)
        encounter_id = metadata_mapping["id"]
        admit_timestamp = metadata_mapping["timestamp"]

        ids_source = metadata.loc[
            ((metadata[admit_timestamp].dt.month.isin(source))),
            encounter_id,
        ]
        ids_target = metadata.loc[
            ((metadata[admit_timestamp].dt.month.isin(target))),
            encounter_id,
        ]
        X_source = X.loc[X.index.get_level_values(0).isin(ids_source)]
        X_target = X.loc[X.index.get_level_values(0).isin(ids_target)]
        return (X_source, X_target)

    def hospital_type(
        self,
        X: Union[np.ndarray, pd.DataFrame],
        metadata: pd.DataFrame,
        metadata_mapping: dict,
        source,
        target,
    ):
        """Shift against hospital type.

        Parameters
        ----------
        X: pd.DataFrame
            Data to apply shift to.
        metadata: pd.DataFrame
            Dataframe containing admin variables to filter on
            (e.g. "hospital_id", "admit_timestamp").
        source_hospitals: list
            List of hospitals for source data.
        target_hospitals: list
            List of hospitals for target data.
        encounter_id: str
            Column name for encounter ids.
        hospital_id: str
            Column name for hospital ids.

        """
        if isinstance(X, np.ndarray):
            X = pd.DataFrame(X)
        encounter_id = metadata_mapping["id"]
        hospital_id = metadata_mapping["hospital_id"]

        ids_source = metadata.loc[
            ((metadata[hospital_id].isin(source))),
            encounter_id,
        ]
        ids_target = metadata.loc[
            ((metadata[hospital_id].isin(target))),
            encounter_id,
        ]
        X_source = X.loc[X.index.get_level_values(0).isin(ids_source)]
        X_target = X.loc[X.index.get_level_values(0).isin(ids_target)]
        return (X_source, X_target)
