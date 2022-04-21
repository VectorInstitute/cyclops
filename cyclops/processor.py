"""Processor API."""

from dataclasses import dataclass
from typing import Union

import pandas as pd

from cyclops.processors.aggregate import (
    Aggregator,
    gather_event_features,
    gather_static_features,
)
from cyclops.processors.column_names import DIAGNOSIS_CODE, ENCOUNTER_ID
from cyclops.processors.diagnoses import group_diagnosis_codes_to_trajectories
from cyclops.processors.events import clean_events
from cyclops.processors.feature_handler import FeatureHandler
from cyclops.processors.utils import check_must_have_columns, gather_columns


@dataclass
class Imputer:
    """Imputation options."""

    strategy: str = "none"


def featurize(
    static_data: Union[list, pd.DataFrame] = None,
    temporal_data: Union[list, pd.DataFrame] = None,
    aggregator: Aggregator = Aggregator(),  # pylint:disable=unused-argument
    imputer: Imputer = Imputer(),  # pylint:disable=unused-argument
) -> FeatureHandler:
    """Process and create features from raw queried data.

    Parameters
    ----------
    data: pandas.DataFrame or list of pandas.DataFrame
    aggregator: cyclops.processor.Aggregator, optional
        Aggregation options.
    imputer: cyclops.processor.Imputer, optional
        Imputation options.

    Returns
    -------
    cyclops.processors.feature_handler.FeatureHandler
        Feature handler object, which is a container for processed features.

    """
    feature_handler = FeatureHandler()

    if isinstance(static_data, pd.DataFrame):
        static_data = [static_data]
    if isinstance(temporal_data, pd.DataFrame):
        temporal_data = [temporal_data]

    if static_data:
        for dataframe in static_data:
            if check_must_have_columns(dataframe, [ENCOUNTER_ID, DIAGNOSIS_CODE]):
                diagnoses_data = gather_columns(
                    dataframe, [ENCOUNTER_ID, DIAGNOSIS_CODE]
                )
                diagnoses_features = group_diagnosis_codes_to_trajectories(
                    diagnoses_data
                )
                feature_handler.add_features(diagnoses_features)
                dataframe.drop(DIAGNOSIS_CODE, axis=1, inplace=True)
            static_features = gather_static_features(dataframe)
            feature_handler.add_features(static_features)

    if temporal_data:
        for dataframe in temporal_data:
            dataframe = clean_events(dataframe)
            temporal_features = gather_event_features(dataframe, aggregator=aggregator)
            feature_handler.add_features(temporal_features)

    return feature_handler
