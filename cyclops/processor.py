"""Processor API."""

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
from cyclops.processors.impute import impute_features, Imputer


def process_diagnoses(dataframe: pd.DataFrame) -> pd.DataFrame:
    """Process diagnoses data (codes) into trajectories, and create features.
    
    Parameters
    ----------
    dataframe: pd.DataFrame
        Input DataFrame with diagnoses code data.
    
    Returns
    -------
    pd.DataFrame
        Diagnoses codes processed into trajectory features.
    
    """
    diagnoses_features = None
    if check_must_have_columns(dataframe, [ENCOUNTER_ID, DIAGNOSIS_CODE]):
        diagnoses_data = gather_columns(
            dataframe, [ENCOUNTER_ID, DIAGNOSIS_CODE]
        )
        diagnoses_features = group_diagnosis_codes_to_trajectories(
            diagnoses_data
        )
    return diagnoses_features


def featurize(
    static_data: Union[list, pd.DataFrame] = None,
    temporal_data: Union[list, pd.DataFrame] = None,
    aggregator: Aggregator = Aggregator(),
    static_imputer: Imputer = Imputer(),
    temporal_imputer: Imputer = Imputer(),
) -> FeatureHandler:
    """Process and create features from raw queried data.
    
    - User inputs static and temporal data in the form of raw dataframes.
    - If diagnoses codes are present in the static data, they are processed
    first.
    - Static data is then processed into static features, and added to
    the FeatureHandler container.
    - Temporal data is then processed into temporal features, and added to
    the FeatureHandler container. For the aggregation of temporal data,
    the user passed in options using `cyclops.processor.Aggregator`.
    - Imputation is called as a public method on the FeatureHandler, which then
    applies imputation on both static and temporal features. For
    imputation, options are passed using `cyclops.processor.Imputer`.

    Parameters
    ----------
    data: pandas.DataFrame or list of pandas.DataFrame
    aggregator: cyclops.processor.Aggregator, optional
        Aggregation options.
    static_imputer: cyclops.processor.Imputer, optional
        Imputation options for static data.
    temporal_imputer: cyclops.processor.Imputer, optional
        Imputation options for temporal data.

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
            diagnoses_features = process_diagnoses(dataframe)
            if diagnoses_features is not None:
                dataframe.drop(DIAGNOSIS_CODE, axis=1, inplace=True)
                feature_handler.add_features(diagnoses_features)
                
            static_features = gather_static_features(dataframe)
            feature_handler.add_features(static_features)

    if temporal_data:
        for dataframe in temporal_data:
            dataframe = clean_events(dataframe)
            temporal_features = gather_event_features(dataframe, aggregator=aggregator)
            feature_handler.add_features(temporal_features)
            
    feature_handler.impute_features(static_imputer=static_imputer, temporal_imputer=temporal_imputer)

    return feature_handler
