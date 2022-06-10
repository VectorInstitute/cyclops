"""Processor API."""

from typing import Optional, Union

import pandas as pd

from cyclops.feature_handler import FeatureHandler
from cyclops.processors.aggregate import Aggregator
from cyclops.processors.column_names import DIAGNOSIS_CODE, ENCOUNTER_ID
from cyclops.processors.diagnoses import group_diagnosis_codes_to_trajectories
from cyclops.processors.events import normalize_events
from cyclops.processors.statics import compute_statics
from cyclops.processors.util import gather_columns, has_columns


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
    if has_columns(dataframe, [ENCOUNTER_ID, DIAGNOSIS_CODE]):
        diagnoses_data = gather_columns(dataframe, [ENCOUNTER_ID, DIAGNOSIS_CODE])
        diagnoses_features = group_diagnosis_codes_to_trajectories(diagnoses_data)

    return diagnoses_features


def run_data_pipeline(  # pylint: disable=too-many-arguments
    static_data: Union[list, pd.DataFrame] = None,
    temporal_data: Union[list, pd.DataFrame] = None,
    aggregator: Aggregator = Aggregator(),
    reference_cols: Optional[list] = None,
) -> FeatureHandler:
    """Run data pipeline in steps just for development purposes (not public API fn.).

    Parameters
    ----------
    data: pandas.DataFrame or list of pandas.DataFrame
    aggregator: cyclops.processor.Aggregator, optional
        Aggregation options.
    static_imputer: cyclops.processor.Imputer, optional
        Imputation options for static data.
    temporal_imputer: cyclops.processor.Imputer, optional
        Imputation options for temporal data.
    reference_cols: list, optional
        Columns from patient statics data to use as reference columns in the
        FeatureHandler, e.g. 'hospital_id', 'admit_timestamp'. These columns
        will not be added as features, instead be stored in
        feature_handler.reference.

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

            feature_handler.add_features(
                compute_statics(dataframe),
                reference_cols=reference_cols,
            )

    aggregated_events = []
    if temporal_data:
        for dataframe in temporal_data:
            dataframe = normalize_events(dataframe)
            aggregated_events.append(aggregator(dataframe))

    return feature_handler, aggregated_events
