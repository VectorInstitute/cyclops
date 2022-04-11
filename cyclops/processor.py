"""Processor API."""

from dataclasses import dataclass
from typing import Union

import pandas as pd

from cyclops.processors.admin import AdminProcessor
from cyclops.processors.column_names import (
    ADMIT_TIMESTAMP,
    AGE,
    ENCOUNTER_ID,
    LAB_TEST_NAME,
    LAB_TEST_RESULT_UNIT,
    LAB_TEST_RESULT_VALUE,
    LAB_TEST_TIMESTAMP,
    REFERENCE_RANGE,
    SEX,
    VITAL_MEASUREMENT_NAME,
    VITAL_MEASUREMENT_TIMESTAMP,
    VITAL_MEASUREMENT_VALUE,
)
from cyclops.processors.feature_handler import FeatureHandler
from cyclops.processors.labs import LabsProcessor
from cyclops.processors.vitals import VitalsProcessor


@dataclass
class Aggregator:
    """Aggregator options."""

    strategy: str = "static"
    range_: int = 168
    window: int = 24


@dataclass
class Imputer:
    """Imputation options."""

    strategy: str = "none"


def featurize(
    data: Union[list, pd.DataFrame],
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
    admin_processor = AdminProcessor(data[0], [ENCOUNTER_ID, AGE, SEX])
    admin_features = admin_processor.process()

    labs_processor = LabsProcessor(
        data[0],
        [
            ENCOUNTER_ID,
            ADMIT_TIMESTAMP,
            LAB_TEST_NAME,
            LAB_TEST_TIMESTAMP,
            LAB_TEST_RESULT_VALUE,
            LAB_TEST_RESULT_UNIT,
            REFERENCE_RANGE,
        ],
    )
    labs_features = labs_processor.process()
    vitals_processor = VitalsProcessor(
        data[1],
        [
            ENCOUNTER_ID,
            ADMIT_TIMESTAMP,
            VITAL_MEASUREMENT_NAME,
            VITAL_MEASUREMENT_VALUE,
            VITAL_MEASUREMENT_TIMESTAMP,
            REFERENCE_RANGE,
        ],
    )
    vitals_features = vitals_processor.process()

    feature_handler = FeatureHandler()
    feature_handler.add_features(admin_features)
    feature_handler.add_features(labs_features)
    feature_handler.add_features(vitals_features)

    return feature_handler
