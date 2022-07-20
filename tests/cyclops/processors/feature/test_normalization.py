"""Test normalization."""

import numpy as np
import pandas as pd
import pytest

from cyclops.processors.column_names import (
    ENCOUNTER_ID,
    EVENT_NAME,
    EVENT_VALUE,
    EVENT_VALUE_UNIT,
    SUBJECT_ID,
)
from cyclops.processors.constants import MIN_MAX, STANDARD
from cyclops.processors.feature.normalization import (
    GroupbyNormalizer,
    SklearnNormalizer,
    VectorizedNormalizer,
)


@pytest.fixture
def test_input():
    """Create a test input."""
    cols = [EVENT_NAME, SUBJECT_ID, ENCOUNTER_ID, EVENT_VALUE, EVENT_VALUE_UNIT]
    data = [
        ["Heart Rate", 10449408, 22698294, 99.0, "bpm"],
        ["High risk (>51) interventions", 10449408, 22698294, 1.0, None],
        ["Inspiratory Ratio", 10449408, 22698294, 1.0, None],
        ["Replacement Rate", 10441335, 25373904, 1400.0, "ml/hr"],
        ["Non-Invasive Blood Pressure Alarm - High", 10449408, 22698294, 160.0, "mmHg"],
        ["Arterial Blood Pressure mean", 10449408, 22698294, 73.0, "mmHg"],
        ["Arterial Blood Pressure mean", 10449408, 22698294, 65.0, "mmHg"],
        ["Inspired O2 Fraction", 10449408, 22698294, 60.0, None],
        ["Dialysate Rate", 10449408, 22698294, 600.0, "ml/hr"],
        ["Eye Care", 10449408, 22698294, 1.0, None],
        ["Peak Insp. Pressure", 10449408, 22698294, 30.0, "cmH2O"],
        ["Total PEEP Level", 10449408, 22698294, 10.0, "cmH2O"],
        ["Peak Insp. Pressure", 10449408, 22698294, 29.0, "cmH2O"],
        ["Fspn High", 10449408, 22698294, 15.0, "insp/min"],
        ["Ultrafiltrate Output", 10449408, 22698294, 442.0, "mL"],
        ["Arterial Blood Pressure diastolic", 10449408, 22698294, 56.0, "mmHg"],
        ["22 Gauge placed in outside facility", 10449297, 29981093, 0.0, None],
        ["Non-Invasive Blood Pressure Alarm - High", 10449408, 22698294, 160.0, "mmHg"],
        ["PCA basal rate (mL/hour)", 10445790, 26253687, 0.0, "ml/hr"],
        ["Vti High", 10449408, 22698294, 1.5, "mL"],
        ["Non Invasive Blood Pressure mean", 10445790, 26253687, 67.0, "mmHg"],
        ["Heart Rate", 10449408, 22698294, 80.0, "bpm"],
        ["Heart rate Alarm - High", 10449408, 22698294, 120.0, "bpm"],
        ["Chloride (serum)", 10449408, 22698294, 108.0, "mEq/L"],
        ["Parameters Checked", 10449408, 22698294, 1.0, None],
        ["Arterial Blood Pressure Alarm - High", 10449408, 22698294, 110.0, "mmHg"],
        ["Sodium (serum)", 10449408, 22698294, 138.0, "mEq/L"],
        ["O2 saturation pulseoxymetry", 10449408, 22698294, 95.0, "%"],
        ["Minute Volume Alarm - High", 10449408, 22698294, 17.0, "L/min"],
        ["Filter Pressure", 10449408, 22698294, 183.0, "mmHg"],
        ["Ultrafiltrate Output", 10449408, 22698294, 317.0, "mL"],
        ["Heart Rate", 10449297, 29981093, 60.0, "bpm"],
        ["Current Goal", 10441335, 25373904, 0.0, "mL"],
        ["High risk (>51) interventions", 10449408, 22698294, 1.0, None],
        ["Arterial Blood Pressure diastolic", 10449408, 22698294, 49.0, "mmHg"],
        ["Respiratory Rate", 10449408, 22698294, 17.0, "insp/min"],
        ["Return Pressure", 10441335, 25373904, 32.0, "mmHg"],
        ["Respiratory Rate", 10449408, 22698294, 21.0, "insp/min"],
        ["Apnea Interval", 10449408, 22698294, 20.0, "sec"],
        ["Resp Alarm - High", 10449408, 22698294, 35.0, "insp/min"],
        ["PCA 1 hour limit", 10445790, 26253687, 2.5, None],
        ["HCO3 (serum)", 10449297, 29981093, 26.0, "mEq/L"],
        ["PBP (Prefilter) Replacement Rate", 10449408, 22698294, 1200.0, "ml/hr"],
        ["Apnea Interval", 10449408, 22698294, 20.0, "sec"],
        ["Fspn High", 10449408, 22698294, 10.0, "insp/min"],
        ["Specific Gravity (urine)", 10449408, 22698294, 1.012, None],
        ["Peak Insp. Pressure", 10449408, 22698294, 28.0, "cmH2O"],
        ["Replacement Rate", 10449408, 22698294, 1400.0, "ml/hr"],
        ["Non Invasive Blood Pressure mean", 10445790, 26253687, 61.0, "mmHg"],
        ["Post Filter Replacement Rate", 10449408, 22698294, 200.0, "ml/hr"],
        ["Arterial Blood Pressure mean", 10449408, 22698294, 69.0, "mmHg"],
        ["PBP (Prefilter) Replacement Rate", 10449408, 22698294, 1200.0, "ml/hr"],
        ["Tidal Volume (spontaneous)", 10449408, 22698294, 949.0, "mL"],
        ["Inspired Gas Temp.", 10449408, 22698294, 37.0, "°C"],
        ["Multi Lumen placed in outside facility", 10449408, 22698294, 0.0, None],
        ["Hourly Patient Fluid Removal", 10449408, 22698294, 330.0, "mL"],
        ["Respiratory Rate (Total)", 10449408, 22698294, 25.0, "insp/min"],
        ["Dialysis Catheter placed in outside facility", 10449408, 22698294, 0.0, None],
        ["PEEP set", 10449408, 22698294, 10.0, "cmH2O"],
        ["Respiratory Rate", 10449408, 22698294, 15.0, "insp/min"],
        ["Temperature Fahrenheit", 10449408, 22698294, 98.6, "°F"],
        ["Heart Rate", 10449297, 29981093, 69.0, "bpm"],
        ["O2 saturation pulseoxymetry", 10449408, 22698294, 100.0, "%"],
        ["Respiratory Rate", 10449408, 22698294, 23.0, "insp/min"],
        ["Self ADL", 10449297, 29981093, 1.0, None],
        ["Citrate (ACD-A)", 10449408, 22698294, 0.0, "ml/hr"],
        ["O2 saturation pulseoxymetry", 10449408, 22698294, 95.0, "%"],
        ["O2 Saturation Pulseoxymetry Alarm - Low", 10449408, 22698294, 92.0, "%"],
        ["Temperature Fahrenheit", 10449408, 22698294, 99.2, "°F"],
        ["Chloride (serum)", 10449408, 22698294, 105.0, "mEq/L"],
        ["Alarms On", 10449408, 22698294, 1.0, None],
        ["HCO3 (serum)", 10449297, 29981093, 26.0, "mEq/L"],
        ["Temperature Celsius", 10449408, 22698294, 37.2, "°C"],
        ["Hematocrit (serum)", 10449408, 22698294, 29.1, "%"],
        ["Impaired Skin Odor #4", 10445790, 26253687, 0.0, None],
        ["Hourly Patient Fluid Removal", 10449408, 22698294, 150.0, "mL"],
        ["Expiratory Ratio", 10449408, 22698294, 3.2, None],
        ["Multi Lumen Dressing Occlusive", 10449408, 22698294, 1.0, None],
        ["TCO2 (calc) Arterial", 10449408, 22698294, 29.0, "mEq/L"],
        ["Arterial Blood Pressure systolic", 10449408, 22698294, 120.0, "mmHg"],
        ["O2 saturation pulseoxymetry", 10449408, 22698294, 95.0, "%"],
        ["Non Invasive Blood Pressure mean", 10449297, 29981093, 63.0, "mmHg"],
        ["Cordis/Introducer Dressing Occlusive", 10449408, 22698294, 1.0, None],
        ["Arterial Blood Pressure systolic", 10449408, 22698294, 99.0, "mmHg"],
        ["Heart Rate", 10449408, 22698294, 74.0, "bpm"],
        ["Sodium (serum)", 10449408, 22698294, 142.0, "mEq/L"],
        ["Cordis/Introducer Dressing Occlusive", 10449408, 22698294, 1.0, None],
        ["Arterial Blood Pressure systolic", 10449408, 22698294, 103.0, "mmHg"],
        ["Inspired O2 Fraction", 10449408, 22698294, 40.0, None],
        ["HCO3 (serum)", 10449408, 22698294, 19.0, "mEq/L"],
        ["Heart Rate", 10449408, 22698294, 82.0, "bpm"],
        ["Cough/Deep Breath", 10449408, 22698294, 1.0, None],
        ["Potassium (serum)", 10445790, 26253687, 3.6, "mEq/L"],
        ["Heart Rate", 10449408, 22698294, 95.0, "bpm"],
        ["Heart Rate", 10449408, 22698294, 100.0, "bpm"],
        ["Inspired O2 Fraction", 10449408, 22698294, 50.0, None],
        ["Blood Flow (ml/min)", 10449408, 22698294, 250.0, "ml/min"],
        ["Arterial Blood Pressure mean", 10449408, 22698294, 74.0, "mmHg"],
        ["Access Pressure", 10449408, 22698294, -77.0, "mmHg"],
        ["Phosphorous", 10449408, 22698294, 3.4, "mg/dL"],
    ]
    data = pd.DataFrame(data, columns=cols)
    return data


def test_nongrouped_normalization(test_input):  # pylint: disable=redefined-outer-name
    """Test normalization without using a groupby."""
    gbn = GroupbyNormalizer({SUBJECT_ID: MIN_MAX, EVENT_VALUE: STANDARD})
    gbn.fit(test_input)
    normalized = gbn.transform(test_input)

    min_max = SklearnNormalizer(MIN_MAX)
    min_max.fit(test_input[SUBJECT_ID])
    min_max_norm = min_max.transform(test_input[SUBJECT_ID])
    assert np.allclose(normalized[SUBJECT_ID].values, min_max_norm.values, atol=1e-07)

    standard = SklearnNormalizer(STANDARD)
    standard.fit(test_input[EVENT_VALUE])
    standard_norm = standard.transform(test_input[EVENT_VALUE])
    assert np.allclose(normalized[EVENT_VALUE].values, standard_norm.values, atol=1e-07)

    denormalized = gbn.inverse_transform(normalized)
    assert np.allclose(
        denormalized[SUBJECT_ID].values, test_input[SUBJECT_ID].values, atol=1e-07
    )
    assert np.allclose(
        denormalized[EVENT_VALUE].values, test_input[EVENT_VALUE].values, atol=1e-07
    )


def test_grouped_normalization(test_input):  # pylint: disable=redefined-outer-name
    """Test normalization using a groupby."""
    gbn = GroupbyNormalizer(
        {SUBJECT_ID: MIN_MAX, EVENT_VALUE: STANDARD}, by=[EVENT_NAME]
    )
    gbn.fit(test_input)

    normalized = gbn.transform(test_input)

    assert normalized[SUBJECT_ID].max() <= 1
    assert normalized[SUBJECT_ID].min() >= 0
    assert (
        np.abs(normalized.groupby(EVENT_NAME)[EVENT_VALUE].apply("mean").values) < 1e-07
    ).all()

    denormalized = gbn.inverse_transform(normalized)

    assert np.allclose(
        denormalized[SUBJECT_ID].values, test_input[SUBJECT_ID].values, atol=1e-07
    )
    assert np.allclose(
        denormalized[EVENT_VALUE].values, test_input[EVENT_VALUE].values, atol=1e-07
    )


def test_vectorized_normalizer():
    """Test VectorizedNormalizer."""
    feat_map = {"A": 0, "B": 1}

    data = np.array(
        [
            [[1, 2, 3], [3, 2, 100]],
            [[4, 5, 2], [9, 20, 10]],
        ]
    ).astype(float)

    feat_map = {"A": 0, "B": 1}

    values0 = np.array([1, 2, 3, 4, 5, 2])
    values1 = np.array([3, 2, 100, 9, 20, 10])

    normalizer0 = SklearnNormalizer(STANDARD)
    normalizer0.fit(values0)
    normalized0 = normalizer0.transform(values0).reshape((2, 3))

    normalizer1 = SklearnNormalizer(STANDARD)
    normalizer1.fit(values1)
    normalized1 = normalizer1.transform(values1).reshape((2, 3))

    normalizer = VectorizedNormalizer(1, {"A": STANDARD, "B": STANDARD})
    normalizer.fit(data, feat_map)

    normalized = normalizer.transform(data, feat_map)

    assert np.array_equal(normalized[:, 0, :], normalized0)
    assert np.array_equal(normalized[:, 1, :], normalized1)
