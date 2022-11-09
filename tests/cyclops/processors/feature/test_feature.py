"""Test feature module."""

import os
import unittest

import numpy as np
import pandas as pd

from cyclops.processors.column_names import ENCOUNTER_ID
from cyclops.processors.constants import (
    BINARY,
    FEATURE_MAPPING_ATTR,
    FEATURE_TARGET_ATTR,
    FEATURE_TYPE_ATTR,
    NUMERIC,
    ORDINAL,
    STRING,
)
from cyclops.processors.feature.feature import FeatureMeta, Features
from cyclops.processors.feature.normalization import GroupbyNormalizer


def test__feature_meta__get_type():
    """Test FeatureMeta.get_type fn."""
    feature_meta_numeric = FeatureMeta(**{FEATURE_TYPE_ATTR: NUMERIC})
    feature_meta_binary = FeatureMeta(**{FEATURE_TYPE_ATTR: BINARY})
    feature_meta_string = FeatureMeta(**{FEATURE_TYPE_ATTR: STRING})
    feature_meta_ordinal = FeatureMeta(**{FEATURE_TYPE_ATTR: ORDINAL})

    assert feature_meta_numeric.get_type() == NUMERIC
    assert feature_meta_binary.get_type() == BINARY
    assert feature_meta_string.get_type() == STRING
    assert feature_meta_ordinal.get_type() == ORDINAL


def test__feature_meta__is_target():
    """Test FeatureMeta.is_target fn."""
    feature_meta_target = FeatureMeta(
        **{FEATURE_TYPE_ATTR: NUMERIC, FEATURE_TARGET_ATTR: True}
    )
    feature_meta = FeatureMeta(**{FEATURE_TYPE_ATTR: NUMERIC})

    assert feature_meta_target.is_target()
    assert not feature_meta.is_target()


def test__feature_meta__get_mapping():
    """Test FeatureMeta.get_mapping fn."""
    feature_meta = FeatureMeta(**{FEATURE_TYPE_ATTR: NUMERIC})
    assert feature_meta.get_mapping() is None

    feature_meta = FeatureMeta(
        **{FEATURE_TYPE_ATTR: NUMERIC, FEATURE_MAPPING_ATTR: {1: "hospital"}}
    )
    assert feature_meta.get_mapping() == {1: "hospital"}


def test__feature_meta__update():
    """Test FeatureMeta.update fn."""
    feature_meta = FeatureMeta(**{FEATURE_TYPE_ATTR: NUMERIC})
    assert feature_meta.get_type() == NUMERIC
    feature_meta.update([(FEATURE_TYPE_ATTR, BINARY)])
    assert feature_meta.get_type() == BINARY
    assert not feature_meta.is_target()
    feature_meta.update([(FEATURE_TARGET_ATTR, True)])
    assert feature_meta.is_target()


def _create_feature(data, features, by_attribute, targets=None):
    return Features(data=data, features=features, by=by_attribute, targets=targets)


def test__feature__get_data():
    """Test Feature.get_data fn."""
    data = pd.DataFrame({"fe": [1], "f": [1], ENCOUNTER_ID: [1]})
    feat = _create_feature(data, ["fe", "f"], ENCOUNTER_ID)
    data = data.set_index([ENCOUNTER_ID])
    returned_data = feat.get_data()
    returned_data = returned_data.reindex(returned_data.columns.sort_values(), axis=1)
    data = data.reindex(data.columns.sort_values(), axis=1)
    assert returned_data.equals(data)

    boolean_data = pd.DataFrame({"fe": [True], "f": [True], ENCOUNTER_ID: [1]})
    feat = _create_feature(boolean_data, ["fe", "f"], ENCOUNTER_ID)
    boolean_data = boolean_data.set_index([ENCOUNTER_ID])
    boolean_data = boolean_data.astype("int")
    returned_data = feat.get_data()
    boolean_data = boolean_data.reindex(boolean_data.columns.sort_values(), axis=1)
    returned_data = returned_data.reindex(returned_data.columns.sort_values(), axis=1)
    assert returned_data.equals(boolean_data)


def test__feature__columns():
    """Test Feature.columns fn."""
    data = pd.DataFrame({"fe": [1], "f": [1], ENCOUNTER_ID: [1]})
    feat = _create_feature(data, ["fe", "f"], ENCOUNTER_ID)
    returned_data = feat.columns
    assert returned_data.sort_values().equals(data.columns.sort_values())


def test__feature__feature_names():
    """Test Feature.feature_names fn."""
    data = pd.DataFrame({"fe": [1], "f": [1], ENCOUNTER_ID: [1]})
    features = ["fe", "f"]
    feat = _create_feature(data, features, ENCOUNTER_ID)
    returned_data = feat.feature_names()
    assert sorted(returned_data) == sorted(features)


def test__feature__types():
    """Test Feature.types fn."""
    data = pd.DataFrame({"fe": [1], "f": [1], ENCOUNTER_ID: [1]})
    feat = _create_feature(data, ["fe", "f"], ENCOUNTER_ID)
    returned_data = feat.types
    assert returned_data == {"fe": NUMERIC, "f": NUMERIC}


def test__feature__targets():
    """Test Feature.targets fn."""
    data = pd.DataFrame({"fe": [1], "f": [1], ENCOUNTER_ID: [1]})
    feat = _create_feature(data, ["fe", "f"], ENCOUNTER_ID)
    returned_data = feat.targets
    assert returned_data == []


def test__feature__features_by_type():
    """Test Feature.features_by_type fn."""
    data = pd.DataFrame({"fe": [1], "f": [1], ENCOUNTER_ID: [1]})
    feat = _create_feature(data, ["fe", "f"], ENCOUNTER_ID)
    returned_data = feat.features_by_type(NUMERIC)
    assert sorted(returned_data) == ["f", "fe"]
    data = pd.DataFrame({"fe": [True], "f": [False], ENCOUNTER_ID: [1]})
    feat = _create_feature(data, ["fe", "f"], ENCOUNTER_ID)
    returned_data = feat.features_by_type(BINARY)
    assert sorted(returned_data) == ["f", "fe"]
    data = pd.DataFrame(
        {
            "fe": [True, True],
            "f": [False, False],
            "fea": ["hi", "hey"],
            ENCOUNTER_ID: [1, 1],
        }
    )
    feat = _create_feature(data, ["fe", "f", "fea"], ENCOUNTER_ID)
    returned_data = feat.features_by_type(BINARY)
    assert sorted(returned_data) == ["f", "fe", "fea"]


def test__feature__compute_value_splits():
    """Test Feature.compute_value_splits fn."""
    data = pd.DataFrame(
        {"fe": [10] * 10, "f": [10] * 10, ENCOUNTER_ID: [1] * 5 + [2] * 5}
    )
    feat = _create_feature(data, ["fe", "f"], ENCOUNTER_ID)
    returned_data = feat.compute_value_splits([0.5, 0.5], seed=42)
    returned_data = [data[0] for data in returned_data]
    assert sorted(returned_data) == [1, 2]


def test__feature__split_by_values():
    """Test Feature.split_by_values fn."""
    data = pd.DataFrame(
        {"fe": [10] * 10, "f": [10] * 10, ENCOUNTER_ID: [1] * 5 + [2] * 5}
    )
    feat = _create_feature(data, ["fe", "f"], ENCOUNTER_ID)
    value_splits = feat.compute_value_splits([0.5, 0.5], seed=42)
    returned_data = feat.split_by_values(value_splits)
    df1 = pd.DataFrame({"fe": [10] * 5, "f": [10] * 5, ENCOUNTER_ID: [1] * 5})
    df2 = pd.DataFrame({"fe": [10] * 5, "f": [10] * 5, ENCOUNTER_ID: [2] * 5})
    df2.index = np.arange(5, 10)
    assert returned_data[0].data.equals(df2)
    assert returned_data[1].data.equals(df1)


def test__feature__split():
    """Test Feature.split fn."""
    data = pd.DataFrame(
        {"fe": [10] * 10, "f": [10] * 10, ENCOUNTER_ID: [1] * 5 + [2] * 5}
    )
    feat = _create_feature(data, ["fe", "f"], ENCOUNTER_ID)
    returned_data = feat.split([0.5, 0.5], seed=42)
    df1 = pd.DataFrame({"fe": [10] * 5, "f": [10] * 5, ENCOUNTER_ID: [1] * 5})
    df2 = pd.DataFrame({"fe": [10] * 5, "f": [10] * 5, ENCOUNTER_ID: [2] * 5})
    df2.index = np.arange(5, 10)
    assert returned_data[0].data.equals(df2)
    assert returned_data[1].data.equals(df1)


def test__feature__add_normalizer():
    """Test Feature.add_normalizer fn."""
    normalizer = GroupbyNormalizer({"f": "standard"})
    data = pd.DataFrame(
        {"fe": [10] * 10, "f": [10] * 10, ENCOUNTER_ID: [1] * 5 + [2] * 5}
    )
    feat = _create_feature(data, ["fe", "f"], ENCOUNTER_ID)
    feat.add_normalizer("normalizer", normalizer)
    norm = feat.normalizers
    assert norm["normalizer"].get_map() == normalizer.get_map()


def test__feature__remove_normalizer():
    """Test Feature.remove_normalizer fn."""
    normalizer = GroupbyNormalizer({"f": "standard"})
    data = pd.DataFrame(
        {"fe": [10] * 10, "f": [10] * 10, ENCOUNTER_ID: [1] * 5 + [2] * 5}
    )
    feat = _create_feature(data, ["fe", "f"], ENCOUNTER_ID)
    feat.add_normalizer("normalizer", normalizer)
    feat.remove_normalizer("normalizer")
    norm = feat.normalizers
    assert not norm


def test__feature__normalize():
    """Test Feature.normalize fn."""
    normalizer = GroupbyNormalizer({"f": "standard"})
    data = pd.DataFrame(
        {"fe": [10] * 10, "f": [10] * 10, ENCOUNTER_ID: [1] * 5 + [2] * 5}
    )
    feat = _create_feature(data, ["fe", "f"], ENCOUNTER_ID)
    feat.add_normalizer("normalizer", normalizer)
    new_data = feat.normalize("normalizer")
    data = pd.DataFrame(
        {"fe": [10] * 10, "f": [0.0] * 10, ENCOUNTER_ID: [1] * 5 + [2] * 5}
    )
    assert new_data.equals(data)


def test__feature__inverse_normalize():
    """Test Feature.inverse_normalize fn."""
    normalizer = GroupbyNormalizer({"f": "standard"})
    data = pd.DataFrame(
        {"fe": [10] * 10, "f": [10] * 10, ENCOUNTER_ID: [1] * 5 + [2] * 5}
    )
    feat = _create_feature(data, ["fe", "f"], ENCOUNTER_ID)
    feat.add_normalizer("normalizer", normalizer)
    feat.normalize("normalizer")
    new_data = feat.inverse_normalize("normalizer")
    data = pd.DataFrame(
        {"fe": [10] * 10, "f": [10.0] * 10, ENCOUNTER_ID: [1] * 5 + [2] * 5}
    )
    assert new_data.equals(data)


def test__feature__save():
    """Test Feature.save fn."""
    data = pd.DataFrame(
        {"fe": [10] * 10, "f": [10] * 10, ENCOUNTER_ID: [1] * 5 + [2] * 5}
    )
    feat = _create_feature(data, ["fe", "f"], ENCOUNTER_ID)
    filename = os.path.join(os.getcwd(), "feature.parquet")
    feat.save(filename)
    assert os.path.exists(filename)
    os.remove(filename)
    assert not os.path.exists(filename)


class TestFeatures(unittest.TestCase):
    """Test Features class."""

    def setUp(self):
        """Create test features to test."""
        self.test_data = pd.DataFrame(
            {
                "feat_A": [False, True, True],
                "feat_B": [1.2, 3, 3.8],
                ENCOUNTER_ID: [101, 201, 301],
            }
        )
        self.features = Features(
            data=self.test_data, features=["feat_A", "feat_B"], by=ENCOUNTER_ID
        )

    def test_slice(self):
        """Test slice method."""
        sliced_by_indices = self.features.slice({"feat_B": 3}, replace=False)
        assert np.array_equal(sliced_by_indices, np.array([201]))

        sliced_by_indices = self.features.slice(
            {"feat_A": True, "feat_B": [3.8, 3]}, replace=False
        )
        assert np.array_equal(sliced_by_indices, np.array([201, 301]))
        sliced_by_indices = self.features.slice({}, replace=True)
        assert np.array_equal(sliced_by_indices, np.array([101, 201, 301]))
        assert len(self.features.data) == 3
        sliced_by_indices = self.features.slice(
            slice_query="feat_A == True & feat_B > 3", replace=False
        )
        assert np.array_equal(sliced_by_indices, np.array([301]))
        filter_list = [3, 3.8]
        sliced_by_indices = self.features.slice(
            slice_query=f"feat_B=={filter_list}", replace=True
        )
        assert np.array_equal(sliced_by_indices, np.array([201, 301]))
        assert len(self.features.data) == 2
