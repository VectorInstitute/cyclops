"""Test type handling fns."""

import pandas as pd
import pytest

from cyclops.processors.constants import BINARY, NUMERIC, ORDINAL, STRING
from cyclops.processors.feature.type_handling import (
    collect_indicators,
    convertible_to_type,
    get_unique,
    infer_types,
    is_valid,
    normalize_data,
    to_dtype,
    to_types,
    valid_feature_type,
)


def test_get_unique():
    """Test get_unique fn."""
    assert (get_unique(pd.Series([1, 3, 5])) == [1, 3, 5]).all()
    assert (get_unique(pd.Series([1, 1, 5])) == [1, 5]).all()


def test_valid_feature_type():
    """Test valid_feature_type fn."""
    assert valid_feature_type(NUMERIC)
    assert valid_feature_type(BINARY)
    assert valid_feature_type(STRING)
    assert valid_feature_type(ORDINAL)
    with pytest.raises(ValueError):
        valid_feature_type("donkey")
    assert not valid_feature_type("donkey", raise_error=False)


def test_to_dtype():
    """Test to_dtype fn."""
    series_one = pd.Series([1, 3, 5])
    assert pd.api.types.is_numeric_dtype(to_dtype(series_one, NUMERIC))
    series_two = pd.Series([True, False, True])
    assert pd.api.types.is_bool_dtype(to_dtype(series_two, BINARY))
    series_three = pd.Series([0, 1, 3])
    assert pd.api.types.is_categorical_dtype(to_dtype(series_three, ORDINAL))
    series_four = pd.Series(["a", "B", "C"])
    assert pd.api.types.is_object_dtype(to_dtype(series_four, STRING))


def test_collect_indicators():
    """Test collect_indicators fn."""
    data = pd.DataFrame()
    data["hospital_A"] = pd.Series([1, 0, 1])
    data["hospital_B"] = pd.Series([0, 1, 0])
    data["room_A"] = pd.Series([0, 1, 1])
    data["room_B"] = pd.Series([1, 0, 0])
    categories = ["hospital", "room"]
    _, meta = collect_indicators(data, categories)
    assert meta == {
        "hospital": {"mapping": {0: "A", 1: "B"}, "type_": "ordinal"},
        "room": {"mapping": {0: "A", 1: "B"}, "type_": "ordinal"},
    }


def test_convertible_to_type():
    """Test convertible_to_type fn."""
    series_one = pd.Series([1, 3, 5])
    assert convertible_to_type(series_one, NUMERIC)
    assert convertible_to_type(series_one, STRING)
    series_two = pd.Series([1, 3, 5])
    assert not convertible_to_type(series_two, BINARY)
    assert convertible_to_type(series_two, ORDINAL)


def test_infer_types():
    """Test infer_types fn."""
    data = pd.DataFrame()
    data["numbers"] = pd.Series([1, 2, 3])
    data["strings"] = pd.Series(["1", "2", "3"])
    data["ordinal"] = pd.Series([1, 2, 3])
    data["binary"] = pd.Series([1, 0, 1])
    features = ["numbers", "strings", "ordinal", "binary"]
    types = infer_types(data, features)
    assert types == {
        "numbers": ORDINAL,
        "strings": ORDINAL,
        "ordinal": ORDINAL,
        "binary": BINARY,
    }


def test_is_valid():
    """Test is_valid fn."""
    series_one = pd.Series([1, 3, 5])
    assert is_valid(series_one, NUMERIC)
    assert not is_valid(series_one, ORDINAL)
    assert not is_valid(series_one, BINARY)
    assert not is_valid(series_one, STRING)

    series_two = pd.Series(["1", "3", "5"])
    assert not is_valid(series_two, BINARY)
    assert not is_valid(series_two, ORDINAL)
    assert not is_valid(series_two, NUMERIC)
    assert is_valid(series_two, STRING)

    series_three = pd.Series([0, 1, 2])
    assert is_valid(series_three, NUMERIC)
    assert is_valid(series_three, ORDINAL)
    assert not is_valid(series_three, STRING)
    assert not is_valid(series_three, BINARY)


def test_normalize_data():
    """Test normalize_data fn."""
    data = pd.DataFrame()
    data["numbers"] = pd.Series([1, 2, 3])
    data["strings"] = pd.Series(["1", "2", "3"])
    data["nones"] = pd.Series(["None", 2, 3])
    features = ["numbers", "strings", "nones"]
    normalized_data = normalize_data(data, features)
    assert pd.isna(normalized_data.loc[0, "nones"])


def test_to_types():
    """Test to_types fn."""
    data = pd.DataFrame()
    data["nums"] = pd.Series([1, 2, 3])
    data["strs"] = pd.Series(["1", "2", "3"])
    new_types = {"nums": STRING, "strs": NUMERIC}
    _, meta = to_types(data, new_types)
    assert meta == {"nums": {"type_": "string"}, "strs": {"type_": "numeric"}}
