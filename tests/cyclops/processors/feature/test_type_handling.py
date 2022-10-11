"""Test type handling fns."""

import pandas as pd
import pytest

from cyclops.processors.constants import BINARY, NUMERIC, ORDINAL, STRING
from cyclops.processors.feature.type_handling import (get_unique,
                                                      valid_feature_type,
                                                      to_dtype,
                                                      collect_indicators,
                                                      convertible_to_type,
                                                      infer_types,
                                                      is_valid,
                                                      normalize_data,
                                                      to_types)


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
    """Test to_dtype fn"""
    s1 = pd.Series([1, 3, 5])
    assert pd.api.types.is_numeric_dtype(to_dtype(s1, NUMERIC))
    s2 = pd.Series([True, False, True])
    assert pd.api.types.is_bool_dtype(to_dtype(s2, BINARY))
    s3 = pd.Series([0, 1, 3])
    assert pd.api.types.is_categorical_dtype(to_dtype(s3, ORDINAL))
    s4 = pd.Series(["a", 'B', "C"])
    assert pd.api.types.is_object_dtype(to_dtype(s4, STRING))


def test_collect_indicators():
    """Test collect_indicators fn"""
    df1 = pd.DataFrame()
    df1['hospital_A'] = pd.Series([1, 0, 1])
    df1['hospital_B'] = pd.Series([0, 1, 0])
    df1['room_A'] = pd.Series([0, 1, 1])
    df1['room_B'] = pd.Series([1, 0, 0])
    categories = ['hospital', 'room']
    data, meta = collect_indicators(df1, categories)
    assert meta == {'hospital': {'mapping': {0: 'A', 1: 'B'}, 'type_': 'ordinal'},
                    'room': {'mapping': {0: 'A', 1: 'B'}, 'type_': 'ordinal'}}


def test_convertible_to_type():
    """Test convertible_to_type fn"""
    s1 = pd.Series([1, 3, 5])
    assert convertible_to_type(s1, NUMERIC)
    assert convertible_to_type(s1, STRING)
    s2 = pd.Series([1, 3, 5])
    assert not convertible_to_type(s2, BINARY)
    assert convertible_to_type(s2, ORDINAL)


def test_infer_types():
    """Test infer_types fn"""
    df1 = pd.DataFrame()
    df1['numbers'] = pd.Series([1, 2, 3])
    df1['strings'] = pd.Series(["1", "2", "3"])
    df1['ordinal'] = pd.Series([1, 2, 3])
    df1['binary'] = pd.Series([1, 0, 1])
    features = ['numbers', 'strings', 'ordinal', 'binary']
    types = infer_types(df1, features)
    assert types == {"numbers": ORDINAL,
                     "strings": ORDINAL,
                     "ordinal": ORDINAL,
                     "binary": BINARY}


def test_is_valid():
    """Test is_valid fn"""
    s1 = pd.Series([1, 3, 5])
    assert is_valid(s1, NUMERIC)
    assert not is_valid(s1, ORDINAL)
    assert not is_valid(s1, BINARY)
    assert not is_valid(s1, STRING)

    s2 = pd.Series(["1", "3", "5"])
    assert not is_valid(s2, BINARY)
    assert not is_valid(s2, ORDINAL)
    assert not is_valid(s2, NUMERIC)
    assert is_valid(s2, STRING)

    s2 = pd.Series([0, 1, 2])
    assert is_valid(s2, NUMERIC)
    assert is_valid(s2, ORDINAL)
    assert not is_valid(s2, STRING)
    assert not is_valid(s2, BINARY)


def test_normalize_data():
    """Test normalize_data fn"""
    df1 = pd.DataFrame()
    df1['numbers'] = pd.Series([1, 2, 3])
    df1['strings'] = pd.Series(["1", "2", "3"])
    df1['nones'] = pd.Series(["None", 2, 3])
    features = ['numbers', 'strings', 'nones']
    data = normalize_data(df1, features)
    assert pd.isna(data.loc[0, 'nones'])


def test_to_types():
    """Test to_types fn"""
    df = pd.DataFrame()
    df['nums'] = pd.Series([1, 2, 3])
    df['strs'] = pd.Series(['1', '2', '3'])
    new_types = {"nums": STRING, "strs": NUMERIC}
    data, meta = to_types(df, new_types)
    assert meta == {'nums': {'type_': 'string'}, 'strs': {'type_': 'numeric'}}
