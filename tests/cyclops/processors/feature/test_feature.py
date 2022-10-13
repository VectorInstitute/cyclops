"""Test feature module."""

from cyclops.processors.constants import (
    BINARY,
    FEATURE_MAPPING_ATTR,
    FEATURE_TARGET_ATTR,
    FEATURE_TYPE_ATTR,
    NUMERIC,
    ORDINAL,
    STRING,
)
from cyclops.processors.feature.feature import FeatureMeta


def test_get_type():
    """Test FeatureMeta.get_type fn."""
    feature_meta_numeric = FeatureMeta(**{FEATURE_TYPE_ATTR: NUMERIC})
    feature_meta_binary = FeatureMeta(**{FEATURE_TYPE_ATTR: BINARY})
    feature_meta_string = FeatureMeta(**{FEATURE_TYPE_ATTR: STRING})
    feature_meta_ordinal = FeatureMeta(**{FEATURE_TYPE_ATTR: ORDINAL})

    assert feature_meta_numeric.get_type() == NUMERIC
    assert feature_meta_binary.get_type() == BINARY
    assert feature_meta_string.get_type() == STRING
    assert feature_meta_ordinal.get_type() == ORDINAL


def test_is_target():
    """Test FeatureMeta.is_target fn."""
    feature_meta_target = FeatureMeta(
        **{FEATURE_TYPE_ATTR: NUMERIC, FEATURE_TARGET_ATTR: True}
    )
    feature_meta = FeatureMeta(**{FEATURE_TYPE_ATTR: NUMERIC})

    assert feature_meta_target.is_target()
    assert not feature_meta.is_target()


def test_get_mapping():
    """Test FeatureMeta.get_mapping fn."""
    feature_meta = FeatureMeta(**{FEATURE_TYPE_ATTR: NUMERIC})
    assert feature_meta.get_mapping() is None

    feature_meta = FeatureMeta(
        **{FEATURE_TYPE_ATTR: NUMERIC, FEATURE_MAPPING_ATTR: {1: "hospital"}}
    )
    assert feature_meta.get_mapping() == {1: "hospital"}


def test_update():
    """Test FeatureMeta.update fn."""
    feature_meta = FeatureMeta(**{FEATURE_TYPE_ATTR: NUMERIC})
    assert feature_meta.get_type() == NUMERIC
    feature_meta.update([(FEATURE_TYPE_ATTR, BINARY)])
    assert feature_meta.get_type() == BINARY
    assert not feature_meta.is_target()
    feature_meta.update([(FEATURE_TARGET_ATTR, True)])
    assert feature_meta.is_target()
