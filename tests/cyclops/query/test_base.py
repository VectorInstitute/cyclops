"""Test base dataset querier."""

from cyclops.query.base import DatasetQuerier


def test_dataset_querier():
    """Test dataset querier instantiation."""
    _ = DatasetQuerier({})
