"""Test low-level query API processing functions."""

import pytest
from sqlalchemy import column, select

from cyclops.query.omop import OMOPQuerier
from cyclops.query.process import (
    QAP,
    ConditionIn,
    ConditionSubstring,
    Drop,
    Literal,
    Rename,
    ckwarg,
    process_checks,
    process_operations,
)
from cyclops.query.util import process_column


@pytest.fixture
def test_table():
    """Test table input."""
    column_a = process_column(column("a"), to_timestamp=True)
    return select(column_a, column("b"), column("c"))


def test_ckwarg():
    """Test ckwarg."""
    assert ckwarg({"arg1": 1}, "arg1") == 1
    assert ckwarg({"arg1": 1}, "arg2") is None


def test_qap():
    """Test QAP."""
    args = QAP("arg1", required=False, transform_fn=int)
    test_arg = args(arg1="2")
    assert isinstance(test_arg, int) and test_arg == 2


def test_process_checks(test_table):  # pylint: disable=redefined-outer-name
    """Test process_checks fn."""
    process_checks(test_table, cols=["a"], cols_not_in=["d"], timestamp_cols=["a"])
    with pytest.raises(ValueError):
        process_checks(test_table, cols_not_in=["a"])


def test_process_operations(test_table):  # pylint: disable=redefined-outer-name
    """Test process_operations fn."""
    process_kwargs = {"b_args": "cat", "c_args": ["dog"]}
    operations = [
        (
            ConditionIn,
            ["b", QAP("b_args")],
            {"to_str": True},
        ),
        (ConditionSubstring, ["c", QAP("c_args")], {}),
    ]
    table = process_operations(test_table, operations, process_kwargs)
    query_lines = str(table).splitlines()
    assert query_lines[0] == "SELECT anon_1.a, anon_1.b, anon_1.c "
    assert query_lines[-1] == "WHERE lower(CAST(anon_1.c AS VARCHAR)) LIKE :lower_1"


@pytest.mark.integration_test
def test_operations():
    """Test Drop operation."""
    synthea = OMOPQuerier("cdm_synthea10", ["database=synthea"])
    visits = synthea.visit_occurrence().query
    visits = Drop("care_site_source_value")(visits)
    visits = Rename({"care_site_name": "hospital_name"})(visits)
    visits = Literal(1, "new_col")(visits)
    visits = synthea.get_interface(visits).run()

    assert "care_site_source_value" not in visits.columns
    assert "care_site_name" not in visits.columns
    assert "hospital_name" in visits.columns
    assert "new_col" in visits.columns
    assert visits["new_col"].unique() == 1
