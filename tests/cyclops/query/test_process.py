"""Test low-level query API processing functions."""

import pytest
from sqlalchemy import column, select

from cyclops.query.omop import OMOPQuerier
from cyclops.query.process import (
    QAP,
    AddNumeric,
    ConditionIn,
    ConditionSubstring,
    Drop,
    ExtractTimestampComponent,
    GroupByAggregate,
    Literal,
    Rename,
    ReorderAfter,
    Substring,
    Trim,
    ckwarg,
    none_add,
    process_checks,
    process_operations,
    remove_kwargs,
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


def test_remove_kwargs():
    """Test remove_kwargs."""
    kwargs = {"arg1": 1, "arg2": 2, "arg3": 3}
    assert "arg2" not in remove_kwargs(kwargs, "arg2")
    assert "arg1" not in remove_kwargs(kwargs, ["arg2", "arg1"])


def test_none_add():
    """Test none_add fn."""
    assert none_add("1", "2") == "12"
    assert none_add("1", None) == "1"
    assert none_add(None, "2") == "2"


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
    """Test query operations."""
    synthea = OMOPQuerier("cdm_synthea10", ["database=synthea_integration_test"])
    visits = synthea.visit_occurrence().query
    visits = Drop("care_site_source_value")(visits)
    visits = Rename({"care_site_name": "hospital_name"})(visits)
    visits = Literal(1, "new_col")(visits)
    visits = Literal("test ", "untrimmed_col")(visits)
    visits = Substring("visit_concept_name", 0, 6, "visit_concept_name_substr")(visits)
    visits = ReorderAfter("visit_concept_name", "care_site_id")(visits)
    visits = Trim("untrimmed_col", "trimmed_col")(visits)
    visits = ExtractTimestampComponent("visit_start_datetime", "year", "visit_year")(
        visits
    )
    visits = AddNumeric("new_col", 2, "new_col_add")(visits)
    visits_agg = GroupByAggregate("person_id", {"person_id": ("count", "visit_count")})(
        visits
    )
    visits_agg = synthea.get_interface(visits_agg).run()
    visits = synthea.get_interface(visits).run()

    assert "care_site_source_value" not in visits.columns
    assert "care_site_name" not in visits.columns
    assert "hospital_name" in visits.columns
    assert "new_col" in visits.columns
    assert visits["new_col"].unique() == 1
    assert (
        visits["visit_concept_name_substr"].unique() == ["Inpat", "Outpa", "Emerg"]
    ).all()
    assert (
        visits.columns[list(visits.columns).index("care_site_id") + 1]
        == "visit_concept_name"
    )
    assert visits["trimmed_col"].unique() == "test"
    assert 2018 in visits["visit_year"]
    assert visits["new_col_add"].unique() == 3
    assert visits_agg[visits_agg["person_id"] == 33]["visit_count"][0] == 25
