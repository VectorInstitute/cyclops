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
    Limit,
    Literal,
    OrderBy,
    Rename,
    ReorderAfter,
    Substring,
    Trim,
    _none_add,
    _process_checks,
    process_operations,
)
from cyclops.query.util import process_column


@pytest.fixture
def test_table():
    """Test table input."""
    column_a = process_column(column("a"), to_timestamp=True)
    return select(column_a, column("b"), column("c"))


def test__none_add():
    """Test _none_add fn."""
    assert _none_add("1", "2") == "12"
    assert _none_add("1", None) == "1"
    assert _none_add(None, "2") == "2"


def test_qap():
    """Test QAP."""
    args = QAP("arg1", required=False, transform_fn=int)
    test_arg = args(arg1="2")
    assert isinstance(test_arg, int) and test_arg == 2


def test__process_checks(test_table):  # pylint: disable=redefined-outer-name
    """Test _process_checks fn."""
    _process_checks(test_table, cols=["a"], cols_not_in=["d"], timestamp_cols=["a"])
    with pytest.raises(ValueError):
        _process_checks(test_table, cols_not_in=["a"])


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
    synthea = OMOPQuerier("cdm_synthea10", database="synthea_integration_test")
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
    visits_ordered = OrderBy("person_id", ascending=True)(visits)
    visits_limited = Limit(100)(visits)
    visits_agg_count = GroupByAggregate(
        "person_id", {"person_id": ("count", "visit_count")}
    )(visits)
    visits_string_agg = GroupByAggregate(
        "person_id",
        {"visit_concept_name": ("string_agg", "visit_concept_names")},
        {"visit_concept_name": ", "},
    )(visits)
    with pytest.raises(ValueError):
        visits_agg_count = GroupByAggregate(
            "person_id", {"person_id": ("donkey", "visit_count")}
        )(visits)
    with pytest.raises(ValueError):
        visits_agg_count = GroupByAggregate(
            "person_id", {"person_id": ("count", "person_id")}
        )(visits)
    visits_agg_median = GroupByAggregate(
        "person_id", {"visit_concept_name": ("median", "visit_concept_name_median")}
    )(visits)
    visits_agg_count = synthea.get_interface(visits_agg_count).run()
    visits_agg_median = synthea.get_interface(visits_agg_median).run()
    visits = synthea.get_interface(visits).run()
    visits_ordered = synthea.get_interface(visits_ordered).run()
    visits_limited = synthea.get_interface(visits_limited).run()
    visits_string_agg = synthea.get_interface(visits_string_agg).run()

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
    assert visits_agg_count[visits_agg_count["person_id"] == 33]["visit_count"][0] == 25
    assert visits_agg_median["visit_concept_name_median"].value_counts()[0] == 107
    assert visits_ordered["person_id"][0] == 1
    assert len(visits_limited) == 100
    test_visit_concept_names = visits_string_agg[visits_string_agg["person_id"] == 33][
        "visit_concept_names"
    ][0].split(",")
    test_visit_concept_names = [item.strip() for item in test_visit_concept_names]
    assert (
        len(test_visit_concept_names) == 25
        and "Outpatient Visit" in test_visit_concept_names
    )
