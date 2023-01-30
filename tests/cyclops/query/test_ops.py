"""Test low-level query API processing functions."""

from math import isclose

import pytest
from sqlalchemy import column, select

from cyclops.query.omop import OMOPQuerier
from cyclops.query.ops import (
    AddNumeric,
    Apply,
    ConditionRegexMatch,
    Drop,
    ExtractTimestampComponent,
    GroupByAggregate,
    Limit,
    Literal,
    OrderBy,
    Rename,
    ReorderAfter,
    Sequential,
    Substring,
    Trim,
    _none_add,
    _process_checks,
)
from cyclops.query.util import process_column

SYNTHEA = OMOPQuerier("cdm_synthea10", database="synthea_integration_test")


@pytest.fixture
def table_input():
    """Test table input."""
    column_a = process_column(column("a"), to_timestamp=True)
    return select(column_a, column("b"), column("c"))


@pytest.fixture
def visits_input():
    """Test visits table input."""
    return SYNTHEA.visit_occurrence().query


@pytest.fixture
def measurements_input():
    """Test measurement table input."""
    return SYNTHEA.measurement().query


def test__none_add():
    """Test _none_add fn."""
    assert _none_add("1", "2") == "12"
    assert _none_add("1", None) == "1"
    assert _none_add(None, "2") == "2"


def test__process_checks(table_input):  # pylint: disable=redefined-outer-name
    """Test _process_checks fn."""
    _process_checks(table_input, cols=["a"], cols_not_in=["d"], timestamp_cols=["a"])
    with pytest.raises(ValueError):
        _process_checks(table_input, cols_not_in=["a"])


@pytest.mark.integration_test
def test_drop(visits_input):  # pylint: disable=redefined-outer-name
    """Test Drop."""
    visits = Drop("care_site_source_value")(visits_input)
    visits = SYNTHEA.get_interface(visits).run()
    assert "care_site_source_value" not in visits.columns


@pytest.mark.integration_test
def test_rename(visits_input):  # pylint: disable=redefined-outer-name
    """Test Rename."""
    visits = Rename({"care_site_name": "hospital_name"})(visits_input)
    visits = SYNTHEA.get_interface(visits).run()
    assert "hospital_name" in visits.columns
    assert "care_site_name" not in visits.columns


@pytest.mark.integration_test
def test_literal(visits_input):  # pylint: disable=redefined-outer-name
    """Test Literal."""
    visits = Literal(1, "new_col")(visits_input)
    visits = Literal("a", "new_col2")(visits)
    visits = SYNTHEA.get_interface(visits).run()
    assert "new_col" in visits.columns
    assert visits["new_col"].iloc[0] == 1
    assert "new_col2" in visits.columns
    assert visits["new_col2"].iloc[0] == "a"


@pytest.mark.integration_test
def test_reorder_after(visits_input):  # pylint: disable=redefined-outer-name
    """Test ReorderAfter."""
    visits = ReorderAfter("visit_concept_name", "care_site_id")(visits_input)
    visits = SYNTHEA.get_interface(visits).run()
    assert list(visits.columns).index("care_site_id") + 1 == list(visits.columns).index(
        "visit_concept_name"
    )


@pytest.mark.integration_test
def test_limit(visits_input):  # pylint: disable=redefined-outer-name
    """Test Limit."""
    visits = Limit(10)(visits_input)
    visits = SYNTHEA.get_interface(visits).run()
    assert len(visits) == 10


@pytest.mark.integration_test
def test_order_by(visits_input):  # pylint: disable=redefined-outer-name
    """Test OrderBy."""
    visits = OrderBy("visit_concept_name")(visits_input)
    visits = SYNTHEA.get_interface(visits).run()
    assert visits["visit_concept_name"].is_monotonic_increasing


@pytest.mark.integration_test
def test_substring(visits_input):  # pylint: disable=redefined-outer-name
    """Test Substring."""
    visits = Substring("visit_concept_name", 0, 3, "visit_concept_name_substr")(
        visits_input
    )
    visits = SYNTHEA.get_interface(visits).run()
    assert visits["visit_concept_name_substr"].iloc[0] == "In"


@pytest.mark.integration_test
def test_trim(visits_input):  # pylint: disable=redefined-outer-name
    """Test Trim."""
    visits = Trim("visit_concept_name", "visit_concept_name_trim")(visits_input)
    visits = SYNTHEA.get_interface(visits).run()
    assert visits["visit_concept_name_trim"].iloc[0] == "Inpatient Visit"


@pytest.mark.integration_test
def test_extract_timestamp_component(
    visits_input,
):  # pylint: disable=redefined-outer-name
    """Test ExtractTimestampComponent."""
    visits = ExtractTimestampComponent(
        "visit_start_date", "year", "visit_start_date_year"
    )(visits_input)
    visits = SYNTHEA.get_interface(visits).run()
    assert visits["visit_start_date_year"].iloc[0] == 2018


@pytest.mark.integration_test
def test_add_numeric(visits_input):  # pylint: disable=redefined-outer-name
    """Test AddNumeric."""
    visits = Literal(1, "new_col")(visits_input)
    visits = AddNumeric("new_col", 1, "new_col_plus_1")(visits)
    visits = SYNTHEA.get_interface(visits).run()
    assert visits["new_col_plus_1"].iloc[0] == 2


@pytest.mark.integration_test
def test_apply(visits_input):  # pylint: disable=redefined-outer-name
    """Test Apply."""
    visits = Apply(
        "visit_concept_name", lambda x: x + "!", "visit_concept_name_exclaim"
    )(visits_input)
    visits = SYNTHEA.get_interface(visits).run()
    assert visits["visit_concept_name_exclaim"].iloc[0] == "Inpatient Visit!"


@pytest.mark.integration_test
def test_condition_regex_match(
    measurements_input,
):  # pylint: disable=redefined-outer-name
    """Test ConditionRegexMatch."""
    measurements = ConditionRegexMatch(
        "value_source_value",
        r"^[0-9]+(\.[0-9]+)?$",
        binarize_col="value_source_value_match",
    )(measurements_input)
    measurements = SYNTHEA.get_interface(measurements).run()
    assert "value_source_value_match" in measurements.columns
    assert (
        measurements["value_source_value_match"].sum()
        == measurements["value_source_value"].str.match(r"^[0-9]+(\.[0-9]+)?$").sum()
    )


@pytest.mark.integration_test
def test_group_by_aggregate(  # pylint: disable=redefined-outer-name
    visits_input, measurements_input
):
    """Test GroupByAggregate."""
    with pytest.raises(ValueError):
        GroupByAggregate("person_id", {"person_id": ("donkey", "visit_count")})(
            visits_input
        )
    with pytest.raises(ValueError):
        GroupByAggregate("person_id", {"person_id": ("count", "person_id")})(
            visits_input
        )

    visits_count = GroupByAggregate(
        "person_id", {"person_id": ("count", "num_visits")}
    )(visits_input)
    visits_string_agg = GroupByAggregate(
        "person_id",
        {"visit_concept_name": ("string_agg", "visit_concept_names")},
        {"visit_concept_name": ", "},
    )(visits_input)
    measurements_sum = GroupByAggregate(
        "person_id", {"value_as_number": ("sum", "value_as_number_sum")}
    )(measurements_input)
    measurements_average = GroupByAggregate(
        "person_id", {"value_as_number": ("average", "value_as_number_average")}
    )(measurements_input)
    measurements_min = GroupByAggregate(
        "person_id", {"value_as_number": ("min", "value_as_number_min")}
    )(measurements_input)
    measurements_max = GroupByAggregate(
        "person_id", {"value_as_number": ("max", "value_as_number_max")}
    )(measurements_input)
    measurements_median = GroupByAggregate(
        "person_id", {"value_as_number": ("median", "value_as_number_median")}
    )(measurements_input)

    visits_count = SYNTHEA.get_interface(visits_count).run()
    visits_string_agg = SYNTHEA.get_interface(visits_string_agg).run()
    measurements_sum = SYNTHEA.get_interface(measurements_sum).run()
    measurements_average = SYNTHEA.get_interface(measurements_average).run()
    measurements_min = SYNTHEA.get_interface(measurements_min).run()
    measurements_max = SYNTHEA.get_interface(measurements_max).run()
    measurements_median = SYNTHEA.get_interface(measurements_median).run()

    assert "num_visits" in visits_count.columns
    assert visits_count[visits_count["person_id"] == 33]["num_visits"][0] == 25
    assert "visit_concept_names" in visits_string_agg.columns
    test_visit_concept_names = visits_string_agg[visits_string_agg["person_id"] == 33][
        "visit_concept_names"
    ][0].split(",")
    test_visit_concept_names = [item.strip() for item in test_visit_concept_names]
    assert (
        len(test_visit_concept_names) == 25
        and "Outpatient Visit" in test_visit_concept_names
    )
    assert "value_as_number_sum" in measurements_sum.columns
    assert (
        measurements_sum[measurements_sum["person_id"] == 33]["value_as_number_sum"][0]
        == 11371.0
    )
    assert "value_as_number_average" in measurements_average.columns
    assert isclose(
        measurements_average[measurements_average["person_id"] == 33][
            "value_as_number_average"
        ][0],
        61.79,
        abs_tol=0.01,
    )
    assert "value_as_number_min" in measurements_min.columns
    assert (
        measurements_min[measurements_min["person_id"] == 33]["value_as_number_min"][0]
        == 0.0
    )
    assert "value_as_number_max" in measurements_max.columns
    assert (
        measurements_max[measurements_max["person_id"] == 33]["value_as_number_max"][0]
        == 460.9
    )
    assert "value_as_number_median" in measurements_median.columns
    assert (
        measurements_median[measurements_median["person_id"] == 33][
            "value_as_number_median"
        ].item()
        == 53.1
    )


@pytest.mark.integration_test
def test_sequential(visits_input):  # pylint: disable=redefined-outer-name
    """Test Sequential."""
    substr_op = Sequential(
        [
            Substring("visit_concept_name", 0, 4, "visit_concept_name_substr"),
        ]
    )
    operations = [
        Literal(33, "const"),
        Rename({"care_site_name": "hospital_name"}),
        Apply("visit_concept_name", lambda x: x + "!", "visit_concept_name_exclaim"),
        OrderBy(["person_id", "visit_start_date"]),
        substr_op,
    ]
    sequential_ops = Sequential(operations)
    visits = SYNTHEA.get_interface(visits_input, ops=sequential_ops).run()
    assert "hospital_name" in visits.columns
    assert "visit_concept_name_exclaim" in visits.columns
    assert list(visits[visits["person_id"] == 33]["visit_concept_name_exclaim"])[0] == (
        "Outpatient Visit!"
    )
    assert "visit_concept_name_substr" in visits.columns
    assert list(visits[visits["person_id"] == 33]["visit_concept_name_substr"])[0] == (
        "Out"
    )
