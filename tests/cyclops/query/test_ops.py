"""Test low-level query API processing functions."""

from math import isclose

import pandas as pd
import pytest
from sqlalchemy import column, select

from cyclops.query.omop import OMOPQuerier
from cyclops.query.ops import (
    AddColumn,
    AddNumeric,
    Apply,
    Cast,
    ConditionAfterDate,
    ConditionBeforeDate,
    ConditionEndsWith,
    ConditionEquals,
    ConditionGreaterThan,
    ConditionIn,
    ConditionInMonths,
    ConditionInYears,
    ConditionLessThan,
    ConditionLike,
    ConditionRegexMatch,
    ConditionStartsWith,
    ConditionSubstring,
    Distinct,
    Drop,
    DropNulls,
    ExtractTimestampComponent,
    FillNull,
    GroupByAggregate,
    Limit,
    Literal,
    OrderBy,
    Rename,
    ReorderAfter,
    Sequential,
    Substring,
    Trim,
    Union,
    _none_add,
    _process_checks,
)
from cyclops.query.util import process_column


QUERIER = OMOPQuerier("cdm_synthea10", database="synthea_integration_test")


@pytest.fixture
def table_input():
    """Test table input."""
    column_a = process_column(column("a"), to_timestamp=True)
    return select(column_a, column("b"), column("c"))


@pytest.fixture
def visits_input():
    """Test visits table input."""
    return QUERIER.visit_occurrence().query


@pytest.fixture
def measurements_input():
    """Test measurement table input."""
    return QUERIER.measurement().query


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
    visits = QUERIER.get_interface(visits).run()
    assert "care_site_source_value" not in visits.columns


@pytest.mark.integration_test
def test_fill_null(visits_input):  # pylint: disable=redefined-outer-name
    """Test FillNull."""
    visits_before = QUERIER.get_interface(visits_input).run()
    unique_before = visits_before["preceding_visit_occurrence_id"].unique()
    visits = FillNull(["preceding_visit_occurrence_id", "care_site_id"], 0)(
        visits_input,
    )
    visits_after = QUERIER.get_interface(visits).run()
    unique_after = visits_after["preceding_visit_occurrence_id"].unique()
    assert visits_after["preceding_visit_occurrence_id"].isna().sum() == 0
    assert visits_after["care_site_id"].isna().sum() == 0
    assert 0 not in unique_before
    assert len(unique_after) == len(unique_before)
    assert len(visits_after["care_site_id"].unique()) == 1

    visits = FillNull(
        ["preceding_visit_occurrence_id", "care_site_id"],
        [0, -99],
        ["col1", "col2"],
    )(visits_input)
    visits_after = QUERIER.get_interface(visits).run()
    assert visits_after["preceding_visit_occurrence_id"].isna().sum() != 0
    assert visits_after["care_site_id"].isna().sum() != 0
    assert visits_after["col1"].isna().sum() == 0
    assert visits_after["col2"].isna().sum() == 0
    assert len(visits_after["col2"].unique()) == 1
    assert -99 in visits_after["col2"].unique()


@pytest.mark.integration_test
def test_add_column(visits_input):  # pylint: disable=redefined-outer-name
    """Test AddColumn."""
    ops = Sequential(
        [
            Literal(2, "test_col1"),
            Literal(3, "test_col2"),
            AddColumn("test_col1", "test_col2", new_col_labels="test_col3"),
        ],
    )
    visits = QUERIER.get_interface(visits_input, ops=ops).run()
    assert "test_col3" in visits.columns
    assert (visits["test_col3"] == 5).all()

    ops = Sequential(
        [
            Literal(2, "test_col1"),
            Literal(3, "test_col2"),
            AddColumn(
                "test_col1",
                "test_col2",
                negative=True,
                new_col_labels="test_col3",
            ),
        ],
    )
    visits = QUERIER.get_interface(visits_input, ops=ops).run()
    assert "test_col3" in visits.columns
    assert (visits["test_col3"] == -1).all()


@pytest.mark.integration_test
def test_rename(visits_input):  # pylint: disable=redefined-outer-name
    """Test Rename."""
    visits = Rename({"care_site_name": "hospital_name"})(visits_input)
    visits = QUERIER.get_interface(visits).run()
    assert "hospital_name" in visits.columns
    assert "care_site_name" not in visits.columns


@pytest.mark.integration_test
def test_literal(visits_input):  # pylint: disable=redefined-outer-name
    """Test Literal."""
    visits = Literal(1, "new_col")(visits_input)
    visits = Literal("a", "new_col2")(visits)
    visits = QUERIER.get_interface(visits).run()
    assert "new_col" in visits.columns
    assert visits["new_col"].iloc[0] == 1
    assert "new_col2" in visits.columns
    assert visits["new_col2"].iloc[0] == "a"


@pytest.mark.integration_test
def test_reorder_after(visits_input):  # pylint: disable=redefined-outer-name
    """Test ReorderAfter."""
    visits = ReorderAfter("visit_concept_name", "care_site_id")(visits_input)
    visits = QUERIER.get_interface(visits).run()
    assert list(visits.columns).index("care_site_id") + 1 == list(visits.columns).index(
        "visit_concept_name",
    )


@pytest.mark.integration_test
def test_limit(visits_input):  # pylint: disable=redefined-outer-name
    """Test Limit."""
    visits = Limit(10)(visits_input)
    visits = QUERIER.get_interface(visits).run()
    assert len(visits) == 10


@pytest.mark.integration_test
def test_order_by(visits_input):  # pylint: disable=redefined-outer-name
    """Test OrderBy."""
    visits = OrderBy("visit_concept_name")(visits_input)
    visits = QUERIER.get_interface(visits).run()
    assert visits["visit_concept_name"].is_monotonic_increasing


@pytest.mark.integration_test
def test_substring(visits_input):  # pylint: disable=redefined-outer-name
    """Test Substring."""
    visits = Substring("visit_concept_name", 0, 3, "visit_concept_name_substr")(
        visits_input,
    )
    visits = QUERIER.get_interface(visits).run()
    assert visits["visit_concept_name_substr"].iloc[0] == "In"


@pytest.mark.integration_test
def test_trim(visits_input):  # pylint: disable=redefined-outer-name
    """Test Trim."""
    visits = Trim("visit_concept_name", "visit_concept_name_trim")(visits_input)
    visits = QUERIER.get_interface(visits).run()
    assert visits["visit_concept_name_trim"].iloc[0] == "Inpatient Visit"


@pytest.mark.integration_test
def test_extract_timestamp_component(
    visits_input,
):  # pylint: disable=redefined-outer-name
    """Test ExtractTimestampComponent."""
    visits = ExtractTimestampComponent(
        "visit_start_date",
        "year",
        "visit_start_date_year",
    )(visits_input)
    visits = QUERIER.get_interface(visits).run()
    assert visits["visit_start_date_year"].iloc[0] == 2021


@pytest.mark.integration_test
def test_add_numeric(visits_input):  # pylint: disable=redefined-outer-name
    """Test AddNumeric."""
    visits = Literal(1, "new_col")(visits_input)
    visits = AddNumeric("new_col", 1, "new_col_plus_1")(visits)
    visits = QUERIER.get_interface(visits).run()
    assert visits["new_col_plus_1"].iloc[0] == 2


@pytest.mark.integration_test
def test_apply(visits_input):  # pylint: disable=redefined-outer-name
    """Test Apply."""
    visits = Apply(
        "visit_concept_name",
        lambda x: x + "!",
        "visit_concept_name_exclaim",
    )(visits_input)
    visits = QUERIER.get_interface(visits).run()
    assert visits["visit_concept_name_exclaim"].iloc[0] == "Inpatient Visit!"
    visits = Apply(
        ["visit_occurrence_id", "preceding_visit_occurrence_id"],
        lambda x, y: x + y,
        "sum_id",
    )(visits_input)
    visits = QUERIER.get_interface(visits).run()
    assert (
        visits["sum_id"].iloc[0]
        == visits["visit_occurrence_id"].iloc[0]
        + visits["preceding_visit_occurrence_id"].iloc[0]
    )
    assert (
        visits["sum_id"].isna().sum()
        == visits["preceding_visit_occurrence_id"].isna().sum()
    )


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
    measurements = QUERIER.get_interface(measurements).run()
    assert "value_source_value_match" in measurements.columns
    assert (
        measurements["value_source_value_match"].sum()
        == measurements["value_source_value"].str.match(r"^[0-9]+(\.[0-9]+)?$").sum()
    )


@pytest.mark.integration_test
def test_group_by_aggregate(  # pylint: disable=redefined-outer-name
    visits_input,
    measurements_input,
):
    """Test GroupByAggregate."""
    with pytest.raises(ValueError):
        GroupByAggregate("person_id", {"person_id": ("donkey", "visit_count")})(
            visits_input,
        )
    with pytest.raises(ValueError):
        GroupByAggregate("person_id", {"person_id": ("count", "person_id")})(
            visits_input,
        )

    visits_count = GroupByAggregate(
        "person_id",
        {"person_id": ("count", "num_visits")},
    )(visits_input)
    visits_string_agg = GroupByAggregate(
        "person_id",
        {"visit_concept_name": ("string_agg", "visit_concept_names")},
        {"visit_concept_name": ", "},
    )(visits_input)
    measurements_sum = GroupByAggregate(
        "person_id",
        {"value_as_number": ("sum", "value_as_number_sum")},
    )(measurements_input)
    measurements_average = GroupByAggregate(
        "person_id",
        {"value_as_number": ("average", "value_as_number_average")},
    )(measurements_input)
    measurements_min = GroupByAggregate(
        "person_id",
        {"value_as_number": ("min", "value_as_number_min")},
    )(measurements_input)
    measurements_max = GroupByAggregate(
        "person_id",
        {"value_as_number": ("max", "value_as_number_max")},
    )(measurements_input)
    measurements_median = GroupByAggregate(
        "person_id",
        {"value_as_number": ("median", "value_as_number_median")},
    )(measurements_input)

    visits_count = QUERIER.get_interface(visits_count).run()
    visits_string_agg = QUERIER.get_interface(visits_string_agg).run()
    measurements_sum = QUERIER.get_interface(measurements_sum).run()
    measurements_average = QUERIER.get_interface(measurements_average).run()
    measurements_min = QUERIER.get_interface(measurements_min).run()
    measurements_max = QUERIER.get_interface(measurements_max).run()
    measurements_median = QUERIER.get_interface(measurements_median).run()

    assert "num_visits" in visits_count.columns
    assert visits_count[visits_count["person_id"] == 33]["num_visits"][0] == 86
    assert "visit_concept_names" in visits_string_agg.columns
    test_visit_concept_names = visits_string_agg[visits_string_agg["person_id"] == 33][
        "visit_concept_names"
    ][0].split(",")
    test_visit_concept_names = [item.strip() for item in test_visit_concept_names]
    assert (
        len(test_visit_concept_names) == 86
        and "Outpatient Visit" in test_visit_concept_names
    )
    assert "value_as_number_sum" in measurements_sum.columns
    assert (
        measurements_sum[measurements_sum["person_id"] == 33]["value_as_number_sum"][0]
        == 9881.3
    )
    assert "value_as_number_average" in measurements_average.columns
    assert isclose(
        measurements_average[measurements_average["person_id"] == 33][
            "value_as_number_average"
        ][0],
        75.42,
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
        == 360.7
    )
    assert "value_as_number_median" in measurements_median.columns
    assert (
        measurements_median[measurements_median["person_id"] == 33][
            "value_as_number_median"
        ].item()
        == 75.7
    )


@pytest.mark.integration_test
def test_drop_nulls(visits_input):  # pylint: disable=redefined-outer-name
    """Test DropNulls."""
    visits = DropNulls("preceding_visit_occurrence_id")(visits_input)
    visits = QUERIER.get_interface(visits).run()
    assert visits["preceding_visit_occurrence_id"].isnull().sum() == 0


@pytest.mark.integration_test
def test_condition_before_date(visits_input):  # pylint: disable=redefined-outer-name
    """Test ConditionBeforeDate."""
    visits = ConditionBeforeDate("visit_start_date", "2018-01-01")(visits_input)
    visits = QUERIER.get_interface(visits).run()
    assert pd.Timestamp(visits["visit_start_date"].max()) < pd.Timestamp("2018-01-01")


@pytest.mark.integration_test
def test_condition_after_date(visits_input):  # pylint: disable=redefined-outer-name
    """Test ConditionAfterDate."""
    visits = ConditionAfterDate("visit_start_date", "2018-01-01")(visits_input)
    visits = QUERIER.get_interface(visits).run()
    assert pd.Timestamp(visits["visit_start_date"].min()) > pd.Timestamp("2018-01-01")


@pytest.mark.integration_test
def test_condition_in(visits_input):  # pylint: disable=redefined-outer-name
    """Test ConditionIn."""
    visits = ConditionIn("visit_concept_name", ["Outpatient Visit"])(visits_input)
    visits = QUERIER.get_interface(visits).run()
    assert all(visits["visit_concept_name"] == "Outpatient Visit")


@pytest.mark.integration_test
def test_condition_in_months(visits_input):  # pylint: disable=redefined-outer-name
    """Test ConditionInMonths."""
    visits_input = Cast("visit_start_date", "timestamp")(visits_input)
    visits = ConditionInMonths("visit_start_date", 6)(visits_input)
    visits = QUERIER.get_interface(visits).run()
    assert (visits["visit_start_date"].dt.month == 6).all()


@pytest.mark.integration_test
def test_condition_in_years(visits_input):  # pylint: disable=redefined-outer-name
    """Test ConditionInYears."""
    visits_input = Cast("visit_start_date", "timestamp")(visits_input)
    visits = ConditionInYears("visit_start_date", 2018)(visits_input)
    visits = QUERIER.get_interface(visits).run()
    assert (visits["visit_start_date"].dt.year == 2018).all()


@pytest.mark.integration_test
def test_condition_substring(visits_input):  # pylint: disable=redefined-outer-name
    """Test ConditionSubstring."""
    visits = ConditionSubstring("visit_concept_name", "Outpatient")(visits_input)
    visits = QUERIER.get_interface(visits).run()
    assert all(visits["visit_concept_name"].str.contains("Outpatient"))


@pytest.mark.integration_test
def test_condition_starts_with(visits_input):  # pylint: disable=redefined-outer-name
    """Test ConditionStartsWith."""
    visits = ConditionStartsWith("visit_concept_name", "Outpatient")(visits_input)
    visits = QUERIER.get_interface(visits).run()
    assert all(visits["visit_concept_name"].str.startswith("Outpatient"))


@pytest.mark.integration_test
def test_condition_ends_with(visits_input):  # pylint: disable=redefined-outer-name
    """Test ConditionEndsWith."""
    visits = ConditionEndsWith("visit_concept_name", "Visit")(visits_input)
    visits = QUERIER.get_interface(visits).run()
    assert all(visits["visit_concept_name"].str.endswith("Visit"))


@pytest.mark.integration_test
def test_condition_equals(visits_input):  # pylint: disable=redefined-outer-name
    """Test ConditionEquals."""
    visits = ConditionEquals("visit_concept_name", "Outpatient Visit")(visits_input)
    visits = QUERIER.get_interface(visits).run()
    assert all(visits["visit_concept_name"] == "Outpatient Visit")
    visits = ConditionEquals("visit_concept_name", "Outpatient Visit", not_=True)(
        visits_input,
    )
    visits = QUERIER.get_interface(visits).run()
    assert all(visits["visit_concept_name"] != "Outpatient Visit")


@pytest.mark.integration_test
def test_condition_greater_than(visits_input):  # pylint: disable=redefined-outer-name
    """Test ConditionGreaterThan."""
    visits = ConditionGreaterThan("visit_concept_id", 9300)(visits_input)
    visits = QUERIER.get_interface(visits).run()
    assert all(visits["visit_concept_id"] > 9300)


@pytest.mark.integration_test
def test_condition_less_than(visits_input):  # pylint: disable=redefined-outer-name
    """Test ConditionLessThan."""
    visits = ConditionLessThan("visit_concept_id", 9300)(visits_input)
    visits = QUERIER.get_interface(visits).run()
    assert all(visits["visit_concept_id"] < 9300)


@pytest.mark.integration_test
def test_union(visits_input):  # pylint: disable=redefined-outer-name
    """Test Union."""
    visits = Union(
        ConditionEquals("visit_concept_name", "Outpatient Visit")(visits_input),
    )(ConditionEquals("visit_concept_name", "Emergency Room Visit")(visits_input))
    visits = QUERIER.get_interface(visits).run()
    assert len(visits) == 4212
    assert all(
        visits["visit_concept_name"].isin(["Outpatient Visit", "Emergency Room Visit"]),
    )
    visits = Union(
        ConditionEquals("visit_concept_name", "Outpatient Visit")(visits_input),
        union_all=True,
    )(ConditionEquals("visit_concept_name", "Outpatient Visit")(visits_input))
    visits = QUERIER.get_interface(visits).run()
    assert len(visits) == 8114


@pytest.mark.integration_test
def test_sequential(visits_input):  # pylint: disable=redefined-outer-name
    """Test Sequential."""
    substr_op = Sequential(
        [
            Substring("visit_concept_name", 0, 4, "visit_concept_name_substr"),
        ],
    )
    operations = [
        Literal(33, "const"),
        Rename({"care_site_name": "hospital_name"}),
        Apply("visit_concept_name", lambda x: x + "!", "visit_concept_name_exclaim"),
        OrderBy(["person_id", "visit_start_date"]),
        substr_op,
    ]
    sequential_ops = Sequential(operations)
    visits = QUERIER.get_interface(visits_input, ops=sequential_ops).run()
    assert "hospital_name" in visits.columns
    assert "visit_concept_name_exclaim" in visits.columns
    assert list(visits[visits["person_id"] == 33]["visit_concept_name_exclaim"])[0] == (
        "Outpatient Visit!"
    )
    assert "visit_concept_name_substr" in visits.columns
    assert list(visits[visits["person_id"] == 33]["visit_concept_name_substr"])[0] == (
        "Out"
    )


@pytest.mark.integration_test
def test_distinct(visits_input):  # pylint: disable=redefined-outer-name
    """Test Distinct."""
    distinct_op = Distinct(["person_id"])
    visits = QUERIER.get_interface(visits_input, ops=distinct_op).run()
    assert len(visits) == 109
    visits = QUERIER.get_interface(visits_input).run()


@pytest.mark.integration_test
def test_condition_like(visits_input):  # pylint: disable=redefined-outer-name
    """Test ConditionLike."""
    like_op = ConditionLike("visit_concept_name", "Outpatient%")
    visits = QUERIER.get_interface(visits_input, ops=like_op).run()
    assert len(visits) == 4057
    assert all(visits["visit_concept_name"].str.startswith("Outpatient"))
