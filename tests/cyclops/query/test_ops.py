"""Test low-level query API processing functions."""

from math import isclose

import pandas as pd
import pytest
from sqlalchemy import column, select

from cyclops.query.omop import OMOPQuerier
from cyclops.query.ops import (
    AddColumn,
    AddNumeric,
    And,
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
    DropEmpty,
    DropNulls,
    ExtractTimestampComponent,
    FillNull,
    GroupByAggregate,
    Limit,
    Literal,
    Or,
    OrderBy,
    QueryOp,
    Rename,
    ReorderAfter,
    Sequential,
    Substring,
    Trim,
    _addindent,
    _none_add,
    _process_checks,
)
from cyclops.query.util import process_column


QUERIER = OMOPQuerier(
    database="synthea_integration_test",
    user="postgres",
    password="pwd",
    schema_name="cdm_synthea10",
)


@pytest.fixture()
def table_input():
    """Test table input."""
    column_a = process_column(column("a"), to_timestamp=True)
    return select(column_a, column("b"), column("c"))


@pytest.fixture()
def visits_table():
    """Test visits table input."""
    return QUERIER.visit_occurrence()


@pytest.fixture()
def measurements_table():
    """Test measurement table input."""
    return QUERIER.measurement()


def test__none_add():
    """Test _none_add fn."""
    assert _none_add("1", "2") == "12"
    assert _none_add("1", None) == "1"
    assert _none_add(None, "2") == "2"


def test__process_checks(table_input):
    """Test _process_checks fn."""
    _process_checks(table_input, cols=["a"], cols_not_in=["d"], timestamp_cols=["a"])
    with pytest.raises(ValueError):
        _process_checks(table_input, cols_not_in=["a"])


class TestAddndent:
    """Test _addindent fn."""

    def test_addindent_multiple_lines(self):
        """Test _addindent fn with multiple lines."""
        input_string = "This is a\nmultiline\nstring"
        expected_output = "This is a\n    multiline\n    string"
        assert _addindent(input_string, 4) == expected_output

    def test_addindent_single_line(self):
        """Test _addindent fn with single line."""
        input_string = "This is a single line string"
        assert _addindent(input_string, 4) == input_string


class TestQueryOp:
    """Test QueryOp class."""

    def test_add_child_operation(self):
        """Test adding a child operation."""
        query_op = QueryOp()
        child_op = QueryOp()
        query_op._add_op("child", child_op)
        assert query_op.child == child_op

    def test_get_query_op_name(self):
        """Test getting the name of the query op."""
        query_op = QueryOp()
        assert query_op._get_name() == "QueryOp"

    def test_set_attribute(self):
        """Test setting an attribute of the query op."""
        query_op = QueryOp()
        child_op = QueryOp()
        query_op.child = child_op
        assert query_op.child == child_op

    def test_string_representation(self):
        """Test string representation of the query op."""
        query_op = QueryOp()
        child_op = QueryOp()
        query_op._add_op("child", child_op)
        assert repr(query_op) == "QueryOp(\n  (child): QueryOp()\n)"

    def test_add_child_operation_empty_name(self):
        """Test adding a child operation with an empty name."""
        query_op = QueryOp()
        child_op = QueryOp()
        with pytest.raises(KeyError):
            query_op._add_op("", child_op)

    def test_add_child_operation_dot_name(self):
        """Test adding a child operation with a dot in the name."""
        query_op = QueryOp()
        child_op = QueryOp()
        with pytest.raises(KeyError):
            query_op._add_op("child.name", child_op)


@pytest.mark.integration_test()
def test_drop(visits_table):
    """Test Drop."""
    visits = visits_table.ops(Drop("care_site_source_value")).run()
    assert "care_site_source_value" not in visits.columns


@pytest.mark.integration_test()
def test_fill_null(visits_table):
    """Test FillNull."""
    visits_before = visits_table.run()
    unique_before = visits_before["preceding_visit_occurrence_id"].unique()
    visits_after = visits_table.ops(
        FillNull(["preceding_visit_occurrence_id", "care_site_id"], 0),
    ).run()
    unique_after = visits_after["preceding_visit_occurrence_id"].unique()
    assert visits_after["preceding_visit_occurrence_id"].isna().sum() == 0
    assert visits_after["care_site_id"].isna().sum() == 0
    assert 0 not in unique_before
    assert len(unique_after) == len(unique_before)
    assert len(visits_after["care_site_id"].unique()) == 1

    visits_after = visits_table.ops(
        FillNull(
            ["preceding_visit_occurrence_id", "care_site_id"],
            [0, -99],
            ["col1", "col2"],
        ),
    ).run()
    assert visits_after["preceding_visit_occurrence_id"].isna().sum() != 0
    assert visits_after["care_site_id"].isna().sum() != 0
    assert visits_after["col1"].isna().sum() == 0
    assert visits_after["col2"].isna().sum() == 0
    assert len(visits_after["col2"].unique()) == 1
    assert -99 in visits_after["col2"].unique()


@pytest.mark.integration_test()
def test_add_column(visits_table):
    """Test AddColumn."""
    ops = Sequential(
        Literal(2, "test_col1"),
        Literal(3, "test_col2"),
        AddColumn("test_col1", "test_col2", new_col_labels="test_col3"),
    )
    visits = visits_table.ops(ops).run()
    assert "test_col3" in visits.columns
    assert (visits["test_col3"] == 5).all()

    ops = Sequential(
        Literal(2, "test_col1"),
        Literal(3, "test_col2"),
        AddColumn(
            "test_col1",
            "test_col2",
            negative=True,
            new_col_labels="test_col3",
        ),
    )
    visits = visits_table.ops(ops).run()
    assert "test_col3" in visits.columns
    assert (visits["test_col3"] == -1).all()


@pytest.mark.integration_test()
def test_rename(visits_table):
    """Test Rename."""
    rename_op = Rename({"care_site_name": "hospital_name"})
    visits = visits_table.ops(rename_op).run()
    assert "hospital_name" in visits.columns
    assert "care_site_name" not in visits.columns


@pytest.mark.integration_test()
def test_literal(visits_table):
    """Test Literal."""
    literal_ops = Sequential(Literal(1, "new_col"), Literal("a", "new_col2"))
    visits = visits_table.ops(literal_ops).run()
    assert "new_col" in visits.columns
    assert visits["new_col"].iloc[0] == 1
    assert "new_col2" in visits.columns
    assert visits["new_col2"].iloc[0] == "a"


@pytest.mark.integration_test()
def test_reorder_after(visits_table):
    """Test ReorderAfter."""
    reorder_op = ReorderAfter("visit_concept_name", "care_site_id")
    visits = visits_table.ops(reorder_op).run()
    assert list(visits.columns).index("care_site_id") + 1 == list(visits.columns).index(
        "visit_concept_name",
    )


@pytest.mark.integration_test()
def test_limit(visits_table):
    """Test Limit."""
    visits = visits_table.ops(Limit(10)).run()
    assert len(visits) == 10


@pytest.mark.integration_test()
def test_order_by(visits_table):
    """Test OrderBy."""
    orderby_op = OrderBy("visit_concept_name")
    visits = visits_table.ops(orderby_op).run()
    assert visits["visit_concept_name"].is_monotonic_increasing


@pytest.mark.integration_test()
def test_substring(visits_table):
    """Test Substring."""
    substring_op = Substring("visit_concept_name", 0, 3, "visit_concept_name_substr")
    visits = visits_table.ops(substring_op).run()
    assert visits["visit_concept_name_substr"].iloc[0] == "In"


@pytest.mark.integration_test()
def test_trim(visits_table):
    """Test Trim."""
    trim_op = Trim("visit_concept_name", "visit_concept_name_trim")
    visits = visits_table.ops(trim_op).run()
    assert visits["visit_concept_name_trim"].iloc[0] == "Inpatient Visit"


@pytest.mark.integration_test()
def test_extract_timestamp_component(
    visits_table,
):
    """Test ExtractTimestampComponent."""
    extract_ts_op = ExtractTimestampComponent(
        "visit_start_date",
        "year",
        "visit_start_date_year",
    )
    visits = visits_table.ops(extract_ts_op).run()
    assert visits["visit_start_date_year"].iloc[0] == 2021


@pytest.mark.integration_test()
def test_add_numeric(visits_table):
    """Test AddNumeric."""
    ops = Sequential(Literal(1, "new_col"), AddNumeric("new_col", 1, "new_col_plus_1"))
    visits = visits_table.ops(ops).run()
    assert visits["new_col_plus_1"].iloc[0] == 2


@pytest.mark.integration_test()
def test_apply(visits_table):
    """Test Apply."""
    apply_op = Apply(
        "visit_concept_name",
        lambda x: x + "!",
        "visit_concept_name_exclaim",
    )
    visits = visits_table.ops(apply_op).run()
    assert visits["visit_concept_name_exclaim"].iloc[0] == "Inpatient Visit!"
    apply_op = Apply(
        ["visit_occurrence_id", "preceding_visit_occurrence_id"],
        lambda x, y: x + y,
        "sum_id",
    )
    visits = visits_table.ops(apply_op).run()
    assert (
        visits["sum_id"].iloc[0]
        == visits["visit_occurrence_id"].iloc[0]
        + visits["preceding_visit_occurrence_id"].iloc[0]
    )
    assert (
        visits["sum_id"].isna().sum()
        == visits["preceding_visit_occurrence_id"].isna().sum()
    )
    apply_op = Apply(
        ["visit_occurrence_id", "preceding_visit_occurrence_id"],
        [lambda x: x + 1, lambda x: x + 2],
        ["sum_id", "sum_id2"],
    )
    visits = visits_table.ops(apply_op).run()
    assert visits["sum_id"].iloc[0] == visits["visit_occurrence_id"].iloc[0] + 1
    assert (
        visits["sum_id2"].iloc[0] == visits["preceding_visit_occurrence_id"].iloc[0] + 2
    )


@pytest.mark.integration_test()
def test_condition_regex_match(
    measurements_table,
):
    """Test ConditionRegexMatch."""
    measurements_op = ConditionRegexMatch(
        "value_source_value",
        r"^[0-9]+(\.[0-9]+)?$",
        binarize_col="value_source_value_match",
    )
    measurements = measurements_table.ops(measurements_op).run()
    assert "value_source_value_match" in measurements.columns
    assert (
        measurements["value_source_value_match"].sum()
        == measurements["value_source_value"].str.match(r"^[0-9]+(\.[0-9]+)?$").sum()
    )


@pytest.mark.integration_test()
def test_group_by_aggregate(
    visits_table,
    measurements_table,
):
    """Test GroupByAggregate."""
    with pytest.raises(ValueError):
        visits_table.ops(
            GroupByAggregate("person_id", {"person_id": ("donkey", "visit_count")}),
        )
    with pytest.raises(ValueError):
        visits_table.ops(
            GroupByAggregate("person_id", {"person_id": ("count", "person_id")}),
        )

    visits_count = visits_table.ops(
        GroupByAggregate(
            "person_id",
            {"person_id": ("count", "num_visits")},
        ),
    ).run()
    visits_string_agg = visits_table.ops(
        GroupByAggregate(
            "person_id",
            {"visit_concept_name": ("string_agg", "visit_concept_names")},
            {"visit_concept_name": ", "},
        ),
    ).run()
    measurements_sum = measurements_table.ops(
        GroupByAggregate(
            "person_id",
            {"value_as_number": ("sum", "value_as_number_sum")},
        ),
    ).run()
    measurements_average = measurements_table.ops(
        GroupByAggregate(
            "person_id",
            {"value_as_number": ("average", "value_as_number_average")},
        ),
    ).run()
    measurements_min = measurements_table.ops(
        GroupByAggregate(
            "person_id",
            {"value_as_number": ("min", "value_as_number_min")},
        ),
    ).run()
    measurements_max = measurements_table.ops(
        GroupByAggregate(
            "person_id",
            {"value_as_number": ("max", "value_as_number_max")},
        ),
    ).run()
    measurements_median = measurements_table.ops(
        GroupByAggregate(
            "person_id",
            {"value_as_number": ("median", "value_as_number_median")},
        ),
    ).run()

    assert "num_visits" in visits_count.columns
    assert visits_count[visits_count["person_id"] == 33]["num_visits"][0] == 86
    assert "visit_concept_names" in visits_string_agg.columns
    test_visit_concept_names = visits_string_agg[visits_string_agg["person_id"] == 33][
        "visit_concept_names"
    ][0].split(",")
    test_visit_concept_names = [item.strip() for item in test_visit_concept_names]
    assert len(test_visit_concept_names) == 86
    assert "Outpatient Visit" in test_visit_concept_names
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


@pytest.mark.integration_test()
def test_drop_nulls(visits_table):
    """Test DropNulls."""
    visits = visits_table.ops(DropNulls("preceding_visit_occurrence_id")).run()
    assert visits["preceding_visit_occurrence_id"].isnull().sum() == 0


@pytest.mark.integration_test()
def test_drop_empty(visits_table):
    """Test DropEmpty."""
    visits = visits_table.ops(DropEmpty("visit_concept_name")).run()
    assert (visits["visit_concept_name"] == "").sum() == 0


@pytest.mark.integration_test()
def test_condition_before_date(visits_table):
    """Test ConditionBeforeDate."""
    visits = visits_table.ops(
        ConditionBeforeDate("visit_start_date", "2018-01-01"),
    ).run()
    assert pd.Timestamp(visits["visit_start_date"].max()) < pd.Timestamp("2018-01-01")


@pytest.mark.integration_test()
def test_condition_after_date(visits_table):
    """Test ConditionAfterDate."""
    visits = visits_table.ops(
        ConditionAfterDate("visit_start_date", "2018-01-01"),
    ).run()
    assert pd.Timestamp(visits["visit_start_date"].min()) > pd.Timestamp("2018-01-01")


@pytest.mark.integration_test()
def test_condition_in(visits_table):
    """Test ConditionIn."""
    visits = visits_table.ops(
        ConditionIn("visit_concept_name", ["Outpatient Visit"]),
    ).run()
    assert all(visits["visit_concept_name"] == "Outpatient Visit")


@pytest.mark.integration_test()
def test_condition_in_months(visits_table):
    """Test ConditionInMonths."""
    ops = Sequential(
        Cast("visit_start_date", "timestamp"),
        ConditionInMonths("visit_start_date", 6),
    )
    visits = visits_table.ops(ops).run()
    assert (visits["visit_start_date"].dt.month == 6).all()


@pytest.mark.integration_test()
def test_condition_in_years(visits_table):
    """Test ConditionInYears."""
    ops = Sequential(
        Cast("visit_start_date", "timestamp"),
        ConditionInYears("visit_start_date", 2018),
    )
    visits = visits_table.ops(ops).run()
    assert (visits["visit_start_date"].dt.year == 2018).all()


@pytest.mark.integration_test()
def test_condition_substring(visits_table):
    """Test ConditionSubstring."""
    visits = visits_table.ops(
        ConditionSubstring("visit_concept_name", "Outpatient"),
    ).run()
    assert all(visits["visit_concept_name"].str.contains("Outpatient"))


@pytest.mark.integration_test()
def test_condition_starts_with(visits_table):
    """Test ConditionStartsWith."""
    visits = visits_table.ops(
        ConditionStartsWith("visit_concept_name", "Outpatient"),
    ).run()
    assert all(visits["visit_concept_name"].str.startswith("Outpatient"))


@pytest.mark.integration_test()
def test_condition_ends_with(visits_table):
    """Test ConditionEndsWith."""
    visits = visits_table.ops(ConditionEndsWith("visit_concept_name", "Visit")).run()
    assert all(visits["visit_concept_name"].str.endswith("Visit"))


@pytest.mark.integration_test()
def test_condition_equals(visits_table):
    """Test ConditionEquals."""
    visits = visits_table.ops(
        ConditionEquals("visit_concept_name", "Outpatient Visit"),
    ).run()
    assert all(visits["visit_concept_name"] == "Outpatient Visit")
    visits = visits_table.ops(
        ConditionEquals("visit_concept_name", "Outpatient Visit", not_=True),
    ).run()
    assert all(visits["visit_concept_name"] != "Outpatient Visit")


@pytest.mark.integration_test()
def test_condition_greater_than(visits_table):
    """Test ConditionGreaterThan."""
    visits = visits_table.ops(ConditionGreaterThan("visit_concept_id", 9300)).run()
    assert all(visits["visit_concept_id"] > 9300)


@pytest.mark.integration_test()
def test_condition_less_than(visits_table):
    """Test ConditionLessThan."""
    visits = visits_table.ops(ConditionLessThan("visit_concept_id", 9300)).run()
    assert all(visits["visit_concept_id"] < 9300)


@pytest.mark.integration_test()
def test_union(visits_table):
    """Test Union."""
    outpatient_filtered = visits_table.ops(
        ConditionEquals("visit_concept_name", "Outpatient Visit"),
    )
    emergency_filtered = visits_table.ops(
        ConditionEquals("visit_concept_name", "Emergency Room Visit"),
    )
    visits = emergency_filtered.union(outpatient_filtered).run()
    assert len(visits) == 4212
    assert all(
        visits["visit_concept_name"].isin(["Outpatient Visit", "Emergency Room Visit"]),
    )
    visits = emergency_filtered.union_all(emergency_filtered).run()
    assert len(visits) == 310


@pytest.mark.integration_test()
def test_sequential(visits_table):
    """Test Sequential."""
    substr_op = Substring("visit_concept_name", 0, 4, "visit_concept_name_substr")
    operations = [
        Literal(33, "const"),
        Rename({"care_site_name": "hospital_name"}),
        Apply("visit_concept_name", lambda x: x + "!", "visit_concept_name_exclaim"),
        OrderBy(["person_id", "visit_start_date"]),
        substr_op,
    ]
    sequential_ops = Sequential(operations)
    visits = visits_table.ops(sequential_ops).run()
    assert "hospital_name" in visits.columns
    assert "visit_concept_name_exclaim" in visits.columns
    assert list(visits[visits["person_id"] == 33]["visit_concept_name_exclaim"])[0] == (
        "Outpatient Visit!"
    )
    assert "visit_concept_name_substr" in visits.columns
    assert list(visits[visits["person_id"] == 33]["visit_concept_name_substr"])[0] == (
        "Out"
    )


@pytest.mark.integration_test()
def test_or(visits_table):
    """Test Or."""
    or_op = Or(
        ConditionEquals("visit_concept_name", "Outpatient Visit"),
        ConditionLike("visit_concept_name", "%Emergency%"),
    )
    visits = visits_table.ops(or_op).run()
    assert len(visits) == 4212
    assert all(
        visits["visit_concept_name"].isin(["Outpatient Visit", "Emergency Room Visit"]),
    )


@pytest.mark.integration_test()
def test_and(visits_table):
    """Test And."""
    and_op = And(
        [
            ConditionEquals("visit_concept_name", "Outpatient Visit"),
            ConditionLike("visit_concept_name", "%Emergency%", not_=True),
        ],
    )
    visits = visits_table.ops(and_op).run()
    assert len(visits) == 4057
    and_op = And(
        ConditionEquals("visit_concept_name", "Outpatient Visit"),
        ConditionLike("visit_concept_name", "%Emergency%", not_=True),
    )
    visits = visits_table.ops(and_op).run()
    assert len(visits) == 4057


@pytest.mark.integration_test()
def test_distinct(visits_table):
    """Test Distinct."""
    distinct_op = Distinct(["person_id"])
    visits = visits_table.ops(distinct_op).run()
    assert len(visits) == 109


@pytest.mark.integration_test()
def test_condition_like(visits_table):
    """Test ConditionLike."""
    like_op = ConditionLike("visit_concept_name", "Outpatient%")
    visits = visits_table.ops(like_op).run()
    assert len(visits) == 4057
    assert all(visits["visit_concept_name"].str.startswith("Outpatient"))
