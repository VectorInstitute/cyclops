from typing import List, Optional, Union
import warnings

import datetime
from datetime import timedelta

from dateutil import parser as du_parser
from dateutil.parser import ParserError

# import datefinder

import numpy as np
import pandas as pd

from fecg.utils.pandas.pandas import check_cols
from fecg.utils.pandas.type import is_datetime_series, is_str_series

# Datetime component names
DATE_COMPONENTS = ["year", "month", "day"]
TIME_COMPONENTS = ["hour", "minute", "second", "microsecond"]
DT_COMPONENTS = DATE_COMPONENTS + TIME_COMPONENTS

# Parsing results for pd.to_datetime (PD_DT) and the dateutil parser (DU_DT)
PD_DT = "pd"
DU_DT = "du"
DU_TO_PD_DT = f"{DU_DT}_to_{PD_DT}"


def datetime_to_unix(series: pd.Series) -> pd.Series:
    """
    Convert a datetime series to UNIX timestamps.

    Parameters
    ----------
    series : pandas.Series
        Datetime series.

    Returns
    -------
    pd.Series
        Series containing UNIX timestamps corresponding to the datetime values.
    """
    is_datetime_series(series, raise_err=True)

    return series.astype(int) / 10**9


def unix_to_datetime(series: pd.Series) -> pd.Series:
    """
    Convert a series of UNIX timestamps to datetime.

    Parameters
    ----------
    series : pandas.Series
        Series containing UNIX timestamps.

    Returns
    -------
    pd.Series
        Series containing datetime values corresponding to the UNIX timestamps.
    """
    return series.astype(int).astype("datetime64[s]")


def round_date(dates: pd.Series) -> pd.Series:
    """
    Round datetimes to the nearest day.

    Parameters
    ----------
    dates : pd.Series
        Datetime series.

    Returns
    -------
    pd.Series
        Series rounded to the nearest day.
    """
    is_datetime_series(dates, raise_err=True)

    return dates.dt.round('1d')


def has_time(
    dates: pd.Series,
    raise_err_on_time: bool = False,
) -> pd.Series:
    """
    Checks whether any datetimes have a time component.

    Parameters
    ----------
    dates : pd.Series
        Datetime series.
    raise_err : bool, default False
        If True, raise an error if any date has a time component.

    Raises
    ------
    ValueError
        If any date has a time component and `raise_err` is True.

    Returns
    -------
    bool
        Whether any dates have a time component.
    """
    # Round datetime values
    rounded = round_date(dates)

    # If the same when rounded, then no time, if different, then has time
    # Since NaN isn't equal to NaN, specifically check to make sure not null
    has_time = (dates != rounded) & ~dates.isna()

    # Check if any dates have times and raise_err is True
    if raise_err_on_time and has_time.any():
        raise ValueError("Dates cannot have a time component.")

    return has_time


# DEPRECIATED IN CONTRAST TO `analyze_dates`???
def invalid_date(dates: pd.Series, **to_datetime_kwargs) -> pd.Series:
    """
    Given a Series of dates, return a boolean Series of whether the dates are invalid.

    Parameters
    ----------
    dates : pandas.Series
        A string series containing (possibly invalid) dates.
    **to_datetime_kwargs
        Additional arguments for pandas.to_datetime.

    Returns
    -------
    pandas.Series
        Series with boolean values indicating whether each date is invalid.

    Raises
    ------
    ValueError
        When "errors" is specified in `to_datetime_kwargs`
    """
    is_str_series(dates, raise_err=True)

    if "errors" in to_datetime_kwargs:
        raise ValueError("Cannot specify 'errors' in to_datetime_kwargs.")

    return pd.isna(pd.to_datetime(dates, errors='coerce', **to_datetime_kwargs))


def filter_date_deltas(
    dates: pd.DataFrame,
    delta_cutoff: Union[str, timedelta] = None,
    left_delta_cutoff: Union[str, timedelta] = None,
    right_delta_cutoff: Union[str, timedelta] = None,
) -> pd.DataFrame:
    """
    Filter DataFrame based on date delta conditions.

    Parameters
    ----------
    dates : pandas.DataFrame
        DataFrame containing 'delta' column.
    delta_cutoff : timedelta, optional
        Maximum delta value allowed.
    left_delta_cutoff : timedelta, optional
        Minimum delta value allowed.
    right_delta_cutoff : timedelta, optional
        Maximum delta value allowed.

    Returns
    -------
    pandas.DataFrame
        Filtered DataFrame based on delta conditions.

    Raises
    ------
    ValueError
        When delta_cutoff specified along with left_delta_cutoff or right_delta_cutoff.
    """
    if delta_cutoff is not None:
        if left_delta_cutoff is not None or right_delta_cutoff is not None:
            raise ValueError(
                "Cannot specify left_delta_cutoff or right_delta_cutoff when "
                "delta_cutoff is specified."
            )

        return dates[abs(dates['delta']) <= pd.to_timedelta(delta_cutoff)]

    if left_delta_cutoff is not None:
        dates = dates[dates['delta'] >= pd.to_timedelta(left_delta_cutoff)]

    if right_delta_cutoff is not None:
        dates = dates[dates['delta'] <= pd.to_timedelta(right_delta_cutoff)]

    return dates


class DatePairHandler:
    """
    Handler to create and manipulate pairs based on dates and IDs.

    Attributes
    ----------
    data_x : pandas.DataFrame
        DataFrame containing data x. Should have the index `id` and a `date` column.
    data_y : pandas.DataFrame
        DataFrame containing data y. Should have the index `id` and a `date` column.
    date_pairs : pandas.DataFrame
        DataFrame containing date pair results.
    _paired_data : pandas.DataFrame, optional
        The paired data coming from the data_x and data_y columns. Computed and stored
        based on `date_pairs` when the `paired_data` method is first called.
    """
    def __init__(
        self,
        data_x: pd.DataFrame,
        data_y: pd.DataFrame,
        delta_cutoff: Union[str, timedelta] = None,
        left_delta_cutoff: Union[str, timedelta] = None,
        right_delta_cutoff: Union[str, timedelta] = None,
        keep_closest_to: Optional[str] = None,
    ):
        assert data_x.index.name == "id"
        assert data_y.index.name == "id"
        assert "idx_x" not in data_x.columns
        assert "idx_y" not in data_y.columns
        assert "date" in data_x.columns
        assert "date" in data_y.columns

        data_x["idx_x"] = np.arange(len(data_x))
        data_y["idx_y"] = np.arange(len(data_y))

        date_pairs = data_x[["date", "idx_x"]].merge(data_y[["date", "idx_y"]], on='id', how='inner')

        if keep_closest_to is not None:
            assert keep_closest_to in ["date_x", "date_y"]

        date_pairs["delta"] = date_pairs["date_x"] - date_pairs["date_y"]
        date_pairs["abs_delta"] = abs(date_pairs["delta"])

        date_pairs = filter_date_deltas(
            date_pairs,
            delta_cutoff=delta_cutoff,
            left_delta_cutoff=left_delta_cutoff,
            right_delta_cutoff=right_delta_cutoff,
        )

        if keep_closest_to is not None:
            date_pairs = date_pairs.reset_index()
            min_deltas = date_pairs.groupby(["id", keep_closest_to]).agg({
                "abs_delta": "min",
            }).reset_index()
            date_pairs = date_pairs.merge(
                min_deltas,
                on=["id", keep_closest_to, "abs_delta"],
                how='inner',
            )

        self.data_x = data_x
        self.data_y = data_y
        self.date_pairs = date_pairs
        self._paired_data = None

    @property
    def paired_data(self) -> pd.DataFrame:
        """
        Get paired data based on the date pairs.

        Returns
        -------
        pandas.DataFrame
            Paired data based on the date pairs.
        """
        if self._paired_data is None:
            self._paired_data = pd.concat([
                self.data_x.set_index("idx_x").loc[self.date_pairs["idx_x"]].reset_index(),
                self.data_y.set_index("idx_y").loc[self.date_pairs["idx_y"]].reset_index(),
            ], axis=1)

        return self._paired_data


def du_parse_date(date: str, **parse_kwargs) -> Union[datetime.datetime, float]:
    """
    Parse a date string using dateutil's parser.

    Parameters
    ----------
    date : str
        Date string to be parsed.
    **parse_kwargs
        Keyword arguments to pass to the parser.

    Returns
    -------
    datetime.datetime or float
        Parsed datetime object or np.nan on failure.
    """
    try:
        return du_parser.parse(date, **parse_kwargs)

    # ParserError = failed to parse
    # TypeError = wrong type, e.g., nan or int
    except (ParserError, TypeError):
        return np.nan


def extract_du_components(
    du_series: pd.Series,
    components: Optional[List[str]] = None,
) -> pd.DataFrame:
    """
    Extract datetime components from dates parsed from dateutil (du).

    Useful for Series full of datetimes that cannot be converted using 
    `pandas.to_datetime` without possibly losing dates to errors like 
    `OutOfBoundsDatetime`.

    Parameters
    ----------
    du_series : pd.Series
        Series of datetimes parsed using dateutil.
    components : list of str, optional
        Components to extract from the datetime. If None, uses `DT_COMPONENTS`.

    Returns
    -------
    pd.DataFrame
        DataFrame containing the extracted datetime components.
    """
    def extract_components(datetime, components):
        if pd.isna(datetime):
            return np.full(len(components), np.nan)
        return np.array([getattr(datetime, comp) for comp in components])

    components = components or DT_COMPONENTS
    component_data = pd.DataFrame(
        np.stack(du_series.apply(extract_components, args=(components,)).values),
        columns=components,
        index=du_series.index,
    )
    return component_data.astype("Int64")


def datetime_components(
    texts: pd.Series,
    components: Optional[List[str]] = None,
) -> pd.DataFrame:
    """
    Extract separate datetime components (NaN when missing) using dateutil.

    Useful because functionalities like `pandas.to_datetime` will return
    NaT if a full date is not present (e.g., missing a year).

    Parameters
    ----------
    texts : pd.Series
        Series of datetime strings.
    components : list of str, optional
        Components to extract from the datetime. If None, uses `DT_COMPONENTS`.

    Returns
    -------
    pd.DataFrame
        DataFrame containing the extracted datetime components and the parsed date.
    """
    # Extract dates with different values across all components
    du = texts.apply(du_parse_date)
    du.rename(DU_DT, inplace=True)

    du2 = texts.apply(du_parse_date, default=datetime.datetime(1, 2, 2, 2, 2, 2, 2))
    du2.rename("du2", inplace=True)

    # Where they are equal is not default, where they aren't is default (i.e., missing)
    components = components or DT_COMPONENTS
    equal = pd.concat([
        extract_du_components(du, components=components),
        extract_du_components(du2, components=components).add_suffix('_2'),
    ], axis=1)

    for i, comp in enumerate(components):
        # If a value is missing (different for different default components),
        # then replace it with NaN
        equal[comp][equal[comp] != equal[f'{comp}_2']] = np.nan

    return pd.concat([du, equal[components]], axis=1)


def analyzed_dates_differ(
    analyzed: pd.DataFrame,
    warn: bool = False,
    raise_err: bool = False,
) -> pd.Series:
    """
    Check where the analyzed `dateutil` and `pd.to_datetime` dates differ.

    Parameters
    ----------
    analyzed : pd.DataFrame
        A result of `analyze_dates`.
    warn : bool, default False
        Whether to warn the user when the dates differ.
    raise_err : bool, default False
        Whether to raise an error when the dates differ.

    Returns
    -------
    pd.Series
        Boolean series indicating where the dates from `pd.to_datetime` and 
        `dateutil` do not match.

    Raises
    ------
    ValueError
        Raised if `raise_err` is True and there are non-matching dates between
        `pd.to_datetime` and `dateutil`.
    """
    check_cols(analyzed, [PD_DT, DU_DT], raise_err_on_missing=True)

    # If the dates parsed from pd and du aren't the same date (and didn't
    # both fail to parse), then flag that something funky might be going on
    matching = (analyzed[PD_DT] == analyzed[DU_DT]) | \
        (analyzed[[PD_DT, DU_DT]].isna().sum(axis=1) == 2)

    if not matching.all():
        msg = (
            "`pd.to_datetime` and `dateutil` produced different results. "
            "Consider manual inspection."
        )

        if raise_err:
            raise ValueError(msg)

        if warn:
            warnings.warn(msg)

    return ~matching


def analyzed_dates_failed_to_convert(
    analyzed: pd.DataFrame,
    warn: bool = False,
    raise_err: bool = False,
) -> pd.Series:
    """
    Check whether any `dateutil` dates which failed to convert using `pd.to_datetime`.

    One common failure is due to a `pandas.errors.OutOfBoundsDatetime`.

    Parameters
    ----------
    analyzed : pd.DataFrame
        A result of `analyze_dates`.
    warn : bool, default False
        Whether to warn the user if there are failures.
    raise_err : bool, default False
        Whether to raise an error if there are failures.

    Returns
    -------
    pd.Series
        Boolean series indicating where the `dateutil` dates failed to convert.

    Raises
    ------
    ValueError
        Raised if `raise_err` is True and there are `dateutil` dates failed to convert.
    """
    check_cols(analyzed, [DU_DT, DU_TO_PD_DT], raise_err_on_missing=True)

    # If du date is not null but the converted date is, then it failed to convert
    failed = analyzed[DU_DT].notnull() & analyzed[DU_TO_PD_DT].isna()

    if failed.any():
        msg = (
            "Failed to convert `dateutil` dates using `pd.to_datetime`. "
            "Consider manual inspection."
        )

        if raise_err:
            raise ValueError(msg)

        if warn:
            warnings.warn(msg)

    return failed


def analyze_dates(
    texts: pd.Series,
    components: Optional[List[str]] = None,
    warn: bool = True,
) -> pd.DataFrame:
    """
    Analyze a series of dates and extract datetime components.

    Parameters
    ----------
    texts : pd.Series
        Series of datetime strings to be analyzed.
    components : list of str, optional
        Components to extract from the datetime. If None, uses `DT_COMPONENTS`.
    warn : bool, default True
        Whether to analyze the dates and warn the user about various anomalies.

    Returns
    -------
    pd.DataFrame
        DataFrame containing the analyzed dates and extracted components.
    """
    is_str_series(texts, raise_err=True)

    texts.rename("text", inplace=True)
    dates = texts.to_frame()

    dates[PD_DT] = pd.to_datetime(dates["text"], infer_datetime_format=True, errors="coerce")

    components = components or DT_COMPONENTS
    dates = pd.concat([
        dates,
        datetime_components(dates["text"], components=components),
    ], axis=1)

    # Drop a component column if the whole column is NaN - it is likely never specified
    dates.drop(
        [comp for comp in components if dates[comp].isna().all()],
        axis=1,
        inplace=True,
    )

    dates[DU_TO_PD_DT] = pd.to_datetime(
        dates[DU_DT],
        infer_datetime_format=True,
        errors="coerce",
    )

    if warn:
        analyzed_dates_differ(dates, warn=True)
        analyzed_dates_failed_to_convert(dates, warn=True)

    return dates


def components_to_datetime(
    comps: pd.DataFrame,
    default_time: Optional[datetime.time] = None,
) -> pd.Series:
    """
    Converts a DataFrame of datetime components into a datetime series.

    Useful for combining separate date and time texts.

    Parameters
    ----------
    comps: pandas.DataFrame
        DataFrame of component columns. Must have `DATE_COMPONENTS` columns and may
        have any in `DT_COMPONENTS`.
    default_time : datetime.time, optional
        Default time for filling null time components. Defaults to midnight (all 0).

    Returns
    -------
    pd.Series
        A datetime series. Null time components will be filled with the components in
        `default_time`. Null date components will result in a null result.

    Notes
    -----
    Consider using `default_time=datetime.time(12)` (noon) to approximate the datetime
    with the least error. If nothing is specified, it defaults to midnight, which is
    a bad default for many events, e.g., few medical procedures take place at night.

    Examples
    --------
    >>> # Convert components to datetime, using noon as the default time
    >>> dts = components_to_datetime(comps, default_time=datetime.time(12))
    """
    # Check component columns
    check_cols(comps, DATE_COMPONENTS, raise_err_on_missing=True)
    check_cols(comps, DT_COMPONENTS, raise_err_on_unexpected=True)
    avail_time_comps = set(comps.columns).intersection(set(TIME_COMPONENTS))

    if not (comps.dtypes.unique().astype(str) == 'Int64').all():
        raise ValueError("Components must have type 'Int64'.")

    # Handle default times
    default_time = default_time or datetime.time(0)
    TIME_COMPONENTS
    for time_comp in TIME_COMPONENTS:
        time_comp_value = getattr(default_time, time_comp)

        # If the column already exists, fill any nulls with the default value
        if time_comp in avail_time_comps:
            comps[time_comp].fillna(time_comp_value, inplace=True)
        # If not, then create the column using the default value
        else:
            comps[time_comp] = time_comp_value
            comps[time_comp] = comps[time_comp].astype("Int64")

    # Convert the components (now filled with time defaults) into datetimes
    cmp = comps.copy()
    index = cmp.index
    cmp.reset_index(drop=True, inplace=True)

    # Convert only the datetimes which are not missing date components,
    # the rest will be filled with NaN during reindexing
    res = pd.to_datetime(cmp[~cmp.isna().any(axis=1)].astype(int)).reindex(cmp.index)
    res.index = index

    return res


def combine_date_and_time_components(
    date_comps: pd.DataFrame,
    time_comps: pd.DataFrame,
) -> pd.DataFrame:
    """
    Combine date components from one DataFrame and time components from another.

    Parameters
    ----------
    date_comps : pandas.DataFrame
        DataFrame containing relevant date components. Non-relevant columns dropped.
    time_comps : pandas.DataFrame
        DataFrame containing relevant time components. Non-relevant columns dropped.

    Returns
    -------
    pd.DataFrame
        A DataFrame with the date components from `date_comps` and time components from
        `time_comps`.

    Examples
    --------
    >>> date_comps = analyze_dates(meta["AcquisitionDate"])
    >>> time_comps = analyze_dates(meta["AcquisitionTime"])
    >>> comps = combine_date_and_time_components(
    >>>     date_comps,
    >>>     time_comps,
    >>>     default_time=datetime.time(12),
    >>> )
    >>> dts = components_to_datetime(datetime)
    """
    if not date_comps.index.equals(date_comps.index):
        raise ValueError(
            "Indexes of `date_comps` and `time_comps` must be the same."
        )

    unexpected_cols_date, _, _ = check_cols(date_comps, DATE_COMPONENTS)
    date_comps = date_comps.drop(unexpected_cols_date, axis=1)

    unexpected_cols_time, _, _ = check_cols(time_comps, TIME_COMPONENTS)
    time_comps = time_comps.drop(unexpected_cols_time, axis=1)

    return pd.concat([date_comps, time_comps], axis=1)


#def find_dates(text):
#    matches = datefinder.find_dates(text, source=True, index=True)
