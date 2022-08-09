"""Functionality relating to the Pandas library."""
import warnings

import pandas as pd
from pandas.errors import PerformanceWarning


def add_years_approximate(
    timestamp_series: pd.Series, years_series: pd.Series
) -> pd.Series:
    """Approximately add together a timestamp series with a years series row-by-row.

    Approximates are typically exact or incorrect by one day, e.g., on leap days.

    Parameters
    ----------
    timestamp_series: pandas.Series
        The series of timestamps to which to add.
    years_series: panadas.Series
        The series of years to add.

    Returns
    -------
    pandas.Series
        The timestamp series with the approximately added years.

    """
    # Add to the years column
    year = timestamp_series.dt.year + years_series

    # Handle the other columns
    month = timestamp_series.dt.month
    day = timestamp_series.dt.day
    hour = timestamp_series.dt.hour
    minute = timestamp_series.dt.minute

    # Create new timestamp column
    data = pd.DataFrame(
        {"year": year, "month": month, "day": day, "hour": hour, "minute": minute}
    )

    # Subtract 1 from potentially invalid leap days to avoid issues
    leap_days = (month == 2) & (day == 29)
    data["day"][leap_days] -= 1

    return pd.to_datetime(data)


def add_years_exact(timestamp_series: pd.Series, years_series: pd.Series) -> pd.Series:
    """Add together a timestamp series with a years series row-by-row.

    Warning: Very slow. It is worth using the add_years_approximate function even
    moderately large data.

    Parameters
    ----------
    timestamp_series: pandas.Series
        The series of timestamps to which to add.
    years_series: panadas.Series
        The series of years to add.

    Returns
    -------
    pandas.Series
        The timestamp series with the approximately added years.

    """
    warnings.warn(
        (
            "Computing the exact addition cannot be vectorized and is very slow. "
            "Consider using the quick, approximate calculation."
        ),
        PerformanceWarning,
    )
    return timestamp_series + years_series.apply(lambda x: pd.DateOffset(years=x))
