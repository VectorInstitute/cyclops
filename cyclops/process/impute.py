"""Imputation functions."""

import logging
from typing import Any, Callable, Dict, Optional, Tuple, Union

import numpy as np
import pandas as pd

from cyclops.process.constants import (
    BFILL,
    DROP,
    EXTRA,
    FFILL,
    FFILL_BFILL,
    IGNORE,
    INTER,
    LINEAR_INTERP,
    MEAN,
    MEDIAN,
    MODE,
)
from cyclops.process.util import has_columns
from cyclops.utils.log import setup_logging
from cyclops.utils.profile import time_function

# Logging.
LOGGER = logging.getLogger(__name__)
setup_logging(print_level="INFO", logger=LOGGER)


def compute_inter_range(null: pd.Series) -> Optional[Tuple[int, int]]:
    """Compute the range of values to be interpolated.

    Computed using the first and last non-null values in the series.

    Parameters
    ----------
    null: pandas.Series
        A boolean mask array indicated the positions of nulls. True is a null.

    Returns
    -------
    tuple
        A tuple of (int, int) with (inter start, inter stop).

    """
    inds = np.argwhere(~null.values)[:, 0]
    if len(inds) == 0:
        return None

    return inds[0], inds[-1] + 1


def np_ffill(arr: np.ndarray) -> np.ndarray:
    """Forward fill a 1D array.

    Parameters
    ----------
    arr: numpy.ndarray
        Array to impute.

    """
    mask = np.isnan(arr)
    idx = np.where(~mask, np.arange(mask.shape[0]), 0)
    idx = np.maximum.accumulate(idx, axis=0, out=idx)
    return arr[idx]


def np_bfill(arr: np.ndarray) -> np.ndarray:
    """Backward fill a 1D array.

    Parameters
    ----------
    arr: numpy.ndarray
        Array to impute.

    Returns
    -------
    numpy.ndarray
        Imputed array.

    """
    mask = np.isnan(arr)
    idx = np.where(~mask, np.arange(mask.shape[0]), mask.shape[0] - 1)
    idx = np.minimum.accumulate(idx[::-1], axis=0)[::-1]
    return arr[idx]


def np_ffill_bfill(arr: np.ndarray) -> np.ndarray:
    """Equivalent to forward filling and then backward filling a 1D array.

    Parameters
    ----------
    arr: numpy.ndarray
        Array to impute.

    Returns
    -------
    numpy.ndarray
        Imputed array.

    """
    arr = np_ffill(arr)
    mask = np.isnan(arr)
    first_non_null_idx = mask.argmin()
    first_non_null = arr[first_non_null_idx]
    arr[:first_non_null_idx] = first_non_null
    return arr


def np_fill_null_num(arr: np.ndarray, num: float) -> np.ndarray:
    """Fill null values with a number.

    Parameters
    ----------
    arr: numpy.ndarray
        Array to impute.

    Returns
    -------
    numpy.ndarray
        Imputed array.

    """
    return np.nan_to_num(arr, nan=num)


def np_fill_null_zero(arr: np.ndarray) -> np.ndarray:
    """Fill null values with zero.

    Parameters
    ----------
    arr: numpy.ndarray
        Array to impute.

    Returns
    -------
    numpy.ndarray
        Imputed array.

    """
    return np_fill_null_num(arr, 0)


def np_fill_null_mean(arr: np.ndarray) -> np.ndarray:
    """Fill null values with the array mean.

    Parameters
    ----------
    arr: numpy.ndarray
        Array to impute.

    Returns
    -------
    numpy.ndarray
        Imputed array.

    """
    return np_fill_null_num(arr, np.nanmean(arr))


def fill_null_with(series: pd.Series, null: pd.Series, value: Any) -> pd.Series:
    """Fill null values with a specified value when the nulls were already located.

    Parameters
    ----------
    series: pandas.Series
        Series for which to fill the nulls.
    null: pandas.Series
        A boolean mask array indicated the positions of nulls. True is a null.
    value
        The value to replace the nulls.

    Returns
    -------
    pandas.Series
        Imputed series.

    """
    series[null] = value
    return series


def efficient_ffill_bfill(series: pd.Series, null: pd.Series) -> pd.Series:
    """Forward and backward fill nulls in an efficient manner.

    An efficient implementation equivalent to forward filling a series and subsequently
    backwards filling to remove any nulls at the very beginning of the series. This
    ensures no nulls are returned.

    Parameters
    ----------
    series: pandas.Series
        Series for which to fill the nulls.
    null: pandas.Series
        A boolean mask array indicated the positions of nulls. True is a null.

    Returns
    -------
    pandas.Series
        Imputed series.

    """
    series = series.ffill()
    first_non_null = null.idxmin()
    series.iloc[:first_non_null] = series.iloc[first_non_null]
    return series


IMPUTEFUNCS = {
    IGNORE: None,
    DROP: lambda series, null: series[~null],
    MEAN: lambda series, null: fill_null_with(series, null, series.mean(skipna=True)),
    MEDIAN: lambda series, null: fill_null_with(
        series, null, series.median(skipna=True)
    ),
    MODE: lambda series, null: fill_null_with(series, null, series.mode().iloc[0]),
    FFILL: lambda series, _: series.ffill(),
    BFILL: lambda series, _: series.bfill(),
    FFILL_BFILL: efficient_ffill_bfill,
    LINEAR_INTERP: lambda series, _: series.interpolate(method="linear"),
}


class SeriesImputer:
    """Imputation of a Pandas Series.

    It may seem counter-intuitive to allow the return of nulls, but many imputation
    functions such as filling with mean/median will return nulls if the series is all
    null, i.e., where no fill value can be computed.

    Attributes
    ----------
    imputefunc: str or callable
        Imputation function. Either function or string, e.g., MEAN.
        If a function, it should accept a series and return a series of same length
        unless the DROP function is used.
    using_drop: bool
        Whether the DROP imputation was used. This is necessary to track to ensure
        no values are otherwise dropped.
    allow_nulls_returned: bool, default = True
        Whether to allow the returning of nulls from an imputation function.
    limit_area: str, optional
        Can specify 'inter' or 'extra'.
        'inter': Only fill NaNs surrounded by valid values (interpolate).
        'extra': Only fill NaNs outside valid values (extrapolate).

    """

    def __init__(
        self,
        imputefunc: Union[str, Callable] = MEAN,
        allow_nulls_returned=True,
        limit_area=None,
    ):
        """Init."""
        self.using_drop = imputefunc == DROP

        if self.using_drop:
            LOGGER.warning(  # pylint: disable=logging-not-lazy
                "The imputer DROP strategy is rarely used. "
                + "The IGNORE strategy may be more fitting, more robust, "
                + "and less expensive."
            )

        self.imputefunc = self._process_imputefunc(imputefunc)
        self.allow_nulls_returned = allow_nulls_returned

        if limit_area is not None:
            if limit_area not in [INTER, EXTRA]:
                raise ValueError(
                    f"If specified, limit_area must be in: {', '.join([INTER, EXTRA])}."
                )

        self.limit_area = limit_area

    def _process_imputefunc(self, imputefunc: Union[str, Callable]) -> Callable:
        """Process imputation function.

        Convert a imputefunc string to an imputefunc if recognized.
        Otherwise, simply return the function.

        Returns
        -------
        callable
            The imputation function.

        """
        if isinstance(imputefunc, str):
            if imputefunc not in IMPUTEFUNCS:
                raise ValueError(
                    f"""Imputefunc string {imputefunc} not supported.
                    Supporting: {','.join(IMPUTEFUNCS)}"""
                )
            func = IMPUTEFUNCS[imputefunc]
        elif callable(imputefunc):
            func = imputefunc
        else:
            raise ValueError("Imputefunc must be a string or callable.")

        return func  # type: ignore

    def is_using_drop(self) -> bool:
        """Return whether the imputer is using the DROP strategy.

        Returns
        -------
        bool
            Whether the imputer is using the DROP strategy.

        """
        return self.using_drop

    def _interpolate(self, series: pd.Series, null: pd.Series) -> pd.Series:
        inter_range = compute_inter_range(null)
        if inter_range is None:
            # If all null, there is nothing to do for inter imputation.
            return series

        inter_start, inter_stop = inter_range
        series[inter_start:inter_stop] = self.imputefunc(
            series[inter_start:inter_stop],
            null[inter_start:inter_stop],
        )
        return series

    def _extrapolate(self, series: pd.Series, null: pd.Series) -> pd.Series:
        inter_range = compute_inter_range(null)
        if inter_range is None:
            # If all null, then it is all considered extra.
            return self.imputefunc(series)

        inter_start, inter_stop = inter_range

        # Otherwise, compute the extra before and after the inter start/stop
        if inter_start != 0:
            series[:inter_start] = self.imputefunc(
                series[:inter_start],
                null[:inter_start],
            )

        if inter_stop != len(series) - 1:
            series[inter_stop:] = self.imputefunc(
                series[inter_stop:],
                null[inter_stop:],
            )

        return series

    def _impute(self, series: pd.Series, null: pd.Series) -> pd.Series:
        if self.limit_area is None:
            return self.imputefunc(series, null)

        # Interpolate
        if self.limit_area == INTER:
            return self._interpolate(series, null)

        # Extrapolate
        return self._extrapolate(series, null)

    def __call__(self, series: pd.Series) -> Tuple[pd.Series, float]:
        """Impute a series.

        Parameters
        ----------
        series: pandas.Series
            The series to impute.

        Returns
        -------
        tuple
            A tuple containing, respectively, a pandas.Series (the imputed Series) and
            a float (the percentage of nulls in the original series).

        """
        # Compute preliminary information
        original_len = len(series)
        null = pd.isnull(series)
        null_count = null.sum()
        null_percent = null_count / len(series)

        # If no imputation is desired
        if self.imputefunc is None:
            return series, null_percent

        # Impute
        series = self._impute(series, null)

        # Check validity of imputation

        # Check that the length is the same unless the DROP function was used
        if original_len != len(series) and not self.using_drop:
            raise ValueError(
                (
                    "Different length returned from imputation function when"
                    "not using the 'DROP' function."
                )
            )

        # Check if nulls were returned when not allowed
        if not self.allow_nulls_returned:
            if series.isnull().values.any():
                raise ValueError(
                    "Nulls returned from imputation function when not allowed."
                )

        return series, null_percent


class TabularImputer:  # pylint: disable=too-few-public-methods
    """Imputation of tabular data.

    Attributes
    ----------
    imputers: dict
        Aggregation functions mapped from column to imputer.

    """

    def __init__(
        self,
        imputers: Dict[str, SeriesImputer],
    ):
        """Init."""
        self.imputers = imputers

        for imputer in self.imputers.values():
            if imputer.is_using_drop():
                raise ValueError(
                    (
                        "TabularImputer does not accept series imputers using the ",
                        "DROP strategy. Explore other strategy like IGNORE, or ",
                        "consider using no imputation.",
                    )
                )

    @time_function
    def __call__(self, data: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, float]]:
        """Impute.

        Parameters
        ----------
        data: pandas.DataFrame
            Tabular data.

        """
        has_columns(data, list(self.imputers.keys()), raise_error=True)
        null_percents = {}
        for col, imputer in self.imputers.items():
            data[col], null_percents[col] = imputer(data[col])

        return data, null_percents


# class GroupbyImputer:
#     """Imputation over groups.

#     Attributes
#     ----------
#     imputers: dict
#         Aggregation functions mapped from column to imputer.
#     by: str
#     """
#     def __init__(
#         self,
#         imputers: Dict[str, SeriesImputer],
#         by: Union[str, List[str]],  # pylint: disable=invalid-name
#     ):
#         self.imputers = imputers
#         self.by = to_list(self.by)  # pylint: disable=invalid-name

#     def __call__(data: pd.DataFrame) -> pd.DataFrame:


# class TemporalImputer:
#     """Imputation of temporal data.

#     """
#     def __init__(self):
#         pass


class AggregatedImputer:
    """Imputation of data being aggregated.

    Attributes
    ----------
    intra_imputer: cyclops.processors.impute.Imputer, optional
        Intra imputer for imputation within a timestep, or bucket.
    inter_imputer: cyclops.processors.impute.Imputer, optional
        Inter imputer for imputation between timesteps, or buckets.
    extra_imputer: cyclops.processors.impute.Imputer, optional
        Extra imputer for imputation beyond existing timesteps, or buckets.

    """

    def __init__(
        self,
        intra_imputer: Optional[TabularImputer] = None,
        inter_imputer: Optional[TabularImputer] = None,
        extra_imputer: Optional[TabularImputer] = None,
    ):
        """Init."""
        self.intra_imputer = intra_imputer
        self.inter_imputer = inter_imputer
        self.extra_imputer = extra_imputer

        if self.inter_imputer is not None:
            for series_imputer in self.inter_imputer.imputers.values():
                if not series_imputer.limit_area == INTER:
                    raise ValueError(
                        f"inter_imputer SeriesImputer limit_area='{INTER}'."
                    )

        if self.extra_imputer is not None:
            for series_imputer in self.extra_imputer.imputers.values():
                if not series_imputer.limit_area == EXTRA:
                    raise ValueError(
                        f"extra_imputer SeriesImputer limit_area='{EXTRA}'."
                    )

    def intra(self, group: pd.DataFrame) -> pd.DataFrame:
        """Perform intra-imputation.

        Intra imputation describes the imputation occurring within a single timestep,
        or bucket, before the aggregation of values takes place.

        Parameters
        ----------
        group: pandas.Series
            The group of non-aggregated data being imputed, which will subsequently
            be aggregated.

        Returns
        -------
        pandas.Series
            The imputed group.

        """
        if self.intra_imputer is None:
            return group

        has_columns(group, list(self.intra_imputer.imputers.keys()), raise_error=True)
        return self.intra_imputer(group)

    def inter(self, group: pd.DataFrame) -> pd.DataFrame:
        """Perform interpolation imputation.

        Inter imputation describes the imputation occurring between timesteps, or
        buckets, where for "missing" timesteps, it fills in information. A "missing"
        timestep is one with no occurences, or where all the occurences were null.

        Parameters
        ----------
        group: pandas.Series
            The group of aggregated timesteps being imputed which may have missing
            timesteps.

        Returns
        -------
        pandas.Series
            The imputed group.

        """
        if self.inter_imputer is None:
            return group

        has_columns(group, list(self.inter_imputer.imputers.keys()), raise_error=True)
        return self.inter_imputer(group)

    def extra(self, group: pd.DataFrame) -> pd.DataFrame:
        """Perform extrapolation imputation.

        Extra imputation describes the imputation occurring after the last observed
        timestep. This is typically desired after padding to fill the resultant missing
        values.

        Parameters
        ----------
        group: pandas.Series
            The group of aggregated timesteps being imputed which may have missing
            timesteps extending beyond the last observation.

        Returns
        -------
        pandas.Series
            The imputed group.

        """
        if self.extra_imputer is None:
            return group

        has_columns(group, list(self.extra_imputer.imputers.keys()), raise_error=True)
        return self.extra_imputer(group)


def numpy_2d_ffill(arr: np.ndarray) -> np.ndarray:
    """Foward fill a 2D array in a row-wise fashion, i.e., filling each row separately.

    Parameters
    ----------
    arr: numpy.ndarray
        A 2-dimensional array.

    Returns
    -------
    numpy.ndarray
        The row-wise forward filled array.

    """
    if arr.ndim != 2:
        raise ValueError("The array must be 2-dimensional.")

    mask = np.isnan(arr)
    idx = np.where(~mask, np.arange(mask.shape[1]), 0)
    np.maximum.accumulate(idx, axis=1, out=idx)
    out = arr[np.arange(idx.shape[0])[:, None], idx]
    return out
