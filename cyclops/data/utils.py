"""Utilities for datasets."""

import logging
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Dict,
    List,
    Optional,
    Union,
    get_args,
)

import pandas as pd
import PIL
import psutil
from datasets import Dataset, DatasetDict
from datasets.features import (
    Array2D,
    Array3D,
    Array4D,
    Array5D,
    ClassLabel,
    Sequence,
    Value,
)

from cyclops.utils.common import to_list
from cyclops.utils.log import setup_logging
from cyclops.utils.optional import import_optional_module


# Logging.
LOGGER = logging.getLogger(__name__)
setup_logging(print_level="INFO", logger=LOGGER)


if TYPE_CHECKING:
    from torchvision.transforms import PILToTensor
else:
    PILToTensor = import_optional_module(
        "torchvision.transforms",
        attribute="PILToTensor",
        error="warn",
    )


NUMERIC_FEATURE_TYPES = (
    "bool",
    "int",
    "uint",
    "float",
    "decimal",
)
DATETIME_FEATURE_TYPES = (
    "date",
    "duration",
    "time",
    "timestamp",
)
FEATURE_TYPES = Union[
    ClassLabel,
    Sequence,
    Value,
    Array2D,
    Array3D,
    Array4D,
    Array5D,
]


def set_decode(
    dataset: Dataset,
    decode: bool = True,
    exclude: Optional[List[str]] = None,
) -> None:
    """Set decode attribute of dataset features that have it.

    Parameters
    ----------
    dataset : Dataset
       A Hugging Face dataset object.
    decode : bool, optional, default=True
        Whether to set decode attribute to True or False for features that have
        it.
    exclude : List[str], optional, default=None
        List of feature names to exclude. If None, no features are excluded.
        An example of when this might be useful is when dealing with image
        segmentation tasks where the target and prediction are images that need
        to be decoded, whereas the original image may not need to be decoded.

    """
    assert isinstance(
        dataset,
        (Dataset, DatasetDict),
    ), "dataset must be a Hugging Face dataset"
    if exclude is not None and (
        not isinstance(exclude, list)
        or not all(feature in dataset.column_names for feature in exclude)
    ):
        raise ValueError(
            "`exclude` must be a list of feature names that are present in "
            f"dataset. Got {exclude} of type `{type(exclude)}` and dataset "
            f"with columns {dataset.column_names}.",
        )
    if isinstance(dataset, DatasetDict):
        # set_decode for all keys in dataset
        for key in dataset:
            for feature_name, feature in dataset[key].features.items():
                if feature_name not in (exclude or []) and hasattr(feature, "decode"):
                    dataset[key].features[feature_name].decode = decode
    else:
        for feature_name, feature in dataset.features.items():
            if feature_name not in (exclude or []) and hasattr(feature, "decode"):
                dataset.features[feature_name].decode = decode


def feature_is_numeric(feature: FEATURE_TYPES) -> bool:
    """Check if Hugging Face dataset feature is numeric.

    Parameters
    ----------
    feature : Union[ClassLabel, Sequence, Value, Array2D, Array3D, Array4D, Array5D]
        Hugging Face dataset feature.

    Returns
    -------
    bool
        True if feature is numeric, False otherwise.

    Raises
    ------
    TypeError
        If `feature` is not a valid Hugging Face dataset feature.

    """
    dtype = feature.dtype
    if dtype == "list":  # Sequence
        return all(feature_is_numeric(subfeature) for subfeature in feature.feature)
    if not isinstance(feature, get_args(FEATURE_TYPES)):
        raise TypeError(f"Invalid type for `feature`: {type(feature)}.")

    return any(dtype.startswith(t) for t in NUMERIC_FEATURE_TYPES)


def feature_is_datetime(feature: FEATURE_TYPES) -> bool:
    """Check if Hugging Face dataset feature is datetime.

    Parameters
    ----------
    feature : Union[ClassLabel, Sequence, Value, Array2D, Array3D, Array4D, Array5D]
        Hugging Face dataset feature.

    Returns
    -------
    bool
        True if feature is datetime, False otherwise.

    Raises
    ------
    TypeError
        If `feature` is not a valid Hugging Face dataset feature.

    """
    dtype = feature.dtype
    if dtype == "list":  # Sequence
        return all(feature_is_datetime(subfeature) for subfeature in feature.feature)
    if not isinstance(feature, get_args(FEATURE_TYPES)):
        raise TypeError(f"Invalid type for `feature`: {type(feature)}.")

    return any(dtype.startswith(t) for t in DATETIME_FEATURE_TYPES)


def is_out_of_core(dataset_size: int) -> Any:
    """Check if dataset is too large to fit in memory.

    Parameters
    ----------
    dataset_size : int
        Size of dataset expressed in bytes

    Returns
    -------
    Any
        Whether dataset can fit in memory

    """
    return dataset_size > psutil.virtual_memory().available


def apply_transforms(
    examples: Dict[str, Any],
    transforms: Callable[..., Any],
) -> Dict[str, Any]:
    """Apply transforms to examples."""
    # examples is a dict of lists; convert to list of dicts.
    # doing a conversion from PIL to tensor is necessary here when working
    # with the Image feature type.
    value_len = len(list(examples.values())[0])
    examples_list = [
        {
            k: PILToTensor()(v[i]) if isinstance(v[i], PIL.Image.Image) else v[i]
            for k, v in examples.items()
        }
        for i in range(value_len)
    ]

    # apply the transforms to each example
    examples_list = [transforms(example) for example in examples_list]

    # convert back to a dict of lists
    return {k: [d[k] for d in examples_list] for k in examples_list[0]}


def generate_timestamps(
    start_time: str,
    end_time: str,
    periods: int,
) -> pd.Series:
    """Generate timestamps between start_time and end_time.

    Parameters
    ----------
    start_time : str
        Start time in the format "MM/DD/YYYY".
    end_time : str
        End time in the format "MM/DD/YYYY".
    periods : int
        Number of timestamps to generate.

    Returns
    -------
    pd.Series
        Series of timestamps.

    """
    return pd.date_range(
        start=start_time,
        end=end_time,
        periods=periods,
    )


def create_indicator_variables(
    features: pd.DataFrame,
    columns: Optional[List[str]] = None,
) -> pd.DataFrame:
    """Create binary indicator variable for each column (or specified).

    Create new indicator variable columns based on NAs for other feature columns.

    Parameters
    ----------
    features: pandas.DataFrame
        Input features with missing values.
    columns: List[str], optional
        Columns to create variables, all if not specified.

    Returns
    -------
    pandas.DataFrame
        Dataframe with indicator variables as columns.

    """
    indicator_features = features[columns] if columns else features

    return indicator_features.notnull().astype(int).add_suffix("_indicator")


def is_timestamp_series(series: pd.Series, raise_error: bool = False) -> Any:
    """Check whether a series has the Pandas Timestamp datatype.

    Parameters
    ----------
    series: pandas.Series
        A series.

    Returns
    -------
    bool
        Whether the series has the Pandas Timestamp datatype.

    """
    is_timestamp = series.dtype == pd.to_datetime(["2069-03-29 02:30:00"]).dtype
    if not is_timestamp and raise_error:
        raise ValueError(f"{series.name} must be a timestamp Series.")

    return is_timestamp


def has_columns(
    data: pd.DataFrame,
    cols: Union[str, List[str]],
    exactly: bool = False,
    raise_error: bool = False,
) -> bool:
    """Check if data has required columns for processing.

    Parameters
    ----------
    data: pandas.DataFrame
        DataFrame to check.
    cols: str or list or str
        List of column names that must be present in data.
    raise_error: bool
        Whether to raise a ValueError if there are missing columns.

    Returns
    -------
    bool
        True if all required columns are present, otherwise False.

    """
    cols = to_list(cols)
    required_set = set(cols)
    columns = set(data.columns)
    present = required_set.issubset(columns)

    if not present and raise_error:
        missing = required_set - columns
        raise ValueError(f"Missing required columns: {', '.join(missing)}.")

    if exactly:
        exact = present and len(data.columns) == len(cols)
        if not exact and raise_error:
            raise ValueError(f"Must have exactly the columns: {', '.join(cols)}.")

    return present


def has_range_index(data: pd.DataFrame) -> Union[bool, pd.Series, pd.DataFrame]:
    """Check whether a DataFrame has a range index.

    Parameters
    ----------
    data: pandas.DataFrame
        Data.

    Returns
    -------
    bool or pandas.Series or pandas.DataFrame
        Whether the data has a range index.

    """
    return (data.index == pd.RangeIndex(stop=len(data))).all()


def to_range_index(data: pd.DataFrame) -> pd.DataFrame:
    """Force a DataFrame to have a range index.

    Parameters
    ----------
    data: pandas.DataFrame
        Data.

    Returns
    -------
    pandas.DataFrame
        Data with a range index.

    """
    if has_range_index(data):
        return data

    name = data.index.name
    data = data.reset_index()
    if name == "index":
        data = data.drop("index", axis=1)

    return data


def gather_columns(data: pd.DataFrame, columns: Union[List[str], str]) -> pd.DataFrame:
    """Gather specified columns, discarding rest and return copy of columns.

    Parameters
    ----------
    data: pandas.DataFrame
        DataFrame to check.
    columns: list of str or str
        Column names to gather from dataframe.

    Returns
    -------
    pandas.DataFrame
        DataFrame with required columns, other columns discarded.

    """
    return data[to_list(columns)].copy()


def log_df_counts(
    data: pd.DataFrame,
    col: str,
    step_description: str,
    rows: bool = True,
    columns: bool = False,
) -> None:
    """Log num. of encounters and num. of samples (rows).

    Parameters
    ----------
    data: pandas.DataFrame
        Encounter specific input data.
    col: str
        Column name to count.
    step_description: str
        Description of intermediate processing step.
    rows: bool
        Log the number of samples, or rows of data.
    columns: bool
        Log the number of data columns.

    """
    LOGGER.info(step_description)
    num_encounters = data[col].nunique()
    if rows:
        num_samples = len(data)
        LOGGER.info("# samples: %d, # encounters: %d", num_samples, num_encounters)
    if columns:
        num_columns = len(data.columns)
        LOGGER.info("# columns: %d, # encounters: %d", num_columns, num_encounters)
