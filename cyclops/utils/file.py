"""Utility functions for saving/loading files."""

import logging
import os
import pickle
from typing import Any, Generator, List, Optional, Union

import dask.dataframe as dd
import numpy as np
import pandas as pd

from cyclops.utils.log import setup_logging

# Logging.
LOGGER = logging.getLogger(__name__)
setup_logging(print_level="INFO", logger=LOGGER)


def join(*paths: str) -> str:
    """Robustly join paths.

    os.path.join only may cause problems with some filepaths (especially on Windows).

    Parameters
    ----------
    paths: str
        file paths

    Returns
    -------
    str
        The joined path of all input paths.

    """
    return os.path.join(*paths).replace("\\", "/")


def exchange_extension(file_path: str, new_ext: str) -> str:
    """Exchange one file extension for another.

    Parameters
    ----------
    file_path: str
        File path in which to exchange the extension.
    new_ext: str
        New extension to replace the existing extension.

    Returns
    -------
    str
        File path with the new extension.

    """
    # Remove a leading dot
    new_ext = new_ext.strip(".")
    _, old_ext = os.path.splitext(file_path)
    return file_path[: -len(old_ext)] + "." + new_ext


def process_file_save_path(
    save_path: str, file_format: str, create_dir: bool = True
) -> str:
    """Process file save path, perform checks, and possibly create a parent directory.

    Parameters
    ----------
    save_path: str
        Path where the file will be saved.
    file_format: str
        File format of the file to save.
    create_dir: bool
        If True, create the parent directory path if needed.

    Returns
    -------
    str
        The processed save path.

    """
    # Create the directory if it doesn't already exist.
    directory, _ = os.path.split(save_path)

    if create_dir:
        # Ignore checking local paths
        if directory != "":
            os.makedirs(directory, exist_ok=True)

    # Add the .parquet extension if it isn't there.
    _, ext = os.path.splitext(save_path)

    if ext == "":
        save_path = save_path + "." + file_format
    elif ext != "." + file_format:
        raise ValueError(
            f"""The file extension on the save path must be {file_format}.
            Alternatively, sesider changing the file format."""
        )

    return save_path


def process_dir_save_path(save_path: str, create_dir: bool = True) -> str:
    """Process directory save path, perform checks, and possibly create the directory.

    Parameters
    ----------
    save_path: str
        Path where the file will be saved.
    create_dir: bool
        If True, create the directory if needed.

    Returns
    -------
    str
        The processed save path.

    """
    if os.path.exists(save_path):
        if os.path.isdir(save_path):
            return save_path
        raise ValueError("If save path exists, it must be a directory.")

    if create_dir:
        os.makedirs(save_path)
        return save_path

    raise ValueError("Directory does not exist.")


def save_dataframe(
    data: Union[pd.DataFrame, dd.DataFrame],
    save_path: str,
    file_format: str = "parquet",
    log: bool = True,
) -> str:
    """Save a pandas.DataFrame or dask.DataFrame object to file.

    Parameters
    ----------
    data: pandas.DataFrame
        Dataframe to save.
    save_path: str
        Path where the file will be saved.
    file_format: str
        File format of the file to save.
    log: bool
        Whether to log the occurence.

    Returns
    -------
    str
        Processed save path for upstream use.

    """
    if not isinstance(data, (pd.DataFrame, dd.DataFrame)):
        raise ValueError("Input data is not a DataFrame.")
    save_path = process_file_save_path(save_path, file_format)
    if isinstance(data, dd.DataFrame):
        save_path, _ = os.path.splitext(save_path)
    if log:
        LOGGER.info("Saving dataframe to %s", save_path)
    if file_format == "parquet":
        if isinstance(data, pd.DataFrame):
            data.to_parquet(save_path, schema=None)
        if isinstance(data, dd.DataFrame):
            data.to_parquet(
                save_path,
                schema=None,
                name_function=lambda x: f"batch-{str(x).zfill(3)}.parquet",
            )
    elif file_format == "csv":
        data.to_csv(save_path)
    else:
        raise ValueError(
            "Invalid file formated provided. Currently supporting 'parquet' and 'csv'."
        )

    return save_path


def load_dataframe(
    load_path: str,
    file_format: str = "parquet",
    log: bool = True,
) -> Union[pd.DataFrame, dd.DataFrame]:
    """Load file to a pandas.DataFrame or dask.DataFrame object.

    Parameters
    ----------
    load_path: str
        Path where the file to load.
    file_format: str
        File format of the file to load.
    log: bool
        Whether to log the occurence.

    Returns
    -------
    pandas.DataFrame or dask.DataFrame
        Loaded data.

    """
    is_dask = True
    if not os.path.isdir(load_path):
        load_path = process_file_save_path(load_path, file_format)
        is_dask = False
    if log:
        LOGGER.info("Loading DataFrame from %s", load_path)
    if file_format == "parquet":
        data_reader = dd.read_parquet if is_dask else pd.read_parquet
        data = data_reader(load_path)
    elif file_format == "csv":
        data = pd.read_csv(load_path, index_col=[0])
    else:
        raise ValueError(
            "Invalid file formated provided. Currently supporting 'parquet' and 'csv'."
        )

    return data


def save_array(
    data: np.ndarray,
    save_path: str,
    file_format: str = "npy",
    log: bool = True,
) -> str:
    """Save a numpy.ndarray object to file.

    Parameters
    ----------
    data: numpy.ndarray
        Array to save.
    save_path: str
        Path where the file will be saved.
    file_format: str
        File format of the file to save.
    log: bool
        Whether to log the occurence.

    Returns
    -------
    str
        Processed save path for upstream use.

    """
    save_path = process_file_save_path(save_path, file_format)

    if not isinstance(data, np.ndarray):
        raise ValueError("Input data is not an array.")

    if log:
        LOGGER.info("Saving array to %s", save_path)

    if file_format == "npy":
        np.save(save_path, data)
    else:
        raise ValueError("Invalid file formated provided. Currently supporting 'npy'.")

    return save_path


def load_array(
    load_path: str,
    file_format: str = "npy",
    log: bool = True,
) -> np.ndarray:
    """Load file to a numpy.ndarray object.

    Parameters
    ----------
    load_path: str
        Path where the file to load.
    file_format: str
        File format of the file to load.
    log: bool
        Whether to log the occurence.

    Returns
    -------
    numpy.ndarray
        Loaded data.

    """
    load_path = process_file_save_path(load_path, file_format)

    if log:
        LOGGER.info("Loading array from %s", load_path)

    if file_format == "npy":
        data = np.load(load_path)
    else:
        raise ValueError("Invalid file formated provided. Currently supporting 'npy'.")

    return data


def save_pickle(
    data: Any,
    save_path: str,
    log: bool = True,
) -> str:
    """Save a object to pickle file.

    Parameters
    ----------
    data: any
        Data to save.
    save_path: str
        Path where the file will be saved.
    log: bool
        Whether to log the occurence.

    Returns
    -------
    str
        Processed save path for upstream use.

    """
    save_path = process_file_save_path(save_path, "pkl")

    if log:
        LOGGER.info("Pickling data to %s", save_path)

    with open(save_path, "wb") as handle:
        pickle.dump(data, handle)

    return save_path


def load_pickle(
    load_path: str,
    log: bool = True,
) -> Any:
    """Load an object from a pickle file.

    Parameters
    ----------
    load_path: str
        Path where the file to load.
    log: bool
        Whether to log the occurence.

    Returns
    -------
    any
        Loaded data.

    """
    load_path = process_file_save_path(load_path, "pkl")

    if log:
        LOGGER.info("Loading pickled data from %s", load_path)

    with open(load_path, "rb") as handle:
        return pickle.load(handle)


def listdir_nonhidden(path: str) -> List[str]:
    """List the non-hidden files of a directory.

    Parameters
    ----------
    path: str
        Directory path.

    Returns
    -------
    list
        List of non-hidden files.

    """
    return [f for f in os.listdir(path) if not f.startswith(".")]


def yield_dataframes(
    dir_path: str,
    sort: bool = True,
    skip_n: Optional[int] = None,
    log: bool = True,
) -> Generator[pd.DataFrame, None, None]:
    """Yield DataFrames loaded from a directory.

    Any non-hidden files in the directory must be loadable as a DataFrame.

    Parameters
    ----------
    dir_path: str
        Directory path of files.
    sort: bool, default = True
        Whether to sort the files and yield them in an ordered manner.
    skip_n: int, optional
        If specified, skip the first n files when yielding the files.
        This is especially useful in lieu of the execution being interrupted.
    log: bool
        Whether to log the occurence.

    Yields
    ------
    pandas.DataFrame
        A DataFrame.

    """
    files = list(listdir_nonhidden(dir_path))

    if sort:
        files.sort()

    if skip_n:
        files = files[skip_n:]

    for file in files:
        yield load_dataframe(join(dir_path, file), log=log)


def yield_pickled_files(
    dir_path: str,
    sort: bool = True,
    skip_n: Optional[int] = None,
    log: bool = True,
) -> Generator[pd.DataFrame, None, None]:
    """Yield pickled files loaded from a directory.

    Any non-hidden files in the directory must be loadable with pickle.

    Parameters
    ----------
    dir_path: str
        Directory path of files.
    sort: bool, default = True
        Whether to sort the files and yield them in an ordered manner.
    skip_n: int, optional
        If specified, skip the first n files when yielding the files.
        This is especially useful in lieu of the execution being interrupted.
    log: bool
        Whether to log the occurence.

    Yields
    ------
    any
        Previously pickled data.

    """
    files = list(listdir_nonhidden(dir_path))

    if sort:
        files.sort()

    if skip_n:
        files = files[skip_n:]

    for file in files:
        yield load_pickle(join(dir_path, file), log=log)
