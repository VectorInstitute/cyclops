"""Utility functions for saving/loading files."""

import logging
import os
from typing import Generator, List

import numpy as np
import pandas as pd

from codebase_ops import get_log_file_path
from cyclops.utils.log import setup_logging

# Logging.
LOGGER = logging.getLogger(__name__)
setup_logging(log_path=get_log_file_path(), print_level="INFO", logger=LOGGER)


def join(path1: str, path2: str) -> str:
    """Robustly join two paths.

    os.path.join only may cause problems with some filepaths (especially on Windows).

    Parameters
    ----------
    path1: str
        Start of file path.
    path2: str
        End of file path.

    Returns
    -------
    str
        The joined path of path1 and path2.

    """
    return os.path.join(path1, path2).replace("\\", "/")


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
    data: pd.DataFrame,
    save_path: str,
    file_format: str = "parquet",
) -> str:
    """Save a pandas.DataFrame object to file.

    Parameters
    ----------
    data: pandas.DataFrame
        Dataframe to save.
    save_path: str
        Path where the file will be saved.
    file_format: str
        File format of the file to save.

    Returns
    -------
    str
        Processed save path for upstream use.

    """
    save_path = process_file_save_path(save_path, file_format)

    if not isinstance(data, pd.DataFrame):
        raise ValueError("Input data is not a DataFrame.")

    LOGGER.info("Saving dataframe to %s", save_path)

    if file_format == "parquet":
        data.to_parquet(save_path)
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
) -> pd.DataFrame:
    """Load file to a pandas.DataFrame object.

    Parameters
    ----------
    load_path: str
        Path where the file to load.
    file_format: str
        File format of the file to load.

    Returns
    -------
    pandas.DataFrame
        Loaded data.

    """
    load_path = process_file_save_path(load_path, file_format)
    LOGGER.info("Loading DataFrame from %s", load_path)

    if file_format == "parquet":
        data = pd.read_parquet(load_path)
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

    Returns
    -------
    str
        Processed save path for upstream use.

    """
    save_path = process_file_save_path(save_path, file_format)
    print(save_path, data)

    if not isinstance(data, np.ndarray):
        raise ValueError("Input data is not an array.")

    LOGGER.info("Saving array to %s", save_path)

    if file_format == "npy":
        np.save(save_path, data)
    else:
        raise ValueError("Invalid file formated provided. Currently supporting 'npy'.")

    return save_path


def load_array(
    load_path: str,
    file_format: str = "npy",
) -> np.ndarray:
    """Load file to a numpy.ndarray object.

    Parameters
    ----------
    load_path: str
        Path where the file to load.
    file_format: str
        File format of the file to load.

    Returns
    -------
    numpy.ndarray
        Loaded data.

    """
    load_path = process_file_save_path(load_path, file_format)
    LOGGER.info("Loading array from %s", load_path)

    if file_format == "npy":
        data = np.load(load_path)
    else:
        raise ValueError("Invalid file formated provided. Currently supporting 'npy'.")

    return data


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
    dir_path: str, sort: bool = True
) -> Generator[pd.DataFrame, None, None]:
    """Yield DataFrames loaded from a directory.

    Any non-hidden files in the directory must be loadable as a DataFrame.

    Parameters
    ----------
    dir_path: str
        Directory path of files.
    sort: bool, default = True
        Whether to sort the files and yield them in an ordered manner.

    Yields
    ------
    pandas.DataFrame
        A DataFrame.

    """
    files = list(listdir_nonhidden(dir_path))

    if sort:
        files.sort()

    for file in files:
        yield load_dataframe(join(dir_path, file))


def concat_consequtive_dataframes(
    dir_path: str, every_n: int
) -> Generator[pd.DataFrame, None, None]:
    """Yield DataFrames concatenated from consequtive files in a directory.

    Any non-hidden files in the directory must be loadable as a DataFrame.

    Parameters
    ----------
    dir_path: str
        Directory path of files. Any non-hidden file must be loadable as a DataFrame.
    every_n: int
        Concatenate and yield every N consequtive files.

    Yields
    ------
    pandas.DataFrame
        Concatenated DataFrame.

    """
    assert every_n > 1

    datas = []
    for i, data in enumerate(yield_dataframes(dir_path)):
        datas.append(data)

        # Yield full batches
        if (i + 1) % every_n == 0:
            print(len(datas))
            yield pd.concat(datas)
            datas = []

    # Yield if any remaining
    if len(datas) > 0:
        yield pd.concat(datas)


def save_consequtive_dataframes(prev_dir: str, new_dir: str, every_n: int) -> None:
    """Save DataFrames concatenated from consequtive files in a directory.

    Parameters
    ----------
    prev_dir: str
        Directory path of files. Any non-hidden file must be loadable as a DataFrame.
    new_dir: str
        Directory in which to save the newly concatenated DataFrames.
    every_n: int
        Concatenate and yield every N consequtive files.

    """
    new_dir = process_dir_save_path(new_dir)
    generator = concat_consequtive_dataframes(prev_dir, every_n)

    save_count = 0
    while True:
        try:
            save_dataframe(next(generator), join(new_dir, "batch_" + str(save_count)))
            save_count += 1
        except StopIteration:
            return
