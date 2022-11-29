"""Tasks and workflows."""

import logging
from typing import Union

import pandas as pd
from prefect import flow, task

from cyclops.process.clean import normalize_events
from cyclops.query.interface import QueryInterface, QueryInterfaceProcessed
from cyclops.utils.log import setup_logging

# Logging.
LOGGER = logging.getLogger(__name__)
setup_logging(print_level="INFO", logger=LOGGER)


@task
def run_query(query_interface: Union[QueryInterface, QueryInterfaceProcessed]):
    """Task to run a query.

    Parameters
    ----------
    query_interface: cyclops.query.interface.QueryInterface or
    cyclops.query.interface.QueryInterfaceProcessed
        Query interface.

    Returns
    -------
    pandas.DataFrame
        Queried data.

    """
    return query_interface.run()


@task
def save_query(
    query_interface: Union[QueryInterface, QueryInterfaceProcessed],
    path: str,
    file_format: str,
):
    """Task to save a query.

    Parameters
    ----------
    query_interface: cyclops.query.interface.QueryInterface or
    cyclops.query.interface.QueryInterfaceProcessed
        Query interface.

    Returns
    -------
    str
        Processed save path for upstream use.

    """
    return query_interface.save(path, file_format=file_format)


@task
def pandas_merge(dataframe_left: pd.DataFrame, dataframe_right: pd.DataFrame, **kwargs):
    """Task to merge two pandas DataFrames.

    Parameters
    ----------
    dataframe_left: pandas.DataFrame
        DataFrame acting as the 'left' table in the join.
    dataframe_right: pandas.DataFrame
        DataFrame acting as the 'right' table in the join.
    **kwargs
        Keyword arguments used in pandas.merge.

    Returns
    -------
    pandas.DataFrame
        Merged data.

    """
    return pd.merge(dataframe_left, dataframe_right, **kwargs)


@flow
def join_queries_flow(query_interface_left, query_interface_right, **pd_merge_kwargs):
    """Workflow to run two queries and join them.

    Parameters
    ----------
    query_interface_left: cyclops.query.interface.QueryInterface or
    cyclops.query.interface.QueryInterfaceProcessed
        Query acting as the 'left' table in the join.
    query_interface_right: cyclops.query.interface.QueryInterface or
    cyclops.query.interface.QueryInterfaceProcessed
        Query acting as the 'right' table in the join.
    **pd_merge_kwargs
        Keyword arguments used in pandas.merge.

    Returns
    -------
    pandas.DataFrame
        The joined data.

    """
    # Run queries
    query_task_left = run_query(query_interface_left)
    query_task_right = run_query(query_interface_right)

    # Join
    join_task = pandas_merge(
        query_task_left.result(raise_on_failure=True),
        query_task_right.result(raise_on_failure=True),
        wait_for=[query_task_left, query_task_right],
        **pd_merge_kwargs
    )

    return join_task.result(raise_on_failure=True)


@flow
def normalize_events_flow(events):
    """Task to clean and normalize event data.

    events: pandas.DataFrame
        Events data.

    Returns
    -------
    pandas.DataFrame
        The normalized events data.

    """
    LOGGER.info("Running normalize events task!")

    return normalize_events(events)
