from typing import Tuple, Union

import numpy as np
import pandas as pd

import networkx as nx

from fecg.utils.pandas.type import to_frame_if_series


def get_pairs(
    data: Union[pd.Series, pd.DataFrame],
    self_match: bool = False,
    combinations: bool = True,
) -> pd.DataFrame:
    """
    Perform a self-cross to generate pairs.

    Parameters
    ----------
    data : pandas.Series or pandas.DataFrame
        Values used to create the pairs.
    self_match : bool, default False
        If False, rows which paired with themselves are excluded.
    combinations : bool, default True
        If True, remove one of two permutations, leaving only pair combinations.

    Returns
    -------
    pandas.DataFrame
        A DataFrame of pairs.

    Notes
    -----
    Often, we are only interested in combinations of pairs, not permutations. For
    example, if evaluating the pairs using a commutative function, where argument order
    does not affect the result, we would want to take only the pair combinations.
    """
    pairs = to_frame_if_series(data).merge(data, how='cross')

    if combinations or not self_match:
        length = len(data)
        idx0 = np.repeat(np.arange(length), length)
        idx1 = np.tile(np.arange(length), length)

        if combinations:
            if self_match:
                pairs = pairs[idx0 <= idx1]
            else:
                pairs = pairs[idx0 < idx1]
        else:
            pairs = pairs[idx0 != idx1]

    return pairs


def split_pairs(pairs: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Split x and y pair columns into two separate DataFrames.

    Parameters
    ----------
    pairs : pandas.DataFrame
        A DataFrame of pairs.

    Returns
    -------
    pandas.DataFrame
        A DataFrame of pairs which had the "_x" columns. Suffix now removed.
    pandas.DataFrame
        A DataFrame of pairs which had the "_y" columns. Suffix now removed.
    """
    half_len = (len(pairs.columns)//2)

    pairs_x = pairs.iloc[:, :half_len]
    pairs_y = pairs.iloc[:, half_len:]

    cols = pairs.columns[:half_len].str.slice(stop=-2)

    pairs_x.columns = cols
    pairs_y.columns = cols

    return pairs_x, pairs_y


def pairs_to_groups(pairs: pd.DataFrame) -> pd.DataFrame:
    """
    Convert pairs of values in a DataFrame to groups of connected values.

    Given a DataFrame with two columns representing pairs of values, this function
    constructs a graph where each value is a node and each pair is an edge. It then
    finds the connected components of this graph, returning each component as a group
    in a DataFrame.

    Parameters
    ----------
    pairs : pandas.DataFrame
        A DataFrame with two columns, each containing values. Each row represents a
        pair of connected values.

    Raises
    ------
    ValueError
        If the input DataFrame does not have exactly two columns.

    Returns
    -------
    pandas.DataFrame
        A DataFrame with columns `value` and `group`. Each row represents a value and
        its associated group ID.
    """
    if pairs.shape[1] != 2:
        raise ValueError("The DataFrame must have exactly two columns.")

    # Create an empty graph
    graph = nx.Graph()

    # Add edges to the graph based on the DataFrame rows
    for _, row in pairs.iterrows():
        graph.add_edge(row[pairs.columns[0]], row[pairs.columns[1]])

    # Find connected components
    components = pd.Series(nx.connected_components(graph))

    # Convert connected components into a groups series
    groups = components.explode()
    groups = pd.Series(groups.index, index=groups.values, name="group")

    return groups
