"""Vectorized data processing."""

from typing import Any, Dict, List, Optional, Union

import numpy as np

from cyclops.utils.common import to_list_optional
from cyclops.utils.indexing import take_indices


class Vectorized:
    """Vectorized data.

    Attributes
    ----------
    data: numpy.ndarray
        Data.
    indexes: list of numpy.ndarray
        AAA.
    index_maps: dict
        Name to index.
    axis_names: list of str, optional
        Axis names.

    """

    def __init__(
        self,
        data: np.ndarray,
        indexes: List[Union[List, np.ndarray]],
        axis_names: Optional[List[str]] = None,
    ) -> None:
        """Init."""
        if not isinstance(data, np.ndarray):
            raise ValueError("Data must be a numpy.ndarray.")

        if data.ndim != len(indexes):
            raise ValueError(
                "Number of array axes and the number of indexes do not match."
            )

        axis_names_list = to_list_optional(axis_names)
        if axis_names_list is not None:
            if data.ndim != len(axis_names_list):
                raise ValueError(
                    "Number of array axes and the number of axis names do not match."
                )

        for i, index in enumerate(indexes):
            if not isinstance(index, list) and not isinstance(index, np.ndarray):
                raise ValueError("Indexes must be a list of list or numpy.ndarray.")

            index = np.array(index)

            if len(index) != data.shape[i]:
                raise ValueError(
                    (
                        f"Axis {i} index has {len(index)} elements, "
                        f"but the axis itself has length {data.shape[i]}."
                    )
                )

            if len(np.unique(index)) != len(index):
                raise ValueError(
                    "Each index must have no duplicate values to uniquely identify."
                )

            indexes[i] = index

        self.data: np.ndarray = data
        self.indexes: List[np.ndarray] = indexes
        self.index_maps: List[Dict[str, int]] = [
            {val: i for i, val in enumerate(index)} for index in indexes
        ]

        self.axis_names: Optional[List[str]] = axis_names_list

    def get_data(self) -> np.ndarray:
        """Get the vectorized data.

        Returns
        -------
        numpy.ndarray
            The data.

        """
        return self.data

    def get_by_index(
        self, indexes: List[Optional[Union[List[int], np.ndarray]]]
    ) -> np.ndarray:
        """Get data by indexing each axis.

        Parameters
        ----------
        indexes
            E.g., ([None, [1, 2, 3], None, [20]]), where each element can be
            None, a list, or a numpy.ndarray.

        Returns
        -------
        numpy.ndarray
            Indexed data.

        """
        # Index the data accordingly
        data = take_indices(self.data, indexes)

        # Create the corresponding indexes
        new_indexes = []
        for i, index in enumerate(indexes):
            if index is None:
                new_indexes.append(self.indexes[i])
                continue

            new_indexes.append([self.indexes[i][ind] for ind in index])  # type: ignore

        return Vectorized(data, new_indexes)

    def get_by_value(
        self, indexes: List[Optional[Union[List[Any], np.ndarray]]]
    ) -> np.ndarray:
        """Get data by indexing using the indexes values for each axis.

        Parameters
        ----------
        indexes
            Length of indexes must equal the number of axes in the data.
            E.g., for array (2, 3, 5), we might specify
            get_by_value([None, ["A", "B"], None]).
            A None value will take all values along an axis

        Returns
        -------
        numpy.ndarray
            Indexed data.

        """
        if len(indexes) != len(self.data.shape):
            raise ValueError(
                "Must have the same number of parameters as axes in the data."
            )

        for i, index in enumerate(indexes):
            if index is None:
                continue

            if not isinstance(index, list) and not isinstance(index, np.ndarray):
                raise ValueError(
                    "Each index must either be None, a list or a NumPy array."
                )

            # Map values to indices
            is_in = [val in self.index_maps[i] for val in index]
            if not all(is_in):
                missing = [val for j, val in enumerate(index) if not is_in[j]]
                raise ValueError(f"Index {i} does not have values {', '.join(missing)}")
            indexes[i] = [self.index_maps[i][val] for val in index]

        return self.get_by_index(indexes)

    def _get_axis(self, axis: Union[int, str]) -> int:
        """Get an array axis by index or by name.

        Parameters
        ----------
        axis: int or str
            Axis index or name.

        Returns
        -------
        int
            Axis index.

        """
        # If an index was given
        if isinstance(axis, int):
            if axis >= len(self.indexes) or axis < 0:
                raise ValueError("Axis out of bounds.")
            return axis

        # If an axis name was given
        if isinstance(axis, str):
            if self.axis_names is None:
                raise ValueError(
                    "Axis cannot be a string unless axis_names were specified."
                )
            if axis not in self.axis_names:
                raise ValueError(
                    f"Axis {axis} does not exist in: {', '.join(self.axis_names)}"
                )
            return self.axis_names.index(axis)

        raise ValueError("Axis is an invalid type. Must be an int or string.")

    # def reorder_axes(Union[List[int], List[str]]):
