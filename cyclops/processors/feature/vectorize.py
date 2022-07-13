"""Vectorized data processing."""

from typing import Any, Dict, List, Optional, Union

import numpy as np

from cyclops.utils.indexing import take_indices


class Vectorized:
    """Vectorized data.

    Attributes
    ----------
    indexes: list of numpy.ndarray
        AAA.

    """

    def __init__(
        self,
        data: np.ndarray,
        indexes: List[Union[List, np.ndarray]],
    ) -> None:
        """Init."""
        if not isinstance(data, np.ndarray):
            raise ValueError("Data must be a numpy.ndarray.")

        if not len(data.shape) == len(indexes):
            raise ValueError(
                "Number of array axes and the number of indexes do not match."
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
        self.index_maps_inv: List[Dict[int, str]] = [
            {v: k for k, v in index.items()} for index in self.index_maps
        ]

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

            new_indexes.append(
                [self.index_maps_inv[i][val] for val in index]  # type: ignore
            )

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
                print("missing", missing)
                raise ValueError(f"Index {i} does not have values {', '.join(missing)}")
            indexes[i] = [self.index_maps[i][val] for val in index]

        return self.get_by_index(indexes)
