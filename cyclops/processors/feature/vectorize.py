"""Vectorized data processing."""

from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np

from cyclops.processors.feature.split import split_idx
from cyclops.utils.indexing import take_indices


class Vectorized:
    """Vectorized data.

    Attributes
    ----------
    data: numpy.ndarray
        Data.
    indexes: list of numpy.ndarray
        Names of each index in each dimension. E.g., for an array with shape
        (2, 10, 5), len(indexes) == 3, and len(indexes[0]) = 2.
    index_maps: list of dict
        A name to index map in each dimension.
    axis_names: list of str
        Axis names.

    """

    def __init__(
        self,
        data: np.ndarray,
        indexes: List[Union[List, np.ndarray]],
        axis_names: List[str],
    ) -> None:
        """Init."""
        if not isinstance(data, np.ndarray):
            raise ValueError("Data must be a numpy.ndarray.")

        if len(indexes) != data.ndim:
            raise ValueError(
                "Number of array axes and the number of indexes do not match."
            )

        if len(axis_names) != data.ndim:
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
        self.axis_names: List[str] = axis_names

    @property
    def shape(self) -> Tuple:
        """Get data shape, as an attribute.

        Returns
        -------
        tuple
            Shape.

        """
        return self.data.shape

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

        return Vectorized(data, new_indexes, self.axis_names)

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

    def get_axis(self, axis: Union[int, str]) -> int:
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

    def split_by_index(
        self,
        axis: Union[str, int],
        indices: List[Union[List[int], np.ndarray]],
        allow_drops: bool = False,
    ):
        """Split the data over an axis using indices.

        Parameters
        ----------
        axis: int or str
            Axis index or name.
        indices: list of numpy.ndarray
            A list of the indices in each split.
        allow_drops:
            If True and certain indices or index names do not appear in any of the
            splits, then drop any which do not appear. Otherwise, raises an error.

        Returns
        -------
        tuple of Vectorized
            Data splits.

        """
        axis_index = self.get_axis(axis)

        # Check for invalid duplicate indices
        all_vals = np.concatenate(indices).ravel()
        if len(all_vals) != len(np.unique(all_vals)):
            raise ValueError(
                "Splits cannot contain duplicate values. "
                "Ensure all values are unique across the splits."
            )

        if not allow_drops:
            required_indices = set(np.arange(self.data.shape[axis_index]))
            diff = required_indices - set(all_vals)
            if len(diff) > 0:
                raise ValueError("Not allowing dropping and missing certain values.")

        get_index: List[Union[int, np.ndarray]] = [None] * len(self.data.shape)
        splits = []
        for split_indices in indices:
            get_index[axis_index] = split_indices
            splits.append(self.get_by_index(get_index))

        return tuple(splits)

    def split_by_index_name(
        self,
        axis: Union[str, int],
        index_names: List[Union[List[Any], np.ndarray]],
        allow_drops: bool = False,
    ):
        """Split the data over an axis using index names.

        Parameters
        ----------
        axis: int or str
            Axis index or name.
        index_names: list of numpy.ndarray or list of any
            A list of the index names in each split.
        allow_drops:
            If True and certain indices or index names do not appear in any of the
            splits, then drop any which do not appear. Otherwise, raises an error.

        Returns
        -------
        tuple of Vectorized
            Data splits.

        """
        axis_index = self.get_axis(axis)
        index_map = self.index_maps[axis_index]

        indices: List[Union[List[int], np.ndarray]] = []
        for names in index_names:
            indices.append([])
            for name in names:
                # If a name is not in the index_map but allowing drops, then ignore it
                if name not in index_map:
                    if allow_drops:
                        continue
                    raise ValueError(f"Invalid index name {name}.")
                indices[-1].append(index_map[name])
            indices[-1] = np.array(indices[-1])

        return self.split_by_index(
            axis=axis_index,
            indices=indices,
            allow_drops=allow_drops,
        )

    def split_by_fraction(
        self,
        axis: Union[str, int],
        fractions: Union[float, List[float]],
        randomize: bool = True,
        seed: int = None,
    ):
        """Split the data over an axis using split fractions.

        Parameters
        ----------
        axis: int or str
            Axis index or name.
        fractions: float or list of float
            Fraction(s) of samples between 0 and 1 to use for each split.
        randomize: bool, default = True
            Whether to randomize the samples in the splits. Otherwise it splits
            the samples in the current order.
        seed: int, optional
            A seed for the randomization.

        Returns
        -------
        tuple of Vectorized
            Data splits.

        """
        axis_index = self.get_axis(axis)

        indices = split_idx(
            fractions=fractions,
            data_len=self.data.shape[axis_index],
            randomize=randomize,
            seed=seed,
        )

        return self.split_by_index(
            axis=axis_index,
            indices=indices,
            allow_drops=False,
        )

    # def reorder_axes(Union[List[int], List[str]]):
