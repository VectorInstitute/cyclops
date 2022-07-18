"""Vectorized data processing."""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np

from codebase_ops import get_log_file_path
from cyclops.processors.feature.normalization import VectorizedNormalizer
from cyclops.processors.feature.split import split_idx
from cyclops.utils.common import list_swap, to_list
from cyclops.utils.file import save_array
from cyclops.utils.indexing import take_indices_over_axis
from cyclops.utils.log import setup_logging

# Logging.
LOGGER = logging.getLogger(__name__)
setup_logging(log_path=get_log_file_path(), print_level="INFO", logger=LOGGER)


def process_axes(
    vecs: List[Vectorized], axes: Union[str, int, List[str], List[int]]
) -> List[int]:
    """Process a common axis (int/str) or list of axes (list of int/str).

    Parameters
    ----------
    vecs: list of Vectorized
        Vectorized datasets.
    axes: str or int or list or str or list of int, optional
        The axis, or axes if different in the different datasets, over which to
        intersect. Can provide axis indices (int) or names (str).

    Returns
    -------
    list of int
        The processed axes.

    """
    axes_list: List[int]
    if isinstance(axes, list):
        axes_list = [vec.get_axis(axes[i]) for i, vec in enumerate(vecs)]
    else:
        axes_list = [vec.get_axis(axes) for vec in vecs]

    if not len(vecs) == len(axes_list):
        raise ValueError(f"Got {len(axes_list)} axes but needed {len(vecs)}.")

    return axes_list


def intersect_vectorized(
    vecs: List[Vectorized],
    axes: Union[str, int, List[str], List[int]] = 0,
) -> Tuple:
    """Perform an intersection over the indexes of vectorized datasets.

    This is especially useful to align the samples of separate datasets.

    Parameters
    ----------
    vecs: list of Vectorized
        Vectorized datasets.
    axes: str or int or list or str or list of int, optional
        The axis, or axes if different in the datasets, over which to
        intersect. Can provide axis indices (int) or names (str).

    Returns
    -------
    tuple
        A tuple of the Vectorized objects in the same order as provided.

    """
    # Process axes
    axes_list: List[int] = process_axes(vecs, axes)

    # Get intersection
    index_sets = [set(vec.get_index(axes_list[i])) for i, vec in enumerate(vecs)]
    intersection = np.array(list(set.intersection(*index_sets)))

    # Return intersected datasets
    intersected_vecs = [
        vec.take_with_index(axes_list[i], intersection) for i, vec in enumerate(vecs)
    ]

    return tuple(intersected_vecs)


def split_vectorized(
    vecs: List[Vectorized],
    fractions: Union[float, List[float]],
    axes: Union[str, int, List[str], List[int]] = 0,
    randomize: bool = True,
    seed: int = None,
) -> Tuple:
    """Split vectorized datasets matching the index.

    Parameters
    ----------
    vecs: list of Vectorized
        Vectorized datasets.
    fractions: float or list of float
        Fraction(s) of samples between 0 and 1 to use for each split.
    axes: str or int or list or str or list of int, optional
        The axis, or axes if different in the datasets, over which to
        intersect. Can provide axis indices (int) or names (str).
    randomize: bool, default = True
        Whether to randomize the samples in the splits. Otherwise it splits
        the samples in the current order.
    seed: int, optional
        A seed for the randomization.

    Returns
    -------
    tuple of tuple of Vectorized
        A tuple of datasets of splits. All splits are Vectorized objects.

    """
    # Process axes
    axes_list: List[int] = process_axes(vecs, axes)

    indexes = [vec.indexes[axes_list[i]] for i, vec in enumerate(vecs)]

    # Check that index lengths are the same - inexpensive check
    index_lens = [len(index) for index in indexes]

    if index_lens.count(index_lens[0]) != len(index_lens):
        raise ValueError("Indexes must be the same. Consider intersecting the data.")

    # Check that indexes are exactly identical
    if not index_lens[0] == len(set.union(*[set(index) for index in indexes])):
        raise ValueError("Indexes must be the same. Consider intersecting the data.")

    index_splits = split_idx(
        fractions=fractions,
        data_len=index_lens[0],
        randomize=randomize,
        seed=seed,
    )

    splits = [
        vec.split_by_indices(axes_list[i], index_splits) for i, vec in enumerate(vecs)
    ]

    return tuple(splits)


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
    normalizer: VectorizedNormalizer, optional
        Normalizer.

    """

    def __init__(  # pylint: disable=too-many-arguments
        self,
        data: np.ndarray,
        indexes: List[Union[List, np.ndarray]],
        axis_names: List[str],
        normalizer: Optional[VectorizedNormalizer] = None,
        is_normalized: bool = False,
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

        self.normalizer: Optional[VectorizedNormalizer]
        if normalizer is not None:
            self.add_normalizer(normalizer)
        else:
            self.normalizer = None

        self.is_normalized = is_normalized

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

    def add_normalizer(self, normalizer: VectorizedNormalizer) -> None:
        """Add a normalizer.

        Parameters
        ----------
        normalizer: VectorizedNormalizer
            Normalizer to add.

        """
        if not isinstance(normalizer, VectorizedNormalizer):
            raise ValueError("Normalizer must be a VectorizedNormalizer.")

        if self.normalizer is not None:
            LOGGER.warning("Replacing an existing normalizer.")

        index_map = self.index_maps[normalizer.axis]
        normalizer.fit(self.data, index_map)
        self.normalizer = normalizer

    def normalize(self) -> None:
        """Normalize.

        Requires a normalizer to be added and that the data is not already normalized.

        """
        if self.normalizer is None:
            raise ValueError("No normalizer was added.")

        if self.is_normalized:
            raise ValueError("Data normalized. Cannot normalize.")

        index_map = self.index_maps[self.normalizer.axis]
        self.normalizer.transform(self.data, index_map)
        self.is_normalized = True

    def inverse_normalize(self) -> None:
        """Inverse normalize.

        Requires a normalizer to be added and that the data is already normalized.

        """
        if self.normalizer is None:
            raise ValueError("No normalizer was added.")

        if not self.is_normalized:
            raise ValueError("Data not normalized. Cannot inverse normalize.")

        index_map = self.index_maps[self.normalizer.axis]
        self.normalizer.inverse_transform(self.data, index_map)
        self.is_normalized = False

    def save(self, save_path: str, file_format: str = "npy") -> None:
        """Save data to file.

        Parameters
        ----------
        save_path: str
            Path where the file will be saved.
        file_format: str
            File format of the file to save.

        Returns
        -------
        str
            Processed save path for upstream use.

        """
        return save_array(self.data, save_path, file_format=file_format)

    def take_with_indices(
        self, axis: Union[str, int], indices: Union[List[int], np.ndarray]
    ) -> Vectorized:
        """Get data by indexing an axis.

        Parameters
        ----------
        axis: int or str
            Axis index or name.
        indices
            Array/list of indices to take along the axis.

        Returns
        -------
        numpy.ndarray
            Indexed data.

        """
        axis_index = self.get_axis(axis)

        # Index the data accordingly
        data = take_indices_over_axis(self.data, axis_index, indices)

        # Create the corresponding indexes
        new_indexes = list(self.indexes)
        new_indexes[axis_index] = [self.indexes[axis_index][ind] for ind in indices]

        return Vectorized(data, new_indexes, self.axis_names)

    def take_with_index(
        self, axis: Union[str, int], index: Union[List[Any], np.ndarray]
    ) -> Vectorized:
        """Get data by indexing an axis using its index.

        Parameters
        ----------
        axis: int or str
            Axis index or name.
        index: numpy.ndarray or list of any
            Array/list of index values to take along the axis.

        Returns
        -------
        numpy.ndarray
            Indexed data.

        """
        axis_index = self.get_axis(axis)
        index_map = self.index_maps[axis_index]

        if not isinstance(index, list) and not isinstance(index, np.ndarray):
            raise ValueError("Index must either be a list or a NumPy array.")

        # Map values to indices
        missing = [val for val in index if val not in index_map]
        if len(missing) > 0:
            raise ValueError(f"Index does not have values {', '.join(missing)}.")

        indices = [index_map[val] for val in index]

        return self.take_with_indices(axis_index, indices)

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

    def get_index(self, axis: Union[int, str]) -> np.ndarray:
        """Get an axis index by index or by name.

        Parameters
        ----------
        axis: int or str
            Axis index or name.

        Returns
        -------
        numpy.ndarray
            Axis index.

        """
        return self.indexes[self.get_axis(axis)]

    def split_by_indices(
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

        splits = []
        for split_indices in indices:
            splits.append(self.take_with_indices(axis_index, split_indices))

        return tuple(splits)

    def split_by_index(
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

        return self.split_by_indices(
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

        return self.split_by_indices(
            axis=axis_index,
            indices=indices,
            allow_drops=False,
        )

    def split_out(
        self,
        axis: Union[str, int],
        index_names: Union[List[Any], np.ndarray],
    ):
        """Split out some indexes by name.

        Parameters
        ----------
        axis: int or str
            Axis index or name.
        index_names: list of numpy.ndarray or list of any
            A list of the index names in each split.
        allow_drops:
            If True and certain indices or index names do not appear in any of the
            splits, then drop any which do not appear. Otherwise, raises an error.

        """
        axis_index = self.get_axis(axis)
        index_names = np.array(index_names)
        remaining = np.setdiff1d(self.indexes[axis_index], index_names)

        return self.split_by_index(
            axis=axis_index,
            index_names=[remaining, index_names],
            allow_drops=False,
        )

    def rename_axis(self, axis: Union[str, int], name: str) -> None:
        """Rename an axis.

        Parameters
        ----------
        axis: int or str
            Old axis index or name.
        name: str
            New axis name.

        """
        axis_index = self.get_axis(axis)
        self.axis_names[axis_index] = name

    def reorder_axes(
        self,
        source: Union[str, int, List[str], List[int]],
        destination: Union[str, int, List[str], List[int]],
    ) -> None:
        """Move axes to new positions, being functionally similar to numpy.moveaxis.

        Other axes remain in their original order.

        Parameters
        ----------
        source: int or list of int
            Original positions of the axes to move. These must be unique.
        destination: int or list of int
            Destination positions for each of the original axes.
            These must also be unique.

        """
        # Process axes
        source_list: List[int] = [self.get_axis(axis) for axis in to_list(source)]
        destination_list: List[int] = [
            self.get_axis(axis) for axis in to_list(destination)
        ]

        # Call moveaxis before meta changes in case there are errors
        self.data = np.moveaxis(self.data, source_list, destination_list)

        # Update meta
        self.indexes = list_swap(self.indexes, source_list, destination_list)
        self.index_maps = list_swap(self.index_maps, source_list, destination_list)
        self.axis_names = list_swap(self.axis_names, source_list, destination_list)
