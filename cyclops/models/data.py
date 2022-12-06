"""Data classes."""

import numpy as np
import torch
from torch.utils.data import Dataset

from cyclops.utils.file import load_pickle
from use_cases.util import get_use_case_params

# pylint: disable=invalid-name, too-many-instance-attributes


class VectorizedLoader:
    """Vectorized data loader."""

    def __init__(self, dataset_name: str, use_case: str, data_type: str) -> None:
        """Initialize loader.

        Parameters
        ----------
        dataset_name : str
            Name of the dataset
        use_case : str
            Use-case to be predicted
        data_type : str
            Type of data (tabular, temporal, or combined)

        """
        self.dataset_name = dataset_name
        self.use_case = use_case
        self.data_type = data_type
        self.use_case_params = get_use_case_params(dataset_name, use_case)

        if self.data_type == "tabular":
            self.data = self._get_tabular_data()
        elif self.data_type == "temporal":
            self.data = self._get_temporal_data()
        else:
            self.data = self._get_combined_data()

        (
            self.X_train,
            self.y_train,
            self.X_val,
            self.y_val,
            self.X_test,
            self.y_test,
        ) = self.data

    @property
    def targets(self) -> list:
        """Get the targets feature names, as an attribute.

        Returns
        -------
        list
            list of target features

        """
        if self.data_type == "tabular":
            return self.use_case_params.TABULAR_FEATURES["targets"]
        if self.data_type in ["temporal", "combined"]:
            return self.use_case_params.TEMPORAL_FEATURES["targets"]
        return []

    @property
    def n_features(self) -> int:
        """Get the number of features, as an attribute.

        Returns
        -------
        int
            number of features

        """
        return self.X_train.shape[-1]

    @property
    def n_classes(self) -> int:
        """Get the number of classes, as an attribute.

        Returns
        -------
        int
            number of classes

        """
        return len(np.unique(self.y_train))

    @property
    def classes(self) -> np.ndarray:
        """Get the list of classes, as an attribute.

        Returns
        -------
        np.ndarray
             list of classes

        """
        return np.unique(self.y_train)

    @property
    def train_len(self) -> int:
        """Get the length of the train set, as an attribute.

        Returns
        -------
        int
            length of the train set

        """
        return len(self.y_train)

    @property
    def val_len(self) -> int:
        """Get the length of the validation set, as an attribute.

        Returns
        -------
        int
            length of the validation set

        """
        return len(self.y_val)

    @property
    def test_len(self) -> int:
        """Get the length of the test set, as an attribute.

        Returns
        -------
        int
            length of the test set

        """
        return len(self.y_test)

    @property
    def train_c_counts(self) -> dict:
        """Get the number of instances for each label in the train set, as an attribute.

        Returns
        -------
        dict
            counts of instances per class

        """
        counts = {}
        for c in self.classes:
            counts["c"] = (self.y_train == c).sum()
        return counts

    @property
    def val_c_counts(self) -> dict:
        """Get the number of instances for each label in the validation set, as an \
        attribute.

        Returns
        -------
        dict
            counts of instances per class

        """
        counts = {}
        for c in self.classes:
            counts["c"] = (self.y_val == c).sum()
        return counts

    @property
    def test_c_counts(self) -> dict:
        """Get the number of instances for each label in the test set, as an attribute.

        Returns
        -------
        dict
            counts of the instances per class

        """
        counts = {}
        for c in self.classes:
            counts["c"] = (self.y_test == c).sum()
        return counts

    def _load_tabular(self) -> tuple:
        """Load the vactorized tabular data from files.

        Returns
        -------
        tuple
            tuple: features and labels for train, validation, and test

        """
        X_train_vec = load_pickle(self.use_case_params.UNALIGNED_PATH + "tab_train_X")
        y_train_vec = load_pickle(self.use_case_params.UNALIGNED_PATH + "tab_train_y")
        X_val_vec = load_pickle(self.use_case_params.UNALIGNED_PATH + "tab_val_X")
        y_val_vec = load_pickle(self.use_case_params.UNALIGNED_PATH + "tab_val_y")
        X_test_vec = load_pickle(self.use_case_params.UNALIGNED_PATH + "tab_test_X")
        y_test_vec = load_pickle(self.use_case_params.UNALIGNED_PATH + "tab_test_y")
        return (X_train_vec, y_train_vec, X_val_vec, y_val_vec, X_test_vec, y_test_vec)

    def _load_temporal(self) -> tuple:
        """Load the vactorized temporal data from files.

        Returns
        -------
        tuple
            features and labels for train, validation, and test

        """
        X_train_vec = load_pickle(self.use_case_params.ALIGNED_PATH + "temp_train_X")
        y_train_vec = load_pickle(self.use_case_params.ALIGNED_PATH + "temp_train_y")
        X_val_vec = load_pickle(self.use_case_params.ALIGNED_PATH + "temp_val_X")
        y_val_vec = load_pickle(self.use_case_params.ALIGNED_PATH + "temp_val_y")
        X_test_vec = load_pickle(self.use_case_params.ALIGNED_PATH + "temp_test_X")
        y_test_vec = load_pickle(self.use_case_params.ALIGNED_PATH + "temp_test_y")
        return (X_train_vec, y_train_vec, X_val_vec, y_val_vec, X_test_vec, y_test_vec)

    def _load_combined(self) -> tuple:
        """Load the vactorized combined data from files.

        Returns
        -------
        tuple
            features and labels for train, validation, and test

        """
        X_train_vec = load_pickle(self.use_case_params.ALIGNED_PATH + "comb_train_X")
        y_train_vec = load_pickle(self.use_case_params.ALIGNED_PATH + "comb_train_y")
        X_val_vec = load_pickle(self.use_case_params.ALIGNED_PATH + "comb_val_X")
        y_val_vec = load_pickle(self.use_case_params.ALIGNED_PATH + "comb_val_y")
        X_test_vec = load_pickle(self.use_case_params.ALIGNED_PATH + "comb_test_X")
        y_test_vec = load_pickle(self.use_case_params.ALIGNED_PATH + "comb_test_y")
        return (X_train_vec, y_train_vec, X_val_vec, y_val_vec, X_test_vec, y_test_vec)

    def _get_tabular_data(self) -> tuple:
        """Get tabular data.

        Returns
        -------
        tuple
            features and labels for train, validation, and test

        """
        data_vectors = self._load_tabular()
        X_train = data_vectors[0].data
        y_train = data_vectors[1].data
        X_val = data_vectors[2].data
        y_val = data_vectors[3].data
        X_test = data_vectors[4].data
        y_test = data_vectors[5].data

        return (X_train, y_train, X_val, y_val, X_test, y_test)

    def _prep_temporal(self, vec: np.ndarray) -> np.ndarray:
        """Prepare data vectors with temporal features.

        Parameters
        ----------
        vec : np.ndarray
            input data vector

        Returns
        -------
        np.ndarray
            output data vector

        """
        arr = np.squeeze(vec.data, 0)
        arr = np.moveaxis(arr, 2, 0)
        arr = np.nan_to_num(arr)
        return arr

    def _get_temporal_data(self) -> tuple:
        """Get temporal data.

        Returns
        -------
        tuple
            features and labels for train, validation, and test

        """
        data_vectors = self._load_temporal()
        X_train = self._prep_temporal(data_vectors[0].data)
        y_train = self._prep_temporal(data_vectors[1].data)
        X_val = self._prep_temporal(data_vectors[2].data)
        y_val = self._prep_temporal(data_vectors[3].data)
        X_test = self._prep_temporal(data_vectors[4].data)
        y_test = self._prep_temporal(data_vectors[5].data)

        return (X_train, y_train, X_val, y_val, X_test, y_test)

    def _get_combined_data(self) -> tuple:
        """Get combined data.

        Returns
        -------
        tuple
            features and labels for train, validation, and test

        """
        data_vectors = self._load_combined()
        X_train = self._prep_temporal(data_vectors[0].data)
        y_train = self._prep_temporal(data_vectors[1].data)
        X_val = self._prep_temporal(data_vectors[2].data)
        y_val = self._prep_temporal(data_vectors[3].data)
        X_test = self._prep_temporal(data_vectors[4].data)
        y_test = self._prep_temporal(data_vectors[5].data)

        return (X_train, y_train, X_val, y_val, X_test, y_test)


class PTDataset(Dataset):
    """Pytorch dataset class."""

    def __init__(self, inputs: np.ndarray, target: np.ndarray) -> None:
        """Initialize dataset.

        Parameters
        ----------
        inputs : np.ndarray
            data features
        target : np.ndarray
            data labels

        """
        self.inputs = torch.from_numpy(inputs).float()
        self.target = torch.from_numpy(target).float()

    def __getitem__(self, idx: int) -> tuple:
        """Get data items per index.

        Parameters
        ----------
        idx : int
            index of the data instance

        Returns
        -------
        tuple
            features and labels for the instance

        """
        return self.inputs[idx], self.target[idx]

    def __len__(self) -> int:
        """Get the length of the dataset.

        Returns
        -------
        int
            length of the dataset

        """
        return len(self.target)
