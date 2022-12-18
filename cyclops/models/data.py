"""Data classes."""
import numpy as np
import torch
from torch.utils.data import Dataset

from cyclops.utils.file import join, load_pickle


class VectorizedLoader:
    """Vectorized data loader."""

    def __init__(
        self, dataset_name: str, use_case: str, data_type: str, data_dir: str
    ) -> None:
        """Initialize loader.

        Parameters
        ----------
        dataset_name : str
            Name of the dataset.
        use_case : str
            Use-case to be predicted.
        data_type : str
            Type of data (tabular, temporal, or combined).
        data_dir : str
            Path to the directory of final vectorized data.

        """
        self.dataset_name = dataset_name
        self.use_case = use_case
        self.data_type = data_type
        self.data_dir = data_dir

        if self.data_type == "tabular":
            self.data_path = join(self.data_dir, "unaligned_")
            self.data = self._get_tabular_data()
        elif self.data_type == "temporal":
            self.data_path = join(self.data_dir, "unaligned_")
            self.data = self._get_temporal_data()
        else:
            self.data_path = join(self.data_dir, "aligned_")
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
    def n_features(self) -> int:
        """Get the number of features, as an attribute.

        Returns
        -------
        int
            The number of features in the dataset.

        """
        return self.X_train.shape[-1]

    @property
    def n_classes(self) -> int:
        """Get the number of classes, as an attribute.

        Returns
        -------
        int
            The number of classes in the dataset.

        """
        return len(np.unique(self.y_train))

    @property
    def classes(self) -> np.ndarray:
        """Get the list of classes, as an attribute.

        Returns
        -------
        np.ndarray
            The list of classes in the dataset.

        """
        return np.unique(self.y_train)

    @property
    def train_len(self) -> int:
        """Get the length of the train set, as an attribute.

        Returns
        -------
        int
            The length of the train set.

        """
        return len(self.y_train)

    @property
    def val_len(self) -> int:
        """Get the length of the validation set, as an attribute.

        Returns
        -------
        int
            The length of the validation set.

        """
        return len(self.y_val)

    @property
    def test_len(self) -> int:
        """Get the length of the test set, as an attribute.

        Returns
        -------
        int
            The length of the test set.

        """
        return len(self.y_test)

    @property
    def train_c_counts(self) -> dict:
        """Get the number of instances for each label in the train set, as an attribute.

        Returns
        -------
        dict
            The counts of instances per class.

        """
        counts = {}
        for klass in self.classes:
            counts["c"] = (self.y_train == klass).sum()
        return counts

    @property
    def val_c_counts(self) -> dict:
        """Get the number of instances for each label in the validation set, as an \
        attribute.

        Returns
        -------
        dict
            The counts of instances per class.

        """
        counts = {}
        for klass in self.classes:
            counts["c"] = (self.y_val == klass).sum()
        return counts

    @property
    def test_c_counts(self) -> dict:
        """Get the number of instances for each label in the test set, as an attribute.

        Returns
        -------
        dict
            The counts of the instances per class.

        """
        counts = {}
        for klass in self.classes:
            counts["c"] = (self.y_test == klass).sum()
        return counts

    def _load_tabular(self) -> tuple:
        """Load the vactorized tabular data from files.

        Returns
        -------
        tuple
            Tuple of features and labels for train, validation, and test sets.

        """
        X_train_vec = load_pickle(self.data_path + "tab_train_X")
        y_train_vec = load_pickle(self.data_path + "tab_train_y")
        X_val_vec = load_pickle(self.data_path + "tab_val_X")
        y_val_vec = load_pickle(self.data_path + "tab_val_y")
        X_test_vec = load_pickle(self.data_path + "tab_test_X")
        y_test_vec = load_pickle(self.data_path + "tab_test_y")
        return (X_train_vec, y_train_vec, X_val_vec, y_val_vec, X_test_vec, y_test_vec)

    def _load_temporal(self) -> tuple:
        """Load the vactorized temporal data from files.

        Returns
        -------
        tuple
            The features and labels for train, validation, and test sets.

        """
        X_train_vec = load_pickle(self.data_path + "temp_train_X")
        y_train_vec = load_pickle(self.data_path + "temp_train_y")
        X_val_vec = load_pickle(self.data_path + "temp_val_X")
        y_val_vec = load_pickle(self.data_path + "temp_val_y")
        X_test_vec = load_pickle(self.data_path + "temp_test_X")
        y_test_vec = load_pickle(self.data_path + "temp_test_y")
        return (X_train_vec, y_train_vec, X_val_vec, y_val_vec, X_test_vec, y_test_vec)

    def _load_combined(self) -> tuple:
        """Load the vactorized combined data from files.

        Returns
        -------
        tuple
            The features and labels for train, validation, and test

        """
        X_train_vec = load_pickle(self.data_path + "comb_train_X")
        y_train_vec = load_pickle(self.data_path + "comb_train_y")
        X_val_vec = load_pickle(self.data_path + "comb_val_X")
        y_val_vec = load_pickle(self.data_path + "comb_val_y")
        X_test_vec = load_pickle(self.data_path + "comb_test_X")
        y_test_vec = load_pickle(self.data_path + "comb_test_y")
        return (X_train_vec, y_train_vec, X_val_vec, y_val_vec, X_test_vec, y_test_vec)

    def _get_tabular_data(self) -> tuple:
        """Get tabular data.

        Returns
        -------
        tuple
            The features and labels for train, validation, and test sets.

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
            The input data vector.

        Returns
        -------
        np.ndarray
            The output data vector.

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
            The features and labels for train, validation, and test.

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
            The features and labels for train, validation, and test.

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
            Data features.
        target : np.ndarray
            Data labels.

        """
        self.inputs = torch.from_numpy(inputs).float()
        self.target = torch.from_numpy(target).float()

    def __getitem__(self, idx: int) -> tuple:
        """Get data items per index.

        Parameters
        ----------
        idx : int
            The index of the data instance.

        Returns
        -------
        tuple
            The features and labels for the instance.

        """
        return self.inputs[idx], self.target[idx]

    def __len__(self) -> int:
        """Get the length of the dataset.

        Returns
        -------
        int
            The length of the dataset.

        """
        return len(self.target)
