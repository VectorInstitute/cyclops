import pandas as pd
import torch
import scipy as sp
import numpy as np
import random
import math
import warnings
from torch.utils.data import (
    Dataset,
    DataLoader,
    RandomSampler,
    BatchSampler,
    WeightedRandomSampler,
)
from torch.utils.data.dataloader import default_collate


class LoaderGenerator:
    """
    A class that constructs data loaders
    """

    def __init__(self, *args, **kwargs):
        self.config_dict = self.get_default_config()
        self.config_dict = self.override_config(**kwargs)

    def init_loaders(self):
        """
        Returns a dictionary of dataloaders with keys indicating phases
        """
        raise NotImplementedError

    def get_default_config(self):
        """
        Defines the default config_dict
        """
        raise NotImplementedError

    def override_config(self):
        """
        Overrides the config dict with provided kwargs
        """
        raise NotImplementedError


class ArrayLoaderGenerator(LoaderGenerator):
    def __init__(
        self,
        *args,
        features=None,
        cohort=None,
        fold_id_test="test",
        train_key="train",
        eval_key="val",
        row_id_col="row_id",
        group_var_name=None,
        include_group_in_dataset=False,
        balance_groups=False,
        weight_var_name=None,
        load_features=True,
        **kwargs
    ):
        super().__init__(
            self,
            *args,
            group_var_name=group_var_name,
            include_group_in_dataset=include_group_in_dataset,
            balance_groups=balance_groups,
            **kwargs,
        )
        if isinstance(fold_id_test, str):
            fold_id_test = [fold_id_test]

        self.num_workers = kwargs.get("num_workers", 0)
        self.data_dict = self.get_data_dict(
            features=features,
            cohort=cohort,
            fold_id_test=fold_id_test,
            train_key=train_key,
            eval_key=eval_key,
            row_id_col=row_id_col,
            group_var_name=group_var_name,
            balance_groups=balance_groups,
            load_features=load_features,
            weight_var_name=weight_var_name,
            **kwargs,
        )

    def get_default_config(self):
        return {
            "batch_size": 256,
            "iters_per_epoch": 100,
            "include_group_in_dataset": False,
            "group_var_name": None,
            "weight_var_name": None,
        }

    def override_config(self, **override_dict):
        return {**self.config_dict, **override_dict}

    def compute_group_weights(self, cohort, group_var_name=None):
        if group_var_name is None:
            raise ValueError("Cannot compute group weights if group_var_name is None")
        if "group_weight" in cohort.columns:
            warnings.warn("group_weight is already a column in cohort")

        group_weight_df = (
            cohort.groupby(group_var_name)
            .size()
            .rename("group_weight")
            .to_frame()
            .reset_index()
            .assign(group_weight=lambda x: 1 / (x.group_weight / cohort.shape[0]))
        )
        return group_weight_df

    def init_datasets(self):
        """
        Creates data loaders from inputs
        """
        phases = self.data_dict["row_id"].keys()
        tensor_dict_dict = {
            key: {
                "features": self.data_dict["features"][key],
                "labels": torch.as_tensor(
                    self.data_dict["labels"][key], dtype=torch.long
                ),
                "row_id": torch.LongTensor(self.data_dict["row_id"][key]),
            }
            for key in phases
        }
        if self.config_dict.get("include_group_in_dataset"):
            for key in phases:
                tensor_dict_dict[key]["group"] = torch.as_tensor(
                    np.copy(self.data_dict["group"][key]), dtype=torch.long
                )
        if self.config_dict.get("weight_var_name") is not None:
            for key in phases:
                tensor_dict_dict[key]["weights"] = torch.as_tensor(
                    self.data_dict["weights"][key]
                )
        
        if self.config_dict.get("ids_var") is not None:
            for key in phases:
                tensor_dict_dict[key]['ids'] = torch.as_tensor(
                    self.data_dict["ids"][key]
                )

        dataset_dict = {
            key: ArrayDataset(
                tensor_dict=tensor_dict_dict[key],
                convert_sparse=self.config_dict.get("sparse_mode") == "convert",
            )
            for key in phases
        }

        return dataset_dict

    def init_loaders(self, sample_keys=None):
        """
        Method that converts data and labels to instances of class torch.utils.data.DataLoader
            Returns:
                a dictionary with the same keys as data_dict and label_dict.
                    Each element of the dictionary is an instance of torch.utils.data.DataLoader
                        that yields paired elements of data and labels
        """
        # Convert the data to Dataset
        dataset_dict = self.init_datasets()

        # If the Dataset implements collate_fn, that is used. Otherwise, default_collate is used
        if hasattr(dataset_dict["train"], "collate_fn") and callable(
            getattr(dataset_dict["train"], "collate_fn")
        ):
            collate_fn = dataset_dict["train"].collate_fn
        else:
            collate_fn = default_collate

        # If 'iters_per_epoch' is defined, then a fixed number of random sample batches from the training set
        # are drawn per epoch.
        # Otherwise, an epoch is defined by a full run through all of the data in the dataloader.
        if self.config_dict.get("iters_per_epoch") is not None:
            num_samples = (
                self.config_dict["iters_per_epoch"] * self.config_dict["batch_size"]
            )

        if sample_keys is None:
            sample_keys = ["train"]

        loaders_dict = {}
        for key in dataset_dict.keys():
            if key in sample_keys:
                if self.config_dict.get("balance_groups"):
                    random_sampler = WeightedRandomSampler(
                        # dataset_dict[key],
                        weights=self.data_dict["group_weight"][key],
                        replacement=True,
                        num_samples=num_samples,
                    )
                else:
                    random_sampler = RandomSampler(
                        dataset_dict[key], replacement=True, num_samples=num_samples
                    )

                loaders_dict[key] = DataLoader(
                    dataset_dict[key],
                    batch_sampler=BatchSampler(
                        random_sampler,
                        batch_size=self.config_dict["batch_size"],
                        drop_last=False,
                    ),
                    collate_fn=collate_fn,
                    num_workers=self.num_workers,
                    pin_memory=False
                    if self.config_dict.get("sparse_mode") == "convert"
                    else True,
                )
            else:
                loaders_dict[key] = DataLoader(
                    dataset_dict[key],
                    batch_size=self.config_dict["batch_size"],
                    collate_fn=collate_fn,
                    num_workers=self.num_workers,
                    pin_memory=False
                    if self.config_dict.get("sparse_mode") == "convert"
                    else True,
                )

        return loaders_dict

    def init_loaders_predict(self, *args):
        """
        Creates data loaders from inputs - for use at prediction time
        """

        # Convert the data to Dataset
        dataset_dict = self.init_datasets()

        # If the Dataset implements collate_fn, that is used. Otherwise, default_collate is used
        if hasattr(dataset_dict["train"], "collate_fn") and callable(
            getattr(dataset_dict["train"], "collate_fn")
        ):
            collate_fn = dataset_dict["train"].collate_fn
        else:
            collate_fn = default_collate

        loaders_dict = {
            key: DataLoader(
                dataset_dict[key],
                batch_size=self.config_dict["batch_size"],
                collate_fn=collate_fn,
                num_workers=self.num_workers,
                pin_memory=False
                if self.config_dict.get("sparse_mode") == "convert"
                else True,
            )
            for key in dataset_dict.keys()
        }

        return loaders_dict

    def get_data_dict(
        self,
        features=None,
        cohort=None,
        fold_id_test="test",
        train_key="train",
        eval_key="val",
        row_id_col="row_id",
        label_col="outcome",
        group_var_name=None,
        balance_groups=False,
        weight_var_name=None,
        ids_var = None,
        load_features=True,
        **kwargs
    ):
        """
        Generates a data_dict from a features array and a cohort dataframe.
        Args:
            features: The input feature matrix
            cohort: A dataframe with a column called "fold_id" that maps to fold_id
            fold_id: The fold_id corresponding to the validation set
            fold_id_test: The fold_id corresponding to the test set
            train_key: A string that will be used to refer to the training set in the result
            eval_key: A string that will be used to refer to the validation set in the result
            test_key: A string that will be used to refer to the test set in the result
        """

        # Get the validation fold
        fold_id = self.config_dict.get("fold_id")

        if fold_id is None:
            fold_id = ""

        fold_id = str(fold_id)

        if balance_groups:
            group_weight_df = self.compute_group_weights(
                cohort, group_var_name=group_var_name
            )
            cohort = cohort.merge(group_weight_df)

        if isinstance(fold_id_test, str):
            fold_id_test = [fold_id_test]

        heldout_dict = {
            key: cohort.query('fold_id == "{}"'.format(key)) for key in fold_id_test
        }
        train_eval_fold_ids = list(set(cohort.fold_id) - set(fold_id_test))

        # train_eval_df = cohort.query("fold_id != @fold_id_test")
        train_eval_df = cohort.query("fold_id in @train_eval_fold_ids")
        # Partition the cohort data into the training phases
        cohort_dict = {
            train_key: train_eval_df.query("fold_id != @fold_id"),
            eval_key: train_eval_df.query("fold_id == @fold_id"),
        }
        cohort_dict = {**cohort_dict, **heldout_dict}
        # # Ensure that each partition is sorted and not empty
        cohort_dict = {
            key: value.sort_values(row_id_col)
            for key, value in cohort_dict.items()
            if value.shape[0] > 0
        }

        # # Initialize the data_dict
        data_dict = {}
        # Save the row_id corresponding to unique predictions
        data_dict["row_id"] = {
            key: value[row_id_col].values for key, value in cohort_dict.items()
        }

        # store the group_var_name
        if group_var_name is not None:
            categories = cohort[group_var_name].sort_values().unique()
            print(categories)
            data_dict["group"] = {
                key: pd.Categorical(value[group_var_name], categories=categories).codes
                for key, value in cohort_dict.items()
            }
            self.config_dict["num_groups"] = len(categories)

            if balance_groups:
                data_dict["group_weight"] = {
                    key: np.int64(value["group_weight"].values) for key, value in cohort_dict.items()
                }

        if weight_var_name is not None:
            data_dict["weights"] = {
                key: value[weight_var_name].values.astype(np.float32)
                for key, value in cohort_dict.items()
            }

        # If features should be loaded
        if load_features:
            data_dict["features"] = {}
            for key in cohort_dict.keys():
                data_dict["features"][key] = features[data_dict["row_id"][key], :]

        data_dict["labels"] = {
            key: np.int64((value[self.config_dict["label_col"]] > 0).values)
            for key, value in cohort_dict.items()
        }
        
        # If identifiers and feature names are provided:
        if ids_var is not None:
            data_dict["ids"] = {
                key: value[ids_var].values.astype(np.int64)
                for key, value in cohort_dict.items()
            }

        return data_dict


class ArrayDataset(Dataset):
    """Dataset wrapping arrays (tensor, numpy, or scipy CSR sparse)
    Each sample will be retrieved by indexing arrays along the first dimension.
    Arguments:
        tensor_dict: a dictionary of array inputs that have the same size in the first dimension
        convert_sparse: whether CSR inputs should be converted to torch.SparseTensor
    """

    def __init__(self, tensor_dict, convert_sparse=False):
        self.convert_sparse = convert_sparse
        self.the_len = list(tensor_dict.values())[0].shape[0]
        assert all(self.the_len == tensor.shape[0] for tensor in tensor_dict.values())
        self.tensor_dict = tensor_dict

    def __getitem__(self, index):
        return {key: tensor[index] for key, tensor in self.tensor_dict.items()}

    def __len__(self):
        return self.the_len

    def collate_fn(self, batch):
        """
        Called by Dataloader to aggregate elements into a batch.
        Delegates to collate_helper for typed aggregation
        Arguments:
            batch: a list of dictionaries with same keys as self.tensor_dict
        """
        result = {}
        keys = batch[0].keys()
        for key in keys:
            result[key] = self.collate_helper(tuple(element[key] for element in batch))
        return result

    def collate_helper(self, batch):
        """
        Aggregates a tuple of elements of the same type
        """
        if isinstance(batch[0], sp.sparse.csr_matrix):
            batch_concat = sp.sparse.vstack(batch)
            if not self.convert_sparse:
                return batch_concat
            else:
                return self.csr_to_tensor(batch_concat)
        else:
            return default_collate(batch)

    def csr_to_tensor(self, x):
        """
        Converts CSR matrix to torch.sparse.Tensor
        """
        x = x.tocoo()
        return torch.sparse.FloatTensor(
            torch.LongTensor([x.row, x.col]),
            torch.FloatTensor(x.data),
            torch.Size(x.shape),
        )