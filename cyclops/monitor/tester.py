"""Tester Module for drift detection with TSTester and DCTester submodules."""

import os
from functools import partial
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import sklearn
import torch
from alibi_detect.cd import (
    ChiSquareDrift,
    ClassifierDrift,
    ContextMMDDrift,
    FETDrift,
    KSDrift,
    LearnedKernelDrift,
    LSDDDrift,
    MMDDrift,
    SpotTheDiffDrift,
    TabularDrift,
)
from alibi_detect.utils.pytorch.kernels import DeepKernel, GaussianRBF
from datasets import Dataset, DatasetDict, concatenate_datasets
from monai.transforms import Lambdad
from scipy.special import expit as sigmoid
from scipy.special import softmax
from sklearn.base import BaseEstimator
from torch import nn
from torch.utils.data import Dataset as TorchDataset

from cyclops.data.utils import apply_transforms
from cyclops.models.catalog import wrap_model
from cyclops.monitor.utils import DetectronModule, DummyCriterion, get_args


class TSTester:
    """Two Sample Statistical Tester.

    This class provides a set of methods to test for a shift in data distribution
    between a reference dataset and a target dataset. It supports several
    two-sample statistical tests, including Kolmogorov-Smirnov (KS), Chi-Square (Chi2),
    Maximum Mean Discrepancy (MMD), Contextual Maximum Mean Discrepancy (CTX-MMD),
    Localized Subspace Distribution Distance (LSDD), Likelihood Ratio (LK),
    Fisher Exact Test (FET), and Tabular Drift (Tabular).

    Parameters
    ----------
    tester_method: str
        two-sample statistical test method
        available methods are:
        "ks", "chi2", "mmd", "ctx_mmd", "lsdd", "lk", "fet", "tabular"
    p_val_threshold: float, optional (default=0.05)
        p-value threshold for statistical significance
    **kwargs: Any
        additional arguments to pass to the statistical test method

    Methods
    -------
    get_available_test_methods() -> List[str]
        Get available test methods
    fit(X_s: np.ndarray, **kwargs: Any) -> None
        Fit statistical test method to reference data
    test_shift(X_t: np.ndarray, **kwargs: Any) -> Tuple[float, float]
        Test for shift in data

    Examples
    --------
    >>> from cyclops.monitor.tester import TSTester
    >>> import numpy as np
    >>> np.random.seed(0)
    >>> X_s = np.random.normal(0, 1, (100, 10))
    >>> X_t = np.random.normal(1, 1, (100, 10))
    >>> tester = TSTester("ks")
    >>> tester.fit(X_s)
    >>> p_val, dist = tester.test_shift(X_t)
    >>> print(p_val, dist)
    0.0 0.3
    """

    def __init__(
        self,
        tester_method: str,
        p_val_threshold: float = 0.05,
        **kwargs: Any,
    ) -> None:
        self.tester_method = tester_method
        self.method: Any = None
        self.p_val_threshold = p_val_threshold

        # dict where the key is the string of each test_method
        # and the value is the class of the test_method
        self.tester_methods = {
            "ks": KSDrift,
            "chi2": ChiSquareDrift,
            "mmd": MMDDrift,
            "ctx_mmd": ContextMMDWrapper,
            "lsdd": LSDDDrift,
            "lk": LKWrapper,
            "fet": FETDrift,
            "tabular": TabularDrift,
        }

        self.method_args = kwargs
        if "backend" not in self.method_args:
            self.method_args["backend"] = "pytorch"

        if self.tester_method not in self.tester_methods:
            raise ValueError(
                f"Tester method {self.tester_method} not supported. \
                    Must be one of {self.tester_methods.keys()}",
            )

    def get_available_test_methods(self) -> List[str]:
        """Return list of available test methods."""
        return list(self.tester_methods.keys())

    def fit(
        self,
        X_s: Union[Dataset, DatasetDict, TorchDataset, np.ndarray, torch.Tensor],
        **kwargs: Any,
    ) -> None:
        """Initialize test method to source data.

        Parameters
        ----------
        X_s: np.ndarray
            reference dataset
        **kwargs: Any
            additional arguments to pass to the statistical test method
        """
        if isinstance(X_s, np.ndarray):
            X_s = X_s.astype("float32")
        # append alternative="two-sided" to method_args"
        # if not already present
        # this is required for the FET test
        # to work properly
        if self.tester_method == "fet" and "alternative" not in self.method_args:
            self.method_args["alternative"] = "two-sided"

        if self.tester_method == "ctx_mmd":
            if "ds_source" in kwargs:
                self.method = self.tester_methods[self.tester_method](
                    X_s,
                    ds_source=kwargs["ds_source"],
                    **get_args(
                        self.tester_methods[self.tester_method],
                        self.method_args,
                    ),
                )
            else:
                raise ValueError(
                    "ds_source must be provided to fit method \
                    for ctx_mmd.",
                )
        else:
            self.method = self.tester_methods[self.tester_method](
                X_s,
                **get_args(self.tester_methods[self.tester_method], self.method_args),
            )

    def test_shift(
        self,
        X_t: Union[Dataset, DatasetDict, TorchDataset, np.ndarray, torch.Tensor],
        **kwargs: Any,
    ) -> Tuple[float, float]:
        """Test for shift in data.

        Parameters
        ----------
        X_t: np.ndarray
            target dataset
        **kwargs: Any
            additional arguments to pass to the statistical test method

        Returns
        -------
        Tuple[float, float]
            p-value and distance between reference and target datasets
        """
        if isinstance(X_t, np.ndarray):
            X_t = X_t.astype("float32")
            num_features = X_t.shape[1]

        if self.tester_method == "ctx_mmd":
            if "ds_target" in kwargs:
                preds = self.method.predict(
                    X_t,
                    ds_target=kwargs["ds_target"],
                    **get_args(self.method.predict, self.method_args),
                )

            else:
                raise ValueError(
                    "ds_target must be provided to test_shift method \
                    for ctx_mmd.",
                )
        else:
            preds = self.method.predict(
                X_t,
                **get_args(self.method.predict, self.method_args),
            )

        p_val = preds["data"]["p_val"]
        dist = preds["data"]["distance"]

        if isinstance(p_val, np.ndarray):
            idx = np.argmin(p_val)
            p_val = p_val[idx]
            dist = dist[idx]

        if self.tester_method in ["ks", "chi2", "fet", "tabular"]:
            self.p_val_threshold = self.p_val_threshold / num_features

        return p_val, dist


class DCTester:
    """Domain Classifier Tester.

    This class provides a set of methods to test for a shift in data distribution
    between a reference dataset and a target dataset. It supports several
    domain classifier tests, including SpotTheDiff (spot_the_diff),
    ClassifierDrift (classifier), and Detectron (detectron).

    SpotTheDiff
    -----------
    The spot-the-diff drift detector is an extension of the Classifier
    drift detector where the classifier is specified in a manner that makes
    detections interpretable at the feature level when they occur.
    Documentation for SpotTheDiff can be found here:
    https://docs.seldon.io/projects/alibi-detect/en/stable/cd/methods/spotthediffdrift.html

    Examples
    --------
    >>> from cyclops.monitor.tester import DCTester
    >>> import numpy as np
    >>> np.random.seed(0)
    >>> X_s = np.random.normal(0, 1, (100, 10))
    >>> X_t = np.random.normal(1, 1, (100, 10))
    >>> tester = DCTester("spot_the_diff")
    >>> tester.fit(X_s)
    >>> p_val, dist = tester.test_shift(X_t)

    ClassifierDrift
    ---------------
    The classifier-based drift detector Lopez-Paz and Oquab, 2017 simply tries
    to correctly distinguish instances from the reference set vs. the test set.
    The classifier is trained to output the probability that a given instance
    belongs to the test set. If the probabilities it assigns to unseen test instances
    are significantly higher (as determined by a Kolmogorov-Smirnov test) to
    those it assigns to unseen reference instances then the test set must differ
    from the reference set and drift is flagged. Documentation for ClassifierDrift
    can be found here:
    https://docs.seldon.io/projects/alibi-detect/en/stable/cd/methods/classifierdrift.html

    Examples
    --------
    >>> from cyclops.monitor.tester import DCTester
    >>> import numpy as np
    >>> np.random.seed(0)
    >>> X_s = np.random.normal(0, 1, (100, 10))
    >>> X_t = np.random.normal(1, 1, (100, 10))
    >>> from sklearn.linear_model import LogisticRegression
    >>> model = LogisticRegression()
    >>> tester = DCTester("classifier", model=model)
    >>> tester.fit(X_s)
    >>> p_val, dist = tester.test_shift(X_t)

    Detectron
    ---------
    "A Learning Based Hypothesis Test for Harmful Covariate Shift".
    The Detectron method utilizes the discordance between an ensemble of
    classifiers trained to agree on training data and disagree on test data.
    A loss function is derived for training this ensemble, and the disagreement
    rate and entropy are shown to be powerful discriminative statistics
    for harmful covariate shift (HCS).

    Examples
    --------
    >>> from cyclops.monitor.tester import DCTester

    >>> nih_ds = load_nihcxr(DATA_DIR)
    >>> base_model = DenseNet(weights="densenet121-res224-nih")
    >>> detectron = DCTester("detectron", model=base_model)
    >>> detectron = DCTester("detectron",
                        base_model=base_model,
                        model=base_model,
                        feature_columns="image",
                        transforms=transforms,
                        task="multilabel",
                        max_epochs_per_model=5,
                        ensemble_size=5,
                        lr=0.01,
                        num_runs=5)

    >>> detectron.fit(source_ds)
    >>> p_val, distance = detectron.predict(target_ds)

    Parameters
    ----------
    tester_method: str
        domain classifier test method
        Must be one of: "spot_the_diff", "classifier" or "detectron"
    p_val_threshold: float
        p-value threshold for statistical test
    model: str
        model to use for chosen test method.
        Must be either a sklearn or pytorch model.

    Methods
    -------
    get_available_test_methods()
        Get available test methods
    fit(X_s: np.ndarray, **kwargs)
        Fit domain classifier to reference data
    test_shift(X_t: np.ndarray, **kwargs)
        Test for shift in data

    """

    def __init__(
        self,
        tester_method: str,
        p_val_threshold: float = 0.05,
        **kwargs: Any,
    ) -> None:
        self.tester_method = tester_method
        self.p_val_threshold = p_val_threshold
        self.method_args = kwargs
        self.tester: Any = None

        self.tester_methods = {
            "spot_the_diff": SpotTheDiffDrift,
            "classifier": ClassifierDrift,
            "detectron": Detectron,
        }
        if self.tester_method not in self.tester_methods:
            raise ValueError(
                f"Tester method {self.tester_method} not supported. \
                Must be one of {self.tester_methods.keys()}",
            )

    def get_available_test_methods(self) -> List[str]:
        """Return list of available test methods."""
        return list(self.tester_methods.keys())

    def fit(
        self,
        X_s: Union[Dataset, DatasetDict, TorchDataset, np.ndarray, torch.Tensor],
    ) -> None:
        """Initialize test method to source data."""
        if isinstance(X_s, np.ndarray):
            X_s = X_s.astype("float32")

        if self.tester_method == "spot_the_diff":
            if not isinstance(X_s, np.ndarray):
                raise ValueError("spot_the_diff only supports numpy arrays as input.")
            self.tester = self.tester_methods[self.tester_method](
                X_s,
                backend="pytorch",
                **get_args(self.tester_methods[self.tester_method], self.method_args),
            )
        elif self.tester_method == "classifier":
            if not isinstance(X_s, np.ndarray):
                raise ValueError("classifier only supports numpy arrays as input.")
            if isinstance(self.method_args["model"], torch.nn.Module):
                self.method_args["backend"] = "pytorch"
            elif isinstance(self.method_args["model"], sklearn.base.ClassifierMixin):
                self.method_args["backend"] = "sklearn"
            else:
                raise ValueError(
                    "Model must be one of: torch.nn.Module or \
                    sklearn.base.BaseEstimator",
                )
            self.tester = self.tester_methods[self.tester_method](
                X_s,
                **get_args(self.tester_methods[self.tester_method], self.method_args),
            )
        elif self.tester_method == "detectron":
            self.tester = self.tester_methods[self.tester_method](
                X_s,
                **get_args(self.tester_methods[self.tester_method], self.method_args),
            )

    def test_shift(
        self,
        X_t: Union[Dataset, DatasetDict, TorchDataset, np.ndarray, torch.Tensor],
    ) -> Tuple[float, float]:
        """Test for shift in data."""
        if isinstance(X_t, np.ndarray):
            X_t = X_t.astype("float32")
        preds = self.tester.predict(X_t)

        p_val = preds["data"]["p_val"]
        dist = preds["data"]["distance"]
        return p_val, dist


class ContextMMDWrapper:
    """Wrapper for ContextMMDDrift."""

    def __init__(
        self,
        X_s: Union[Dataset, DatasetDict, TorchDataset, np.ndarray, torch.Tensor],
        ds_source: Dataset,
        context_generator: Union[torch.nn.Module, BaseEstimator],
        *,
        backend: str = "tensorflow",
        p_val: float = 0.05,
        preprocess_x_ref: bool = False,
        update_ref: Optional[Dict[str, int]] = None,
        preprocess_fn: Optional[Callable[..., Any]] = None,
        x_kernel: Optional[Callable[..., Any]] = None,
        c_kernel: Optional[Callable[..., Any]] = None,
        n_permutations: int = 1000,
        prop_c_held: float = 0.25,
        n_folds: int = 5,
        batch_size: Optional[int] = 256,
        device: Optional[Union[str, torch.device]] = None,
        input_shape: Optional[Tuple[int, ...]] = None,
        data_type: Optional[str] = None,
        verbose: bool = False,
    ):
        self.context_generator = context_generator

        c_source = context_generator.transform(ds_source)

        args = [
            backend,
            p_val,
            preprocess_x_ref,
            update_ref,
            preprocess_fn,
            x_kernel,
            c_kernel,
            n_permutations,
            prop_c_held,
            n_folds,
            batch_size,
            device,
            input_shape,
            data_type,
            verbose,
        ]

        self.tester = ContextMMDDrift(X_s, c_source, *args)

    def predict(
        self,
        X_t: np.ndarray[float, np.dtype[np.float64]],
        ds_target: Dataset,
        **kwargs: Dict[str, Any],
    ) -> Any:
        """Predict if there is drift in the data."""
        c_target = self.context_generator.transform(ds_target)
        return self.tester.predict(
            X_t,
            c_target,
            **get_args(self.tester.predict, kwargs),
        )


class LKWrapper:
    """Wrapper for LKWrapper."""

    def __init__(
        self,
        X_s: Union[Dataset, DatasetDict, TorchDataset, np.ndarray, torch.Tensor],
        projection: torch.nn.Module,
        *,
        backend: str = "pytorch",
        p_val: float = 0.05,
        x_ref_preprocessed: bool = False,
        preprocess_at_init: bool = True,
        update_x_ref: Optional[Dict[str, int]] = None,
        preprocess_fn: Optional[Callable] = None,
        n_permutations: int = 100,
        batch_size_permutations: int = 1000000,
        var_reg: float = 1e-5,
        reg_loss_fn: Callable = (lambda kernel: 0),
        train_size: Optional[float] = 0.75,
        retrain_from_scratch: bool = True,
        optimizer: Optional[Callable] = None,
        learning_rate: float = 1e-3,
        batch_size: int = 32,
        batch_size_predict: int = 32,
        preprocess_batch_fn: Optional[Callable] = None,
        epochs: int = 3,
        num_workers: int = 0,
        verbose: int = 0,
        train_kwargs: Optional[dict] = None,
        device: Optional[str] = None,
        dataset: Optional[Callable] = None,
        dataloader: Optional[Callable] = None,
        input_shape: Optional[tuple] = None,
        data_type: Optional[str] = None,
        kernel_a: nn.Module = None,
        kernel_b: nn.Module = None,
        eps: str = "trainable",
    ):
        self.proj = projection

        kernel_a = GaussianRBF(trainable=True) if kernel_a is None else kernel_a
        kernel_b = GaussianRBF(trainable=True) if kernel_b is None else kernel_b
        kernel = DeepKernel(self.proj, kernel_a, kernel_b, eps)

        args = [
            backend,
            p_val,
            x_ref_preprocessed,
            preprocess_at_init,
            update_x_ref,
            preprocess_fn,
            n_permutations,
            batch_size_permutations,
            var_reg,
            reg_loss_fn,
            train_size,
            retrain_from_scratch,
            optimizer,
            learning_rate,
            batch_size,
            batch_size_predict,
            preprocess_batch_fn,
            epochs,
            num_workers,
            verbose,
            train_kwargs,
            device,
            dataset,
            dataloader,
            input_shape,
            data_type,
        ]
        self.tester = LearnedKernelDrift(X_s, kernel, *args)

    def predict(
        self,
        X_t: np.ndarray[float, np.dtype[np.float64]],
        **kwargs: Dict[str, Any],
    ) -> Any:
        """Predict if there is drift in the data."""
        return self.tester.predict(X_t, **get_args(self.tester.predict, kwargs))


class Detectron:
    """Implementation of the ICLR 2023 paper.

    "A Learning Based Hypothesis Test for Harmful Covariate Shift".

    The Detectron method utilizes the discordance between an ensemble of
    classifiers trained to agree on training data and disagree on test data.
    A loss function is derived for training this ensemble, and the disagreement
    rate and entropy are shown to be powerful discriminative statistics
    for harmful covariate shift (HCS).

    @inproceedings{
    ginsberg2023a,
    title = {A Learning Based Hypothesis Test for Harmful Covariate Shift},
    author = {Tom Ginsberg and Zhongyuan Liang and Rahul G Krishnan},
    booktitle = {The Eleventh International Conference on Learning Representations },
    year = {2023},
    url = {https://openreview.net/forum?id=rdfgqiwz7lZ}
    }

    Parameters
    ----------
    X_s
        reference dataset
    base_model
        pre-trained base model to use for Detectron
    feature_column
        feature column to use for Detectron
    model
        optional model to use for Detectron,
        if different from base_model
    transforms
        optional transforms to apply to data
    splits_mapping
        optional mapping of splits to use for Detectron
        Defaults to {"train": "train", "test": "test"}
    num_runs
        number of runs to use for Detectron. Defaults to 100.
    sample_size
        sample size to use for Detectron. Defaults to 50.
    batch_size
        batch size to use for Detectron. Defaults to 32.
    ensemble_size
        ensemble size for each CDC. Defaults to 5.
    max_epochs_per_model
        maximum number of epochs to use for each ensemble
    lr
        learning rate to use for fitting CDCs. Defaults to 1e-3.
    num_workers
        number of workers to use for data loading
    save_dir
        directory to save Detectron models to. Defaults to "detectron".
    task
        task to use for Detectron. Defaults to "multiclass".
        Used to infer loss function and activation function.

    Methods
    -------
    fit(X_s: np.ndarray, **kwargs: Any) -> None
        Fit Detectron to reference data
    test_shift(X_t: np.ndarray, **kwargs: Any) -> Tuple[float, float]
        Test for shift in target data
    """

    def __init__(
        self,
        X_s: Union[Dataset, DatasetDict, TorchDataset, np.ndarray, torch.Tensor],
        base_model: Union[nn.Module, BaseEstimator],
        feature_column: str,
        model: nn.Module = None,
        transforms: Callable = None,
        splits_mapping=None,
        num_runs: int = 100,
        sample_size: int = 50,
        batch_size: int = 32,
        ensemble_size: int = 5,
        max_epochs_per_model: int = 10,
        lr: float = 1e-3,
        num_workers: int = os.cpu_count(),
        save_dir: str = None,
        task="multiclass",
    ):
        if splits_mapping is None:
            splits_mapping = {"train": "train", "test": "test"}
        if model is None:
            self.model = base_model
        else:
            self.model = model
        if isinstance(base_model, nn.Module):
            self.base_model = wrap_model(
                base_model,
                batch_size=batch_size,
            )
            self.base_model.initialize()
        else:
            self.base_model = wrap_model(base_model)
            self.base_model.initialize()
        self.feature_column = feature_column
        if transforms:
            self.transforms = partial(apply_transforms, transforms=transforms)
            model_transforms = transforms
            model_transforms.transforms = model_transforms.transforms + (
                Lambdad(
                    keys=("mask", "labels"),
                    func=lambda x: np.array(x),
                    allow_missing_keys=True,
                ),
            )
            self.model_transforms = partial(
                apply_transforms,
                transforms=model_transforms,
            )
        else:
            self.transforms = transforms
            self.model_transforms = transforms
        self.splits_mapping = splits_mapping
        self.num_runs = num_runs
        self.sample_size = sample_size
        self.batch_size = batch_size
        self.ensemble_size = ensemble_size
        self.max_epochs_per_model = max_epochs_per_model
        self.lr = lr
        self.num_workers = num_workers
        self.task = task
        if save_dir is None:
            self.save_dir = "detectron"

        self.fit(X_s)

    def fit(self, X_s: Union[Dataset, DatasetDict, np.ndarray, TorchDataset]):
        """Fit the Detectron model."""
        X_s = self.split_dataset(X_s)
        self.p = X_s
        self.cal_record = {
            "seed": [],
            "ensemble": [],
            "count": [],
            "rejection_rate": [],
        }
        for seed in range(self.num_runs):
            # train ensemble of for split 'p*'
            for e in range(1, self.ensemble_size + 1):
                alpha = 1 / (len(X_s) * self.sample_size + 1)
                model = wrap_model(
                    DetectronModule(
                        self.model,
                        feature_column=self.feature_column,
                        alpha=alpha,
                    ),
                    batch_size=self.batch_size,
                    criterion=DummyCriterion,
                    max_epochs=self.max_epochs_per_model,
                    lr=self.lr,
                    num_workers=self.num_workers,
                    save_dir=self.save_dir,
                    concatenate_features=False,
                )
                if isinstance(X_s, (Dataset, DatasetDict)):
                    # create p/p* splits

                    p = (
                        X_s[self.splits_mapping["train"]]
                        .shuffle()
                        .select(range(self.sample_size))
                    )
                    p = p.add_column("mask", [1] * len(p))
                    p_pseudolabels = self.base_model.predict(
                        X=p,
                        feature_columns=self.feature_column,
                        transforms=self.transforms,
                        only_predictions=True,
                    )
                    p_pseudolabels = self.format_pseudolabels(np.array(p_pseudolabels))
                    p = p.add_column("labels", p_pseudolabels.tolist())

                    pstar = (
                        X_s[self.splits_mapping["test"]]
                        .shuffle()
                        .select(range(self.sample_size))
                    )
                    pstar = pstar.add_column("mask", [0] * len(pstar))
                    pstar_pseudolabels = self.base_model.predict(
                        X=pstar,
                        feature_columns=self.feature_column,
                        transforms=self.transforms,
                        only_predictions=True,
                    )
                    pstar_pseudolabels = self.format_pseudolabels(
                        np.array(pstar_pseudolabels),
                    )
                    pstar = pstar.add_column("labels", pstar_pseudolabels.tolist())

                    p_pstar = concatenate_datasets([p, pstar], axis=0)
                    p_pstar = p_pstar.train_test_split(test_size=0.5, shuffle=True)

                    train_features = [self.feature_column]
                    train_features.extend(["labels", "mask"])
                    model.fit(
                        X=p_pstar,
                        feature_columns=train_features,
                        target_columns="mask",  # placeholder, not used in dummycriterion
                        transforms=self.model_transforms,
                        splits_mapping={"train": "train", "validation": "test"},
                    )

                    model.load_model(
                        os.path.join(
                            self.save_dir,
                            "saved_models/DetectronModule/best_model.pt",
                        ),
                    )
                    pstar_logits = model.predict(
                        X=pstar,
                        feature_columns=self.feature_column,
                        transforms=self.model_transforms,
                        only_predictions=True,
                    )
                    count = (
                        self.format_pseudolabels(np.array(pstar_logits))
                        != pstar_pseudolabels
                    ).sum()
                    self.cal_record["seed"].append(seed)
                    self.cal_record["ensemble"].append(e)
                    self.cal_record["count"].append(count)
                    self.cal_record["rejection_rate"].append(
                        count / pstar_pseudolabels.size,
                    )

                elif isinstance(X_s, np.ndarray):
                    raise NotImplementedError("Numpy arrays are not supported yet.")
                elif isinstance(X_s, TorchDataset):
                    raise NotImplementedError("PyTorch datasets are not supported yet.")

    def predict(self, X_t: Union[Dataset, DatasetDict, np.ndarray, TorchDataset]):
        """Detect shift in target dataset."""
        X_t = self.split_dataset(X_t)
        self.test_record = {
            "seed": [],
            "ensemble": [],
            "count": [],
            "rejection_rate": [],
        }
        for seed in range(self.num_runs):
            # train ensemble of for split 'p*'
            for e in range(1, self.ensemble_size + 1):
                alpha = 1 / (len(X_t) * self.sample_size + 1)
                model = wrap_model(
                    DetectronModule(
                        self.model,
                        feature_column=self.feature_column,
                        alpha=alpha,
                    ),
                    batch_size=self.batch_size,
                    criterion=DummyCriterion,
                    max_epochs=self.max_epochs_per_model,
                    lr=self.lr,
                    num_workers=self.num_workers,
                    save_dir=self.save_dir,
                    concatenate_features=False,
                )
                model.initialize()
                if isinstance(X_t, (Dataset, DatasetDict)):
                    # create p/q splits
                    p = (
                        self.p[self.splits_mapping["train"]]
                        .shuffle()
                        .select(range(self.sample_size))
                    )
                    p = p.add_column("mask", [1] * len(p))
                    p_pseudolabels = self.base_model.predict(
                        X=p,
                        feature_columns=self.feature_column,
                        transforms=self.transforms,
                        only_predictions=True,
                    )
                    p_pseudolabels = self.format_pseudolabels(np.array(p_pseudolabels))
                    p = p.add_column("labels", p_pseudolabels.tolist())
                    q = (
                        X_t[self.splits_mapping["test"]]
                        .shuffle()
                        .select(range(self.sample_size))
                    )
                    q = q.add_column("mask", [0] * len(q))
                    q_pseudolabels = self.base_model.predict(
                        X=q,
                        feature_columns=self.feature_column,
                        transforms=self.transforms,
                        only_predictions=True,
                    )
                    q_pseudolabels = self.format_pseudolabels(np.array(q_pseudolabels))
                    q = q.add_column("labels", q_pseudolabels.tolist())
                    p_q = concatenate_datasets([p, q], axis=0)
                    p_q = p_q.train_test_split(test_size=0.5, shuffle=True)
                    train_features = [self.feature_column]
                    train_features.extend(["labels", "mask"])
                    model.fit(
                        X=p_q,
                        feature_columns=train_features,
                        target_columns="mask",  # placeholder, not used in dummycriterion
                        transforms=self.model_transforms,
                        splits_mapping={"train": "train", "validation": "test"},
                    )

                    model.load_model(
                        os.path.join(
                            self.save_dir,
                            "saved_models/DetectronModule/best_model.pt",
                        ),
                    )
                    q_logits = model.predict(
                        X=q,
                        feature_columns=self.feature_column,
                        transforms=self.model_transforms,
                        only_predictions=True,
                    )
                    count = (
                        self.format_pseudolabels(np.array(q_logits)) != q_pseudolabels
                    ).sum()
                    self.test_record["seed"].append(seed)
                    self.test_record["ensemble"].append(e)
                    self.test_record["count"].append(count)
                    self.test_record["rejection_rate"].append(
                        count / q_pseudolabels.size,
                    )

                elif isinstance(X_t, np.ndarray):
                    raise NotImplementedError("Numpy arrays are not supported yet.")
                elif isinstance(X_t, TorchDataset):
                    raise NotImplementedError("PyTorch datasets are not supported yet.")
        return self.get_results()

    def format_pseudolabels(self, labels):
        """Format pseudolabels."""
        if self.task in ("binary", "multilabel"):
            labels = (
                (labels > 0.5).astype("float32")
                if ((labels <= 1).all() and (labels >= 0).all())
                else (sigmoid(labels) > 0.5).astype("float32")
            )
        elif self.task == "multiclass":
            labels = (
                labels.argmax(dim=-1)
                if np.isclose(labels.sum(axis=-1), 1).all()
                else softmax(labels, axis=-1).argmax(axis=-1)
            )
        else:
            raise ValueError(
                f"Task must be either 'binary', 'multiclass' or 'multilabel', got {self.task} instead.",
            )
        return labels

    def get_record(self, record_type):
        """Get record."""
        if record_type == "calibration":
            record = self.cal_record
        elif record_type == "test":
            record = self.test_record
        if not isinstance(record, pd.DataFrame):
            record = pd.DataFrame(record)
        return record

    def counts(self, record_type, max_ensemble_size=None) -> np.ndarray:
        """Get counts."""
        assert (
            max_ensemble_size is None or max_ensemble_size > 0
        ), "max_ensemble_size must be positive or None"
        rec = self.get_record(record_type)
        counts = []
        for i in rec.seed.unique():
            run = rec.query(f"seed=={i}")
            if max_ensemble_size is not None:
                run = run.iloc[: max_ensemble_size + 1]
            counts.append(run.iloc[-1]["count"])
        return np.array(counts)

    @staticmethod
    def ecdf(x):
        """
        Compute the empirical cumulative distribution function.

        :param x: array of 1-D numerical data
        :return: a function that takes a value and returns the probability that
            a random sample from x is less than or equal to that value.
        """
        x = np.sort(x)

        def result(v):
            """Get the probability that a random sample from x is <= v."""
            return np.searchsorted(x, v, side="right") / x.size

        return result

    def get_results(self, max_ensemble_size=None) -> float:
        """Get p-value and distance along with calibration and test records."""
        cal_counts = self.counts("calibration", max_ensemble_size)
        test_count = self.counts("test", max_ensemble_size)[0]
        cdf = self.ecdf(cal_counts)
        p_value = cdf(test_count)
        return {
            "data": {
                "p_val": p_value,
                "distance": test_count,
                "cal_record": self.cal_record,
                "test_record": self.test_record,
            },
        }

    @staticmethod
    def split_dataset(X: Union[Dataset, DatasetDict]) -> DatasetDict:
        """Split dataset into train and test splits."""
        if isinstance(X, Dataset):
            X = X.train_test_split(test_size=0.5, shuffle=True)
        return X
