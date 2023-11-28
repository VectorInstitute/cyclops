"""PyTorch model wrapper."""

import contextlib
import logging
import os
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Dict,
    List,
    Literal,
    Optional,
    Sequence,
    Union,
)

import numpy as np
import torch
from datasets import Dataset, DatasetDict
from datasets.combine import concatenate_datasets
from torch import nn
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler as TorchLRScheduler
from torch.utils.data import DataLoader
from torch.utils.data import Dataset as TorchDataset

from cyclops.models.data import PTDataset
from cyclops.models.torch_utils import (
    DefaultCriterion,
    LossMeter,
    get_device,
    get_module,
)
from cyclops.models.utils import (
    get_split,
    is_pytorch_instance,
    is_pytorch_model,
)
from cyclops.models.wrappers.base import ModelWrapper
from cyclops.models.wrappers.utils import (
    DatasetColumn,
    check_is_fitted,
    set_random_seed,
    to_numpy,
    to_tensor,
)
from cyclops.utils.file import join, process_dir_save_path
from cyclops.utils.log import setup_logging
from cyclops.utils.optional import import_optional_module


if TYPE_CHECKING:
    from monai.data.meta_tensor import MetaTensor
else:
    MetaTensor = import_optional_module(
        "monai.data.meta_tensor",
        attribute="MetaTensor",
        error="warn",
    )


LOGGER = logging.getLogger(__name__)
setup_logging(print_level="INFO", logger=LOGGER)


# ruff: noqa: PLR0912


class PTModel(ModelWrapper):
    """PyTorch model wrapper.

    Parameters
    ----------
    model : nn.Module
        A PyTorch model instance or class to wrap. If class, provide parameters
        as kwargs in the format `model__<param_name>=<param_value>`.
    criterion : str or torch.nn.Module, default=DefaultCriterion
        The loss function to use. Accepts a string representing the name of a
        PyTorch loss function as defined in `torch.nn.modules.loss` or a
        `torch.nn.Module` instance/class. If class, provide parameters as kwargs
        in the format `criterion__<param_name>=<param_value>`.
        default criterion is a placeholder that takes the mean value of logits.
    optimizer : str or torch.optim.Optimizer, default=torch.optim.SGD
        The optimizer to use. Accepts a string representing the name of a
        PyTorch optimizer as defined in `torch.optim` or a `torch.optim.Optimizer`
        instance/class. If class, provide parameters as kwargs in the format
        `optimizer__<param_name>=<param_value>`.
    lr : float, default=0.01
        The learning rate to use.
    lr_scheduler : str or torch.optim.lr_scheduler._LRScheduler, default="ConstantLR"
        The learning rate scheduler to use. Accepts a string representing the name
        of a PyTorch learning rate scheduler as defined in `torch.optim.lr_scheduler`
        or a `torch.optim.lr_scheduler._LRScheduler` instance/class. If class,
        provide parameters as kwargs in the format
        `lr_scheduler__<param_name>=<param_value>`.
    lr_update_per_batch : bool, default=False
        Whether to update the learning rate after each batch or after each epoch.
        Set to `True` if using a learning rate scheduler that updates per batch
        like `torch.optim.lr_scheduler.StepLR`.
    batch_size : int, default=32
        The batch size to use. Set to -1 for full batch.
    max_epochs : int, default=10
        The maximum number of epochs to train for.
    activation : str or torch.nn.Module, default=nn.Identity
        The activation function to use. Accepts a string representing the name of
        a PyTorch activation function as defined in `torch.nn.modules.activation`
        or a `torch.nn.Module` instance/class. If class, provide parameters as
        kwargs in the format `activation__<param_name>=<param_value>`.
    train_loader : torch.utils.data.DataLoader, default=torch.utils.data.DataLoader
        An iterator for loading the training data in batches. The class assumes
        it is a `torch.utils.data.DataLoader`. Arguments can be passed as kwargs
        in the format `train_loader__<param_name>=<param_value>`.
    test_loader : torch.utils.data.DataLoader, default=torch.utils.data.DataLoader
        An iterator for loading the test/validation data in batches. The class
        assumes it is a `torch.utils.data.DataLoader`. Arguments can be passed
        as kwargs in the format `test_loader__<param_name>=<param_value>`.
    num_workers : int, default=os.cpu_count()
        The number of workers to use for loading the train/test data.
    warm_start : bool, default=False
        Whether to re-use the weights from the previous fit call. If `True`, the
        model will continue training from the weights of the previous fit call.
    save_every : int, default=-1
        The number of epochs to train before saving the model. If it is a negative
        only the latest model will be saved.
    save_best_only : bool, default=True
        Whether to save only the best model based on the validation loss.
    device : str or torch.device, default="cpu"
        The device to use for training.
    seed : int, default=None
        The random seed to use for reproducibility. If `None`, runs will be
        stochastic.
    deterministic : bool, default=False
        Whether to use deterministic algorithms for training. This will make
        runs reproducible but will be slower. It is ignored if `seed` is `None`.
    concatenate_features : bool, default=True
        Whether to concatenate the features in the dataset before passing them.
        This is useful when the input is a Hugging Face Dataset and the features
        are stored in different columns.

    """

    def __init__(
        self,
        model: nn.Module,
        criterion: Union[str, nn.Module] = DefaultCriterion,
        optimizer: Union[str, Optimizer] = torch.optim.SGD,
        lr: float = 0.01,
        lr_scheduler: Optional[Union[str, TorchLRScheduler]] = "ConstantLR",
        lr_update_per_batch: bool = False,
        batch_size: int = 32,
        max_epochs: int = 10,
        activation: Optional[Union[str, nn.Module]] = nn.Identity,
        train_loader=DataLoader,
        test_loader=DataLoader,
        num_workers=1,
        warm_start: bool = False,
        save_every: int = -1,
        save_best_only: bool = True,
        save_dir: Optional[str] = None,
        device: Optional[Union[str, torch.device]] = None,
        seed: Optional[int] = None,
        deterministic: bool = False,
        concatenate_features: bool = True,
        **kwargs,
    ) -> None:
        assert is_pytorch_model(
            model,
        ), "`model` must be an instance or subclass of `torch.nn.Module`."

        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.lr = lr
        self.lr_scheduler = lr_scheduler
        self.lr_update_per_batch = lr_update_per_batch
        self.batch_size = batch_size
        self.max_epochs = max_epochs
        self.activation = activation
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.num_workers = num_workers
        self.warm_start = warm_start
        self.save_every = save_every
        self.save_best_only = save_best_only
        self.save_dir = save_dir
        if device is None:
            device = get_device()
        self.device = device
        self.seed = seed
        self.deterministic = deterministic
        self.concatenate_features = concatenate_features

        vars(self).update(kwargs)  # add any additional kwargs to the class

        self.initialized_ = False
        self.train_loss_ = LossMeter("train")
        self.val_loss_ = LossMeter("val")

    @property
    def model_name(self) -> str:
        """The model name.

        Returns
        -------
        str
            The model name.

        """
        return self.model_.__class__.__name__

    def collect_params_for(self, prefix: str) -> Dict:
        """Collect parameters for a given prefix.

        Parameters
        ----------
        prefix : str
            The prefix to collect parameters for.

        Returns
        -------
        Dict
            A dictionary of parameters for the given prefix.

        """
        if not prefix.endswith("__"):
            prefix += "__"

        return {
            k.replace(prefix, ""): v
            for k, v in vars(self).items()
            if k.startswith(prefix)
        }

    def get_initialized_instance(self, instance_or_class: Any, kwargs: Dict) -> Any:
        """Initialize instance or class.

        Parameters
        ----------
        instance_or_class : Any
            Instance or class to initialize with kwargs.
        kwargs : Dict
            Parameters for instance or class initialization.

        Returns
        -------
        Any
            Initialized instance.

        """
        if is_pytorch_instance(instance_or_class):
            if not kwargs:
                return instance_or_class
            return type(instance_or_class)(**kwargs)

        return instance_or_class(**kwargs)

    def _initialize_module(
        self,
        module_name: str,
        default: Optional[str] = None,
        **extra_kwargs,
    ):
        """Initialize a module.

        Parameters
        ----------
        module_name : str
            Name of the module to initialize.
        default : str, optional
            Default value to use if module is not defined.
        **extra_kwargs
            Additional keyword arguments to pass to module initialization.
            These are mainly for known required parameters that may be missing
            from the class attributes.

        Returns
        -------
        self

        """
        module_or_name = getattr(self, module_name, default)
        if isinstance(module_or_name, str):
            setattr(self, module_name, get_module(module_name, module_or_name))

        kwargs = self.collect_params_for(prefix=f"{module_name}__")
        kwargs.update(extra_kwargs)

        setattr(
            self,
            f"{module_name}_",
            self.get_initialized_instance(getattr(self, module_name, default), kwargs),
        )

        return self

    def _load_module_to_device(self, name: str):
        """Load initialized torch.nn.Module to device.

        Parameters
        ----------
        name : str
            Name of the module to load.

        Returns
        -------
        self

        """
        if not name.endswith("_"):
            name += "_"

        module = getattr(self, name)
        if is_pytorch_instance(module):
            setattr(self, name, module.to(self.device))

        return self

    def initialize_model(self):
        """Initialize the model.

        Returns
        -------
        self

        """
        self._initialize_module("model")
        self._load_module_to_device(name="model")

        return self

    def initialize_criterion(self):
        """Initialize the criterion.

        Returns
        -------
        self

        """
        self._initialize_module("criterion")
        self._load_module_to_device(name="criterion")

        return self

    def get_all_learnable_params(self) -> List[nn.Parameter]:
        """Get all learnable parameters.

        Returns
        -------
        List[nn.Parameter]
            all learnable parameters

        """
        model = self.model_
        criterion = self.criterion_

        if model is None or criterion is None:
            raise ValueError(
                "Model and criterion must be initialized before getting"
                " learnable parameters.",
            )
        model_parameters = model.named_parameters()
        criterion_parameters = criterion.named_parameters()

        parameters = dict(model_parameters)
        for param_name, param in criterion_parameters:
            if param_name not in parameters:
                parameters[param_name] = param

        return [param for param in parameters.values() if param.requires_grad]

    def initialize_optimizer(self):
        """Initialize the optimizer.

        Returns
        -------
        self

        """
        params = self.get_all_learnable_params()
        return self._initialize_module(
            module_name="optimizer",
            default="SGD",
            params=params,
            lr=self.lr,
        )

    def initialize_activation(self):
        """Initialize the activation function.

        Returns
        -------
        nn.modules.activation
            The activation function

        Raises
        ------
        ValueError
            Invalid activation name

        """
        return self._initialize_module("activation", default="Identity")

    def initialize_lr_scheduler(self) -> TorchLRScheduler:
        """Initialize the lr scheduler.

        Returns
        -------
        torch.optim.lr_scheduler._LRScheduler
            The scheduler object

        Raises
        ------
        ValueError
            Invalid scheduler name

        """
        return self._initialize_module(
            "lr_scheduler",
            default="ConstantLR",
            optimizer=self.optimizer_,  # type: ignore[attr-defined]
        )

    def initialize(self):
        """Initialize the components of the model.

        Returns
        -------
        self

        """
        set_random_seed(self.seed, self.deterministic)
        self.initialize_model()
        self.initialize_criterion()
        self.initialize_activation()
        self.initialize_optimizer()
        self.initialize_lr_scheduler()

        self.initialized_ = True

        return self

    def _set_mode(self, training: bool = True) -> None:
        """Set mode for the model.

        Parameters
        ----------
        training : bool, default=True
            Whether to set the model to training mode or evaluation mode.

        """
        if not self.initialized_:
            self.initialize()

        self.model_.train(training)  # type: ignore[attr-defined]

    def _forward_pass(self, batch, **fit_params):
        """Run the forward pass.

        Parameters
        ----------
        batch
            Batch of data.
        **fit_params : dict, optional
            Additional parameters to pass to the model's `forward` method.

        Returns
        -------
        Any
            The model's output.

        """
        X = to_tensor(batch, device=self.device)
        if self.concatenate_features:
            out = self.model_(X, **fit_params)  # type: ignore[attr-defined]
        else:
            out = self.model_(**X, **fit_params)  # type: ignore[attr-defined]
        return out

    def _get_loss(self, target: torch.Tensor, preds: torch.Tensor) -> torch.Tensor:
        """Apply criterion and get the loss value.

        Parameters
        ----------
        target : torch.Tensor
            Target tensor.
        preds : torch.Tensor
            Predictions tensor.

        Returns
        -------
        loss : torch.Tensor
            Loss tensor.

        """
        return self.criterion_(  # type: ignore[attr-defined]
            preds.squeeze(),
            target.squeeze(),
        )

    def _train_step(self, batch, **fit_params) -> Dict[str, torch.Tensor]:
        """Train the model for one step.

        Parameters
        ----------
        batch
            Batch of data.
        **fit_params : dict, optional
            Additional parameters to pass to the model's `forward` method.

        Returns
        -------
        Dict[str, torch.Tensor]
            A dictionary containing the loss tensor.

        """
        self._set_mode(training=True)
        self.optimizer_.zero_grad(set_to_none=True)  # type: ignore[attr-defined]

        if isinstance(batch, (tuple, list)):
            X, target = batch
            target = to_tensor(target, device=self.device)
        else:
            X = batch
            target = None

        preds = self._forward_pass(X, **fit_params)
        if target is not None:
            loss = self._get_loss(target, preds)
        else:
            # XXX: batch may not always be a tuple of two elements
            raise NotImplementedError

        loss.backward()
        self.optimizer_.step()  # type: ignore[attr-defined]

        if self.lr_update_per_batch:
            self.lr_scheduler_.step()  # type: ignore[attr-defined]

        return {"loss": loss}

    def _validation_step(self, batch, **fit_params) -> Dict[str, torch.Tensor]:
        """Perform a validation step.

        Parameters
        ----------
        batch : tuple
            The batch of data.
        training : bool, default=False
            Whether to set the model to training mode.
        **predict_params : dict, optional
            Additional parameters to pass to the model's `forward` method.

        Returns
        -------
        Dict[str, torch.Tensor]
            A dictionary containing the loss tensor and the predictions tensor.

        """
        self._set_mode(training=False)

        if isinstance(batch, (tuple, list)):
            X, y = batch
            y = to_tensor(y, device=self.device)
        else:
            X = batch
            y = None

        with torch.no_grad():
            preds = self._forward_pass(X, **fit_params)
            if y is not None:
                loss = self._get_loss(y, preds)
            else:
                # XXX: `batch` may not always be a tuple
                raise NotImplementedError

        return {"loss": loss, "preds": preds}

    def _run_one_epoch(
        self,
        data_loader,
        step_fn: Callable,
        training: bool,
        feature_columns: Optional[Union[str, List[str]]] = None,
        target_columns: Optional[Union[str, List[str]]] = None,
        **fit_params,
    ):
        """Run one epoch of training or validation.

        Parameters
        ----------
        data_loader
            Data loader for the data.
        step_fn : Callable
            Function to call for each step.
        training : bool
            Whether the run is for training or not.
        feature_columns : Optional[Union[str, List[str]]], optional
            List of feature columns in the dataset. This is required when the input is \
                a Hugging Face Dataset, by default None
        target_columns : Optional[Union[str, List[str]]], optional
            List of target columns in the dataset. This is required when \
                the input is a Hugging Face Dataset, by default None
        **fit_params : dict, optional
            Additional parameters to pass to the model's `forward` method.

        Returns
        -------
        self : `PTModel`

        Raises
        ------
        AssertionError
            If the loss is NaN at any point during training.

        """
        batch_losses = []
        # if data is a HF dataset
        if feature_columns is not None:
            if target_columns is not None:
                for batch in data_loader:
                    if self.concatenate_features:
                        batch_features = torch.cat(
                            [batch[feature] for feature in feature_columns],
                            dim=1,
                        )
                    else:
                        batch_features = {k: batch[k] for k in feature_columns}
                    try:
                        batch_labels = torch.cat(
                            [batch[target] for target in target_columns],
                            dim=1,
                        )
                    except IndexError:
                        batch_labels = torch.cat(
                            [batch[target].unsqueeze(1) for target in target_columns],
                            dim=1,
                        )
                    batch = (batch_features, batch_labels)  # noqa: PLW2901
                    output = step_fn(batch, **fit_params)
                    loss = output["loss"].item()
                    assert not np.isnan(loss).any(), "Loss is NaN. Aborting training."
                    batch_losses.append(loss)
            else:
                raise NotImplementedError
        else:
            for batch in data_loader:
                output = step_fn(batch, **fit_params)
                loss = output["loss"].item()
                assert not np.isnan(loss).any(), "Loss is NaN. Aborting training."
                batch_losses.append(loss)

        if training:
            self.train_loss_.add(np.mean(batch_losses))
        else:
            self.val_loss_.add(np.mean(batch_losses))

        if training and not self.lr_update_per_batch:
            self.lr_scheduler_.step()  # type: ignore[attr-defined]

    def _get_dataset(
        self,
        X: Union[Dataset, DatasetDict, TorchDataset, np.ndarray, torch.Tensor],
        y: Optional[Union[np.ndarray, torch.Tensor]] = None,
    ) -> Union[Dataset, DatasetDict, TorchDataset]:
        """Get dataset.

        Parameters
        ----------
        X : Union[Dataset, DatasetDict, TorchDataset, np.ndarray, torch.Tensor]
            The features of the data.
        y : np.ndarray, optional
            The labels of the data.

        Returns
        -------
        TorchDataset
            The dataset object.

        """
        if isinstance(X, (Dataset, TorchDataset, DatasetDict)):
            return X

        if MetaTensor is not None and isinstance(X, MetaTensor):
            return PTDataset(X.data, y)

        if isinstance(X, (np.ndarray, torch.Tensor)):
            return PTDataset(X, y)

        raise ValueError(
            "`X` must be a numpy array or a `torch.utils.data.Dataset` instance."
            f" Got {type(X)} instead.",
        )

    def _get_dataloader(
        self,
        dataset: Union[Dataset, TorchDataset],
        test: bool = False,
    ):
        """Get PyTorch DataLoader for the data.

        Parameters
        ----------
        dataset : TorchDataset
            Data to load.
        test : bool, default=False
            Whether to load the data for testing or not.

        Returns
        -------
        A dataloader for the data.

        """
        assert isinstance(dataset, (TorchDataset, Dataset)), (
            "`dataset` must be a `torch.utils.data.Dataset` or"
            f"`datasets.Dataset` instance. Got {type(dataset)} instead."
        )

        if test:
            kwargs = self.collect_params_for(prefix="test_loader")
            data_loader = self.test_loader
        else:
            kwargs = self.collect_params_for(prefix="train_loader")
            data_loader = self.train_loader

        if "batch_size" not in kwargs:
            kwargs["batch_size"] = self.batch_size

        if kwargs["batch_size"] == -1:
            kwargs["batch_size"] = len(dataset)

        return data_loader(dataset, num_workers=self.num_workers, **kwargs)

    def _train_loop(
        self,
        X: Union[Dataset, DatasetDict, np.ndarray, TorchDataset],
        y: Optional[np.ndarray] = None,
        feature_columns: Optional[Union[str, List[str]]] = None,
        target_columns: Optional[Union[str, List[str]]] = None,
        splits_mapping: Optional[dict] = None,
        **fit_params,
    ):
        """Run the training loop.

        Parameters
        ----------
        X : Union[Dataset, DatasetDict, np.ndarray, TorchDataset],
            The features of the data.
        y : np.ndarray, optional
            The labels of the data.
        feature_columns : Optional[Union[str, List[str]]], optional
            List of feature columns in the dataset. This is required when the input is \
                a Hugging Face Dataset, by default None
        target_columns : Optional[Union[str, List[str]]], optional
            List of target columns in the dataset. This is required when \
                the input is a Hugging Face Dataset, by default None
        splits_mapping: Optional[dict], optional
            Mapping from 'train', 'validation' and 'test' to dataset splits names, \
                used when input is a dataset dictionary, by default None
        **fit_params : dict, optional
            Additional parameters to pass to the model's `forward` method.

        Returns
        -------
        self : `PTModel`

        """
        dataset = self._get_dataset(X, y)

        do_validation = isinstance(dataset, DatasetDict)

        if do_validation:
            train_dataset = dataset[splits_mapping["train"]]
            val_dataset = dataset[splits_mapping["validation"]]
        else:
            train_dataset = dataset

        # get the data loaders
        train_loader = self._get_dataloader(train_dataset)
        if do_validation:
            val_loader = self._get_dataloader(val_dataset, test=True)

        save_dir = self.save_dir if self.save_dir else os.getcwd()
        model_dir = join(save_dir, "saved_models", self.model_name)

        best_loss = np.inf
        for epoch in range(1, self.max_epochs + 1):
            self._run_one_epoch(
                data_loader=train_loader,
                step_fn=self._train_step,
                training=True,
                feature_columns=feature_columns,
                target_columns=target_columns,
                **fit_params,
            )

            LOGGER.info(
                "[%d/%d] \
                Training loss: %0.4f \t",
                epoch,
                self.max_epochs,
                self.train_loss_.pop(),
            )

            if do_validation:
                self._run_one_epoch(
                    data_loader=val_loader,
                    step_fn=self._validation_step,
                    training=False,
                    feature_columns=feature_columns,
                    target_columns=target_columns,
                    **fit_params,
                )

                val_loss = self.val_loss_.pop()
                LOGGER.info(
                    "[%d/%d] \
                    Validation loss: %0.4f \t",
                    epoch,
                    self.max_epochs,
                    val_loss,
                )

                if val_loss < best_loss:
                    LOGGER.info("Best model saved at epoch %d in %s", epoch, model_dir)
                    self.save_model(filepath=model_dir, epoch=epoch, is_best=True)

            if (
                self.save_every < 0 or epoch % self.save_every == 0
            ) and not self.save_best_only:
                self.save_model(filepath=model_dir, epoch=epoch)

        return self

    def partial_fit(
        self,
        X: Union[Dataset, DatasetDict, np.ndarray, TorchDataset],
        y: Optional[np.ndarray] = None,
        feature_columns: Optional[Union[str, List[str]]] = None,
        target_columns: Optional[Union[str, List[str]]] = None,
        splits_mapping: Optional[dict] = None,
        **fit_params,
    ):
        """Fit the model to the data.

        Parameters
        ----------
        X : Union[Dataset, DatasetDict, np.ndarray, TorchDataset],
            The features of the data.
        y : np.ndarray, optional
            The labels of the data.
        feature_columns : Optional[Union[str, List[str]]], optional
            List of feature columns in the dataset. This is required when the input is \
                a Hugging Face Dataset, by default None
        target_columns : Optional[Union[str, List[str]]], optional
            List of target columns in the dataset. This is required when \
                the input is a Hugging Face Dataset, by default None
        splits_mapping: Optional[dict], optional
            Mapping from 'train', 'validation' and 'test' to dataset splits names, \
                used when input is a dataset dictionary, by default None
        **fit_params : dict, optional
            Additional parameters to pass to the model's `forward` method.

        Returns
        -------
        self : `PTModel`

        """
        if not self.initialized_:
            self.initialize()

        with contextlib.suppress(KeyboardInterrupt):
            self._train_loop(
                X,
                y=y,
                feature_columns=feature_columns,
                target_columns=target_columns,
                splits_mapping=splits_mapping,
                **fit_params,
            )

        return self

    def fit(
        self,
        X: Union[Dataset, DatasetDict, np.ndarray, TorchDataset],
        y: Optional[np.ndarray] = None,
        feature_columns: Optional[Union[str, List[str]]] = None,
        target_columns: Optional[Union[str, List[str]]] = None,
        transforms: Optional[Callable] = None,
        splits_mapping: dict = None,
        **fit_params,
    ):
        """Fit the model.

        Parameters
        ----------
        X : Union[Dataset, np.ndarray, TorchDataset]
            The data features or a Hugging Face Dataset containing features and labels.
        y : Optional[ArrayLike], optional
            The labels of the data. This is required when the input data is not \
                a Hugging Face Dataset and only contains features, by default None
        feature_columns : Optional[Union[str, List[str]]], optional
            List of feature columns in the dataset. This is required when the input is \
                a Hugging Face Dataset, by default None
        target_columns : Optional[Union[str, List[str]]], optional
            List of target columns in the dataset. This is required when the input is \
                a Hugging Face Dataset, by default None
        transforms : Optional[Callable], optional
            Transform function to be applied when __getitem__ is called \
            when the input is a Hugging Face Dataset, by default None
        splits_mapping: Optional[dict], optional
            Mapping from 'train', 'validation' and 'test' to dataset splits names, \
                used when input is a dataset dictionary,
                by default {"train": "train", "validation": "validation"}

        Returns
        -------
        self : `PTModel`

        Raises
        ------
        ValueError
            If `X` is a Hugging Face Dataset and the feature column(s) is not provided.

        """
        if splits_mapping is None:
            splits_mapping = {"train": "train", "validation": "validation"}
        if not self.warm_start or not self.initialized_:
            self.initialize()

        if isinstance(X, (Dataset, DatasetDict)):
            if feature_columns is None:
                raise ValueError(
                    "Missing feature columns 'feature_columns'. Please provide \
                    the name of feature columns when using a \
                    Hugging Face dataset as the input.",
                )
            if isinstance(feature_columns, str):
                feature_columns = [feature_columns]

            if target_columns is None:
                LOGGER.warning(
                    "Missing target columns 'target_columns'. Please provide \
                    the name of target columns when using a \
                    Hugging Face dataset for supervised training.",
                )
            if isinstance(target_columns, str):
                target_columns = [target_columns]

            if isinstance(X, DatasetDict):
                train_split = get_split(X, "train", splits_mapping)
                try:
                    val_split = get_split(X, "validation", splits_mapping)
                except ValueError:
                    LOGGER.info("No validation split was found.")
                    val_split = None

                if val_split is None:
                    return self.fit(
                        X[train_split],
                        feature_columns=feature_columns,
                        target_columns=target_columns,
                        transforms=transforms,
                    )

                splits_mapping["train"] = train_split
                splits_mapping["validation"] = val_split

                format_kwargs = {} if transforms is None else {"transform": transforms}
                with X[train_split].formatted_as(
                    "custom" if transforms is not None else "torch",
                    columns=feature_columns + target_columns,
                    **format_kwargs,
                ), X[val_split].formatted_as(
                    "custom" if transforms is not None else "torch",
                    columns=feature_columns + target_columns,
                    **format_kwargs,
                ):
                    self.partial_fit(
                        X,
                        feature_columns=feature_columns,
                        target_columns=target_columns,
                        splits_mapping=splits_mapping,
                        **fit_params,
                    )
            else:
                format_kwargs = {} if transforms is None else {"transform": transforms}
                with X.formatted_as(
                    "custom" if transforms is not None else "torch",
                    columns=feature_columns + target_columns,
                    **format_kwargs,
                ):
                    self.partial_fit(
                        X,
                        feature_columns=feature_columns,
                        target_columns=target_columns,
                        **fit_params,
                    )
        else:
            if y is None:
                LOGGER.warning(
                    "Missing data labels 'y'. Please provide the labels \
                    for supervised training when not using a \
                    Hugging Face dataset as the input.",
                )
            self.partial_fit(X, y, **fit_params)

        return self

    def find_best(
        self,
        parameters: Union[Dict, List[Dict]],
        X: Union[Dataset, DatasetDict],
        y: Optional[np.ndarray] = None,
        feature_columns: Optional[Union[str, List[str]]] = None,
        target_columns: Optional[Union[str, List[str]]] = None,
        transforms: Optional[Callable] = None,
        metric: Optional[Union[str, Callable, Sequence, Dict]] = None,
        method: Literal["grid", "random"] = "grid",
        splits_mapping: dict = None,
        **kwargs,
    ):
        """Find the best model from hyperparameter search.

        Parameters
        ----------
        parameters : dict or list of dicts
            The hyperparameters to be tuned.
        X : Union[Dataset, DatasetDict]
            The data features or a Hugging Face dataset containing features and labels.
        y : Optional[np.ndarray], optional
            The labels of the data. This is required when the input dataset is not \
                a huggingface dataset and only contains features, by default None
        feature_columns : Optional[Union[str, List[str]]], optional
            List of feature columns in the dataset. This is required when the input is \
                a Hugging Face Dataset, by default None
        target_columns : Optional[Union[str, List[str]]], optional
            List of target columns in the dataset. This is required when the input is \
                a Hugging Face Dataset, by default None
        transforms : Optional[Union[Callable], optional
            The transformation to be applied to the data before prediction, \
                This is used when the input is a Hugging Face Dataset, \
                by default None
        metric : str, callable, sequence, dict, optional
            The metric to be used for model evaluation.
        method : Literal["grid", "random"], default="grid"
            The tuning method to be used.
        splits_mapping: Optional[dict], optional
            Mapping from 'train', 'validation' and 'test' to dataset splits names, \
                used when input is a dataset dictionary,
                by default {"train": "train", "validation": "validation"}
        **kwargs : dict, optional
            Additional keyword arguments to be passed to the search method.

        Returns
        -------
        self : `PTModel`

        """
        if splits_mapping is None:
            splits_mapping = {"train": "train", "validation": "validation"}
        raise NotImplementedError

    def _evaluation_step(self, batch, training: bool = False, **fit_params):
        """Run the forward pass for evaluation.

        Parameters
        ----------
        batch : tuple
            The batch of data
        training : bool, default=False
            Whether to run the model in training mode.
        **predict_params : dict, optional
            Additional parameters to pass to the model's `forward` method.

        Returns
        -------
        output : torch.Tensor
            The output of the model.

        """
        if isinstance(batch, (tuple, list)):
            X, _ = batch
        else:
            X = batch

        with torch.set_grad_enabled(mode=training):
            self._set_mode(training=training)
            return self._forward_pass(X, **fit_params)

    def predict_proba(
        self,
        X: Union[Dataset, np.ndarray, TorchDataset],
        feature_columns: Optional[Union[str, List[str]]] = None,
        prediction_column=None,
        **predict_params,
    ):
        """Return the output probabilities of the model output for the given input.

        Parameters
        ----------
        X : Union[Dataset, np.ndarray, TorchDataset]
            The input to the model.
        feature_columns : Optional[Union[str, List[str]]], optional
            List of feature columns in the dataset. This is required when the input is \
                a Hugging Face Dataset, by default None
        prediction_column : Optional[Union[str, List[str]]],
            Name of the prediction column to be added to the dataset
        **predict_params : dict, optional
            Additional parameters for the prediction.

        Returns
        -------
            The probabilities of the output of the model.

        """
        check_is_fitted(
            estimator=self,
            attributes=["model_", "criterion_", "optimizer_", "lr_scheduler_"],
            all_or_any=all,
        )

        dataset = self._get_dataset(X)
        dataloader = self._get_dataloader(dataset, test=True)

        if isinstance(X, Dataset):
            preds = Dataset.from_dict({prediction_column: []})
            for batch in dataloader:
                if self.concatenate_features:
                    batch = torch.cat(  # noqa: PLW2901
                        [batch[feature] for feature in feature_columns],
                        dim=1,
                    )
                else:
                    batch = {k: batch[k] for k in feature_columns}  # noqa: PLW2901
                output = self._evaluation_step(batch, training=False, **predict_params)
                output = self.activation_(output)
                batch_ds = Dataset.from_dict({prediction_column: output})
                preds = concatenate_datasets([preds, batch_ds], axis=0)
        else:
            preds = []
            for batch in dataloader:
                output = self._evaluation_step(batch, training=False, **predict_params)
                output = self.activation_(output)
                preds.append(to_numpy(output))
            preds = np.concatenate(preds)

        return preds

    def predict(
        self,
        X: Union[Dataset, DatasetDict, np.ndarray, TorchDataset],
        feature_columns: Optional[Union[str, List[str]]] = None,
        prediction_column_prefix: str = "predictions",
        model_name: Optional[str] = None,
        transforms: Optional[Callable] = None,
        only_predictions: bool = False,
        splits_mapping: dict = None,
        **predict_params,
    ) -> Union[Dataset, DatasetColumn, np.ndarray]:
        """Predict the output of the model.

        Parameters
        ----------
        X : Dataset
            The data features or a Hugging Face Dataset containing features and labels.
        feature_columns : Optional[Union[str, List[str]]], optional
            List of feature columns in the dataset. This is required when the input is \
                a Hugging Face Dataset, by default None
        prediction_column_prefix : str, optional
            Name of the prediction column to be added to the dataset, This is used \
                when the input is a Hugging Face Dataset, by default "predictions"
        model_name : Optional[str], optional
            Model name used as suffix to the prediction column, This is used \
                when the input is a Hugging Face Dataset, by default None
        transforms : Optional[Callable], optional
            Transform function to be applied when __getitem__ is called,
                This is used when the input is a Hugging Face Dataset, \
                by default None
        only_predictions : bool, optional
            Whether to return only the predictions rather than the dataset \
                with predictions when the input is a Hugging Face Datset, \
                by default False
        splits_mapping: Optional[dict], optional
            Mapping from 'train', 'validation' and 'test' to dataset splits names, \
                used when input is a dataset dictionary, by default {"test": "test"}

        Returns
        -------
        Union[Dataset, DatasetColumn, np.ndarray]
            Dataset containing the predictions or the predictions array.

        Raises
        ------
        ValueError
            If `X` is a Hugging Face Dataset and the feature column(s) is not provided.

        """
        # Input is a Hugging Face Dataset Dictionary
        if splits_mapping is None:
            splits_mapping = {"test": "test"}
        if isinstance(X, DatasetDict):
            test_split = get_split(X, "test", splits_mapping=splits_mapping)
            return self.predict(
                X[test_split],
                feature_columns=feature_columns,
                prediction_column_prefix=prediction_column_prefix,
                model_name=model_name,
                transforms=transforms,
                only_predictions=only_predictions,
            )
        # Input is a Hugging Face Dataset
        if isinstance(X, Dataset):
            if feature_columns is None:
                raise ValueError(
                    "Missing feature columns 'feature_columns'. Please provide \
                    the name of feature columns when using \
                    a Hugging Face dataset as the input.",
                )
            if isinstance(feature_columns, str):
                feature_columns = [feature_columns]

            if model_name:
                pred_column = f"{prediction_column_prefix}.{model_name}"
            else:
                pred_column = (
                    f"{prediction_column_prefix}.{self.model_.__class__.__name__}"
                )

            format_kwargs = {} if transforms is None else {"transform": transforms}
            with X.formatted_as(
                "custom" if transforms is not None else "torch",
                columns=feature_columns,
                **format_kwargs,
            ):
                preds_ds = self.predict_proba(
                    X,
                    feature_columns=feature_columns,
                    prediction_column=pred_column,
                    **predict_params,
                )

                if only_predictions:
                    return DatasetColumn(preds_ds.with_format("numpy"), pred_column)
                return concatenate_datasets([X, preds_ds], axis=1)

        # Input is not a Hugging Face Dataset
        return self.predict_proba(X, **predict_params)

    def save_model(self, filepath: str, overwrite: bool = True, **kwargs):
        """Save the model to a file.

        Parameters
        ----------
        filepath : str
            The path to save the model. If the path is a directory, the model
            is saved in the directory with the filename `model.pt`. If the
            epoch is specified, the model is saved as `checkpoint-<epoch>.pt`.
        overwrite : bool, default=True
            Whether to overwrite the existing model.
        **kwargs : dict, optional
            Additional keyword arguments for saving the model.
            The following arguments are used:
            * include_optimizer: bool, default=True
                Whether to include the optimizer in the saved model.
            * include_lr_scheduler: bool, default=True
                Whether to include the learning rate scheduler in the saved model.
            * epoch: int, default=None
                The epoch to save the model at. This is used for checkpointing.
                The epoch number is appended to the filename.
            * is_best: bool, default=False
                Whether the model is the best model so far. The best model is
                saved as `best_model.pt`. A symbolic link is created between
                the best model and the checkpoint model.

        """
        use_default_filepath = False  # whether the filepath is the default

        if len(os.path.basename(filepath).split(".")) == 1:
            process_dir_save_path(filepath)

        if os.path.isdir(filepath):
            filepath = join(filepath, "model.pt")
            use_default_filepath = True

        dir_path = os.path.dirname(filepath)
        if dir_path == "":
            dir_path = "."
            filepath = join(dir_path, filepath)
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)

        latest_model_path = None
        # TODO: create versions for model directories?

        if os.path.exists(filepath) and not overwrite:
            LOGGER.warning("The file already exists and will not be overwritten.")
            return

        # prepare the state dictionary
        state_dict = {"model": self.model_.state_dict()}  # type: ignore[attr-defined]

        include_optimizer = kwargs.get("include_optimizer", True)
        if include_optimizer:
            state_dict[
                "optimizer"
            ] = self.optimizer_.state_dict()  # type: ignore[attr-defined]

        include_lr_scheduler = kwargs.get("include_lr_scheduler", True)
        if include_lr_scheduler:
            state_dict[
                "lr_scheduler"
            ] = self.lr_scheduler_.state_dict()  # type: ignore[attr-defined]

        epoch = kwargs.get("epoch", None)
        if epoch is not None:
            filename, extension = os.path.basename(filepath).split(".")
            filepath = join(
                dir_path,
                f"checkpoint_{epoch}.pt"
                if use_default_filepath
                else join(dir_path, f"{filename}_checkpoint{epoch}.{extension}"),
            )

            state_dict["epoch"] = epoch

            # prepare the latest model path
            # if a specific filename is provided, append 'latest' to it and preseve
            # the extension. Otherwise, use 'latest.pt' as the filename.
            latest_model_path = join(dir_path, "latest.pt")
            if not use_default_filepath:
                filename, extension = os.path.basename(filepath).split(".")
                latest_model_path = join(dir_path, f"{filename}_latest.{extension}")

        # save model
        if self.save_every >= 1:
            torch.save(state_dict, filepath)

            # create a symlink to the latest model
            if latest_model_path:
                if os.path.exists(latest_model_path):
                    os.remove(latest_model_path)
                os.symlink(filepath, latest_model_path)
        else:
            if latest_model_path is None:
                latest_model_path = filepath
            torch.save(state_dict, latest_model_path)
            filepath = latest_model_path  # for creating the symlink below

        # save best model
        best_model_path = join(dir_path, "best_model.pt")
        is_best = kwargs.get("is_best", False)
        if is_best:
            if os.path.exists(best_model_path):
                os.remove(best_model_path)
            torch.save(state_dict, best_model_path)

    def load_model(self, filepath: str, **kwargs):
        """Load a model from a checkpoint.

        Parameters
        ----------
        filepath : str
            The path to the checkpoint file.
        **kwargs : dict, optional
            Additional keyword arguments for loading the model.

        Returns
        -------
        self

        """
        state_dict = torch.load(filepath, map_location=self.device)
        if not isinstance(state_dict, dict):
            raise ValueError(
                "Expected the file to be a checkpoint file."
                " Probably, the file is a model file."
                " Please call `save_model` instead of `torch.save`.",
            )

        if not self.initialized_:
            self.initialize()

        to_load = ["model", "optimizer", "lr_scheduler"]
        for key in to_load:
            if key in state_dict:
                getattr(self, f"{key}_").load_state_dict(state_dict[key])

        return self

    def plot_losses(self) -> None:
        """Plot train and validation losses per epoch."""
        pass
