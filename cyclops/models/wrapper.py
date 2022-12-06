"""Model Wrappers for PyTorch and Scikit-learn."""

import logging
import math
from ctypes import Union
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.model_selection import GridSearchCV, PredefinedSplit
from torch import nn
from torch.optim import Optimizer, lr_scheduler
from torch.utils.data import DataLoader

from cyclops.models.catalog import MODELS, _PTModel, _SKModel
from cyclops.models.data import PTDataset
from cyclops.models.util import (
    ACTIVATIONS,
    CRITERIONS,
    OPTIMIZERS,
    SCHEDULERS,
    LossMeter,
    get_device,
)
from cyclops.utils.file import join, load_pickle, save_pickle
from cyclops.utils.log import setup_logging

LOGGER = logging.getLogger(__name__)
setup_logging(print_level="INFO", logger=LOGGER)

# mypy: ignore-errors
# pylint: disable=invalid-name, too-many-instance-attributes


class PTModel:
    """PyTorch model wrapper."""

    def __init__(self, model_name: str, save_path: str, **kwargs) -> None:
        """Initialize wrapper.

        Parameters
        ----------
        model_name : str
            pyTorch model name (rnn, gru, etc.)
        save_path : str
            path to save and/or load trained model

        """
        self.model_name = model_name
        self.save_path = save_path

        self.model_params: dict = kwargs.get("model_params")
        self.train_params: dict = kwargs.get("train_params")
        self.opt_params: dict = kwargs.get("opt_params")
        self.lr_params: dict = kwargs.get("lr_params")
        self.data_params: dict = kwargs.get("data_params")
        self.test_params: dict = kwargs.get("test_params")

        self.n_epochs: int = self.train_params.get("n_epochs")
        self.per_epoch: bool = self.train_params.get("per_epoch")
        self.reweight: Union[str, float] = self.train_params.get("reweight")
        self.batch_size: int = self.data_params.get("batch_size")

        self.device = get_device()

        self.train_loss = LossMeter("train")
        self.val_loss = LossMeter("val")

        self.model = self._init_model()
        self.criterion = self._init_criterion()
        self.activation = self._init_activation()
        self.optimizer = self._init_optimizer()
        self.lr_scheduler = self._init_lr_scheduler()

    def _init_model(self) -> _PTModel:
        """Initialize model.

        Returns
        -------
        _PTModel
            the PyTorch model

        Raises
        ------
        ValueError
            Invalid model name

        """
        model = self.model_name.lower()
        if model not in MODELS:
            raise ValueError("[!] Invalid Model name.")
        return MODELS[model](**self.model_params).to(self.device)

    def _init_criterion(self) -> nn.modules.loss:
        """Initialize the criterion.

        Returns
        -------
        nn.modules.loss
            the loss function

        Raises
        ------
        ValueError
            Invalid criterion name

        """
        criterion = self.train_params["criterion"].lower()
        if criterion not in CRITERIONS:
            raise ValueError("[!] Invalid criterion.")
        return CRITERIONS[criterion](reduction="none")

    def _init_activation(self) -> nn.modules.activation:
        """Initialize the activation function.

        Returns
        -------
        nn.modules.activation
            the activation function

        Raises
        ------
        ValueError
            Invalid activation name

        """
        activation = self.train_params["activation"].lower()
        if activation not in ACTIVATIONS:
            raise ValueError("[!] Invalid activation function.")
        return ACTIVATIONS[activation]

    def _init_optimizer(self) -> Optimizer:
        """Initialize the optimizer.

        Returns
        -------
        Optimizer
            the optimizer object

        Raises
        ------
        ValueError
            Invalid optimizer name

        """
        optimizer = self.train_params["optimizer"]
        if optimizer not in OPTIMIZERS:
            raise ValueError("[!] Invalid optimizer.")
        return OPTIMIZERS[optimizer](self.model.parameters(), **self.opt_params)

    def _init_lr_scheduler(self) -> lr_scheduler:
        """Initialize the lr scheduler.

        Returns
        -------
        lr_scheduler
            the scheduler object

        Raises
        ------
        ValueError
            Invalid scheduler name

        """
        scheduler = self.train_params["lr_scheduler"]
        if scheduler not in SCHEDULERS:
            raise ValueError("[!] Invalid scheduler.")
        return SCHEDULERS[scheduler](self.optimizer, **self.lr_params)

    def _get_loss(
        self,
        y: torch.Tensor,
        y_pred: torch.Tensor,
    ) -> torch.Tensor:
        """Apply criterion and get the loss value.

        Parameters
        ----------
        y : torch.Tensor
            data labels
        y_pred : torch.Tensor
            predicted labels

        Returns
        -------
        torch.Tensor
            loss tensor

        """
        loss = self.criterion(y_pred.squeeze(), y.squeeze())
        loss = self._reweight_loss(loss, y)
        loss *= ~y.eq(-1).squeeze()
        loss = loss.sum() / (~y.eq(-1)).sum()
        return loss

    def _reweight_loss(
        self,
        loss: torch.Tensor,
        y: torch.Tensor,
    ) -> torch.Tensor:
        """Reweight loss for unbalanced data.

        Parameters
        ----------
        loss : torch.Tensor
            loss tensor
        y : torch.Tensor
            data labels

        Returns
        -------
        torch.Tensor
            reweighted loss

        """
        if isinstance(self.reweight, (float, np.float64)):
            loss[y.squeeze() == 1] *= self.reweight
        elif self.reweight == "mini-batch":
            loss[y.squeeze() == 1] *= (y == 0).sum() / (y == 1).sum()
        else:
            pass
        return loss

    def _train_step(self, batch: torch.Tensor) -> float:
        """Run training step per mini-batch.

        Parameters
        ----------
        batch : torch.Tensor
            train data batch (features, labels)

        Returns
        -------
        float
            loss value

        """
        self.model.train()
        self.optimizer.zero_grad()
        x_train, y_train = batch
        x_train = x_train.to(self.device)
        y_train = y_train.to(self.device)
        y_pred = self.model(x_train)
        loss = self._get_loss(y_train, y_pred)
        loss.backward()
        self.optimizer.step()
        return loss.item()

    def _train_epoch(self, train_loader: DataLoader) -> None:
        """Train for one epoch.

        Parameters
        ----------
        train_loader : DataLoader
            train data loader

        """
        batch_losses = []
        for _, batch in enumerate(train_loader):
            loss = self._train_step(batch)
            assert not np.isnan(loss).any()
            batch_losses.append(loss)
        self.train_loss.add(np.mean(batch_losses))
        self.lr_scheduler.step()

    def _validation_step(self, batch: torch.Tensor) -> float:
        """Run validation step per mini-batch.

        Parameters
        ----------
        batch : torch.Tensor
            validation data batch (features, labels)

        Returns
        -------
        float
            loss value

        """
        self.model.eval()
        x_val, y_val = batch
        x_val = x_val.to(self.device)
        y_val = y_val.to(self.device)
        y_pred = self.model(x_val)
        val_loss = self._get_loss(y_val, y_pred)
        return val_loss.item()

    def _validate_epoch(self, val_loader: DataLoader) -> None:
        """Validate for one epoch.

        Parameters
        ----------
        val_loader : DataLoader
            validation data loader

        """
        batch_losses = []
        with torch.no_grad():
            for _, batch in enumerate(val_loader):
                val_loss = self._validation_step(batch)
                assert not np.isnan(val_loss).any()
                batch_losses.append(val_loss)
            self.val_loss.add(np.mean(batch_losses))

    def get_dataloader(
        self,
        X: np.ndarray,
        y: np.ndarray,
        test: bool = False,
    ) -> DataLoader:
        """Get PyTorch dataloader.

        Parameters
        ----------
        X : np.ndarray
            data features
        y : np.ndarray
            data labels
        test : bool, optional
            data is the test set

        Returns
        -------
        DataLoader
            _description_

        """
        dataset = PTDataset(X, y)
        if test:
            return DataLoader(dataset, batch_size=1, shuffle=False)
        return DataLoader(dataset, **self.data_params)

    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
    ) -> _PTModel:
        """Train the model.

        Parameters
        ----------
        X_train : np.ndarray
            train data features
        y_train : np.ndarray
            train data labels
        X_val : np.ndarray
            test data features
        y_val : np.ndarray
            test data labels

        Returns
        -------
        _PTModel
            trained pyTorch model

        """
        best_loss = math.inf
        best_model = None
        train_loader = self.get_dataloader(X_train, y_train)
        val_loader = self.get_dataloader(X_val, y_val)

        for epoch in range(1, self.n_epochs + 1):
            self._train_epoch(train_loader)
            self._validate_epoch(val_loader)

            if self.per_epoch:
                self.save_model(epoch)
            else:
                self._save_last(epoch)

            if self.val_loss.pop() < best_loss:
                LOGGER.info("Best model saved at epoch %d in %s", epoch, self.save_path)
                best_model = self.model
                self._save_best(epoch)

            LOGGER.info(
                "[%d/%d] \
                Training loss: %0.4f \t \
                Validation loss: %0.4f",
                epoch,
                self.n_epochs,
                self.train_loss.pop(),
                self.val_loss.pop(),
            )

        return best_model

    def save_model(self, epoch: int, model_path: Optional[str] = None) -> None:
        """Checkpoint model.

        Parameters
        ----------
        epoch : int
            current epoch number
        model_path : Optional[str], optional
            directory to save the model, by default None

        """
        if not model_path:
            model_path = join(self.save_path, f"{self.model_name}_{epoch}.pt")
        torch.save(
            {
                "epoch": epoch,
                "model": self.model.state_dict(),
                "optimizer": self.optimizer.state_dict(),
                "hyper_params": self.model_params,
            },
            model_path,
        )

    def _save_best(self, epoch: int) -> None:
        """Checkpoint the best model.

        Parameters
        ----------
        epoch : int
            current epoch number

        """
        model_path = join(self.save_path, f"{self.model_name}_best.pt")
        self.save_model(epoch, model_path)

    def _save_last(self, epoch: int) -> None:
        """Checkpoint the last model.

        Parameters
        ----------
        epoch : int
            current epoch number

        """
        model_path = join(self.save_path, f"{self.model_name}_last.pt")
        self.save_model(epoch, model_path)

    def load_model(self, model_path: Optional[str]) -> _PTModel:
        """Load a model from checkpoint.

        Parameters
        ----------
        model_path : Optional[str]
            path to the checkpoint file

        Returns
        -------
        _PTModel
            pyTorch model

        """
        if not model_path:
            model_path = join(self.save_path, f"{self.model_name}_best.pt")
        checkpoint = torch.load(model_path)
        model = self.model.__class__(**checkpoint["hyper_params"]).to(self.device)
        model.load_state_dict(checkpoint["model"])
        return model

    def load_ckp(self, ckp_path: str) -> tuple:
        """Load a checkpoint.

        Parameters
        ----------
        ckp_path : str
            ckp_path (str): path to the checkpoint file

        Returns
        -------
        tuple
            pytorch model, optimizer, and the last epoch

        """
        checkpoint = torch.load(ckp_path)
        model = self.model.__class__(**checkpoint["hyper_params"]).to(self.device)
        optimizer = self.optimizer
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        return model, optimizer, checkpoint["epoch"]

    def predict(
        self,
        model: _PTModel,
        X_test: np.ndarray,
        y_test: np.ndarray,
    ) -> tuple:
        """Make prediction by a trained model.

        Parameters
        ----------
        model : _PTModel
            pyTorch model object
        X_test : np.ndarray
            test data features
        y_test : np.ndarray
            test data labels

        Returns
        -------
        tuple
            data labels, predicted values (probabilities), predicted labels

        """
        y_pred_values = []
        y_test_labels = []
        y_pred_labels = []
        test_loader = self.get_dataloader(X_test, y_test, test=True)
        with torch.no_grad():
            for _, batch in enumerate(test_loader):
                x_test, y_test = batch
                x_test = x_test.to(self.device)
                y_test = y_test.to(self.device)
                model.eval()
                y_pred = self.activation(model(x_test))
                y_pred_values.append(y_pred.cpu().detach())
                y_test_labels.append(y_test.cpu().detach())
                y_pred_labels.append(torch.round(y_pred).cpu().detach())

        y_test_labels = np.concatenate(y_test_labels)
        y_pred_labels = np.concatenate(y_pred_labels)
        y_pred_values = np.concatenate(y_pred_values)

        if self.test_params["flatten"]:
            return (y.flatten() for y in [y_test_labels, y_pred_values, y_pred_labels])

        return y_test_labels, y_pred_values, y_pred_labels

    def plot_losses(self) -> None:
        """Plot train and validation losses per epoch."""
        plt.plot(self.train_loss.losses, label="Training loss")
        plt.plot(self.val_loss.losses, label="Validation loss")
        plt.legend()
        plt.title("Losses")
        plt.show()
        plt.close()


class SKModel:
    """Scikit-learn model wrapper."""

    def __init__(self, model_name: str, save_path, **kwargs) -> None:
        """Initialize wrapper.

        Parameters
        ----------
        model_name : str
            sklearn model name (xgb, lr, etc.)
        save_path : _type_
            path to save and/or load trained model

        """
        self.model_name = model_name
        self.save_path = save_path

        self.model_params: dict = kwargs.get("model_params")
        self.best_model: bool = kwargs.get("train_params")["best_model"]
        self.best_model_params: dict = kwargs.get("train_params")["best_model_params"]
        self.best_model_metric: dict = kwargs.get("train_params")["best_model_metric"]
        self.test_params: dict = kwargs.get("test_params")

    @property
    def model(self) -> _SKModel:
        """Get model as an attribute.

        Returns
        -------
        _SKModel
            scikit-learn model object

        Raises
        ------
        ValueError
            Invalid model name

        """
        model = self.model_name.lower()
        if model not in MODELS:
            raise ValueError("[!] Invalid Model name.")
        return MODELS[model]

    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
    ) -> _SKModel:
        """Train the model.

        Parameters
        ----------
        X_train : np.ndarray
            train data features
        y_train : np.ndarray
            train data labels
        X_val : np.ndarray
            test data features
        y_val : np.ndarray
            test data labels

        Returns
        -------
        _SKModel
            trained pyTorch model

        """
        y_train = y_train.ravel()
        y_val = y_val.ravel()

        if self.best_model:
            model = self.find_best(X_train, y_train, X_val, y_val)
        else:
            model = self.model(**self.model_params)
            model.fit(X_train, y_train)

        self.save_model(model)
        LOGGER.info("Model saved in %s", self.save_path)

        return model

    def find_best(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
    ) -> _SKModel:
        """Find the best model by grid search.

        Parameters
        ----------
        X_train : np.ndarray
            train data features
        y_train : np.ndarray
            trian data labels
        X_val : np.ndarray
            validation data features
        y_val : np.ndarray
            validation data labels

        Returns
        -------
        _SKModel
            best-performing sklearn model

        """
        model = self.model(**self.model_params)
        split_index = [-1] * len(X_train) + [0] * len(X_val)
        X = np.concatenate((X_train, X_val), axis=0)
        y = np.concatenate((y_train, y_val), axis=0)
        pds = PredefinedSplit(test_fold=split_index)
        clf = GridSearchCV(
            model,
            param_grid=self.best_model_params,
            scoring=self.best_model_metric,
            cv=pds,
            n_jobs=2,
            verbose=2,
        )
        clf.fit(X, y)
        for k, v in clf.best_params_.items():
            LOGGER.info("Best %s: %f", k, v)

        return clf.best_estimator_

    def save_model(self, model: _SKModel) -> None:
        """Save model to file.

        Parameters
        ----------
        model : _SKModel
            model to save

        """
        model_path = join(self.save_path, f"{self.model_name}.pkl")
        save_pickle(model, model_path)

    def load_model(self, model_path: Optional[str] = None) -> _SKModel:
        """Load a saved model.

        Parameters
        ----------
        model_path : Optional[str], optional
            path to the saved model file, by default None

        Returns
        -------
        _SKModel
            loaded model

        """
        if not model_path:
            model_path = join(self.save_path, f"{self.model_name}.pkl")

        try:
            model = load_pickle(model_path)
        except FileNotFoundError:
            LOGGER.error("No saved model was found to load!")

        return model

    def predict(
        self,
        model,
        X_test: np.ndarray,
        y_test: np.ndarray,
    ) -> tuple:
        """Make prediction by a trained model.

        Parameters
        ----------
        model : _type_
            sklearn model object
        X_test : np.ndarray
            test data features
        y_test : np.ndarray
            test data labels

        Returns
        -------
        tuple
            data labels, predicted values (probabilities), predicted labels

        """
        y_pred_values = model.predict_proba(X_test)[:, 1]
        y_pred_labels = model.predict(X_test)
        if self.test_params["flatten"]:
            y_test = y_test.ravel()
            y_pred_values = y_pred_values.ravel()
            y_pred_labels = y_pred_labels.ravel()

        return (y_test, y_pred_values, y_pred_labels)
