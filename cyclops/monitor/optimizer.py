"""Optimizer to train the baseline model."""

from datetime import datetime
from typing import Any, Callable, Generator, List, Optional, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader


class Optimizer:
    """Optimizer Class.

    Attributes
    ----------
    model: torch.nn.Module
        Pytorch model to optimize (e.g. RNNModel, LSTMModel, GRUModel)
    loss_fn: function
        Loss function
    optimizer: torch.optim
        Optimization algorithm (e.g. Adam)

    """

    def __init__(
        self,
        model: torch.nn.Module,
        loss_fn: Callable[..., torch.Tensor],
        optimizer: torch.optim.Optimizer,
        activation: Callable[..., torch.Tensor],
        lr_scheduler: torch.optim.lr_scheduler._LRScheduler,
        reweight_positive: Union[None, float, np.dtype[np.float64], str] = None,
    ):
        self.model = model
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.activation = activation
        self.lr_scheduler = lr_scheduler
        self.train_losses: List[float] = []
        self.val_losses: List[float] = []
        self.device = model.device
        self.reweight_positive = reweight_positive

    def reweight_loss(self, loss: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Reweight the losses."""
        if isinstance(self.reweight_positive, (float, np.float64)):
            loss[y.squeeze() == 1] *= self.reweight_positive
        elif self.reweight_positive == "mini-batch":
            loss[y.squeeze() == 1] *= (y == 0).sum() / (y == 1).sum()
        return loss

    def train_step(self, x: torch.Tensor, y: torch.Tensor) -> float:
        """Train model for one step."""
        # Sets model to train mode
        self.model.train(True)

        # Makes predictions
        yhat = self.model(x)

        # Computes loss
        loss: torch.Tensor = self.loss_fn(yhat.squeeze(), y.squeeze())

        # Reweight the losses
        loss = self.reweight_loss(loss, y)

        # Mask out the loss for -1 labels.
        loss *= ~y.eq(-1).squeeze()

        # Take mean of loss.
        loss = loss.sum() / (~y.eq(-1)).sum()

        # Computes gradients
        loss.backward()  # type: ignore

        # self.model.float()
        # Updates parameters and zeroes gradients
        self.optimizer.step()
        self.optimizer.zero_grad()

        # Returns the loss
        return loss.item()

    def train(
        self,
        train_loader: DataLoader[Any],
        val_loader: DataLoader[Any],
        n_epochs: int = 50,
        model_path: Optional[str] = None,
    ) -> None:
        """Train pytorch model.

        Parameters
        ----------
        train_loader: DataLoader
            Dataset object containing training set.
        val_loader: DataLoader
            Dataset object containing validation set.
        n_epochs: int
            Number of complete passes through the training set.
        model_path: str
            Path to save model.

        """
        if model_path is None:
            model_path = f'checkpoint_{datetime.now().strftime("%Y-%m-%d %H:%M:%S")}'

        for epoch in range(1, n_epochs + 1):
            batch_losses = []
            for x_batch, y_batch in train_loader:
                x_batch = x_batch.to(self.device)
                y_batch = y_batch.to(self.device)
                loss = self.train_step(x_batch, y_batch)

                assert not np.isnan(loss).any()

                batch_losses.append(loss)
            training_loss = np.mean(batch_losses)
            self.train_losses.append(training_loss)

            with torch.no_grad():
                batch_val_losses = []
                for x_val, y_val in val_loader:
                    x_val = x_val.to(self.device)
                    y_val = y_val.to(self.device)
                    self.model.eval()
                    yhat = self.model(x_val)
                    val_loss = self.loss_fn(yhat.squeeze(), y_val.squeeze())

                    # Reweight the losses
                    val_loss = self.reweight_loss(val_loss, y_val)

                    # Mask out the loss for -1 labels.
                    val_loss *= ~y_val.eq(-1).squeeze()

                    # Take mean of loss.
                    val_loss = (val_loss.sum() / (~y_val.eq(-1)).sum()).item()

                    assert not np.isnan(val_loss).any()

                    batch_val_losses.append(val_loss)
                validation_loss = np.mean(batch_val_losses)
                self.val_losses.append(validation_loss)

            torch.save(self.model.state_dict(), model_path)
            print(
                f"[{epoch}/{n_epochs}] Training loss: {training_loss:.4f}\t \
                Validation loss: {validation_loss:.4f}"
            )
            self.lr_scheduler.step()

    def evaluate(
        self,
        test_loader: DataLoader[Any],
        flatten: bool = True,
    ) -> Union[
        Tuple[
            np.ndarray[float, np.dtype[np.float64]],
            np.ndarray[float, np.dtype[np.float64]],
            np.ndarray[float, np.dtype[np.float64]],
        ],
        Generator[np.ndarray[float, np.dtype[np.float64]], None, None],
    ]:
        """Evaluate pytorch model.

        Parameters
        ----------
        test_loader: DataLoader
            Dataset object containing test set.
        batch_size: int
            Number of samples in each batch.

        """
        with torch.no_grad():
            y_pred_values: List[torch.Tensor] = []
            y_test_labels: List[torch.Tensor] = []
            y_pred_labels: List[torch.Tensor] = []
            for x_test, y_test in test_loader:
                # x_test = x_test.view([batch_size, timesteps, n_features]).to(
                #    self.device
                # )
                x_test = x_test.to(self.device)
                y_test = y_test.to(self.device)
                self.model.eval()
                y_hat = self.activation(self.model(x_test))
                y_pred_values.append(y_hat.cpu().detach())
                y_test_labels.append(y_test.cpu().detach())
                y_pred_labels.append(torch.round(y_hat).cpu().detach())

        y_test_labels_array: np.ndarray[float, np.dtype[np.float64]] = np.concatenate(
            y_test_labels
        )
        y_pred_labels_array: np.ndarray[float, np.dtype[np.float64]] = np.concatenate(
            y_pred_labels
        )
        y_pred_values_array: np.ndarray[float, np.dtype[np.float64]] = np.concatenate(
            y_pred_values
        )

        if flatten:
            return (
                y.flatten()
                for y in [y_test_labels_array, y_pred_values_array, y_pred_labels_array]
            )

        return y_test_labels_array, y_pred_values_array, y_pred_labels_array

    def plot_losses(self) -> None:
        """Plot training and validation losses."""
        plt.plot(self.train_losses, label="Training loss")
        plt.plot(self.val_losses, label="Validation loss")
        plt.legend()
        plt.title("Losses")
        plt.show()
        plt.close()
