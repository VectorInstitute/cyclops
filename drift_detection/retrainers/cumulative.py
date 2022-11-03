"""Retrainer that uses a cumulative set of data to retrain the model."""
import pandas as pd
import torch
from tqdm import tqdm

from drift_detection.baseline_models.temporal.pytorch.metrics import (
    print_metrics_binary,
)
from drift_detection.baseline_models.temporal.pytorch.optimizer import Optimizer
from drift_detection.baseline_models.temporal.pytorch.utils import get_data
from drift_detection.drift_detector.detector import Detector

# from drift_detection.drift_detector.rolling_window import RollingWindow
from drift_detection.drift_detector.utils import scale
from drift_detection.gemini.utils import process


class CumulativeRetrainer:
    """Retrainer that uses a cumulative set of data to retrain the model.

    Parameters
    ----------
    shift_detector : Detector
        Detector that is used to detect drift.
    optimizer : Optimizer
        Optimizer that is used to retrain the model.
    model : torch.nn.Module
        Model type to be retrained.
    model_name : str
        Name of the model that is retrained.
    retrain_model_path : str
        Path to the model that is retrained.
    verbose : bool
        Whether to print the tracking of drift detection.

    """

    def __init__(
        self,
        shift_detector: Detector,
        optimizer: Optimizer,
        model=None,
        model_name: str = None,
        retrain_model_path: str = None,
        verbose: bool = False,
    ):

        self.shift_detector = shift_detector
        self.optimizer = optimizer
        self.model = model
        self.model_name = model_name
        self.retrain_model_path = retrain_model_path
        self.verbose = verbose

    def retrain(
        self,
        data_streams: dict,
        sample: int = 1000,
        stat_window: int = 30,
        lookup_window: int = 0,
        stride: int = 1,
        p_val_threshold: float = 0.05,
        # batch_size: int = 64,
        n_epochs: int = 1,
        **kwargs
    ):
        """Retrain the model.

        Parameters
        ----------
        data_streams : dict
            Dictionary of data streams.
        sample : int
            Number of samples to be used.
        stat_window : int
            Size of the statistical window.
        lookup_window : int
            Size of the lookup window.
        stride : int
            Stride size.
        p_val_threshold : float
            Threshold for the p-value.
        batch_size : int
            Size of the batches.
        n_epochs : int
            Number of epochs.
        **kwargs
            Keyword arguments.

        Returns
        -------
        dict
            Dictionary of results to be sent to the Plotter.

        """
        rolling_metrics = []
        run_length = stat_window
        i = stat_window
        p_val = 1
        total_alarms = 0
        num_timesteps = data_streams["X"][0].index.get_level_values(1).nunique()
        # n_features = data_streams["X"][0].shape[1]

        aggregation_type = kwargs.get("aggregation_type", "mean")

        pbar_total = len(data_streams["X"]) - stat_window - lookup_window + 1
        pbar = tqdm(total=pbar_total, miniters=int(pbar_total / 100))

        while i + stat_window + lookup_window < len(data_streams["X"]):
            pbar.update(1)

            if p_val < p_val_threshold:

                X_update = pd.concat(
                    data_streams["X"][max(int(i) - run_length, 0) : int(i)]
                )
                X_update = X_update[~X_update.index.duplicated(keep="first")]
                ind = X_update.index.get_level_values(0).unique()
                # X_next = scale(X_next)
                X_next = scale(X_update)
                X_next = process(X_next, aggregation_type, num_timesteps)

                y_update = pd.concat(
                    data_streams["y"][max(int(i) - run_length, 0) : int(i)]
                )
                y_update.index = ind
                y_update = y_update[~y_update.index.duplicated(keep="first")].to_numpy()

                if self.verbose:
                    print(
                        "Retrain ",
                        self.model_name,
                        " on: ",
                        max(int(i) - run_length, 0),
                        "-",
                        int(i),
                    )

                if self.model_name in ["rnn", "gru", "lstm"]:
                    update_dataset = get_data(X_update, y_update)
                    update_loader = torch.utils.data.DataLoader(
                        update_dataset, batch_size=1, shuffle=False
                    )

                    self.retrain_model_path = "_".join(
                        ["cumulative", str(n_epochs), str(sample), "retrain.model"]
                    )

                    # train
                    self.optimizer.train(
                        update_loader,
                        update_loader,
                        # batch_size=batch_size,
                        n_epochs=n_epochs,
                        # n_features=n_features,
                        # timesteps=num_timesteps,
                        model_path=self.retrain_model_path,
                    )

                    self.model.load_state_dict(torch.load(self.retrain_model_path))
                    self.optimizer.model = self.model
                    self.shift_detector.reductor.model_path = self.retrain_model_path

                elif self.model_name == "gbt":
                    # X_retrain, y_retrain not defined
                    X_retrain, y_retrain = None, None
                    self.model = self.model.fit(
                        X_retrain, y_retrain, xgb_model=self.model.get_booster()
                    )

                else:
                    print("Invalid Model Name")

                i += stride

            # Get covariates of next window
            X_next = pd.concat(
                data_streams["X"][
                    max(int(i) + lookup_window, 0) : int(i)
                    + stat_window
                    + lookup_window
                ]
            )
            X_next = X_next[~X_next.index.duplicated(keep="first")]
            next_ind = X_next.index.get_level_values(0).unique()
            X_next = scale(X_next)
            X_next = process(X_next, aggregation_type, num_timesteps)
            # Get labels of next window
            y_next = pd.concat(
                data_streams["y"][
                    max(int(i) + lookup_window, 0) : int(i)
                    + stat_window
                    + lookup_window
                ]
            )
            y_next.index = next_ind
            y_next = y_next[~y_next.index.duplicated(keep="first")].to_numpy()

            if X_next.shape[0] <= 2:
                break

            # Check performance of next window
            test_dataset = get_data(X_next, y_next)
            test_loader = torch.utils.data.DataLoader(
                test_dataset, batch_size=1, shuffle=False
            )
            y_test_labels, y_pred_values, y_pred_labels = self.optimizer.evaluate(
                test_loader,
                # batch_size=1,
                # n_features=X_next.shape[-1],
                # timesteps=num_timesteps,
            )
            assert y_test_labels.shape == y_pred_labels.shape == y_pred_values.shape
            y_pred_values = y_pred_values[y_test_labels != -1]
            y_pred_labels = y_pred_labels[y_test_labels != -1]
            y_test_labels = y_test_labels[y_test_labels != -1]
            performance_metrics = print_metrics_binary(
                y_test_labels, y_pred_values, y_pred_labels, verbose=self.verbose
            )

            # Run distribution shift test of next window
            drift_metrics = self.shift_detector.detect_shift(X_next, sample, **kwargs)

            metrics = {**drift_metrics, **performance_metrics}

            p_val = drift_metrics["p_val"]

            rolling_metrics.append(metrics)

            run_length += stat_window

            if self.verbose:
                print(
                    "P-value below threshold for ",
                    data_streams["timesteps"][i + lookup_window],
                    "-",
                    data_streams["timesteps"][i + stat_window + lookup_window],
                    "\tP-Value: ",
                    drift_metrics["p_val"],
                )

            if p_val < p_val_threshold:
                total_alarms += 1
            else:
                i += stride
                run_length += stat_window

        pbar.close()

        rolling_metrics_dict = {
            k: [d.get(k) for d in rolling_metrics]
            for k in set().union(*rolling_metrics)
        }

        return rolling_metrics_dict
