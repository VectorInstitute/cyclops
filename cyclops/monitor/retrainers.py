"""Retrainer that uses a cumulative set of data to retrain the model."""

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from cyclops.monitor.baseline_models.temporal.pytorch.metrics import (
    print_metrics_binary,
)
from cyclops.monitor.baseline_models.temporal.pytorch.optimizer import Optimizer
from cyclops.monitor.baseline_models.temporal.pytorch.utils import get_data
from cyclops.monitor.detector import Detector
from cyclops.monitor.gemini.utils import process
from cyclops.monitor.utils import scale


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
        # datastreams is dictionary of numpy arrays, static typing
        data_streams,
        sample: int = 1000,
        stat_window: int = 30,
        lookup_window: int = 0,
        stride: int = 1,
        p_val_threshold: float = 0.05,
        n_epochs: int = 1,
        correct_only: bool = False,
        aggregation_type="time",
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
        correct_only: bool
            Whether to use all samples for retraining or only those predicted
            correctly by the model.
        aggregation_type: str
            How to aggregate data (e.g. time, mean)
        **kwargs
            Keyword arguments.

        Returns
        -------
        dict
            Dictionary of retraining drift and performance results.

        """
        rolling_metrics = []
        run_length = stat_window
        i = stat_window
        p_val = 1
        num_timesteps = data_streams["X"][0].index.get_level_values(1).nunique()
        # n_features = data_streams["X"][0].shape[1]

        pbar_total = len(data_streams["X"]) - stat_window - lookup_window + 1
        pbar = tqdm(total=pbar_total, miniters=int(pbar_total / 100))

        while i + stat_window + lookup_window < len(data_streams["X"]):
            pbar.update(1)

            if p_val < p_val_threshold:

                X_update_streams = pd.concat(
                    data_streams["X"][max(int(i) - run_length, 0) : int(i)]
                )
                X_update_streams = X_update_streams[
                    ~X_update_streams.index.duplicated(keep="first")
                ]
                ind = X_update_streams.index.get_level_values(0).unique()
                encounter_ids = np.repeat(ind, num_timesteps)
                X_update = scale(X_update_streams)
                X_update = process(X_update, aggregation_type, num_timesteps)

                y_update = pd.concat(
                    data_streams["y"][max(int(i) - run_length, 0) : int(i)]
                )
                y_update.index = ind
                y_update = y_update[~y_update.index.duplicated(keep="first")]

                if self.verbose:
                    print(
                        "Retrain ",
                        self.model_name,
                        " on: ",
                        data_streams["timestamps"][max(int(i) - run_length, 0)],
                        "-",
                        data_streams["timestamps"][int(i)],
                    )

                if self.model_name in ["rnn", "gru", "lstm"]:
                    update_dataset = get_data(X_update, y_update.to_numpy())
                    update_loader = torch.utils.data.DataLoader(
                        update_dataset, batch_size=1, shuffle=False
                    )

                    if correct_only:
                        (
                            y_test_labels,
                            y_pred_values,
                            y_pred_labels,
                        ) = self.optimizer.evaluate(
                            update_loader,
                        )

                        y_pred_values = y_pred_values[y_pred_labels != y_test_labels]
                        encounter_ids = encounter_ids[y_pred_labels != y_test_labels]
                        y_test_labels = y_pred_labels = y_pred_labels[
                            y_pred_labels != y_test_labels
                        ]

                        assert len(encounter_ids) == len(y_test_labels)

                        X_update = X_update_streams.loc[
                            X_update_streams.index.get_level_values(0).isin(
                                encounter_ids
                            )
                        ]
                        X_update = scale(X_update_streams)
                        X_update = process(X_update, aggregation_type, num_timesteps)
                        y_update = y_update.loc[y_update.index.isin(encounter_ids)]

                        update_dataset = get_data(X_update, y_update.to_numpy())
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
                        n_epochs=n_epochs,
                    )

                    self.model.load_state_dict(torch.load(self.retrain_model_path))
                    self.optimizer.model = self.model
                    setattr(self.shift_detector, "model_path", self.retrain_model_path)

                elif self.model_name == "gbt":
                    # undefined name: X_retrain, y_retrain
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
            test_loader = torch.utils.data.DataLoader(test_dataset, shuffle=False)
            y_test_labels, y_pred_values, y_pred_labels = self.optimizer.evaluate(
                test_loader
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

            if p_val < p_val_threshold:
                if self.verbose:
                    print(
                        "P-value below threshold for ",
                        data_streams["timestamps"][i + lookup_window],
                        "-",
                        data_streams["timestamps"][i + stat_window + lookup_window],
                        "\tP-Value: ",
                        drift_metrics["p_val"],
                    )
            else:
                i += stride
                run_length += stat_window

        pbar.close()

        rolling_metrics_final = {
            k: [d.get(k) for d in rolling_metrics]
            for k in set().union(*rolling_metrics)
        }

        return rolling_metrics_final


class MostRecentRetrainer:
    """Retrainer that uses the most recent data points to retrain the model.

    Attributes
    ----------
    shift_detector : Detector
        Detector object for detecting data shift.
    optimizer : Optimizer
        Optimizer object for training the model.
    model : torch.nn.Module
        Model to be trained.
    model_name : str
        Name of the model.
    retrain_model_path : str
        Path to save the retrained model.
    verbose : bool
        Whether to print out the training progress.

    Methods
    -------
    retrain(X_s, X_t, **kwargs)
        Retrains the model on the target data.

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
        data_streams,
        retrain_window: int = 30,
        sample: int = 1000,
        stat_window: int = 30,
        lookup_window: int = 0,
        stride: int = 1,
        p_val_threshold: float = 0.05,
        n_epochs: int = 1,
        correct_only: bool = False,
        aggregation_type="time",
        **kwargs
    ):
        """Retrain the model on the target data.

        Parameters
        ----------
        data_streams : dict
            Dictionary of data streams.
        retrain_window : int
            Number of days to retrain the model.
        sample : int
            Number of samples to use for retraining.
        stat_window : int
            Number of days to compute the statistics.
        lookup_window : int
            Number of days to look ahead for the shift.
        stride : int
            Stride for the rolling window.
        p_val_threshold : float
            Threshold for the p-value.
        batch_size : int
            Batch size for training.
        n_epochs : int
            Number of epochs to train the model.
        aggregation_type: str
            How to aggregate data (e.g. time, mean)
        correct_only: bool
            Whether to use all samples for retraining or only those predicted
            correctly by the model.

        Returns
        -------
        results : dict
            Dictionary of retraining drift and performance results.

        """
        rolling_metrics = []
        run_length = stat_window
        i = stat_window
        p_val = 1
        num_timesteps = data_streams["X"][0].index.get_level_values(1).nunique()
        # n_features = data_streams["X"][0].shape[1]

        pbar_total = len(data_streams["X"]) - stat_window - lookup_window + 1
        pbar = tqdm(total=pbar_total, miniters=int(pbar_total / 100))

        while i + stat_window + lookup_window < len(data_streams["X"]):
            pbar.update(1)

            if p_val < p_val_threshold:

                X_update_streams = pd.concat(
                    data_streams["X"][max(int(i) - run_length, 0) : int(i)]
                )
                X_update_streams = X_update_streams[
                    ~X_update_streams.index.duplicated(keep="first")
                ]
                ind = X_update_streams.index.get_level_values(0).unique()
                encounter_ids = np.repeat(ind, num_timesteps)
                X_update = scale(X_update_streams)
                X_update = process(X_update, aggregation_type, num_timesteps)

                y_update = pd.concat(
                    data_streams["y"][max(int(i) - run_length, 0) : int(i)]
                )
                y_update.index = ind
                y_update = y_update[~y_update.index.duplicated(keep="first")]

                if self.verbose:
                    print(
                        "Retrain ",
                        self.model_name,
                        " on: ",
                        data_streams["timestamps"][max(int(i) - run_length, 0)],
                        "-",
                        data_streams["timestamps"][int(i)],
                    )

                if self.model_name in ["rnn", "gru", "lstm"]:
                    update_dataset = get_data(X_update, y_update.to_numpy())
                    update_loader = torch.utils.data.DataLoader(
                        update_dataset, batch_size=1, shuffle=False
                    )

                    # Remove all incorrectly predicted labels for retraining
                    # undefined name: input_dim
                    if correct_only:
                        (
                            y_test_labels,
                            y_pred_values,
                            y_pred_labels,
                        ) = self.optimizer.evaluate(
                            update_loader,
                        )

                        y_pred_values = y_pred_values[y_pred_labels != y_test_labels]
                        encounter_ids = encounter_ids[y_pred_labels != y_test_labels]
                        y_test_labels = y_pred_labels = y_pred_labels[
                            y_pred_labels != y_test_labels
                        ]

                        assert len(encounter_ids) == len(y_test_labels)

                        X_update = X_update_streams.loc[
                            X_update_streams.index.get_level_values(0).isin(
                                encounter_ids
                            )
                        ]
                        X_update = scale(X_update_streams)
                        X_update = process(X_update, aggregation_type, num_timesteps)
                        y_update = y_update.loc[y_update.index.isin(encounter_ids)]

                        update_dataset = get_data(X_update, y_update.to_numpy())
                        update_loader = torch.utils.data.DataLoader(
                            update_dataset, batch_size=1, shuffle=False
                        )

                    if self.retrain_model_path is None:
                        self.retrain_model_path = "_".join(
                            [
                                "mostrecent",
                                str(retrain_window),
                                str(n_epochs),
                                str(sample),
                                "retrain.model",
                            ]
                        )

                    # train
                    self.optimizer.train(
                        update_loader,
                        update_loader,
                        n_epochs=n_epochs,
                        model_path=self.retrain_model_path,
                    )

                    self.model.load_state_dict(torch.load(self.retrain_model_path))
                    self.optimizer.model = self.model
                    setattr(self.shift_detector, "model_path", self.retrain_model_path)

                elif self.model_name == "gbt":
                    # undefined name: X_retrain, y_retrain
                    X_retrain, y_retrain = None, None

                    self.model = self.model.fit(
                        X_retrain, y_retrain, xgb_model=self.model.get_booster()
                    )

                else:
                    print("Invalid Model Name")

                i += stride

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

            y_next = pd.concat(
                data_streams["y"][
                    max(int(i) + lookup_window, 0) : int(i)
                    + stat_window
                    + lookup_window
                ]
            )
            y_next.index = next_ind
            y_next = y_next[~y_next.index.duplicated(keep="first")].to_numpy()

            # Check if there are patient encounters in the next timestep
            if X_next.shape[0] <= 2:
                break

            # Check Performance
            test_dataset = get_data(X_next, y_next)
            test_loader = torch.utils.data.DataLoader(
                test_dataset, batch_size=1, shuffle=False
            )
            y_test_labels, y_pred_values, y_pred_labels = self.optimizer.evaluate(
                test_loader,
            )
            assert y_test_labels.shape == y_pred_labels.shape == y_pred_values.shape
            y_pred_values = y_pred_values[y_test_labels != -1]
            y_pred_labels = y_pred_labels[y_test_labels != -1]
            y_test_labels = y_test_labels[y_test_labels != -1]
            performance_metrics = print_metrics_binary(
                y_test_labels, y_pred_values, y_pred_labels, verbose=self.verbose
            )

            # Detect Distribution Shift
            drift_metrics = self.shift_detector.detect_shift(X_next, sample, **kwargs)
            p_val = drift_metrics["p_val"]
            metrics = {**drift_metrics, **performance_metrics}
            rolling_metrics.append(metrics)

            if p_val >= p_val_threshold:
                run_length += stride
                i += stride
                if self.verbose:
                    print(
                        "P-value below threshold for ",
                        data_streams["timestamps"][i + lookup_window],
                        "-",
                        data_streams["timestamps"][i + stat_window + lookup_window],
                        "\tP-Value: ",
                        drift_metrics["p_val"],
                    )
            else:
                run_length = retrain_window

        pbar.close()

        rolling_metrics = pd.concat(rolling_metrics).reset_index(drop=True)

        return rolling_metrics
