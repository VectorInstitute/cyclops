"""Rolling window class for detecting drift on time series data."""
import pandas as pd
import torch
from tqdm import tqdm

from cyclops.monitor.detector import Detector
from cyclops.monitor.gemini.utils import get_label, process
from cyclops.monitor.optimizer import Optimizer
from cyclops.monitor.utils import get_data, print_metrics_binary, scale


class RollingWindow:
    """RollingWindow Class.

    Parameters
    ----------
    shift_detector: Detector
        Shift detector object to use in rolling window.
    optimizer: Optimizer
        Deep learning model optimizer to use in rolling window.

    Methods
    -------
    mean(X: dict, window: int)
        Get rolling mean of time series data.
    stdev(X: dict, window: int)
        Get rolling standard deviation of time series data.
    performance(data_streams: dict, stat_window: int, lookup_window: int, stride: int)
        Rolling window to measure performance over time series.

    """

    def __init__(
        self,
        admin_data: pd.DataFrame,
        shift_detector: Detector,
        optimizer: Optimizer = None,
        model=None,
        verbose: bool = False,
    ):

        self.admin_data = admin_data
        self.shift_detector = shift_detector
        self.optimizer = optimizer
        self.model = model
        self.verbose = verbose

    def mean(self, X: pd.DataFrame, window: int = 30):
        """Get rolling mean of time series data.

        Parameters
        ----------
        X: pd.DataFrame
            time series data
        window: int
            window length

        Returns
        -------
        X: pd.DataFrame
            time series data with rolling mean

        """
        return X.rolling(window).mean().dropna(inplace=True)

    def stdev(self, X: pd.DataFrame, window: int = 30):
        """Get rolling standard deviation of time series data.

        Parameters
        ----------
        X: pd.DataFrame
            time series data
        window: int
            window length

        Returns
        -------
        X: dict
            time series data with rolling standard deviation

        """
        return X.rolling(window).stdev().dropna(inplace=True)

    def performance(
        self,
        data_streams: dict,
        stat_window: int = 30,
        lookup_window: int = 0,
        stride: int = 1,
        aggregation_type="time",
        outcome="mortality",
    ):
        """Perform rolling window to measure performance over time series.

        Parameters
        ----------
        data_streams: dict
            time series data
        stat_window: int
            window length
        lookup_window: int
            window length
        stride: int
            window length
        aggregation_type: str
            type of aggregation to use
        outcome: str
            outcome to predict

        Returns
        -------
        performance_metrics: dict
            dataframe containing performance metrics across time series.

        """
        performance_metrics = []
        i = 0
        num_timesteps = data_streams["X"][0].index.get_level_values(1).nunique()
        # n_features = data_streams["X"][0].shape[1]
        pbar_total = len(data_streams["X"]) - stat_window - lookup_window + 1
        pbar = tqdm(total=pbar_total, miniters=int(pbar_total / 100))
        while i + stat_window + lookup_window < len(data_streams["X"]):
            pbar.update(1)

            X_next = pd.concat(
                data_streams["X"][i + lookup_window : i + lookup_window + stat_window]
            )
            X_next = X_next[~X_next.index.duplicated(keep="first")]
            y_test_labels = get_label(self.admin_data, X_next, outcome)
            ind = X_next.index.get_level_values(0).unique()
            X_next = scale(X_next)
            X_next = process(X_next, aggregation_type, num_timesteps)

            y_next = pd.concat(
                data_streams["y"][i + lookup_window : i + lookup_window + stat_window]
            )
            y_next.index = ind
            y_next = y_next[~y_next.index.duplicated(keep="first")].to_numpy()
            assert y_next.shape[0] == X_next.shape[0]

            if X_next.shape[0] <= 2:
                break

            if self.optimizer is not None:
                test_dataset = get_data(X_next, y_next)
                test_loader = torch.utils.data.DataLoader(
                    test_dataset, batch_size=1, shuffle=False
                )
                y_test_labels, y_pred_values, y_pred_labels = self.optimizer.evaluate(
                    test_loader,
                )
                y_pred_values = y_pred_values[y_test_labels != -1]
                y_pred_labels = y_pred_labels[y_test_labels != -1]
                y_test_labels = y_test_labels[y_test_labels != -1]
                pred_metrics = print_metrics_binary(
                    y_test_labels, y_pred_values, y_pred_labels, verbose=0
                )
            else:
                y_pred_values = self.model.predict_proba(X_next)[:, 1]
                y_pred_labels = self.model.predict(X_next)
                pred_metrics = print_metrics_binary(
                    y_test_labels, y_pred_values, y_pred_labels, verbose=0
                )

            performance_metrics.append(pred_metrics)

            i += stride

        pbar.close()

        performance_metrics_dict = {
            k: [d.get(k) for d in performance_metrics]
            for k in set().union(*performance_metrics)
        }

        return performance_metrics_dict

    def drift(
        self,
        data_streams: dict,
        sample: int = 1000,
        stat_window: int = 30,
        lookup_window: int = 0,
        stride: int = 1,
        threshold: float = 0.05,
        aggregation_type="time",
        **kwargs
    ):
        """Perform rolling window to measure drift over time series.

        Parameters
        ----------
        data_streams: dict
            time series data
        sample: int
            number of samples to use
        stat_window: int
            window length
        lookup_window: int
            window length
        stride: int
            window length
        threshold: float
            threshold for drift detection
        aggregation_type: str
            type of aggregation to use

        Returns
        -------
        drift_metrics: dict
            dataframe containing drift p-value and distance metrics across time series.

        """
        rolling_drift_metrics = []
        num_timesteps = data_streams["X"][0].index.get_level_values(1).nunique()
        pbar_total = len(data_streams["X"]) - stat_window - lookup_window + 1
        pbar = tqdm(total=pbar_total, miniters=int(pbar_total / 100))

        i = 0

        while i + stat_window + lookup_window < len(data_streams["X"]):
            pbar.update(1)

            X_next = pd.concat(
                data_streams["X"][i + lookup_window : i + lookup_window + stat_window]
            )
            X_next = X_next[~X_next.index.duplicated(keep="first")]
            X_next = scale(X_next)
            X_next = process(X_next, aggregation_type, num_timesteps)

            if X_next.shape[0] <= 2:
                break

            drift_metrics = self.shift_detector.detect_shift(X_next, sample, **kwargs)

            if drift_metrics["p_val"] < threshold:
                print(
                    "P-value below threshold for ",
                    data_streams["timesteps"][i + lookup_window],
                    "-",
                    data_streams["timesteps"][i + stat_window + lookup_window],
                    "\tP-Value: ",
                    drift_metrics["p_val"],
                )

            rolling_drift_metrics.append(drift_metrics)
            i += stride

        pbar.close()

        rolling_drift_metrics_dict = {
            k: [d.get(k) for d in rolling_drift_metrics]
            for k in set().union(*rolling_drift_metrics)
        }

        return rolling_drift_metrics_dict
