import numpy as np
import random
import sys
import pandas as pd

sys.path.append("..")

from baseline_models.temporal.pytorch.utils import Optimizer
from drift_detector.utils import *
    
class RollingWindow:
    
    """
    RollingWindow Class.
    ----------
    shift_detector: Detectpr
        Shift detector object to use in rolling window.
    optimizer: Optimizer
        Deep learning model optimizer to use in rolling window.

    """
        
    def __init__(
        self, 
        shift_detector: Detector = None, 
        optimizer: Optimizer = None
    ):
        
        self.shift_detector = shift_detector
        self.optimizer = optimizer
        
    def rolling_window_mean(
        x: pd.DataFrame = None, 
        window: int = 30
    ):
        """
        Get rolling mean of time series data.
        Parameters
        ----------
        x: 
            time series data
        window: int
            window length
        """
        return x.rolling(window).mean().dropna(inplace=True)
        
    def rolling_window_stdev(
        x: pd.DataFrame = None, 
        window: int =30
    ):
        """
        Get rolling standard deviation of time series data.
        Parameters
        ----------
        x: 
            time series data
        window: int
            window length
        """
        return x.rolling(window).stdev().dropna(inplace=True)
        
    def rolling_window_performance(self, 
                                   X_test: pd.DataFrame = None, 
                                   y_test: pd.DataFrame = None, 
                                   stat_window: int = 30, 
                                   lookup_window: int = 0, 
                                   stride: int = 1):  
        """
        Rolling window to measure performance over time series.

        Returns
        -------
        performance_metrics: pd.DataFrame
            dataframe containing performance metrics across time series.
        """
    
        performance_metrics = []
        i = 0 
        
        while i+stat_window+lookup_window < len(X_test):
            feat_index = 0

            X_next = pd.concat(X_test[i+lookup_window:i+lookup_window+stat_window])
            X_next = X_next[~X_next.index.duplicated(keep='first')]
            ind = X_next.index.get_level_values(0).unique()
            X_next = reshape_inputs(X_next, num_timesteps)

            y_next = pd.concat(y_test[i+lookup_window:i+lookup_window+stat_window])
            y_next.index = ind
            y_next = y_next[~y_next.index.duplicated(keep='first')].to_numpy()

            assert y_next.shape[0] == X_next.shape[0]

            if X_next.shape[0]<=2:
                break

            test_dataset = get_data(X_next, y_next)
            test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle=False)
                    
            if self.optimizer is not None:
                y_test_labels, y_pred_values, y_pred_labels = self.optimizer.evaluate(
                    test_loader, batch_size=1, n_features=X.shape[2], timesteps=X.shape[1]
                )
                
            y_pred_values = y_pred_values[y_test_labels != -1]
            y_pred_labels = y_pred_labels[y_test_labels != -1]
            y_test_labels = y_test_labels[y_test_labels != -1]
            pred_metrics = print_metrics_binary(y_test_labels, y_pred_values, y_pred_labels, verbose=0)
            performance_metrics.append(pd.DataFrame(pred_metrics.values(),index=pred_metrics.keys()).T)

            i += stride

        performance_metrics = pd.concat(performance_metrics).reset_index(drop=True)

        return performance_metrics

    def rolling_window_drift(self, 
                             X_train: pd.DataFrame = None, 
                             X_ref: pd.DataFrame = None, 
                             X_test: list = None, 
                             y_test: list = None, 
                             sample: int = 1000, 
                             stat_window: int = 30, 
                             lookup_window: int = 0, 
                             stride: int = 1, 
                             threshold: float = 0.05): 
        """
        Rolling window to measure drift over time series.

        Returns
        -------
        drift metrics: pd.DataFrame
            dataframe containing drift p-value and distance metrics across time series.
        """
        
        p_vals = []
        dist_vals =[]

        i = 0 
        if X_ref is not None:
            X_prev = X_ref

        while i+stat_window+lookup_window < len(X_test):
            feat_index = 0

            if X_ref is None:
                X_prev = pd.concat(X_test[i:i+stat_window])
                X_prev = X_prev[~X_prev.index.duplicated(keep='first')]

            X_next = pd.concat(X_stream[i+lookup_window:i+lookup_window+stat_window])
            X_next = X_next[~X_next.index.duplicated(keep='first')]
            X_next = reshape_inputs(X_next, num_timesteps)

            if X_next.shape[0]<=2 or X_prev.shape[0]<=2:
                break

            (p_val, dist, val_acc, te_acc) = self.shift_detector.detect_data_shift(X_train, 
                                                                              X_prev[:1000,:], 
                                                                              X_next[:sample,:]
            )

            if p_val < threshold:
                print("P-value below threshold.")
                print("Ref -->",i+lookup_window,"-",i+stat_window+lookup_window,"\tP-Value: ",p_val)
                
            dist_vals.append(dist)
            p_vals.append(p_val)
            i += stride

        drift_metrics = pd.DataFrame({'dist': dist_vals, 
                         'pval': p_vals,
                        })

        return drift_metrics
