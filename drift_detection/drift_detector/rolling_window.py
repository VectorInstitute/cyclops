import numpy as np
import random
import sys
import pandas as pd
from tqdm import tqdm
from drift_detection.gemini.utils import *
from drift_detection.baseline_models.temporal.pytorch.optimizer import Optimizer
from drift_detection.baseline_models.temporal.pytorch.utils import *
from drift_detection.drift_detector.detector import Detector
from drift_detection.drift_detector.utils import *
    
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
        admin_data = None,
        shift_detector: Detector = None, 
        optimizer: Optimizer = None,
        model = None
    ):
        
        self.admin_data = admin_data
        self.shift_detector = shift_detector
        self.optimizer = optimizer
        self.model = model
        
    def mean(
        self,
        X: dict = None, 
        window: int = 30
    ):
        """
        Get rolling mean of time series data.
        Parameters
        ----------
        X: dict
            time series data
        window: int
            window length
        """
        return X.rolling(window).mean().dropna(inplace=True)
        
    def stdev(
        self,
        X: dict = None, 
        window: int = 30
    ):
        """
        Get rolling standard deviation of time series data.
        Parameters
        ----------
        X: dict
            time series data
        window: int
            window length
        """
        return X.rolling(window).stdev().dropna(inplace=True)
        
    def performance(
        self, 
        data_streams: dict, 
        stat_window: int = 30, 
        lookup_window: int = 0, 
        stride: int = 1,
        aggregation_type= "time",
        outcome = "mortality"
    ):  
        """
        Rolling window to measure performance over time series.

        Returns
        -------
        performance_metrics: dict
            dataframe containing performance metrics across time series.
        """
    
        performance_metrics = []
        i = 0 
        num_timesteps = data_streams['X'][0].index.get_level_values(1).nunique()
        pbar_total=len(data_streams['X'])-stat_window-lookup_window+1
        pbar = tqdm(total = pbar_total, miniters = int(pbar_total/100))
        while i+stat_window+lookup_window < len(data_streams['X']):
            pbar.update(1)
            feat_index = 0

            X_next = pd.concat(data_streams['X'][i+lookup_window:i+lookup_window+stat_window])
            X_next = X_next[~X_next.index.duplicated(keep='first')]
            y_test_labels = get_label(self.admin_data, X_next, outcome)
            ind = X_next.index.get_level_values(0).unique()
            X_next = scale(X_next)
            X_next = process(X_next, aggregation_type, num_timesteps)

            y_next = pd.concat(data_streams['y'][i+lookup_window:i+lookup_window+stat_window])
            y_next.index = ind
            y_next = y_next[~y_next.index.duplicated(keep='first')].to_numpy()
            assert y_next.shape[0] == X_next.shape[0]

            if X_next.shape[0]<=2:
                break
                    
            if self.optimizer is not None:
                test_dataset = get_data(X_next, y_next)
                test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle=False)
                y_test_labels, y_pred_values, y_pred_labels = self.optimizer.evaluate(
                    test_loader, batch_size=1, n_features=data_streams['X'][0].shape[1], timesteps=num_timesteps
                )                
                y_pred_values = y_pred_values[y_test_labels != -1]
                y_pred_labels = y_pred_labels[y_test_labels != -1]
                y_test_labels = y_test_labels[y_test_labels != -1]
                pred_metrics = print_metrics_binary(y_test_labels, y_pred_values, y_pred_labels, verbose=0)
            else:
                y_pred_values = self.model.predict_proba(X_next)[:, 1]
                y_pred_labels = self.model.predict(X_next)
                pred_metrics = print_metrics_binary(y_test_labels, y_pred_values, y_pred_labels, verbose=0)
                
            performance_metrics.append(pred_metrics)

            i += stride
            
        pbar.close()

        performance_metrics = {
            k: [d.get(k) for d in performance_metrics] for k in set().union(*performance_metrics)
        }

        return performance_metrics

    def drift(
        self, 
        data_streams: dict, 
        sample: int = 1000, 
        stat_window: int = 30, 
        lookup_window: int = 0, 
        stride: int = 1, 
        threshold: float = 0.05,
        aggregation_type= "time",
        **kwargs
    ): 
        """
        Rolling window to measure drift over time series.

        Returns
        -------
        drift_metrics: dict
            dataframe containing drift p-value and distance metrics across time series.
        """
        
        rolling_drift_metrics = []
        num_timesteps = data_streams['X'][0].index.get_level_values(1).nunique()
        pbar_total=len(data_streams['X'])-stat_window-lookup_window+1
        pbar = tqdm(total = pbar_total, miniters = int(pbar_total/100))

        i = 0 

        while i+stat_window+lookup_window < len(data_streams['X']):
            pbar.update(1)
            feat_index = 0

            X_next = pd.concat(data_streams['X'][i+lookup_window:i+lookup_window+stat_window])
            X_next = X_next[~X_next.index.duplicated(keep='first')]
            X_next = scale(X_next)
            X_next = process(X_next, aggregation_type, num_timesteps)

            if X_next.shape[0]<=2:
                break

            drift_metrics = self.shift_detector.detect_shift(
                X_next,
                sample,
                **kwargs
            )

            if drift_metrics['p_val'] < threshold:
                print("P-value below threshold.")
                print("Ref -->",i+lookup_window,"-",i+stat_window+lookup_window,"\tP-Value: ",drift_metrics['p_val'])
                
            rolling_drift_metrics.append(drift_metrics)
            i += stride
        
        pbar.close()
        
        rolling_drift_metrics = {
            k: [d.get(k) for d in rolling_drift_metrics] for k in set().union(*rolling_drift_metrics)
        }
        
        return rolling_drift_metrics
