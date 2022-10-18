import os
import random
import sys
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch.nn as nn
import torch.optim as optim
import scipy.stats as st
from sklearn.preprocessing import StandardScaler
from matplotlib.colors import ListedColormap
from datetime import date, timedelta
from tqdm import tqdm
from drift_detection.gemini.utils import *
from drift_detection.drift_detector.utils import *
from drift_detection.drift_detector.rolling_window import *
from drift_detection.drift_detector.detector import Detector
from drift_detection.drift_detector.reductor import Reductor
from drift_detection.drift_detector.tester import TSTester, DCTester
from drift_detection.baseline_models.temporal.pytorch.optimizer import Optimizer
from drift_detection.baseline_models.temporal.pytorch.utils import *

class CumulativeRetrainer:
    def __init__(
        self, 
        shift_detector: Detector = None, 
        optimizer: Optimizer = None,
        model = None,
        model_name: str = None,
        retrain_model_path: str = None,
        verbose: bool = False
    ):
        
        self.shift_detector = shift_detector
        self.optimizer = optimizer
        self.model = model
        self.model_name = model_name
        self.retrain_model_path = retrain_model_path
        self.verbose = verbose
        
    def retrain(
        self,
        data_streams: dict = None, 
        sample: int = 1000, 
        stat_window: int = 30, 
        lookup_window: int = 0, 
        stride: int = 1, 
        p_val_threshold: float = 0.05,
        batch_size: int = 64,
        n_epochs: int = 1,
        **kwargs
    ):    
        rolling_metrics = []
        run_length = stat_window 
        i = stat_window
        p_val = 1
        total_alarms = 0 
        num_timesteps = data_streams['X'][0].index.get_level_values(1).nunique()
        n_features = data_streams['X'][0].shape[1]
        
        pbar_total=len(data_streams['X'])-stat_window-lookup_window+1
        pbar = tqdm(total = pbar_total, miniters = int(pbar_total/100))
        
        while i+stat_window+lookup_window < len(data_streams['X']):
            pbar.update(1)
            feat_index = 0

            if p_val < p_val_threshold:

                X_update = pd.concat(data_streams['X'][max(int(i)-run_length,0):int(i)])
                X_update = X_update[~X_update.index.duplicated(keep='first')]
                ind = X_update.index.get_level_values(0).unique()
                X_next = scale(X_next)
                X_next = process(X_next, aggregation_type, num_timesteps)             

                y_update = pd.concat(data_streams['y'][max(int(i)-run_length,0):int(i)])
                y_update.index = ind
                y_update = y_update[~y_update.index.duplicated(keep='first')].to_numpy()

                if self.verbose:
                    print("Retrain ",self.model_name," on: ",max(int(i)-run_length,0),"-",int(i))

                if self.model_name in ["rnn","gru","lstm"]:
                    update_dataset = get_data(X_update, y_update)
                    update_loader = torch.utils.data.DataLoader(update_dataset, batch_size=1, shuffle=False)

                    self.retrain_model_path="_".join(["cumulative",str(n_epochs),str(sample),"retrain.model"])

                    ## train 
                    self.optimizer.train(
                        update_loader,
                        update_loader,
                        batch_size=batch_size,
                        n_epochs=n_epochs,
                        n_features=n_features,
                        timesteps=num_timesteps,
                        model_path=self.retrain_model_path,
                    )

                    self.model.load_state_dict(torch.load(self.retrain_model_path))
                    self.optimizer.model = self.model
                    self.shift_detector.model_path = self.retrain_model_path

                elif model_name == "gbt":
                    model = self.model.fit(X_retrain, y_retrain, xgb_model=self.model.get_booster())

                else:
                    print("Invalid Model Name")

                i += stride

            # Get covariates of next window
            X_next = pd.concat(data_streams['X'][max(int(i)+lookup_window,0):int(i)+stat_window+lookup_window])
            X_next = X_next[~X_next.index.duplicated(keep='first')]
            next_ind = X_next.index.get_level_values(0).unique()
            X_next = scale(X_next)
            X_next = process(X_next, aggregation_type, num_timesteps)
            # Get labels of next window
            y_next = pd.concat(data_streams['y'][max(int(i)+lookup_window,0):int(i)+stat_window+lookup_window])
            y_next.index = next_ind
            y_next = y_next[~y_next.index.duplicated(keep='first')].to_numpy()

            if X_next.shape[0]<=2:
                break

            ## Check performance of next window
            test_dataset = get_data(X_next, y_next)
            test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle=False)
            y_test_labels, y_pred_values, y_pred_labels = self.optimizer.evaluate(
                test_loader, batch_size=1, n_features=input_dim, timesteps=num_timesteps
            )
            assert y_test_labels.shape == y_pred_labels.shape == y_pred_values.shape
            y_pred_values = y_pred_values[y_test_labels != -1]
            y_pred_labels = y_pred_labels[y_test_labels != -1]
            y_test_labels = y_test_labels[y_test_labels != -1]
            performance_metrics = print_metrics_binary(
                y_test_labels, 
                y_pred_values, 
                y_pred_labels, 
                verbose=self.verbose
            )

            ## Run distribution shift test of next window
            drift_metrics = self.shift_detector.detect_shift(
                X_next,
                sample,
                **kwargs
            )

            metrics = {
                **drift_metrics, 
                **performance_metrics
            }
            
            p_val = drift_metrics['p_val']
            
            rolling_metrics.append(metrics)

            run_length += stat_window
            
            if self.verbose:
                print("P-value below threshold for ",data_streams['timesteps'][i+lookup_window],"-",data_streams['timesteps'][i+stat_window+lookup_window],"\tP-Value: ",drift_metrics['p_val'])

            if p_val < p_val_threshold:
                total_alarms += 1
            else:
                i += stride
                run_length += stat_window
                
        pbar.close()

        rolling_metrics = {
            k: [d.get(k) for d in rolling_metrics] for k in set().union(*rolling_metrics)
        }

        return rolling_metrics