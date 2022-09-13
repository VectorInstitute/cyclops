import numpy as np
import os
import random
import sys
import pandas as pd

sys.path.append("..")

from baseline_models.temporal.pytorch.utils import *
from gemini.utils import *
    
class RollingWindow:
    
    """RollingWindow Class.

    Attributes
    ----------
    data_parameters: dictionary
        Dictionary containing training, validation and test data
    drift_parameters: dictionary
        Dictionary containing drift parameters: stat_window, lookup_window, stride
    model_parameters: dictionary
        Dictionary containing model parameters: num_timesteps, optimizer, input_dim

    """
        
    def __init__(self, data_parameters, drift_parameters, model_parameters):
        
        self.data_parameters = data_parameters
        self.drift_parameters = drift_parameters
        self.model_parameters = model_parameters

    def rolling_window_performance(self):  
    
        rolling_metrics = []
        i = 0 
        
        if data_parameters['X_ref'] is not None:
            X_prev = data_parameters['X_ref']

        while i+drift_parameters['stat_window']+drift_parameters['lookup_window'] < len(data_parameters['X_test']):
            feat_index = 0

            if data_parameters['X_ref'] is None:
                X_prev = pd.concat(data_parameters['X_test'][i:i+drift_parameters['stat_window']])
                X_prev = X_prev[~X_prev.index.duplicated(keep='first')]

            X_next = pd.concat(data_parameters['X_test'][i+drift_parameters['lookup_window']:i+drift_parameters['lookup_window']+drift_parameters['stat_window']])
            X_next = X_next[~X_next.index.duplicated(keep='first')]
            ind = X_next.index.get_level_values(0).unique()
            X_next = reshape_inputs(X_next, model_parameters['num_timesteps'])

            y_next = pd.concat(data_parameters['y_test'][i+drift_parameters['lookup_window']:i+drift_parameters['lookup_window']+drift_parameters['stat_window']])
            y_next.index = ind
            y_next = y_next[~y_next.index.duplicated(keep='first')].to_numpy()

            assert y_next.shape[0] == X_next.shape[0]

            if X_next.shape[0]<=2 or X_prev.shape[0]<=2:
                break

            test_dataset = get_data(X_next, y_next)
            test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle=False)
            y_test_labels, y_pred_values, y_pred_labels = model_parameters['optimizer'].evaluate(
                test_loader, batch_size=1, n_features=model_parameters['input_dim'], timesteps=model_parameters['num_timesteps']
            )
            y_pred_values = y_pred_values[y_test_labels != -1]
            y_pred_labels = y_pred_labels[y_test_labels != -1]
            y_test_labels = y_test_labels[y_test_labels != -1]
            pred_metrics = print_metrics_binary(y_test_labels, y_pred_values, y_pred_labels, verbose=0)
            rolling_metrics.append(pd.DataFrame(pred_metrics.values(),index=pred_metrics.keys()).T)

            i += drift_parameters['stride']

        rolling_metrics = pd.concat(rolling_metrics).reset_index(drop=True)

        return rolling_metrics

    def rolling_window(self):
        p_vals = []
        dist_vals =[]

        i = 0 
        if data_parameters['X_ref'] is not None:
            X_prev = data_parameters['X_ref']

        while i+drift_parameters['stat_window']+drift_parameters['lookup_window']< len(data_parameters['X_test']):
            feat_index = 0

            if data_parameters['X_ref'] is None:
                X_prev = pd.concat(data_parameters['X_test'][i:i+drift_parameters['stat_window']])
                X_prev = X_prev[~X_prev.index.duplicated(keep='first')]

            X_next = pd.concat(X_stream[i+drift_parameters['lookup_window']:i+drift_parameters['lookup_window']+drift_parameters['stat_window']])
            X_next = X_next[~X_next.index.duplicated(keep='first')]
            X_next = reshape_inputs(X_next, model_parameters['num_timesteps'])

            if X_next.shape[0]<=2 or X_prev.shape[0]<=2:
                break

            (p_val, dist, val_acc, te_acc) = drift_parameters['shift_detector'].detect_data_shift(data_parameters['X_train'], 
                                                                              X_prev[:1000,:], 
                                                                              X_next[:drift_parameters['sample'],:]
            )

            if p_val < drift_parameters['threshold']:
                print("P-value below threshold.")
                print("Ref -->",i+drift_parameters['lookup_window'],"-",i+drift_parameters['stat_window']+drift_parameters['lookup_window'],"\tP-Value: ",p_val)
                
            dist_vals.append(dist)
            p_vals.append(p_val)
            i += drift_parameters['stride']

        drift_metrics = {'dist': dist_vals, 
                         'pval': p_vals,
                        }

        return drift_metrics
