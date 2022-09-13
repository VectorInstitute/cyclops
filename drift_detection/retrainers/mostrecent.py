import os
import random
import sys
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import StandardScaler
from matplotlib.colors import ListedColormap

sys.path.append('..')

from drift_detector.rolling_window import *
from baseline_models.temporal.pytorch.optimizer import Optimizer
from baseline_models.temporal.pytorch.utils import *

class MostRecentRollingWindow:
    
    """MostRecentRollingWindow Class.

    Attributes
    ----------
    data_parameters: dictionary
        Dictionary containing training, validation and test data
    drift_parameters: dictionary
        Dictionary containing drift parameters: stat_window, lookup_window, stride
    model_parameters: dictionary
        Dictionary containing model parameters: num_timesteps, optimizer, input_dim

    """
    
    def __init__(self, data_parameters, model_parameters, retrain_parameters):
        self.data_parameters = data_parameters
        self.retrain_parameters = retrain_parameters
        self.model_parameters = model_parameters
    
    def retrain(self):
        p_vals = []
        dist_vals = []
        rolling_metrics = []
        run_length = retrain_parameters['stat_window'] 
        i = retrain_parameters['stat_window'] 
        p_val = 1
        val_dataset = get_data(data_parameters['X_val'], data_parameters['y_val'])
        val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=model_parameters['batch_size'] , shuffle=False)
        X_ref = data_parameters['X_val']

        while i+retrain_parameters['stat_window']+retrain_parameters['lookup_window']  <= len(data_parameters['X_test']):
            feat_index = 0

            if p_val < retrain_parameters['drift_threshold'] :

                if retrain_parameters['retrain_type'] is not None:
                    X_update = pd.concat(data_parameters['X_test'][max(int(i)-run_length,0):int(i)])
                    X_update = X_update[~X_update.index.duplicated(keep='first')]
                    ind = X_update.index.get_level_values(0).unique()
                    X_update = reshape_inputs(X_update, model_parameters['num_timesteps'])

                    ## Get updated source data for two-sample test (including data for retraining) 
                    X_ref = np.concatenate((X_ref, X_update), axis=0)
                    tups = [tuple(row) for row in X_ref]
                    X_ref = np.unique(tups, axis=0)
                    np.random.shuffle(X_ref)                 

                    y_update = pd.concat(y_stream[max(int(i)-run_length,0):int(i)])
                    y_update.index = ind
                    y_update = y_update[~y_update.index.duplicated(keep='first')].to_numpy()

                    if verbose:
                        print("Retrain ",model_parameters['model_name']," on: ",max(int(i)-run_length,0),"-",int(i))

                    if model_parameters['model_name'] == "rnn":
                        ## create train loader 
                        update_dataset = get_data(X_update, y_update)
                        update_loader = torch.utils.data.DataLoader(update_dataset, batch_size=model_parameters['batch_size'], shuffle=False)
                        retrain_model_path = 'mostrecent_' + retrain_parameters['stat_window'] + '_' + model_parameters['n_epochs'] + 'epoch_n' + retrain_parameters['sample'] + '_window_retrain.model'

                        ## train 
                        opt.train(
                             update_loader,
                             val_loader,
                             batch_size=model_parameters['batch_size'],
                             n_epochs=model_parameters['n_epochs'],
                             n_features=model_parameters['input_dim'],
                             timesteps=model_parameters['num_timesteps'],
                             model_path=retrain_model_path,
                        )

                        model.load_state_dict(torch.load(retrain_model_path))
                        opt.model = model
                        shift_detector.model_path = retrain_model_path

                    elif model_parameters['model_name'] == "gbt":
                        model = model.fit(X_retrain, y_retrain, xgb_model=model.get_booster())

                    else:
                        print("Invalid Model Name")

                i += retrain_parameters['stride']
                
            if data_parameters['X_val'] is None:
                X_ref = pd.concat(data_parameters['X_test'][max(int(i)-run_length,0):int(i)+retrain_parameters['stat_window']])
                X_ref = X_ref[~X_ref.index.duplicated(keep='first')]
                X_ref = reshape_inputs(X_ref, model_parameters['num_timesteps'])
                #X_ref = X_ref.reshape(X_ref.shape[0]*X_ref.shape[1],X_ref.shape[2])
                
            X_next = pd.concat(data_parameters['X_test'][max(int(i)+retrain_parameters['lookup_window'],0):int(i)+retrain_parameters['stat_window']+retrain_parameters['lookup_window']])
            X_next = X_next[~X_next.index.duplicated(keep='first')]
            next_ind = X_next.index.get_level_values(0).unique()
            X_next = reshape_inputs(X_next, model_parameters['num_timesteps'])

            y_next = pd.concat(data_parameters['y_test'][max(int(i)+retrain_parameters['lookup_window'],0):int(i)+retrain_parameters['stat_window']+retrain_parameters['lookup_window']])
            y_next.index = next_ind
            y_next = y_next[~y_next.index.duplicated(keep='first')].to_numpy()

            if X_next.shape[0]<=2:
                print("No more data, ending retraining.")
                return
            
            if X_ref.shape[0]<=2:
                print("Reference is empty, exiting retraining.")
                return

            ## Check Performance 
            test_dataset = get_data(X_next, y_next)
            test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle=False)
            y_test_labels, y_pred_values, y_pred_labels = model_parameters['optimizer'].evaluate(
                test_loader, batch_size=1, n_features=model_parameters['input_dim'], timesteps=model_parameters['num_timesteps'] 
            )
            assert y_test_labels.shape == y_pred_labels.shape == y_pred_values.shape
            y_pred_values = y_pred_values[y_test_labels != -1]
            y_pred_labels = y_pred_labels[y_test_labels != -1]
            y_test_labels = y_test_labels[y_test_labels != -1]  
            pred_metrics = print_metrics_binary(y_test_labels, y_pred_values, y_pred_labels, verbose=0)
            rolling_metrics.append(pd.DataFrame(pred_metrics.values(),index=pred_metrics.keys()).T)

            ## Detect Distribution Shift 
            (p_val, dist, val_acc, te_acc) = retrain_parameters['shift_detector'].detect_data_shift(data_parameters['X_train'], 
                                                                              X_ref[:1000,:], 
                                                                              X_next[:retrain_parameters['sample'] ,:]
            )

            if retrain_parameters['verbose']:
                print("Drift on ",max(int(i)+retrain_parameters['lookup_window'],0),"-",int(i)+retrain_parameters['stat_window']+retrain_parameters['lookup_window']," P-Value: ",p_val,)

            dist_vals.append(dist)
            p_vals.append(p_val)

            if p_val >= retrain_parameters['drift_threshold']:
                run_length += retrain_parameters['stride'] 
                i += retrain_parameters['stride'] 
            else:
                run_length= retrain_parameters['retrain_window'] 

        rolling_metrics = pd.concat(rolling_metrics).reset_index(drop=True)
        
        drift_metrics = {'dist': dist_vals, 
                         'pval': p_vals,
                        }

        return drift_metrics, rolling_metrics
