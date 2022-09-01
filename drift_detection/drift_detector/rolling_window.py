import numpy as np
import os
import random
import sys
import pandas as pd
from datetime import date, timedelta
from drift_detection.baseline_models.temporal.pytorch.utils import *

sys.path.append("..")

from utils.utils import *

def daterange(start_date, end_date, stride, window):
    for n in range(int((end_date - start_date).days)):
        if start_date + timedelta(n*stride+window) < end_date:
            yield start_date+ timedelta(n*stride)
            
def get_streams(x, y, admin_data, start_date, end_date, stride, window, ids_to_exclude=None):
    target_stream_x = []
    target_stream_y = [] 
    measure_dates = []

    admit_df = admin_data[[ENCOUNTER_ID,ADMIT_TIMESTAMP]].sort_values(by=ADMIT_TIMESTAMP)
    for single_date in daterange(start_date, end_date, stride, window):
        if single_date.month ==1 and single_date.day == 1:
            print(single_date.strftime("%Y-%m-%d"),"-",(single_date+timedelta(days=window)).strftime("%Y-%m-%d"))
        encounters_inwindow = admit_df.loc[((single_date+timedelta(days=window)).strftime("%Y-%m-%d") > admit_df[ADMIT_TIMESTAMP].dt.strftime("%Y-%m-%d")) 
                           & (admit_df[ADMIT_TIMESTAMP].dt.strftime("%Y-%m-%d") >= single_date.strftime("%Y-%m-%d")), ENCOUNTER_ID].unique()
        if ids_to_exclude is not None:
            encounters_inwindow = [x for x in encounters_inwindow if x not in ids_to_exclude]
        encounter_ids = x.index.get_level_values(0).unique()
        x_inwindow = x.loc[x.index.get_level_values(0).isin(encounters_inwindow)]
        y_inwindow = pd.DataFrame(y[np.in1d(encounter_ids, encounters_inwindow)])
        if not x_inwindow.empty:
            target_stream_x.append(x_inwindow)
            target_stream_y.append(y_inwindow)
            measure_dates.append((single_date+timedelta(days=window)).strftime("%Y-%m-%d"))
    return(target_stream_x, target_stream_y, measure_dates)

def rolling_window_performance(X_stream, y_stream, opt, sample, stat_window, lookup_window, stride, num_timesteps, input_dim, threshold, custom_ref=None):
    
    rolling_metrics = []
    
    i = 0 
    if custom_ref is not None:
        X_prev = custom_ref
    
    while i+stat_window+lookup_window < len(X_stream):
        feat_index = 0
        
        if custom_ref is None:
            X_prev = pd.concat(X_stream[i:i+stat_window])
            X_prev = X_prev[~X_prev.index.duplicated(keep='first')]
            
        X_next = pd.concat(X_stream[i+lookup_window:i+lookup_window+stat_window])
        X_next = X_next[~X_next.index.duplicated(keep='first')]
        ind = X_next.index.get_level_values(0).unique()
        X_next = reshape_inputs(X_next, num_timesteps)
        
        y_next = pd.concat(y_stream[i+lookup_window:i+lookup_window+stat_window])
        y_next.index = ind
        y_next = y_next[~y_next.index.duplicated(keep='first')].to_numpy()
       
        assert y_next.shape[0] == X_next.shape[0]
        
        if X_next.shape[0]<=2 or X_prev.shape[0]<=2:
            break
        
        test_dataset = get_data(X_next, y_next)
        test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle=False)
        y_test_labels, y_pred_values, y_pred_labels = opt.evaluate(
            test_loader, batch_size=1, n_features=input_dim, timesteps=num_timesteps
        )

        y_pred_values = y_pred_values[y_test_labels != -1]
        y_pred_labels = y_pred_labels[y_test_labels != -1]
        y_test_labels = y_test_labels[y_test_labels != -1]

        pred_metrics = print_metrics_binary(y_test_labels, y_pred_values, y_pred_labels, verbose=0)
        rolling_metrics.append(pd.DataFrame(pred_metrics.values(),index=pred_metrics.keys()).T)
        
        i += stride
        
    rolling_metrics = pd.concat(rolling_metrics).reset_index(drop=True)
    
    return rolling_metrics

def rolling_window_drift(X_train, X_stream, shift_detector, sample, stat_window, lookup_window, stride, num_timesteps, threshold, custom_ref=None):

    p_vals = np.asarray([])
    dist_vals = np.asarray([])
    
    i = 0 
    if custom_ref is not None:
        X_prev = custom_ref
    
    while i+stat_window+lookup_window < len(X_stream):
        feat_index = 0
        
        if custom_ref is None:
            X_prev = pd.concat(X_stream[i:i+stat_window])
            X_prev = X_prev[~X_prev.index.duplicated(keep='first')]
            
        X_next = pd.concat(X_stream[i+lookup_window:i+lookup_window+stat_window])
        X_next = X_next[~X_next.index.duplicated(keep='first')]
        X_next = reshape_inputs(X_next, num_timesteps)
       
        if X_next.shape[0]<=2 or X_prev.shape[0]<=2:
            break
        
        (p_val, dist, val_acc, te_acc) = shift_detector.detect_data_shift(X_train, 
                                                                          X_prev[:1000,:], 
                                                                          X_next[:sample,:]
        )
        
        if p_val < threshold:
            print("P-value below threshold.")
            print("Ref -->",i+lookup_window,"-",i+stat_window+lookup_window,"\tP-Value: ",p_val)
        dist_vals = np.concatenate((dist_vals, np.repeat(dist, 1)))
        p_vals = np.concatenate((p_vals, np.repeat(p_val, 1)))
        i += stride
        
    return dist_vals, p_vals