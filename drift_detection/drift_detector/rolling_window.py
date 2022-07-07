import numpy as np
import os
import random
import sys
import pandas as pd

#####################################################
## rolling window - not cumulatively including data and no adjustment made when drift occurs
##################################################### 
def rolling_window(shift_detector, sample, stat_window, lookup_window, stride, num_timesteps, series, threshold,custom_ref=None):

    p_vals = np.asarray([])
    dist_vals = np.asarray([])
    i = 0 
    if custom_ref is not None:
        prev = reshape_inputs(custom_ref, num_timesteps)
        
    while i+stat_window+lookup_window < len(series):
        feat_index = 0
        
        if custom_ref is None:
            prev = pd.concat(series[i:i+stat_window])
            prev = prev[~prev.index.duplicated(keep='first')]
            prev = reshape_inputs(prev, num_timesteps)
            
        next = pd.concat(series[i+lookup_window:i+lookup_window+stat_window])
        next = next[~next.index.duplicated(keep='first')]
        next = reshape_inputs(next, num_timesteps)
        
        if next.shape[0]<=2 or prev.shape[0]<=2:
            break
        
        (p_val, dist, val_acc, te_acc) = shift_detector.detect_data_shift(X_tr, 
                                                                          y_tr, 
                                                                          X_val, 
                                                                          y_val, 
                                                                          X_t[:sample, :], 
                                                                          y_t[:sample], 
                                                                          orig_dims
        )
            
        if p_val < threshold:
            print("P-value below threshold.")
            print(i,"-", i+stat_window,"-->",i+lookup_window,"-",i+stat_window+lookup_window,"\tP-Value: ",p_val)
        dist_val = preds['data']['distance']
        dist_vals = np.concatenate((dist_vals, np.repeat(dist_val, 1)))
        p_vals = np.concatenate((p_vals, np.repeat(p_val, 1)))
        i += stride
            
    return dist_vals, p_vals
