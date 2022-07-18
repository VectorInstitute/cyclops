import numpy as np
import os
import random
import sys
import pandas as pd
       
#####################################################
## rolling window - not cumulatively including data and no adjustment made when drift occurs
##################################################### 
def rolling_window(X_train, shift_detector, sample, stat_window, lookup_window, stride, num_timesteps, X_stream, threshold, custom_ref=None):

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
                                                                          X_prev, 
                                                                          X_next[:sample,:], 
                                                                          orig_dims,
        )
        
        if p_val < threshold:
            print("P-value below threshold.")
            print("Ref -->",i+lookup_window,"-",i+stat_window+lookup_window,"\tP-Value: ",p_val)
        dist_vals = np.concatenate((dist_vals, np.repeat(dist, 1)))
        p_vals = np.concatenate((p_vals, np.repeat(p_val, 1)))
        i += stride
        
    return dist_vals, p_vals