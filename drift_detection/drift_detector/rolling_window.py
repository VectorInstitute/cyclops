import numpy as np

def dynamic_rolling_window(x_val, stride, window, timesteps, series, threshold):

    p_vals = np.asarray([])
    dist_vals = np.asarray([])

    run_length = int(window)
    i = window

#    for single_date in daterange(start_date, end_date):
#        print(single_date.strftime("%Y-%m-%d"))
    #(single_date+timedelta(days=window)).strftime("%Y-%m-%d")
    
    
    while i+stride+window <= len(series):
        feat_index = 0
        
        prev = pd.concat(series[max(int(i)-run_length,0):int(i)])
        prev = prev[~prev.index.duplicated(keep='first')]
        prev = reshape_inputs(prev, num_timesteps)
        next = pd.concat(series[max(int(i)+window,0):int(i)+stride+window])
        next = next[~next.index.duplicated(keep='first')]
        next = reshape_inputs(next, num_timesteps)
        if next.shape[0]<=2 or prev.shape[0]<=2:
            break
            
        ## run distribution shift check here
        cd = MMDDrift(prev, backend='pytorch', p_val=.05)
        preds = cd.predict(next, return_p_val=True, return_distance=True)
        p_val = preds['data']['p_val']
        print(max(int(i)-run_length,0),"-", int(i),"-->",max(int(i)+window,0),"-",int(i)+stride+window,"\tP-Value: ",p_val)
        dist_val = preds['data']['distance']

        #####################################
        if p_val >= threshold:
            dist_vals = np.concatenate((dist_vals, np.repeat(dist_val, 1)))
            dist_vals = np.concatenate((dist_vals, np.repeat(0, stride-1)))
            p_vals = np.concatenate((p_vals, np.repeat(p_val, 1)))
            p_vals = np.concatenate((p_vals, np.repeat(0, stride-1)))
            i += stride
            run_length += stride
        else:
            dist_vals = np.concatenate((dist_vals, np.repeat(dist_val, 1)))
            p_vals = np.concatenate((p_vals, np.repeat(p_val, 1)))
            i+=1
            run_length = stride
        
        if i == 500:
            break
            
    return dist_vals, p_vals

