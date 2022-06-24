import numpy as np

def rolling_window(n_start_window, n_end_window, n_window, series, threshold):

    p_vals = np.asarray([])
    dist_vals = np.asarray([])

    run_length = int(n_start_window)
    i = n_start_window

    while i+n_end_window+n_window <= series.shape[1]:
        feat_index = 0
        print(max(int(i)-run_length,0),"-", int(i),"-->",max(int(i)+n_window,0),"-",int(i)+n_end_window+n_window)
        prev = series[: , max(int(i)-run_length,0):int(i), :]
        prev = prev.reshape(prev.shape[0]*prev.shape[1],prev.shape[2])
        next = series[: , max(int(i)+n_window,0):int(i)+n_end_window+n_window, :]
        next = next.reshape(next.shape[0]*next.shape[1],next.shape[2])
        if next.shape[0]<=2 or prev.shape[0]<=2:
            break
            
        ## run distribution shift check here
        cd = MMDDrift(prev, backend='pytorch', p_val=.05)
        preds = cd.predict(next, return_p_val=True, return_distance=True)
        p_val = preds['data']['p_val']
        dist_val = preds['data']['distance']

        #####################################
        if p_val >= threshold:
            dist_vals = np.concatenate((dist_vals, np.repeat(dist_val, 1)))
            dist_vals = np.concatenate((dist_vals, np.repeat(0, n_end_window-1)))
            i += n_end_window
            run_length += n_start_window
        else:
            dist_vals = np.concatenate((dist_vals, np.repeat(dist_val, 1)))
            i+=1
            run_length = n_start_window

    return dist_vals, 
