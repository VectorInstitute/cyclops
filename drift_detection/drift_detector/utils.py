import numpy as np
import os
import random
import sys
import pandas as pd
from datetime import date, timedelta
from utils.utils import *

def daterange(start_date, end_date, stride, window):
    for n in range(int((end_date - start_date).days)):
        if start_date + timedelta(n*stride+window) < end_date:
            yield start_date+ timedelta(n*stride)  

def get_streams(x, y, admin_data, start_date, end_date, stride, window, ids_to_exclude=None, encounter_id=ENCOUNTER_ID, admit_timestamp=ADMIT_TIMESTAMP):
    
    target_stream_x = []
    target_stream_y = [] 
    timestamps = []

    admit_df = admin_data[[encounter_id,admit_timestamp]].sort_values(by=admit_timestamp)
    for single_date in daterange(start_date, end_date, stride, window):
        if single_date.month ==1 and single_date.day == 1:
            print(single_date.strftime("%Y-%m-%d"),"-",(single_date+timedelta(days=window)).strftime("%Y-%m-%d"))
        encounters_inwindow = admit_df.loc[((single_date+timedelta(days=window)).strftime("%Y-%m-%d") > admit_df[admit_timestamp].dt.strftime("%Y-%m-%d")) 
                            & (admit_df[admit_timestamp].dt.strftime("%Y-%m-%d") >= single_date.strftime("%Y-%m-%d")), encounter_id].unique()
        if ids_to_exclude is not None:
            encounters_inwindow = [x for x in encounters_inwindow if x not in ids_to_exclude]
        encounter_ids = x.index.get_level_values(0).unique()
        x_inwindow = x.loc[x.index.get_level_values(0).isin(encounters_inwindow)]
        y_inwindow = pd.DataFrame(y[np.in1d(encounter_ids, encounters_inwindow)])
        if not x_inwindow.empty:
            target_stream_x.append(x_inwindow)
            target_stream_y.append(y_inwindow)
            timestamps.append((single_date+timedelta(days=window)).strftime("%Y-%m-%d"))
    target_data = { 'timestamps': timestamps,
                    'target_stream_x': target_stream_y,
                   'target_stream_y': target_stream_y 
                  }                 
    return(target_data)