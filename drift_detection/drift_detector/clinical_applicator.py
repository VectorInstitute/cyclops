import datetime
import random
import sys
import numpy as np
import pandas as pd
from .utils import get_args

class ClinicalShiftApplicator(object):
    
    """
    The ClinicalShiftApplicator class is used induce synthetic dataset shift.
    --------
    
    Parameters
    ----------
    shift_type: str
        Method used to induce shift in data. Options include: "seasonal", "hospital_type".
    
    """   
    def __init__(self, 
                 shift_type: str, 
                 **kwargs
                ):
        
        self.shift_type = shift_type
        
        self.shift_types = { 
            "month": month, 
            "hospital_type": hospital_type,
            "time": time,
        }
           
        if self.shift_type not in self.shift_types.keys():
            raise ValueError(
                "Shift not supported, must be one of: {}".format(
                    self.shift_types.keys()
                )
            )
        
    def apply_shift(
        self, 
        X, 
        admin_data,
        **kwargs
    ):

        """apply_shift.

        Returns
        ----------
        X: pd.DataFrame
            Data to apply shift to.
        y: 
            Outcome labels.
        admin_data: pd.DataFrame
            Dataframe containing admin variables to filter on (e.g. "hospital_id", "admit_timestamp").
        """
        
        encounter_ids = list(X.index.get_level_values(0).unique())
        X = X[np.in1d(X.index.get_level_values(0),admin_data['encounter_id'])]
    
        X_s, X_t = self.shift_types[self.shift_type](
                X, admin_data, **get_args(self.shift_types[self.shift_type], kwargs)
            )
        
        encounter_ids = list(X.index.get_level_values(0).unique())
     
        return (X_s, X_t)  

            
def time(
    X, 
    admin_data, 
    start_date,
    cutoff_date,
    end_date,
    admit_timestamp='admit_timestamp', 
    encounter_id='encounter_id'
):
    """time.

    Parameters
    ----------
    X: pd.DataFrame
        Data to apply shift to.
    admin_data: pd.DataFrame
        Dataframe containing admin variables to filter on (e.g. "hospital_id", "admit_timestamp").
    start_date: datetime.date
        Start of source data.
    cutoff_date: datetime.date
        Cutoff to separate source and target data.
    end_date: datetime.date
        End of target data.
    encounter_id: str
        Column name for encounter ids.
    admit_timestamp: str
        Column name for admission timestamps.

    """
    ids_source = admin_data.loc[
        ((admin_data[admit_timestamp].dt.date > start_date) 
            & (admin_data[admit_timestamp].dt.date < cutoff_date),
        ),
        encounter_id,
    ]
    ids_target = admin_data.loc[
        ((admin_data[admit_timestamp].dt.date > cutoff_date) 
            & (admin_data[admit_timestamp].dt.date < end_date),
        ),
        encounter_id,
    ]
    X_s = X.loc[X.index.get_level_values(0).isin(ids_source)]
    X_t = X.loc[X.index.get_level_values(0).isin(ids_target)]
    return (X_s, X_t)

def month(
    X, 
    admin_data, 
    source,
    target,
    encounter_id='encounter_id', 
    admit_timestamp='admit_timestamp'
):
    """month.

    Parameters
    ----------
    X: pd.DataFrame
        Data to apply shift to.
    admin_data: pd.DataFrame
        Dataframe containing admin variables to filter on (e.g. "hospital_id", "admit_timestamp").
    encounter_id: str
        Column name for encounter ids.
    admit_timestamp: str
        Column name for admission timestamps.

    """
    ids_source = admin_data.loc[
        (
            (admin_data[admit_timestamp].dt.month.isin(source))
        ),
            encounter_id,
    ]
    ids_target = admin_data.loc[
        (
            (admin_data[admit_timestamp].dt.month.isin(target))
        ),
        encounter_id,
    ]
    X_s = X.loc[X.index.get_level_values(0).isin(ids_source)]
    X_t = X.loc[X.index.get_level_values(0).isin(ids_target)]
    return (X_s, X_t)

def hospital_type(
    X, 
    admin_data,
    source, 
    target, 
    encounter_id='encounter_id', 
    hospital_id='hospital_id'
):
    
    """hospital_type.

    Parameters
    ----------
    X: pd.DataFrame
        Data to apply shift to.
    admin_data: pd.DataFrame
        Dataframe containing admin variables to filter on (e.g. "hospital_id", "admit_timestamp").
    source_hospitals: list
        List of hospitals for source data.
    target_hospitals: list
        List of hospitals for target data.
    encounter_id: str
        Column name for encounter ids.
    hospital_id: str
        Column name for hospital ids.

    """
    
    ids_source = admin_data.loc[
        (
            (admin_data[hospital_id].isin(source))
        ),
        encounter_id,
    ]
    ids_target = admin_data.loc[
        (
            (admin_data[hospital_id].isin(target))
        ),
        encounter_id,
    ]
    X_s = X.loc[X.index.get_level_values(0).isin(ids_source)]
    X_t = X.loc[X.index.get_level_values(0).isin(ids_target)]
    return (X_s, X_t) 
