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
        
        self.shift_types = { 
            "summer": summer, 
            "winter": winter, 
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
        admin_data, 
        X, 
        y, 
        **kwargs
    ):

        """apply_shift.

        Returns
        ----------
        admin_data: pd.DataFrame
            Dataframe containing admin variables to filter on (e.g. "hospital_id", "admit_timestamp").
        X: numpy.matrix
            Data to apply shift to.
        y: 
            Outcome labels.
        """
        
        
        X_s, X_t = self.shift_types[self.shift_type](
                admin_data, X, **get_args(self.shift_types[self.shift_type], kwargs)
            )
        
        encounter_ids = list(X.index.get_level_values(0).unique())
        y_s = y[np.in1d(encounter_ids, X_s.index.get_level_values(0).unique())]
        y_t = y[np.in1d(encounter_ids, X_t.index.get_level_values(0).unique())]

        assert len(X_s.index.get_level_values(0).unique()) == len(y_s)
        assert len(X_t.index.get_level_values(0).unique()) == len(y_t)
        
        return (X_s, y_s, X_t, y_t)  

            
def time(
    admin_data, 
    X, 
    start_date,
    cutoff_date,
    end_date,
    admit_timestamp='admit_timestamp', 
    encounter_id='encounter_id'
):
    """time.

    Parameters
    ----------
    admin_data: pd.DataFrame
        Dataframe containing admin variables to filter on (e.g. "hospital_id", "admit_timestamp").
    X: numpy.ndarray
        Data to apply shift to.
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

def winter(
    admin_data, 
    X, 
    encounter_id='encounter_id', 
    admit_timestamp='admit_timestamp'
):
    """winter.

    Parameters
    ----------
    admin_data: pd.DataFrame
        Dataframe containing admin variables to filter on (e.g. "hospital_id", "admit_timestamp").
    X: numpy.matrix
        Data to apply shift to.
    encounter_id: str
        Column name for encounter ids.
    admit_timestamp: str
        Column name for admission timestamps.

    """
    ids_source = admin_data.loc[
        (
            (admin_data[admit_timestamp].dt.month.isin([11, 12, 1, 2]))
        ),
            encounter_id,
    ]
    ids_target = admin_data.loc[
        (
            (admin_data[admit_timestamp].dt.month.isin([3, 4, 5, 6, 7, 8, 9, 10]))
        ),
        encounter_id,
    ]
    X_s = X.loc[X.index.get_level_values(0).isin(ids_source)]
    X_t = X.loc[X.index.get_level_values(0).isin(ids_target)]
    return (X_s, X_t)

def summer(
    admin_data, 
    X, 
    encounter_id='encounter_id', 
    admit_timestamp='admit_timestamp'
):
    """summer.

    Parameters
    ----------
    admin_data: pd.DataFrame
        Dataframe containing admin variables to filter on (e.g. "hospital_id", "admit_timestamp").
    X: numpy.matrix
        Data to apply shift to.
    encounter_id: str
        Column name for encounter ids.
    admit_timestamp: str
        Column name for admission timestamps.

    """
    ids_source = admin_data.loc[
        (
            (admin_data[admit_timestamp].dt.month.isin([6, 7, 8, 9]))
        ),
        encounter_id,
    ]
    ids_target = admin_data.loc[
        (
            (admin_data[admit_timestamp].dt.month.isin([1, 2, 3, 4, 5, 10, 11, 12]))
        ),
        encounter_id,
    ]
    X_s = X.loc[X.index.get_level_values(0).isin(ids_source)]
    X_t = X.loc[X.index.get_level_values(0).isin(ids_target)]
    return (X_s, X_t)

def hospital_type(
    admin_data, 
    X, 
    source_hospitals, 
    target_hospitals, 
    encounter_id='encounter_id', 
    hospital_id='hospital_id'
):
    
    """hospital_type.

    Parameters
    ----------
    admin_data: pd.DataFrame
        Dataframe containing admin variables to filter on (e.g. "hospital_id", "admit_timestamp").
    X: numpy.matrix
        Data to apply shift to.
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
            (admin_data[hospital_id].isin(source_hospitals))
        ),
        encounter_id,
    ]
    ids_target = admin_data.loc[
        (
            (admin_data[hospital_id].isin(target_hospitals))
        ),
        encounter_id,
    ]
    X_s = X.loc[X.index.get_level_values(0).isin(ids_source)]
    X_t = X.loc[X.index.get_level_values(0).isin(ids_target)]
    return (X_s, X_t)
