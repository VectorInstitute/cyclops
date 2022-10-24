"""Common use-case constants."""

import numpy as np

READMISSION_MAP = {
    "planned_from_acute": [1, "1", "Yes"],
    "unplanned_7_day_acute": [2, "2"],
    "unplanned_8_to_28_day_acute": [3, "3"],
    "unplanned_7_day_day_surg": [4, "4"],
    "new_to_acute": [5, "5"],
    "nota": [9, "9", "No"],
    "no_info": np.nan,
}

ADMIT_VIA_AMBULANCE_MAP = {
    "ground": ["G", "GROUND", "GROUND AMBULANCE", "Y"],
    "no_ambulance": ["N", "No Ambulance"],
    "air": ["A", "C", "COMBINED", "COMBINATION OF AMBULANCES - Includes Air Ambulance"],
    "no_info": np.nan,
}

TRIAGE_LEVEL_MAP = {
    "resuscitation": ["1", "L1", "RESUSCITATION"],
    "emergent": ["2", "L2", "EMERGENCY"],
    "urgent": ["3", "L3", "URGENT"],
    "semi-urgent": ["4", "L4", "SEMI-URGENT"],
    "non-urgent": ["5", "NON-URGENT"],
    "no_info": ["9", np.nan],
}
