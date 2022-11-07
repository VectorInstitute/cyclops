"""Clinical Shift Applicator module."""
import datetime

# import numpy as np
# from .utils import get_args


class ClinicalShiftApplicator:
    """The ClinicalShiftApplicator class is used induce synthetic clinical shifts.

    Parameters
    ----------
    shift_type: str
        Method used to induce shift in data.
        Options include: "seasonal", "hospital_type".

    """

    def __init__(self, shift_type: str):

        self.shift_type = shift_type

        self.shift_types = {
            "source_target": source_target,
            "month": month,
            "hospital_type": hospital_type,
            "time": time,
        }

        if self.shift_type not in self.shift_types:
            raise ValueError(f"Shift type {self.shift_type} not supported. ")

    def apply_shift(self, X):
        """apply_shift.

        Returns
        -------
        X: pd.DataFrame
            Data to apply shift to.
        y:
            Outcome labels.
        admin_data: pd.DataFrame
            Dataframe containing admin variables to filter on
            (e.g. "hospital_id", "admit_timestamp").

        """
        # list(X.index.get_level_values(0).unique())
        X_s, X_t = X, X
        # X = X[np.in1d(X.index.get_level_values(0), admin_data["encounter_id"])]

        # X_s, X_t = self.shift_types[self.shift_type](
        #     X, admin_data, **get_args(self.shift_types[self.shift_type], kwargs)
        # )

        # list(X.index.get_level_values(0).unique())

        return (X_s, X_t)


def source_target(
    X,
    admin_data,
    train_frac: float = 0.5,
    encounter_id="encounter_id",
    admit_timestamp="admit_timestamp",
):
    """Shift in time across source and target data.

    Parameters
    ----------
    X: pd.DataFrame
        Data to apply shift to.
    admin_data: pd.DataFrame
        Dataframe containing admin variables to filter on
        (e.g. "hospital_id", "admit_timestamp").
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
    dataset_ids = admin_data.loc[
        (
            (admin_data[admit_timestamp].dt.date > datetime.date(2015, 1, 1))
            & (admin_data[admit_timestamp].dt.date < datetime.date(2019, 1, 1)),
        ),
        encounter_id,
    ]
    X = X.loc[X.index.get_level_values(0).isin(dataset_ids)]
    num_train = int(train_frac * len(dataset_ids))
    ids_source = dataset_ids[0:num_train]
    ids_target = dataset_ids[num_train:]

    X_s = X.loc[X.index.get_level_values(0).isin(ids_source)]
    X_t = X.loc[X.index.get_level_values(0).isin(ids_target)]
    return (X_s, X_t)


def time(
    X,
    admin_data,
    source,
    target,
    admit_timestamp="admit_timestamp",
    encounter_id="encounter_id",
):
    """Shift in time.

    Parameters
    ----------
    X: pd.DataFrame
        Data to apply shift to.
    admin_data: pd.DataFrame
        Dataframe containing admin variables to filter on
        (e.g. "hospital_id", "admit_timestamp").
    source: list[datetime.date]
        Start and end of source data.
    target: list[datetime.date]
        Start and end of target data.
    encounter_id: str
        Column name for encounter ids.
    admit_timestamp: str
        Column name for admission timestamps.

    """
    ids_source = admin_data.loc[
        (
            (admin_data[admit_timestamp].dt.date > source[0])
            & (admin_data[admit_timestamp].dt.date < source[1]),
        ),
        encounter_id,
    ]
    ids_target = admin_data.loc[
        (
            (admin_data[admit_timestamp].dt.date > target[0])
            & (admin_data[admit_timestamp].dt.date < target[1]),
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
    encounter_id="encounter_id",
    admit_timestamp="admit_timestamp",
):
    """Shift for selection of months.

    Parameters
    ----------
    X: pd.DataFrame
        Data to apply shift to.
    admin_data: pd.DataFrame
        Dataframe containing admin variables to filter on
        (e.g. "hospital_id", "admit_timestamp").
    encounter_id: str
        Column name for encounter ids.
    admit_timestamp: str
        Column name for admission timestamps.

    """
    ids_source = admin_data.loc[
        ((admin_data[admit_timestamp].dt.month.isin(source))),
        encounter_id,
    ]
    ids_target = admin_data.loc[
        ((admin_data[admit_timestamp].dt.month.isin(target))),
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
    encounter_id="encounter_id",
    hospital_id="hospital_id",
):
    """Shift against hospital type.

    Parameters
    ----------
    X: pd.DataFrame
        Data to apply shift to.
    admin_data: pd.DataFrame
        Dataframe containing admin variables to filter on
        (e.g. "hospital_id", "admit_timestamp").
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
        ((admin_data[hospital_id].isin(source))),
        encounter_id,
    ]
    ids_target = admin_data.loc[
        ((admin_data[hospital_id].isin(target))),
        encounter_id,
    ]
    X_s = X.loc[X.index.get_level_values(0).isin(ids_source)]
    X_t = X.loc[X.index.get_level_values(0).isin(ids_target)]
    return (X_s, X_t)
