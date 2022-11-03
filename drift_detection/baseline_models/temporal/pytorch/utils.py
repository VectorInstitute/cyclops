"""Utilities for baseline temporal pytorch models."""
import numpy as np
import pandas as pd
import torch

from .dataset import Data
from .models import GRUModel, LSTMModel, RNNModel

# impute_forward does not exist
# from metrics import impute_forward


def load_ckp(checkpoint_fpath, model):
    """Load checkpoint."""
    checkpoint = torch.load(checkpoint_fpath)
    model.load_state_dict(checkpoint["model"])
    optimizer = checkpoint["optimizer"]
    return model, optimizer, checkpoint["n_epochs"]


def reshape_2d_to_3d(data, num_timesteps):
    """Reshape 2D data to 3D data."""
    data = data.unstack()
    num_encounters = data.shape[0]
    data = data.values.reshape((num_encounters, num_timesteps, -1))
    return data


def get_device():
    """Get device."""
    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    return device


def format_dataset(X, imputation_method="simple"):
    """Clean the data into machine-learnable matrices.

    Parameters
    ----------
        X (pd.DataFrame)
            a multiindex dataframe of GEMINI data
            level (string or bytes): the level of the column index
            to use as the features level
            imputation_method (string): the method for imputing
            the data, either forward, or simple
        Target (string)
            the heading to select from the labels in the outcomes_df,
            i.e. er LOS or mortality

    Returns
    -------
        X (pd.DataFrame)
            the X data to input to the model
        y (array)
            the labels for the corresponding X-data

    """
    # Imputation
    if imputation_method == "forward":
        # X_imputed = impute_forward(X, "timestep")
        raise NotImplementedError("impute_forward does not exist.")
    if imputation_method == "simple":
        X_imputed = impute_simple(X, "timestep")
    else:
        X_imputed = X.unstack().sort_index(axis=1).copy()

    # add categorical/demographic data
    # make both dataframes have the same number of column levels
    # while  len(y_df.columns.names)!=len(imputed_df.columns.names):
    # if len(y_df.columns.names)>len(imputed_df.columns.names):
    # y_df.columns=y_df.columns.droplevel(0)
    # elif len(y_df.columns.names)<len(imputed_df.columns.names):
    #            raise Exception("number of y_df columns is \
    #                             less than the number of \
    #                             imputed_df columns")
    #            y_df=pd.concat([y_df], names=['Firstlevel'])
    # make both dataframes have the same column names
    # y_df.columns.names=imputed_df.columns.names

    # Stack columns
    X_imputed = X_imputed.stack(level="timestep", dropna=False)

    # Remove columns with no variability
    keep_cols = X_imputed.columns.tolist()
    keep_cols = [col for col in keep_cols if len(X_imputed[col].unique()) != 1]
    X_imputed = X_imputed[keep_cols]

    # Unstack the timestep and sort them to the same as the original dataframe
    X_imputed = X_imputed.unstack()
    if imputation_method:
        X_imputed.columns = X_imputed.columns.swaplevel("timestep", "simple_impute")

    # Scale
    df_means = X_imputed.mean(skipna=True, axis=0)
    df_stds = X_imputed.std(skipna=True, axis=0)
    df_stds.loc[df_stds.values == 0, :] = df_means.loc[
        df_stds.values == 0, :
    ]  # If std dev is 0, replace with mean
    X_scaled = (X_imputed[X_imputed.columns] - df_means) / df_stds
    X_final = pd.DataFrame(
        X_scaled, index=X_imputed.index.tolist(), columns=X_imputed.columns.tolist()
    )

    # keep this df for adding categorical data after
    # demo_df=y_df.copy()
    # demo_df.columns=demo_df.columns.droplevel(demo_df.columns.names.index('timestep'))

    X_final = X_final.stack(level="timestep", dropna=False)
    # X_final=X_final.join(demo_df.loc[:, ['M', 'F']],
    #                      how='inner', on=['encounter_id'])

    X_final.index.names = ["encounter_id", "timestep"]
    X_final[X_final.columns] = X_final[X_final.columns].values.astype(np.float32)
    # gender=X_final.loc[(slice(None), slice(None), 0), 'F'].values.ravel()

    if imputation_method:
        X_final = flatten_to_sequence(X_final)

    X_final = np.swapaxes(X_final, 1, 2)
    return X_final


def flatten_to_sequence(X):
    """Turn pandas dataframe into sequence.

    Parameters
    ----------
        X (pd.DataFrame)
            a multiindex dataframe of GEMINI data
        vect (tuple) (optional): (timesteps, non-time-varying columns,
        time-varying columns, vectorizer(dict)(a mapping between an item and its index))

    Returns
    -------
        X_seq (np.ndarray)
            Input 3-dimensional input data
            with the shape [n_samples, n_features, n_hours]

    """
    timestep_in_values = X.index.get_level_values(X.index.names.index("timestep"))

    output = np.dstack(
        (X.loc[(slice(None), i), :].values for i in sorted(set(timestep_in_values)))
    )

    return output


def forward_imputer(df):
    """Forward impute the data."""
    imputed_df = df.fillna(method="ffill").unstack().fillna(0)
    imputed_df.sort_index(axis=1, inplace=True)
    return df


def get_data(X, y):
    """Convert pandas dataframe to dataset.

    Parameters
    ----------
    X: numpy matrix
        Data containing features in the form of [samples, timesteps, features].
    y: list
        List of labels.

    """
    inputs = torch.tensor(X, dtype=torch.float32)
    target = torch.tensor(y, dtype=torch.float32)
    return Data(inputs, target)


def process_outcome(outcome, static):
    """Process the outcome data."""
    if outcome == "mortality":
        static["mortality_derived"] = np.where(
            static["discharge_disposition"].isin([7, 66, 72, 73]), 1, 0
        )
    elif outcome == "length_of_stay_in_er":
        mort1 = (static[outcome] >= 0) & (static[outcome] < 7)
        mort2 = (static[outcome] >= 7) & (static[outcome] < 14)
        mort3 = (static[outcome] >= 14) & (static[outcome] < 30)

        vals = [1, 2, 3]
        default = 4
        static["los_er_derived"] = np.select(
            [mort1, mort2, mort3], vals, default=default
        )
    return static


def get_temporal_model(model, model_params):
    """Get temporal model.

    Parameters
    ----------
    model: string
        String with model name (e.g. rnn, lstm, gru).

    """
    models = {"rnn": RNNModel, "lstm": LSTMModel, "gru": GRUModel}
    return models.get(model.lower())(**model_params)


def impute_simple(df, time_index=None):
    """Imputation of data with simple schema.

    Concatenates the forward filled value, the mask of the measurement,
    and the time of the last measurement refer to paper below.

    Che, S. Purushotham, K. Cho, D. Sontag, and Y. Liu,
    "Recurrent Neural Networks for Multivariate Time Series with Missing Values,"
    Scientific Reports, vol. 8, no. 1, p. 6085, Apr 2018.

    Parameters
    ----------
        df (pandas.DataFrame)
            the dataframe with timeseries data in the index.
        time_index (string, optional)
            the heading name for the time-series index.

    Returns
    -------
        df (pandas.DataFrame)
            a dataframe according to the simple impute
            algorithm described in the paper.

    """
    # masked data
    masked_df = pd.isna(df)
    masked_df = masked_df.apply(pd.to_numeric)

    # time since last measurement
    index_of_time = list(df.index.names).index(time_index)
    time_in = [item[index_of_time] for item in df.index.tolist()]
    time_df = df.copy()
    for col in time_df.columns.tolist():
        time_df[col] = time_in
    time_df[masked_df] = np.nan

    # concatenate the dataframes
    df_prime = pd.concat(
        [df, masked_df, time_df], axis=1, keys=["measurement", "mask", "time"]
    )
    df_prime.columns = df_prime.columns.rename(
        "simple_impute", level=0
    )  # rename the column level

    # fill each dataframe using either ffill or mean
    df_prime = df_prime.fillna(method="ffill").unstack().fillna(0)

    # swap the levels so that the simple imputation feature is the lowest value
    col_level_names = list(df_prime.columns.names)
    col_level_names.append(col_level_names.pop(0))

    df_prime = df_prime.reorder_levels(col_level_names, axis=1)
    df_prime.sort_index(axis=1, inplace=True)

    return df_prime
