from sklearn.preprocessing import MinMaxScaler, StandardScaler, MaxAbsScaler, RobustScaler
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
import torch
from dataset import Data
from temporal_models import RNNModel, LSTMModel, GRUModel, LSTMCellModel

def get_device():
    if torch.cuda.is_available():
        return torch.device('cuda')
    else:
        return torch.device('cpu')
    
def format_dataset(X, level="features",imputation_method="simple"):   
    """
    Clean the data into machine-learnable matrices.
    Inputs:
        X (pd.DataFrame): a multiindex dataframe of GEMINI data
        level (string or bytes): the level of the column index to use as the features level
        imputation_method (string): the method for imputing the data, either forward, or simple
        Target (string): the heading to select from the labels in the outcomes_df, i.e. er LOS or mortality
    Returns:
        X (pd.DataFrame): the X data to input to the model
        y (array): the labels for the corresponding X-data
    """
    scaler = None

    # Imputation
    if imputation_method=='forward':
        X_imputed=impute_forward(X,"timestep")
    elif imputation_method=='simple':
        X_imputed=impute_simple(X,"timestep")
    else:
        X_imputed=X.unstack().sort_index(axis=1).copy()

    #add categorical/demographic data
###    #make both dataframes have the same number of column levels
###    while  len(y_df.columns.names)!=len(imputed_df.columns.names):
###        if len(y_df.columns.names)>len(imputed_df.columns.names):
###            y_df.columns=y_df.columns.droplevel(0)
###        elif len(y_df.columns.names)<len(imputed_df.columns.names):
###            raise Exception("number of y_df columns is less than the number of imputed_df columns")
###            y_df=pd.concat([y_df], names=['Firstlevel'])
###    # make both dataframes have the same column names
###    y_df.columns.names=imputed_df.columns.names

    # Stack columns
    X_imputed=X_imputed.stack(level='timestep', dropna=False)

    # Remove columns with no variability
    keep_cols=X_imputed.columns.tolist()
    keep_cols=[col for col in keep_cols if len(X_imputed[col].unique())!=1]
    X_imputed=X_imputed[keep_cols]

    # Unstack the timestep and sort them to the same as the original dataframe
    X_imputed=X_imputed.unstack()
    if imputation_method:
        X_imputed.columns=X_imputed.columns.swaplevel('timestep', 'simple_impute')

    # Scale
    df_means=X_imputed.mean(skipna=True, axis=0)
    df_stds=X_imputed.std(skipna=True, axis=0)
    df_stds.loc[df_stds.values==0, :]=df_means.loc[df_stds.values==0, :] # If std dev is 0, replace with mean
    scaler=(df_means, df_stds)  
    X_scaled=(X_imputed[X_imputed.columns]-df_means)/df_stds
    X_final=pd.DataFrame(X_scaled, index=X_imputed.index.tolist(), columns=X_imputed.columns.tolist())

    #keep this df for adding categorical data after
###     demo_df=y_df.copy()
###     demo_df.columns=demo_df.columns.droplevel(demo_df.columns.names.index('timestep'))

    X_final=X_final.stack(level='timestep', dropna=False) 
### X_final=X_final.join(demo_df.loc[:, ['M', 'F']], how='inner', on=['encounter_id'])

    X_final.index.names=['encounter_id', 'timestep']
    X_final[X_final.columns]=X_final[X_final.columns].values.astype(np.float32)
### gender=X_final.loc[(slice(None), slice(None), 0), 'F'].values.ravel()

    if imputation_method:
        X_final = flatten_to_sequence(X_final)
    
    X_final = np.swapaxes(X_final,1,2)
    return(X_final)

def flatten_to_sequence(X):
    """
    Turn pandas dataframe into sequence
    Inputs:
        X (pd.DataFrame): a multiindex dataframe of MIMICIII data
        vect (tuple) (optional): (timesteps, non-time-varying columns, time-varying columns, vectorizer(dict)(a mapping between an item and its index))
    Returns:
        X_seq (np.ndarray): Input 3-dimensional input data with the [n_samples, n_features, n_hours]
    """
    timestep_in_values=X.index.get_level_values(X.index.names.index('timestep'))

    output=np.dstack((X.loc[(slice(None), i), :].values for i in sorted(set(timestep_in_values))))

    return output

def forward_imputer(df):
    imputed_df=df.fillna(method='ffill').unstack().fillna(0)
    imputed_df.sort_index(axis=1, inplace=True)
    return(df)

def simple_imputer(df,train_subj):
    idx = pd.IndexSlice
    df = df.copy()
    
    df_out = df.loc[:, idx[:, ['mean', 'count']]]
    icustay_means = df_out.loc[:, idx[:, 'mean']].groupby(ID_COLS).mean()
    global_means = df_out.loc[idx[train_subj,:], idx[:, 'mean']].mean(axis=0)
    
    df_out.loc[:,idx[:,'mean']] = df_out.loc[:,idx[:,'mean']].groupby(ID_COLS).fillna(
        method='ffill'
    ).groupby(ID_COLS).fillna(icustay_means).fillna(global_means)
    
    df_out.loc[:, idx[:, 'count']] = (df.loc[:, idx[:, 'count']] > 0).astype(float)
    df_out.rename(columns={'count': 'mask'}, level='Aggregation Function', inplace=True)
    
    is_absent = (1 - df_out.loc[:, idx[:, 'mask']])
    hours_of_absence = is_absent.cumsum()
    time_since_measured = hours_of_absence - hours_of_absence[is_absent==0].fillna(method='ffill')
    time_since_measured.rename(columns={'mask': 'time_since_measured'}, level='Aggregation Function', inplace=True)

    df_out = pd.concat((df_out, time_since_measured), axis=1)
    df_out.loc[:, idx[:, 'time_since_measured']] = df_out.loc[:, idx[:, 'time_since_measured']].fillna(100)
    
    df_out.sort_index(axis=1, inplace=True)
    return df_out

def get_data(X,y):
    """Convert pandas dataframe to dataset.

    Parameters
    ----------
    X: numpy matrix
        Data containing features in the form of [samples, timesteps, features].
    y: list
        List of labels.

    """
    inputs = torch.tensor(X,dtype=torch.float32)
    target = torch.tensor(y,dtype=torch.float32)
    return Data(inputs, target)

def get_scaler(scaler):
    """Get scaler.

    Parameters
    ----------
    scaler: string
        String indicating which scaler to retrieve.

    """ 
    scalers = {
        "minmax": MinMaxScaler,
        "standard": StandardScaler,
        "maxabs": MaxAbsScaler,
        "robust": RobustScaler,
    }
    return scalers.get(scaler.lower())()

def get_temporal_model(model, model_params):
    """Get temporal model.

    Parameters
    ----------
    model: string
        String with model name (e.g. rnn, lstm, gru).

    """
    models = {
        "rnn": RNNModel,
        "lstm": LSTMModel,
        "gru": GRUModel,
        "lstmcell": LSTMCellModel
    }
    return models.get(model.lower())(**model_params)

def format_predictions(predictions, values, tags, df_test):
    vals = np.concatenate(values, axis=0).ravel()
    preds = np.concatenate(predictions, axis=0).ravel()
    tags = np.concatenate(tags, axis=0).ravel()
    df_result = pd.DataFrame(data={"value": vals, "prediction": preds, "tag": tags}, index=df_test.head(len(vals)).index)
    df_result = df_result.sort_index()
    return df_result

def impute_simple(df, time_index=None):
    """
    Concatenate the forward filled value, the mask of the measurement, and the time of the last measurement
    refer to paper
    Z. Che, S. Purushotham, K. Cho, D. Sontag, and Y. Liu, "Recurrent Neural Networks for Multivariate Time Series with Missing Values," Scientific Reports, vol. 8, no. 1, p. 6085, Apr 2018.
    Input:
        df (pandas.DataFrame): the dataframe with timeseries data in the index.
        time_index (string, optional): the heading name for the time-series index.
    Returns:
        df (pandas.DataFrame): a dataframe according to the simple impute algorithm described in the paper.
    """

    ID_COLS = ['encounter_id']
    
    #masked data
    masked_df=pd.isna(df)
    masked_df=masked_df.apply(pd.to_numeric)

    #time since last measurement
    index_of_time=list(df.index.names).index(time_index)
    time_in=[item[index_of_time] for item in df.index.tolist()]
    time_df=df.copy()
    for col in time_df.columns.tolist():
        time_df[col]=time_in
    time_df[masked_df]=np.nan

    #concatenate the dataframes
    df_prime=pd.concat([df,masked_df, time_df],axis=1,keys=['measurement','mask', 'time'])
    df_prime.columns=df_prime.columns.rename("simple_impute", level=0)#rename the column level

    #fill each dataframe using either ffill or mean
    df_prime=df_prime.fillna(method='ffill').unstack().fillna(0)

    #swap the levels so that the simple imputation feature is the lowest value
    col_level_names=list(df_prime.columns.names)
    col_level_names.append(col_level_names.pop(0))

    df_prime=df_prime.reorder_levels(col_level_names, axis=1)
    df_prime.sort_index(axis=1, inplace=True)

    return  df_prime
