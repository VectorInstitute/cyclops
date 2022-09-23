import numpy as np
import os
import random
import sys
import pandas as pd
from datetime import date, timedelta
import inspect
import torch
import torch.nn as nn
import pickle
from scipy.special import softmax
from typing import Callable, Dict, List, Optional, Tuple, Union, Any
from alibi_detect.utils.pytorch.kernels import GaussianRBF
from alibi_detect.cd import ContextMMDDrift, LearnedKernelDrift
from alibi_detect.utils.pytorch.kernels import DeepKernel
from drift_detection.baseline_models.temporal.pytorch.utils import get_temporal_model, get_device


def get_args(obj, kwargs):
    '''
    Get valid arguments from kwargs to pass to object.
    
    Parameters
    ----------
    obj
        object to get arguments from.
    kwargs  
        Dictionary of arguments to pass to object.
    
    Returns
    -------
    args
        Dictionary of valid arguments to pass to class object.
    '''
    args = {}
    for key in kwargs:
        if inspect.isclass(obj):
            if key in obj.__init__.__code__.co_varnames:
                args[key] = kwargs[key]
        elif inspect.ismethod(obj) or inspect.isfunction(obj):
            if key in obj.__code__.co_varnames:
                args[key] = kwargs[key]
    return args

def load_model(self, model_path: str):
    '''Load pre-trained model from path.
    For scikit-learn models, a pickle is loaded from disk.
    For the torch models, the "state_dict" is loaded from disk.
    
    Returns
    -------
    model
        loaded pre-trained model
    '''
    file_type = self.model_path.split(".")[-1]
    if file_type == "pkl" or file_type == "pickle":
        model = pickle.load(open(model_path, "rb"))
    elif file_type == "pt":
        model = torch.load(self.model_path)
    return model


def save_model(self, model, output_path: str):
    '''Saves the model to disk.
    For scikit-learn models, a pickle is saved to disk.
    For the torch models, the "state_dict" is saved to disk.
    
    Parameters
    ----------
    output_path: String
        path to save the model to
    '''
    file_type = output_path.split(".")[-1]
    if file_type == "pkl" or file_type == "pickle":
        pickle.dump(model, open(output_path, "wb"))
    elif file_type == "pt":
        torch.save(model.state_dict(), output_path)


class ContextMMDWrapper:
    '''
    Wrapper for ContextMMDDrift
    
    Parameters
    ----------
     
    
    '''
    def __init__(self, X_s, backend= 'tensorflow', p_val = 0.05, preprocess_x_ref = True, update_ref = None, preprocess_fn = None, 
    x_kernel = None, c_kernel = None, n_permutations= 1000, prop_c_held = 0.25, n_folds = 5, batch_size = 256, device = None, 
    input_shape = None, data_type = None, verbose = False, context_type='rnn', model_path=None):

        self.context_type = context_type
        self.model_path = model_path
        C_s = context(X_s, self.context_type, self.model_path)
        self.tester = ContextMMDDrift(X_s, C_s)

    def predict(self, X_t, **kwargs):
        
        C_t = context(X_t, self.context_type, self.model_path)
        return self.tester.predict(X_t, C_t, **get_args(self.tester.predict, kwargs))

class LKWrapper:
    '''
    Wrapper for LKWrapper
    
    Parameters
    ----------
     
    
    '''
    def __init__(self, X_s, *, backend = 'tensorflow', p_val = 0.05, preprocess_x_ref = True, update_x_ref = None, 
    preprocess_fn = None, n_permutations = 100, var_reg = 0.00001, reg_loss_fn = lambda kernel: 0, train_size = 0.75,
    retrain_from_scratch = True, optimizer = None, learning_rate = 0.001, batch_size = 32, preprocess_batch = None, 
    epochs = 3, verbose = 0, train_kwargs = None, device = None, dataset = None, dataloader = None, data_type = None, 
    kernel_a = GaussianRBF(trainable=True), kernel_b = GaussianRBF(trainable=True), eps = 'trainable', proj_type = 'ffnn'):

        self.proj = self.choose_proj(X_s, proj_type)

        kernel = DeepKernel(self.proj, kernel_a, kernel_b, eps)

        kwargs = locals()
        args = [kwargs['backend'], kwargs['p_val'], kwargs['preprocess_x_ref'], kwargs['update_x_ref'], kwargs['preprocess_fn'],
        kwargs['n_permutations'], kwargs['var_reg'], kwargs['reg_loss_fn'], kwargs['train_size'], kwargs['retrain_from_scratch'],
        kwargs['optimizer'], kwargs['learning_rate'], kwargs['batch_size'], kwargs['preprocess_batch'], kwargs['epochs'], 
        kwargs['verbose'], kwargs['train_kwargs'], kwargs['device'], kwargs['dataset'], kwargs['dataloader'], kwargs['data_type']]
        self.tester = LearnedKernelDrift(X_s, kernel, *args)


    def predict(self, X_t, **kwargs):
        return self.tester.predict(X_t, **get_args(self.tester.predict, kwargs))

    def choose_proj(self, X_s, proj_type):
        if proj_type == 'rnn':
            return recurrent_neural_network("lstm", X_s.shape[-1])
        elif proj_type == 'ffnn':
            return feed_forward_neural_network(X_s.shape[-1])
        elif proj_type == 'cnn':
            return convolutional_neural_network(X_s.shape[-1])
        else:
            raise ValueError("Invalid projection type.")

def context(x: pd.DataFrame, context_type='rnn', model_path=None):
    '''
    Get context for context mmd drift detection.
    
    Parameters
    ----------
    x
        data to build context for context mmd drift detection
    
    '''
    device = get_device()

    if context_type == "rnn":
        model = recurrent_neural_network("lstm", x.shape[-1])
        model.load_state_dict(load_model(model_path))
        model.eval()
        with torch.no_grad():
            logits = model(torch.from_numpy(x).to(device)).cpu().numpy()
        return softmax(logits, -1)
    elif context_type == "gmm":
        gmm = load_model(model_path)
        c_gmm_proba = gmm.predict_proba(x) 
        return c_gmm_proba


def recurrent_neural_network(model_name: str, input_dim: int, hidden_dim = 64, layer_dim = 2, dropout = 0.2, output_dim = 1, last_timestep_only = False):
    '''
    Creates a recurrent neural network model.

    Parameters
    ----------
    model_name
        type of rnn model, one of: "rnn", "lstm", "gru"
    input_dim
        number of features
        
    Returns
    -------
    model: torch.nn.Module
        recurrent neural network model.
    '''
    model_params = {
        "device": get_device(),
        "input_dim": input_dim,
        "hidden_dim": hidden_dim,
        "layer_dim": layer_dim,
        "output_dim": output_dim,
        "dropout_prob": dropout,
        "last_timestep_only": last_timestep_only,
    }
    model = get_temporal_model(model_name, model_params)
    return model


def feed_forward_neural_network(input_dim: int):
    '''
    Creates a feed forward neural network model.

    Parameters
    ----------
    input_dim
        number of features

    Returns
    -------
    model: torch.nn.Module
        feed forward neural network model.
    '''
    ffnn = nn.Sequential(
            nn.Linear(input_dim, 16),
            nn.SiLU(),
            nn.Linear(16, 8),
            nn.SiLU(),
            nn.Linear(8, 1),
    )
    return ffnn


def convolutional_neural_network(input_dim: int):
    '''
    Creates a convolutional neural network model.
    
    Parameters
    ----------
    input_dim
        number of features
    
    Returns
    -------
    torch.nn.Module
        convolutional neural network.
    '''
    cnn = nn.Sequential(
            nn.Conv2d(input_dim, 4, stride=2, padding=0),
            nn.ReLU(),
            nn.Conv2d(8, 16, 4, stride=2, padding=0),
            nn.ReLU(),
            nn.Conv2d(16, 32, 4, stride=2, padding=0),
            nn.ReLU(),
            nn.Flatten(),
    )
    return cnn
        
def scale_temporal(x: pd.DataFrame):
    '''
    Scale columns of temporal dataframe.
    
    Returns
    -------
    model: torch.nn.Module
        feed forward neural network model.
    '''
    numerical_cols = [col for col in x[numerical_cols] 
             if not np.isin(x[col].dropna().unique(), [0, 1]).all()]

    for col in numerical_cols:
        scaler = StandardScaler().fit(x[col].values.reshape(-1, 1))
        x[col] = pd.Series(
            np.squeeze(scaler.transform(x[col].values.reshape(-1, 1))),
            index=x[col].index,
        )
        
    return(x)

def daterange(start_date, end_date, stride: int, window: int):
    '''
    Outputs a range of dates after applying a shift of a given stride and window adjustment.
    
    Returns
    -------
    datetime.date
        range of dates after stride and window adjustment.
    '''
    for n in range(int((end_date - start_date).days)):
        if start_date + timedelta(n*stride+window) < end_date:
            yield start_date+ timedelta(n*stride)  
            
def get_serving_data(X, y, admin_data, start_date, end_date, stride=1, window=1, ids_to_exclude=None, encounter_id='encounter_id', admit_timestamp='admit_timestamp'):
    '''
    Transforms a static set of patient encounters with timestamps into serving data that ranges from a given start date and goes until a given end date with a constant window and stride length.
    
    Returns
    -------
    dictionary
        dictionary containing keys timestamp, X and y
    '''
    
    target_stream_X = []
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
        encounter_ids = X.index.get_level_values(0).unique()
        X_inwindow = X.loc[X.index.get_level_values(0).isin(encounters_inwindow)]
        y_inwindow = pd.DataFrame(y[np.in1d(encounter_ids, encounters_inwindow)])
        if not X_inwindow.empty:
            target_stream_X.append(X_inwindow)
            target_stream_y.append(y_inwindow)
            timestamps.append((single_date+timedelta(days=window)).strftime("%Y-%m-%d"))
    target_data = { 'timestamps': timestamps,
                    'X': target_stream_X,
                    'y': target_stream_y 
                  }                 
    return(target_data)