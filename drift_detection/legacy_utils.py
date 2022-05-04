import re
from collections import OrderedDict
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import sqlalchemy
from sqlalchemy import func, select, desc
from sqlalchemy.sql.expression import and_, or_

import sys
from functools import reduce
import datetime
import joblib
import matplotlib.pyplot as plt
import os
import pickle

sys.path.append("../..")

from sklearn.metrics import (
    auc,
    accuracy_score,
    confusion_matrix,
    roc_auc_score,
    roc_curve,
    precision_recall_curve,
    average_precision_score,
)

from sklearn.model_selection import GridSearchCV, train_test_split
import pandas as pd
import numpy as np
import sqlalchemy
from sqlalchemy import select, func, extract, desc
from sqlalchemy.sql.expression import and_

from evidently import ColumnMapping
from evidently.dashboard import Dashboard
from evidently.dashboard.tabs import DataQualityTab
from evidently.model_profile import Profile
from evidently.model_profile.sections import DataQualityProfileSection

FEATURES = ['age', 'icd10_A00_B99','icd10_C00_D49', 'icd10_D50_D89', 'icd10_E00_E89', 'icd10_F01_F99',
       'icd10_G00_G99', 'icd10_H00_H59', 'icd10_H60_H95', 'icd10_I00_I99',
       'icd10_J00_J99', 'icd10_K00_K95', 'icd10_L00_L99', 'icd10_M00_M99',
       'icd10_N00_N99', 'icd10_O00_O99', 'icd10_Q00_Q99', 'icd10_R00_R99', 
       'icd10_S00_T88', 'icd10_U07_U08', 'icd10_Z00_Z99']
    
def import_dataset(outcome, features=FEATURES, shuffle=True):
    data = pd.read_csv('/mnt/nfs/project/delirium/data/data_2019.csv')
    data = data.loc[data['hospital_id'].isin([3])]
    data[features] = data[features].fillna(0)
    x = data[features].to_numpy()
    y = data[outcome].to_numpy()
    
    # Add 3-way split
    x_spl = np.array_split(x, 3)
    y_spl = np.array_split(y, 3)
    x_train = x_spl[0] ; x_val = x_spl[1] ; x_test = x_spl[2]
    y_train = y_spl[0] ; y_val = y_spl[1] ; y_test = x_spl[2]
    
    if shuffle:
        (x_train, y_train), (x_val, y_val) = random_shuffle_and_split(x_train, y_train, x_val, y_val, len(x_train))
        
    orig_dims = x_train.shape[1:]
    
    return (x_train, y_train), (x_val, y_val), (x_test, y_test), orig_dims


def random_shuffle_and_split(x_train, y_train, x_test, y_test, split_index):
    x = np.append(x_train, x_test, axis=0)
    y = np.append(y_train, y_test, axis=0)

    x, y = __unison_shuffled_copies(x, y)

    x_train = x[:split_index, :]
    x_test = x[split_index:, :]
    y_train = y[:split_index]
    y_test = y[split_index:]

    return (x_train, y_train), (x_test, y_test)

def __unison_shuffled_copies(a, b):
    assert len(a) == len(b)
    p = np.random.permutation(len(a))
    return a[p], b[p]

def import_dataset_year(outcome, year, features=None, shuffle=True,):
    data_s = pd.read_csv('/mnt/nfs/project/delirium/data/data_2019.csv')
    data_s = data_s.loc[data_s['hospital_id'].isin([3])]
    if year == "2019":
        data_t = pd.read_csv('/mnt/nfs/project/delirium/data/data_2019.csv')
    elif year == "2020":
        data_t = pd.read_csv('/mnt/nfs/project/delirium/data/data_2020.csv')
    data_t = data_t.loc[data_t['hospital_id'].isin([3])]
    data_s[features] = data_s[features].fillna(0)
    data_t[features] = data_t[features].fillna(0)
    
    x_train = data_s[features].to_numpy()
    y_train = data_s[outcome].to_numpy()
    
    x_test = data_t[features].to_numpy()
    y_test = data_t[outcome].to_numpy()  
    
    # Add 3-way split
    x_train_spl = np.split(x_train, 2)
    y_train_spl = np.split(y_train, 2)
    x_train = x_train_spl[0]
    x_val = x_train_spl[1]
    y_train = y_train_spl[0]
    y_val = y_train_spl[1]

    if shuffle:
        (x_train, y_train), (x_val, y_val) = random_shuffle_and_split(x_train, y_train, x_val, y_val, len(x_train))
        
    orig_dims = x_train.shape[1:]
    
    return (x_train, y_train), (x_val, y_val), (x_test, y_test), orig_dims

def import_dataset_hospital(outcome, train_hosp, test_hosp, features=None, shuffle=True):
    data = pd.read_csv('/mnt/nfs/project/delirium/data/data_2019.csv')
    data_s = data.loc[data['hospital_id'].isin(train_hosp)]
    data_t = data.loc[data['hospital_id'].isin(test_hosp)]
    data_s[features] = data_s[features].fillna(0)
    data_t[features] = data_t[features].fillna(0)
    
    x_train = data_s[features].to_numpy()
    y_train = data_s[outcome].to_numpy()
    
    x_test = data_t[features].to_numpy()
    y_test = data_t[outcome].to_numpy()  
    
    # Add 3-way split
    x_train_spl = np.split(x_train, 2)
    y_train_spl = np.split(y_train, 2)
    x_train = x_train_spl[0]
    x_val = x_train_spl[1]
    y_train = y_train_spl[0]
    y_val = y_train_spl[1]

    if shuffle:
        (x_train, y_train), (x_val, y_val) = random_shuffle_and_split(x_train, y_train, x_val, y_val, len(x_train))
        
    orig_dims = x_train.shape[1:]
    
    return (x_train, y_train), (x_val, y_val), (x_test, y_test), orig_dims

def run_year_shift_experiment(year, path, dr_technique, md_test, samples, dataset,sign_level,features=None, random_runs=5,calc_acc=True):
    # Stores p-values for all experiments of a shift class.
    samples_rands = np.ones((len(samples), random_runs)) * (-1)

    # Stores accuracy values for malignancy detection.
    val_accs = np.ones((random_runs, len(samples))) * (-1)
    te_accs = np.ones((random_runs, len(samples))) * (-1)
    dcl_accs = np.ones((len(samples), random_runs)) * (-1)
    
    # Average over a few random runs to quantify robustness.
    for rand_run in range(0,random_runs-1):
        rand_run = int(rand_run)
        rand_run_path = path + str(rand_run) + '/'
        # if not os.path.exists(rand_run_path):
        #     os.makedirs(rand_run_path)

        np.random.seed(rand_run)

        (X_tr, y_tr), (X_val, y_val), (X_t, y_t), orig_dims = import_dataset_year('los',year,features, shuffle=True)
        
        for si, sample in enumerate(samples):

            red_model = None
            print("Random Run %s : Sample %s" % (rand_run,sample))
            
            sample_path = rand_run_path + str(sample) + '/'

            # if not os.path.exists(sample_path):
            #     os.makedirs(sample_path)

            # Detect shift.

            shift_detector = ShiftDetector(dr_technique, md_test, sign_level, red_model, sample, dataset)
            p_val, dist, red_dim, red_model, val_acc, te_acc = shift_detector.detect_data_shift(X_tr, y_tr, X_val, y_val, X_t[:sample,:], y_t[:sample], orig_dims)
            
            val_accs[rand_run, si] = val_acc
            te_accs[rand_run, si] = te_acc

            print("Shift p-vals: ", p_val)

            samples_rands[si,rand_run] = p_val

    mean_p_vals = np.mean(samples_rands, axis=1)
    std_p_vals = np.std(samples_rands, axis=1)

    mean_val_accs = np.mean(val_accs, axis=0)
    std_val_accs = np.std(val_accs, axis=0)

    mean_te_accs = np.mean(te_accs, axis=0)
    std_te_accs = np.std(te_accs, axis=0)
        
    return(mean_p_vals, std_p_vals)


def run_hosp_shift_experiment(test_hosp, path, dr_technique, md_test, samples, dataset,sign_level,features=None,random_runs=5,calc_acc=True):
    # Stores p-values for all experiments of a shift class.
    samples_rands = np.ones((len(samples), random_runs)) * (-1)

    # Stores accuracy values for malignancy detection.
    val_accs = np.ones((random_runs, len(samples))) * (-1)
    te_accs = np.ones((random_runs, len(samples))) * (-1)
    dcl_accs = np.ones((len(samples), random_runs)) * (-1)
    
    # Average over a few random runs to quantify robustness.
    for rand_run in range(0,random_runs-1):
        rand_run = int(rand_run)
        rand_run_path = path + str(rand_run) + '/'
        # if not os.path.exists(rand_run_path):
        #     os.makedirs(rand_run_path)

        np.random.seed(rand_run)

        (X_tr, y_tr), (X_val, y_val), (X_t, y_t), orig_dims = import_dataset_hospital('los',[HOSPITAL_ID['SMH']], [HOSPITAL_ID[test_hosp]], features, shuffle=True)
                
        for si, sample in enumerate(samples):

            red_model = None
            print("Hospital %s: Random Run %s : Sample %s" % (str(test_hosp),rand_run,sample))
            
            sample_path = rand_run_path + str(sample) + '/'

            # if not os.path.exists(sample_path):
            #     os.makedirs(sample_path)

            # Detect shift.

            shift_detector = ShiftDetector(dr_technique, md_test, sign_level, red_model, sample, dataset)
            p_val, dist, red_dim, red_model, val_acc, te_acc = shift_detector.detect_data_shift(X_tr, y_tr, X_val, y_val, X_t[:sample,:], y_t[:sample], orig_dims)

            val_accs[rand_run, si] = val_acc
            te_accs[rand_run, si] = te_acc

            print("Shift p-vals: ", p_val)

            samples_rands[si,rand_run] = p_val

    mean_p_vals = np.mean(samples_rands, axis=1)
    std_p_vals = np.std(samples_rands, axis=1)

    mean_val_accs = np.mean(val_accs, axis=0)
    std_val_accs = np.std(val_accs, axis=0)

    mean_te_accs = np.mean(te_accs, axis=0)
    std_te_accs = np.std(te_accs, axis=0)
        
    return(mean_p_vals, std_p_vals)
