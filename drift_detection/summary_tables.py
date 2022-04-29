import sys
import pandas as pd
import numpy as np
import os
from functools import reduce
import datetime
import joblib
import matplotlib.pyplot as plt
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

sys.path.append("..")

from shift_detector import *
from shift_reductor import *
from shift_experiments import *
from shift_explainer import *
from shift_tester import *
from shift_utils import *
from shift_models import *
from shift_constants import *
from shift_plot_utils import *

sys.path.append("../..")

import sqlalchemy
from sqlalchemy import select, func, extract, desc
from sqlalchemy.sql.expression import and_

from evidently import ColumnMapping
from evidently.dashboard import Dashboard
from evidently.dashboard.tabs import DataQualityTab
from evidently.model_profile import Profile
from evidently.model_profile.sections import DataQualityProfileSection

import config
import cyclops
from cyclops.processors.column_names import (
    ENCOUNTER_ID,
    HOSPITAL_ID,
    ADMIT_TIMESTAMP,
    DISCHARGE_TIMESTAMP,
    DISCHARGE_DISPOSITION,
    READMISSION,
    AGE,
    SEX,
    TOTAL_COST,
    CITY,
    PROVINCE,
    COUNTRY,
    LANGUAGE,
    LENGTH_OF_STAY_IN_ER,
    VITAL_MEASUREMENT_NAME,
    VITAL_MEASUREMENT_VALUE,
    VITAL_MEASUREMENT_TIMESTAMP,
    LAB_TEST_NAME,
    LAB_TEST_TIMESTAMP,
    LAB_TEST_RESULT_VALUE,
    LAB_TEST_RESULT_UNIT,
    REFERENCE_RANGE,
)
from cyclops.processors.constants import EMPTY_STRING
from cyclops.processors.admin import AdminProcessor
from cyclops.processors.vitals import VitalsProcessor
from cyclops.processors.labs import LabsProcessor
from cyclops.processors.outcomes import OutcomesProcessor
from cyclops.processors.feature_handler import FeatureHandler
from cyclops.orm import Database

%reload_ext autoreload
%reload_ext nb_black

SHIFTS = ["pre-covid", "covid","summer","winter","seasonal"]
OUTCOMES = ["length_of_stay_in_er","mortality_in_hospital"]
HOSPITALS = ["SMH","MSH","THPC","THPM","UHNTG", "UHNTW"]
MODELS = ["xgb","rf"]
NA_CUTOFF = 4000
PATH = "/mnt/nfs/project/delirium/drift_exp/"
DATASET = "gemini"
SAMPLES = [10, 20, 50, 100, 200, 500, 1000, 2000]
RANDOM_RUNS = 5
SIGN_LEVEL = 0.05
CALC_ACC = True
DR_TECHNIQUES = ["NoRed", "PCA", "BBSDs_FFNN", "SRP", "Isomap","kPCA"]
MD_TESTS = ["MMD", "LK", "LSDD"]

# Run model fitting
shift_auc = np.ones((len(SHIFTS), len(OUTCOMES), len(HOSPITALS), len(MODELS), 2)) * (-1)
shift_pr = np.ones((len(SHIFTS), len(OUTCOMES), len(HOSPITALS), len(MODELS), 2)) * (-1)
for si, SHIFT in enumerate(SHIFTS):
    for oi, OUTCOME in enumerate(OUTCOMES):
        for hi, HOSPITAL in enumerate(HOSPITALS):
            for mi, MODEL in enumerate(MODELS):
                print("{} | {} | {} | {}".format(SHIFT, OUTCOME, HOSPITAL, MODEL))
                (
                    (X_train, y_train),
                    (X_val, y_val),
                    (X_test, y_test),
                    feats,
                    orig_dims,
                ) = import_dataset_hospital(
                    SHIFT, OUTCOME, HOSPITAL, NA_CUTOFF, shuffle=True
                )

                optimised_model = run_model(MODEL, X_train, y_train, X_val, y_val)

                # calc metrics for validation set
                y_val_pred_prob = optimised_model.predict_proba(X_val)[:, 1]
                val_fpr, val_tpr, val_thresholds = roc_curve(
                    y_val, y_val_pred_prob, pos_label=1
                )
                val_roc_auc = auc(val_fpr, val_tpr)
                val_precision, val_recall, val_thresholds = precision_recall_curve(
                    y_val, y_val_pred_prob
                )
                val_avg_pr = average_precision_score(y_val, y_val_pred_prob)

                # calc metrics for test set
                y_test_pred_prob = optimised_model.predict_proba(X_test)[:, 1]
                test_fpr, test_tpr, test_thresholds = roc_curve(
                    y_test, y_test_pred_prob, pos_label=1
                )
                test_roc_auc = auc(test_fpr, test_tpr)
                test_precision, test_recall, test_thresholds = precision_recall_curve(
                    y_test, y_test_pred_prob
                )
                test_avg_pr = average_precision_score(y_test, y_test_pred_prob)

                shift_auc[si, oi, hi, mi, :] = [val_roc_auc, test_roc_auc]
                shift_pr[si, oi, hi, mi, :] = [val_avg_pr, test_avg_pr]

auc_file = PATH + "/driftexp_auc.csv"
auc = np.rollaxis(shift_auc, 2, 0)
cols = pd.MultiIndex.from_product([OUTCOMES, HOSPITALS, MODELS])
index = pd.MultiIndex.from_product([SHIFTS, ["VAL_ROC_AUC", "TEST_ROC_AUC"]])
auc = auc.T.reshape(len(SHIFTS) * 2, len(OUTCOMES) * len(HOSPITALS) * len(MODELS))
auc = pd.DataFrame(auc, columns=cols, index=index)
auc.to_csv(auc_file, sep="\t")

pr_file = PATH + "/driftexp_pr.csv"
pr = np.rollaxis(shift_pr, 2, 0)
cols = pd.MultiIndex.from_product([OUTCOMES, HOSPITALS, MODELS])
index = pd.MultiIndex.from_product([SHIFTS, ["VAL_AVG_PR", "TEST_AVG_PR"]])
pr = pr.T.reshape(len(SHIFTS) * 2, len(OUTCOMES) * len(HOSPITALS) * len(MODELS))
pr = pd.DataFrame(pr, columns=cols, index=index)
pr.to_csv(pr_file, sep="\t")

# Run shift experiments
mean_dr_md = np.ones(
    (len(SHIFTS), len(HOSPITALS), len(DR_TECHNIQUES), len(MD_TESTS), len(SAMPLES))
) * (-1)
std_dr_md = np.ones(
    (len(SHIFTS), len(HOSPITALS), len(DR_TECHNIQUES), len(MD_TESTS), len(SAMPLES))
) * (-1)
for si, SHIFT in enumerate(SHIFTS):
    for hi, HOSPITAL in enumerate(HOSPITALS):
        for di, DR_TECHNIQUE in enumerate(DR_TECHNIQUES):
            for mi, MD_TEST in enumerate(MD_TESTS):
                print(
                    "{} | {} | {} | {}".format(SHIFT, HOSPITAL, DR_TECHNIQUE, MD_TEST)
                )
                if np.any(mean_dr_md[si, hi, di, mi, :] == -1):
                    try:
                        mean_p_vals, std_p_vals = run_shift_experiment(
                            SHIFT,
                            OUTCOME,
                            HOSPITAL,
                            PATH,
                            DR_TECHNIQUE,
                            MD_TEST,
                            SAMPLES,
                            DATASET,
                            SIGN_LEVEL,
                            NA_CUTOFF,
                            RANDOM_RUNS,
                            calc_acc=True,
                        )
                        mean_dr_md[si, hi, di, mi, :] = mean_p_vals
                        std_dr_md[si, hi, di, mi, :] = std_p_vals
                    except ValueError as e:
                        print("Value Error")
                        pass
         
        
means_file = PATH + "/driftexp_means.csv"
means = np.moveaxis(mean_dr_md, 4, 2)
cols = pd.MultiIndex.from_product([DR_TECHNIQUES, MD_TESTS])
index = pd.MultiIndex.from_product([SHIFTS, HOSPITALS, SAMPLES])
means = means.reshape(
    len(SHIFTS) * len(HOSPITALS) * len(SAMPLES), len(DR_TECHNIQUES) * len(MD_TESTS)
)
means = pd.DataFrame(means, columns=cols, index=index)means = pd.DataFrame(means, columns=cols, index=SAMPLES)
means.to_csv(means_file, sep="\t")

stds_file = PATH + "/driftexp_stds.csv"
stds = np.moveaxis(std_dr_md, 4, 2)
stds = stds.reshape(
    len(SHIFTS) * len(HOSPITALS) * len(SAMPLES), len(DR_TECHNIQUES) * len(MD_TESTS)
)
stds = pd.DataFrame(stds, columns=cols, index=index)
stds.to_csv(stds_file, sep="\t")

                      