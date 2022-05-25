import datetime
import os
import sys
from functools import reduce

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    auc,
    average_precision_score,
    confusion_matrix,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import GridSearchCV, train_test_split

sys.path.append("..")

from shift_constants import *
from shift_detector import *
from shift_experiments import *
from shift_explainer import *
from shift_models import *
from shift_plot_utils import *
from shift_reductor import *
from shift_tester import *
from shift_utils import *

sys.path.append("../..")

import config
import sqlalchemy
from evidently import ColumnMapping
from evidently.dashboard import Dashboard
from evidently.dashboard.tabs import DataQualityTab
from evidently.model_profile import Profile
from evidently.model_profile.sections import DataQualityProfileSection
from sqlalchemy import desc, extract, func, select
from sqlalchemy.sql.expression import and_

import cyclops
from cyclops.orm import Database
from cyclops.processors.admin import AdminProcessor
from cyclops.processors.column_names import (
    ADMIT_TIMESTAMP,
    AGE,
    CITY,
    COUNTRY,
    DISCHARGE_DISPOSITION,
    DISCHARGE_TIMESTAMP,
    ENCOUNTER_ID,
    HOSPITAL_ID,
    LAB_TEST_NAME,
    LAB_TEST_RESULT_UNIT,
    LAB_TEST_RESULT_VALUE,
    LAB_TEST_TIMESTAMP,
    LANGUAGE,
    LENGTH_OF_STAY_IN_ER,
    PROVINCE,
    READMISSION,
    REFERENCE_RANGE,
    SEX,
    TOTAL_COST,
    VITAL_MEASUREMENT_NAME,
    VITAL_MEASUREMENT_TIMESTAMP,
    VITAL_MEASUREMENT_VALUE,
)
from cyclops.processors.constants import EMPTY_STRING
from cyclops.processors.feature_handler import FeatureHandler
from cyclops.processors.labs import LabsProcessor
from cyclops.processors.outcomes import OutcomesProcessor
from cyclops.processors.vitals import VitalsProcessor

datset = sys.argv[1]
dr_technique = sys.argv[3]

# Define results path and create directory.
path = "./results/"
path += test_type + "/"
path += datset + "_"
path += sys.argv[2] + "/"
if not os.path.exists(path):
    os.makedirs(path)

samples = [10, 20, 50, 100, 200, 500, 1000]

# Number of random runs to average results over.
random_runs = 5

# Significance level.
sign_level = 0.05

# Whether to calculate accuracy for malignancy quantification.
calc_acc = True

# Define shift types.
if sys.argv[2] == "small_gn_shift":
    shifts = ["small_gn_shift_0.1", "small_gn_shift_0.5", "small_gn_shift_1.0"]
elif sys.argv[2] == "medium_gn_shift":
    shifts = ["medium_gn_shift_0.1", "medium_gn_shift_0.5", "medium_gn_shift_1.0"]
elif sys.argv[2] == "large_gn_shift":
    shifts = ["large_gn_shift_0.1", "large_gn_shift_0.5", "large_gn_shift_1.0"]
elif sys.argv[2] == "ko_shift":
    shifts = ["ko_shift_0.1", "ko_shift_0.5", "ko_shift_1.0"]
elif sys.argv[2] == "mfa_shift":
    shifts = ["mfa_shift_0.25", "mfa_shift_0.5", "mfa_shift_0.75"]
elif sys.argv[2] == "cp_shift":
    shifts = ["cp_shift_0.25", "cp_shift_0.75"]
elif sys.argv[2] == "covid_shift":
    shifts = ["precovid", "covid"]
elif sys.argv[2] == "seasonal_shift":
    shifts = ["summer", "winter", "seasonal"]

# -------------------------------------------------
# PARAMETERS
# -------------------------------------------------

# Output path
PATH = "/mnt/nfs/project/delirium/drift_exp/"
# Number of samples in the test set
SAMPLES = [10, 20, 50, 100, 200, 500, 1000]
# Number of random runs to average results over.
RANDOM_RUNS = 5
# Significance level.
SIGN_LEVEL = 0.05
# Whether to calculate accuracy for malignancy quantification.
CALC_ACC = True
# Dimensionality Reduction Techniques
DR_TECHNIQUES = ["NoRed", "PCA", "BBSDs_FFNN", "SRP", "Isomap", "kPCA"]
# Statistical Tests
MD_TESTS = ["MMD", "LK", "LSDD"]

# -------------------------------------------------
# PIPELINE START
# -------------------------------------------------

OUTCOMES = ["length_of_stay_in_er", "mortality_in_hospital"]
HOSPITALS = ["SMH", "MSH", "THPC", "THPM", "UHNTG", "UHNTW"]
MODELS = ["lr", "xgb", "rf"]
NA_CUTOFF = 4000

# Run model fitting
shift_auc = np.ones((len(shifts), len(OUTCOMES), len(HOSPITALS), len(MODELS), 2)) * (-1)
shift_pr = np.ones((len(shifts), len(OUTCOMES), len(HOSPITALS), len(MODELS), 2)) * (-1)
for si, SHIFT in enumerate(shifts):
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
index = pd.MultiIndex.from_product([shifts, ["VAL_ROC_AUC", "TEST_ROC_AUC"]])
auc = auc.T.reshape(len(shifts) * 2, len(OUTCOMES) * len(HOSPITALS) * len(MODELS))
auc = pd.DataFrame(auc, columns=cols, index=index)
auc.to_csv(auc_file, sep="\t")

pr_file = PATH + "/driftexp_pr.csv"
pr = np.rollaxis(shift_pr, 2, 0)
cols = pd.MultiIndex.from_product([OUTCOMES, HOSPITALS, MODELS])
index = pd.MultiIndex.from_product([shifts, ["VAL_AVG_PR", "TEST_AVG_PR"]])
pr = pr.T.reshape(len(shifts) * 2, len(OUTCOMES) * len(HOSPITALS) * len(MODELS))
pr = pd.DataFrame(pr, columns=cols, index=index)
pr.to_csv(pr_file, sep="\t")

# Run shift experiments
mean_dr_md = np.ones(
    (len(shifts), len(HOSPITALS), len(DR_TECHNIQUES), len(MD_TESTS), len(SAMPLES))
) * (-1)
std_dr_md = np.ones(
    (len(shifts), len(HOSPITALS), len(DR_TECHNIQUES), len(MD_TESTS), len(SAMPLES))
) * (-1)
for si, SHIFT in enumerate(shifts):
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
index = pd.MultiIndex.from_product([shifts, HOSPITALS, SAMPLES])
means = means.reshape(
    len(shifts) * len(HOSPITALS) * len(SAMPLES), len(DR_TECHNIQUES) * len(MD_TESTS)
)
means = pd.DataFrame(means, columns=cols, index=index)
means = pd.DataFrame(means, columns=cols, index=SAMPLES)
means.to_csv(means_file, sep="\t")

stds_file = PATH + "/driftexp_stds.csv"
stds = np.moveaxis(std_dr_md, 4, 2)
stds = stds.reshape(
    len(shifts) * len(HOSPITALS) * len(SAMPLES), len(DR_TECHNIQUES) * len(MD_TESTS)
)
stds = pd.DataFrame(stds, columns=cols, index=index)
stds.to_csv(stds_file, sep="\t")
