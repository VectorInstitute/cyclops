"""Script to run drift detection for GEMINI use case with set of chosen parameters."""
import os
import sys

import numpy as np
import pandas as pd
from sklearn.metrics import (
    auc,
    average_precision_score,
    precision_recall_curve,
    roc_curve,
)

from .baseline_models.static.utils import run_model

# from utils.constants import *
# from utils.utils import (
#     import_dataset_hospital,
#     run_shift_experiment,
#     run_synthetic_shift_experiment,
# )
from .gemini.utils import import_dataset_hospital

# from drift_detector import Experimenter


DATASET = sys.argv[1]
SHIFT = sys.argv[2]

# Define results PATH and create directory.
PATH = "/mnt/nfs/project/delirium/drift_exp/results/"
PATH += DATASET + "_"
PATH += SHIFT + "_"

if not os.path.exists(PATH):
    os.makedirs(PATH)

# Define shift types.
if SHIFT == "small_gn_shift":
    shifts = ["small_gn_shift_0.1", "small_gn_shift_0.5", "small_gn_shift_1.0"]
elif SHIFT == "medium_gn_shift":
    shifts = ["medium_gn_shift_0.1", "medium_gn_shift_0.5", "medium_gn_shift_1.0"]
elif SHIFT == "large_gn_shift":
    shifts = ["large_gn_shift_0.1", "large_gn_shift_0.5", "large_gn_shift_1.0"]
elif SHIFT == "ko_shift":
    shifts = ["ko_shift_0.1", "ko_shift_0.5", "ko_shift_1.0"]
elif SHIFT == "small_bn_shift":
    shifts = ["small_bn_shift_0.1", "small_bn_shift_0.5", "small_bn_shift_1.0"]
elif SHIFT == "medium_bn_shift":
    shifts = ["medium_bn_shift_0.1", "medium_bn_shift_0.5", "medium_bn_shift_1.0"]
elif SHIFT == "large_bn_shift":
    shifts = ["large_bn_shift_0.1", "large_bn_shift_0.5", "large_bn_shift_1.0"]
elif SHIFT == "mfa_shift":
    shifts = ["mfa_shift_0.25", "mfa_shift_0.5", "mfa_shift_0.75"]
elif SHIFT == "cp_shift":
    shifts = ["cp_shift_0.25", "cp_shift_0.75"]
elif SHIFT == "covid_shift":
    shifts = ["precovid", "covid"]
elif SHIFT == "seasonal_shift":
    shifts = ["summer", "winter", "seasonal"]

# -------------------------------------------------
# PARAMETERS
# -------------------------------------------------
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
MD_TESTS = ["MMD", "LK", "LSDD", "Classifier"]
# Outcomes
OUTCOMES = ["length_of_stay_in_er", "mortality_in_hospital"]
# Hospital
HOSPITALS = ["SMH", "MSH", "THPC", "THPM", "UHNTG", "UHNTW"]
# Model
MODELS = ["lr", "xgb", "rf"]
# NA Cutoff
NA_CUTOFF = 0.6
# -------------------------------------------------
# PIPELINE START
# -------------------------------------------------

# Run model fitting
shift_auc = np.ones((len(shifts), len(OUTCOMES), len(HOSPITALS), len(MODELS), 2)) * (-1)
shift_pr = np.ones((len(shifts), len(OUTCOMES), len(HOSPITALS), len(MODELS), 2)) * (-1)
for si, SHIFT in enumerate(shifts):
    for oi, OUTCOME in enumerate(OUTCOMES):
        for hi, HOSPITAL in enumerate(HOSPITALS):
            for mi, MODEL in enumerate(MODELS):
                print(f"Running {SHIFT} {OUTCOME} {HOSPITAL} {MODEL}")
                admin_data, X, y = None, None, None
                if SHIFT in ["covid", "seasonal"]:
                    (
                        (X_train, y_train),
                        (X_val, y_val),
                        (X_test, y_test),
                        feats,
                        orig_dims,
                    ) = import_dataset_hospital(
                        admin_data,
                        X,
                        y,
                        SHIFT,
                        OUTCOME,
                        HOSPITAL,
                        NA_CUTOFF,
                        shuffle=True,
                    )

                else:
                    (
                        (X_train, y_train),
                        (X_val, y_val),
                        (X_test, y_test),
                        feats,
                        orig_dims,
                    ) = import_dataset_hospital(
                        admin_data,
                        X,
                        y,
                        "baseline",
                        OUTCOME,
                        HOSPITAL,
                        NA_CUTOFF,
                        shuffle=True,
                    )
                    X_t_1, y_t_1 = X_train.copy(), y_train.copy()
                    # (X_t_1, y_t_1) = apply_shift(X_tr, y_tr, X_t_1, y_t_1, shift)
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

AUC_FILE = PATH + "/driftexp_auc.csv"
auc = np.rollaxis(shift_auc, 2, 0)
cols = pd.MultiIndex.from_product([OUTCOMES, HOSPITALS, MODELS])
index = pd.MultiIndex.from_product([shifts, ["VAL_ROC_AUC", "TEST_ROC_AUC"]])
auc = auc.T.reshape(len(shifts) * 2, len(OUTCOMES) * len(HOSPITALS) * len(MODELS))
auc = pd.DataFrame(auc, columns=cols, index=index)
auc.to_csv(AUC_FILE, sep="\t")

PR_FILE = PATH + "/driftexp_pr.csv"
pr = np.rollaxis(shift_pr, 2, 0)
cols = pd.MultiIndex.from_product([OUTCOMES, HOSPITALS, MODELS])
index = pd.MultiIndex.from_product([shifts, ["VAL_AVG_PR", "TEST_AVG_PR"]])
pr = pr.T.reshape(len(shifts) * 2, len(OUTCOMES) * len(HOSPITALS) * len(MODELS))
pr = pd.DataFrame(pr, columns=cols, index=index)
pr.to_csv(PR_FILE, sep="\t")

# Run shift experiments
mean_dr_md_pval = np.ones(
    (len(shifts), len(HOSPITALS), len(DR_TECHNIQUES), len(MD_TESTS), len(SAMPLES))
) * (-1)
std_dr_md_pval = np.ones(
    (len(shifts), len(HOSPITALS), len(DR_TECHNIQUES), len(MD_TESTS), len(SAMPLES))
) * (-1)
mean_dr_md_dist = np.ones(
    (len(shifts), len(HOSPITALS), len(DR_TECHNIQUES), len(MD_TESTS), len(SAMPLES))
) * (-1)
std_dr_md_dist = np.ones(
    (len(shifts), len(HOSPITALS), len(DR_TECHNIQUES), len(MD_TESTS), len(SAMPLES))
) * (-1)
for si, SHIFT in enumerate(shifts):
    for hi, HOSPITAL in enumerate(HOSPITALS):
        for di, DR_TECHNIQUE in enumerate(DR_TECHNIQUES):
            for mi, MD_TEST in enumerate(MD_TESTS):
                print(f"Running {SHIFT} {HOSPITAL} {DR_TECHNIQUE} {MD_TEST} {SAMPLES}")
                if np.any(mean_dr_md_pval[si, hi, di, mi, :] == -1):
                    try:
                        if SHIFT in ["covid", "seasonal"]:
                            # mean_p_vals, std_p_vals,
                            # mean_dist, std_dist = run_shift_experiment(
                            #     shift=SHIFT,
                            #     outcome=OUTCOME,
                            #     hospital=HOSPITAL,
                            #     PATH=PATH,
                            #     dr_technique=DR_TECHNIQUE,
                            #     md_test=MD_TEST,
                            #     samples=SAMPLES,
                            #     dataset=DATASET,
                            #     sign_level=SIGN_LEVEL,
                            #     na_cutoff=NA_CUTOFF,
                            #     random_runs=RANDOM_RUNS,
                            #     calc_acc=CALC_ACC,
                            # )
                            pass
                        else:
                            pass
                            # (
                            #     mean_p_vals,
                            #     std_p_vals,
                            #     mean_dist,
                            #     std_dist,
                            # ) = run_synthetic_shift_experiment(
                            #     shift=SHIFT,
                            #     outcome=OUTCOME,
                            #     hospital=HOSPITAL,
                            #     PATH=PATH,
                            #     dr_technique=DR_TECHNIQUE,
                            #     md_test=MD_TEST,
                            #     samples=SAMPLES,
                            #     dataset=DATASET,
                            #     sign_level=SIGN_LEVEL,
                            #     na_cutoff=NA_CUTOFF,
                            #     random_runs=RANDOM_RUNS,
                            #     calc_acc=CALC_ACC,
                            # )
                        # mean_dr_md_pval[si, hi, di, mi, :] = mean_p_vals
                        # std_dr_md_pval[si, hi, di, mi, :] = std_p_vals
                        # mean_dr_md_dist[si, hi, di, mi, :] = mean_dist
                        # std_dr_md_dist[si, hi, di, mi, :] = std_dist
                    except ValueError:
                        print("Value Error")


MEANS_PVAL_FILE = PATH + "/driftexp_pval_means.csv"
means_pval = np.moveaxis(mean_dr_md_pval, 4, 2)
cols = pd.MultiIndex.from_product([DR_TECHNIQUES, MD_TESTS])
index = pd.MultiIndex.from_product([shifts, HOSPITALS, SAMPLES])
means_pval = means_pval.reshape(
    len(shifts) * len(HOSPITALS) * len(SAMPLES), len(DR_TECHNIQUES) * len(MD_TESTS)
)
means_pval = pd.DataFrame(means_pval, columns=cols, index=index)
means_pval = pd.DataFrame(means_pval, columns=cols, index=SAMPLES)
means_pval.to_csv(MEANS_PVAL_FILE, sep="\t")

STDS_PVAL_FILE = PATH + "/driftexp_pval_stds.csv"
stds_pval = np.moveaxis(std_dr_md_pval, 4, 2)
stds_pval = stds_pval.reshape(
    len(shifts) * len(HOSPITALS) * len(SAMPLES), len(DR_TECHNIQUES) * len(MD_TESTS)
)
stds_pval = pd.DataFrame(stds_pval, columns=cols, index=index)
stds_pval.to_csv(STDS_PVAL_FILE, sep="\t")
