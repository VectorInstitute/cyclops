import datetime
import os
import sys
from functools import reduce
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

from drift_detector.explainer import ShiftExplainer
from experiments import *
from utils.constants import *
from utils.plot_utils import errorfill, plot_roc, plot_pr
from utils.utils import import_dataset_hospital, run_shift_experiment
from baseline_models.static.utils import run_model

DATASET = sys.argv[1]
DR_TECHNIQUE = sys.argv[2]
MD_TEST = sys.argv[3]
OUTCOME = sys.argv[4]
HOSPITAL = sys.argv[5]

# Define results path and create directory.
PATH = "/mnt/nfs/project/delirium/drift_exp/results/"
PATH += DATASET + "_"
PATH += DR_TECHNIQUE + "_"
PATH += MD_TEST + "/"

if not os.path.exists(path):
    os.makedirs(path)

# Define shift types.


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
if DR_TECHNIQUE not in DR_TECHNIQUES:
    raise Exception("Not a valid dimensionality reduction technique")
# Statistical Tests
MD_TESTS = ["MMD", "LK", "LSDD"]
if MD_TEST not in MD_TESTS:
    raise Exception("Not a valid two-sample test")
# Outcomes
OUTCOMES = ["length_of_stay_in_er", "mortality_in_hospital"]
if OUTCOME not in OUTCOMES:
    raise Exception("Not a valid outcome")
# Hospital
HOSPITALS = ["SMH", "MSH", "THPC", "THPM", "UHNTG", "UHNTW","PMH"]
if HOSPITAL in HOSPITALS:
    raise Exception("Not a valid hospital")
# Model
MODELS = ["lr", "xgb", "rf"]
# NA Cutoff
NA_CUTOFF = 0.6

# -------------------------------------------------
# PIPELINE START
# -------------------------------------------------

mean_dr_md_pval = np.ones(
    (len(EXPERIMENTS), len(DR_TECHNIQUES), len(MD_TESTS), len(SAMPLES))
) * (-1)
mean_dr_md_dist = np.ones(
    (len(EXPERIMENTS), len(DR_TECHNIQUES), len(MD_TESTS), len(SAMPLES))
) * (-1)
std_dr_md_pval = np.ones(
    (len(EXPERIMENTS), len(DR_TECHNIQUES), len(MD_TESTS), len(SAMPLES))
) * (-1)
std_dr_md_dist = np.ones(
    (len(EXPERIMENTS), len(DR_TECHNIQUES), len(MD_TESTS), len(SAMPLES))
) * (-1)

for si, SHIFT in enumerate(EXPERIMENTS):
    for di, DR_TECHNIQUE in enumerate(DR_TECHNIQUES):
        for mi, MD_TEST in enumerate(MD_TESTS):
            if np.any(mean_dr_md_pval[si, di, mi, :] == -1):
                print(
                    "{} | {} | {} | {}".format(
                        SHIFT, HOSPITAL, DR_TECHNIQUE, MD_TEST
                    )
                )
                try:
                    mean_p_vals, std_p_vals, mean_dist, std_dist = run_shift_experiment(
                        shift=SHIFT,
                        outcome=OUTCOME,
                        hospital=HOSPITAL,
                        path=PATH,
                        dr_technique=DR_TECHNIQUE,
                        md_test=MD_TEST,
                        samples=SAMPLES,
                        dataset=DATASET,
                        sign_level=SIGN_LEVEL,
                        na_cutoff=NA_CUTOFF,
                        random_runs=RANDOM_RUNS,
                        calc_acc=CALC_ACC,
                        bucket_size=6, 
                        window=6
                    )
                    mean_dr_md_pval[si, di, mi, :] = mean_p_vals
                    std_dr_md_pval[si, di, mi, :] = std_p_vals
                    mean_dr_md_dist[si, di, mi, :] = mean_dist
                    std_dr_md_dist[si, di, mi, :] = std_dist
                except ValueError as e:
                    print("Value Error")
                    pass

fig = plt.figure(figsize=(8, 6))
for si, shift in enumerate(EXPERIMENTS):
    for di, dr_technique in enumerate(DR_TECHNIQUES):
        for mi, md_test in enumerate(MD_TESTS):
            if (
                    dr_technique == DIM_RED
                    and md_test == MD_TEST
            ):
                errorfill(
                        np.array(SAMPLES),
                        mean_dr_md_pval[si, di, mi, :],
                        std_dr_md_pval[si, di, mi, :],
                        fmt=linestyles[si] + markers[si],
                        color=colorscale(colors[si], brightness[si]),
                        label="%s" % "_".join([shift, dr_technique, md_test]),
                )
plt.xlabel("Number of samples from test data")
plt.ylabel("$p$-value")
plt.axhline(y=SIGN_LEVEL, color="k")
plt.legend()
plt.show()

means_pval_file = PATH + "/driftexp_pval_means.csv"
means_pval = np.moveaxis(mean_dr_md_pval, 3, 1)
cols = pd.MultiIndex.from_product([DR_TECHNIQUES, MD_TESTS])
index = pd.MultiIndex.from_product([shifts, SAMPLES])
means_pval = means_pval.reshape(
    len(shifts) * len(SAMPLES), len(DR_TECHNIQUES) * len(MD_TESTS)
)
means_pval = pd.DataFrame(means_pval, columns=cols, index=index)
means_pval = pd.DataFrame(means_pval, columns=cols, index=SAMPLES)
means_pval.to_csv(means_pval_file, sep="\t")

stds_pval_file = PATH + "/driftexp_pval_stds.csv"
stds_pval = np.moveaxis(std_dr_md_pval, 3, 1)
stds_pval = stds_pval.reshape(
    len(shifts) * len(SAMPLES), len(DR_TECHNIQUES) * len(MD_TESTS)
)
stds_pval = pd.DataFrame(stds_pval, columns=cols, index=index)
stds_pval.to_csv(stds_pval_file, sep="\t")
