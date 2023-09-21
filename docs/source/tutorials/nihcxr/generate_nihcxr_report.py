"""Chest X-ray Disease Classification."""

# get args from command line
import argparse
import shutil
from functools import partial

import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from monai.transforms import Compose, Lambdad, Resized  # type: ignore
from torchxrayvision.models import DenseNet

from cyclops.data.loader import load_nihcxr
from cyclops.data.slicer import (
    SliceSpec,
    filter_value,  # noqa: E402
)
from cyclops.data.utils import apply_transforms
from cyclops.evaluate import evaluator
from cyclops.evaluate.fairness import evaluate_fairness  # type: ignore
from cyclops.evaluate.metrics.factory import create_metric
from cyclops.models.wrappers import PTModel  # type: ignore
from cyclops.report import ModelCardReport  # type: ignore


parser = argparse.ArgumentParser()
parser.add_argument(
    "--report_type",
    type=str,
    default="baseline",
    choices=["baseline", "periodic"],
)
parser.add_argument("--sample_size", type=int, default=4000)
parser.add_argument("--synthetic_timestamp", type=str, default=None)

args = parser.parse_args()

report = ModelCardReport()

data_dir = "/mnt/data/clinical_datasets/NIHCXR"
if args.report_type == "baseline":
    nih_ds = load_nihcxr(data_dir)["val"]
elif args.report_type == "periodic":
    nih_ds = load_nihcxr(data_dir)["test"]

nih_ds = nih_ds.select(range(args.sample_size))

transforms = Compose(
    [
        Resized(
            keys=("image",),
            spatial_size=(224, 224),
            allow_missing_keys=True,
        ),
        Lambdad(
            keys=("image",),
            func=lambda x: ((2 * (x / 255.0)) - 1.0) * 1024,
            allow_missing_keys=True,
        ),
        Lambdad(
            ("image",),
            func=lambda x: np.mean(x, axis=0)[np.newaxis, :] if x.shape[0] != 1 else x,
        ),
    ],
)

model = PTModel(DenseNet(weights="densenet121-res224-nih"))
model.initialize()  # type: ignore
nih_ds = model.predict(
    nih_ds,
    feature_columns=["image"],
    transforms=partial(apply_transforms, transforms=transforms),
    model_name="densenet",
)

# remove any rows with No Finding == 1
nih_ds = nih_ds.filter(
    partial(filter_value, column_name="No Finding", value=1, negate=True),
    batched=True,
)

# remove the No Finding column and adjust the predictions to account for it
nih_ds = nih_ds.map(
    lambda x: {
        "predictions.densenet": x["predictions.densenet"][:14],
    },
    remove_columns=["No Finding"],
)


pathologies = list(model.model.pathologies[:14])  # type: ignore

auroc = create_metric(
    metric_name="auroc",
    task="multilabel",  # type: ignore
    num_labels=len(pathologies),  # type: ignore
    thresholds=np.arange(0, 1, 0.01),  # type: ignore
)

# define the slices
slices_sex = [
    {"Patient Gender": {"value": "M"}},
    {"Patient Gender": {"value": "F"}},
]

# create the slice functions
slice_spec = SliceSpec(spec_list=slices_sex)

nih_eval_results_gender = evaluator.evaluate(
    dataset=nih_ds,
    metrics=auroc,
    feature_columns="image",
    target_columns=pathologies,
    prediction_column_prefix="predictions",
    remove_columns="image",
    slice_spec=slice_spec,
)

# plot the results
plots = []

for slice_name, slice_results in nih_eval_results_gender["densenet"].items():
    plots.append(
        go.Scatter(
            x=pathologies,
            y=slice_results["MultilabelAUROC"],
            name="Overall" if slice_name == "overall" else slice_name,
            mode="markers",
        ),
    )

perf_metric_gender = go.Figure(data=plots)
perf_metric_gender.update_layout(
    title="Multilabel AUROC by Pathology and Sex",
    title_x=0.5,
    title_font_size=20,
    xaxis_title="Pathology",
    yaxis_title="Multilabel AUROC",
    width=1024,
    height=768,
)
perf_metric_gender.update_traces(
    marker={"size": 12, "line": {"width": 2}},
    selector={"mode": "markers"},
)


auroc = create_metric(
    metric_name="auroc",
    task="multilabel",  # type: ignore
    num_labels=len(pathologies),  # type: ignore
    thresholds=np.arange(0, 1, 0.01),  # type: ignore
)

# define the slices
slices_age = [
    {"Patient Age": {"min_value": 19, "max_value": 35}},
    {"Patient Age": {"min_value": 35, "max_value": 65}},
    {"Patient Age": {"min_value": 65, "max_value": 100}},
]

# create the slice functions
slice_spec = SliceSpec(spec_list=slices_age)

nih_eval_results_age = evaluator.evaluate(
    dataset=nih_ds,
    metrics=auroc,
    feature_columns="image",
    target_columns=pathologies,
    prediction_column_prefix="predictions",
    remove_columns="image",
    slice_spec=slice_spec,
)


# plot the results
plots = []

for slice_name, slice_results in nih_eval_results_age["densenet"].items():
    plots.append(
        go.Scatter(
            x=pathologies,
            y=slice_results["MultilabelAUROC"],
            name="Overall" if slice_name == "overall" else slice_name,
            mode="markers",
        ),
    )

perf_metric_age = go.Figure(data=plots)
perf_metric_age.update_layout(
    title="Multilabel AUROC by Pathology and Age",
    title_x=0.5,
    title_font_size=20,
    xaxis_title="Pathology",
    yaxis_title="Multilabel AUROC",
    width=1024,
    height=768,
)
perf_metric_age.update_traces(
    marker={"size": 12, "line": {"width": 2}},
    selector={"mode": "markers"},
)


fig = px.pie(
    values=[nih_ds["Patient Gender"].count("M"), nih_ds["Patient Gender"].count("F")],
    names=["Male", "Female"],
)

fig.update_layout(title="Gender Distribution")

report.log_plotly_figure(
    fig=fig,
    caption="Gender Distribution",
    section_name="datasets",
)

fig = px.histogram(nih_ds["Patient Age"])
fig.update_traces(showlegend=False)
fig.update_layout(
    title="Age Distribution",
    xaxis_title="Age",
    yaxis_title="Count",
    bargap=0.2,
)

report.log_plotly_figure(
    fig=fig,
    caption="Age Distribution",
    section_name="datasets",
)

fig = px.bar(x=pathologies, y=[np.array(nih_ds[p]).sum() for p in pathologies])
fig.update_layout(
    title="Pathology Distribution",
    xaxis_title="Pathology",
    yaxis_title="Count",
    bargap=0.2,
)

report.log_plotly_figure(
    fig=fig,
    caption="Pathology Distribution",
    section_name="datasets",
)


specificity = create_metric(
    metric_name="specificity",
    task="multilabel",  # type: ignore
    num_labels=len(pathologies),  # type: ignore
)
sensitivity = create_metric(
    metric_name="sensitivity",
    task="multilabel",  # type: ignore
    num_labels=len(pathologies),  # type: ignore
)


fpr = 1 - specificity
fnr = 1 - sensitivity

balanced_error_rate = (fpr + fnr) / 2

nih_fairness_result_age = evaluate_fairness(
    metrics=balanced_error_rate,
    metric_name="BalancedErrorRate",
    dataset=nih_ds,
    remove_columns="image",
    target_columns=pathologies,
    prediction_columns="predictions.densenet",
    groups=["Patient Age"],
    group_bins={"Patient Age": [35, 65]},
    group_base_values={"Patient Age": 50},
)

# plot metrics per slice
plots = []
for slice_name, slice_results in nih_fairness_result_age.items():
    plots.append(
        go.Scatter(
            x=pathologies,
            y=slice_results["BalancedErrorRate"],
            name=slice_name,
            mode="markers",
        ),
    )
fairness_age = go.Figure(data=plots)
fairness_age.update_layout(
    title="Balanced Error Rate by Age vs. Pathology",
    title_x=0.5,
    title_font_size=20,
    xaxis_title="Pathology",
    yaxis_title="Balanced Error Rate",
    width=1024,
    height=768,
)
fairness_age.update_traces(
    marker={"size": 12, "line": {"width": 2}},
    selector={"mode": "markers"},
)


# plot parity difference per slice
plots = []

for slice_name, slice_results in nih_fairness_result_age.items():
    plots.append(
        go.Scatter(
            x=pathologies,
            y=slice_results["BalancedErrorRate Parity"],
            name=slice_name,
            mode="markers",
        ),
    )

fairness_age_parity = go.Figure(data=plots)
fairness_age_parity.update_layout(
    title="Balanced Error Rate Parity by Age vs. Pathology",
    title_x=0.5,
    title_font_size=20,
    xaxis_title="Pathology",
    yaxis_title="Balanced Error Rate Parity",
    width=1024,
    height=768,
)
fairness_age_parity.update_traces(
    marker={"size": 12, "line": {"width": 2}},
    selector={"mode": "markers"},
)


results_flat = {}
for slice_, metrics in nih_eval_results_age["densenet"].items():
    for name, metric in metrics.items():
        results_flat[f"{slice_}/{name}"] = metric.mean()
        for itr, m in enumerate(metric):
            results_flat[f"{slice_} ({pathologies[itr]})/{name}"] = m
for slice_, metrics in nih_eval_results_gender["densenet"].items():
    for name, metric in metrics.items():
        results_flat[f"{slice_}/{name}"] = metric.mean()
        for itr, m in enumerate(metric):
            results_flat[f"{slice_} ({pathologies[itr]})/{name} "] = m

for name, metric in results_flat.items():
    split, name = name.split("/")  # noqa: PLW2901
    report.log_quantitative_analysis(
        "performance",
        name=name,
        value=metric,
        metric_slice=split,
        pass_fail_thresholds=0.7,
        pass_fail_threshold_fns=lambda x, threshold: bool(x >= threshold),
    )


# model details for NIH Chest X-Ray model
report.log_from_dict(
    data={
        "name": "NIH Chest X-Ray Multi-label Classification Model",
        "description": "This model is a DenseNet121 model trained on the NIH Chest \
            X-Ray dataset, which contains 112,120 frontal-view X-ray images of 30,805 \
            unique patients with the fourteen text-mined disease labels from the \
            associated radiological reports. The labels are Atelectasis, Cardiomegaly, \
            Effusion, Infiltration, Mass, Nodule, Pneumonia, Pneumothorax, \
            Consolidation, Edema, Emphysema, Fibrosis, Pleural Thickening, and Hernia. \
            The model was trained on 80% of the data and evaluated on the remaining \
            20%.",
        "references": [{"link": "https://arxiv.org/abs/2111.00595"}],
    },
    section_name="Model Details",
)

report.log_citation(
    citation="""@inproceedings{Cohen2022xrv,
    title = {{TorchXRayVision: A library of chest X-ray datasets and models}},
    author = {Cohen, Joseph Paul and Viviano, Joseph D. and Bertin, \
    Paul and Morrison,Paul and Torabian, Parsa and Guarrera, \
    Matteo and Lungren, Matthew P and Chaudhari,\
    Akshay and Brooks, Rupert and Hashir, \
    Mohammad and Bertrand, Hadrien},
    booktitle = {Medical Imaging with Deep Learning},
    url = {https://github.com/mlmed/torchxrayvision},
    arxivId = {2111.00595},
    year = {2022}
    }""",
)

report.log_citation(
    citation="""@inproceedings{cohen2020limits,
    title={On the limits of cross-domain generalization\
            in automated X-ray prediction},
    author={Cohen, Joseph Paul and Hashir, Mohammad and Brooks, \
        Rupert and Bertrand, Hadrien},
    booktitle={Medical Imaging with Deep Learning},
    year={2020},
    url={https://arxiv.org/abs/2002.02497}
    }""",
)

report.log_owner(name="Machine Learning and Medicine Lab", contact="mlmed.org")

# considerations
report.log_user(description="Radiologists")
report.log_user(description="Data Scientists")

report.log_use_case(
    description="The model can be used to predict the presence of 14 pathologies \
        in chest X-ray images.",
    kind="primary",
)
report.log_descriptor(
    name="limitations",
    description="The limitations of this model include its inability to detect \
                pathologies that are not included in the 14 labels of the NIH \
                Chest X-Ray dataset. Additionally, the model may not perform \
                well on images that are of poor quality or that contain \
                artifacts. Finally, the model may not generalize well to\
                populations that are not well-represented in the training \
                data, such as patients from different geographic regions or \
                with different demographics.",
    section_name="considerations",
)
report.log_descriptor(
    name="tradeoffs",
    description="The model can help radiologists to detect pathologies in \
        chest X-ray images, but it may not generalize well to populations \
        that are not well-represented in the training data.",
    section_name="considerations",
)
report.log_risk(
    risk="One ethical risk of the model is that it may not generalize well to \
        populations that are not well-represented in the training data,\
        such as patients from different geographic regions \
        or with different demographics. ",
    mitigation_strategy="A mitigation strategy for this risk is to ensure \
        that the training data is diverse and representative of the population \
            that the model will be used on. Additionally, the model should be \
            regularly evaluated and updated to ensure that it continues to \
            perform well on diverse populations. Finally, the model should \
            be used in conjunction with human expertise to ensure that \
            any biases or limitations are identified and addressed.",
)
report.log_fairness_assessment(
    affected_group="Patients with rare pathologies",
    benefit="The model can help radiologists to detect pathologies in \
        chest X-ray images.",
    harm="The model may not generalize well to populations that are not \
        well-represented in the training data.",
    mitigation_strategy="A mitigation strategy for this risk is to ensure that \
        the training data is diverse and representative of the population.",
)


# qualitative analysis
report.log_plotly_figure(
    fig=perf_metric_gender,
    caption="MultiLabel AUROC by Pathology",
    section_name="Quantitative Analysis",
)

report.log_plotly_figure(
    fig=perf_metric_age,
    caption="MultiLabel AUROC by Pathology",
    section_name="Quantitative Analysis",
)

report.log_plotly_figure(
    fig=fairness_age,
    caption="Balanced Error Rate by Age vs. Pathology",
    section_name="Fairness Analysis",
)
report.log_plotly_figure(
    fig=fairness_age_parity,
    caption="Balanced Error Rate Parity by Age vs. Pathology",
    section_name="Fairness Analysis",
)

report_path = report.export(
    output_filename=f"nihcxr_report_{args.report_type}.html",
    report_type=args.report_type,
    synthetic_timestamp=args.synthetic_timestamp,
)
shutil.copy(f"{report_path}", ".")
