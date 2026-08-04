"""Chest X-ray Disease Classification."""

# get args from command line
import argparse
from functools import partial
from typing import Any, Dict, List

import numpy as np
import plotly.express as px
from torchvision.transforms import Compose
from torchxrayvision.models import DenseNet

from cyclops.data.loader import load_nihcxr
from cyclops.data.slicer import (
    SliceSpec,
    filter_value,  # noqa: E402
)
from cyclops.data.transforms import Lambdad, Resized
from cyclops.data.utils import apply_transforms
from cyclops.evaluate import evaluator
from cyclops.evaluate.metrics.factory import create_metric
from cyclops.models.wrappers import PTModel  # type: ignore[attr-defined]
from cyclops.report import ModelCardReport  # type: ignore[attr-defined]


parser = argparse.ArgumentParser()

parser.add_argument("--sample_size", type=int, default=1000)
parser.add_argument("--synthetic_timestamp", type=str, default=None)
parser.add_argument("--split", type=str, default="val")
parser.add_argument("--seed", type=int, default=42)

args = parser.parse_args()

report = ModelCardReport()

data_dir = "/mnt/data2/clinical_datasets/NIHCXR"
nih_ds = load_nihcxr(data_dir)[args.split]

# select a subset of the data
np.random.seed(args.seed)
nih_ds = nih_ds.select(np.random.choice(len(nih_ds), args.sample_size))

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
            keys=("image",),
            func=lambda x: x[0][np.newaxis, :] if x.shape[0] != 1 else x,
        ),
    ],
)
pathologies = DenseNet(weights="densenet121-res224-nih").pathologies[:14]

model = PTModel(DenseNet(weights="densenet121-res224-nih"))
model.initialize()  # type: ignore
nih_ds = model.predict(
    nih_ds,
    feature_columns=["image"],
    transforms=partial(apply_transforms, transforms=transforms),
    model_name="densenet",
)

# remove any rows with No Finding == 1
nih_ds = nih_ds.filter(  # type: ignore[union-attr]
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

# define the slices
slices_sex = [
    {"Patient Gender": {"value": "M"}},
    {"Patient Gender": {"value": "F"}},
]

num_labels = len(pathologies)
ppv = create_metric(
    metric_name="multilabel_ppv",
    experimental=True,
    num_labels=num_labels,
    average=None,
)

npv = create_metric(
    metric_name="multilabel_npv",
    experimental=True,
    num_labels=num_labels,
    average=None,
)

specificity = create_metric(
    metric_name="multilabel_specificity",
    experimental=True,
    num_labels=num_labels,
    average=None,
)

sensitivity = create_metric(
    metric_name="multilabel_sensitivity",
    experimental=True,
    num_labels=num_labels,
    average=None,
)
# create the slice functions
slice_spec = SliceSpec(spec_list=slices_sex)

nih_eval_results_gender = evaluator.evaluate(
    dataset=nih_ds,
    metrics=[ppv, npv, sensitivity, specificity],  # type: ignore[list-item]
    target_columns=pathologies,
    prediction_columns="predictions.densenet",
    ignore_columns="image",
    slice_spec=slice_spec,
)

# define the slices
slices_age: List[Dict[str, Dict[str, Any]]] = [
    {"Patient Age": {"min_value": 19, "max_value": 35}},
    {"Patient Age": {"min_value": 35, "max_value": 65}},
    {"Patient Age": {"min_value": 65, "max_value": 100}},
    {
        "Patient Age": {"min_value": 19, "max_value": 35},
        "Patient Gender": {"value": "M"},
    },
    {
        "Patient Age": {"min_value": 19, "max_value": 35},
        "Patient Gender": {"value": "F"},
    },
    {
        "Patient Age": {"min_value": 35, "max_value": 65},
        "Patient Gender": {"value": "M"},
    },
    {
        "Patient Age": {"min_value": 35, "max_value": 65},
        "Patient Gender": {"value": "F"},
    },
    {
        "Patient Age": {"min_value": 65, "max_value": 100},
        "Patient Gender": {"value": "M"},
    },
    {
        "Patient Age": {"min_value": 65, "max_value": 100},
        "Patient Gender": {"value": "F"},
    },
]

# create the slice functions
slice_spec = SliceSpec(spec_list=slices_age)

nih_eval_results_age = evaluator.evaluate(
    dataset=nih_ds,
    metrics=[ppv, npv, sensitivity, specificity],  # type: ignore[list-item]
    target_columns=pathologies,
    prediction_columns="predictions.densenet",
    ignore_columns="image",
    slice_spec=slice_spec,
)


fig = px.pie(
    values=[nih_ds["Patient Gender"].count("M"), nih_ds["Patient Gender"].count("F")],
    names=["Male", "Female"],
)
fig.update_layout(
    title="Gender Distribution",
    width=1200,
    height=600,
)
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
    width=1200,
    height=600,
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
    width=1200,
    height=600,
)
report.log_plotly_figure(
    fig=fig,
    caption="Pathology Distribution",
    section_name="datasets",
)


results_flat = {}
for slice_, metrics in nih_eval_results_age["model_for_predictions.densenet"].items():
    for name, metric in metrics.items():
        if "sample_size" in name:
            results_flat[f"{slice_}/{name}"] = metric
        else:
            results_flat[f"{slice_}/{name}"] = metric.mean()
            for itr, m in enumerate(metric):
                if slice_ == "overall":
                    results_flat[f"pathology:{pathologies[itr]}/{name}"] = m
                else:
                    results_flat[f"{slice_}&pathology:{pathologies[itr]}/{name}"] = m
for slice_, metrics in nih_eval_results_gender[
    "model_for_predictions.densenet"
].items():
    for name, metric in metrics.items():
        if "sample_size" in name:
            results_flat[f"{slice_}/{name}"] = metric
        else:
            results_flat[f"{slice_}/{name}"] = metric.mean()
            for itr, m in enumerate(metric):
                if slice_ == "overall":
                    results_flat[f"pathology:{pathologies[itr]}/{name}"] = m
                else:
                    results_flat[f"{slice_}&pathology:{pathologies[itr]}/{name}"] = m
sample_sizes = {
    key.split("/")[0]: value
    for key, value in results_flat.items()
    if "sample_size" in key.split("/")[1]
}
for name, metric in results_flat.items():
    split, name = name.split("/")  # noqa: PLW2901
    descriptions = {
        "MultilabelPPV": "The proportion of correctly predicted positive instances among all instances predicted as positive. Also known as precision.",
        "MultilabelNPV": "The proportion of correctly predicted negative instances among all instances predicted as negative.",
        "MultilabelSensitivity": "The proportion of actual positive instances that are correctly predicted. Also known as recall or true positive rate.",
        "MultilabelSpecificity": "The proportion of actual negative instances that are correctly predicted.",
    }
    # remove the "&pathology" from the split name
    # check if splits contains "pathology:{pathology}" and remove it
    if "pathology" in split:
        if len(split.split("&")) == 1:
            sample_size_split = "overall"
        else:
            sample_size_split = "&".join(split.split("&")[:-1])
    else:
        sample_size_split = split
    if name != "sample_size":
        report.log_quantitative_analysis(
            "performance",
            name=name,
            value=metric.tolist() if isinstance(metric, np.generic) else metric,
            description=descriptions.get(name),
            metric_slice=split,
            pass_fail_thresholds=0.7,
            pass_fail_threshold_fns=lambda x, threshold: bool(x >= threshold),
            sample_size=sample_sizes[sample_size_split],
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

report.log_owner(
    name="Machine Learning and Medicine Lab",
    contact="mlmed.org",
    email="mlmed@gmail.com",
)

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

report_path = report.export(
    output_filename="nihcxr_report_periodic.html",
    synthetic_timestamp=args.synthetic_timestamp,
)
