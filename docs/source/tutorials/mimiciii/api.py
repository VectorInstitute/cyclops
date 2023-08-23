"""Showcase use of evaluate API endpoint using FastAPI."""


import json
import typing
import plotly.express as px
import plotly.graph_objects as go
from reactpy import component, html
from reactpy.backend.fastapi import configure
from fastapi import Body, FastAPI
from fastapi.responses import JSONResponse
from datasets import Dataset
from datasets.features import ClassLabel
import kaggle
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler
import numpy as np

from cyclops.data.slicer import SliceSpec
from cyclops.evaluate.metrics import MetricCollection, create_metric
from cyclops.models.catalog import create_model
from cyclops.process.feature.feature import TabularFeatures
from cyclops.tasks.mortality_prediction import MortalityPredictionTask
from cyclops.utils.file import join, load_dataframe


DATA_DIR = "./data"
RANDOM_SEED = 85
NAN_THRESHOLD = 0.75
TRAIN_SIZE = 0.8


kaggle.api.dataset_download_files(
    "saurabhshahane/in-hospital-mortality-prediction", path=DATA_DIR, unzip=True
)
df = load_dataframe(join(DATA_DIR, "data01.csv"), file_format="csv")
thresh_nan = int(NAN_THRESHOLD * len(df))
df = df.dropna(axis=1, thresh=thresh_nan)
df = df.dropna(axis=0, subset=["outcome"])
df["outcome"] = df["outcome"].astype("int")


features_list = [
    "Anion gap",
    "Lactic acid",
    "Blood calcium",
    "Lymphocyte",
    "Leucocyte",
    "heart rate",
    "Blood sodium",
    "Urine output",
    "Platelets",
    "Urea nitrogen",
    "age",
    "MCH",
    "RBC",
    "Creatine kinase",
    "PCO2",
    "Blood potassium",
    "Diastolic blood pressure",
    "Respiratory rate",
    "Renal failure",
    "NT-proBNP",
]
features_list = sorted(features_list)
tab_features = TabularFeatures(
    data=df.reset_index(),
    features=features_list,
    by="ID",
    targets="outcome",
)


numeric_transformer = Pipeline(
    steps=[("imputer", SimpleImputer(strategy="mean")), ("scaler", MinMaxScaler())]
)

binary_transformer = Pipeline(
    steps=[("imputer", SimpleImputer(strategy="most_frequent"))]
)


numeric_features = sorted((tab_features.features_by_type("numeric")))
numeric_indices = [
    df[features_list].columns.get_loc(column) for column in numeric_features
]
binary_features = sorted(tab_features.features_by_type("binary"))
binary_features.remove("outcome")
binary_indices = [
    df[features_list].columns.get_loc(column) for column in binary_features
]
preprocessor = ColumnTransformer(
    transformers=[
        ("num", numeric_transformer, numeric_indices),
        ("bin", binary_transformer, binary_indices),
    ],
    remainder="passthrough",
)


dataset = Dataset.from_pandas(df)
dataset.cleanup_cache_files()
dataset = dataset.cast_column("outcome", ClassLabel(num_classes=2))
dataset = dataset.train_test_split(
    train_size=TRAIN_SIZE, stratify_by_column="outcome", seed=RANDOM_SEED
)


model_name = "sgd_classifier"
model = create_model(model_name, random_state=123, verbose=0, class_weight="balanced")
mortality_task = MortalityPredictionTask(
    {model_name: model}, task_features=features_list, task_target="outcome"
)
mortality_task.list_models()
best_model_params = {
    "alpha": [0.0001, 0.001, 0.01, 0.1, 1, 10, 100],
    "learning_rate": ["constant", "optimal", "invscaling", "adaptive"],
    "eta0": [0.1, 0.01, 0.001, 0.0001],
    "metric": "roc_auc",
    "method": "grid",
}


mortality_task.train(
    dataset["train"],
    model_name=model_name,
    transforms=preprocessor,
    best_model_params=best_model_params,
)

y_pred = mortality_task.predict(
    dataset["test"],
    model_name=model_name,
    transforms=preprocessor,
    proba=False,
    only_predictions=True,
)
print(len(y_pred))


metric_names = ["accuracy", "precision", "recall", "f1_score", "auroc", "roc_curve"]
metrics = [create_metric(metric_name, task="binary") for metric_name in metric_names]
metric_collection = MetricCollection(metrics)


spec = {
    "age": {
        "min_value": 30,
        "max_value": 50,
        "min_inclusive": True,
        "max_inclusive": False,
    },
    "gendera": {"value": 1},
    "Anion gap": {
        "min_value": 14.73,
        "min_inclusive": False,
    }
}

with open("spec.json", "w") as outfile:
    json.dump(spec, outfile, indent=4)


class NumpyEncoder(json.JSONEncoder):
    """ Special json encoder for numpy types """
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)


app = FastAPI()
@app.post('/evaluate')
def evaluate(
        spec: dict = Body(...)
):
    slice_spec = SliceSpec([spec])
    results, _ = mortality_task.evaluate(
        dataset["test"],
        metric_collection,
        model_names=model_name,
        transforms=preprocessor,
        prediction_column_prefix="preds",
        slice_spec=slice_spec,
        batch_size=64,
    )
    results_json = json.dumps(results, indent=4, cls=NumpyEncoder)
    return JSONResponse(content=results_json)


@component
def plot_auroc():
    # results_json_string = evaluate(spec)
    # results = json.loads(results_json_string)
    # key = list(results[model_name].keys())[0]
    # fpr, tpr, _ = results[model_name][key]["BinaryROCCurve"]
    # aurocs = results[model_name][key]["BinaryAUROC"]

    # return html.pre(f"{results_json_string}")
    return html.pre(f"Hello World")


configure(app, plot_auroc)
